"""Sequential case sampler for the ML tab.

Evaluates a list of X points - one full pipeline case per point
(geometry + mesh, SU2 case setup, solve, post-processing) - in a worker
thread, mirroring the optimization runner's eval flow. Events land on
``self.queue`` for the GUI to drain:

    ("sample_start",  {index, case})
    ("sample_done",   {index, case, y})     y = {col: value}
    ("sample_failed", {index, case, error})
    ("log",           {line})
    ("done",          {ok, filled, failed})
"""

import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from .optimizer import set_design_value
from .runner import _popen_flags
from .state import get_path, set_path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


class MLSamplingRun:
    def __init__(self, base_config, run_dir, run_tag, points, x_cols, y_cols,
                 su2_exe=None, threads=4, max_iterations=None,
                 stage_timeout=None, x_kinds=None):
        self.base_config = copy.deepcopy(base_config)
        self.run_dir = Path(run_dir)
        self.run_tag = run_tag
        self.points = [dict(p) for p in points]
        self.x_cols = list(x_cols)
        self.y_cols = list(y_cols)
        # col -> "int" | "float": int-kind inputs are rounded before they
        # reach input.json (a float blade count crashes pyturbo/meshing)
        self.x_kinds = dict(x_kinds or {})
        self.su2_exe = su2_exe
        self.threads = int(threads)
        self.max_iterations = max_iterations
        self.stage_timeout = stage_timeout
        self.queue = queue.Queue()
        self.stop_requested = False
        self.thread = None
        self.proc = None
        self._stage_timed_out = False

    # ------------------------------------------------------------- control
    def start(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"ml-{self.run_tag}")
        self.thread.start()

    def stop(self):
        self.stop_requested = True
        self._kill_tree()

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def _emit(self, kind, **data):
        self.queue.put((kind, data))

    def _log(self, line):
        self._emit("log", line=str(line))

    # ------------------------------------------------------------ main loop
    def _run(self):
        filled = failed = 0
        try:
            self._log(f"ML sampling run folder: {self.run_dir}")
            for i, xvalues in enumerate(self.points):
                if self.stop_requested:
                    break
                case = f"{self.run_tag}_p{i:03d}"
                self._emit("sample_start", index=i, case=case)
                y, error = self._evaluate(i, xvalues, case)
                if self.stop_requested:
                    self._emit("sample_failed", index=i, case=case,
                               error="stopped by user")
                    failed += 1
                    break
                if error:
                    failed += 1
                    self._log(f"[p{i:03d}] failed: {error}")
                    self._emit("sample_failed", index=i, case=case,
                               error=error)
                else:
                    filled += 1
                    vals = ", ".join(f"{c}={y[c]:.5g}" for c in self.y_cols
                                     if isinstance(y.get(c), (int, float)))
                    self._log(f"[p{i:03d}] ok ({vals})")
                    self._emit("sample_done", index=i, case=case, y=y)
            self._emit("done", ok=True, filled=filled, failed=failed)
        except Exception as e:                      # pragma: no cover
            self._log(f"[error] internal error: {e}")
            self._emit("done", ok=False, filled=filled, failed=failed)
        finally:
            self._close_log()

    def _evaluate(self, index, xvalues, case):
        """Run one case; returns ({y col: value}, None) or (None, error)."""
        t0 = time.time()
        try:
            eval_dir = self._write_input(index, xvalues, case)
            case_dir = REPO / "cases" / case
            stages = [
                ("geometry + mesh",
                 [PY, str(REPO / "pipeline" / "run_pipeline.py"),
                  str(eval_dir / "input.json")], str(REPO), None),
                ("SU2 case setup",
                 [PY, str(REPO / "tools" / "setup_cascade_case.py"),
                  str(eval_dir), "--name", case, "--skip-validate"],
                 str(REPO), None),
                (f"SU2 solve ({self.threads} threads)",
                 [str(self.su2_exe), "-t", str(self.threads), "turbine.cfg"],
                 str(case_dir), str(case_dir / "su2_run.log")),
                ("post-processing",
                 [PY, str(REPO / "tools" / "plot_case.py"), str(case_dir)],
                 str(case_dir), None),
            ]
            for title, argv, cwd, logfile in stages:
                if self.stop_requested:
                    return None, "stopped by user"
                self._log(f"[p{index:03d}] {title} ...")
                rc, timed_out = self._run_stage(argv, cwd, logfile,
                                                timeout=self.stage_timeout)
                if self.stop_requested:
                    return None, "stopped by user"
                if rc != 0:
                    if timed_out:
                        minutes = (self.stage_timeout or 0) / 60.0
                        return None, (f"{title} killed after {minutes:.1f} "
                                      "min (hang) - recorded as failed")
                    return None, f"{title} failed (exit code {rc})"
            results_path = case_dir / "results.json"
            if not results_path.exists():
                return None, "no results.json after post-processing"
            results = json.loads(results_path.read_text())
            y = {}
            for col in self.y_cols:
                node, value = results, None
                for part in col.split("."):
                    if not isinstance(node, dict) or part not in node:
                        value = None
                        break
                    node = value = node[part]
                y[col] = value
            missing = [c for c, v in y.items() if v is None]
            if missing:
                return None, (f"quantity not found in results.json: "
                              f"{', '.join(missing)}")
            self._log(f"[p{index:03d}] done in {time.time() - t0:.0f} s")
            return y, None
        except Exception as e:
            return None, str(e)

    def _write_input(self, index, xvalues, case):
        """Write the eval's input.json (design overrides, R1=R2 sync,
        morph enabled when cage offsets are among the X columns)."""
        eval_dir = self.run_dir / f"p{index:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(self.base_config)
        for col, value in xvalues.items():
            if value is None:
                continue
            if self.x_kinds.get(col) == "int":
                # int-kind inputs (blade count, flow-guidance exponent, ...)
                # crash pyturbo/meshing when they arrive as floats
                value = int(round(float(value)))
            set_design_value(cfg, col, value)
        if any("airfoil_source.morph." in c and "@" in c
               for c in xvalues):
            set_path(cfg, "airfoil_source.morph.enabled", True)
        # pitch exploration must keep the SU2 single-translation pair valid
        # (the 3D wedge's rotational pair instead supports R1 != R2)
        if get_path(cfg, "domain.R1") is not None and \
                get_path(cfg, "domain.periodicity") != "axisymmetric3d":
            set_path(cfg, "domain.R2", get_path(cfg, "domain.R1"))
        if self.max_iterations:
            set_path(cfg, "solver_settings.max_iterations",
                     int(self.max_iterations))
        set_path(cfg, "case.name", case)
        set_path(cfg, "case.output_dir", "cases")
        set_path(cfg, "case.show_images", False)
        (eval_dir / "input.json").write_text(
            json.dumps(cfg, indent=2) + "\n")
        try:
            (REPO / "cases" / case).mkdir(parents=True, exist_ok=True)
            (REPO / "cases" / case / "input.json").write_text(
                json.dumps(cfg, indent=2) + "\n")
        except OSError:
            pass
        return eval_dir

    # --------------------------------------------------------- stage runner
    def _run_stage(self, argv, cwd, logfile, timeout=None):
        self._stage_timed_out = False
        logf = None
        try:
            if logfile:
                Path(logfile).parent.mkdir(parents=True, exist_ok=True)
                logf = open(logfile, "w")
                out = logf
            else:
                out = subprocess.PIPE
            self.proc = subprocess.Popen(
                argv, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                **_popen_flags())
        except OSError as e:
            self._log(f"[error] could not start {argv[0]}: {e}")
            return -1, False
        timer = None
        if timeout:
            timer = threading.Timer(timeout, self._on_stage_timeout)
            timer.daemon = True
            timer.start()
        try:
            if out is subprocess.PIPE:
                for raw in self.proc.stdout:
                    self._log("    " + raw.decode("utf-8",
                                                  errors="replace").rstrip())
            while self.proc.poll() is None:
                time.sleep(0.2)
        finally:
            if timer:
                timer.cancel()
            if logf:
                logf.close()
        rc = self.proc.returncode if self.proc.returncode is not None else -1
        return rc, self._stage_timed_out

    def _on_stage_timeout(self):
        self._stage_timed_out = True
        self._kill_tree()

    def _kill_tree(self):
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID",
                                str(proc.pid)], capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                proc.terminate()
            except OSError:
                pass

    def _close_log(self):
        pass
