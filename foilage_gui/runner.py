"""Background job runner for the GUI: mesh / setup / solve stages.

Jobs are a list of stages; each stage is a subprocess whose output is
streamed line-by-line onto an event queue. The Tk side polls the queue
with after() - no Tk calls happen on worker threads.

The solve stage additionally spawns the detached post-processing watchdog
(tools/postprocess_watchdog.py) so plots + results.json are produced even
if the GUI is closed while SU2 runs.
"""

import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


class Job:
    def __init__(self, name, stages, on_event=None):
        self.name = name
        self.stages = stages          # list of dicts: title, argv, cwd, watch_pid
        self.queue = queue.Queue()
        self.thread = None
        self.proc = None              # currently running subprocess
        self.stop_requested = False
        self.running_stage = None
        self.t_start = None

    # ------------------------------------------------------------- control
    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"job-{self.name}")
        self.t_start = time.time()
        self.thread.start()

    def stop(self):
        self.stop_requested = True
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def elapsed(self):
        return time.time() - self.t_start if self.t_start else 0.0

    # ---------------------------------------------------------------- work
    def _emit(self, kind, **data):
        self.queue.put((kind, data))

    def _run(self):
        try:
            for idx, stage in enumerate(self.stages):
                if self.stop_requested:
                    break
                self.running_stage = stage["title"]
                self._emit("stage", index=idx, title=stage["title"],
                           state="start")
                try:
                    self._run_stage(stage)
                except StageFailed as e:
                    self._emit("stage", index=idx, title=stage["title"],
                               state="failed", message=str(e))
                    self._emit("done", ok=False, mode=self.name,
                               reason=f"{stage['title']} failed")
                    return
                if self.stop_requested:
                    self._emit("stage", index=idx, title=stage["title"],
                               state="stopped")
                    self._emit("done", ok=False, mode=self.name,
                               reason="stopped by user")
                    return
                self._emit("stage", index=idx, title=stage["title"],
                           state="ok")
            self._emit("done", ok=True, mode=self.name,
                       reason="all stages complete")
        except Exception as e:                      # pragma: no cover
            self._emit("done", ok=False, mode=self.name,
                       reason=f"internal error: {e}")

    def _run_stage(self, stage):
        # a solve stage with warm start enabled picks up restart.dat: the
        # patch must happen here because the setup stage re-renders the cfg
        if stage.get("watch_pid") and getattr(self, "restart", False):
            self._patch_restart(stage)
        logf = None
        try:
            if stage.get("logfile"):
                logf = open(stage["logfile"], "w")
                out = logf
            else:
                out = subprocess.PIPE
            self.proc = subprocess.Popen(
                stage["argv"], cwd=str(stage.get("cwd", REPO)),
                stdout=out, stderr=subprocess.STDOUT,
                **_popen_flags())
        except OSError as e:
            raise StageFailed(str(e))

        # the solve stage detaches a watchdog on the SU2 pid before waiting
        if stage.get("watch_pid") and os.name == "nt":
            self._spawn_watchdog(stage)

        if out is subprocess.PIPE:
            for raw in self.proc.stdout:
                self._emit("log", line=raw.decode("utf-8", errors="replace")
                           .rstrip())
        rc = self.proc.wait()
        if logf:
            logf.close()
        if self.stop_requested:
            return
        if rc != 0:
            raise StageFailed(f"exit code {rc}")

    def _patch_restart(self, stage):
        """Point the solve cfg at restart.dat (RESTART_SOL= YES).

        SU2 v8 also needs the restart-input filenames pinned: their
        defaults are 'solution.dat', while every solve writes
        'restart.dat' - without this the warm start cannot find the
        file."""
        case_dir = Path(stage["cwd"])
        self._check_restart_matches_mesh(case_dir)
        cfg_path = case_dir / "turbine.cfg"
        try:
            text = cfg_path.read_text()
            for opt, val in (("RESTART_SOL", "YES"),
                             ("RESTART_FILENAME", "restart.dat"),
                             ("SOLUTION_FILENAME", "restart.dat")):
                if re.search(rf"(?m)^{opt}=", text):
                    text = re.sub(rf"(?m)^{opt}=.*$", f"{opt}= {val}", text)
                else:
                    text = text.rstrip("\n") + f"\n{opt}= {val}\n"
            cfg_path.write_text(text)
            self._emit("log", line=f"[restart] RESTART_SOL= YES, "
                                   f"restart files -> restart.dat "
                                   f"({cfg_path.name})\n")
        except OSError as e:
            self._emit("log", line=f"[restart] could not patch cfg: {e}\n")

    @staticmethod
    def _restart_rows(path):
        """Data rows in an ASCII SU2 restart file (lines minus header)."""
        with open(path, "r", errors="replace") as f:
            return max(sum(1 for _ in f) - 1, 0)

    @staticmethod
    def _mesh_node_count(path):
        with open(path, "r", errors="replace") as f:
            for line in f:
                if line.startswith("NNODES="):
                    return int(line.split("=")[1].strip())
        return None

    def _check_restart_matches_mesh(self, case_dir):
        """A restart.dat written for a different mesh (the geometry changed
        since the last solve) makes SU2 die with a cryptic size error -
        fail the stage here with an actionable message instead."""
        restart, mesh = case_dir / "restart.dat", case_dir / "mesh.su2"
        if not restart.exists() or not mesh.exists():
            return
        try:
            rows = self._restart_rows(restart)
            nodes = self._mesh_node_count(mesh)
        except (OSError, ValueError):
            return
        if nodes and rows and rows != nodes:
            raise StageFailed(
                f"restart.dat holds {rows} points but mesh.su2 has {nodes} "
                "- it was written for a different mesh. Turn 'Initialize "
                "from previous solution' off and re-run the case once (or "
                "restore a matching restart.dat).")

    def _spawn_watchdog(self, stage):
        case_dir = stage["case_dir"]
        wlog = open(Path(case_dir) / "watchdog.log", "w")
        subprocess.Popen(
            [PY, str(REPO / "tools" / "postprocess_watchdog.py"),
             str(case_dir), str(self.proc.pid), PY],
            cwd=str(case_dir), stdout=wlog, stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP)


class StageFailed(Exception):
    pass


def _popen_flags():
    if os.name == "nt":
        # CREATE_NO_WINDOW keeps SU2/python children from popping up a
        # console window when the GUI itself runs without one
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


# ------------------------------------------------------------ stage builders
def mesh_stage(state):
    """Stage dict running pipeline/run_pipeline.py on the current input."""
    return {
        "title": "geometry + mesh (pyturbo-aero, Gmsh)",
        "argv": [PY, str(REPO / "pipeline" / "run_pipeline.py"),
                 str(state.path)],
        "cwd": str(REPO),
    }


def setup_stage(state, skip_validate=False):
    argv = [PY, str(REPO / "tools" / "setup_cascade_case.py"),
            str(state.path.parent), "--name", state.case_name()]
    if skip_validate:
        argv.append("--skip-validate")
    return {
        "title": "SU2 case setup (BCs, mesh scaling, periodic validation)",
        "argv": argv,
        "cwd": str(REPO),
    }


def solve_stage(state, case_dir, threads, su2_exe):
    case_dir = Path(case_dir)
    return {
        "title": f"SU2 solve ({threads} threads)",
        "argv": [str(su2_exe), "-t", str(threads), "turbine.cfg"],
        "cwd": str(case_dir),
        # SU2 output goes to su2_run.log; the Solution tab tails history.csv
        "logfile": str(case_dir / "su2_run.log"),
        "watch_pid": True,
        "case_dir": str(case_dir),
    }


def build_job(state, mode, threads, su2_exe):
    """Assemble a Job for mode: full | setup | mesh | solve."""
    case_dir = state.case_dir()
    stages = []
    if mode in ("full", "setup", "mesh"):
        stages.append(mesh_stage(state))
    if mode in ("full", "setup"):
        stages.append(setup_stage(state))
    if mode in ("full", "solve"):
        stages.append(solve_stage(state, case_dir, threads, su2_exe))
    job = Job(mode, stages)
    job.case_dir = case_dir
    job.restart = bool(state.get("solver_settings.restart"))
    return job
