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
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
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
    return job
