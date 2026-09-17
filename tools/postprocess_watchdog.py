"""Detached post-processing watchdog for a SU2 run.

Spawned by run_monitor.py when the solver starts. It waits for the SU2
process to exit - no matter how (Stop button, monitor window closed, monitor
or solver killed externally, natural finish) - then runs plot_case exactly
once to produce the plots and results.json.

Usage (spawn detached):
    python postprocess_watchdog.py <case_dir> <su2_pid> [python_exe]
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def wait_for_exit(pid: int):
    """Block until the given pid exits (Windows handle wait)."""
    import ctypes
    SYNCHRONIZE = 0x00100000
    INFINITE = 0xFFFFFFFF
    h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if h:
        try:
            ctypes.windll.kernel32.WaitForSingleObject(h, INFINITE)
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
        return
    # process already gone (or not ours): give the OS a moment to flush files
    time.sleep(5.0)


def main():
    case_dir = Path(sys.argv[1]).resolve()
    pid = int(sys.argv[2])
    py = sys.argv[3] if len(sys.argv) > 3 else sys.executable

    wait_for_exit(pid)

    lock = case_dir / "postprocess.lock"
    for _ in range(120):  # another post-process may be running; wait for it
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(5.0)
    else:
        sys.exit("lock never released")

    try:
        results = case_dir / "results.json"
        history = case_dir / "history.csv"
        # skip if a results.json newer than the history already exists
        if (results.exists() and history.exists()
                and results.stat().st_mtime >= history.stat().st_mtime):
            return
        subprocess.run([py, str(REPO / "tools" / "plot_case.py"), str(case_dir)],
                       cwd=str(case_dir))
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


if __name__ == "__main__":
    main()
