"""Live SU2 convergence GUI - launches the solver and watches history.csv.

Usage:
    python tools/run_monitor.py <case_dir> [cfg_name] [-t THREADS]
        [--config foilage.cfg]

Launches SU2_CFD on <case_dir>/<cfg_name> (default cylinder.cfg) as a
background process and opens a Tk window with live panels:
    - residuals (all rms[*] columns, log scale)
    - drag/lift coefficients + running mean CD
    - mass flow per unit depth (SURFACE_MASSFLOW, if present in history)
    - status strip (state, progress, ETA) and log tail

Self-test (no solver, no window):
    python tools/run_monitor.py --selftest
"""

import csv
import argparse
import os
import re
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from foilage_config import resolve_su2_executable  # noqa: E402

POLL_MS = 500
LOG_TAIL_LINES = 8


# --------------------------------------------------------------- data source
class HistoryTail:
    """Incremental tailer for SU2's history.csv (handles partial writes)."""

    def __init__(self, path: Path):
        self.path = path
        self._offset = 0
        self._partial = b""
        self.columns = None
        self.data = {}          # column -> list[float]
        self.rows = 0

    def poll(self):
        """Read newly completed lines; update self.data. Returns rows added."""
        if not self.path.exists():
            return 0
        added = 0
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = self._partial + f.read()
            self._offset = f.tell()
        lines = chunk.split(b"\n")
        self._partial = lines.pop()          # last piece may be incomplete
        for raw in lines:
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            parts = [p.strip().strip('"').strip("'") for p in line.split(",")]
            if self.columns is None:
                if parts and parts[0].startswith("Time_Iter"):
                    self.columns = parts
                    self.data = {c: [] for c in parts}
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) != len(self.columns):
                continue
            for c, v in zip(self.columns, vals):
                self.data[c].append(v)
            self.rows += 1
            added += 1
        return added

    def get(self, col):
        return self.data.get(col)


# ----------------------------------------------------------------- utilities
def parse_cfg_totals(cfg_path: Path):
    """Return (kind, total) - total solver units for progress estimation."""
    text = cfg_path.read_text(errors="ignore") if cfg_path.exists() else ""
    m = re.search(r"^TIME_ITER=\s*(\d+)", text, re.M)
    if m:
        return "unsteady", int(m.group(1))
    m = re.search(r"^ITER=\s*(\d+)", text, re.M)
    if m:
        return "steady", int(m.group(1))
    return "steady", None


def log_state(log_path: Path, running: bool):
    """Classify run state from the log tail."""
    if log_path.exists():
        tail = log_path.read_text(errors="ignore")[-4000:]
        if "Exit Success" in tail:
            return "finished"
        if "NaN detected" in tail or "Error in" in tail:
            return "failed"
    return "running" if running else "stopped"


def read_log_tail(log_path: Path, n=LOG_TAIL_LINES):
    if not log_path.exists():
        return "waiting for su2_run.log ..."
    with open(log_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 4000))
        text = f.read().decode("ascii", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


# ------------------------------------------------------------------ selftest
def selftest():
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "history.csv"
    tail = HistoryTail(tmp)
    assert tail.poll() == 0, "empty file should give 0 rows"

    tmp.write_text('"Time_Iter","Inner_Iter","rms[Rho]","CD"\n')
    assert tail.poll() == 0
    assert tail.columns and tail.columns[0] == "Time_Iter", "header should register"

    with open(tmp, "a") as f:
        f.write(" 0, 0, -6.2, 48.4\n")          # complete row, no newline yet
        f.write(" 0, 1, -7.5, 4")                # partial row (cut mid-value)
    added = tail.poll()
    assert added == 1, f"partial line must not parse, got {added}"

    with open(tmp, "a") as f:
        f.write("8.4\n 0, 2, -8.1, 2.0\n")
    added = tail.poll()
    assert added == 2, f"completed partial + new row expected, got {added}"
    assert tail.get("rms[Rho]") == [-6.2, -7.5, -8.1]
    assert tail.get("CD") == [48.4, 48.4, 2.0]
    assert tail.rows == 3

    # real file from the finished laminar case
    real = REPO / "cases" / "cylinder_mach05" / "history.csv"
    if real.exists():
        t2 = HistoryTail(real)
        n = t2.poll()
        assert n == t2.rows and t2.rows > 100
        assert "rms[Rho]" in t2.data and "CD" in t2.data
        print(f"selftest: parsed real history ({t2.rows} rows, "
              f"final CD={t2.get('CD')[-1]:.3f})")
    print("selftest OK")


# ----------------------------------------------------------------------- GUI
class MonitorApp:
    def __init__(self, case_dir: Path, cfg_name: str, threads: int,
                 su2_exe: Path):
        self.case_dir = case_dir
        self.cfg_name = cfg_name
        self.su2_exe = su2_exe
        self.cfg_path = case_dir / cfg_name
        self.log_path = case_dir / "su2_run.log"
        self.tail = HistoryTail(case_dir / "history.csv")
        self.kind, self.total = parse_cfg_totals(self.cfg_path)
        self.t0 = time.time()
        self.proc = None
        self.running = False

        self._build_ui()
        self._start_solver(threads)
        self.root.after(POLL_MS, self._tick)

    # ---------------- UI
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title(f"SU2 monitor - {self.case_dir.name}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)
        self.status = tk.Label(top, text="starting ...", anchor="w",
                               font=("TkDefaultFont", 10, "bold"))
        self.status.pack(side=tk.LEFT, padx=8, pady=4)
        self.stop_btn = tk.Button(top, text="Stop run", command=self._stop)
        self.stop_btn.pack(side=tk.RIGHT, padx=8, pady=4)

        self.fig = Figure(figsize=(10.5, 7.2), dpi=96)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[2.4, 1.8, 1.5], hspace=0.75)
        self.ax_res = self.fig.add_subplot(gs[0])
        self.ax_force = self.fig.add_subplot(gs[1], sharex=self.ax_res)
        self.ax_mass = self.fig.add_subplot(gs[2], sharex=self.ax_res)
        for ax, title in ((self.ax_res, "residuals (log10)"),
                          (self.ax_force, "force coefficients"),
                          (self.ax_mass, "mass flow per unit depth  [kg/(s·m)]")):
            ax.set_title(title, fontsize=9, loc="left")
            ax.grid(True, alpha=0.3)
        self.ax_res.set_yscale("linear")     # residuals are log10 values already

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.log_box = tk.Text(self.root, height=LOG_TAIL_LINES, font=("Consolas", 8),
                               bg="#111", fg="#8f8")
        self.log_box.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_box.insert("1.0", "waiting for output ...")

    def _start_solver(self, threads):
        cmd = [str(self.su2_exe), "-t", str(threads), self.cfg_name]
        try:
            logf = open(self.log_path, "w")
            self.proc = subprocess.Popen(cmd, cwd=str(self.case_dir),
                                         stdout=logf, stderr=subprocess.STDOUT)
            self.running = True
            # detached watchdog: waits for the solver to exit (however that
            # happens) and runs the post-processing independent of this GUI
            wlog = open(self.case_dir / "watchdog.log", "w")
            subprocess.Popen(
                [sys.executable, str(REPO / "tools" / "postprocess_watchdog.py"),
                 str(self.case_dir), str(self.proc.pid), sys.executable],
                cwd=str(self.case_dir), stdout=wlog, stderr=subprocess.STDOUT,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        except OSError as e:
            self.status.config(text=f"failed to start SU2: {e}", fg="red")

    # ---------------- polling
    def _tick(self):
        if self.running:
            self.tail.poll()

        xcol = self._x_column()
        x = self.tail.get(xcol) or []
        self._draw_residuals(x)
        self._draw_forces(x)
        self._draw_massflow(x)
        self._update_status(x)
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", read_log_tail(self.log_path))

        self.canvas.draw_idle()
        if self.running and self.proc and self.proc.poll() is not None:
            self.running = False
        if not self.running and (self.case_dir / "results.json").exists():
            self.status.config(text=self.status.cget("text").split(
                "   |   post-processing")[0] + "   |   results.json ready",
                fg="green")
        self.root.after(POLL_MS, self._tick)

    def _x_column(self):
        t = self.tail.get("Time_Iter")
        if t and max(t) > 0:
            return "Time_Iter"
        if self.tail.rows:
            cols = self.tail.columns or []
            return "Inner_Iter" if "Inner_Iter" in cols else None
        return "Inner_Iter"

    def _draw_residuals(self, x):
        ax = self.ax_res
        ax.clear()
        ax.set_title("residuals (log10)", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        plotted = False
        if x:
            for col in (self.tail.columns or []):
                if col.startswith("rms["):
                    ax.plot(x, self.tail.get(col), lw=1, label=col.replace("rms", "rms "))
                    plotted = True
            if plotted:
                ax.legend(fontsize=7, ncol=2, loc="upper right")

    def _draw_forces(self, x):
        ax = self.ax_force
        ax.clear()
        ax.set_title("force coefficients", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        cd = self.tail.get("CD")
        cl = self.tail.get("CL")
        if x and cd:
            ax.plot(x, cd, lw=1.2, color="tab:red", label="CD")
            ax.plot(x, cl, lw=1.2, color="tab:blue", label="CL")
            if self.kind == "unsteady" and len(cd) > 20:      # running mean CD
                win = max(1, len(cd) // 8)
                mean = np.convolve(cd, np.ones(win) / win, mode="valid")
                ax.plot(x[win - 1:], mean, lw=2, color="black", alpha=0.6,
                        label="CD running mean")
            ax.legend(fontsize=7, loc="upper right")

    def _draw_massflow(self, x):
        ax = self.ax_mass
        ax.clear()
        ax.set_title("mass flow per unit depth  [kg/(s·m)]", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        mdot = None
        for col in ("SURFACE_MASSFLOW", "Avg_Massflow"):
            mdot = self.tail.get(col)
            if mdot:
                break
        if x and mdot:
            ax.plot(x, mdot, lw=1.2, color="tab:green")
            tail_n = max(1, len(mdot) // 5)
            mean = float(np.mean(mdot[-tail_n:]))
            ax.axhline(mean, color="k", ls=":", lw=1)
            ax.annotate(f"mean {mean:.4f}", xy=(0.99, mean),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no mass-flow column in history.csv",
                    transform=ax.transAxes, ha="center", fontsize=8, color="0.4")

    def _update_status(self, x):
        state = log_state(self.log_path, self.running)
        elapsed = time.time() - self.t0
        n = len(x) if x else 0
        parts = [f"state: {state}", f"rows: {self.tail.rows}",
                 f"elapsed: {elapsed:.0f}s"]
        if self.total and x:
            cur = x[-1]
            frac = min(1.0, cur / self.total) if self.kind == "unsteady" else \
                min(1.0, (x[-1] + 1) / self.total)
            parts.append(f"progress: {cur}/{self.total} ({100 * frac:.0f}%)")
            if frac > 0.02 and n > 1:
                rate = elapsed / n
                parts.append(f"ETA: {rate * (n / frac - n):.0f}s")
        color = {"running": "black", "finished": "green",
                 "failed": "red", "stopped": "orange"}[state]
        self.status.config(text="   |   ".join(parts), fg=color)

    # ---------------- lifecycle
    def _stop(self):
        if self.proc and self.proc.poll() is None:
            if not tk.messagebox.askokcancel("Stop run", "Terminate the SU2 run?"):
                return
            self.proc.terminate()
            self.running = False
            self.status.config(text="state: stopped by user", fg="orange")

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            if not tk.messagebox.askokcancel("Quit", "Closing kills the running SU2 process. Quit?"):
                return
            self.proc.terminate()
            # the detached watchdog post-processes once the solver exits
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Run SU2 with a live convergence monitor")
    ap.add_argument("case_dir", nargs="?", type=Path)
    ap.add_argument("cfg_name", nargs="?", default="cylinder.cfg")
    ap.add_argument("-t", "--threads", type=int, default=6)
    ap.add_argument("--config", type=Path, default=None,
                    help="Foilage runtime config (default: foilage.cfg)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.case_dir is None:
        ap.error("case_dir is required unless --selftest is used")
    try:
        su2_exe = resolve_su2_executable(args.config)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    if not su2_exe.exists():
        sys.exit(f"SU2 not found at {su2_exe}")
    case_dir = args.case_dir
    if not case_dir.exists():
        sys.exit(f"case dir not found: {case_dir}")
    MonitorApp(case_dir.resolve(), args.cfg_name, args.threads, su2_exe).run()


if __name__ == "__main__":
    main()
