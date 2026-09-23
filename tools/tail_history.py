"""Read-only live view of an SU2 history.csv.

Usage:
    python tools/tail_history.py <case_dir> [--interval 2.0]

Plots residuals, force coefficients and the inlet/outlet mass flow from
<case_dir>/history.csv while a solve is running. This NEVER launches or
touches the solver - safe to attach to a run whose monitor window is gone
(the monitor itself launches SU2; this script only reads).

Self-test (no window):
    python tools/tail_history.py --selftest
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

POLL_S = 2.0


def read_history(path: Path):
    """Parse history.csv into {column: array} (empty dict if unusable)."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = f.read()
    lines = raw.decode("ascii", errors="replace").splitlines()
    if not lines:
        return {}
    header = [h.strip().strip('"').strip("'") for h in
              lines[0].split(",")]
    arrays = [[] for _ in header]
    for ln in lines[1:]:
        parts = [p.strip().strip('"').strip("'") for p in ln.split(",")]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) != len(header):
            continue
        for a, v in zip(arrays, vals):
            a.append(v)
    return {name: np.array(arr) for name, arr in zip(header, arrays)
            if len(arr)}


def selftest():
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "history.csv"
    tmp.write_text('"Time_Iter","Inner_Iter","rms[Rho]","CD"\n'
                   "0, 0, -3.2, 1.0\n"
                   "0, 1, -4.2, 1.1\n")
    cols = read_history(tmp)
    assert list(cols) == ["Time_Iter", "Inner_Iter", "rms[Rho]", "CD"]
    assert cols["rms[Rho]"].tolist() == [-3.2, -4.2]
    assert read_history(Path(tempfile.mkdtemp()) / "missing.csv") == {}
    print("selftest OK")


def watch(case_dir: Path, interval: float):
    path = case_dir / "history.csv"
    plt.ion()
    fig, (ax_res, ax_force, ax_mass) = plt.subplots(
        3, 1, figsize=(9.5, 7.2), num=f"history tail - {case_dir.name}")
    fig.tight_layout(pad=3.0)

    while plt.fignum_exists(fig.number):
        cols = read_history(path)
        it = cols.get("Inner_Iter")
        ax_res.clear()
        ax_force.clear()
        ax_mass.clear()
        plotted = False
        if it is not None and len(it):
            for name, arr in cols.items():
                if name.startswith("rms[") and len(arr) == len(it):
                    ax_res.plot(it, arr, lw=1, label=name.replace("rms", "rms "))
                    plotted = True
            if plotted:
                ax_res.legend(fontsize=7, ncol=2, loc="upper right")
            ax_res.set_title(f"residuals (log10) - {len(it)} iterations",
                             fontsize=9, loc="left")
            for name, color in (("CD", "tab:red"), ("CL", "tab:blue")):
                if name in cols and len(cols[name]) == len(it):
                    ax_force.plot(it, cols[name], lw=1.2, color=color,
                                  label=name)
            ax_force.legend(fontsize=7, loc="upper right")
            ax_force.set_title("force coefficients", fontsize=9, loc="left")
            mi = cols.get("Avg_Massflow(inlet)")
            mo = cols.get("Avg_Massflow(outlet)")
            if mi is not None and len(mi) == len(it):
                ax_mass.plot(it, mi, lw=1.2, color="tab:green",
                             label="inlet")
                ax_mass.plot(it, -np.asarray(mo, dtype=float), lw=1.2,
                             color="tab:orange", label="-outlet")
                ax_mass.legend(fontsize=7, loc="upper right")
                ax_mass.set_title("mass flow in vs out "
                                  "(curves coincide when conservative)",
                                  fontsize=9, loc="left")
            else:
                ax_mass.set_title("mass flow (needs FLOW_COEFF_SURF)",
                                  fontsize=9, loc="left")
            for ax in (ax_res, ax_force, ax_mass):
                ax.grid(True, alpha=0.3)
            fig.canvas.draw_idle()
        plt.pause(interval)


def main():
    ap = argparse.ArgumentParser(description="Read-only live history plot")
    ap.add_argument("case_dir", nargs="?", type=Path)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.case_dir is None:
        ap.error("case_dir is required unless --selftest is used")
    watch(args.case_dir.resolve(), args.interval)


if __name__ == "__main__":
    main()
