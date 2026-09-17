"""One-command pipeline: geometry -> mesh -> SU2 solve -> post-process.

Usage (any cwd, from anywhere):
    python run_full_pipeline [INPUT_JSON] [--name NAME] [--threads N]
                             [--skip-mesh] [--no-run]

INPUT_JSON defaults to <repo>/input.json; a per-case input like
cases/turbine_blade_4/input.json works too. Stages:

    1. geometry + mesh   pipeline/run_pipeline.py (pyturbo-aero + gmsh +
                         Gmsh Blossom recombination), driven by input.json
    2. SU2 case setup    tools/setup_cascade_case.py reads the BCs block from
                         the same input.json, scales the mesh to meters,
                         writes case.json + turbine.cfg, validates the
                         periodic pairing (2 iterations)
    3. solve + monitor   tools/run_monitor.py opens the live convergence GUI;
                         post-processing (plots + results.json) runs
                         automatically when the solve stops or finishes

The command returns when you close the monitor window, then prints a summary
from results.json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = REPO / "venv_2d_cfd" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)  # fall back to whatever python is running us


def stage(title, cmd, **kw):
    print(f"\n{'=' * 70}\n== {title}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, **kw)
    dt = time.time() - t0
    print(f"== {title}: {'OK' if r.returncode == 0 else 'FAILED'} "
          f"({dt:.0f} s)", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def print_summary(res_path):
    r = json.loads(res_path.read_text())
    print(f"\n{'=' * 70}\n== RESULTS ({res_path})\n{'=' * 70}")
    c = r.get("convergence", {})
    print(f"  iterations run : {c.get('iterations_run')}   "
          f"rms target reached: {c.get('target_reached')}")
    for k, v in c.get("rms_final", {}).items():
        print(f"    {k:10s} = {v: .2f}")
    f = r.get("forces", {})
    if f:
        print(f"  forces         : CD = {f.get('CD', 0):.4f}  "
              f"CL = {f.get('CL', 0):.4f}")
    if "inlet" in r and "outlet" in r:
        i, o = r["inlet"], r["outlet"]
        print(f"  mass flow      : {i['mass_flow_kg_s_m']:.3f} kg/(s.m)   "
              f"imbalance {r.get('mass_balance', {}).get('imbalance_pct', 0):.3f}%")
        print(f"  outlet state   : M {o['mach']:.3f}   V {o['velocity_m_s']:.0f} m/s   "
              f"angle {o['flow_angle_deg']:+.1f} deg   p {o['static_p_pa']:.0f} Pa")
        print(f"  loss Yp        : {r.get('losses', {}).get('total_pressure_loss_coeff_Yp', 0):.4f}")
    w = r.get("wall", {})
    if w:
        print(f"  wall y+        : median {w.get('yplus_median', 0):.1f}   "
              f"p95 {w.get('yplus_p95', 0):.1f}")
    print(f"  max Mach       : {r.get('fields', {}).get('max_mach', 0):.3f}")


def main():
    ap = argparse.ArgumentParser(
        description="geometry -> mesh -> SU2 -> post-process, one command")
    ap.add_argument("input", nargs="?", default=None,
                    help="pipeline input.json (default: <repo>/input.json)")
    ap.add_argument("--name", default=None,
                    help="case name (default: case.name from the input json)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--skip-mesh", action="store_true",
                    help="reuse the existing mesh, start at SU2 setup")
    ap.add_argument("--no-run", action="store_true",
                    help="stop after setup (no solve)")
    args = ap.parse_args()

    inp = Path(args.input) if args.input else REPO / "input.json"
    if not inp.is_absolute():
        inp = (Path.cwd() / inp)
    if not inp.exists():
        cands = sorted(p for p in (REPO / "cases").glob("*/input.json"))
        sys.exit(f"input not found: {inp}\n"
                 f"available case inputs:\n" +
                 "\n".join(f"  {p}" for p in cands))
    inp = inp.resolve()
    cfg = json.loads(inp.read_text())
    name = args.name or cfg.get("case", {}).get("name", inp.parent.name)

    print(f"repo : {REPO}\ninput: {inp}\ncase : {name}")

    if not args.skip_mesh:
        stage("1/3  geometry + mesh (pyturbo-aero, gmsh Blossom)",
              [str(PY), str(REPO / "pipeline" / "run_pipeline.py"), str(inp)],
              cwd=str(REPO))

    stage("2/3  SU2 case setup (BCs from input.json, mesh scaled to meters)",
          [str(PY), str(REPO / "tools" / "setup_cascade_case.py"),
           str(inp.parent), "--name", name])

    if args.no_run:
        print(f"\ncase ready: {REPO / 'cases' / name}  "
              f"(solve later with tools\\run_monitor.py)")
        return

    print(f"\nThe live convergence window is opening; post-processing runs "
          f"automatically when the solve stops or finishes.\n")
    stage(f"3/3  SU2 solve + live monitor ({args.threads} threads)",
          [str(PY), str(REPO / "tools" / "run_monitor.py"),
           str(REPO / "cases" / name), "turbine.cfg",
           "-t", str(args.threads)])

    res = REPO / "cases" / name / "results.json"
    if res.exists():
        print_summary(res)
    else:
        print("\nresults.json not found - check postprocess.log / su2_run.log "
              f"in {REPO / 'cases' / name}")


if __name__ == "__main__":
    main()
