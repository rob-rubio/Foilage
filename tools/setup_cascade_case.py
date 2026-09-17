"""Bridge the Airfoil_generate_and_mesh pipeline to the SU2 cascade setup.

Reads a mesh-project case directory (input.json with airfoil scale, domain,
BCs, and optional solver_settings/numerics), copies + scales the generated mesh into
this repo's cases/ folder, writes case.json with the BCs and solver settings
applied automatically, renders turbine.cfg, validates the periodic pairing
with a 2-iteration run, and prints (or with --run launches) the live monitor.

Usage:
    python tools/setup_cascade_case.py <mesh_project_case_dir> [--name NAME]
        [--scale SCALE_M] [--remesh] [--skip-validate] [--run]
        [--config foilage.cfg]

    <dir> points at e.g. ...\\Airfoil_generate_and_mesh\\cases\\turbine_blade_3
    (input.json sits there; the mesh lands in <dir>\\cases\\<mesh name>\\).

Units: the mesh pipeline normalizes geometry to axial chord = 1; axial_chord
values >= 1 are treated as millimeters, smaller values as meters. Override
with --scale (meters per chord unit).
"""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from foilage_config import resolve_su2_executable  # noqa: E402

SUPPORTED_TURBULENCE_MODELS = {"SA", "SST"}
DEFAULT_TURBULENCE_MODEL = "SA"
DEFAULT_MAX_ITERATIONS = 6000
DEFAULT_NUMERICS = {
    "limiter": "VENKATAKRISHNAN",
    "gradient": "WEIGHTED_LEAST_SQUARES",
    "cfl": 10.0,
    "cfl_adapt": [0.1, 1.2, 5.0, 40.0],
    "linear_solver_iter": 100,
}


def load_bc(cfg):
    """Robust lookup of the BCs block (keys may contain spaces/case)."""
    def norm(d):
        return {str(k).strip().lower(): v for k, v in d.items()}
    bcs = norm(cfg.get("BCs", {}))
    if "inlet" not in bcs or "outlet" not in bcs:
        sys.exit("input.json has no usable 'BCs' block (need inlet + outlet)")
    inl = norm(bcs["inlet"])
    out = norm(bcs["outlet"])
    need = ["total pressure", "total temperature", "gas angle"]
    missing = [k for k in need if k not in inl]
    if missing or "static pressure" not in out:
        sys.exit(f"BCs block incomplete: missing {missing + ['outlet static pressure']}")
    return (float(inl["total pressure"]), float(inl["total temperature"]),
            float(inl["gas angle"]), float(out["static pressure"]))


def load_solver_settings(cfg):
    """Read and validate the SU2 settings exposed by the mesh input file."""
    settings = cfg.get("solver_settings", {})
    if not isinstance(settings, dict):
        sys.exit("input.json 'solver_settings' must be an object")

    model = settings.get("turbulence_model", DEFAULT_TURBULENCE_MODEL)
    if not isinstance(model, str) or not model.strip():
        sys.exit("solver_settings.turbulence_model must be a non-empty string")
    model = model.strip().upper()
    if model not in SUPPORTED_TURBULENCE_MODELS:
        supported = ", ".join(sorted(SUPPORTED_TURBULENCE_MODELS))
        sys.exit(f"unsupported turbulence model {model!r}; choose one of: {supported}")

    max_iterations = settings.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if (isinstance(max_iterations, bool) or
            not isinstance(max_iterations, int) or max_iterations < 1):
        sys.exit("solver_settings.max_iterations must be a positive integer")

    return model, max_iterations


def load_numerics(cfg):
    """Read numerical settings from input.json for the generated case.json.

    Accept the block at the top level or nested under solver_settings so both
    input layouts remain usable.
    """
    settings = cfg.get("solver_settings", {})
    if not isinstance(settings, dict):
        sys.exit("input.json 'solver_settings' must be an object")
    values = cfg["numerics"] if "numerics" in cfg else settings.get("numerics", {})
    if not isinstance(values, dict):
        sys.exit("input.json 'numerics' must be an object")

    numerics = DEFAULT_NUMERICS.copy()
    numerics.update(values)
    if isinstance(numerics["cfl_adapt"], list):
        numerics["cfl_adapt"] = list(numerics["cfl_adapt"])
    return numerics


def measure_surface_probe(pts, af_nodes, frac=0.35):
    """Suction-side wall point + outward normal at frac of axial chord,
    plus a zoom window around the blade (all in mesh units)."""
    a = pts[af_nodes]
    xle = a[:, 0].min()
    cax = a[:, 0].max() - xle

    def envelope(nodes, upper):
        out = []
        for xq in np.linspace(0.15, 0.85, 141):
            w = nodes[np.abs(nodes[:, 0] - (xle + xq * cax)) < 0.004 * cax]
            if len(w):
                out.append(w[np.argmax(w[:, 1]) if upper else np.argmin(w[:, 1])])
        return np.array(out)

    mid = a[((a[:, 0] - xle) / cax > 0.1) & ((a[:, 0] - xle) / cax < 0.9)]
    ss = envelope(mid, True)   # suction side = upper envelope (left-to-right vane)
    k = int(np.argmin(np.abs((ss[:, 0] - xle) / cax - frac)))
    p0 = ss[k]
    i0, i1 = max(k - 4, 0), min(k + 4, len(ss) - 1)
    tan = ss[i1] - ss[i0]
    tan /= np.linalg.norm(tan)
    nrm = np.array([tan[1], -tan[0]])
    if nrm[1] < 0:
        nrm = -nrm
    nrm /= np.linalg.norm(nrm)
    zoom = {"xmin": -0.15 * cax, "xmax": 0.55 * cax,
            "ymin": -0.85 * cax, "ymax": 0.90 * cax}
    return p0, nrm, zoom


def parse_airfoil_mesh(path):
    with open(path) as f:
        lines = f.read().splitlines()
    i, pts, af = 0, None, None
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("NPOIN="):
            n = int(ln.split("=")[1].split()[0])
            pts = np.zeros((n, 2))
            for j in range(n):
                p = lines[i + 1 + j].split()
                pts[j, 0], pts[j, 1] = float(p[0]), float(p[1])
            i += n + 1
            continue
        if ln.startswith("MARKER_TAG= airfoil"):
            cnt = int(lines[i + 1].split("=")[1].split()[0])
            nodes = set()
            for j in range(cnt):
                p = [int(x) for x in lines[i + 2 + j].split()]
                nodes.update(p[1:-1] if len(p) > 3 else p[1:3])
            af = np.array(sorted(nodes))
            break
        i += 1
    return pts, af


def scale_mesh(path, scale):
    with open(path) as f:
        lines = f.read().splitlines()
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("NPOIN="):
            n = int(ln.split("=")[1].split()[0])
            out.append(ln)
            for j in range(n):
                parts = lines[i + 1 + j].split()
                x, y = float(parts[0]) * scale, float(parts[1]) * scale
                idx = parts[2] if len(parts) > 2 else ""
                out.append(f"\t{repr(x)}\t{repr(y)}\t{idx}".rstrip())
            i += n + 1
            continue
        out.append(ln)
        i += 1
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh_case_dir", type=Path)
    ap.add_argument("--name", default=None, help="SU2 case name (default: folder name)")
    ap.add_argument("--scale", type=float, default=None,
                    help="meters per chord unit (default: axial_chord mm->m if >=1)")
    ap.add_argument("--remesh", action="store_true",
                    help="re-run the meshing pipeline first")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="launch the live monitor when setup is done")
    ap.add_argument("--config", type=Path, default=None,
                    help="Foilage runtime config (default: foilage.cfg)")
    args = ap.parse_args()

    if not args.skip_validate:
        try:
            su2_exe = resolve_su2_executable(args.config)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(str(e))
        if not su2_exe.exists():
            sys.exit(f"SU2 not found at {su2_exe}")

    case_root = args.mesh_case_dir.resolve()
    cfg = json.loads((case_root / "input.json").read_text())
    turbulence_model, max_iterations = load_solver_settings(cfg)
    numerics = load_numerics(cfg)
    mesh_name = cfg["case"]["name"]
    su2_name = args.name or case_root.name
    su2_dir = REPO / "cases" / su2_name
    su2_dir.mkdir(parents=True, exist_ok=True)

    # ---- mesh (regenerate on request or if missing)
    mesh_src = case_root / "cases" / mesh_name / "mesh_quad.su2"
    if not mesh_src.exists():
        mesh_src = case_root / "cases" / mesh_name / "mesh.su2"
    if args.remesh or not mesh_src.exists():
        proj = case_root.parent.parent
        py = Path(sys.executable)
        print(f"[mesh] running pipeline ({'--remesh' if args.remesh else 'mesh missing'}) ...")
        subprocess.run([str(py), str(proj / "pipeline" / "run_pipeline.py"),
                        str(case_root / "input.json")], check=True)
    if not mesh_src.exists():
        sys.exit(f"mesh not found at {mesh_src} (run with --remesh)")

    # ---- geometry scale
    ac = float(cfg["airfoil"]["axial_chord"])
    scale = args.scale if args.scale is not None else (ac / 1000.0 if ac >= 1.0 else ac)
    print(f"[scale] axial_chord = {ac} -> {scale} m per chord unit")

    # ---- domain / periodic translation
    dom = cfg["domain"]
    if abs(dom["R1"] - dom["R2"]) > 1e-9 * max(dom["R1"], dom["R2"], 1.0):
        sys.exit("R1 != R2 (varying pitch) cannot be paired by SU2 - "
                 "regenerate the mesh with R1 = R2 (see the_process.md)")
    pitch_m = 2.0 * math.pi * dom["R1"] / dom["airfoil_count"] * scale

    # ---- BCs from input.json
    p01, T01, ang, p2 = load_bc(cfg)
    rad = math.radians(ang)
    direction = [math.cos(rad), math.sin(rad), 0.0]
    print(f"[bcs] inlet: p0={p01} Pa, T0={T01} K, angle={ang} deg | outlet: p={p2} Pa")

    # ---- copy + scale mesh
    (su2_dir / "mesh.su2").write_bytes(mesh_src.read_bytes())
    scale_mesh(su2_dir / "mesh.su2", scale)

    # ---- postprocess hints measured on this airfoil
    pts, af = parse_airfoil_mesh(su2_dir / "mesh.su2")
    p0, nrm, zoom = measure_surface_probe(pts, af)

    # ---- reference / init state
    g, R = 1.4, 287.058
    T2 = T01 * (p2 / p01) ** ((g - 1) / g)
    V2 = math.sqrt(2 * g / (g - 1) * R * (T01 - T2))
    rho2 = p2 / (R * T2)
    mu2 = 1.716e-5 * (T2 / 273.15) ** 1.5 * (383.55 / (T2 + 110.4))
    Re = rho2 * V2 * scale / mu2
    print(f"[refs] isentropic exit: M={V2/math.sqrt(g*R*T2):.3f} V={V2:.1f} m/s | "
          f"Re_axial = {Re:.3g}, pitch = {pitch_m:.6f} m")

    case = {
        "name": su2_name,
        "description": (f"Turbine vane cascade auto-setup from {case_root}. "
                        f"BCs and solver settings read from input.json; "
                        f"mesh scaled {scale} m/chord."),
        "mesh": "mesh.su2",
        "physics": {
            "solver": "RANS",
            "turbulence_model": turbulence_model,
            "mach": 0.65,
            "reynolds": Re,
            "reynolds_length": scale,
            "init_pressure": 100000.0,
            "init_temperature": 660.0,
            "gamma": g,
            "gas_constant": R,
        },
        "cascade": {
            "inlet": {"marker": "inlet", "total_pressure": p01,
                      "total_temperature": T01, "direction": direction},
            "outlet": {"marker": "outlet", "static_pressure": p2},
            "periodic": [{"markers": ["periodic_bottom", "periodic_top"],
                          "translation": [0.0, pitch_m, 0.0]}],
        },
        "markers": {
            "airfoil": {"bc": "wall_adiabatic"},
            "inlet": {"bc": "inlet", "analyze": True},
        },
        "numerics": numerics,
        "convergence": {"fields": ["RMS_DENSITY"], "minval": -6.0,
                        "startiter": 100, "iterations": max_iterations},
        "postprocess": {
            "wall_hint": [round(float(p0[0]), 9), round(float(p0[1]), 9)],
            "probe": {"point": [round(float(p0[0]), 9), round(float(p0[1]), 9)],
                      "normal": [round(float(nrm[0]), 4), round(float(nrm[1]), 4)],
                      "length": round(0.5 * scale, 6)},
            "zoom": {k: round(v, 6) for k, v in zoom.items()},
        },
    }
    (su2_dir / "case.json").write_text(json.dumps(case, indent=2))
    print(f"[case] wrote {su2_dir / 'case.json'}")

    # ---- render config
    subprocess.run([sys.executable, str(REPO / "tools" / "render_config.py"),
                    str(su2_dir / "case.json"),
                    str(REPO / "templates" / "su2_cascade.cfg"),
                    str(su2_dir / "turbine.cfg")], check=True)

    # ---- validate periodic pairing + BCs (2 iterations)
    if not args.skip_validate:
        # the validate run overwrites history/restart/solution outputs -
        # preserve any previous results first
        prev = [f for f in ("history.csv", "restart.dat", "vol_solution.vtk",
                            "vol_solution.vtu", "surface.vtk", "surface.vtu")
                if (su2_dir / f).exists()]
        if prev:
            from datetime import datetime
            bdir = su2_dir / f"results_backup_{datetime.now():%Y%m%d_%H%M%S}"
            bdir.mkdir()
            for f in prev:
                (su2_dir / f).replace(bdir / f)
            print(f"[validate] previous results moved to {bdir.name}")
        text = (su2_dir / "turbine.cfg").read_text()
        text = re.sub(r"(?m)^ITER=[ \t]+\d+[ \t]*$", "ITER= 2", text)
        text = re.sub(r"(?m)^CONV_STARTITER=[ \t]+\d+[ \t]*$",
                      "CONV_STARTITER= 2", text)
        (su2_dir / "validate.cfg").write_text(text)
        r = subprocess.run([str(su2_exe),
                            "-t", "6", "validate.cfg"],
                           cwd=str(su2_dir), capture_output=True, text=True, timeout=300)
        log = r.stdout + r.stderr
        if r.returncode != 0 or "Matched" not in log:
            sys.exit(f"validation FAILED:\n{log[-1500:]}")
        for ln in log.splitlines():
            if "Matched" in ln:
                print(f"[validate] {ln.strip()}")
        (su2_dir / "validate.cfg").unlink()
        for f in ("history.csv", "restart.dat", "vol_solution.vtk", "vol_solution.vtu",
                  "surface.vtk", "surface.vtu"):
            (su2_dir / f).unlink(missing_ok=True)

    run_cmd = (f'python tools\\run_monitor.py cases\\{su2_name} '
               f'turbine.cfg -t 6')
    print(f"[done] SU2 case ready: {su2_dir}")
    print(f"[next] {run_cmd}")
    if args.run:
        monitor_cmd = [sys.executable, str(REPO / "tools" / "run_monitor.py"),
                       str(su2_dir), "turbine.cfg", "-t", "6"]
        if args.config:
            monitor_cmd += ["--config", str(args.config)]
        subprocess.Popen(monitor_cmd)


if __name__ == "__main__":
    main()
