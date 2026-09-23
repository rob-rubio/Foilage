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
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# keep spawned solver/python processes from opening console windows when
# this script runs from a windowless parent (e.g. the Foilage GUI)
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))
from foilage_config import resolve_su2_executable  # noqa: E402
from freestream import gamma_of_air, sutherland_mu  # noqa: E402

SUPPORTED_TURBULENCE_MODELS = {"SA", "SST"}
DEFAULT_TURBULENCE_MODEL = "SA"
DEFAULT_MAX_ITERATIONS = 6000
CONV_FIELD_PRESETS = {
    # residuals SU2 must drive below the convergence floor (ALL of them)
    "density_energy": ["RMS_DENSITY", "RMS_ENERGY"],
    "density": ["RMS_DENSITY"],
}
DEFAULT_CONV_CRITERION = "density_energy"
DEFAULT_NUMERICS = {
    "limiter": "VENKATAKRISHNAN",
    "gradient": "WEIGHTED_LEAST_SQUARES",
    "cfl": 10.0,
    "cfl_adapt": [0.1, 1.2, 5.0, 40.0],
    "linear_solver_iter": 100,
}


def load_conv_fields(cfg):
    """Convergence fields from input.json (solver_settings.convergence_criterion).

    'density_energy' (default) requires BOTH the density and energy
    residuals to fall below the floor: stopping on density alone can
    declare a false 'converged' while the energy equation is still
    unconverged (seen on 3D wedge cases that choke and pressurize)."""
    crit = (cfg.get("solver_settings") or {}).get("convergence_criterion") \
        or DEFAULT_CONV_CRITERION
    fields = CONV_FIELD_PRESETS.get(str(crit).strip().lower())
    if fields is None:
        sys.exit(f"solver_settings.convergence_criterion = {crit!r} is not "
                 f"one of: {', '.join(CONV_FIELD_PRESETS)}")
    return list(fields)


def streamtube_exit_check(p01, p2, area_ratio, gamma):
    """Quasi-1D feasibility of the (Pt_in, p_out) pair on a streamtube
    whose exit area is `area_ratio` x its inlet area.

    The mass flow the exit demands at p_out with the isentropic exit Mach
    is A2*Pt_in*C*phi(M2_is); the inlet can swallow at most the choking
    flow A1*Pt_in*C*phi(1). If demanded > capacity, no steady solution
    exists for the prescribed pair (the passage diffuses/pressurizes
    instead). Returns a dict of the diagnostic numbers."""
    g = float(gamma)

    def phi(m):
        return m * (1.0 + 0.5 * (g - 1.0) * m * m) \
            ** (-(g + 1.0) / (2.0 * (g - 1.0)))

    ratio = p01 / p2 if p2 > 0 else float("inf")
    m2_is = math.sqrt(max(2.0 / (g - 1.0) * (ratio ** ((g - 1.0) / g) - 1.0),
                          0.0))
    phi_max = phi(1.0)
    phi_req = area_ratio * phi(m2_is)
    # minimum feasible p_out: solve phi(m2*) = phi_max / area_ratio
    target = phi_max / area_ratio
    lo, hi = 0.0, 1.0
    for _ in range(60):                  # phi rises monotonically on (0, 1)
        mid = 0.5 * (lo + hi)
        if phi(mid) < target:
            lo = mid
        else:
            hi = mid
    m2_star = 0.5 * (lo + hi)
    p2_min = p01 * (1.0 + 0.5 * (g - 1.0) * m2_star * m2_star) \
        ** (-g / (g - 1.0))
    return {"area_ratio": area_ratio, "m2_is": m2_is,
            "phi_req": phi_req, "phi_max": phi_max,
            "over_capacity": phi_req / phi_max,
            "p2_min": p2_min, "feasible": p2 >= p2_min}


def cascade_flow_states(p01, T01, p2, gamma, R=287.058):
    """Return the pressure-driven outlet state and a gentle startup state.

    The cascade BCs provide inlet total pressure/temperature and outlet
    static pressure. The isentropic outlet state is a useful physical
    reference for Reynolds number and post-processing, but using its Mach
    number as the uniform initial velocity can inject too much mass flow into
    a turning passage. The startup state therefore keeps the predicted
    outlet static pressure/temperature and uses a bounded low Mach number.

    ``init_pressure`` and ``init_temperature`` are static freestream values.
    That is what SU2 expects with ``FREESTREAM_OPTION= TEMPERATURE_FS`` and
    ``INIT_OPTION= TD_CONDITIONS``; ``MACH_NUMBER`` supplies the initial
    velocity independently.
    """
    p01, T01, p2, gamma, R = map(float, (p01, T01, p2, gamma, R))
    if p01 <= 0.0 or T01 <= 0.0 or p2 <= 0.0 or R <= 0.0:
        raise ValueError("cascade pressures, temperature, and gas constant "
                         "must be positive")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    if p2 >= p01:
        raise ValueError("outlet static pressure must be below inlet total pressure")

    pressure_ratio = p01 / p2
    m2_sq = 2.0 / (gamma - 1.0) * (
        pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0)
    mach_exit = math.sqrt(max(m2_sq, 0.0))
    t2 = T01 / (1.0 + 0.5 * (gamma - 1.0) * mach_exit ** 2)
    a2 = math.sqrt(gamma * R * t2)
    velocity_exit = mach_exit * a2
    rho2 = p2 / (R * t2)
    mu2 = sutherland_mu(t2)

    # Start at the outlet pressure equilibrium, but with deliberately modest
    # velocity. Scaling from the physical exit Mach keeps the startup flow
    # proportional for low-pressure-ratio cases; the cap prevents an
    # excessive first-step mass flux for high-pressure-ratio cases.
    mach_init = min(0.20, 0.25 * mach_exit)
    init_temperature = t2
    init_pressure = p2

    return {
        "mach_exit": mach_exit,
        "temperature_exit": t2,
        "velocity_exit": velocity_exit,
        "density_exit": rho2,
        "viscosity_exit": mu2,
        "mach_init": mach_init,
        "init_temperature": init_temperature,
        "init_pressure": init_pressure,
    }


def load_gamma(cfg):
    """Ratio of specific heats: explicit, or computed from the inlet total
    temperature with the temperature-dependent specific heats of air."""
    g = cfg.get("solver_settings", {}).get("gamma")
    if g is not None:
        g = float(g)
        if not 1.05 < g < 1.75:
            sys.exit(f"solver_settings.gamma = {g} looks wrong for a gas "
                     "(expected ~1.2-1.7, air: 1.4)")
        return g, f"set in input.json (gamma = {g})"
    T01 = load_bc(cfg)[1]
    g = gamma_of_air(T01)
    return g, f"computed from inlet total temperature T01 = {T01:g} K " \
              f"(gamma = {g:.4f})"


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
    ndim = 2
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("NDIME="):
            ndim = int(float(ln.split("=")[1].split()[0]))
            i += 1
            continue
        if ln.startswith("NPOIN="):
            n = int(ln.split("=")[1].split()[0])
            pts = np.zeros((n, ndim))
            for j in range(n):
                p = lines[i + 1 + j].split()
                for k in range(ndim):
                    pts[j, k] = float(p[k])
            i += n + 1
            continue
        if ln.startswith("MARKER_TAG= airfoil"):
            cnt = int(lines[i + 1].split("=")[1].split()[0])
            nodes = set()
            for j in range(cnt):
                p = [int(x) for x in lines[i + 2 + j].split()]
                nodes.update(p[1:3] if len(p) == 3 else p[1:5])
            if ndim == 3:
                # 3D (wedge) mesh: collapse the airfoil surface to its
                # mid-span node layer. The span direction is radial, so
                # nodes are grouped into spanwise stacks by (x, theta)
                # and the mid-ranked member of each stack is kept.
                idx = sorted(nodes)
                xs3 = np.round(pts[idx, 0], 9)
                th3 = np.round(np.arctan2(pts[idx, 1], pts[idx, 2]), 9)
                r3 = np.hypot(pts[idx, 1], pts[idx, 2])
                stacks = {}
                for k, n_id in enumerate(idx):
                    stacks.setdefault((xs3[k], th3[k]), []).append(k)
                n_layers = max(len(v) for v in stacks.values()) \
                    if stacks else 0
                mid = max(0, (n_layers - 1) // 2)
                keep = set()
                for members in stacks.values():
                    if len(members) != n_layers:
                        continue
                    mem = np.asarray(members, dtype=np.int64)
                    keep.add(idx[int(mem[np.argsort(r3[mem])][mid])])
                nodes = keep
            af = np.array(sorted(nodes), dtype=int)
            break
        i += 1
    return pts, af


def scale_mesh(path, scale):
    with open(path) as f:
        lines = f.read().splitlines()
    ndim = 2
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("NDIME="):
            ndim = int(float(ln.split("=")[1].split()[0]))
            out.append(ln)
            i += 1
            continue
        if ln.startswith("NPOIN="):
            n = int(ln.split("=")[1].split()[0])
            out.append(ln)
            for j in range(n):
                parts = lines[i + 1 + j].split()
                coords = [repr(float(v) * scale) for v in parts[:ndim]]
                out.append("\t" + "\t".join(coords + parts[ndim:]))
            i += n + 1
            continue
        out.append(ln)
        i += 1
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def restore_warm_start_restart(su2_dir, backup_dir):
    """Put the archived restart.dat back after the validation run.

    'Initialize from previous solution' warm-starts the following solve
    from restart.dat - but the validation step both moves that file into
    the results backup and overwrites/deletes the validate-run's own
    restart. Restoring the archived copy keeps the warm start working
    across full (mesh + setup + solve) runs. Returns True when a restart
    file was restored.
    """
    if backup_dir is None:
        return False
    candidate = Path(backup_dir) / "restart.dat"
    if candidate.exists():
        candidate.replace(Path(su2_dir) / "restart.dat")
        return True
    return False


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
    gamma, gamma_source = load_gamma(cfg)
    print(f"[gamma] {gamma_source}")
    numerics = load_numerics(cfg)
    mesh_name = cfg["case"]["name"]
    su2_name = args.name or case_root.name
    su2_dir = REPO / "cases" / su2_name
    su2_dir.mkdir(parents=True, exist_ok=True)

    # ---- domain / periodicity (read before mesh selection: the 3D wedge
    # mode uses a different mesh file)
    dom = cfg["domain"]
    mode = dom.get("periodicity", "axisymmetric")
    wedge3d = mode == "axisymmetric3d"
    periodic = mode in ("axisymmetric", "axisymmetric3d", "offset")

    # ---- mesh (regenerate on request or if missing)
    mesh_dir = case_root / "cases" / mesh_name
    if wedge3d:
        mesh_src = mesh_dir / "mesh_quad3d.su2"
    else:
        mesh_src = mesh_dir / "mesh_quad.su2"
        if not mesh_src.exists():
            mesh_src = mesh_dir / "mesh.su2"
    if args.remesh or not mesh_src.exists():
        proj = case_root.parent.parent
        py = Path(sys.executable)
        print(f"[mesh] running pipeline ({'--remesh' if args.remesh else 'mesh missing'}) ...")
        subprocess.run([str(py), str(proj / "pipeline" / "run_pipeline.py"),
                        str(case_root / "input.json")], check=True,
                       creationflags=CREATE_NO_WINDOW)
    if not mesh_src.exists():
        wanted = ("mesh_quad3d.su2 (axisymmetric3d mode)" if wedge3d
                  else "mesh_quad.su2")
        sys.exit(f"mesh not found at {mesh_src} (run with --remesh; "
                 f"expected {wanted})")

    # ---- geometry scale
    ac = float(cfg["airfoil"]["axial_chord"])
    scale = args.scale if args.scale is not None else (ac / 1000.0 if ac >= 1.0 else ac)
    print(f"[scale] axial_chord = {ac} -> {scale} m per chord unit")

    if periodic and mode == "axisymmetric" and \
            abs(dom["R1"] - dom["R2"]) > 1e-9 * max(dom["R1"], dom["R2"], 1.0):
        sys.exit("R1 != R2 (varying pitch) cannot be paired by SU2 - "
                 "regenerate the mesh with R1 = R2 (see the_process.md)")
    # R1/R2 are actual radii (same units as axial_chord): convert to meters
    units_to_m = 0.001 if ac >= 1.0 else 1.0
    pitch_le_m = 2.0 * math.pi * dom["R1"] * units_to_m / dom["airfoil_count"] \
        if periodic else 0.0
    pitch_te_m = 2.0 * math.pi * dom["R2"] * units_to_m / dom["airfoil_count"] \
        if periodic else 0.0
    pitch_m = pitch_le_m

    # ---- BCs from input.json
    p01, T01, ang, p2 = load_bc(cfg)
    rad = math.radians(ang)
    direction = [math.cos(rad), math.sin(rad), 0.0]
    print(f"[bcs] inlet: p0={p01} Pa, T0={T01} K, angle={ang} deg | outlet: p={p2} Pa")

    # ---- quasi-1D feasibility of the BC pair on the streamtube: an
    # expanding wedge (R2*h2 > R1*h1) with p_out too low has NO steady
    # solution - the domain pressurizes to Pt_in and the through-flow
    # collapses instead
    if wedge3d:
        h1 = float(dom.get("h1") or 0.0)
        h2 = float(dom.get("h2") or h1)
        if h1 > 0.0 and h2 > 0.0:
            area_ratio = (float(dom["R2"]) * h2) / (float(dom["R1"]) * h1)
            chk = streamtube_exit_check(p01, p2, area_ratio, gamma)
            print(f"[bcs] streamtube A2/A1 = {area_ratio:.3f} | isentropic "
                  f"exit M = {chk['m2_is']:.3f} | minimum feasible p_out "
                  f"~ {chk['p2_min']:,.0f} Pa")
            if not chk["feasible"]:
                print(f"[bcs] WARNING: prescribed p_out = {p2:,.0f} Pa is "
                      f"{p2 / chk['p2_min']:.2f}x below the feasible floor "
                      f"for this expanding streamtube (inlet choking "
                      f"over capacity {chk['over_capacity']:.2f}x). No "
                      "steady solution exists: the passage will "
                      "pressurize toward Pt_in, the through-flow will "
                      "collapse and the exit flow will reverse. Raise "
                      "p_out or contract the streamtube (R2*h2).")
            elif chk["over_capacity"] > 0.9:
                print("[bcs] WARNING: p_out is close to the choking limit "
                      "of this streamtube")

    # ---- copy + scale mesh
    (su2_dir / "mesh.su2").write_bytes(mesh_src.read_bytes())
    scale_mesh(su2_dir / "mesh.su2", scale)

    # ---- 0D geometry metrics from the pipeline (throat width, unguided
    # turning, pitch) - merged into results.json by plot_case.py
    metrics_src = mesh_src.parent / "geometry_metrics.json"
    if metrics_src.exists():
        (su2_dir / "geometry_metrics.json").write_bytes(
            metrics_src.read_bytes())
        print(f"[metrics] geometry_metrics.json -> {su2_dir}")

    # ---- postprocess hints measured on this airfoil
    pts, af = parse_airfoil_mesh(su2_dir / "mesh.su2")
    p0, nrm, zoom = measure_surface_probe(pts, af)
    probe_point = [round(float(p0[0]), 9), round(float(p0[1]), 9)]
    probe_normal = [round(float(nrm[0]), 4), round(float(nrm[1]), 4)]
    if len(p0) > 2:                  # 3D wedge mesh: keep the mid-span z
        probe_point.append(round(float(p0[2]), 9))
        probe_normal.append(0.0)

    # ---- reference / init state
    R = 287.058
    flow = cascade_flow_states(p01, T01, p2, gamma, R)
    T2 = flow["temperature_exit"]
    V2 = flow["velocity_exit"]
    Re = flow["density_exit"] * V2 * scale / flow["viscosity_exit"]
    print(f"[refs] isentropic exit: M={flow['mach_exit']:.3f} "
          f"V={V2:.1f} m/s | Re_axial = {Re:.3g}, pitch = {pitch_m:.6f} m")
    print(f"[init] pressure-balanced outlet state: M={flow['mach_init']:.3f} "
          f"p={flow['init_pressure']:.1f} Pa, "
          f"T={flow['init_temperature']:.1f} K")

    mode_desc = {
        "axisymmetric": "Turbine vane cascade auto-setup",
        "axisymmetric3d": ("Conical wedge cascade (3D streamtube, "
                           "slip hub/shroud) auto-setup"),
        "offset": "Linear-cascade (offset periodics) auto-setup",
        "freestream": "Isolated-airfoil (freestream boundaries) auto-setup",
    }.get(mode, "Turbine vane cascade auto-setup")
    case = {
        "name": su2_name,
        "description": (f"{mode_desc} from {case_root}. "
                        f"BCs and solver settings read from input.json; "
                        f"mesh scaled {scale} m/chord."),
        "boundary_mode": "cascade" if periodic else "freestream",
        "mesh": "mesh.su2",
        "physics": {
            "solver": "RANS",
            "turbulence_model": turbulence_model,
            # Physical reference state: the isentropic exit estimate from
            # the supplied inlet/outlet thermodynamic conditions.
            "mach": flow["mach_exit"],
            "reynolds": Re,
            "reynolds_length": scale,
            # Solver startup state: separate from the physical reference Mach
            # so the initial uniform field does not over-feed the passage.
            "freestream_temperature": T2,
            "init_mach": flow["mach_init"],
            "init_pressure": flow["init_pressure"],
            "init_temperature": flow["init_temperature"],
            "gamma": gamma,
            "gas_constant": R,
        },
        "cascade": {
            "inlet": {"marker": "inlet", "total_pressure": p01,
                      "total_temperature": T01, "direction": direction},
            "outlet": {"marker": "outlet", "static_pressure": p2},
        },
        "markers": {
            "airfoil": {"bc": "wall_adiabatic"},
            "inlet": {"bc": "inlet", "analyze": True},
            # outlet analysis -> per-surface Avg_*(outlet) history columns
            # (GUI live mass-flow / imbalance plots)
            "outlet": {"bc": "outlet", "analyze": True},
        },
        "numerics": numerics,
        "convergence": {"fields": load_conv_fields(cfg), "minval": -6.0,
                        "startiter": 100, "iterations": max_iterations},
        "postprocess": {
            "wall_hint": probe_point[:2],
            "probe": {"point": probe_point,
                      "normal": probe_normal,
                      "length": round(0.5 * scale, 6)},
            "zoom": {k: round(v, 6) for k, v in zoom.items()},
            # suction side is the upper surface for clockwise turning (CAP),
            # the lower one for counter-clockwise turning (CUP)
            "ss_upper": float(cfg["airfoil"]["alpha2"])
                        < float(cfg["airfoil"]["alpha1"]),
        },
    }
    if wedge3d:
        # Slip hub/shroud streamtube walls + planform reference area so
        # the 3D CD/CL stay comparable with the 2D per-unit-depth values
        case["markers"]["hub"] = {"bc": "slip"}
        case["markers"]["shroud"] = {"bc": "slip"}
        h1 = float(dom.get("h1") or 0.0)
        h2 = float(dom.get("h2") or h1)
        case["physics"]["ref_area"] = scale * (0.5 * (h1 + h2) * units_to_m)
        case["cascade"]["pitch_le_m"] = pitch_le_m
        case["cascade"]["pitch_te_m"] = pitch_te_m
    if periodic:
        if wedge3d:
            # rotational periodic pair: the constant wedge angle 2 pi / N
            # about the machine axis (x) maps one face onto the other, so a
            # varying radius R1 -> R2 is legal. The extrusion puts
            # periodic_top at theta = theta_bottom + 2 pi / N, and SU2
            # rotates the DONOR (second marker) onto the first, so the
            # working angle is the negative wedge angle (validation still
            # settles the sign empirically - see below).
            case["cascade"]["periodic"] = [
                {"markers": ["periodic_bottom", "periodic_top"],
                 "center": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0],
                 "rotation_deg": -360.0 / float(dom["airfoil_count"]),
                 "translation": [0.0, 0.0, 0.0]}]
        else:
            # translational periodic pair (R1 = R2 -> constant pitch in meters)
            case["cascade"]["periodic"] = [
                {"markers": ["periodic_bottom", "periodic_top"],
                 "translation": [0.0, pitch_m, 0.0]}]
    else:
        # Freestream mode: all four outer edges are farfield.  Keep the
        # separate mesh tags so existing meshes remain usable, but do not
        # mark the former inlet/outlet edges for cascade analysis.
        case["markers"]["inlet"] = {"bc": "farfield"}
        case["markers"]["outlet"] = {"bc": "farfield"}
        case["markers"]["farfield"] = {"bc": "farfield"}
        print("[bcs] freestream outer boundary: inlet, outlet, farfield -> MARKER_FAR")
    (su2_dir / "case.json").write_text(json.dumps(case, indent=2))
    print(f"[case] wrote {su2_dir / 'case.json'}")

    # ---- render config
    def render_cfg():
        subprocess.run([sys.executable, str(REPO / "tools" / "render_config.py"),
                        str(su2_dir / "case.json"),
                        str(REPO / "templates" / "su2_cascade.cfg"),
                        str(su2_dir / "turbine.cfg")], check=True,
                       creationflags=CREATE_NO_WINDOW)

    render_cfg()

    # ---- validate periodic pairing + BCs (2 iterations)
    bdir = None
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
        # the rotational pair's sign matters for the halo construction: a
        # wrong sign still reports "Matched" (SU2 checks both directions)
        # but builds halos from wrong neighbours and diverges - so accept a
        # candidate ONLY when the pairing log is clean of bad-match lines
        for sign in ([-1.0, 1.0] if wedge3d else [1.0]):
            if wedge3d:
                case["cascade"]["periodic"][0]["rotation_deg"] = \
                    sign * 360.0 / float(dom["airfoil_count"])
                (su2_dir / "case.json").write_text(
                    json.dumps(case, indent=2))
                render_cfg()
            text = (su2_dir / "turbine.cfg").read_text()
            text = re.sub(r"(?m)^ITER=[ \t]+\d+[ \t]*$", "ITER= 2", text)
            text = re.sub(r"(?m)^CONV_STARTITER=[ \t]+\d+[ \t]*$",
                          "CONV_STARTITER= 2", text)
            (su2_dir / "validate.cfg").write_text(text)
            r = subprocess.run([str(su2_exe),
                                "-t", "6", "validate.cfg"],
                               cwd=str(su2_dir), capture_output=True,
                               text=True, timeout=300,
                               creationflags=CREATE_NO_WINDOW)
            log = r.stdout + r.stderr
            # freestream cases have no periodic pair, so there is no
            # "Matched" pairing line to look for
            ok = r.returncode == 0 and (not periodic or "Matched" in log)
            if wedge3d:
                ok = ok and "Bad match" not in log
            if ok:
                if wedge3d and sign < 0:
                    print("[validate] rotational periodic: negative wedge "
                          "angle pairs cleanly")
                break
        else:
            sys.exit(f"validation FAILED:\n{log[-1500:]}")
        for ln in log.splitlines():
            if "Matched" in ln:
                print(f"[validate] {ln.strip()}")
        (su2_dir / "validate.cfg").unlink()
        for f in ("history.csv", "restart.dat", "vol_solution.vtk", "vol_solution.vtu",
                  "surface.vtk", "surface.vtu"):
            (su2_dir / f).unlink(missing_ok=True)
        # the validate run must not destroy the warm-start state: put the
        # archived restart.dat back for the 'Initialize from previous
        # solution' solve that follows this setup
        if restore_warm_start_restart(su2_dir, bdir):
            print("[validate] restart.dat restored (warm-start state "
                  "preserved)")

    run_cmd = (f'python tools\\run_monitor.py cases\\{su2_name} '
               f'turbine.cfg -t 6')
    print(f"[done] SU2 case ready: {su2_dir}")
    print(f"[next] {run_cmd}")
    if args.run:
        monitor_cmd = [sys.executable, str(REPO / "tools" / "run_monitor.py"),
                       str(su2_dir), "turbine.cfg", "-t", "6"]
        if args.config:
            monitor_cmd += ["--config", str(args.config)]
        subprocess.Popen(monitor_cmd, creationflags=CREATE_NO_WINDOW)


if __name__ == "__main__":
    main()
