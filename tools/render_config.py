"""Render a SU2 .cfg from a template + case settings (JSON).

Usage:
    python render_config.py <case.json> <template.cfg> <output.cfg>

Template tokens (@TOKEN@, %%MARKER_LINES%%) are replaced from case.json:
    @SOLVER@ @TURB_MODEL_LINE@ @MACH@ @INIT_MACH@ @REYNOLDS@ @REYNOLDS_LENGTH@ @GAMMA@
    @GAS_CONSTANT@ @T_INF@ @P_INF@ @T0@ @P0@ @ITER@ @CONV_FIELDS@
    @CONV_MINVAL@ @CONV_STARTITER@ @CFL@ @CFL_ADAPT@ @LIMITER@ @GRADIENT@
    @LINEAR_SOLVER_ITER@ @OUTPUT_FILES@ @MESH_FILE@
%%MARKER_LINES%% expands to MARKER_* lines grouped by boundary type, plus
MARKER_MONITORING covering all wall-type markers.  A case with
``boundary_mode == "freestream"`` maps every outer marker to MARKER_FAR.

Supported marker bc types: inlet, outlet, farfield, wall_adiabatic,
wall_isothermal (needs "temperature"), symmetry.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freestream import fmt, state  # noqa: E402

WALL_TYPES = {"wall_adiabatic", "wall_isothermal"}


def is_freestream_case(case):
    """Return whether the case uses an external-flow outer boundary.

    ``boundary_mode`` is authoritative for newly generated cases.  The
    marker-based fallback keeps older freestream case.json files usable: the
    earlier generator wrote a farfield marker but had no explicit mode field.
    """
    mode = case.get("boundary_mode")
    if mode == "freestream":
        return True
    if mode:
        return False
    markers = case.get("markers", {})
    cascade = case.get("cascade", {})
    return ("cascade" in case and not cascade.get("periodic") and
            any(spec.get("bc") == "farfield" for spec in markers.values()))


def marker_lines(markers, fs):
    groups = {}
    for name, spec in markers.items():
        groups.setdefault(spec["bc"], []).append(name)

    lines = []
    for name in groups.get("inlet", []):
        d = markers[name].get("direction", [1.0, 0.0, 0.0])
        lines.append(f"MARKER_INLET= ( {name}, {fmt(fs['T0'])}, {fmt(fs['p0'])}, "
                     f"{fmt(d[0])}, {fmt(d[1])}, {fmt(d[2])} )")
    for name in groups.get("outlet", []):
        lines.append(f"MARKER_OUTLET= ( {name}, {fmt(fs['p_inf'])} )")
    if groups.get("farfield"):
        lines.append("MARKER_FAR= ( " + ", ".join(groups["farfield"]) + " )")
    for name in groups.get("wall_adiabatic", []):
        lines.append(f"MARKER_HEATFLUX= ( {name}, 0.0 )")
    for name in groups.get("wall_isothermal", []):
        lines.append(f"MARKER_ISOTHERMAL= ( {name}, {fmt(markers[name]['temperature'])} )")
    if groups.get("symmetry"):
        lines.append("MARKER_SYM= ( " + ", ".join(groups["symmetry"]) + " )")

    walls = [n for n, s in markers.items() if s["bc"] in WALL_TYPES]
    if walls:
        lines.append("MARKER_MONITORING= ( " + ", ".join(walls) + " )")
    return "\n".join(lines)


def analyzed_markers(markers):
    return [n for n, s in markers.items() if s.get("analyze")]


def cascade_marker_lines(case):
    """Turbomachinery BCs from case["cascade"]: MARKER_INLET (total T then
    total P per SU2 v8), MARKER_OUTLET (static p), MARKER_PERIODIC
    (12-token v8 format: marker, donor, rot center, rot angles, translation),
    plus MARKER_MONITORING / MARKER_ANALYZE from the shared markers dict.

    Freestream cases deliberately do not emit inlet, outlet, or periodic
    entries.  Their complete outer boundary (including mesh tags named
    ``inlet`` and ``outlet``) is one SU2 farfield boundary.
    """
    cas = case["cascade"]
    freestream = is_freestream_case(case)
    lines = []

    def nums(vals):
        return ", ".join(fmt(v) for v in vals)

    inl = cas["inlet"]
    out = cas["outlet"]
    if not freestream:
        d = inl["direction"]
        lines.append(f"MARKER_INLET= ( {inl['marker']}, {fmt(inl['total_temperature'])}, "
                     f"{fmt(inl['total_pressure'])}, {nums(d)} )")
        lines.append(f"MARKER_OUTLET= ( {out['marker']}, {fmt(out['static_pressure'])} )")
        for pair in cas.get("periodic", []):
            a, b = pair["markers"]
            zero3 = "0.0, 0.0, 0.0"
            # full precision: the periodic pairing tolerance is far tighter than fmt()
            trans = ", ".join(repr(float(v)) for v in pair["translation"])
            lines.append(f"MARKER_PERIODIC= ( {a}, {b}, {zero3}, {zero3}, {trans} )")

    markers = case.get("markers", {})
    for name, spec in markers.items():
        if spec["bc"] in WALL_TYPES:
            lines.append(f"MARKER_HEATFLUX= ( {name}, 0.0 )")
    # In external flow, every edge around the fluid domain is farfield.  The
    # mesh retains separate names for the former inlet/outlet edges so that
    # old meshes remain usable, but SU2 must receive one complete MARKER_FAR
    # definition rather than a pressure-driven inlet/outlet pair.
    if freestream:
        farfields = []
        for name in (inl["marker"], out["marker"]):
            if name not in farfields:
                farfields.append(name)
        for name, spec in markers.items():
            if spec["bc"] == "farfield" and name not in farfields:
                farfields.append(name)
    else:
        farfields = [n for n, s in markers.items() if s["bc"] == "farfield"]
    if farfields:
        lines.append("MARKER_FAR= ( " + ", ".join(farfields) + " )")
    walls = [n for n, s in markers.items() if s["bc"] in WALL_TYPES]
    if walls:
        lines.append("MARKER_MONITORING= ( " + ", ".join(walls) + " )")
    if analyzed_markers(markers):
        lines.append("MARKER_ANALYZE= ( " + ", ".join(analyzed_markers(markers)) + " )")
    return "\n".join(lines)


def time_domain_block(unsteady, conv):
    """Steady block by default; dual-time URANS block when case.json has "unsteady"."""
    if not unsteady:
        return "TIME_DOMAIN= NO"
    lines = [
        "TIME_DOMAIN= YES",
        "TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER",
        f"TIME_STEP= {fmt(unsteady['time_step'])}",
        f"TIME_ITER= {unsteady['time_steps']}",
        f"INNER_ITER= {unsteady.get('inner_iter', 40)}",
        "WINDOW_CAUCHY_CRIT= YES",
        "CONV_WINDOW_FIELD= ( TAVG_DRAG )",
        f"CONV_WINDOW_CAUCHY_ELEMS= {unsteady.get('window_cauchy_elems', 100)}",
        "CONV_WINDOW_CAUCHY_EPS= 1e-03",
        f"WINDOW_START_ITER= {unsteady.get('window_start_iter', 300)}",
    ]
    return "\n".join(lines)


def main():
    case_file, template_file, out_file = (Path(a) for a in sys.argv[1:4])
    case = json.loads(case_file.read_text())
    physics = case["physics"]
    numerics = case.get("numerics", {})
    conv = case.get("convergence", {})
    fs = state(physics["mach"], physics["reynolds"],
               physics.get("reynolds_length", 1.0),
               physics.get("freestream_temperature", 288.15),
               physics.get("gamma", 1.4), physics.get("gas_constant", 287.058))

    tokens = {
        "SOLVER": physics["solver"],
        "TURB_MODEL_LINE": (f"KIND_TURB_MODEL= {physics['turbulence_model']}"
                            if physics.get("turbulence_model") else ""),
        "TIME_DOMAIN_BLOCK": time_domain_block(case.get("unsteady"), conv),
        "MACH": fmt(physics["mach"]),
        "INIT_MACH": fmt(physics.get("init_mach", physics["mach"])),
        # For a farfield boundary SU2 uses MACH_NUMBER as part of the
        # physical free-stream definition, so the reduced cascade startup
        # Mach cannot be used in external-flow mode.
        "FLOW_MACH": fmt(physics["mach"] if is_freestream_case(case)
                          else physics.get("init_mach", physics["mach"])),
        "REYNOLDS": fmt(physics["reynolds"]),
        "REYNOLDS_LENGTH": fmt(physics.get("reynolds_length", 1.0)),
        "GAMMA": fmt(physics.get("gamma", 1.4)),
        "GAS_CONSTANT": fmt(physics.get("gas_constant", 287.058)),
        "T_INF": fmt(fs["T_inf"]),
        "P_INF": fmt(fs["p_inf"]),
        "T0": fmt(fs["T0"]),
        "P0": fmt(fs["p0"]),
        "INIT_TEMPERATURE": fmt(physics.get("init_temperature", fs["T_inf"])),
        "INIT_PRESSURE": fmt(physics.get("init_pressure", fs["p_inf"])),
        "CASCADE_MARKER_LINES": cascade_marker_lines(case) if "cascade" in case else "",
        "ITER_LINE": ("ITER= " + str(conv.get("iterations", 2000))
                      if "unsteady" not in case else
                      "% ITER is not valid for unsteady problems (TIME_ITER governs)"),
        "CONV_FIELDS": ", ".join(conv.get("fields", ["RMS_DENSITY", "RMS_ENERGY"])),
        "CONV_MINVAL": fmt(conv.get("minval", -6)),
        "CONV_STARTITER": str(conv.get("startiter", 50)),
        "CFL": fmt(numerics.get("cfl", 10.0)),
        "CFL_ADAPT_BLOCK": ("CFL_ADAPT= NO" if not numerics.get("cfl_adapt", True) else
                            "CFL_ADAPT= YES\nCFL_ADAPT_PARAM= ( "
                            + ", ".join(fmt(x) for x in numerics.get("cfl_adapt", [0.1, 1.5, 100.0, 1e10]))
                            + " )"),
        "OUTPUT_WRT_FREQ_LINE": ("" if "unsteady" not in case else
                                 "HISTORY_WRT_FREQ_INNER= " + str(case["unsteady"].get("history_wrt_freq_inner", 5))
                                 + "\nOUTPUT_WRT_FREQ= ( "
                                 + ", ".join(str(x) for x in case["unsteady"].get("output_wrt_freq", []))
                                 + " )"),
        "LIMITER": numerics.get("limiter", "NONE"),
        "GRADIENT": numerics.get("gradient", "WEIGHTED_LEAST_SQUARES"),
        "LINEAR_SOLVER_ITER": str(numerics.get("linear_solver_iter", 100)),
        "OUTPUT_FILES": ", ".join(case.get("output", {}).get(
            "files", ["RESTART", "PARAVIEW", "SURFACE_PARAVIEW",
                      "PARAVIEW_LEGACY", "SURFACE_PARAVIEW_LEGACY"])),
        "MESH_FILE": case.get("mesh", "mesh.su2"),
        "MARKER_LINES": marker_lines(case["markers"], fs),
        "MARKER_ANALYZE_LINE": ("MARKER_ANALYZE= ( " + ", ".join(analyzed_markers(case["markers"])) + " )"
                                if analyzed_markers(case["markers"]) else ""),
        "EXTRA_HISTORY": (", FLOW_COEFF" if analyzed_markers(case["markers"]) else ""),
    }

    text = template_file.read_text()
    for key, val in tokens.items():
        text = text.replace(f"@{key}@", val)

    import re
    leftover = sorted(set(re.findall(r"@([A-Z_]+)@", text)))
    if leftover:
        sys.exit(f"unresolved template tokens: {leftover}")

    out_file.write_text(text)
    print(f"wrote {out_file}")
    print("--- freestream state ---")
    for k in ("mach", "reynolds", "T_inf", "p_inf", "rho_inf", "mu_inf",
              "U_inf", "q_inf", "T0", "p0"):
        print(f"  {k} = {fs[k]:.6g}" if isinstance(fs[k], float) else f"  {k} = {fs[k]}")


if __name__ == "__main__":
    main()
