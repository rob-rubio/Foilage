"""Logic selftest for the GUI package (no window required).

    python -m foilage_gui --selftest
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from foilage_gui.schema import DEFAULTS, FIELDS, FIELDS_BY_PATH  # noqa: E402
from foilage_gui.state import CaseState, get_path, set_path      # noqa: E402
from foilage_gui.su2_mesh import read_su2_mesh, mesh_edges       # noqa: E402


def test_schema():
    assert FIELDS, "schema must define fields"
    paths = [f.path for f in FIELDS]
    assert len(paths) == len(set(paths)), "duplicate schema paths"
    # every default path must resolve inside DEFAULTS
    for f in FIELDS:
        v = get_path(DEFAULTS, f.path)
        assert v is not None or f.kind.startswith("nullable"), f.path
    for f in FIELDS:
        if f.choices:
            stored = [v for _lbl, v in f.choices]
            assert f.default in stored, f"default of {f.path} not in choices"


def test_state_roundtrip():
    inputs = sorted((REPO / "cases").glob("*/input.json"))
    assert inputs, "no case inputs found for the roundtrip test"
    with tempfile.TemporaryDirectory() as tmp:
        for src in inputs:
            state = CaseState(src)
            # unknown keys survive, defaults get filled in
            raw = json.loads(src.read_text())
            for k, v in raw.items():
                assert k in state.config, f"{src.name}: lost key {k}"
            state.save(Path(tmp) / src.name)
            re = CaseState(Path(tmp) / src.name)
            for f in FIELDS:
                a, b = state.get(f.path), re.get(f.path)
                assert a == b, f"{f.path}: {a!r} != {b!r} ({src.name})"
    print(f"state roundtrip OK ({len(inputs)} case inputs)")


def test_validation():
    state = CaseState(REPO / "cases" / "turbine_blade_9" / "input.json")
    assert state.validate() == [] or all(
        s != "error" for s, _m in state.validate())
    set_path(state.config, "BCs.outlet.static pressure",
             get_path(state.config, "BCs.inlet.total pressure") + 1)
    issues = state.validate()
    assert any(s == "error" and "below" in m for s, m in issues), \
        "p2 >= p01 must raise an error"
    set_path(state.config, "domain.R2",
             get_path(state.config, "domain.R1") + 1)
    issues = state.validate()
    assert any("R1" in m for _s, m in issues), "R1 != R2 must warn"
    print("validation OK")


def test_derived():
    state = CaseState(REPO / "cases" / "turbine_blade_9" / "input.json")
    d = state.derived()
    assert 0 < d["M2"] < 1.2, f"exit Mach out of range: {d['M2']}"
    assert abs(d["pr"] - d.get("pr", d["pr"])) < 1e-12
    assert d["V2"] > 100 and d["T2"] > 300
    assert state.scale_m_per_chord() == 0.1
    print(f"derived OK (M2={d['M2']:.3f}, Re={d['Re_axial']:.2g})")


def test_gamma_model():
    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from freestream import gamma_of_air
    assert abs(gamma_of_air(300.0) - 1.3994) < 0.002
    assert abs(gamma_of_air(700.0) - 1.3638) < 0.003
    assert abs(gamma_of_air(1000.0) - 1.3351) < 0.004
    state = CaseState(REPO / "cases" / "turbine_blade_9" / "input.json")
    g, src = state.effective_gamma()
    T01 = float(state.get("BCs.inlet.total temperature"))
    assert abs(g - gamma_of_air(T01)) < 1e-12 and "computed" in src
    set_path(state.config, "solver_settings.gamma", 1.35)
    g2, src2 = state.effective_gamma()
    assert g2 == 1.35 and "set" in src2
    d = state.derived()
    assert d["M2"] > 0
    print(f"gamma OK (auto {g:.4f} @ {T01:g} K; explicit 1.35 overrides)")


def test_cascade_initialization():
    """Startup state is pressure-balanced and lower-flow than the reference."""
    from tools.setup_cascade_case import cascade_flow_states

    s = cascade_flow_states(125000.0, 700.0, 81266.7, 1.36383165686)
    assert s["mach_init"] < s["mach_exit"]
    assert s["mach_init"] <= 0.2
    assert abs(s["init_pressure"] - 81266.7) < 1e-8
    assert abs(s["init_temperature"] - s["temperature_exit"]) < 1e-8
    print(f"cascade init OK (Mexit={s['mach_exit']:.3f}, "
          f"Minit={s['mach_init']:.3f})")


def test_geomturbo_import():
    import copy
    import tempfile
    if str(REPO / "pipeline") not in sys.path:
        sys.path.insert(0, str(REPO / "pipeline"))
    from pipeline.airfoil import build_airfoil, build_geometry
    from pipeline.cascade_metrics import (channel_widths, curvature,
                                           throat_metrics, true_chord)
    from pipeline.geomturbo import (infer_cascade_parameters,
                                    parse_geomturbo, section_to_airfoil,
                                    write_geomturbo)

    cfg = json.loads((REPO / "cases" / "turbine_blade_9" / "input.json")
                     .read_text())
    ref = build_airfoil(cfg["airfoil"])

    # Exercise the native NUMECA layout: separate SECTIONAL suction and
    # pressure blocks, two spanwise sections, file-order Z Y X rows, a
    # blade count, and CHANNEL hub/shroud zrcurves (constant radii 0.45 /
    # 0.55 m so the inferred mid-span radius is 0.50 m at any station).
    ss, ps = ref["ss"], ref["ps"]
    with tempfile.TemporaryDirectory() as tmp:
        gt = Path(tmp) / "test.geomTurbo"
        scale = 0.105
        lines = [
            "TYPE GEOMTURBO", "GEOMETRY TURBO VERSION 5", "units 1",
            "NI_BEGIN CHANNEL",
            " NI_BEGIN basic_curve", " NAME hub",
            " NI_BEGIN zrcurve", "ZR", "2",
            f"{-0.05 * scale:.9f} 0.450000000",
            f"{0.20 * scale:.9f} 0.450000000",
            " NI_END zrcurve", " NI_END basic_curve",
            " NI_BEGIN basic_curve", " NAME shroud",
            " NI_BEGIN zrcurve", "ZR", "2",
            f"{-0.05 * scale:.9f} 0.550000000",
            f"{0.20 * scale:.9f} 0.550000000",
            " NI_END zrcurve", " NI_END basic_curve",
            "NI_END CHANNEL",
            "NI_BEGIN nibladegeometry", "number_of_blades 20",
            "suction", "SECTIONAL", "2",
        ]
        for section_z in (0.0, 0.025):
            lines += [f"# SECTION {1 if section_z == 0.0 else 2}",
                      "XYZ", str(len(ss))]
            # NUMECA row convention: Z -Y X (the stored Y is the negative
            # of the canonical Cartesian y)
            lines += [f"{section_z:.9f} {-y * scale:.9f} {x * scale:.9f}"
                      for x, y in ss]
        lines += ["pressure", "SECTIONAL", "2"]
        for section_z in (0.0, 0.025):
            lines += [f"# SECTION {1 if section_z == 0.0 else 2}",
                      "XYZ", str(len(ps))]
            lines += [f"{section_z:.9f} {-y * scale:.9f} {x * scale:.9f}"
                      for x, y in ps]
        lines += ["NI_END nibladegeometry"]
        gt.write_text("\n".join(lines))

        parsed = parse_geomturbo(gt)
        assert parsed["blade_count"] == 20
        assert len(parsed["sections"]) == 2
        assert parsed["sections"][0]["ss"].shape == (len(ss), 3)
        assert parsed["sections"][0]["ps"].shape == (len(ps), 3)
        assert parsed["hub"] is not None and parsed["shroud"] is not None
        # the stored -Y is flipped on read: the canonical y matches the
        # generator's y even though the file rows carry the negated value
        assert np.allclose(parsed["sections"][0]["ss"][:, 1],
                           ss[:, 1] * scale), "read-side -Y flip"

        inf = infer_cascade_parameters(parsed, 0)
        assert inf["blade_count"] == 20
        assert abs(inf["axial_chord"] - 0.105 * 1.01) < 5e-3, \
            f"axial chord {inf['axial_chord']}"
        # R1/R2 = LE/TE point radii about the machine axis: sqrt(y^2+z^2).
        # Section 0 has z = 0, so the radius is |y| at each edge point.
        assert abs(inf["R1"] - abs(ref["ss"][0, 1]) * scale) < 1e-6, \
            f"R1 {inf['R1']}"
        assert abs(inf["R2"] - abs(ref["ss"][-1, 1]) * scale) < 1e-6, \
            f"R2 {inf['R2']}"

        cfg["airfoil_source"] = {"type": "geomturbo", "geomturbo_file": str(gt),
                                 "section": 0}
        af = build_geometry(cfg)
    assert af["axial_chord"] == float(cfg["airfoil"]["axial_chord"])
    n = int(cfg["airfoil"]["n_points"])
    assert len(af["ss"]) in (n, n + 1), f"ss points {len(af['ss'])}"
    xmin = min(af["ss"][:, 0].min(), af["ps"][:, 0].min())
    xmax = max(af["ss"][:, 0].max(), af["ps"][:, 0].max())
    assert abs(xmin) < 1e-9 and 0.98 <= xmax <= 1.1, \
        f"normalization wrong (x range {xmin}, {xmax})"
    # roundtrip fidelity: suction side stays the suction side (same mean-y
    # ordering relative to ps as the reference)
    assert (af["ss"][:, 1].mean() > af["ps"][:, 1].mean()) == \
           (ref["ss"][:, 1].mean() > ref["ps"][:, 1].mean())
    assert af["style"] == "geomTurbo import"

    # real-file check: the igv643 sample's section-0 LE point is
    # (z, y, x) = (5.3983301E-02, -1.8840518E-02, -5.8218479E-02), so the
    # LE radius is sqrt(z^2 + y^2) ~= 0.0572 m (user-reported value)
    igv = REPO / "sample" / "geomturbo_external" / "igv643.geomTurbo"
    if igv.exists():
        inf2 = infer_cascade_parameters(parse_geomturbo(igv), 0)
        assert abs(inf2["R1"] - 0.05717) < 2e-3, \
            f"igv643 R1 {inf2['R1']:.5f} != ~0.0572"
        print(f"igv643 check: R1 = {inf2['R1']:.4f} m, "
              f"R2 = {inf2['R2']:.4f} m, ac = {inf2['axial_chord']:.4f} m")

    # channel/throat/curvature sanity on the reference blade at a
    # realistic pitch (blade_9's own pitch is very coarse, s/c = 1.26,
    # which degenerates the throat toward the TE tail). R1/R2 are actual
    # units (mm) and are normalized by the axial chord internally.
    cfg["domain"]["R1"] = cfg["domain"]["R2"] = 700.0    # mm -> 7.0 c_ax
    cfg["domain"]["airfoil_count"] = 90          # s/c ~ 0.49
    ref["axial_chord"] = float(cfg["airfoil"]["axial_chord"])
    from pipeline.mesh_tris import pitch_profile
    prof = pitch_profile(cfg["domain"], ref)
    s_ch, _j = channel_widths(ref["ss"], ref["ps"], prof["p"],
                              ref["ss_upper"])
    # CUP mirror check: flipping the blade vertically must give the same
    # channel widths (the neighbor direction flips with the surfaces)
    ss_m = ref["ss"].copy()
    ps_m = ref["ps"].copy()
    ss_m[:, 1] *= -1.0
    ps_m[:, 1] *= -1.0
    s_ch_m, _j = channel_widths(ss_m, ps_m, prof["p"],
                                not ref["ss_upper"])
    assert np.abs(s_ch - s_ch_m).max() < 1e-9, "CUP/CAP channel mismatch"
    assert 0.05 < s_ch.min() < prof["p_le"], "throat width out of range"
    t = throat_metrics(ref["ss"], ref["ps"], prof["p"], ref["ss_upper"])
    assert 0.1 < t["x_over_cax"] < 0.95
    assert 0.0 < t["unguided_turning_deg"] < 45.0
    assert t["width"] > 0
    chord_cax = true_chord(ref["ss"], ref["ps"])
    assert chord_cax > 0.0
    assert abs(t["pitch_throat"] - prof["p"](t["x_over_cax"])) < 1e-12
    ptc = t["pitch_throat"] / chord_cax
    assert ptc > 0.0
    k_ss, _ = curvature(ref["ss"])
    assert np.isfinite(k_ss).all() and k_ss.max() > k_ss.min()

    # geomTurbo export -> import roundtrip (export is in meters; the
    # import re-normalizes to axial chord = 1)
    from pipeline.geomturbo import write_geomturbo
    with tempfile.TemporaryDirectory() as tmp2:
        export = Path(tmp2) / "export.geomTurbo"
        r_m = 0.35        # export annulus radius, meters
        write_geomturbo(export, ref["ss"] * 0.105, ref["ps"] * 0.105,
                        r1=r_m, r2=r_m)
        # rows carry NUMECA's inverted Y and the spanwise Z computed from
        # the radius and the Cartesian y: z = sqrt(R^2 - y^2)
        text_lines = export.read_text().splitlines()
        i = text_lines.index("XYZ")
        count = int(text_lines[i + 1])
        rows = [ln.split() for ln in text_lines[i + 2:i + 2 + count]]
        y_stored = np.array([float(r[1]) for r in rows])
        z_stored = np.array([float(r[0]) for r in rows])
        y_can = ref["ss"][:, 1] * 0.105
        assert np.allclose(y_stored, -y_can, atol=1e-9), "write-side -Y flip"
        assert np.allclose(z_stored, np.sqrt(r_m ** 2 - y_can ** 2),
                           atol=1e-9), "writer Z from radius + Cartesian y"
        back = section_to_airfoil(parse_geomturbo(export)["sections"][0],
                                  n_points=len(ref["ss"]))
    assert len(back["ss"]) == len(ref["ss"])
    # the import normalization anchors at the LE and the suction-side
    # trailing point - pyturbo's axial-chord parameter differs from that
    # geometric distance by ~1%, so sub-1.5% tail deviation is expected
    dev = np.abs(back["ss"] - ref["ss"]).max()
    assert dev < 1.5e-2, f"export/import ss deviation {dev:.5f} too large"
    print(f"geomTurbo import OK (roundtrip {len(af['ss'])} pts; throat "
          f"{s_ch.min():.4f} c_ax, unguided {t['unguided_turning_deg']:.1f} deg; "
          f"export deviation {dev:.2e})")


def test_su2_mesh_reader():
    meshes = sorted((REPO / "cases").glob("*/*/mesh*.su2")) + \
        sorted((REPO / "cases").glob("*/mesh*.su2"))
    target = None
    for m in meshes:
        if "turbine_blade_4" in str(m):
            target = m
            break
    assert target is not None, "turbine_blade_4 mesh not found"
    mesh = read_su2_mesh(target)
    assert len(mesh["points"]) > 1000
    assert mesh["quads"] is not None and len(mesh["quads"]) > 1000
    assert "airfoil" in mesh["markers"] and "inlet" in mesh["markers"]
    edges = mesh_edges(mesh["tris"], mesh["quads"])
    assert len(edges) > len(mesh["quads"])
    print(f"su2 mesh reader OK ({target.name}: {len(mesh['points'])} nodes, "
          f"{len(mesh['quads'])} quads, {len(mesh['markers'])} markers)")


def test_geomturbo_external_samples():
    """Parse locally downloaded public samples when available."""
    sample_dir = REPO / "sample" / "geomturbo_external"
    files = sorted(sample_dir.glob("*.geomTurbo"))
    if not files:
        print("external geomTurbo samples SKIPPED (none downloaded)")
        return

    from pipeline.geomturbo import parse_geomturbo
    for path in files:
        parsed = parse_geomturbo(path)
        assert parsed["blade_count"] > 0
        assert parsed["sections"]
        assert parsed["blade_geometries"]
        for geometry in parsed["blade_geometries"]:
            assert geometry["sections"]
            for section in geometry["sections"]:
                for side in ("ss", "ps"):
                    points = section[side]
                    assert points.ndim == 2 and points.shape[1] == 3
                    assert len(points) >= 2 and np.isfinite(points).all()
    print(f"external geomTurbo samples OK ({len(files)} files)")


def test_postproc_integration():
    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from plot_case import fields_from_volume, volume_triangulation
    vols = sorted((REPO / "cases").glob("turbine_blade_4/vol_solution*.vtk"))
    if not vols:
        print("postproc integration SKIPPED (no vol_solution.vtk)")
        return
    d = read_legacy_vtk_local(vols[-1])
    pts, conn = volume_triangulation(d)
    fields = fields_from_volume(d)
    assert "Mach" in fields and "Pressure_Coefficient" in fields
    assert len(pts) == len(next(iter(fields.values()))[0])
    print(f"postproc integration OK ({vols[-1].name}: "
          f"{len(fields)} fields, {len(pts)} nodes)")


def read_legacy_vtk_local(path):
    from su2_vtk import read_legacy_vtk
    return read_legacy_vtk(str(path))


def test_optimizer():
    """Optimization engine core: NSGA-II operators, hypervolume,
    constraint handling, catalogs, and the per-evaluation input writer
    (no CFD runs)."""
    import random
    from foilage_gui.optimizer import (CONSTRAINT_QUANTITIES, DESIGN_VARS,
                                       FREESTREAM_OBJECTIVES, OBJECTIVES,
                                       PENALTY, OptimizationRun,
                                       apply_kind, constraint_violation,
                                       constrained_dominates, dominates,
                                       hypervolume, lhs_sample,
                                       nondominated_fronts,
                                       objectives_for_case,
                                       constraint_quantities_for_case,
                                       polynomial_mutation, sbx_crossover)

    F = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.4, 1.5), (0.5, 0.9),
         (2.0, 2.0)]
    fronts = nondominated_fronts(F)
    assert sorted(fronts[0]) == [0, 1, 2] and sorted(fronts[1]) == [3, 4] \
        and fronts[2] == [5], fronts
    assert dominates((1, 1), (2, 2)) and not dominates((1, 2), (2, 1))

    # constrained domination (Deb): feasibility beats objective quality
    assert constrained_dominates((0.1, 0.1), 0.0, (0.0, 0.0), 0.5)
    assert constrained_dominates((1.0, 1.0), 0.0, (0.0, 0.0), 1e9)
    assert constrained_dominates((0.0, 0.0), 2.0, (5.0, 5.0), 3.0)
    assert not constrained_dominates((0.0, 0.0), 3.0, (5.0, 5.0), 2.0)
    assert constrained_dominates((0.0, 0.0), 0.0, (0.5, 0.5), 0.0)
    # fronts with CV: an infeasible point never enters the first front
    F2 = [(0.0, 0.0), (0.2, 0.2)]
    CV2 = [0.5, 0.0]
    assert 1 in nondominated_fronts(F2, CV2)[0] \
        and 0 not in nondominated_fronts(F2, CV2)[0]

    # constraint violation accounting (normalized by bound magnitude)
    cons = [{"path": "a", "op": ">=", "value": 1.0},
            {"path": "b", "op": "<=", "value": 0.5}]
    assert constraint_violation({"a": 2.0, "b": 0.2}, cons) == 0.0
    assert constraint_violation({"a": 0.5, "b": 0.2}, cons) == 0.5
    assert constraint_violation({"a": 0.5, "b": 1.0}, cons) == 0.5 + 1.0
    assert constraint_violation({"a": 2.0}, cons) >= 1.0   # missing b

    # hypervolume: unit-square front monotone in the point set
    hv_small = hypervolume([(0.5, 0.5)])
    hv_big = hypervolume([(0.5, 0.5), (0.2, 0.8), (0.8, 0.2)])
    assert hv_big > hv_small > 0.0, (hv_small, hv_big)

    bounds = [(0.0, 1.0), (10.0, 20.0), (-5.0, 5.0)]
    rng = random.Random(42)
    sample = lhs_sample(bounds, 7, rng)
    assert len(sample) == 7 and all(len(v) == 3 for v in sample)
    for d in range(3):
        col = [v[d] for v in sample]
        assert all(bounds[d][0] <= c <= bounds[d][1] for c in col)
    for _ in range(50):
        c1, c2 = sbx_crossover([0.9, 15.0, 0.0], [0.1, 18.0, -4.0],
                               bounds, 15.0, rng)
        m = polynomial_mutation(c1, bounds, 20.0, rng)
        for v, (lo, hi) in zip(list(c1) + list(c2) + m, bounds * 3):
            assert lo <= v <= hi
    assert apply_kind([2.4, 3.6], ["int", "float"]) == [2, 3.6]

    # catalogs must reference real schema paths
    schema_paths = {f.path for f in FIELDS_BY_PATH.values()}
    for dv in DESIGN_VARS:
        assert dv["path"] in schema_paths, dv["path"]
    assert len({o["path"] for o in OBJECTIVES}) == len(OBJECTIVES)
    assert "forces.CL" not in {o["path"] for o in OBJECTIVES}
    assert {o["path"] for o in FREESTREAM_OBJECTIVES} == {"forces.LD"}
    assert "forces.LD" not in {o["path"] for o in objectives_for_case(False)}
    assert "forces.LD" in {o["path"] for o in objectives_for_case(True)}
    assert "forces.LD" not in {
        c["path"] for c in constraint_quantities_for_case(False)}
    assert "forces.LD" in {
        c["path"] for c in constraint_quantities_for_case(True)}
    # Zweifel loading: objectives + constraints for periodic cascades,
    # absent for freestream (no pitch -> no Zweifel in results.json)
    assert {"zweifel.incompressible", "zweifel.compressible"} <= {
        o["path"] for o in objectives_for_case(False)}
    assert not any(p.startswith("zweifel") for p in
                   {o["path"] for o in objectives_for_case(True)})
    assert {"zweifel.incompressible", "zweifel.compressible",
            "geometry.zweifel_geometric"} <= {
        c["path"] for c in constraint_quantities_for_case(False)}
    assert not any(p.startswith("zweifel") for p in {
        c["path"] for c in constraint_quantities_for_case(True)})

    from tools.plot_case import lift_to_drag
    assert lift_to_drag(2.4, 0.12) == 20.0
    assert lift_to_drag(2.4, 0.0) is None
    assert lift_to_drag(2.4, -0.12) is None

    # the per-evaluation input writer: dv overrides, R1=R2 sync, case copy
    base = {"case": {"name": "x", "output_dir": "cases"},
            "domain": {"R1": 9.0, "R2": 9.0}}
    objs = [{"path": "a", "label": "a", "sense": "min"}]
    dvs = [{"path": "domain.airfoil_count", "label": "N", "kind": "int",
            "min": 2, "avg": 3, "max": 6}]
    run = OptimizationRun(base, Path(tempfile.mkdtemp()) / "w", "wtag",
                          objs, dvs, pop_size=4)
    eval_dir = run._write_eval_input(1, 0, [5], "wtag_g001i00")
    cfg_w = json.loads((eval_dir / "input.json").read_text())
    assert cfg_w["domain"]["airfoil_count"] == 5
    assert cfg_w["domain"]["R2"] == cfg_w["domain"]["R1"]
    assert (REPO / "cases" / "wtag_g001i00" / "input.json").exists()

    # a hanging stage is killed at its timeout and reported as timed out
    import time as _time
    t0 = _time.time()
    rc, timed_out = run._run_stage(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        str(REPO), None, timeout=3)
    assert timed_out and rc != 0 and _time.time() - t0 < 15, \
        (rc, timed_out, _time.time() - t0)
    rc2, to2 = run._run_stage([sys.executable, "-c", "print('ok')"],
                              str(REPO), None, timeout=10)
    assert rc2 == 0 and not to2

    import shutil
    shutil.rmtree(REPO / "cases" / "wtag_g001i00", ignore_errors=True)
    shutil.rmtree(run.run_dir.parent, ignore_errors=True)
    print(f"optimizer OK ({len(OBJECTIVES)} objectives, "
          f"{len(DESIGN_VARS)} design vars; front/hypervolume/bounds)")


def test_optimizer_pause_state():
    """Pause/resume control and the optimizer_state.json save/load cycle
    (instant fake evaluations - no CFD runs)."""
    import queue as _queue
    import shutil
    import time as _time
    from foilage_gui.optimizer import OptimizationRun, validate_state

    base = {"case": {"name": "x", "output_dir": "cases"}}
    objs = [{"path": "a", "label": "a", "sense": "min"}]
    dvs = [{"path": "domain.airfoil_count", "label": "N", "kind": "int",
            "min": 2, "avg": 3, "max": 6}]

    def make(tag, run_dir):
        r = OptimizationRun(base, run_dir, tag, objs, dvs, pop_size=4,
                            max_generations=10000)

        def fake_eval(gen, idx, values):
            _time.sleep(0.005)          # let pause/stop land between evals
            rec = r._make_record(gen, idx, values,
                                 f"{tag}_g{gen:03d}i{idx:02d}")
            rec["F"] = [float(idx) + 0.1 * gen]
            rec["cv"] = 0.0
            rec["status"] = "ok"
            rec["raw"] = {"a": rec["F"][0]}
            return r._finish_record(rec, _time.time())

        r._evaluate = fake_eval
        return r

    def drain(run):
        out = []
        while True:
            try:
                out.append(run.queue.get_nowait())
            except _queue.Empty:
                return out

    def kinds(run):
        return [k for k, _d in drain(run)]

    def wait_until(cond, timeout=15.0):
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            if cond():
                return True
            _time.sleep(0.02)
        return False

    tmp = Path(tempfile.mkdtemp())
    try:
        # run, pause mid-generation (auto-save), resume, grow, stop
        run = make("ptag", tmp / "p1")
        run.start()
        assert wait_until(lambda: len(run.records) >= 3), "no evaluations"
        run.pause()
        assert wait_until(lambda: "paused" in kinds(run)), "no paused event"
        state_path = run.run_dir / "optimizer_state.json"
        assert state_path.exists(), "pause did not auto-save the state"
        run.save_state()            # deferred save request while paused
        assert wait_until(lambda: "state_saved" in kinds(run)), \
            "no state_saved event"
        run.resume()
        assert wait_until(lambda: "resumed" in kinds(run)), \
            "no resumed event"
        n_before = len(run.records)
        assert wait_until(lambda: len(run.records) > n_before), \
            "run did not continue after resume"
        run.stop()
        run.thread.join(15)
        assert not run.thread.is_alive(), "worker did not stop"
        done = [d for k, d in drain(run) if k == "done"]
        assert done and done[-1]["reason"] == "stopped by user", done

        # the stop auto-saved too; reload and resume in a new run folder
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_state(state)
        assert len(state["records"]) == len(run.records)
        assert 0 <= state["pending_idx"] <= len(state["pending"])
        r2 = OptimizationRun.from_state(state, tmp / "p2", "ptag_res")
        assert len(r2.records) == len(state["records"])
        assert r2._generation == state["generation"]

        def fake2(gen, idx, values):
            _time.sleep(0.005)
            rec = r2._make_record(gen, idx, values,
                                  f"ptag_res_g{gen:03d}i{idx:02d}")
            rec["F"] = [float(idx) + 0.1 * gen]
            rec["cv"] = 0.0
            rec["status"] = "ok"
            rec["raw"] = {"a": rec["F"][0]}
            return r2._finish_record(rec, _time.time())

        r2._evaluate = fake2
        n_hist = len(r2.records)    # r2.records aliases the state list
        r2.start()
        assert wait_until(lambda: len(r2.records) > n_hist), \
            "resumed run did not evaluate"
        nxt = r2.records[n_hist]
        assert nxt["id"] >= state["counters"]["next_id"] - 1 \
            and nxt["gen"] >= state["generation"], \
            (nxt["id"], nxt["gen"], state["generation"])
        # the resumed run's csv contains the carried-over history rows
        assert wait_until(lambda: r2._csv is not None)
        r2.pause()
        assert wait_until(lambda: "paused" in kinds(r2))
        r2.stop()                   # stop while paused must wake the worker
        r2.thread.join(15)
        assert not r2.thread.is_alive()
        lines = (r2.run_dir / "evaluations.csv").read_text().splitlines()
        assert len(lines) >= n_hist + 2, \
            f"csv missing carried rows ({len(lines)} lines, " \
            f"{n_hist} carried)"
        assert (r2.run_dir / "optimizer_state.json").exists()

        # corrupt state files must be rejected with a message
        for bad in ({}, {**state, "version": 99},
                    {**state, "algorithm": "Nope"},
                    {**state, "options": {**state["options"],
                                          "pop_size": 2}}):
            try:
                validate_state(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"validate_state accepted {bad}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("optimizer pause/state OK (pause-resume, save/load, "
          "stop-while-paused, validation)")


def test_ffd_morph():
    """FFD morph cage: Bernstein basis, cage deformation behavior, and
    the end-to-end geomTurbo build with morphing (no pyturbo needed)."""
    import shutil
    import numpy as np
    from pipeline.ffd import (MARGIN_FRACTION, apply_morph, bernstein_basis,
                              cage_box, cage_control_points, is_active)

    # Bernstein basis: partition of unity, exact endpoints
    for n in (2, 3, 4, 6):
        t = np.linspace(-0.2, 1.2, 29)
        B = bernstein_basis(n, t)
        assert np.allclose(B.sum(axis=1), 1.0), n
        assert np.isclose(B[0, 0], 1.0) and np.allclose(B[0, 1:], 0.0)
        assert np.isclose(B[-1, -1], 1.0) and np.allclose(B[-1, :-1], 0.0)

    # synthetic normalized section (LE at origin, axial chord = 1)
    x = np.linspace(0.0, 1.0, 101)
    cam = 0.12 * np.sin(np.pi * x)
    thk = 0.05 * (1.0 - x) + 0.004
    ss0 = np.column_stack([x, cam + thk / 2])
    ps0 = np.column_stack([x, cam - thk / 2])

    def morph(n, dx, dy, enabled=True):
        flat = lambda v: [v] * (n * n) if isinstance(v, (int, float)) else v
        return {"enabled": enabled, "n": n, "dx": flat(dx), "dy": flat(dy)}

    # disabled: no-op, no cage info
    ss, ps, cage = apply_morph(ss0, ps0, morph(4, [0.0] * 16, [0.0] * 16,
                                              enabled=False))
    assert cage is None and np.array_equal(ss, ss0) and np.array_equal(ps, ps0)
    assert not is_active(None) and not is_active({"enabled": False})

    # zero offsets: identical geometry, valid cage info
    ss, ps, cage = apply_morph(ss0, ps0, morph(4, [0.0] * 16, [0.0] * 16))
    assert np.allclose(ss, ss0) and np.allclose(ps, ps0)
    assert np.allclose(cage["cp"], cage["cp0"])
    box = cage_box(ss0, ps0)
    assert np.allclose(cage_control_points(box, 4), cage["cp0"])
    assert cage["box"][0] < 0.0 and cage["box"][2] < float(ps0[:, 1].min())

    # rigid translation: all offsets equal -> the section moves as one
    ss, ps, _c = apply_morph(ss0, ps0, morph(4, 0.05, -0.02))
    assert np.allclose(ss, ss0 + [0.05, -0.02])
    assert np.allclose(ps, ps0 + [0.05, -0.02])

    # TE-only bump: the last cage column at the u = 1 edge moves the TE
    # by the Bernstein weight of its u position (the blade sits inside
    # the cage margin, so the edge column's influence blends, as in any
    # FFD lattice); the LE (u = 0) only sees the first column
    n = 4
    dx = [0.0] * (n * n)
    for j in range(n):
        dx[(n - 1) * n + j] = 0.2
    ss, ps, _c = apply_morph(ss0, ps0, morph(n, dx, [0.0] * (n * n)))
    bx = cage_box(ss0, ps0)
    u_of = lambda xq: (float(xq) - bx[0]) / (bx[1] - bx[0])
    u_te, u_le = u_of(ss0[-1, 0]), u_of(ss0[0, 0])
    # with only the last column offset, a point at u moves by
    # 0.2 * B_{n-1}(u) (y offsets are zero, and the v basis sums to 1)
    expected = 0.2 * float(bernstein_basis(n, [u_te])[0, -1])
    assert 0.0 < expected < 0.2, expected
    assert np.isclose(ss[-1, 0], ss0[-1, 0] + expected)
    assert np.isclose(ps[-1, 0], ps0[-1, 0] + expected)
    expected_le = 0.2 * float(bernstein_basis(n, [u_le])[0, -1])
    assert np.isclose(ss[0, 0] - ss0[0, 0], expected_le)
    assert np.isclose(ps[0, 0] - ps0[0, 0], expected_le)
    assert expected_le < 0.005 * expected            # LE barely moves
    i_mid = len(ss0) // 2
    assert ss[i_mid, 0] - ss0[i_mid, 0] < expected   # TE moves the most

    # inconsistent offsets are rejected with a clear message
    try:
        apply_morph(ss0, ps0, {"enabled": True, "n": 3,
                               "dx": [0.0] * 16, "dy": [0.0] * 9})
    except ValueError as e:
        assert "n*n" in str(e), e
    else:
        raise AssertionError("length mismatch accepted")

    # ---- end-to-end: geomTurbo build with a morphed section ----
    from pipeline.airfoil import build_geometry
    from pipeline.geomturbo import write_geomturbo

    tmp = Path(tempfile.mkdtemp())
    try:
        gt = write_geomturbo(tmp / "sec.geomTurbo", ss0 * 0.1, ps0 * 0.1,
                             z=0.0, units="m")
        base = {"airfoil": {"n_points": 121, "axial_chord": 100.0},
                "domain": {"R1": 9.0, "R2": 9.0, "airfoil_count": 45,
                           "x_min": -0.5, "x_max": 2.5}}

        def build(m=None):
            cfg = {**base, "airfoil_source": {
                "type": "geomturbo", "geomturbo_file": str(gt),
                "section": 0, "show_reference": False}}
            if m is not None:
                cfg["airfoil_source"]["morph"] = m
            return build_geometry(cfg)

        af0 = build(None)
        assert af0["morph_cage"] is None
        m = morph(4, 0.05, 0.0)
        af1 = build(m)
        assert af1["morph_cage"] is not None
        assert np.allclose(af1["ss"], af0["ss"] + [0.05, 0.0])
        assert af1["outline"].shape[1] == 2
        assert len(af1["outline"]) == len(af1["ss"]) + len(af1["ps"]) - 2
        # the pinned shared LE/TE points receive identical displacements
        # (the deformation is a pure function of position) -> the section
        # stays closed for meshing
        assert np.allclose(af1["ss"][0], af1["ps"][0])
        assert np.allclose(af1["ss"][-1], af1["ps"][-1])
        # axial chord is NOT renormalized after morphing: the stretch
        # would otherwise be designed away
        assert np.isclose(af1["ss"][-1, 0], af0["ss"][-1, 0] + 0.05)

        try:
            build(morph(3, [0.0] * 16, [0.0] * 9))
        except ValueError:
            pass
        else:
            raise AssertionError("bad morph accepted by build_geometry")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- CaseState validation of the morph block ----
    inputs = sorted((REPO / "cases").glob("*/input.json"))
    if inputs:
        from foilage_gui.state import CaseState
        st = CaseState(inputs[0])
        st.set("airfoil_source.type", "geomturbo")
        st.set("airfoil_source.morph.enabled", True)
        st.set("airfoil_source.morph.n", 3)
        msgs = [m for _s, m in st.validate()]
        assert any("dx" in m and "9" in m for m in msgs), msgs
        st.set("airfoil_source.morph.dx", [0.0] * 9)
        st.set("airfoil_source.morph.dy", [0.0] * 9)
        msgs = [m for _s, m in st.validate()]
        assert not any("offsets" in m for m in msgs), msgs
    print("FFD morph OK (basis/translation/TE-bump/closure, "
          "end-to-end geomTurbo build, validation)")


def test_plugins():
    """Extension plugins: manifest discovery, CLI run contract, the
    build_geometry plugin path (with morph), and validation."""
    import shutil
    import numpy as np
    from pipeline.plugins import (discover_plugins, get_plugin,
                                  run_plugin_generator, type_choices)

    # ---- shipped sample plugin is discoverable and well-formed
    plugins = discover_plugins()
    naca = get_plugin("naca")
    assert naca is not None and not naca["error"], naca
    assert naca["cli"] and Path(naca["cli"]).is_file()
    assert [p["path"] for p in naca["parameters"]] == \
        ["max_camber", "camber_position", "thickness", "n_points_surface"]
    assert len(naca["groups"]) == 2
    assert ("NACA 4-digit generator", "naca") in type_choices(plugins)

    # ---- CLI contract: the script runs and returns valid sections
    cfg = {"airfoil": {"n_points": 161, "axial_chord": 100.0},
           "case": {"name": "plugincase"},
           "airfoil_source": {"type": "naca", "params": {
               "max_camber": 0.02, "camber_position": 0.4,
               "thickness": 0.12, "n_points_surface": 1}}}
    res = run_plugin_generator("naca", cfg)
    assert len(res["ss"]) >= 100 and len(res["ss"]) == len(res["ps"])
    ss = np.asarray(res["ss"])
    ps = np.asarray(res["ps"])
    assert abs(ss[0, 0]) < 1e-9 and abs(ss[-1, 0] - 1.0) < 1e-9
    assert ss[:, 1].max() > ps[:, 1].max()          # cambered: SS above PS
    # the parameter cache: an unchanged call returns the same result
    assert run_plugin_generator("naca", cfg) is res
    cfg2 = {**cfg, "airfoil_source": {**cfg["airfoil_source"],
                                      "params": {**cfg["airfoil_source"]["params"],
                                                 "thickness": 0.2}}}
    res2 = run_plugin_generator("naca", cfg2)
    assert np.ptp(np.asarray(res2["ss"])[:, 1]) > np.ptp(ss[:, 1])  # thicker

    # ---- build_geometry plugin path, with and without the morph cage
    from pipeline.airfoil import build_geometry

    def build(morph=None, params=None):
        src = {"type": "naca", "params": params or cfg["airfoil_source"]["params"]}
        if morph is not None:
            src["morph"] = morph
        return build_geometry({**cfg, "airfoil_source": src})

    af = build()
    assert af["morph_cage"] is None and af["airfoil2d"] is None
    assert abs(af["ss"][0, 0]) < 1e-6               # LE pinned at x = 0
    assert af["ss"][-1, 0] > 0.99                   # axial chord = 1
    dy = [0.0] * 16
    m = {"enabled": True, "n": 4, "dx": [0.0] * 16, "dy": dy}
    af_m = build(morph={**m, "dy": [0.05] * 16})
    assert af_m["morph_cage"] is not None
    assert np.allclose(af_m["ss"], af["ss"] + [0.0, 0.05])

    # ---- broken / unknown plugins fail with clear messages
    tmp = Path(tempfile.mkdtemp())
    try:
        bad = tmp / "broken_one"
        bad.mkdir()
        (bad / "plugin.json").write_text('{"name": "Broken", "cli": "nope.py"}')
        plugins = discover_plugins(tmp)
        assert len(plugins) == 1 and plugins[0]["error"]
        assert type_choices(plugins) == []
        try:
            run_plugin_generator("broken_one", cfg, extensions_dir=tmp)
        except ValueError as e:
            assert "cli script not found" in str(e), e
        else:
            raise AssertionError("broken plugin accepted")
        try:
            run_plugin_generator("does_not_exist", cfg, extensions_dir=tmp)
        except ValueError as e:
            assert "unknown geometry source plugin" in str(e), e
        else:
            raise AssertionError("unknown plugin accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- CaseState validation of plugin sources
    inputs = sorted((REPO / "cases").glob("*/input.json"))
    if inputs:
        from foilage_gui.state import CaseState
        st = CaseState(inputs[0])
        st.set("airfoil_source.type", "naca")
        assert not any("plugin" in m.lower()
                       for _s, m in st.validate())
        st.set("airfoil_source.type", "no_such_plugin")
        msgs = [m for _s, m in st.validate()]
        assert any("known plugin" in m for m in msgs), msgs
    print(f"plugins OK ({len(plugins)} discovered here; sample NACA runs, "
          "builds, morphs, validates)")


def test_optimizer_dvs():
    """Source-dependent design-variable catalogs and the FFD cage
    offset variables (list-element writing, morph auto-enable)."""
    import shutil
    from foilage_gui.optimizer import (DESIGN_VARS, OptimizationRun,
                                       design_variables_for,
                                       set_design_value)

    # pyturbo keeps the classic catalog
    assert [d["path"] for d in design_variables_for("pyturbo")] == \
        [d["path"] for d in DESIGN_VARS]

    # plugin source: manifest params + domain + cage offsets
    dvs = design_variables_for("naca", morph_n=3)
    by_path = {d["path"]: d for d in dvs}
    assert "airfoil_source.params.thickness" in by_path
    assert by_path["airfoil_source.params.thickness"]["min"] == 0.02
    assert by_path["airfoil_source.params.thickness"]["max"] == 0.4
    assert "domain.R1" in by_path and "domain.airfoil_count" in by_path
    assert len([p for p in by_path if "morph.dx@" in p]) == 9
    assert len([p for p in by_path if "morph.dy@" in p]) == 9
    assert all(d["min"] == -0.1 and d["max"] == 0.1
               for d in by_path.values() if "morph." in d["path"])
    # non-numeric or unbounded plugin parameters are not offered
    assert all("kind" not in d or d["kind"] in ("float", "int")
               for d in dvs)

    # geomturbo: no plugin params, but domain + cage
    dvg = design_variables_for("geomturbo", morph_n=2)
    assert not any(d["path"].startswith("airfoil_source.params")
                   for d in dvg)
    assert len([d for d in dvg if "@" in d["path"]]) == 8

    # set_design_value addresses list elements, growing with zeros
    cfg = {"airfoil_source": {"morph": {"enabled": False, "n": 2,
                                        "dx": [0.0] * 4, "dy": [0.0] * 4}}}
    set_design_value(cfg, "airfoil_source.morph.dx@2", 0.07)
    assert cfg["airfoil_source"]["morph"]["dx"][2] == 0.07
    set_design_value(cfg, "airfoil_source.morph.dy@9", -0.02)   # grows
    assert cfg["airfoil_source"]["morph"]["dy"][9] == -0.02
    set_design_value(cfg, "domain.R1", 3.5)
    assert cfg["domain"]["R1"] == 3.5

    # an eval input writer with cage DVs writes the offsets into the
    # arrays and force-enables the morph
    base = {"case": {"name": "x", "output_dir": "cases"},
            "airfoil": {"n_points": 121, "axial_chord": 100.0},
            "airfoil_source": {"type": "geomturbo",
                               "geomturbo_file": "whatever.geomTurbo",
                               "morph": {"enabled": False, "n": 2,
                                         "dx": [0.0] * 4, "dy": [0.0] * 4}},
            "domain": {"R1": 3.0, "R2": 3.0, "airfoil_count": 30}}
    objs = [{"path": "a", "label": "a", "sense": "min"}]
    run = OptimizationRun(base, Path(tempfile.mkdtemp()) / "w", "cwtag",
                          objs, dvg, pop_size=4)
    values = [3.7, 41] + [0.0, 0.0, 0.05, 0.0,   # dx: index 2 set
                          0.0, -0.03, 0.0, 0.0]  # dy: index 1 set
    eval_dir = run._write_eval_input(1, 0, values, "cwtag_g001i00")
    written = json.loads((eval_dir / "input.json").read_text())
    m = written["airfoil_source"]["morph"]
    assert m["enabled"] is True, "morph not auto-enabled"
    assert m["dx"][2] == 0.05 and m["dy"][1] == -0.03
    assert written["domain"]["R1"] == 3.7
    assert written["domain"]["airfoil_count"] == 41
    assert written["domain"]["R2"] == written["domain"]["R1"]

    import shutil as _sh
    _sh.rmtree(REPO / "cases" / "cwtag_g001i00", ignore_errors=True)
    _sh.rmtree(run.run_dir.parent, ignore_errors=True)
    print(f"optimizer DVs OK (pyturbo {len(DESIGN_VARS)}, plugin+cage "
          f"{len(dvs)}, geomturbo+cage {len(dvg)}; morph auto-enable)")


def test_periodicity_modes():
    """Periodicity modes: offset (constant pitch, straight periodics),
    freestream (no periodics, far-field box), and the Cartesian-y <->
    unwrapped-y conversion of imported sections in axisymmetric mode."""
    import shutil
    import numpy as np
    from pipeline.mesh_tris import periodic_edges, pitch_profile
    from pipeline.unwrap import (cartesian_to_uy, is_periodic,
                                 periodicity_of, uy_to_cartesian)

    # mode helpers
    assert periodicity_of({}) == "axisymmetric"
    assert periodicity_of({"periodicity": "weird"}) == "axisymmetric"
    assert not is_periodic({"periodicity": "freestream"})
    assert is_periodic({"periodicity": "offset"})

    # conversions: exact definitions and a lossless roundtrip
    y = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
    sec = np.column_stack([np.linspace(0, 1, 5), y])
    R = 0.9
    uy, _ = cartesian_to_uy(sec, sec, R)
    assert np.allclose(uy[:, 1], R * np.arcsin(y / R))
    back, _ = uy_to_cartesian(uy, uy, R)
    assert np.allclose(back, sec)
    same, _ = cartesian_to_uy(sec, sec, None)      # no radius -> identity
    assert np.array_equal(same, sec)

    af = {"ss": np.column_stack([np.linspace(0, 1, 50),
                                 np.linspace(0.05, -0.05, 50)]),
          "ps": np.column_stack([np.linspace(0, 1, 50),
                                 np.linspace(-0.05, 0.05, 50)]),
          "axial_chord": 100.0}
    dom = {"R1": 90.0, "R2": 110.0, "airfoil_count": 45,
           "x_min": -0.5, "x_max": 2.5}

    # axisymmetric: pitch follows the radius
    prof = pitch_profile({**dom, "periodicity": "axisymmetric"}, af)
    assert prof["periodic"] and abs(prof["p_le"] - prof["p_te"]) > 1e-9

    # offset: constant pitch (R2 ignored), straight periodic lines at
    # y = -/+ p/2 with no tangential offset
    profo = pitch_profile({**dom, "periodicity": "offset"}, af)
    assert profo["periodic"] and abs(profo["p_le"] - profo["p_te"]) < 1e-12
    assert abs(profo["p_le"] - 2 * np.pi * 0.9 / 45) < 1e-12
    bot, top = periodic_edges(profo, -0.5, 2.5)
    assert np.allclose(bot[:, 1], -profo["p_le"] / 2.0)
    assert np.allclose(top[:, 1] - bot[:, 1], profo["p_le"])

    # freestream: no periodics, straight far-field lines at y_min / y_max
    proff = pitch_profile({**dom, "periodicity": "freestream",
                           "y_min": -1.5, "y_max": 1.5}, af)
    assert not proff["periodic"]
    botf, topf = periodic_edges(proff, -0.5, 2.5)
    assert np.allclose(botf[:, 1], -1.5) and np.allclose(topf[:, 1], 1.5)

    # ---- imported sections unwrap only in axisymmetric mode; the export
    # re-wraps them back to Cartesian y (lossless roundtrip)
    from pipeline.airfoil import build_geometry
    from pipeline.geomturbo import write_geomturbo
    tmp = Path(tempfile.mkdtemp())
    try:
        x = np.linspace(0.0, 0.1, 61)
        cam = 0.02 * np.sin(np.pi * x / 0.1)
        thk = 0.004 * (1 - x / 0.1) + 0.0008
        gt = write_geomturbo(
            tmp / "s.geomTurbo",
            np.column_stack([x, cam + thk / 2]),
            np.column_stack([x, cam - thk / 2]), z=0.0, units="m")

        def build(mode):
            return build_geometry({
                "airfoil": {"n_points": 121, "axial_chord": 100.0},
                "domain": {"periodicity": mode, "R1": 90.0, "R2": 90.0,
                           "airfoil_count": 45, "x_min": -0.5, "x_max": 2.5},
                "airfoil_source": {"type": "geomturbo",
                                   "geomturbo_file": str(gt),
                                   "section": 0}})

        af_ax = build("axisymmetric")
        af_off = build("offset")
        assert np.allclose(af_ax["ss"][:, 0], af_off["ss"][:, 0])
        assert np.max(np.abs(af_ax["ss"][:, 1])) > \
            np.max(np.abs(af_off["ss"][:, 1])), "unwrap must stretch |y|"
        # export direction: re-wrap -> exactly the Cartesian shape
        back_ss, back_ps = uy_to_cartesian(af_ax["ss"], af_ax["ps"], 0.9)
        assert np.allclose(back_ss, af_off["ss"])
        assert np.allclose(back_ps, af_off["ps"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- validation follows the mode
    inputs = sorted((REPO / "cases").glob("*/input.json"))
    if inputs:
        from foilage_gui.state import CaseState
        st = CaseState(inputs[0])
        st.set("domain.R2", st.get("domain.R1") + 1.0)
        assert any("R1" in m for _s, m in st.validate())
        st.set("domain.periodicity", "offset")
        assert not any("R1" in m for _s, m in st.validate())
        st.set("domain.periodicity", "freestream")
        st.set("domain.y_min", 1.0)
        st.set("domain.y_max", -1.0)
        assert any("y_max" in m for _s, m in st.validate())

    # ---- freestream case.json renders MARKER_FAR and no MARKER_PERIODIC
    from render_config import cascade_marker_lines, is_freestream_case
    from pipeline.quadify import (FREESTREAM_MARKERS, PERIODIC_MARKERS,
                                  _marker_layout)
    assert _marker_layout(("fluid", "airfoil", "inlet", "outlet",
                           "farfield")) == FREESTREAM_MARKERS
    assert _marker_layout(("fluid", "airfoil", "inlet", "outlet",
                           "periodic_bottom", "periodic_top")) == \
        PERIODIC_MARKERS
    case = {"boundary_mode": "cascade", "cascade": {
        "inlet": {"marker": "inlet", "total_pressure": 125000.0,
                  "total_temperature": 700.0, "direction": [1.0, 0.0, 0.0]},
        "outlet": {"marker": "outlet", "static_pressure": 60000.0}},
        "markers": {"airfoil": {"bc": "wall_adiabatic"},
                    "farfield": {"bc": "farfield"}}}
    txt = cascade_marker_lines(case)
    assert "MARKER_FAR= ( farfield )" in txt and "MARKER_PERIODIC" not in txt
    assert not is_freestream_case(case)
    case["boundary_mode"] = "freestream"
    case["markers"] = {
        "airfoil": {"bc": "wall_adiabatic"},
        "inlet": {"bc": "farfield"},
        "outlet": {"bc": "farfield"},
        "farfield": {"bc": "farfield"},
    }
    txt = cascade_marker_lines(case)
    assert "MARKER_FAR= ( inlet, outlet, farfield )" in txt
    assert "MARKER_INLET" not in txt and "MARKER_OUTLET" not in txt
    case["cascade"]["periodic"] = [
        {"markers": ["periodic_bottom", "periodic_top"],
         "translation": [0.0, 0.012, 0.0]}]
    assert "MARKER_PERIODIC" not in cascade_marker_lines(case)
    case["boundary_mode"] = "cascade"
    assert "MARKER_PERIODIC" in cascade_marker_lines(case)
    print("periodicity OK (offset const-pitch, freestream box, "
          "Cartesian<->uy unwrap, cfg markers, validation)")


def test_zweifel():
    """Zweifel loading coefficients: the exact incompressible criterion,
    the geometric metal-angle predictor, and the post-processing block
    written for periodic cases."""
    import shutil
    import numpy as np
    from pipeline.cascade_metrics import metal_angles, zweifel_geometric

    # exact-math check of the classic criterion: s/bx = 1, a1 = 30 deg,
    # a2 = -60 deg -> 2 * cos^2(60) * (tan30 + tan60) = 1.1547
    assert abs(zweifel_geometric(1.0, 30.0, -60.0) - 1.154700538) < 1e-6
    # metal angles of a straight (uncambered) section are ~0
    ss = np.column_stack([np.linspace(0, 1, 30), np.zeros(30)])
    ps = np.column_stack([np.linspace(0, 1, 30), -0.1 * np.ones(30)])
    a1, a2 = metal_angles(ss, ps)
    assert abs(a1) < 1e-6 and abs(a2) < 1e-6, (a1, a2)

    # post-processing block from a solved periodic case
    src = REPO / "cases" / "turbine_blade_4"
    if not (src / "vol_solution.vtk").exists() or \
            not (src / "case.json").exists():
        print("zweifel SKIPPED (no solved periodic case on disk)")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        for f in ("case.json", "history.csv", "vol_solution.vtk", "mesh.su2"):
            shutil.copy2(src / f, tmp / f)
        from plot_case import write_results
        from freestream import state as freestream_state
        case = json.loads((tmp / "case.json").read_text())
        ph = case["physics"]
        fs = freestream_state(
            ph["mach"], ph["reynolds"],
            ph.get("reynolds_length", 1.0),
            ph.get("freestream_temperature", 288.15),
            ph.get("gamma", 1.4), ph.get("gas_constant", 287.058))
        write_results(tmp, case, fs, dist=None)
        res = json.loads((tmp / "results.json").read_text())
        zw = res.get("zweifel")
        assert zw, "no zweifel block for a periodic case"
        assert zw["incompressible"] > 0.0 and zw["compressible"] > 0.0
        ratio = zw["compressible"] / zw["incompressible"]
        assert 0.5 <= ratio <= 3.0, ratio       # rho1/rho2 of a turbine
        assert zw["alpha1_deg"] > 0.0 > zw["alpha2_deg"]
        print(f"zweifel OK (inc {zw['incompressible']:.3f}, "
              f"comp {zw['compressible']:.3f}, rho-ratio {ratio:.3f}, "
              f"s/bx {zw['pitch_over_axial_chord']:.3f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_warm_start_restart():
    """Warm-start (RESTART_SOL) robustness: the validation step keeps the
    previous restart.dat, and a restart from a different mesh fails with
    an actionable message instead of a cryptic SU2 crash."""
    from setup_cascade_case import restore_warm_start_restart
    from foilage_gui.runner import Job, StageFailed

    # 1) the validate backup/restore cycle preserves the warm-start file
    su2 = Path(tempfile.mkdtemp())
    bdir = su2 / "results_backup_x"
    bdir.mkdir()
    (bdir / "restart.dat").write_text('"ID"\n1\n2\n')
    (su2 / "restart.dat").write_text('"ID"\nvalidate-run-junk\n')
    assert restore_warm_start_restart(su2, bdir) is True
    assert "validate-run-junk" not in (su2 / "restart.dat").read_text()
    assert restore_warm_start_restart(su2, None) is False
    empty = su2 / "results_backup_empty"
    empty.mkdir()
    (su2 / "restart.dat").write_text("kept\n")
    assert restore_warm_start_restart(su2, empty) is False
    assert (su2 / "restart.dat").read_text() == "kept\n"

    # 2) the solve stage patches RESTART_SOL plus the restart-input
    #    filenames (SU2 v8 defaults them to solution.dat) and rejects
    #    mismatched restart files before SU2 starts
    job = Job("t", [])
    case = Path(tempfile.mkdtemp())
    (case / "turbine.cfg").write_text("SOLVER= RANS\n")
    (case / "mesh.su2").write_text("NDIME= 2\nNNODES= 5\n")
    (case / "restart.dat").write_text('"ID"\n1\n2\n3\n4\n5\n')
    job._patch_restart({"cwd": str(case)})
    text = (case / "turbine.cfg").read_text()
    assert "RESTART_SOL= YES" in text
    assert "RESTART_FILENAME= restart.dat" in text
    assert "SOLUTION_FILENAME= restart.dat" in text
    (case / "restart.dat").write_text('"ID"\n1\n2\n3\n4\n5\n6\n7\n')
    try:
        job._patch_restart({"cwd": str(case)})
    except StageFailed as e:
        assert "different mesh" in str(e), e
    else:
        raise AssertionError("mesh/restart mismatch not detected")
    # no mesh on disk -> check skipped, patch still applied
    (case / "mesh.su2").unlink()
    job._patch_restart({"cwd": str(case)})
    assert "RESTART_FILENAME= restart.dat" in (case / "turbine.cfg").read_text()
    shutil.rmtree(su2, ignore_errors=True)
    shutil.rmtree(case, ignore_errors=True)
    print("warm-start restart OK (validate preserves it, mismatch "
          "detected before SU2)")


def test_ml():
    """ML tab building blocks: SQLite sample store, the pure-numpy MLP
    (train + predict on a synthetic function), and the sampler's input
    writer (design overrides, morph auto-enable, R1=R2 sync)."""
    from foilage_gui.ml_store import MLStore
    from foilage_gui.ml_net import predict as mlp_predict
    from foilage_gui.ml_net import train_mlp
    from foilage_gui.ml_runner import MLSamplingRun

    tmp = Path(tempfile.mkdtemp())
    try:
        # ---- store roundtrip
        store = MLStore(tmp / "ml.db")
        ds_id = store.create_dataset("d1", ["a", "b"], ["y1", "y2"])
        assert store.dataset(ds_id)["x_columns"] == ["a", "b"]
        ids = store.add_samples(ds_id, [{"a": 1.0, "b": 2.0},
                                        {"a": 3.0, "b": 4.0}])
        pend = store.pending(ds_id)
        assert len(pend) == 2 and all(s["status"] == "pending" for s in pend)
        store.record_result(ids[0], {"y1": 10.0, "y2": 20.0}, "case1")
        store.record_failure(ids[1], "mesh hang", "case2")
        assert [s["status"] for s in store.pending(ds_id)] == ["failed"]
        # importing rows that already carry y fills them immediately
        store.add_samples(ds_id, [{"a": 5.0, "b": 6.0}],
                          [{"y1": 1.0, "y2": 2.0}])
        assert len(store.filled(ds_id)) == 2

        # ---- per-column LHS bounds: persisted with the dataset
        assert store.dataset(ds_id)["x_bounds"] == {}
        store.set_x_bounds(ds_id, {"a": (-0.5, 0.5)})
        assert store.dataset(ds_id)["x_bounds"] == {"a": (-0.5, 0.5)}
        assert MLStore(tmp / "ml.db").dataset(ds_id)["x_bounds"] == \
            {"a": (-0.5, 0.5)}          # survives a fresh connection
        ds3 = store.create_dataset("d3", ["a"], ["y3"],
                                   x_bounds={"a": (1, 2)})
        assert store.dataset(ds3)["x_bounds"] == {"a": (1.0, 2.0)}

        # ---- MLP learns a smooth 2-in 2-out function
        rng = np.random.default_rng(7)
        X = rng.uniform(0.0, 1.0, size=(140, 2))
        y = np.column_stack([np.sin(2 * np.pi * X[:, 0]) * X[:, 1],
                             X[:, 0] + X[:, 1] ** 2])
        model, info = train_mlp(X, y, hidden=(48, 48), epochs=300,
                                lr=2e-3, batch=16, val_frac=0.2, seed=3)
        pred = mlp_predict(model, X)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        assert rmse < 0.05, f"MLP too inaccurate: RMSE {rmse}"
        assert len(info["history"]) >= 5 and info["val_rmse"] < 0.1

        # ---- sampler input writer: overrides, morph enable, R1=R2
        base = {"case": {"name": "x", "output_dir": "cases"},
                "domain": {"R1": 3.0, "R2": 3.0},
                "airfoil_source": {"type": "geomturbo",
                                   "morph": {"enabled": False, "n": 2,
                                             "dx": [0.0] * 4,
                                             "dy": [0.0] * 4}}}
        runner = MLSamplingRun(
            base, tmp / "run", "mltag",
            [{"airfoil.alpha1": 30.0}],
            ["airfoil.alpha1"], ["losses.total_pressure_loss_coeff_Yp"])
        run_dir = runner._write_input(
            0, {"airfoil.alpha1": 30.0, "airfoil_source.morph.dx@1": 0.05},
            "mltag_p000")
        cfgw = json.loads((run_dir / "input.json").read_text())
        assert cfgw["airfoil"]["alpha1"] == 30.0
        assert cfgw["airfoil_source"]["morph"]["enabled"] is True
        assert cfgw["airfoil_source"]["morph"]["dx"][1] == 0.05
        assert cfgw["domain"]["R2"] == cfgw["domain"]["R1"]
        assert cfgw["case"]["name"] == "mltag_p000"

        # int-kind X columns are rounded before they reach input.json: a
        # float blade count / flow-guidance exponent crashes the pipeline
        runner2 = MLSamplingRun(
            base, tmp / "run2", "mltag2",
            [{"domain.airfoil_count": 30.6}],
            ["domain.airfoil_count", "airfoil.ss_flow_guidance.n"],
            ["losses.total_pressure_loss_coeff_Yp"],
            x_kinds={"domain.airfoil_count": "int",
                     "airfoil.ss_flow_guidance.n": "int"})
        run_dir2 = runner2._write_input(
            0, {"domain.airfoil_count": 30.6,
                "airfoil.ss_flow_guidance.n": 9.7}, "mltag2_p000")
        cfgw2 = json.loads((run_dir2 / "input.json").read_text())
        n = cfgw2["domain"]["airfoil_count"]
        assert isinstance(n, int) and n == 31, n
        nguid = cfgw2["airfoil"]["ss_flow_guidance"]["n"]
        assert isinstance(nguid, int) and nguid == 10, nguid

        # dataset deletion removes the dataset and its samples
        store2 = MLStore(tmp / "ml2.db")
        ds2 = store2.create_dataset("doomed", ["a"], ["y"])
        store2.add_samples(ds2, [{"a": 1.0}], [{"y": None}])
        assert store2.delete_dataset(ds2) is True
        assert store2.dataset(ds2) is None and store2.samples(ds2) == []
        assert store2.delete_dataset(ds2) is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(REPO / "cases" / "mltag_p000", ignore_errors=True)
    print("ml OK (store roundtrip, MLP train/predict, sampler writer)")


def main():
    test_schema()
    test_state_roundtrip()
    test_validation()
    test_derived()
    test_gamma_model()
    test_cascade_initialization()
    test_optimizer()
    test_optimizer_pause_state()
    test_ffd_morph()
    test_plugins()
    test_optimizer_dvs()
    test_periodicity_modes()
    test_zweifel()
    test_warm_start_restart()
    test_ml()
    test_geomturbo_import()
    test_geomturbo_external_samples()
    test_su2_mesh_reader()
    test_postproc_integration()
    print("selftest OK")


if __name__ == "__main__":
    main()
