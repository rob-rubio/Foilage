"""Logic selftest for the GUI package (no window required).

    python -m foilage_gui --selftest
"""

import json
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


def test_geomturbo_import():
    import copy
    import tempfile
    if str(REPO / "pipeline") not in sys.path:
        sys.path.insert(0, str(REPO / "pipeline"))
    from pipeline.airfoil import build_airfoil, build_geometry
    from pipeline.cascade_metrics import channel_widths, curvature, \
        throat_metrics
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
            lines += [f"{section_z:.9f} {y * scale:.9f} {x * scale:.9f}"
                      for x, y in ss]
        lines += ["pressure", "SECTIONAL", "2"]
        for section_z in (0.0, 0.025):
            lines += [f"# SECTION {1 if section_z == 0.0 else 2}",
                      "XYZ", str(len(ps))]
            lines += [f"{section_z:.9f} {y * scale:.9f} {x * scale:.9f}"
                      for x, y in ps]
        lines += ["NI_END nibladegeometry"]
        gt.write_text("\n".join(lines))

        parsed = parse_geomturbo(gt)
        assert parsed["blade_count"] == 20
        assert len(parsed["sections"]) == 2
        assert parsed["sections"][0]["ss"].shape == (len(ss), 3)
        assert parsed["sections"][0]["ps"].shape == (len(ps), 3)
        assert parsed["hub"] is not None and parsed["shroud"] is not None

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
    s_ch, _j = channel_widths(ref["ss"], ref["ps"], prof["p_le"],
                              ref["ss_upper"])
    # CUP mirror check: flipping the blade vertically must give the same
    # channel widths (the neighbor direction flips with the surfaces)
    ss_m = ref["ss"].copy()
    ps_m = ref["ps"].copy()
    ss_m[:, 1] *= -1.0
    ps_m[:, 1] *= -1.0
    s_ch_m, _j = channel_widths(ss_m, ps_m, prof["p_le"],
                                not ref["ss_upper"])
    assert np.abs(s_ch - s_ch_m).max() < 1e-9, "CUP/CAP channel mismatch"
    assert 0.05 < s_ch.min() < prof["p_le"], "throat width out of range"
    t = throat_metrics(ref["ss"], ref["ps"], prof["p_le"], ref["ss_upper"])
    assert 0.1 < t["x_over_cax"] < 0.95
    assert 0.0 < t["unguided_turning_deg"] < 45.0
    assert t["width"] > 0
    k_ss, _ = curvature(ref["ss"])
    assert np.isfinite(k_ss).all() and k_ss.max() > k_ss.min()

    # geomTurbo export -> import roundtrip (export is in meters; the
    # import re-normalizes to axial chord = 1)
    from pipeline.geomturbo import write_geomturbo
    with tempfile.TemporaryDirectory() as tmp2:
        export = Path(tmp2) / "export.geomTurbo"
        write_geomturbo(export, ref["ss"] * 0.105, ref["ps"] * 0.105, z=0.0)
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


def main():
    test_schema()
    test_state_roundtrip()
    test_validation()
    test_derived()
    test_gamma_model()
    test_geomturbo_import()
    test_geomturbo_external_samples()
    test_su2_mesh_reader()
    test_postproc_integration()
    print("selftest OK")


if __name__ == "__main__":
    main()
