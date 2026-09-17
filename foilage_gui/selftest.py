"""Logic selftest for the GUI package (no window required).

    python -m foilage_gui --selftest
"""

import json
import sys
import tempfile
from pathlib import Path

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
    test_su2_mesh_reader()
    test_postproc_integration()
    print("selftest OK")


if __name__ == "__main__":
    main()
