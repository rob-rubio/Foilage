"""Declarative schema of input.json - drives the smart form widgets.

Every editable quantity is described once here with its type, allowed
choices, slider range, default, unit and tooltip. The tabs build their
forms from SECTIONS, and CaseState uses DEFAULTS to fill missing keys
when loading an existing input.json.
"""

from dataclasses import dataclass, field as dc_field
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class FieldSpec:
    path: str                      # dotted path into input.json, e.g. "mesh.max_size"
    label: str                     # human label shown in the form
    kind: str = "float"            # float|int|str|bool|choice|float_array|nullable_float
    default: Any = None
    min: Optional[float] = None    # slider / validation range
    max: Optional[float] = None
    step: Optional[float] = None   # entry hint; int fields use it as spin step
    choices: Optional[List[Tuple[str, Any]]] = None   # (display label, stored value)
    slider: bool = False           # pair the entry with a slider
    unit: str = ""
    tooltip: str = ""
    array_fixed: bool = False      # float_array: forbid changing the length

    @property
    def tab(self) -> str:
        return _path_tab[self.path]


@dataclass(frozen=True)
class Section:
    tab: str
    title: str
    fields: List[FieldSpec] = dc_field(default_factory=list)


def _c(label, value):
    return (label, value)


# ---------------------------------------------------------------- choices
TURBULENCE_MODELS = [_c("SA  (Spalart-Allmaras)", "SA"),
                     _c("SST  (k-omega Menter)", "SST")]
LIMITERS = [_c("Venkatakrishnan (robust default)", "VENKATAKRISHNAN"),
            _c("None (unlimited, most accurate)", "NONE"),
            _c("Venkatakrishnan-Wang", "VENKATAKRISHNAN_WANG"),
            _c("Barth-Jespersen", "BARTH_JESPERSEN"),
            _c("Nishikawa R3", "NISHIKAWA_R3"),
            _c("Nishikawa R4", "NISHIKAWA_R4"),
            _c("Nishikawa R5", "NISHIKAWA_R5"),
            _c("Van Albada (edge)", "VAN_ALBADA_EDGE"),
            _c("Sharp edges", "SHARP_EDGES"),
            _c("Wall distance", "WALL_DISTANCE")]
GRADIENTS = [_c("Weighted least squares", "WEIGHTED_LEAST_SQUARES"),
             _c("Green-Gauss", "GREEN_GAUSS"),
             _c("Least squares", "LEAST_SQUARES")]
MESH_ALGORITHMS = [_c("6 - Frontal-Delaunay (recommended)", 6),
                   _c("5 - Delaunay", 5),
                   _c("1 - MeshAdapt", 1),
                   _c("7 - BAMG", 7),
                   _c("8 - Frontal-Delaunay for Quads", 8),
                   _c("9 - Packing of Parallelograms", 9)]


SECTIONS = [
    # ------------------------------------------------------------------ setup
    Section("setup", "Case", [
        FieldSpec("case.name", "Case name", "str", "case",
                  tooltip="Output folder name under cases/ and SU2 case name."),
        FieldSpec("case.output_dir", "Output directory", "str", "cases",
                  tooltip="Where the mesh-project outputs go, relative to the "
                          "input.json location."),
        FieldSpec("case.show_images", "Open PNG images", "bool", False,
                  tooltip="Let the mesh pipeline pop up generated images "
                          "(they are always written to the case folder)."),
    ]),
    Section("setup", "Boundary conditions", [
        FieldSpec("BCs.inlet.total pressure", "Inlet total pressure", "float",
                  125000.0, 1e3, 5e6, slider=True, unit="Pa",
                  tooltip="Total (stagnation) pressure p01 at the inlet."),
        FieldSpec("BCs.inlet.total temperature", "Inlet total temperature",
                  "float", 700.0, 100.0, 2500.0, slider=True, unit="K",
                  tooltip="Total (stagnation) temperature T01 at the inlet."),
        FieldSpec("BCs.inlet.gas angle", "Inlet flow angle", "float",
                  0.0, -89.0, 89.0, slider=True, unit="deg",
                  tooltip="Flow direction at the inlet, measured from +x "
                          "(axial chord direction)."),
        FieldSpec("BCs.outlet.static pressure", "Outlet static pressure",
                  "float", 60000.0, 1e3, 5e6, slider=True, unit="Pa",
                  tooltip="Static back pressure p2 at the outlet. "
                          "Must be below the inlet total pressure."),
    ]),
    Section("setup", "Solver", [
        FieldSpec("solver_settings.turbulence_model", "Turbulence model",
                  "choice", "SA", choices=TURBULENCE_MODELS,
                  tooltip="RANS closure. SA is robust for cascade sweeps; "
                          "SST usually predicts separation and losses better."),
        FieldSpec("solver_settings.gamma", "Ratio of specific heats (gamma)",
                  "nullable_float", None, 1.1, 1.7, step=0.01,
                  tooltip="Set a value to fix gamma (air: 1.4), or tick auto "
                          "to compute it from the inlet total temperature "
                          "using the temperature-dependent specific heats of "
                          "air (NASA Glenn model: 1.400 at 300 K, 1.364 at "
                          "700 K)."),
        FieldSpec("solver_settings.max_iterations", "Max iterations", "int",
                  6000, 100, 100000, step=100,
                  tooltip="SU2 stops earlier once CONV_RESIDUAL_MINVAL is "
                          "reached for all fields."),
        FieldSpec("solver_settings.restart", "Initialize from previous solution",
                  "bool", False,
                  tooltip="Warm-start the solve from restart.dat in the case "
                          "folder (written by the previous solve). The "
                          "solution vtk you see in the Solution tab is "
                          "written from the same state."),
    ]),
    Section("setup", "Numerics", [
        FieldSpec("solver_settings.numerics.limiter", "Slope limiter",
                  "choice", "VENKATAKRISHNAN", choices=LIMITERS,
                  tooltip="MUSCL slope limiter. Venkatakrishnan is the "
                          "robust default for turbine cascades."),
        FieldSpec("solver_settings.numerics.gradient", "Gradient method",
                  "choice", "WEIGHTED_LEAST_SQUARES", choices=GRADIENTS),
        FieldSpec("solver_settings.numerics.cfl", "CFL number", "float",
                  10.0, 0.1, 100.0, slider=True,
                  tooltip="Implicit time step. With CFL adaptation the "
                          "solver ramps it up automatically."),
        FieldSpec("solver_settings.numerics.cfl_adapt", "CFL adaptation parameters",
                  "float_array", [0.1, 1.2, 10.0, 40.0], array_fixed=True,
                  tooltip="SU2 CFL_ADAPT_PARAM: (reduce_factor, increase_factor, "
                          "lower_bound, upper_bound)."),
        FieldSpec("solver_settings.numerics.linear_solver_iter", "Linear solver iterations",
                  "int", 200, 5, 500, step=10,
                  tooltip="FGMRES+ILU inner iterations per implicit step."),
    ]),

    # --------------------------------------------------------------- geometry
    Section("geometry", "Airfoil source", [
        FieldSpec("airfoil_source.type", "Geometry source", "choice", "pyturbo",
                  choices=[_c("pyturbo-aero", "pyturbo"),
                           _c("geomTurbo import", "geomturbo")],
                  tooltip="pyturbo-aero generates the blade from the "
                          "parameters below; geomTurbo imports a section "
                          "from a NUMECA .geomTurbo file and uses it for "
                          "meshing as-is (normalized to axial chord = 1)."),
        FieldSpec("airfoil_source.geomturbo_file", "geomTurbo file", "file", "",
                  tooltip="Path to a NUMECA .geomTurbo file. Sections found "
                          "in the blade block become available in the "
                          "Section dropdown."),
        FieldSpec("airfoil_source.section", "Section", "choice", 0,
                  choices=[_c("(load a geomTurbo file)", 0)],
                  tooltip="Blade section to use, in file order. The label "
                          "shows the section's Z value."),
        FieldSpec("airfoil_source.show_reference", "Show geomTurbo reference",
                  "bool", True,
                  tooltip="While pyturbo-aero is the source, overlay the "
                          "imported geomTurbo section in gray so the "
                          "generated blade can be matched to it manually."),
    ]),
    Section("geometry", "geomTurbo morph (FFD cage)", [
        FieldSpec("airfoil_source.morph.enabled", "Enable FFD morphing",
                  "bool", False,
                  tooltip="Apply the morph cage to the imported section "
                          "before meshing (geomTurbo source only). The "
                          "blade preview updates live; the deformation is "
                          "stored in input.json so the mesh and any "
                          "optimization runs use the same shape."),
        FieldSpec("airfoil_source.morph.n", "Morph points per direction",
                  "int", 4, 2, 8, step=1,
                  tooltip="N_morph: the cage is an N_morph x N_morph grid "
                          "of control points around the section "
                          "(N_morph^2 points, each with a dx and dy "
                          "offset). The cage box is derived from the "
                          "section bounding box plus a small margin."),
        FieldSpec("airfoil_source.morph.dx", "Cage point dx offsets",
                  "float_array", [0.0] * 16, array_fixed=True,
                  tooltip="x offsets of the cage control points in axial-"
                          "chord units, row-major over the N_morph x "
                          "N_morph grid (edited in the morph panel)."),
        FieldSpec("airfoil_source.morph.dy", "Cage point dy offsets",
                  "float_array", [0.0] * 16, array_fixed=True,
                  tooltip="y offsets of the cage control points in axial-"
                          "chord units, row-major over the N_morph x "
                          "N_morph grid (edited in the morph panel)."),
    ]),
    Section("geometry", "Camberline", [
        FieldSpec("airfoil.alpha1", "Inlet metal angle alpha1", "float",
                  10.0, -70.0, 70.0, slider=True, unit="deg",
                  tooltip="Camberline angle at the leading edge, from +x."),
        FieldSpec("airfoil.alpha2", "Outlet metal angle alpha2", "float",
                  -60.0, -90.0, 40.0, slider=True, unit="deg",
                  tooltip="Camberline angle at the trailing edge, from +x. "
                          "alpha1 - alpha2 is the total flow turning."),
        FieldSpec("airfoil.stagger", "Stagger", "float", -45.0,
                  -90.0, 30.0, slider=True, unit="deg",
                  tooltip="Blade stagger angle of the pyturbo-aero camberline."),
        FieldSpec("airfoil.axial_chord", "Axial chord", "float", 100.0,
                  10.0, 500.0, slider=True, unit="mm",
                  tooltip="Values >= 1 are millimetres (mesh is normalised to "
                          "axial chord = 1, then scaled to metres for SU2); "
                          "values < 1 are metres."),
    ]),
    Section("geometry", "Thickness", [
        FieldSpec("airfoil.le_thickness", "Leading-edge thickness", "float",
                  0.04, 0.005, 0.2, slider=True,
                  tooltip="LE thickness scale (pyturbo-aero units)."),
        FieldSpec("airfoil.ss_thickness", "Suction-side thickness profile",
                  "float_array", [0.2, 0.3, 0.2, 0.1],
                  tooltip="Thickness shape array along the suction side "
                          "(leading -> trailing)."),
        FieldSpec("airfoil.ps_thickness", "Pressure-side thickness profile",
                  "float_array", [0.02, 0.02, 0.01],
                  tooltip="Thickness shape array along the pressure side."),
        FieldSpec("airfoil.camber_percent", "Camber percent", "float", 0.8,
                  0.1, 1.0, slider=True,
                  tooltip="Where the thickness peak sits, as a fraction of "
                          "surface length."),
        FieldSpec("airfoil.expansion_ratio", "Expansion ratio", "float", 1.2,
                  1.0, 3.0, slider=True,
                  tooltip="Thickness distribution skew towards the TE."),
    ]),
    Section("geometry", "Trailing edge", [
        FieldSpec("airfoil.te_radius", "TE radius", "float", 2.0,
                  0.2, 6.0, slider=True,
                  tooltip="Blunt trailing-edge radius (pyturbo-aero units)."),
        FieldSpec("airfoil.wedge_ss", "SS wedge angle", "float", 5.0,
                  0.0, 20.0, slider=True, unit="deg"),
        FieldSpec("airfoil.wedge_ps", "PS wedge angle", "float", 5.0,
                  0.0, 20.0, slider=True, unit="deg"),
    ]),
    Section("geometry", "Suction-side flow guidance", [
        FieldSpec("airfoil.ss_flow_guidance.s_c", "Straightener extent s_c",
                  "float", 0.8, 0.0, 1.0, slider=True,
                  tooltip="Fraction of the suction side held near the metal "
                          "angle downstream of the throat."),
        FieldSpec("airfoil.ss_flow_guidance.n", "Straightener exponent n",
                  "int", 10, 1, 50,
                  tooltip="Blending exponent of the flow-guidance law."),
    ]),
    Section("geometry", "Discretisation", [
        FieldSpec("airfoil.n_points", "Surface points per side", "int",
                  401, 60, 2000, step=20,
                  tooltip="Spline sampling points per surface; 300-600 is "
                          "plenty for RANS meshes."),
    ]),
    Section("geometry", "Domain (periodic passage)", [
        FieldSpec("domain.R1", "Annulus radius at LE (R1)", "float", 9.0,
                  0.01, 2000.0, slider=True, unit="mm",
                  tooltip="Actual annulus radius at the leading-edge axial "
                          "station, in the same units as axial chord (mm by "
                          "default). Sets the pitch there: p_LE = 2 pi R1 / "
                          "N. Must equal R2 for SU2 periodicity. "
                          "Auto-filled when a geomTurbo file is imported."),
        FieldSpec("domain.R2", "Annulus radius at TE (R2)", "float", 9.0,
                  0.01, 2000.0, slider=True, unit="mm",
                  tooltip="Actual annulus radius at the trailing-edge axial "
                          "station, in the same units as axial chord. Sets "
                          "the pitch there: p_TE = 2 pi R2 / N. Must equal "
                          "R1 for a single-translation periodic pair. "
                          "Auto-filled when a geomTurbo file is imported."),
        FieldSpec("domain.airfoil_count", "Blade count N", "int", 45,
                  2, 300, step=1,
                  tooltip="Blades around the full annulus; pitch = 2 pi R / N."),
        FieldSpec("domain.x_min", "Domain inlet extent", "float", -0.5,
                  -3.0, 0.0, slider=True, unit="x/c_ax",
                  tooltip="Upstream extent of the passage, in axial-chord "
                          "units relative to the LE."),
        FieldSpec("domain.x_max", "Domain outlet extent", "float", 2.5,
                  1.0, 8.0, slider=True, unit="x/c_ax",
                  tooltip="Downstream extent of the passage, in axial-chord "
                          "units relative to the LE."),
    ]),

    # ------------------------------------------------------------------- mesh
    Section("mesh", "Sizes (axial-chord units)", [
        FieldSpec("mesh.max_size", "Far-field element size", "float", 0.025,
                  0.002, 0.2, slider=True,
                  tooltip="Largest edge length in the free stream."),
        FieldSpec("mesh.near_wall_size", "Near-wall element size", "float",
                  0.006, 0.0005, 0.05, slider=True,
                  tooltip="Edge length just outside the boundary-layer "
                          "region."),
        FieldSpec("mesh.refine_dist", "Near-wall refinement distance", "float",
                  0.6, 0.05, 2.0, slider=True,
                  tooltip="Distance from the wall over which the mesh "
                          "refines towards near_wall_size."),
        FieldSpec("mesh.periodic_size", "Periodic-edge element size", "float",
                  0.025, 0.002, 0.1, slider=True,
                  tooltip="Edge length along the periodic passage edges."),
        FieldSpec("mesh.algorithm", "2D mesh algorithm", "choice", 6,
                  choices=MESH_ALGORITHMS,
                  tooltip="Gmsh 2D algorithm. Frontal-Delaunay (6) gives the "
                          "best boundary-layer quality."),
    ]),
    Section("mesh", "Wall discretisation", [
        FieldSpec("mesh.airfoil_points", "Nodes per surface side", "nullable_int",
                  None, 20, 2000, step=10,
                  tooltip="Fixes the number of mesh nodes along each airfoil "
                          "surface (SS and PS each). Auto lets Gmsh derive "
                          "the wall spacing from the size fields. This is "
                          "independent of airfoil.n_points, which only sets "
                          "the geometric spline resolution."),
    ]),
    Section("mesh", "Boundary layer", [
        FieldSpec("mesh.boundary_layer.first_layer_height",
                  "First-layer height", "float", 2.0e-4, 1e-6, 2e-3,
                  slider=True,
                  tooltip="Height of the first wall-normal layer. Pick it "
                          "for the y+ you want (GUI reports achieved y+ "
                          "after the run)."),
        FieldSpec("mesh.boundary_layer.growth_rate", "Growth rate", "float",
                  1.25, 1.02, 2.0, slider=True,
                  tooltip="Geometric growth of the wall-normal layers."),
        FieldSpec("mesh.boundary_layer.n_layers", "Layer count", "int", 14,
                  2, 60, step=1,
                  tooltip="Number of structured wall-normal layers."),
    ]),
    Section("mesh", "Wake refinement", [
        FieldSpec("mesh.wake.size", "Wake element size", "float", 0.02,
                  0.002, 0.1, slider=True,
                  tooltip="Edge length inside the trailing-edge wake band."),
        FieldSpec("mesh.wake.half_width", "Wake half-width", "float", 0.15,
                  0.01, 0.6, slider=True,
                  tooltip="Half-width of the wake band behind the TE."),
        FieldSpec("mesh.wake.transition", "Wake transition distance", "float",
                  0.4, 0.02, 2.0, slider=True,
                  tooltip="Distance over which wake sizing blends back to "
                          "the far-field size."),
        FieldSpec("mesh.wake.metal_angle_deg", "Wake direction", "nullable_float",
                  None, -90.0, 40.0, unit="deg",
                  tooltip="Orientation of the wake band. Auto uses the TE "
                          "metal angle; set a value to override."),
    ]),
    Section("mesh", "Plot view window", [
        FieldSpec("plot.zoom_window", "Zoom window (xmin, xmax, ymin, ymax)",
                  "float_array", [-0.2, 0.45, -0.85, 0.75], array_fixed=True,
                  tooltip="Window used for the zoomed mesh/near-wall PNGs "
                          "(axial-chord units)."),
    ]),
]

FIELDS = [f for s in SECTIONS for f in s.fields]
FIELDS_BY_PATH = {f.path: f for f in FIELDS}
_path_tab = {f.path: tab for tab in
             ("setup", "geometry", "mesh")
             for f in [x for s in SECTIONS if s.tab == tab for x in s.fields]}

# nested defaults filled from the specs
DEFAULTS = {}
for _f in FIELDS:
    _node = DEFAULTS
    _parts = _f.path.split(".")
    for _p in _parts[:-1]:
        _node = _node.setdefault(_p, {})
    _node[_parts[-1]] = _f.default


def sections_for(tab):
    return [s for s in SECTIONS if s.tab == tab]


def field_from_dict(data, prefix=""):
    """Build a FieldSpec from a plugin-manifest parameter dict (see
    pipeline/plugins.py). Unknown kinds fall back to float; 'choices'
    entries are [label, value] pairs."""
    path = str(data.get("path") or "").strip()
    kind = data.get("kind") or "float"
    if kind not in ("float", "int", "str", "bool", "choice", "float_array",
                    "nullable_float", "nullable_int", "file"):
        kind = "float"
    choices = None
    if kind == "choice":
        choices = [(str(c[0]), c[1]) for c in (data.get("choices") or [])
                   if isinstance(c, (list, tuple)) and len(c) >= 2]
    return FieldSpec(path=prefix + path,
                     label=str(data.get("label") or path or "parameter"),
                     kind=kind, default=data.get("default"),
                     min=data.get("min"), max=data.get("max"),
                     step=data.get("step"), choices=choices,
                     slider=bool(data.get("slider")),
                     unit=str(data.get("unit") or ""),
                     tooltip=str(data.get("tooltip") or ""),
                     array_fixed=bool(data.get("array_fixed")))
