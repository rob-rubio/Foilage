"""Optimization engine: NSGA-II and adaptive hybrid CFD searches.

Each individual is one full pipeline run (geometry + mesh, SU2 case setup,
solve, post-processing). Objective values are read from the case
results.json written by tools/plot_case.py. Every evaluation keeps its own
case folder under cases/ and is appended to evaluations.csv in the run
folder, so a run is fully recorded and auditable.

The genetic operators (NSGA-II non-dominated sorting, crowding distance,
SBX crossover, polynomial mutation) live here so no extra dependency is
needed. The adaptive hybrid mode combines global differential-evolution
proposals, elite local refinement, and genetic proposals, adapting the mix
from the observed progress. All objectives are handled internally as minimization;
the signed vector stored in a record's "F" is the raw value multiplied by -1
for maximization.

A run can be paused between evaluations and resumed. Whenever it pauses,
finishes, or is asked to save, its complete state is written to
optimizer_state.json in the run folder: configuration, objectives,
constraints, design variables, the full evaluation history, the current
population, the pending proposals of a partially evaluated generation, the
convergence counters, and the random-number generator state.
OptimizationRun.from_state() rebuilds a run from such a file so the search
continues exactly where it left off (in a new run folder).
"""

import copy
import csv
import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from .runner import _popen_flags
from .schema import FIELDS_BY_PATH
from .state import get_path, set_path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

PENALTY = 1e12             # objective vector for failed / unconverged evals
ALGORITHMS = ("NSGA-II", "Adaptive Hybrid Search")

# --------------------------------------------------------------- catalogs
# Postprocessed quantities offered as objectives (dotted paths into
# results.json). "sense" is only the radio-button default.
OBJECTIVES = [
    {"path": "losses.total_pressure_loss_coeff_Yp",
     "label": "Total-pressure loss Yp", "sense": "min"},
    {"path": "back_surface_diffusion.ss.DF",
     "label": "SS back-surface diffusion DF", "sense": "min"},
    {"path": "back_surface_diffusion.ps.DF",
     "label": "PS back-surface diffusion DF", "sense": "min"},
    {"path": "outlet.flow_angle_deg",
     "label": "Outlet flow angle [deg]", "sense": "max"},
    {"path": "outlet.mass_flow_kg_s_m",
     "label": "Outlet mass flow [kg/(s.m)]", "sense": "max"},
    {"path": "inlet.corrected_flow_kg_s_m",
     "label": "Corrected flow, inlet [kg/(s.m)]", "sense": "max"},
    {"path": "outlet.corrected_flow_kg_s_m",
     "label": "Corrected flow, outlet [kg/(s.m)]", "sense": "max"},
    {"path": "fields.max_mach", "label": "Max Mach number", "sense": "min"},
    {"path": "wall.yplus_median", "label": "Wall y+ (median)", "sense": "min"},
    {"path": "convergence.iterations_run",
     "label": "Iterations to converge", "sense": "min"},
    {"path": "mass_balance.imbalance_pct",
     "label": "Mass-flow imbalance [%]", "sense": "min"},
    {"path": "geometry.pitch_to_chord",
     "label": "Pitch-to-chord (PTC)", "sense": "min"},
]

# External-airfoil coefficients are meaningful optimization quantities for a
# freestream case.  Keep them out of the cascade catalog: cascade runs use
# passage quantities such as loss, turning, and corrected flow instead.
FREESTREAM_OBJECTIVES = [
    {"path": "forces.LD", "label": "Lift-to-drag ratio L/D (CL/CD)",
     "sense": "max"},
]

# Quantities that can be *constrained* (bound with >= or <=): everything
# above plus the 0D geometry metrics from the mesh pipeline. Corrected
# flow is W*sqrt(theta)/delta with theta = T0/288.15 K, delta =
# p0/101325 Pa (NASA Glenn reference conditions).
CONSTRAINT_QUANTITIES = OBJECTIVES + [
    {"path": "geometry.throat_width_cax",
     "label": "Throat width [c_ax]", "sense": None},
    {"path": "geometry.throat_x_over_cax",
     "label": "Throat location x/c_ax", "sense": None},
    {"path": "geometry.unguided_turning_deg",
     "label": "Unguided turning [deg]", "sense": None},
    {"path": "geometry.pitch_le_cax",
     "label": "Pitch LE [c_ax]", "sense": None},
]

# Zweifel loading coefficients (results.json "zweifel" block, written by
# the post-processing plane audit for periodic cascades only - a
# freestream run has no pitch, so no Zweifel). The classic incompressible
# criterion and its compressible form (inlet tangential momentum
# re-framed on the outlet axial velocity, Vz1/Vz2 = rho2/rho1; the
# compressible value stays at or below the incompressible one for an
# accelerating row); the traditional design band is Zw ~ 0.8, so as
# objectives they default to maximize, while the constraint rows let a
# target band (e.g. 0.7-0.9) be held.
ZWEIFEL_QUANTITIES = [
    {"path": "zweifel.incompressible",
     "label": "Zweifel incompressible", "sense": "max"},
    {"path": "zweifel.compressible",
     "label": "Zweifel compressible", "sense": "max"},
]

# the geometry-based predictor from the mesh pipeline (metal angles +
# throat pitch over axial chord) - bandable as a CFD-free constraint
ZWEIFEL_GEOMETRIC_CONSTRAINT = [
    {"path": "geometry.zweifel_geometric",
     "label": "Zweifel (geometric predictor)", "sense": None},
]

FREESTREAM_CONSTRAINT_QUANTITIES = FREESTREAM_OBJECTIVES


# Real wedge-sector flow objectives: swapped in for the per-unit-depth
# entries when the case runs in 3D wedge mode (results.json then reports
# actual kg/s through the sector, not kg/(s.m)).
WEDGE_FLOW_OBJECTIVES = [
    {"path": "outlet.mass_flow_kg_s",
     "label": "Outlet mass flow [kg/s (wedge)]", "sense": "max"},
    {"path": "inlet.corrected_flow_kg_s",
     "label": "Corrected flow, inlet [kg/s (wedge)]", "sense": "max"},
    {"path": "outlet.corrected_flow_kg_s",
     "label": "Corrected flow, outlet [kg/s (wedge)]", "sense": "max"},
]

_PER_DEPTH_TO_WEDGE = {
    "outlet.mass_flow_kg_s_m": WEDGE_FLOW_OBJECTIVES[0],
    "inlet.corrected_flow_kg_s_m": WEDGE_FLOW_OBJECTIVES[1],
    "outlet.corrected_flow_kg_s_m": WEDGE_FLOW_OBJECTIVES[2],
}


def objectives_for_case(freestream=False, wedge=False):
    """Return the output-objective catalog for the selected flow mode.

    The Zweifel loading coefficients are offered for periodic cascades
    only - freestream runs have no pitch, so no Zweifel is written to
    results.json. ``wedge`` swaps the per-unit-depth flow objectives for
    the real wedge-sector ones."""
    base = ([_PER_DEPTH_TO_WEDGE.get(o["path"], o) for o in OBJECTIVES]
            if wedge else list(OBJECTIVES))
    extra = []
    if not freestream:
        extra += ZWEIFEL_QUANTITIES
    return base + extra + (list(FREESTREAM_OBJECTIVES)
                           if freestream else [])


def constraint_quantities_for_case(freestream=False, wedge=False):
    """Return the output-constraint catalog for the selected flow mode.

    Periodic cascades can additionally band the Zweifel loading - the
    CFD values or the geometric predictor from the mesh pipeline.
    ``wedge`` swaps the per-unit-depth flow quantities for the real
    wedge-sector ones."""
    base = CONSTRAINT_QUANTITIES
    if wedge:
        base = [_PER_DEPTH_TO_WEDGE.get(o["path"], o) for o in base]
    extra = []
    if not freestream:
        extra += ZWEIFEL_QUANTITIES + ZWEIFEL_GEOMETRIC_CONSTRAINT
    return list(base) + extra + (
        list(FREESTREAM_CONSTRAINT_QUANTITIES) if freestream else [])

# pyturbo generator parameters offered as design variables; suggested
# bounds come from the input.json schema, the "avg" default is filled by
# the tab from the current case. design_variables_for() builds the
# source-dependent catalog (pyturbo / geomTurbo+FFD cage / plugins).
_DESIGN_PATHS = [
    "airfoil.alpha1", "airfoil.alpha2", "airfoil.stagger",
    "airfoil.axial_chord", "airfoil.le_thickness", "airfoil.camber_percent",
    "airfoil.expansion_ratio", "airfoil.te_radius", "airfoil.wedge_ss",
    "airfoil.wedge_ps", "airfoil.ss_flow_guidance.s_c",
    "airfoil.ss_flow_guidance.n", "domain.R1", "domain.airfoil_count",
]

DESIGN_VARS = [{"path": f.path,
                "label": f.label,
                "kind": "int" if f.kind == "int" else "float",
                "min": f.min, "max": f.max}
               for f in (FIELDS_BY_PATH[p] for p in _DESIGN_PATHS)]

# design variables that act on the passage rather than the blade
# generator - offered for every airfoil source
_DOMAIN_DV_PATHS = ("domain.R1", "domain.airfoil_count")

# extra design variables for the 3D wedge mode: the TE radius is free
# (rotational periodics support R1 != R2) and the streamtube contraction
# becomes optimizable
_WEDGE_DV_PATHS = ("domain.R2", "domain.h1", "domain.h2")

CAGE_DV_DEFAULT_BOUND = 0.1     # +/- offset bound prefilled for cage DVs


def ffd_design_vars(n, bound=CAGE_DV_DEFAULT_BOUND):
    """Design variables for the FFD morph cage: one per dx/dy offset of
    the n x n control lattice. The list element is addressed with '@'
    (airfoil_source.morph.dx@5), applied by set_design_value(); bounds
    are symmetric +/- bound in axial-chord units."""
    dvs = []
    for key in ("dx", "dy"):
        for idx in range(n * n):
            i, j = idx // n, idx % n
            dvs.append({"path": f"airfoil_source.morph.{key}@{idx}",
                        "label": f"FFD {key} (u{i},v{j})",
                        "kind": "float", "min": -bound, "max": bound,
                        "cage": True})
    return dvs


def _field_dvs(paths):
    return [{"path": p, "label": FIELDS_BY_PATH[p].label,
             "kind": "int" if FIELDS_BY_PATH[p].kind == "int" else "float",
             "min": FIELDS_BY_PATH[p].min, "max": FIELDS_BY_PATH[p].max}
            for p in paths]


def design_variables_for(source_type, morph_n=0, extensions_dir=None,
                         wedge=False):
    """Design-variable catalog for the active airfoil source.

    pyturbo: the generator parameters (DESIGN_VARS). A geomTurbo import
    or an extension plugin additionally gets the plugin's declared
    parameters (with the manifest's min/max), the domain variables, and
    the FFD morph-cage offsets when a morph lattice is defined. The 3D
    wedge mode (``wedge``) exposes R2 and the streamtube depths h1/h2 on
    top - its rotational periodics support R1 != R2."""
    wedge_dvs = _field_dvs(_WEDGE_DV_PATHS) if wedge else []
    if source_type == "pyturbo":
        return [dict(d) for d in DESIGN_VARS] + wedge_dvs
    dvs = []
    if source_type != "geomturbo":
        try:
            from pipeline.plugins import get_plugin
        except ImportError:
            from plugins import get_plugin
        plugin = get_plugin(source_type, extensions_dir)
        if plugin and not plugin["error"]:
            for spec in plugin["parameters"]:
                if spec.get("kind") not in ("float", "int"):
                    continue
                try:
                    lo = float(spec.get("min"))
                    hi = float(spec.get("max"))
                except (TypeError, ValueError):
                    continue
                if not lo < hi:
                    continue
                dvs.append({"path": f"airfoil_source.params.{spec['path']}",
                            "label": str(spec.get("label") or spec["path"]),
                            "kind": spec["kind"], "min": lo, "max": hi,
                            "default": spec.get("default")})
    for path in _DOMAIN_DV_PATHS:
        f = FIELDS_BY_PATH[path]
        dvs.append({"path": path, "label": f.label,
                    "kind": "int" if f.kind == "int" else "float",
                    "min": f.min, "max": f.max})
    dvs += wedge_dvs
    if morph_n and morph_n >= 2:
        dvs += ffd_design_vars(int(morph_n))
    return dvs


def set_design_value(cfg, dotted, value):
    """Apply one design variable to a config dict. A '@' in the path
    addresses a list element (airfoil_source.morph.dx@5), growing the
    list with zeros when needed."""
    if "@" not in dotted:
        set_path(cfg, dotted, value)
        return
    base, idx = dotted.rsplit("@", 1)
    lst = get_path(cfg, base)
    if not isinstance(lst, list):
        lst = []
        set_path(cfg, base, lst)
    idx = int(idx)
    while len(lst) <= idx:
        lst.append(0.0)
    lst[idx] = value


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "case"


# ------------------------------------------------------- NSGA-II operators
def dominates(f1, f2):
    """Minimization dominance."""
    return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in
                                                       zip(f1, f2))


def constrained_dominates(f1, cv1, f2, cv2):
    """Deb's constrained domination: a feasible point always beats an
    infeasible one; two infeasible points compare by total constraint
    violation; two feasible points compare by the objectives."""
    if cv1 <= 0.0 and cv2 <= 0.0:
        return dominates(f1, f2)
    if cv1 <= 0.0:
        return True
    if cv2 <= 0.0:
        return False
    return cv1 < cv2


def constraint_violation(raw, constraints):
    """Total normalized violation of output-based constraints.

    constraints: [{path, op (">=" | "<="), value}]. A quantity missing
    from results.json counts as strongly violated. Violations are
    normalized by the bound magnitude so differently scaled constraints
    sum comparably."""
    total = 0.0
    for c in constraints:
        v = raw.get(c["path"])
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            total += 1.0                      # missing -> violated
            continue
        bound = float(c["value"])
        scale = max(abs(bound), 1e-6)
        if c["op"] == ">=":
            total += max(0.0, (bound - v) / scale)
        else:
            total += max(0.0, (v - bound) / scale)
    return total


def nondominated_fronts(F, CV=None):
    """NSGA-II fast-non-dominated-sort. Returns a list of fronts, each a
    list of indices into F. CV (constraint violations, 0 = feasible)
    switches the comparison to Deb's constrained domination."""
    n = len(F)
    dominated_by = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(i + 1, n):
            if CV is None:
                i_dom_j = dominates(F[i], F[j])
                j_dom_i = dominates(F[j], F[i])
            else:
                i_dom_j = constrained_dominates(F[i], CV[i], F[j], CV[j])
                j_dom_i = constrained_dominates(F[j], CV[j], F[i], CV[i])
            if i_dom_j:
                dominated_by[i].append(j)
                dom_count[j] += 1
            elif j_dom_i:
                dominated_by[j].append(i)
                dom_count[i] += 1
    for i in range(n):
        if dom_count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(F, front):
    """NSGA-II crowding distance for one front -> {index: distance}."""
    dist = {i: 0.0 for i in front}
    n_obj = len(F[front[0]]) if front else 0
    for m in range(n_obj):
        order = sorted(front, key=lambda i: F[i][m])
        fmin, fmax = F[order[0]][m], F[order[-1]][m]
        dist[order[0]] = dist[order[-1]] = math.inf
        if fmax - fmin < 1e-30:
            continue
        scale = fmax - fmin
        for k in range(1, len(order) - 1):
            i = order[k]
            if math.isinf(dist[i]):
                continue
            dist[i] += (F[order[k + 1]][m] - F[order[k - 1]][m]) / scale
    return dist


def front_ids(F, CV=None):
    """Indices of the single non-dominated front of a population."""
    return nondominated_fronts(F, CV)[0] if F else []


HV_MC_SAMPLES = 20000        # Monte-Carlo samples for 3+ objective volume
HV_TOL = 0.005               # relative gain below this counts as "stalled"

STATE_VERSION = 1
STATE_NAME = "optimizer_state.json"


def hypervolume(F, rng_seed=1234):
    """Hypervolume of a minimization point set w.r.t. a reference point
    built from the worst values (5% margin).

    1 objective: interval length. 2: exact sweep over the sorted front.
    3+: Monte-Carlo estimate with a fixed seed so consecutive generations
    are comparable. Used as the convergence indicator: when the front
    stops growing measurably, the run has stalled.
    """
    pts = [p for p in F if p and all(math.isfinite(v) for v in p)]
    if not pts:
        return 0.0
    n_obj = len(pts[0])
    lo = [min(p[d] for p in pts) for d in range(n_obj)]
    ref = [lo[d] + 1.05 * (max(p[d] for p in pts) - lo[d]) + 1e-12
           for d in range(n_obj)]
    if n_obj == 1:
        return ref[0] - lo[0]
    nd = [p for i, p in enumerate(pts)
          if not any(dominates(q, p) for j, q in enumerate(pts) if j != i)]
    if n_obj == 2:
        nd = sorted(nd)             # x ascending -> y ascending on a front
        total = 0.0
        for k in range(len(nd)):
            nxt = nd[k + 1][0] if k + 1 < len(nd) else ref[0]
            total += (nxt - nd[k][0]) * (ref[1] - nd[k][1])
        return total
    rng = random.Random(rng_seed)
    span = [ref[d] - lo[d] for d in range(n_obj)]
    vol = 1.0
    for s in span:
        vol *= s
    dominated = 0
    for _ in range(HV_MC_SAMPLES):
        q = [lo[d] + rng.random() * span[d] for d in range(n_obj)]
        if any(all(p[d] <= q[d] for d in range(n_obj)) for p in nd):
            dominated += 1
    return vol * dominated / HV_MC_SAMPLES


def lhs_sample(bounds, n, rng):
    """Latin-hypercube sample of n vectors inside (lo, hi) bounds."""
    dims = len(bounds)
    out = []
    for d in range(dims):
        lo, hi = bounds[d]
        cuts = [(k + rng.random()) / n for k in range(n)]
        rng.shuffle(cuts)
        for i, t in enumerate(cuts):
            if d == 0:
                out.append([])
            out[i].append(lo + t * (hi - lo))
    return out


def _clip(v, lo, hi):
    return min(max(v, lo), hi)


def sbx_crossover(x1, x2, bounds, eta, rng, pc=0.9):
    """Simulated-binary crossover; returns two children."""
    c1, c2 = list(x1), list(x2)
    if rng.random() > pc:
        return c1, c2
    for k, (lo, hi) in enumerate(bounds):
        if rng.random() > 0.5 or hi - lo < 1e-15:
            continue
        u = rng.random()
        beta = (2.0 * u) ** (1.0 / (eta + 1.0)) if u <= 0.5 \
            else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1[k] = 0.5 * ((1 + beta) * x1[k] + (1 - beta) * x2[k])
        c2[k] = 0.5 * ((1 - beta) * x1[k] + (1 + beta) * x2[k])
        c1[k] = _clip(c1[k], lo, hi)
        c2[k] = _clip(c2[k], lo, hi)
    return c1, c2


def polynomial_mutation(x, bounds, eta, rng, pm=None):
    """Polynomial mutation (penalty-free boundary handling)."""
    n = len(x)
    if pm is None:
        pm = 1.0 / max(n, 1)
    out = list(x)
    for k, (lo, hi) in enumerate(bounds):
        if rng.random() > pm or hi - lo < 1e-15:
            continue
        u = rng.random()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
            out[k] = x[k] + delta * (x[k] - lo)
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
            out[k] = x[k] + delta * (hi - x[k])
        out[k] = _clip(out[k], lo, hi)
    return out


def apply_kind(values, kinds):
    """Round integer design variables after sampling / breeding."""
    return [int(round(v)) if k == "int" else float(v)
            for v, k in zip(values, kinds)]


# ------------------------------------------------------------ saved state
def validate_state(state):
    """Check a dict loaded from optimizer_state.json.

    Raises ValueError with a user-facing message when the file cannot be
    resumed; returns the state unchanged otherwise."""
    if not isinstance(state, dict):
        raise ValueError("not an optimizer state file (expected a JSON "
                         "object)")
    if state.get("version") != STATE_VERSION:
        raise ValueError(f"unsupported state file version "
                         f"{state.get('version')!r} (expected "
                         f"{STATE_VERSION})")
    if state.get("algorithm") not in ALGORITHMS:
        raise ValueError(f"unknown algorithm in state file: "
                         f"{state.get('algorithm')!r}")
    if not isinstance(state.get("base_config"), dict):
        raise ValueError("state file has no base case configuration")
    objectives = state.get("objectives")
    design_vars = state.get("design_vars")
    records = state.get("records")
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("state file lists no objectives")
    if not isinstance(design_vars, list) or not design_vars:
        raise ValueError("state file lists no design variables")
    if not isinstance(records, list):
        raise ValueError("state file has no evaluation history")
    opts = state.get("options") or {}
    try:
        pop_size = int(opts.get("pop_size"))
    except (TypeError, ValueError):
        raise ValueError("state file has no valid population size")
    if pop_size < 4:
        raise ValueError("population size in state file must be at least 4")
    try:
        if int(state.get("generation") or 0) < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("state file has an invalid generation counter")
    n_dv, n_obj = len(design_vars), len(objectives)
    for r in list(records) + list(state.get("population") or []):
        if len(r.get("values") or []) != n_dv:
            raise ValueError("saved history does not match the design "
                             "variables")
        if len(r.get("F") or []) != n_obj:
            raise ValueError("saved history does not match the objectives")
    pending = state.get("pending") or []
    for v in pending:
        if len(v) != n_dv:
            raise ValueError("saved pending proposals do not match the "
                             "design variables")
    idx = int(state.get("pending_idx") or 0)
    if not 0 <= idx <= len(pending):
        raise ValueError("saved pending-proposal index out of range")
    return state


# ------------------------------------------------------------- run worker
class OptimizationRun:
    """One optimization run: a worker thread evaluating generations.

    Events onto self.queue (drained by the GUI):
        ("started", {run_dir, n_obj, n_dv, pop_size, single})
        ("log", {line})
        ("eval_start", {gen, idx, case})
        ("eval_done", {record})       record dict, see _make_record
        ("generation_done", {gen, best_text, front_ids})
        ("paused", {})                holding between evaluations
        ("resumed", {})
        ("state_saved", {path, count})
        ("done", {reason, converged})
    """

    def __init__(self, base_config, run_dir, run_tag, objectives, design_vars,
                 pop_size=8, patience=10, max_generations=0,
                 require_converged=True, threads=6, su2_exe=None, seed=None,
                 constraints=None, max_iterations=None, stage_timeout=None,
                 algorithm="NSGA-II"):
        self.base_config = copy.deepcopy(base_config)
        self.run_dir = Path(run_dir)
        self.run_tag = run_tag
        self.objectives = objectives          # [{path, sense, label}]
        self.design_vars = design_vars        # [{path, label, min, avg, max, kind}]
        self.constraints = constraints or []  # [{path, op, value, label}]
        self.max_iterations = max_iterations  # None = keep the case value
        self.stage_timeout = stage_timeout    # s per stage; None = no limit
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown optimization algorithm: {algorithm}")
        self.algorithm = algorithm
        self.pop_size = int(pop_size)
        self.patience = int(patience)
        self.max_generations = int(max_generations)
        self.require_converged = bool(require_converged)
        self.threads = int(threads)
        self.su2_exe = su2_exe
        self.seed = seed

        self.queue = queue.Queue()
        self.stop_requested = False
        self.thread = None
        self.proc = None
        self._stage_timed_out = False
        self.records = []            # every evaluated record, in eval order
        self.population = []         # current parent-population records
        self._next_id = 0
        self._stale_gens = 0
        self._last_hv = None
        self._csv_file = None
        self._csv = None
        # pause/resume: the worker holds between evaluations while paused
        self.paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        # resumable search state (also what optimizer_state.json stores)
        self._generation = 0         # generation currently being evaluated
        self._pending = []           # proposals left in the current gen
        self._pending_idx = 0        # next index into _pending
        self._finish_reason = ("stopped", False)
        self._save_requested = False
        self._rng = random.Random(seed)
        self._rng_state = None       # serialized rng state (saved file)
        self._resumed = False
        self.state_path = self.run_dir / STATE_NAME

        self.kinds = [d["kind"] for d in design_vars]
        self.bounds = [(float(d["min"]), float(d["max"]))
                       for d in design_vars]
        self.avg_vector = [float(d.get("avg", d["min"])) for d in design_vars]
        self.signs = [-1.0 if o["sense"] == "max" else 1.0
                      for o in objectives]
        self._adaptive_scale = 0.18
        self._adaptive_global_weight = 0.40

    # ------------------------------------------------------------ control
    def start(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"opt-{self.run_tag}")
        self.thread.start()

    def pause(self):
        """Hold the run: the evaluation in flight finishes, then the
        worker waits between evaluations. Also snapshots the state."""
        if not self.paused:
            self.paused = True
            self._pause_event.clear()

    def resume(self):
        """Continue a paused run."""
        self.paused = False
        self._pause_event.set()

    def save_state(self):
        """Snapshot parameters and search history to optimizer_state.json.

        While the worker is alive it writes the snapshot itself at the
        next evaluation boundary, which keeps the rng and history exactly
        consistent; otherwise the caller's thread writes it directly."""
        if self.running:
            self._save_requested = True
        else:
            self._write_state()

    def stop(self):
        self.stop_requested = True
        self.paused = False
        self._pause_event.set()      # wake the worker if it is paused
        self._kill_tree()

    def _kill_tree(self):
        """Kill the running stage process and its children (gmsh/SU2 can
        retry-loop for a very long time on a degenerate geometry)."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                proc.terminate()
            except OSError:
                pass

    def _on_stage_timeout(self):
        """Timer callback: the stage exceeded its allowance."""
        self._stage_timed_out = True
        self._kill_tree()

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def _emit(self, kind, **data):
        self.queue.put((kind, data))

    def _log(self, line):
        self._emit("log", line=str(line))

    # ------------------------------------------------------------ main loop
    def _wait_if_paused(self):
        """Block between evaluations while paused. Returns False when the
        run should stop instead of continuing. Pausing auto-saves the
        state so the pause survives an application crash."""
        if not self.paused:
            return True
        self._emit("paused")
        self._save_requested = False
        self._write_state()
        while self.paused and not self.stop_requested:
            if self._save_requested:
                self._save_requested = False
                self._write_state()
            self._pause_event.wait(0.2)
        if not self.stop_requested:
            self._emit("resumed")
            return True
        return False

    def _checkpoint_if_requested(self):
        """Honor a save_state() request at an evaluation boundary."""
        if self._save_requested:
            self._save_requested = False
            self._write_state()

    def _run(self):
        try:
            rng = self._rng
            if self._rng_state is not None:
                rng.setstate(self._rng_state)
                self._rng_state = None
            self._open_outputs()
            self._emit("started", run_dir=str(self.run_dir),
                       n_obj=len(self.objectives),
                       n_dv=len(self.design_vars),
                       pop_size=self.pop_size,
                       single=len(self.objectives) == 1,
                       algorithm=self.algorithm)
            self._log(f"run folder: {self.run_dir}")
            self._log(f"algorithm: {self.algorithm}")
            self._log(f"{len(self.design_vars)} design variables, "
                      f"{len(self.objectives)} objective(s), "
                      f"{len(self.constraints)} constraint(s), population "
                      f"{self.pop_size}")
            if self._resumed:
                self._log(f"resumed from {STATE_NAME}: "
                          f"{len(self.records)} evaluation(s) carried over, "
                          f"continuing at generation {self._generation}")
            if self.max_generations:
                self._log(f"generation cap: {self.max_generations} "
                          "(absolute, shared with any resumed history)")
            else:
                self._log("no generation cap - stops on convergence "
                          f"({self.patience} generations without front "
                          "improvement), Pause/Stop, or Save state")

            while not self.stop_requested:
                if not self._wait_if_paused():
                    break
                gen = self._generation
                if not self._pending:
                    # generation 0 of a fresh run evaluates the initial
                    # population; afterwards every generation is a bred /
                    # proposed batch
                    if not self.records:
                        batch = self._initial_population(rng)
                    elif self.algorithm == "Adaptive Hybrid Search":
                        batch = self._adaptive_hybrid_proposals(rng, gen)
                    else:
                        batch = self._breed(rng)
                    self._pending = [list(v) for v in batch]
                    self._pending_idx = 0
                self._evaluate_generation(gen)
                if self._check_generation(gen):
                    break
                self._generation = gen + 1
            if self.stop_requested:
                self._finish_reason = ("stopped by user", False)
            reason, converged = self._finish_reason
            self._emit("done", reason=reason, converged=converged)
        except Exception as e:                      # pragma: no cover
            self._log(f"[error] internal error: {e}")
            self._emit("done", reason=f"internal error: {e}", converged=False)
        finally:
            self._write_state()
            self._close_csv()

    def _initial_population(self, rng):
        pop = [list(self.avg_vector)]               # individual 0 = nominal
        samples = lhs_sample(self.bounds, self.pop_size - 1, rng)
        pop.extend(samples)
        return [apply_kind(v, self.kinds) for v in pop]

    def _evaluate_generation(self, gen):
        """Evaluate the pending proposals of generation `gen`, one at a
        time, honoring Pause/Save between evaluations. Stopping or pausing
        mid-generation keeps the remaining proposals in self._pending, so
        a resumed run continues with exactly the same batch."""
        while self._pending_idx < len(self._pending):
            if self.stop_requested:
                return
            self._checkpoint_if_requested()
            if not self._wait_if_paused():
                return
            idx = self._pending_idx
            record = self._evaluate(gen, idx, self._pending[idx])
            self._pending_idx = idx + 1
            self.records.append(record)
            self.population.append(record)
            self._emit("eval_done", record=record)
        self._pending, self._pending_idx = [], 0
        # environmental selection keeps the parent population at pop_size
        combined = self.population
        if len(combined) > self.pop_size:
            self.population = self._select(combined, self.pop_size)
        front = self._current_front_ids()
        best_text = self._best_text()
        self._emit("generation_done", gen=gen, best_text=best_text,
                   front_ids=front)

    def _select(self, records, n):
        """NSGA-II environmental selection on signed objective vectors."""
        F = [r["F"] for r in records]
        CV = [r["cv"] for r in records]
        chosen = []
        for front in nondominated_fronts(F, CV):
            if len(chosen) + len(front) <= n:
                chosen.extend(front)
                continue
            cd = crowding_distance(F, front)
            front = sorted(front, key=lambda i: cd[i], reverse=True)
            chosen.extend(front[:n - len(chosen)])
            break
        return [records[i] for i in chosen]

    def _breed(self, rng):
        """One NSGA-II mating step: tournament select, SBX, mutate."""
        F = [r["F"] for r in self.population]
        CV = [r["cv"] for r in self.population]
        fronts = nondominated_fronts(F, CV)
        rank = {}
        for r, front in enumerate(fronts):
            for i in front:
                rank[i] = r
        cd = {}
        for front in fronts:
            cd.update(crowding_distance(F, front))

        def better(i, j):
            ri, rj = rank.get(i, 10 ** 9), rank.get(j, 10 ** 9)
            if ri != rj:
                return i if ri < rj else j
            ci, cj = cd.get(i, 0.0), cd.get(j, 0.0)
            return i if ci >= cj else j

        def tournament():
            i, j = rng.randrange(len(self.population)), \
                rng.randrange(len(self.population))
            return better(i, j)

        children = []
        eta_c, eta_m = 15.0, 20.0
        while len(children) < self.pop_size:
            p1, p2 = tournament(), tournament()
            c1, c2 = sbx_crossover([self.population[p1]["values"][k] for k in
                                    range(len(self.bounds))],
                                   [self.population[p2]["values"][k] for k in
                                    range(len(self.bounds))],
                                   self.bounds, eta_c, rng)
            c1 = apply_kind(polynomial_mutation(c1, self.bounds, eta_m, rng),
                            self.kinds)
            c2 = apply_kind(polynomial_mutation(c2, self.bounds, eta_m, rng),
                            self.kinds)
            children.extend((c1, c2))
        return children[:self.pop_size]

    def _adaptive_hybrid_proposals(self, rng, generation):
        """Generate an adaptive hybrid search batch.

        The proposal mix maintains diversity, searches globally and locally
        at the same time, and adapts the balance as the run progresses.
        Environmental selection remains the shared constrained Pareto
        selector used by both modes.
        """
        if not self.population:
            return self._initial_population(rng)

        self._adaptive_scale = max(0.035, 0.18 * (0.93 ** max(0, generation - 1)))
        self._adaptive_global_weight = max(
            0.20, 0.40 - 0.015 * max(0, generation - 1))

        valid = [r for r in self.population
                 if r["F"] and max(r["F"]) < PENALTY and math.isfinite(r["cv"])]
        if not valid:
            valid = list(self.population)
        fronts = nondominated_fronts(
            [r["F"] for r in valid], [r["cv"] for r in valid])
        elite = [valid[i] for i in (fronts[0] if fronts else range(len(valid)))]
        if not elite:
            elite = valid

        def vector(record):
            return [float(v) for v in record["values"]]

        def local_candidate():
            center = vector(rng.choice(elite))
            out = []
            for value, (lo, hi) in zip(center, self.bounds):
                sigma = (hi - lo) * self._adaptive_scale
                out.append(_clip(value + rng.gauss(0.0, sigma), lo, hi))
            return apply_kind(out, self.kinds)

        def global_candidate():
            # Differential-evolution style mutation supplies broad proposals
            # without requiring a fitted surrogate model.
            pool = self.population
            a, b = (rng.choice(pool) for _ in range(2))
            factor = rng.uniform(0.45, 0.95)
            base = vector(rng.choice(elite))
            out = []
            for k, (lo, hi) in enumerate(self.bounds):
                value = base[k] + factor * (a["values"][k] - b["values"][k])
                if rng.random() < 0.15:
                    value += rng.gauss(0.0, (hi - lo) * 0.08)
                out.append(_clip(value, lo, hi))
            return apply_kind(out, self.kinds)

        def genetic_candidate():
            p1, p2 = rng.choice(elite), rng.choice(self.population)
            child, _ = sbx_crossover(vector(p1), vector(p2), self.bounds,
                                     15.0, rng)
            return apply_kind(polynomial_mutation(
                child, self.bounds, 20.0, rng), self.kinds)

        proposals = []
        for _ in range(self.pop_size):
            roll = rng.random()
            if roll < self._adaptive_global_weight:
                proposals.append(global_candidate())
            elif roll < 0.65:
                proposals.append(genetic_candidate())
            else:
                proposals.append(local_candidate())
        return proposals

    # -------------------------------------------------------- convergence
    def _current_front_ids(self):
        F = [r["F"] for r in self.records]
        CV = [r["cv"] for r in self.records]
        return sorted(self.records[i]["id"] for i in front_ids(F, CV))

    def _check_generation(self, gen):
        """Bookkeeping after each generation.

        Convergence = the hypervolume of the accumulated front improved by
        less than HV_TOL (relative) over the previous generation, for
        `patience` consecutive generations. With constraints, the
        hypervolume tracks the feasible front (all points until one is
        feasible, so progress toward feasibility still counts). Returns
        True when the run should stop; the reason is stored in
        self._finish_reason."""
        valid = [r for r in self.records if max(r["F"]) < PENALTY]
        feasible = [r for r in valid if r["cv"] <= 0.0]
        hv_set = feasible if feasible else valid
        hv = hypervolume([r["F"] for r in hv_set])
        if self._last_hv is not None and self._last_hv > 0.0:
            gain = (hv - self._last_hv) / self._last_hv
            # only a genuine plateau counts toward convergence; a large
            # drop (e.g. the feasible set switching on) is a real change
            self._stale_gens = self._stale_gens + 1 \
                if abs(gain) < HV_TOL else 0
            if self.algorithm == "Adaptive Hybrid Search":
                # A stalled front calls for more broad exploration; steady
                # improvement allows the next batch to exploit elite regions.
                if abs(gain) < HV_TOL:
                    self._adaptive_global_weight = min(
                        0.70, self._adaptive_global_weight + 0.05)
                    self._adaptive_scale = min(0.30, self._adaptive_scale * 1.08)
                elif gain > 0.0:
                    self._adaptive_global_weight = max(
                        0.20, self._adaptive_global_weight - 0.02)
        else:
            self._stale_gens = 0
        self._last_hv = hv

        if self.stop_requested:
            self._finish_reason = ("stopped by user", False)
            return True
        if self.max_generations and gen + 1 >= self.max_generations:
            self._finish_reason = (f"generation cap reached "
                                   f"({self.max_generations})", False)
            return True
        if self._stale_gens >= self.patience:
            self._finish_reason = (
                f"converged: front hypervolume stalled below "
                f"{HV_TOL * 100:.1f}% gain for {self.patience} generations",
                True)
            return True
        self._finish_reason = ("stopped", False)
        return False

    def _best_text(self):
        """Human summary: best value per objective over all valid evals."""
        parts = []
        for k, obj in enumerate(self.objectives):
            vals = [r["F"][k] * self.signs[k] for r in self.records
                    if r["F"] is not None and r["F"][k] < PENALTY]
            if not vals:
                continue
            best = min(vals) if self.signs[k] > 0 else max(vals)
            parts.append(f"{obj['label']} {best:.5g}")
        return " | ".join(parts) if parts else "no valid evaluations yet"

    # -------------------------------------------------------- evaluation
    def _make_record(self, gen, idx, values, case):
        # a failed/aborted evaluation keeps the penalty vector and is
        # infeasible: dominated by everything, never enters the front
        return {"id": self._next_id, "gen": gen, "idx": idx,
                "case": case, "values": list(values),
                "F": [PENALTY] * len(self.objectives), "raw": None,
                "cv": math.inf, "status": "failed", "error": "",
                "converged": None, "seconds": 0.0}

    def _evaluate(self, gen, idx, values):
        case = f"{self.run_tag}_g{gen:03d}i{idx:02d}"
        record = self._make_record(gen, idx, values, case)
        self._next_id += 1
        self._emit("eval_start", gen=gen, idx=idx, case=case)
        t0 = time.time()
        try:
            eval_dir = self._write_eval_input(gen, idx, values, case)
            case_dir = REPO / "cases" / case
            stages = [
                ("geometry + mesh",
                 [PY, str(REPO / "pipeline" / "run_pipeline.py"),
                  str(eval_dir / "input.json")], str(REPO), None),
                ("SU2 case setup",
                 [PY, str(REPO / "tools" / "setup_cascade_case.py"),
                  str(eval_dir), "--name", case, "--skip-validate"],
                 str(REPO), None),
                (f"SU2 solve ({self.threads} threads)",
                 [str(self.su2_exe), "-t", str(self.threads), "turbine.cfg"],
                 str(case_dir), str(case_dir / "su2_run.log")),
                ("post-processing",
                 [PY, str(REPO / "tools" / "plot_case.py"), str(case_dir)],
                 str(case_dir), None),
            ]
            for title, argv, cwd, logfile in stages:
                if self.stop_requested:
                    record["error"] = "stopped by user"
                    return self._finish_record(record, t0)
                self._log(f"[g{gen} i{idx}] {title} ...")
                rc, timed_out = self._run_stage(argv, cwd, logfile,
                                                timeout=self.stage_timeout)
                if self.stop_requested:
                    record["error"] = "stopped by user"
                    return self._finish_record(record, t0)
                if rc != 0:
                    if timed_out:
                        minutes = (self.stage_timeout or 0) / 60.0
                        record["error"] = (
                            f"{title} killed after {minutes:.1f} min (hang, "
                            "e.g. degenerate geometry) - treated as failed")
                        self._log(f"[g{gen} i{idx}] {record['error']}")
                    else:
                        record["error"] = f"{title} failed (exit code {rc})"
                    return self._finish_record(record, t0)

            results_path = case_dir / "results.json"
            if not results_path.exists():
                record["error"] = "no results.json after post-processing"
                return self._finish_record(record, t0)
            results = json.loads(results_path.read_text())
            self._fill_objectives(record, results)
        except Exception as e:
            record["error"] = str(e)
        return self._finish_record(record, t0)

    def _fill_objectives(self, record, results):
        record["converged"] = bool(
            (results.get("convergence") or {}).get("target_reached"))
        raw = {}
        F = []
        for obj, sign in zip(self.objectives, self.signs):
            v = get_path(results, obj["path"])
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(
                    f"objective '{obj['path']}' not found in results.json")
            raw[obj["path"]] = float(v)
            F.append(sign * float(v))
        for c in self.constraints:          # constraint quantities too
            v = get_path(results, c["path"])
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                raw[c["path"]] = float(v)
        record["raw"] = raw
        if self.require_converged and not record["converged"]:
            record["F"] = [PENALTY] * len(self.objectives)
            record["cv"] = math.inf
            record["error"] = "solver did not reach the residual target"
        else:
            record["F"] = F
            record["cv"] = constraint_violation(raw, self.constraints) \
                if self.constraints else 0.0
            record["status"] = "ok"

    def _write_eval_input(self, gen, idx, values, case):
        eval_dir = self.run_dir / f"g{gen:03d}i{idx:02d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(self.base_config)
        for dv, v in zip(self.design_vars, values):
            set_design_value(cfg, dv["path"], v)
        # cage-offset variables only shape the blade with the morph on
        if any(d["path"].startswith("airfoil_source.morph.")
               and "@" in d["path"] for d in self.design_vars):
            set_path(cfg, "airfoil_source.morph.enabled", True)
        # pitch exploration must keep the SU2 single-translation pair valid
        # (the 3D wedge's rotational pair instead supports R1 != R2)
        if get_path(cfg, "domain.R1") is not None and \
                get_path(cfg, "domain.periodicity") != "axisymmetric3d":
            set_path(cfg, "domain.R2", get_path(cfg, "domain.R1"))
        # local solver override: applies to this evaluation only
        if self.max_iterations:
            set_path(cfg, "solver_settings.max_iterations",
                     int(self.max_iterations))
        set_path(cfg, "case.name", case)
        set_path(cfg, "case.output_dir", "cases")
        set_path(cfg, "case.show_images", False)
        path = eval_dir / "input.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        # the SU2 case folder keeps its own copy of what produced it
        try:
            (REPO / "cases" / case).mkdir(parents=True, exist_ok=True)
            (REPO / "cases" / case / "input.json").write_text(
                json.dumps(cfg, indent=2) + "\n")
        except OSError:
            pass
        return eval_dir

    def _run_stage(self, argv, cwd, logfile, timeout=None):
        """Run one pipeline stage. Returns (rc, timed_out).

        The stage is killed (process tree) when the run is stopped or
        when `timeout` seconds elapse - a hanging stage must never block
        the optimization loop."""
        self._stage_timed_out = False
        logf = None
        try:
            if logfile:
                Path(logfile).parent.mkdir(parents=True, exist_ok=True)
                logf = open(logfile, "w")
                out = logf
            else:
                out = subprocess.PIPE
            self.proc = subprocess.Popen(
                argv, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                **_popen_flags())
        except OSError as e:
            self._log(f"[error] could not start {argv[0]}: {e}")
            return -1, False
        timer = None
        if timeout:
            timer = threading.Timer(timeout, self._on_stage_timeout)
            timer.daemon = True
            timer.start()
        try:
            if out is subprocess.PIPE:
                for raw in self.proc.stdout:
                    self._log("    " + raw.decode("utf-8",
                                                  errors="replace").rstrip())
            while self.proc.poll() is None:
                time.sleep(0.2)
        finally:
            if timer:
                timer.cancel()
            if logf:
                logf.close()
        rc = self.proc.returncode if self.proc.returncode is not None else -1
        return rc, self._stage_timed_out

    def _finish_record(self, record, t0):
        record["seconds"] = time.time() - t0
        record["time"] = time.strftime("%H:%M:%S")
        self._write_csv_row(record)
        status = record["status"]
        vals = ""
        if record["raw"]:
            vals = ", ".join(f"{o['label']}={record['raw'][o['path']]:.5g}"
                             for o in self.objectives)
        if record["cv"] is not None and record["cv"] != math.inf \
                and record["cv"] > 0.0:
            vals += f", CV={record['cv']:.3g}"
        self._log(f"[g{record['gen']} i{record['idx']}] {status} "
                  f"({record['seconds']:.0f} s) {vals}"
                  + (f" - {record['error']}" if record["error"] else ""))
        return record

    # ------------------------------------------------------------- output
    def _open_outputs(self):
        settings = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_case": self.base_config.get("case", {}).get("name"),
            "objectives": self.objectives,
            "constraints": self.constraints,
            "design_vars": [{"path": d["path"], "min": d["min"],
                             "avg": d.get("avg"), "max": d["max"],
                             "kind": d["kind"]} for d in self.design_vars],
            "pop_size": self.pop_size,
            "patience": self.patience,
            "max_generations": self.max_generations,
            "require_converged": self.require_converged,
            "max_iterations": self.max_iterations,
            "threads": self.threads,
            "stage_timeout_min": (self.stage_timeout / 60.0
                                  if self.stage_timeout else None),
            "seed": self.seed,
            "resumed": self._resumed,
            "resumed_evaluations": len(self.records) if self._resumed else 0,
        }
        (self.run_dir / "settings.json").write_text(
            json.dumps(settings, indent=2) + "\n")
        self._csv_file = self.run_dir / "evaluations.csv"
        self._csv = open(self._csv_file, "w", newline="")
        header = (["time", "gen", "idx", "id", "status", "converged",
                   "case", "seconds"]
                  + [d["path"] for d in self.design_vars]
                  + [o["path"] for o in self.objectives]
                  + [c.get("col", c["path"]) for c in self.constraints]
                  + ["cv", "error"])
        self._csv.writer = csv.writer(self._csv)
        self._csv.writer.writerow(header)
        if self._resumed:
            for record in self.records:     # carried-over history rows
                self._write_csv_row(record)
        self._csv.flush()

    def _write_csv_row(self, record):
        row = [record.get("time") or time.strftime("%H:%M:%S"),
               record["gen"], record["idx"],
               record["id"], record["status"],
               "" if record["converged"] is None else record["converged"],
               record["case"], f"{record['seconds']:.1f}"]
        row += list(record["values"])
        for obj in self.objectives:
            if record["raw"] and obj["path"] in record["raw"]:
                row.append(record["raw"][obj["path"]])
            else:
                row.append("")
        for c in self.constraints:
            if record["raw"] and c["path"] in record["raw"]:
                row.append(record["raw"][c["path"]])
            else:
                row.append("")
        row.append(record["cv"] if record["cv"] != math.inf else "inf")
        row.append(record["error"])
        self._csv.writer.writerow(row)
        self._csv.flush()

    def _close_csv(self):
        if self._csv:
            try:
                self._csv.close()
            except OSError:
                pass
            self._csv = None

    # ------------------------------------------------------- saved state
    @classmethod
    def from_state(cls, state, run_dir, run_tag, su2_exe=None):
        """Build a run that continues from a dict loaded from
        optimizer_state.json (validate_state() is applied first).

        The new run evaluates into `run_dir`; the carried history is
        rewritten into its evaluations.csv so the resumed run stays
        self-contained and auditable."""
        validate_state(state)
        opts = dict(state.get("options") or {})
        run = cls(state["base_config"], Path(run_dir), run_tag,
                  state["objectives"], state["design_vars"],
                  constraints=state.get("constraints") or [],
                  pop_size=int(opts.get("pop_size", 8)),
                  patience=int(opts.get("patience", 10)),
                  max_generations=int(opts.get("max_generations", 0)),
                  require_converged=bool(opts.get("require_converged", True)),
                  threads=int(opts.get("threads") or 6),
                  su2_exe=su2_exe,
                  seed=opts.get("seed"),
                  max_iterations=opts.get("max_iterations"),
                  stage_timeout=opts.get("stage_timeout"),
                  algorithm=state["algorithm"])
        run._load_state(state)
        return run

    def _load_state(self, state):
        """Adopt the saved search state (called via from_state)."""
        self.records = state["records"]
        self.population = state.get("population") or []
        self._pending = [list(v) for v in (state.get("pending") or [])]
        self._pending_idx = int(state.get("pending_idx") or 0)
        if self._pending_idx >= len(self._pending):
            self._pending, self._pending_idx = [], 0   # batch was finished
        self._generation = int(state.get("generation") or 0)
        c = state.get("counters") or {}
        self._next_id = int(c.get("next_id", len(self.records)))
        self._stale_gens = int(c.get("stale_gens", 0))
        self._last_hv = c.get("last_hv")
        self._adaptive_scale = float(
            c.get("adaptive_scale", self._adaptive_scale))
        self._adaptive_global_weight = float(
            c.get("adaptive_global_weight", self._adaptive_global_weight))
        rs = state.get("rng_state")
        self._rng_state = (rs[0], tuple(rs[1]), rs[2]) if rs else None
        self._resumed = True

    def _state_dict(self):
        """The complete resumable state of this run as a JSON-ready dict."""
        if self._rng is not None:
            self._rng_state = self._rng.getstate()
        return {
            "version": STATE_VERSION,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_tag": self.run_tag,
            "algorithm": self.algorithm,
            "base_config": self.base_config,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "design_vars": self.design_vars,
            "options": {
                "pop_size": self.pop_size,
                "patience": self.patience,
                "max_generations": self.max_generations,
                "require_converged": self.require_converged,
                "max_iterations": self.max_iterations,
                "threads": self.threads,
                "stage_timeout": self.stage_timeout,
                "seed": self.seed,
            },
            "rng_state": self._rng_state,
            "generation": self._generation,
            "pending": self._pending,
            "pending_idx": self._pending_idx,
            "counters": {
                "next_id": self._next_id,
                "stale_gens": self._stale_gens,
                "last_hv": self._last_hv,
                "adaptive_scale": self._adaptive_scale,
                "adaptive_global_weight": self._adaptive_global_weight,
            },
            "population": self.population,
            "records": self.records,
        }

    def _write_state(self, path=None):
        """Write optimizer_state.json (atomic replace). Safe to call from
        the worker thread only while it is between evaluations - the
        worker checkpoints itself via _checkpoint_if_requested() and
        _wait_if_paused()."""
        try:
            path = Path(path) if path else self.state_path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(
                json.dumps(self._state_dict(), indent=2) + "\n",
                encoding="utf-8")
            tmp.replace(path)
        except (OSError, TypeError, ValueError) as e:
            self._log(f"[error] could not save optimizer state: {e}")
            return None
        self._emit("state_saved", path=str(path), count=len(self.records))
        return path
