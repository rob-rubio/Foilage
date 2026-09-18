"""Shared case state: one input.json dict edited by all tabs."""

import copy
import json
import math
import sys
from pathlib import Path

from .schema import DEFAULTS, FIELDS_BY_PATH

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from freestream import gamma_of_air  # noqa: E402

GAMMA, R_AIR = 1.4, 287.058


def get_path(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_path(cfg, dotted, value):
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


class CaseState:
    """Holds the input.json dict, its path, and notifies tabs on changes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.config = {}
        self.dirty = False
        self._observers = []      # callables(changed_paths: set)
        self.load(self.path)

    # ------------------------------------------------------------ io
    def load(self, path):
        path = Path(path)
        raw = json.loads(path.read_text())
        # keep unknown user keys; add schema defaults for missing known keys
        merged = copy.deepcopy(DEFAULTS)
        _deep_merge(merged, raw)
        self.config = merged
        self.path = path
        self.dirty = False
        self._notify(set(FIELDS_BY_PATH))

    def save(self, path=None):
        path = Path(path) if path else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.config, indent=2) + "\n")
        self.path = path
        self.dirty = False

    # ---------------------------------------------------------- access
    def get(self, dotted):
        return get_path(self.config, dotted)

    def set(self, dotted, value, notify=True):
        old = self.get(dotted)
        if _same(old, value):
            return False
        set_path(self.config, dotted, value)
        self.dirty = True
        if notify:
            self._notify({dotted})
        return True

    def case_name(self):
        return str(self.get("case.name") or self.path.parent.name)

    def case_dir(self):
        """SU2 case directory (repo/cases/<name>)."""
        repo = Path(__file__).resolve().parent.parent
        return repo / "cases" / self.case_name()

    def mesh_project_dir(self):
        """Mesh-project output dir <input_dir>/<output_dir>/<name>."""
        out = Path(str(self.get("case.output_dir") or "cases"))
        if not out.is_absolute():
            out = self.path.parent / out
        return out / self.case_name()

    # ------------------------------------------------------------ observers
    def observe(self, callback):
        self._observers.append(callback)

    def _notify(self, paths):
        for cb in list(self._observers):
            try:
                cb(paths)
            except Exception:            # a broken tab must not kill others
                pass

    # ------------------------------------------------------------ validation
    def validate(self):
        """Cross-field checks: list of (severity, message)."""
        issues = []
        r1, r2 = self.get("domain.R1"), self.get("domain.R2")
        if r1 is not None and r2 is not None and abs(r1 - r2) > 1e-9:
            issues.append(("warning",
                           f"R1 ({r1}) != R2 ({r2}): the mesh can be built, "
                           "but SU2 needs R1 = R2 for a single-translation "
                           "periodic pair."))
        p01 = self.get("BCs.inlet.total pressure")
        p2 = self.get("BCs.outlet.static pressure")
        if p01 is not None and p2 is not None and p2 >= p01:
            issues.append(("error",
                           f"Outlet static pressure ({p2} Pa) must be below "
                           f"the inlet total pressure ({p01} Pa)."))
        src_type = self.get("airfoil_source.type") or "pyturbo"
        if src_type == "pyturbo":
            a1 = self.get("airfoil.alpha1")
            a2 = self.get("airfoil.alpha2")
            if a1 is not None and a2 is not None and abs(a2 - a1) < 1.0:
                issues.append(("warning",
                               "alpha1 ~= alpha2: the blade has almost no "
                               "turning; check the metal angles."))
            if a1 is not None and a2 is not None and a2 < a1:
                issues.append(("info",
                               "alpha2 < alpha1: clockwise turning, built as "
                               "a CAP profile (suction side on top)."))
        if src_type == "geomturbo" and not self.get(
                "airfoil_source.geomturbo_file"):
            issues.append(("error",
                           "geomTurbo source selected but no geomTurbo file "
                           "is set."))
        n = self.get("domain.airfoil_count")
        if n is not None and n < 2:
            issues.append(("error", "Blade count N must be >= 2."))
        x0 = self.get("domain.x_min")
        x1 = self.get("domain.x_max")
        if x0 is not None and x1 is not None and x1 <= x0:
            issues.append(("error", "Domain x_max must be greater than x_min."))
        fl = self.get("mesh.boundary_layer.first_layer_height")
        nlay = self.get("mesh.boundary_layer.n_layers")
        if fl and nlay and fl * ((1.25) ** nlay) > 0.05:
            issues.append(("warning",
                           "Boundary-layer stack is thicker than ~0.05 "
                           "axial chord; reduce layers or growth rate."))
        return issues

    # ------------------------------------------------------------- derived
    def scale_m_per_chord(self):
        ac = float(self.get("airfoil.axial_chord") or 1.0)
        return ac / 1000.0 if ac >= 1.0 else ac

    def effective_gamma(self):
        """(gamma, source): the user-set value, or computed from the inlet
        total temperature with temperature-dependent specific heats of air."""
        g = self.get("solver_settings.gamma")
        if g:
            return float(g), f"set to {g:g}"
        T01 = self.get("BCs.inlet.total temperature")
        if not T01:
            return GAMMA, "default (no temperature set)"
        return gamma_of_air(float(T01)), f"computed from T01 = {T01:g} K"

    def derived(self):
        """Live quantities computed from the current BCs/geometry."""
        d = {}
        p01 = self.get("BCs.inlet.total pressure")
        T01 = self.get("BCs.inlet.total temperature")
        p2 = self.get("BCs.outlet.static pressure")
        r1, n = self.get("domain.R1"), self.get("domain.airfoil_count")
        scale = self.scale_m_per_chord()
        ac = float(self.get("airfoil.axial_chord") or 1.0)
        d["scale_m"] = scale
        gamma, gamma_src = self.effective_gamma()
        d["gamma_eff"] = gamma
        d["gamma_source"] = gamma_src
        units_to_m = 0.001 if (ac or 0.0) >= 1.0 else 1.0
        if r1 and n:
            d["pitch_le_m"] = 2.0 * math.pi * r1 * units_to_m / n
        if r1 and n and self.get("domain.R2"):
            d["pitch_te_m"] = 2.0 * math.pi * float(self.get("domain.R2")) \
                * units_to_m / n
        if None not in (p01, T01, p2) and p01 and p2 and p2 < p01:
            g = gamma
            T2 = T01 * (p2 / p01) ** ((g - 1) / g)
            V2 = math.sqrt(2 * g / (g - 1) * R_AIR * (T01 - T2))
            rho2 = p2 / (R_AIR * T2)
            mu2 = 1.716e-5 * (T2 / 273.15) ** 1.5 * (383.55 / (T2 + 110.4))
            d["pr"] = p2 / p01
            d["T2"] = T2
            d["V2"] = V2
            d["M2"] = V2 / math.sqrt(g * R_AIR * T2)
            d["Re_axial"] = rho2 * V2 * scale / mu2
        return d


def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _same(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b
