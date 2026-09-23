"""Named mesh profiles: reusable mesh-tab configurations.

Profiles live in <repo>/mesh_profiles.json (next to foilage.cfg) and map
a profile name to its type plus one value per Mesh-tab field:

    {"profiles": {"High Fidelity": {"type": "protected",
                                    "description": "...",
                                    "values": {"mesh.max_size": ...}}}}

The Mesh tab loads a profile from the drop-down (its values are pushed
into the case's input.json state) and saves/deletes user profiles from
inside the tab. The shipped "High Fidelity" and "Optimization Mesh"
entries are type "protected": they can be updated from the tab but never
deleted there, and a missing or type-mangled entry is re-seeded from the
defaults below on every load. A file that fails to parse is left
untouched and only the shipped profiles are offered (saving disabled).
"""

import json
from pathlib import Path

from .schema import DEFAULTS, sections_for
from .state import get_path

REPO = Path(__file__).resolve().parent.parent
PROFILES_FILE = REPO / "mesh_profiles.json"

PROTECTED = "protected"
USER = "user"

# overrides on top of the schema defaults for the shipped profiles; every
# other Mesh-tab field keeps its schema default in the profile
SHIPPED = {
    "High Fidelity": {
        "mesh.max_size": 0.014,
        "mesh.near_wall_size": 0.003,
        "mesh.refine_dist": 0.8,
        "mesh.periodic_size": 0.015,
        "mesh.airfoil_points": 600,
        "mesh.boundary_layer.first_layer_height": 1.0e-4,
        "mesh.boundary_layer.growth_rate": 1.2,
        "mesh.boundary_layer.n_layers": 24,
        "mesh.span.layers": 41,
        "mesh.wake.size": 0.01,
        "mesh.wake.transition": 0.5,
    },
    "Optimization Mesh": {
        "mesh.max_size": 0.04,
        "mesh.near_wall_size": 0.012,
        "mesh.refine_dist": 0.5,
        "mesh.periodic_size": 0.04,
        "mesh.airfoil_points": 250,
        "mesh.boundary_layer.first_layer_height": 4.0e-4,
        "mesh.boundary_layer.growth_rate": 1.4,
        "mesh.boundary_layer.n_layers": 10,
        "mesh.span.layers": 11,
        "mesh.wake.size": 0.035,
        "mesh.wake.half_width": 0.12,
        "mesh.wake.transition": 0.3,
    },
}

SHIPPED_DESCRIPTIONS = {
    "High Fidelity":
        "Fine reference mesh: small far-field/near-wall sizes, 24 "
        "boundary-layer layers, 41 spanwise layers. Best quality, "
        "slowest to generate and solve.",
    "Optimization Mesh":
        "Coarse, fast mesh for optimization and ML sweeps where many "
        "meshes are generated; keeps a boundary-layer stack for solver "
        "robustness.",
}


def mesh_field_paths():
    """Dotted paths of every field shown in the Mesh tab, schema order."""
    return [f.path for section in sections_for("mesh")
            for f in section.fields]


def default_values(overrides=None):
    """Complete Mesh-tab value set: schema defaults plus overrides."""
    values = {p: get_path(DEFAULTS, p) for p in mesh_field_paths()}
    if overrides:
        values.update(overrides)
    return values


def capture_values(state):
    """Snapshot the current Mesh-tab values from a CaseState."""
    return {p: state.get(p) for p in mesh_field_paths()}


def apply_values(state, values):
    """Push profile values into a CaseState (marks the case dirty)."""
    for path, value in values.items():
        state.set(path, value)


class MeshProfiles:
    """Load/save/delete mesh profiles in one JSON file."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else PROFILES_FILE
        self.error = None             # unreadable file: message for the tab
        self.profiles = {}
        self._reload()

    # --------------------------------------------------------------- io
    def _reload(self):
        self.error = None
        raw = {}
        if self.path.exists():
            try:
                parsed = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict) \
                        and isinstance(parsed.get("profiles"), dict):
                    raw = parsed["profiles"]
                else:
                    self.error = "missing the 'profiles' object"
            except (OSError, json.JSONDecodeError) as e:
                self.error = str(e)
        if self.error:
            # keep only the shipped profiles in memory; the file on disk
            # stays untouched so hand-editing can repair it
            self.profiles = {}
            self._seed_protected()
            return
        self.profiles = raw
        self._seed_protected()
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"profiles": self.profiles}, indent=2) + "\n",
            encoding="utf-8")

    def _seed_protected(self):
        """(Re-)add the shipped profiles and force their protected type."""
        for name, overrides in SHIPPED.items():
            entry = self.profiles.get(name)
            if isinstance(entry, dict) and entry.get("values"):
                entry["type"] = PROTECTED
            else:
                self.profiles[name] = {
                    "type": PROTECTED,
                    "description": SHIPPED_DESCRIPTIONS.get(name, ""),
                    "values": default_values(overrides),
                }

    # ------------------------------------------------------------ access
    def names(self):
        """Profile names, protected (shipped) ones first."""
        return sorted(self.profiles,
                      key=lambda n: (0 if self.type_of(n) == PROTECTED
                                     else 1, n))

    def get(self, name):
        return self.profiles.get(name)

    def type_of(self, name):
        entry = self.get(name)
        return entry.get("type") if isinstance(entry, dict) else None

    def values(self, name):
        """The profile's Mesh-tab values (unknown file keys dropped)."""
        entry = self.get(name)
        if not isinstance(entry, dict):
            return {}
        known = set(mesh_field_paths())
        return {p: v for p, v in entry.get("values", {}).items()
                if p in known}

    # ------------------------------------------------------- persistence
    def save(self, name, values):
        """Create or update a profile. An existing protected profile keeps
        its type (it can be updated, just not deleted)."""
        name = str(name).strip()
        if not name:
            raise ValueError("profile name must not be empty")
        entry = self.profiles.get(name)
        if not isinstance(entry, dict):
            entry = {"type": USER, "description": ""}
        entry["values"] = dict(values)
        self.profiles[name] = entry
        self.flush()

    def delete(self, name):
        """False (and no change) for missing or protected profiles."""
        if self.get(name) is None or self.type_of(name) == PROTECTED:
            return False
        del self.profiles[name]
        self.flush()
        return True
