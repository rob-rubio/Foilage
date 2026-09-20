"""Plugin system for custom geometry generators.

Every subdirectory of ``extensions/`` that contains a ``plugin.json``
manifest becomes an airfoil source: it appears automatically in the
Geometry tab's "Geometry source" dropdown, renders its own parameter
panels (sliders, dropdowns, arrays - the same widgets as the built-in
forms), and its CLI script is run automatically whenever a preview or a
mesh build needs the geometry.

Manifest (extensions/<plugin>/plugin.json):

    {
      "id": "naca",                          // optional, default: dir name
      "name": "NACA 4-digit generator",      // dropdown label
      "description": "Generates a NACA...",  // shown under the panels
      "cli": "generate_naca.py",             // script to run (required)
      "timeout_s": 120,                       // CLI timeout
      "groups": [                            // optional panel grouping
        {"title": "Camber", "params": ["camber", "camber_pos"]}
      ],
      "parameters": [
        {"path": "camber", "label": "Max camber", "kind": "float",
         "default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005,
         "slider": true, "unit": "c", "tooltip": "..."},
        ...
      ]
    }

Parameter values are stored in input.json under
``airfoil_source.params.<path>``. The CLI contract:

    <python> <cli> --config <config.json> --output <result.json>

config.json carries ``{"parameters": {...}, "n_points": ..., "axial_chord":
..., "case_name": ...}``. The script writes result.json with

    {"ss": [[x, y], ...], "ps": [[x, y], ...], "blade_count": 45}

- SS/PS polylines ordered LE -> TE, in any consistent unit (the section
is normalized to axial chord = 1 exactly like a geomTurbo import);
``blade_count`` is optional. Anything on the script's stderr is reported
in the GUI status when the run fails.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EXTENSIONS_DIR = Path(__file__).resolve().parent.parent / "extensions"
MANIFEST_NAME = "plugin.json"

RESERVED_IDS = ("pyturbo", "geomturbo")
DEFAULT_TIMEOUT_S = 120.0
SPEC_KINDS = ("float", "int", "str", "bool", "choice", "float_array",
              "nullable_float", "nullable_int", "file")


# ---------------------------------------------------------- discovery
def _sanitize_id(name):
    return re.sub(r"[^a-z0-9_]+", "_", str(name).lower()).strip("_") \
        or "plugin"


def _load_manifest(directory, manifest_path):
    """One plugin entry from its directory; never raises - problems are
    reported in the returned dict's "error" field."""
    plugin = {"id": _sanitize_id(directory.name), "name": directory.name,
              "dir": str(directory), "description": "", "cli": None,
              "timeout_s": DEFAULT_TIMEOUT_S, "parameters": [],
              "groups": [], "error": None}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("manifest must be a JSON object")
    except Exception as e:
        plugin["error"] = f"cannot read {MANIFEST_NAME}: {e}"
        return plugin

    if data.get("id"):
        plugin["id"] = _sanitize_id(data["id"])
    plugin["name"] = str(data.get("name") or plugin["name"])
    plugin["description"] = str(data.get("description") or "")
    plugin["timeout_s"] = _positive_float(data.get("timeout_s"),
                                          DEFAULT_TIMEOUT_S)

    cli = str(data.get("cli") or "").strip()
    if not cli:
        plugin["error"] = "manifest declares no cli script"
        return plugin
    cli_path = directory / cli
    if not cli_path.is_file():
        plugin["error"] = f"cli script not found: {cli}"
        return plugin
    plugin["cli"] = str(cli_path)

    raw_params = data.get("parameters")
    if raw_params is None:
        raw_params = []
    if not isinstance(raw_params, list):
        plugin["error"] = "'parameters' must be a list"
        return plugin
    seen_paths = set()
    for spec in raw_params:
        if not isinstance(spec, dict) or not str(spec.get("path") or "").strip():
            plugin["error"] = "a parameter is missing its 'path'"
            return plugin
        spec = dict(spec)
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(spec["path"])).strip("_")
        if not clean:
            plugin["error"] = f"parameter path {spec['path']!r} is invalid"
            return plugin
        spec["path"] = clean
        if spec["path"] in seen_paths:
            plugin["error"] = f"duplicate parameter path '{clean}'"
            return plugin
        seen_paths.add(spec["path"])
        if spec.get("kind") not in SPEC_KINDS:
            spec["kind"] = "float"
        plugin["parameters"].append(spec)

    by_key = {p["path"]: p for p in plugin["parameters"]}
    groups = data.get("groups")
    if groups:
        if not isinstance(groups, list):
            plugin["error"] = "'groups' must be a list"
            return plugin
        for g in groups:
            keys = [str(k) for k in (g.get("params") or [])
                    if str(k) in by_key]
            plugin["groups"].append({"title": str(g.get("title") or "Parameters"),
                                     "params": [by_key[k] for k in keys]})
    if not plugin["groups"] or not any(g["params"] for g in plugin["groups"]):
        plugin["groups"] = [{"title": plugin["name"],
                             "params": list(plugin["parameters"])}]
    return plugin


def _positive_float(value, default):
    try:
        v = float(value)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def discover_plugins(extensions_dir=None):
    """All plugin manifests under the extensions directory (broken ones
    are included with an "error" message instead of being raised)."""
    root = Path(extensions_dir) if extensions_dir else EXTENSIONS_DIR
    plugins = []
    if not root.is_dir():
        return plugins
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        mf = child / MANIFEST_NAME
        if not mf.is_file():
            continue
        plugins.append(_load_manifest(child, mf))
    seen = {}
    for p in plugins:
        if not p["error"]:
            if p["id"] in RESERVED_IDS:
                p["error"] = (f"plugin id '{p['id']}' is reserved by the "
                              "built-in sources")
            elif p["id"] in seen:
                p["error"] = f"duplicate plugin id '{p['id']}'"
                if not seen[p["id"]]["error"]:
                    seen[p["id"]]["error"] = p["error"]
        seen[p["id"]] = p
    return plugins


def get_plugin(plugin_id, extensions_dir=None):
    """The plugin with this id, or None."""
    for p in discover_plugins(extensions_dir):
        if p["id"] == plugin_id:
            return p
    return None


def type_choices(plugins=None):
    """(label, id) pairs for the Geometry source dropdown."""
    if plugins is None:
        plugins = discover_plugins()
    return [(p["name"], p["id"]) for p in plugins if not p["error"]]


def parameter_defaults(plugin):
    """{param path: default} for the plugin's declared parameters."""
    return {p["path"]: p.get("default")
            for p in plugin["parameters"] if p.get("default") is not None}


# --------------------------------------------------------------- runner
def _get_path(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _check_points(points, what):
    arr = []
    for row in points:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(f"plugin result '{what}' must hold [x, y] pairs")
        x, y = float(row[0]), float(row[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"plugin result '{what}' has a non-finite point")
        arr.append([x, y])
    if len(arr) < 2:
        raise ValueError(f"plugin result '{what}' needs at least 2 points")
    return arr


def run_plugin_generator(plugin_id, cfg, extensions_dir=None):
    """Run a plugin's CLI and return {"ss", "ps", "blade_count"}.

    The CLI receives --config/--output JSON paths (see module docstring);
    the result is cached per parameter set so slider previews do not
    re-spawn the script for unchanged inputs. Raises ValueError with a
    user-facing message on any failure."""
    plugin = get_plugin(plugin_id, extensions_dir)
    if plugin is None:
        raise ValueError(f"unknown geometry source plugin '{plugin_id}' - "
                         f"check {EXTENSIONS_DIR}")
    if plugin["error"]:
        raise ValueError(f"geometry plugin '{plugin_id}': {plugin['error']}")

    params = {}
    for spec in plugin["parameters"]:
        value = _get_path(cfg, f"airfoil_source.params.{spec['path']}")
        if value is None:
            value = spec.get("default")
        params[spec["path"]] = value
    payload = {
        "parameters": params,
        "n_points": int(_get_path(cfg, "airfoil.n_points") or 401),
        "axial_chord": _get_path(cfg, "airfoil.axial_chord"),
        "case_name": _get_path(cfg, "case.name"),
    }

    key = (plugin_id, _mtime(plugin["cli"]),
           json.dumps(payload, sort_keys=True, default=str))
    if _CACHE["key"] == key:
        return _CACHE["result"]

    cli = plugin["cli"]
    timeout = float(plugin["timeout_s"] or DEFAULT_TIMEOUT_S)
    try:
        with tempfile.TemporaryDirectory(
                prefix=f"foilage_plugin_{plugin_id}_") as td:
            cfg_path = Path(td) / "config.json"
            out_path = Path(td) / "result.json"
            cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            argv = [sys.executable, cli,
                    "--config", str(cfg_path), "--output", str(out_path)]
            try:
                proc = subprocess.run(
                    argv, cwd=plugin["dir"], capture_output=True, text=True,
                    errors="replace", timeout=timeout,
                    creationflags=(subprocess.CREATE_NO_WINDOW
                                   if os.name == "nt" else 0))
            except subprocess.TimeoutExpired:
                raise ValueError(
                    f"geometry plugin '{plugin_id}' did not finish within "
                    f"{timeout:.0f} s - raise timeout_s in its manifest or "
                    "check the script for hangs")
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip()[-800:]
                raise ValueError(
                    f"geometry plugin '{plugin_id}' failed (exit code "
                    f"{proc.returncode})" + (f":\n{tail}" if tail else ""))
            if not out_path.is_file():
                raise ValueError(
                    f"geometry plugin '{plugin_id}' wrote no result.json - "
                    "the script must write its section to the --output path")
            try:
                result = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise ValueError(f"geometry plugin '{plugin_id}' produced "
                                 f"invalid JSON: {e}")
    except ValueError:
        raise
    except OSError as e:
        raise ValueError(f"could not run geometry plugin "
                         f"'{plugin_id}': {e}")

    if not isinstance(result, dict) or "ss" not in result or "ps" not in result:
        raise ValueError(
            f"geometry plugin '{plugin_id}' result must be a JSON object "
            "with 'ss' and 'ps' point arrays")
    blade_count = result.get("blade_count")
    if blade_count is not None:
        try:
            blade_count = int(blade_count)
        except (TypeError, ValueError):
            blade_count = None
    out = {"ss": _check_points(result["ss"], "ss"),
           "ps": _check_points(result["ps"], "ps"),
           "blade_count": blade_count}
    _CACHE["key"] = key
    _CACHE["result"] = out
    return out


def _mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


_CACHE = {"key": None, "result": None}
