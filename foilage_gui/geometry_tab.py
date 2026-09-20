"""Geometry tab: edit the airfoil/passage with sliders + exact entries.

The preview recomputes the airfoil (pyturbo-aero or an imported geomTurbo
section) and the periodic passage envelope whenever a geometry field
changes (debounced), on a worker thread - the GUI stays responsive while
splines rebuild. Below the blade view, two inspection charts show the
channel width (with the throat) and the surface curvature.

Panels adapt to the airfoil source: pyturbo shows the generator panels
(camberline, thickness, trailing edge, flow guidance); a geomTurbo
import shows the file/section fields plus the FFD morph cage instead.
The discretisation and domain panels apply to both sources. With the
morph enabled, the imported section is deformed by an N_morph x N_morph
cage of control points (dx/dy per point) and the cage is drawn over the
blade preview.
"""

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.airfoil import (build_geometry,                 # noqa: E402
                              _imported_section_unwrap)
from pipeline.cascade_metrics import (channel_widths, curvature, metal_angles,
                                       throat_metrics, true_chord,
                                       zweifel_geometric)
from pipeline.ffd import MAX_N, MIN_N, bernstein_basis  # noqa: E402
from pipeline.mesh_tris import pitch_profile, periodic_edges  # noqa: E402
from pipeline.plugins import (discover_plugins, type_choices)  # noqa: E402
from pipeline.unwrap import (cartesian_to_uy, periodicity_of,        # noqa: E402
                             uy_to_cartesian)

from .schema import field_from_dict, sections_for      # noqa: E402
from .widgets import INVALID_BG, FieldWidget, ScrolledFrame, Tooltip  # noqa: E402

DEBOUNCE_MS = 250
MORPH_TITLE = "geomTurbo morph (FFD cage)"
PLUGIN_KEY = "@plugin-generators"       # synthetic key for the plugin host
PLUGIN_PREFIX = "airfoil_source.params."
# sections that apply to every airfoil source (everything else is
# pyturbo-generator specific and hidden for a geomTurbo import)
SHARED_SECTIONS = {"Airfoil source", MORPH_TITLE, "Discretisation",
                   "Domain (periodic passage)"}


def _compute_geometry(cfg):
    """Worker: build the airfoil + passage edges + channel/curvature data.

    Returns (data dict) or ("error", message)."""
    try:
        airfoil = build_geometry(cfg)
        dom = cfg.get("domain", {})
        prof = pitch_profile(dom, airfoil)
        x0 = float(dom.get("x_min", -0.5))
        x1 = float(dom.get("x_max", 2.5))
        bot, top = periodic_edges(prof, x0, x1)
        outline = airfoil["outline"]
        if prof.get("periodic", True):
            offs = np.column_stack(
                [np.zeros(len(outline)), prof["p"](outline[:, 0])])
            ghost_p, ghost_m = outline + offs, outline - offs
        else:
            # freestream mode: there are no neighbouring blades
            ghost_p = ghost_m = None
        ss, ps = airfoil["ss"], airfoil["ps"]

        # ---- channel + throat (blade vs pitch-translated neighbor) ----
        # a freestream domain has no neighbor and no cascade throat
        pitch = prof["p"]
        if prof.get("periodic", True):
            s_ch, _j = channel_widths(ss, ps, pitch, airfoil["ss_upper"])
            throat = throat_metrics(ss, ps, pitch, airfoil["ss_upper"])
        else:
            s_ch, throat = None, None
        chord_cax = true_chord(ss, ps)
        # geometric Zweifel predictor (periodic modes): metal angles from
        # the section tangents + throat pitch over the axial chord (= 1)
        zw_geo = None
        a1_metal = a2_metal = None
        if throat is not None:
            a1_metal, a2_metal = metal_angles(ss, ps)
            zw_geo = zweifel_geometric(float(throat["pitch_throat"]),
                                       a1_metal, a2_metal)

        # ---- surface curvature ----
        k_ss, _s_ss = curvature(ss)
        k_ps, _s_ps = curvature(ps)

        # ---- imported geomTurbo reference overlay ----
        # the section is normalized to axial chord = 1 (same frame as the
        # generated blade) - raw file coordinates may be in any unit; in
        # axisymmetric mode it is unwrapped to uy like the active blade
        src = cfg.get("airfoil_source") or {}
        import_pts = None
        ref_err = None
        import_mtime = None
        if src.get("geomturbo_file"):
            try:
                from pipeline.geomturbo import (parse_geomturbo,
                                                section_to_airfoil)
                gt_path = Path(src["geomturbo_file"])
                parsed = parse_geomturbo(gt_path)
                secs = parsed["sections"]
                idx = int(src.get("section") or 0)
                sec = secs[max(0, min(idx, len(secs) - 1))]
                n_ref = int(cfg.get("airfoil", {}).get("n_points", 401))
                ref_af = section_to_airfoil(sec, n_points=n_ref)
                ref_ss, ref_ps = ref_af["ss"], ref_af["ps"]
                unwrap = _imported_section_unwrap(cfg)
                if unwrap is not None:
                    ref_ss, ref_ps = unwrap(ref_ss, ref_ps)
                import_pts = (ref_ss, ref_ps)
                import_mtime = gt_path.stat().st_mtime
            except Exception as e:
                ref_err = f"{type(e).__name__}: {e}"

        a1 = cfg.get("airfoil", {}).get("alpha1")
        a2 = cfg.get("airfoil", {}).get("alpha2")
        return {
            "ss": ss, "ps": ps, "outline": outline,
            "bot": bot, "top": top, "ghost_p": ghost_p, "ghost_m": ghost_m,
            "mode": prof.get("mode", "axisymmetric"),
            "periodic": prof.get("periodic", True),
            "style": airfoil["style"], "ss_upper": airfoil["ss_upper"],
            "n_points": len(ss), "x0": x0, "x1": x1,
            "turning": (float(a1) - float(a2))
            if None not in (a1, a2) and src.get("type", "pyturbo") == "pyturbo"
            else None,
            "p_le": prof["p_le"], "p_te": prof["p_te"],
            "true_chord_cax": chord_cax,
            "radius_throat_cax": float(prof["radius"](
                throat["x_over_cax"])) if throat else None,
            "pitch_to_chord": (throat["pitch_throat"] / max(chord_cax, 1e-30))
            if throat else None,
            "source": src.get("type", "pyturbo"),
            "blade_count": airfoil.get("blade_count"),
            "morph_cage": airfoil.get("morph_cage"),
            "import_pts": import_pts,
            "import_err": ref_err,
            "import_mtime": import_mtime,
            "show_reference": bool(src.get("show_reference", True)),
            "channel_x": ss[:, 0], "channel_s": s_ch,
            "k_ss": k_ss, "k_ps": k_ps,
            "throat": throat,
            "zweifel_geometric": zw_geo,
            "angle_metal_inlet_deg": a1_metal,
            "angle_metal_exit_deg": a2_metal,
        }
    except Exception as e:
        return ("error", f"{type(e).__name__}: {e}")


class GeometryTab(ttk.Frame):
    def __init__(self, app):
        super().__init__(app.notebook)
        self.app = app
        self.fields = {}
        self._pending = None
        self._busy = False
        self._dirty_while_busy = False
        self._results = queue.Queue()
        self._last_good = None
        self._ac_before_infer = None   # pyturbo axial chord, remembered

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ------------------------------------------------------------ form
        left = ttk.Frame(pane)
        pane.add(left, weight=1)
        form = ScrolledFrame(left)
        form.pack(fill="both", expand=True)
        self._section_frames = {}
        self._section_visible = {}
        self._morph_cells = {}
        self._morph_grid = None
        self._grid_n = 0
        self._morph_loading = False
        self._plugins = discover_plugins()
        self._plugin_warnings = [f"{p['name']}: {p['error']}"
                                 for p in self._plugins if p["error"]]
        self._active_plugin_id = None
        self._plugin_field_paths = []
        for section in sections_for("geometry"):
            box = ttk.LabelFrame(form.inner, text=f" {section.title} ")
            box.pack(fill="x", padx=6, pady=(8, 2))
            self._section_frames[section.title] = box
            self._section_visible[section.title] = True
            if section.title == MORPH_TITLE:
                self._build_morph_panel(box)
                continue
            for spec in section.fields:
                w = FieldWidget(box, spec, self._commit)
                self.fields[spec.path] = w
            if section.title == "Airfoil source":
                # the export button lives in its own row so the source
                # panel can be relaid out with the fields
                btn_row = ttk.Frame(box)
                ttk.Button(btn_row, text="Save as geomTurbo ...",
                           command=self._export_geomturbo).pack(
                    anchor="e", padx=4, pady=(2, 4))
                btn_row.pack(fill="x")
                self._source_extra = btn_row
                # host for the extension-plugin parameter panels; it sits
                # directly under the source selection like the morph panel
                host = ttk.Frame(form.inner)
                host.pack(fill="x")
                self._plugin_host = host
                self._section_frames[PLUGIN_KEY] = host
                self._section_visible[PLUGIN_KEY] = False
        # rows of the Airfoil source panel in display order; the second
        # item is the source each row belongs to (None = always shown)
        self._source_rows = [
            ("airfoil_source.type", None),
            ("airfoil_source.geomturbo_file", "geomturbo"),
            ("airfoil_source.section", "geomturbo"),
            ("airfoil_source.show_reference", "pyturbo"),
        ]
        # extension plugins join the source dropdown (labels + ids)
        choices = list(self.fields["airfoil_source.type"].spec.choices) \
            + type_choices(self._plugins)
        self.fields["airfoil_source.type"].set_choices(choices)

        hint = tk.Label(left, text="Sliders update the preview on release; "
                        "type exact values in the boxes and press Enter.",
                        font=("TkDefaultFont", 8), fg="#595959", wraplength=280,
                        justify="left")
        hint.pack(fill="x", padx=8, pady=4)

        # ------------------------------------------------- blade + charts
        right = ttk.Frame(pane)
        pane.add(right, weight=3)
        vpane = ttk.PanedWindow(right, orient="vertical")

        blade = ttk.Frame(vpane)
        vpane.add(blade, weight=3)
        self.fig = Figure(figsize=(7.2, 4.6), dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=blade)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, blade, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        self.metrics_var = tk.StringVar(value="")
        tk.Label(vpane, textvariable=self.metrics_var, anchor="w",
                 font=("Consolas", 8)).pack(fill="x")

        charts = ttk.Notebook(vpane)
        vpane.add(charts, weight=2)
        self._chart_tabs = {}
        for name in ("Channel", "Curvature"):
            tab = ttk.Frame(charts)
            fig = Figure(figsize=(7.2, 2.0), dpi=96)
            ax = fig.add_subplot(111)
            cvs = FigureCanvasTkAgg(fig, master=tab)
            cvs.get_tk_widget().pack(fill="both", expand=True)
            charts.add(tab, text=f" {name} ")
            self._chart_tabs[name] = (fig, ax, cvs)
        vpane.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.status_var, anchor="w",
                 font=("TkDefaultFont", 8), fg="#595959").pack(fill="x",
                                                            side="bottom")

        self.reload_fields()
        self._refresh_sections(commit=False)
        self._infer_from_geomturbo()
        self._update_source_panels()
        self.schedule_regeneration(immediate=True)

    # ------------------------------------------------- source panel layout
    def _plugin_label(self, plugin_id):
        for p in self._plugins:
            if p["id"] == plugin_id:
                return p["name"]
        return plugin_id

    def _update_source_panels(self):
        """Show/hide the source-specific panels: the pyturbo generator
        panels for the pyturbo source; the geomTurbo import fields, the
        FFD morph cage and the selected plugin's parameter panels for
        non-pyturbo sources. Discretisation and Domain apply to both and
        stay visible."""
        src = self.app.state.get("airfoil_source.type") or "pyturbo"
        is_plugin = src not in ("pyturbo", "geomturbo")
        if is_plugin:
            if self._active_plugin_id != src:
                self._rebuild_plugin_panel(src)
        elif self._active_plugin_id is not None:
            self._rebuild_plugin_panel(None)     # tear the panel down
        for title in self._section_frames:
            if title == MORPH_TITLE:
                self._section_visible[title] = (src != "pyturbo")
            elif title == PLUGIN_KEY:
                self._section_visible[title] = is_plugin
            elif title in SHARED_SECTIONS:
                continue
            else:
                self._section_visible[title] = (src == "pyturbo")
        # forget everything and repack in schema order so the visible
        # panels keep their original stacking
        for title, box in self._section_frames.items():
            box.pack_forget()
            if self._section_visible[title]:
                box.pack(fill="x", padx=6, pady=(8, 2))
        box = self._section_frames["Airfoil source"]
        for path, when in self._source_rows:
            self.fields[path].frame.pack_forget()
        self._source_extra.pack_forget()
        for path, when in self._source_rows:
            if when is None or when == src:
                self.fields[path].frame.pack(fill="x", padx=4, pady=1)
        self._source_extra.pack(fill="x")

    # ------------------------------------------------------ plugin panels
    def _rebuild_plugin_panel(self, plugin_id):
        """(Re)build the parameter panels of the selected extension
        plugin from its manifest; plugin_id None tears the panel down."""
        self._active_plugin_id = plugin_id
        for path in self._plugin_field_paths:
            self.fields.pop(path, None)
        self._plugin_field_paths = []
        for child in self._plugin_host.winfo_children():
            child.destroy()
        if plugin_id is None:
            return
        plugin = next((p for p in self._plugins if p["id"] == plugin_id),
                      None)
        if plugin is None:
            tk.Label(self._plugin_host,
                     text=f"unknown geometry plugin '{plugin_id}' - check "
                          "the extensions directory",
                     fg="#a00", wraplength=280,
                     justify="left").pack(fill="x", padx=4, pady=4)
            return
        if plugin["error"]:
            tk.Label(self._plugin_host,
                     text=f"plugin '{plugin['name']}' cannot be used: "
                          f"{plugin['error']}",
                     fg="#a00", wraplength=280,
                     justify="left").pack(fill="x", padx=4, pady=4)
            return
        # seed declared defaults so the CLI always receives full values
        for spec in plugin["parameters"]:
            full = PLUGIN_PREFIX + spec["path"]
            if self.app.state.get(full) is None \
                    and spec.get("default") is not None:
                self.app.state.set(full, spec["default"], notify=False)
        for group in plugin["groups"]:
            if not group["params"]:
                continue
            gbox = ttk.LabelFrame(self._plugin_host,
                                  text=f" {group['title']} ")
            gbox.pack(fill="x", pady=(4, 2))
            for spec_dict in group["params"]:
                spec = field_from_dict(spec_dict, prefix=PLUGIN_PREFIX)
                w = FieldWidget(gbox, spec, self._plugin_commit)
                self.fields[spec.path] = w
                self._plugin_field_paths.append(spec.path)
                w.set_value(self.app.state.get(spec.path))
        if plugin["description"]:
            tk.Label(self._plugin_host, text=plugin["description"],
                     justify="left", wraplength=280,
                     font=("TkDefaultFont", 8),
                     fg="#595959").pack(fill="x", padx=4, pady=(2, 4))

    def _plugin_commit(self, spec, value):
        """A plugin parameter changed: store it; the state notification
        triggers a preview regeneration, which re-runs the plugin CLI."""
        self.app.state.set(spec.path, value)
        self.status_var.set(f"{self._plugin_label(self._active_plugin_id)}: "
                            f"{spec.label} = {value} - regenerating ...")

    # ------------------------------------------------------------ morph UI
    def _build_morph_panel(self, box):
        """Controls for the FFD morph cage: enable, N_morph, reset, and
        the N_morph x N_morph grid of dx/dy entries laid out like the
        cage itself (u across, v upwards)."""
        m = self.app.state.get("airfoil_source.morph") or {}
        bar = ttk.Frame(box)
        bar.pack(fill="x", padx=4, pady=(2, 2))
        self._morph_enabled_var = tk.BooleanVar(value=bool(m.get("enabled")))
        cb = ttk.Checkbutton(bar, text="Enable morphing",
                             variable=self._morph_enabled_var,
                             command=self._morph_enabled_commit)
        cb.pack(side="left")
        Tooltip(cb, "Deform the imported section with the control cage "
                    "below before meshing. The preview, the mesh and any "
                    "optimization evaluations all use the morphed shape "
                    "(stored in input.json).")
        tk.Label(bar, text="N_morph:",
                 font=("TkDefaultFont", 8)).pack(side="left", padx=(12, 2))
        self._morph_n_var = tk.IntVar(
            value=max(MIN_N, min(MAX_N, int(m.get("n") or 4))))
        spin = ttk.Spinbox(bar, from_=MIN_N, to=MAX_N, width=3,
                           textvariable=self._morph_n_var,
                           command=self._morph_n_commit)
        spin.pack(side="left")
        spin.bind("<Return>", self._morph_n_commit)
        spin.bind("<FocusOut>", self._morph_n_commit)
        Tooltip(spin, "The cage is an N_morph x N_morph grid of control "
                      "points around the section; changing it resamples "
                      "the current deformation onto the new lattice.")
        ttk.Button(bar, text="Reset offsets",
                   command=self._morph_reset).pack(side="left", padx=(12, 0))

        self._morph_grid_host = ttk.Frame(box)
        self._morph_grid_host.pack(fill="x", padx=4, pady=(0, 2))
        self._rebuild_morph_grid()
        note = tk.Label(
            box, justify="left", wraplength=280, font=("TkDefaultFont", 8),
            fg="#595959",
            text="Each cell is one cage control point (top: dx, bottom: "
                 "dy, in axial-chord units; drag-free - type and press "
                 "Enter). Equal offsets translate the blade, corner "
                 "offsets rotate/shear, symmetric offsets stretch, single "
                 "points make local bumps. The cage box hugs the section "
                 "bounding box (+8% margin).")
        note.pack(fill="x", padx=4, pady=(0, 3))

    def _rebuild_morph_grid(self):
        """Recreate the cage entry grid for the current N_morph."""
        if self._morph_grid is not None:
            self._morph_grid.destroy()
        n = max(MIN_N, min(MAX_N, int(self._morph_n_var.get() or 4)))
        self._grid_n = n
        grid = ttk.Frame(self._morph_grid_host)
        grid.pack(fill="x")
        self._morph_grid = grid
        self._morph_cells = {}
        dx_vals, dy_vals = self._state_offsets(n)
        width = 6 if n <= 4 else (5 if n <= 6 else 4)
        small = ("TkDefaultFont", 7)
        tk.Label(grid, text="dx / dy", font=small, fg="#737373").grid(
            row=0, column=0, padx=(0, 2))
        for i in range(n):
            tk.Label(grid, text=f"u{i}", font=small,
                     fg="#737373").grid(row=0, column=1 + i, padx=2)
        for r in range(n):
            j = n - 1 - r                     # top row of the grid = v max
            tk.Label(grid, text=f"v{j}", font=small, fg="#737373").grid(
                row=1 + r, column=0, sticky="e", padx=(0, 2))
            for i in range(n):
                cell = ttk.Frame(grid)
                cell.grid(row=1 + r, column=1 + i, padx=1, pady=1)
                entries = []
                for val in (dx_vals[i * n + j], dy_vals[i * n + j]):
                    e = tk.Entry(cell, width=width, font=("Consolas", 8),
                                 justify="right", relief="solid", bd=1)
                    e.insert(0, FieldWidget._fmt(val))
                    e.pack(side="top", anchor="w")
                    e.bind("<Return>", self._morph_cell_commit)
                    e.bind("<FocusOut>", self._morph_cell_commit)
                    Tooltip(e, f"cage point (u{i}, v{j}): offset in "
                               f"{'x' if len(entries) == 0 else 'y'} "
                               "[axial chord]")
                    entries.append(e)
                self._morph_cells[(i, j)] = tuple(entries)

    def _state_offsets(self, n):
        """dx/dy flat lists from the case state, normalized to n*n values
        (resampled from an older grid size, or padded/truncated)."""
        m = self.app.state.get("airfoil_source.morph") or {}
        return (self._coerce_offsets(m.get("dx"), n),
                self._coerce_offsets(m.get("dy"), n))

    @staticmethod
    def _resample_offsets(old_n, flat, new_n):
        """Evaluate the deformation field of an old_n x old_n cage at the
        control points of a new_n x new_n lattice."""
        us = np.linspace(0.0, 1.0, new_n)
        uu, vv = np.meshgrid(us, us, indexing="ij")
        bu = bernstein_basis(old_n, uu.ravel())
        bv = bernstein_basis(old_n, vv.ravel())
        mat = np.asarray(flat, dtype=float).reshape(old_n, old_n)
        return np.einsum("ki,ij,kj->k", bu, mat, bv).tolist()

    def _coerce_offsets(self, vals, n):
        vals = [float(v) for v in (vals or [])]
        if len(vals) == n * n:
            return vals
        old_n = int(round(len(vals) ** 0.5)) if vals else 0
        if MIN_N <= old_n <= MAX_N and old_n * old_n == len(vals):
            return self._resample_offsets(old_n, vals, n)
        return (vals + [0.0] * (n * n))[:n * n]

    def _morph_cell_commit(self, _e=None):
        """Read the whole cage grid, store it in the case state and
        regenerate the preview. Invalid entries are flagged red."""
        if self._morph_loading:
            return
        n = self._grid_n
        dx, dy, bad = [], [], []
        for (i, j) in sorted(self._morph_cells):
            for e, acc in zip(self._morph_cells[(i, j)], (dx, dy)):
                text = e.get().strip()
                try:
                    acc.append(float(text.replace(",", ".")) if text
                               else 0.0)
                    e.configure(background="white")
                except ValueError:
                    bad.append(e)
                    acc.append(0.0)
        for e in bad:
            e.configure(background=INVALID_BG)
        if bad:
            return
        self.app.state.set("airfoil_source.morph.dx", dx)
        self.app.state.set("airfoil_source.morph.dy", dy)
        moved = sum(1 for v in dx + dy if abs(v) > 1e-12)
        self.status_var.set(f"morph cage: {moved} offset(s) set, "
                            "rebuilding ...")

    def _morph_n_commit(self, _e=None):
        if self._morph_loading:
            return
        try:
            n = int(self._morph_n_var.get())
        except (ValueError, tk.TclError):
            self._morph_n_var.set(self._grid_n)
            return
        n = max(MIN_N, min(MAX_N, n))
        self._morph_n_var.set(n)
        if n == self._grid_n:
            return
        # keep the current deformation: sample its field at the new
        # control points (entries with invalid text fall back to state)
        old_n = self._grid_n
        try:
            dx_old, dy_old = self._read_entries_flat(old_n)
        except ValueError:
            dx_old, dy_old = self._state_offsets(old_n)
        self.app.state.set("airfoil_source.morph.n", n)
        self.app.state.set("airfoil_source.morph.dx",
                           self._resample_offsets(old_n, dx_old, n))
        self.app.state.set("airfoil_source.morph.dy",
                           self._resample_offsets(old_n, dy_old, n))
        self._rebuild_morph_grid()

    def _read_entries_flat(self, n):
        dx, dy = [], []
        for (i, j) in sorted(self._morph_cells):
            ex, ey = self._morph_cells[(i, j)]
            for e, acc in ((ex, dx), (ey, dy)):
                text = e.get().strip()
                acc.append(float(text.replace(",", ".")) if text else 0.0)
        if len(dx) != n * n:
            raise ValueError("entry grid does not match N_morph")
        return dx, dy

    def _morph_enabled_commit(self):
        if self._morph_loading:
            return
        self.app.state.set("airfoil_source.morph.enabled",
                           bool(self._morph_enabled_var.get()))
        self.status_var.set(
            "morphing enabled - the imported section is deformed by the "
            "cage" if self._morph_enabled_var.get()
            else "morphing disabled - imported section used as-is")

    def _morph_reset(self):
        if self._morph_loading:
            return
        n = self._grid_n
        self.app.state.set("airfoil_source.morph.dx", [0.0] * (n * n))
        self.app.state.set("airfoil_source.morph.dy", [0.0] * (n * n))
        self._rebuild_morph_grid()
        self.status_var.set("morph cage offsets reset to zero")

    def _morph_load_from_state(self):
        """Sync the morph panel with the case state (config reload)."""
        self._morph_loading = True
        try:
            m = self.app.state.get("airfoil_source.morph") or {}
            n = max(MIN_N, min(MAX_N, int(m.get("n") or 4)))
            self._morph_n_var.set(n)
            self._morph_enabled_var.set(bool(m.get("enabled")))
            if self._grid_n != n:
                self._rebuild_morph_grid()      # fills from the state
            else:
                dx_vals, dy_vals = self._state_offsets(n)
                for (i, j), (ex, ey) in self._morph_cells.items():
                    ex.delete(0, "end")
                    ex.insert(0, FieldWidget._fmt(dx_vals[i * n + j]))
                    ey.delete(0, "end")
                    ey.insert(0, FieldWidget._fmt(dy_vals[i * n + j]))
                    ex.configure(background="white")
                    ey.configure(background="white")
        finally:
            self._morph_loading = False

    # ------------------------------------------------------------- fields
    def _commit(self, spec, value):
        self.app.state.set(spec.path, value)
        if spec.path == "airfoil_source.geomturbo_file":
            self._refresh_sections(commit=True)
            self._infer_from_geomturbo()
        elif spec.path == "airfoil_source.type":
            self._refresh_sections(commit=False)
            self._update_source_panels()
            if value == "geomturbo":
                self._infer_from_geomturbo()
            elif value == "pyturbo" and self._ac_before_infer is not None:
                # switching back to the generator: restore its own axial
                # chord (the geomTurbo-inferred value would wreck the
                # pyturbo thickness proportions)
                self.app.state.set("airfoil.axial_chord",
                                   self._ac_before_infer)
                self._ac_before_infer = None
                self.reload_fields()
        elif spec.path == "airfoil_source.section":
            self._infer_from_geomturbo()
        elif spec.path == "airfoil_source.show_reference":
            if self._last_good:                     # overlay-only change
                self._draw(self._last_good)

    def _infer_from_geomturbo(self):
        """Populate axial chord / blade count / R1 / R2 from the imported
        geomTurbo (only while the geomTurbo source is active).

        Inferred values are converted from file units into the units the
        case currently uses (mm when axial_chord >= 1, else meters). The
        previous axial chord is remembered so switching back to pyturbo
        can restore it.
        """
        if self.app.state.get("airfoil_source.type") != "geomturbo":
            return
        path = self.app.state.get("airfoil_source.geomturbo_file")
        if not path:
            return
        try:
            from pipeline.geomturbo import (infer_cascade_parameters,
                                            parse_geomturbo)
            parsed = parse_geomturbo(path)
            inf = infer_cascade_parameters(
                parsed, self.app.state.get("airfoil_source.section") or 0)
        except Exception as e:
            self.status_var.set(f"geomTurbo inference failed: {e}")
            return
        if not inf:
            return
        cur_ac = float(self.app.state.get("airfoil.axial_chord") or 1.0)
        # file units -> the case's current mm/m convention
        to_current = (1000.0 if cur_ac >= 1.0 else 1.0) \
            * (parsed.get("units") or 1.0)
        if self._ac_before_infer is None:
            self._ac_before_infer = cur_ac     # remember the generator value
        updates = [("airfoil.axial_chord", inf["axial_chord"] * to_current),
                   ("domain.airfoil_count", inf["blade_count"])]
        if inf["R1"] is not None and inf["R1"] > 0.0:
            # a 2D section with its LE exactly on the axis infers R1 = 0,
            # which would collapse the passage - keep the current value
            updates.append(("domain.R1", inf["R1"] * to_current))
        if inf["R2"] is not None and inf["R2"] > 0.0:
            updates.append(("domain.R2", inf["R2"] * to_current))
        applied = []
        for path_key, value in updates:
            if value is not None and self.app.state.set(path_key, value):
                applied.append(f"{path_key.split('.')[-1]} = {value:.6g}")
        self.reload_fields()                   # sync every panel display
        if applied:
            self.status_var.set("inferred from geomTurbo: "
                                + ", ".join(applied))
        else:
            self.status_var.set("geomTurbo parameters already up to date")

    def _export_geomturbo(self):
        """Save the active blade section (SS/PS) as a .geomTurbo file.

        Coordinates are exported in meters: the normalized preview arrays
        are scaled by the physical axial chord. In axisymmetric mode the
        preview lives in unwrapped y (arc length), while geomTurbo files
        hold Cartesian y - the section is re-wrapped (y = R sin(uy / R))
        before writing. Offset/freestream modes have nothing to unwrap.
        The spanwise Z of each point is computed by the writer from the
        annulus radius and the Cartesian y (z = sqrt(R^2 - y^2)), and the
        rows carry NUMECA's inverted-Y convention (Z -Y X)."""
        from tkinter import filedialog, messagebox
        g = self._last_good
        if g is None:
            messagebox.showinfo(
                "Save geomTurbo",
                "The geometry has not been built yet - wait for the "
                "preview to finish and try again.")
            return
        from pipeline.geomturbo import write_geomturbo
        dom = self.app.state.config.get("domain", {})
        ac = float(self.app.state.get("airfoil.axial_chord") or 1.0)
        scale = self.app.state.scale_m_per_chord()      # m per chord unit
        units_to_m = 0.001 if ac >= 1.0 else 1.0
        ss_out, ps_out = g["ss"], g["ps"]
        r1_m = r2_m = None
        r1 = dom.get("R1")
        if r1 and ac > 0 and dom.get("periodicity", "axisymmetric") \
                != "freestream":
            # radii in the exported unit (meters) for the writer's Z
            r1_m = float(r1) * units_to_m
            r2_m = float(dom.get("R2") or r1) * units_to_m
        if g.get("periodic", True) and \
                periodicity_of(dom) == "axisymmetric":
            ss_out, ps_out = uy_to_cartesian(
                ss_out, ps_out,
                float(r1) / ac if r1 and ac > 0 else None)
        path = filedialog.asksaveasfilename(
            title="Save blade section as geomTurbo",
            defaultextension=".geomTurbo",
            initialfile=f"{self.app.state.case_name()}.geomTurbo",
            filetypes=[("geomTurbo", "*.geomTurbo"), ("All files", "*.*")])
        if not path:
            return
        write_geomturbo(path, ss_out * scale, ps_out * scale,
                        r1=r1_m, r2=r2_m, units="m")
        # remember the file so switching the source to geomTurbo picks it up
        self.app.state.set("airfoil_source.geomturbo_file", path,
                           notify=False)
        self._refresh_sections(commit=False)
        self.status_var.set(f"saved geomTurbo: {path}")

    def _refresh_sections(self, commit=True):
        """Repopulate the Section dropdown from the geomTurbo file."""
        widget = self.fields.get("airfoil_source.section")
        path = self.app.state.get("airfoil_source.geomturbo_file")
        pairs = []
        err = None
        if path:
            try:
                from pipeline.geomturbo import parse_geomturbo
                parsed = parse_geomturbo(path)
                secs = parsed["sections"]
                pairs = [(f"Z = {s['z']:g}  [#{i}]", i)
                         for i, s in enumerate(secs)]
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        widget.set_choices(pairs)
        if err:
            self.status_var.set(f"geomTurbo: {err}")
        elif commit and pairs:
            # keep the chosen section in sync and rebuild
            self.app.state.set("airfoil_source.section",
                               widget.get_value()[1])

    def reload_fields(self):
        for path, widget in self.fields.items():
            widget.set_value(self.app.state.get(path))
        self._morph_load_from_state()

    def on_state_changed(self, paths):
        if any(p.startswith(("airfoil.", "domain.", "airfoil_source."))
               for p in paths):
            self.schedule_regeneration()

    # ------------------------------------------------------ regeneration
    def schedule_regeneration(self, immediate=False):
        if self._busy:
            # an edit arrived while a build is running: rebuild once more
            self._dirty_while_busy = True
            return
        if self._pending:
            self.app.root.after_cancel(self._pending)
        delay = 0 if immediate else DEBOUNCE_MS
        self._pending = self.app.root.after(delay, self._regenerate)

    def _regenerate(self):
        self._pending = None
        if self._busy:
            self._dirty_while_busy = True
            return
        self._busy = True
        self.status_var.set("rebuilding geometry ...")
        cfg = {
            "airfoil": self.app.state.config["airfoil"],
            "domain": self.app.state.config.get("domain", {}),
            "airfoil_source": self.app.state.config.get("airfoil_source", {}),
        }
        threading.Thread(target=self._worker, args=(cfg,), daemon=True,
                         name="geometry-build").start()
        self.app.root.after(60, self._poll)

    def _worker(self, cfg):
        self._results.put(_compute_geometry(cfg))

    def _poll(self):
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            self.app.root.after(60, self._poll)
            return
        self._busy = False
        if isinstance(result, tuple) and result and result[0] == "error":
            self.status_var.set(f"geometry error: {result[1]}")
        else:
            self._last_good = result
            self._draw(result)
            self._draw_channel(result)
            self._draw_curvature(result)
            self._update_metrics(result)
            self.status_var.set(self._status_text(result))
        # if edits arrived while computing, run once more with the latest
        if self._dirty_while_busy:
            self._dirty_while_busy = False
            self.schedule_regeneration(immediate=True)

    def _status_text(self, g):
        parts = [f"{g['style'].upper()} profile"]
        if g["source"] not in ("pyturbo", "geomturbo"):
            parts.append(f"plugin: {self._plugin_label(g['source'])}")
        if g["turning"] is not None:
            parts.append(f"turning {g['turning']:.1f} deg")
        parts.append(f"pitch LE {g['p_le']:.4f}, TE {g['p_te']:.4f}")
        if g.get("pitch_to_chord") is not None:
            parts.append(f"PTC {g['pitch_to_chord']:.4f}")
        parts.append({True: "periodic", False: "freestream"}[
            g.get("periodic", True)])
        parts.append(f"{g['n_points']} points/side")
        if g.get("blade_count") is not None:
            parts.append(f"{g['blade_count']} blades")
        if g.get("morph_cage"):
            parts.append(f"FFD-morphed ({g['morph_cage']['n']}x"
                         f"{g['morph_cage']['n']} cage)")
        if g["import_err"]:
            parts.append(f"geomTurbo reference failed ({g['import_err']})")
        if self._plugin_warnings:
            parts.append(f"{len(self._plugin_warnings)} extension(s) "
                         "skipped (bad manifest)")
        return "  |  ".join(parts)

    def _update_metrics(self, g):
        scale_mm = self.app.state.scale_m_per_chord() * 1000.0
        if g.get("throat") is None:
            # freestream mode: no pitch-translated neighbor, no throat
            self.metrics_var.set(
                f"true chord {g['true_chord_cax']:.4f} c_ax "
                f"({g['true_chord_cax'] * scale_mm:.2f} mm)   |   "
                "freestream mode - cascade throat metrics not applicable")
            return
        t = g["throat"]
        self.metrics_var.set(
            f"throat {t['width']:.4f} c_ax ({t['width'] * scale_mm:.2f} mm) "
            f"@ x/c_ax {t['x_over_cax']:.3f}   |   "
            f"true chord {g['true_chord_cax']:.4f} c_ax   |   "
            f"PTC {g['pitch_to_chord']:.4f}   |   "
            f"Zw_geo {g.get('zweifel_geometric', 0):.3f} "
            f"(metal {g.get('angle_metal_inlet_deg', 0):.1f}/"
            f"{g.get('angle_metal_exit_deg', 0):.1f} deg)   |   "
            f"\u03b1_throat {t['angle_throat_deg']:.1f}\u00b0   |   "
            f"unguided {t['unguided_turning_deg']:.1f}\u00b0")

    # -------------------------------------------------------------- draw
    def _draw(self, g):
        ax = self.ax
        ax.clear()
        outline = g["outline"]

        # passage envelope + ghost blades for context
        edge_lbl = "periodic edges" if g.get("periodic", True) \
            else "far-field boundaries"
        ax.plot(g["bot"][:, 0], g["bot"][:, 1], "--", color="0.55", lw=1,
                label=edge_lbl)
        ax.plot(g["top"][:, 0], g["top"][:, 1], "--", color="0.55", lw=1)
        if g.get("ghost_p") is not None:
            for ghost in (g["ghost_p"], g["ghost_m"]):
                closed = np.vstack([ghost, ghost[0]])
                ax.plot(closed[:, 0], closed[:, 1], color="0.8", lw=1)

        # geomTurbo reference under the active blade
        if g["source"] == "pyturbo" and g["show_reference"] \
                and g["import_pts"] is not None:
            ss_ref, ps_ref = g["import_pts"]
            stamp = ""
            if g.get("import_mtime"):
                stamp = time.strftime(" (%H:%M:%S)",
                                      time.localtime(g["import_mtime"]))
            lbl = f"geomTurbo reference{stamp}"
            ax.plot(ss_ref[:, 0], ss_ref[:, 1], color="0.45", lw=1.2,
                    ls="--", zorder=3)
            ax.plot(ps_ref[:, 0], ps_ref[:, 1], color="0.45", lw=1.2,
                    ls="--", label=lbl, zorder=3)

        # FFD morph cage: dotted = undeformed lattice, orange = deformed
        if g.get("morph_cage"):
            self._draw_cage(ax, g["morph_cage"])

        closed = np.vstack([outline, outline[0]])
        ax.fill(closed[:, 0], closed[:, 1], color="#dbe9f6", zorder=4)
        ax.plot(closed[:, 0], closed[:, 1], color="#1f4e79", lw=1.6, zorder=5)

        ss_lbl = f"suction side ({'upper' if g['ss_upper'] else 'lower'})"
        ps_lbl = f"pressure side ({'lower' if g['ss_upper'] else 'upper'})"
        ax.plot(g["ss"][:, 0], g["ss"][:, 1], color="tab:red", lw=1, alpha=0.7,
                label=ss_lbl, zorder=6)
        ax.plot(g["ps"][:, 0], g["ps"][:, 1], color="tab:blue", lw=1,
                alpha=0.7, label=ps_lbl, zorder=6)
        ax.plot(g["ss"][0, 0], g["ss"][0, 1], "k.", ms=7, zorder=7)
        ax.annotate("LE", g["ss"][0], textcoords="offset points",
                    xytext=(8, 8), fontsize=8)
        ax.annotate("TE", g["ss"][-1], textcoords="offset points",
                    xytext=(8, 8), fontsize=8)

        # throat marker: the narrowest point of the channel
        t = g["throat"]
        if t and t["x_over_cax"] is not None:
            ss, ps = g["ss"], g["ps"]
            i = int(round(t["u"] * (len(ss) - 1)))
            ax.plot(ss[i, 0], ss[i, 1], "o", ms=5, color="tab:green",
                    zorder=8)
            ax.annotate("throat", ss[i], textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color="tab:green")

        ax.set_aspect("equal")
        pad = 0.15
        ax.set_xlim(g["x0"] - pad, g["x1"] + pad)
        ymin = min(g["bot"][:, 1].min(), outline[:, 1].min())
        ymax = max(g["top"][:, 1].max(), outline[:, 1].max())
        ax.set_ylim(ymin - 0.1, ymax + 0.1)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x / axial chord")
        ax.set_ylabel("y / axial chord")
        ax.set_title("Turbine cascade passage (normalized: axial chord = 1)",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_cage(self, ax, cage):
        """Overlay the morph cage: dotted gray = undeformed lattice,
        orange = control points displaced by their (dx, dy)."""
        n = cage["n"]
        cp0 = np.asarray(cage["cp0"])
        cp1 = np.asarray(cage["cp"])
        rows0 = cp0.reshape(n, n, 2)
        for r in range(n):
            ax.plot(rows0[r][:, 0], rows0[r][:, 1], color="0.75", lw=0.7,
                    ls=":", zorder=2)
            ax.plot(rows0[:, r][:, 0], rows0[:, r][:, 1], color="0.75",
                    lw=0.7, ls=":", zorder=2)
        label = "FFD morph cage"
        rows1 = cp1.reshape(n, n, 2)
        for r in range(n):
            ax.plot(rows1[r][:, 0], rows1[r][:, 1], color="#e6821e",
                    lw=0.9, alpha=0.85, zorder=3,
                    label=label if r == 0 else None)
            ax.plot(rows1[:, r][:, 0], rows1[:, r][:, 1], color="#e6821e",
                    lw=0.9, alpha=0.85, zorder=3)
        ax.plot(cp1[:, 0], cp1[:, 1], "o", ms=3.5, color="#e6821e",
                zorder=3)

    def _draw_channel(self, g):
        fig, ax, cvs = self._chart_tabs["Channel"]
        ax.clear()
        if g.get("channel_s") is None:
            ax.text(0.5, 0.5, "no cascade channel in freestream mode",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    color="0.4")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            cvs.draw_idle()
            return
        ax.plot(g["channel_x"], g["channel_s"], color="#1f4e79", lw=1.4)
        i = int(np.argmin(g["channel_s"]))
        w = float(g["channel_s"][i])
        ax.plot([g["channel_x"][i]], [w], "o", ms=5, color="tab:green")
        ax.annotate(f"throat {w:.4f} c_ax", (g["channel_x"][i], w),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8, color="tab:green")
        ax.set_title("channel width (blade to pitch-translated neighbor)",
                     fontsize=9, loc="left")
        ax.set_xlabel("x / axial chord (suction side)")
        ax.set_ylabel("channel width [c_ax]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        cvs.draw_idle()

    def _draw_curvature(self, g):
        fig, ax, cvs = self._chart_tabs["Curvature"]
        ax.clear()
        u_ss = np.linspace(0.0, 1.0, len(g["k_ss"]))
        u_ps = np.linspace(0.0, 1.0, len(g["k_ps"]))
        ax.plot(u_ss, g["k_ss"], color="tab:red", lw=1.2, label="suction side")
        ax.plot(u_ps, g["k_ps"], color="tab:blue", lw=1.2,
                label="pressure side")
        allk = np.concatenate([g["k_ss"], g["k_ps"]])
        ymax = float(np.percentile(allk, 98)) * 1.5
        ax.set_ylim(-0.02 * max(ymax, 1.0), max(ymax, 1.0))
        ax.set_title("surface curvature |k| (LE peak at u = 0; y clipped to "
                     "readable range)", fontsize=9, loc="left")
        ax.set_xlabel("u (LE -> TE)")
        ax.set_ylabel("|k| [1/c_ax]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        cvs.draw_idle()
