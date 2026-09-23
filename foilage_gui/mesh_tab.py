"""Mesh tab: interactive mesh viewer (zoom/pan) + mesh generation."""

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from .su2_mesh import read_su2_mesh, mesh_edges  # noqa: E402
from .schema import sections_for                 # noqa: E402
from .mesh_profiles import (PROTECTED, MeshProfiles, apply_values,  # noqa: E402
                            capture_values)
from .widgets import FieldWidget, ScrolledFrame  # noqa: E402

PROFILE_CUSTOM = "(custom)"     # dropdown placeholder: values not from a profile

MARKER_COLORS = {"airfoil": "#111111", "inlet": "#1f77b4",
                 "outlet": "#d62728", "periodic_bottom": "#2ca02c",
                 "periodic_top": "#2ca02c", "farfield": "#9467bd",
                 "hub": "#8c564b", "shroud": "#e377c2"}


class MeshTab(ttk.Frame):
    def __init__(self, app):
        super().__init__(app.notebook)
        self.app = app
        self._load_queue = queue.Queue()
        self._current = None            # parsed mesh dict
        self._current_path = None
        self.profiles = MeshProfiles()

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=(4, 0))

        self.gen_btn = tk.Button(bar, text="Generate mesh",
                                 command=lambda: app.run_job("mesh"),
                                 bg="#dff0d8")
        self.gen_btn.pack(side="left", padx=2)
        TooltipHelper(bar, "Run pipeline/run_pipeline.py on the current "
                           "input.json (airfoil + Gmsh mesh + quality gate).")

        ttk.Label(bar, text="source:").pack(side="left", padx=(10, 2))
        self.source_var = tk.StringVar()
        self.source_box = ttk.Combobox(bar, textvariable=self.source_var,
                                       state="readonly", width=42)
        self.source_box.pack(side="left")
        self.source_box.bind("<<ComboboxSelected>>",
                             lambda _e: self.load_selected())
        ttk.Button(bar, text="Refresh", width=7, command=self.refresh_sources
                   ).pack(side="left", padx=4)

        self.view3d_btn = ttk.Button(bar, text="3D outer view", width=12,
                                     command=self._open_3d_view)
        self.view3d_btn.pack(side="left", padx=4)
        TooltipHelper(self.view3d_btn,
                      "Open a GPU 3D view (moderngl) of the wedge mesh's "
                      "outer surface: hub/shroud streamtube walls, periodic "
                      "faces, blade and inlet/outlet, with translucent "
                      "shells so the blade is visible. Uses the 3D wedge "
                      "mesh (mesh_quad3d.su2) or the 3D solver mesh.")

        self.marker_vars = {}
        for name in MARKER_COLORS:
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(bar, text=name, variable=var,
                                font=("TkDefaultFont", 8),
                                fg=MARKER_COLORS[name],
                                command=self._redraw)
            cb.pack(side="left", padx=(8, 0))
            self.marker_vars[name] = var

        self.info_var = tk.StringVar(value="no mesh loaded")
        ttk.Label(self, textvariable=self.info_var, anchor="w",
                  font=("Consolas", 8)).pack(fill="x", padx=6)

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # --------------------------------------------- mesh parameter form
        form_holder = ttk.Frame(pane)
        pane.add(form_holder, weight=1)
        self.fields = {}
        self._build_profile_bar(form_holder)
        form = ScrolledFrame(form_holder)
        form.pack(fill="both", expand=True)
        for section in sections_for("mesh"):
            box = ttk.LabelFrame(form.inner, text=f" {section.title} ")
            box.pack(fill="x", padx=6, pady=(8, 2))
            for spec in section.fields:
                w = FieldWidget(box, spec, self._commit)
                self.fields[spec.path] = w
        self.reload_fields()

        # --------------------------------------------- viewer + details
        viewer = ttk.Frame(pane)
        pane.add(viewer, weight=4)
        vpane = ttk.PanedWindow(viewer, orient="vertical")

        left = ttk.Frame(vpane)
        vpane.add(left, weight=4)
        self.fig = Figure(figsize=(8.6, 6.0), dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, left, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        right = ttk.Frame(vpane)
        vpane.add(right, weight=1)
        tk.Label(right, text="mesh details", font=("TkDefaultFont", 9,
                 "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        from tkinter.scrolledtext import ScrolledText
        self.stats = ScrolledText(right, font=("Consolas", 8), width=44,
                                  state="disabled", wrap="none", height=12)
        self.stats.pack(fill="both", expand=True, padx=4, pady=(0, 6))
        vpane.pack(fill="both", expand=True)

        self.refresh_sources()

    # ------------------------------------------------------------- fields
    def _commit(self, spec, value):
        self.app.state.set(spec.path, value)

    def reload_fields(self):
        for path, widget in self.fields.items():
            widget.set_value(self.app.state.get(path))
        # a (re)loaded input.json is not necessarily on a profile any more
        self.profile_var.set(PROFILE_CUSTOM)
        self._refresh_profile_list()

    # ----------------------------------------------------------- profiles
    def _build_profile_bar(self, parent):
        """Profile drop-down + save/delete, above the parameter form."""
        bar = ttk.LabelFrame(parent, text=" Mesh profile ")
        bar.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(bar, text="profile:").pack(side="left", padx=(4, 2))
        self.profile_var = tk.StringVar(value=PROFILE_CUSTOM)
        self.profile_box = ttk.Combobox(bar, textvariable=self.profile_var,
                                        state="readonly", width=22)
        self.profile_box.pack(side="left", padx=(0, 4))
        self.profile_box.bind("<<ComboboxSelected>>",
                              lambda _e: self._apply_selected_profile())
        TooltipHelper(self.profile_box,
                      "Mesh profiles stored in mesh_profiles.json. "
                      "Choosing one applies all of its mesh-tab "
                      "parameters to the current case (input.json is "
                      "only written when you save the case or run a "
                      "job). '(custom)' means the current values are "
                      "not from a profile.")
        self.profile_save_btn = ttk.Button(bar, text="Save", width=6,
                                           command=self._profile_save)
        self.profile_save_btn.pack(side="left", padx=2)
        TooltipHelper(self.profile_save_btn,
                      "Overwrite the selected profile with the current "
                      "mesh-tab parameters. The shipped 'High Fidelity' "
                      "and 'Optimization Mesh' profiles can be updated "
                      "this way, but not deleted.")
        self.profile_saveas_btn = ttk.Button(bar, text="Save as ...",
                                             width=10,
                                             command=self._profile_save_as)
        self.profile_saveas_btn.pack(side="left", padx=2)
        TooltipHelper(self.profile_saveas_btn,
                      "Save the current mesh-tab parameters under a new "
                      "profile name (or overwrite an existing one).")
        self.profile_del_btn = ttk.Button(bar, text="Delete", width=7,
                                          command=self._profile_delete)
        self.profile_del_btn.pack(side="left", padx=2)
        TooltipHelper(self.profile_del_btn,
                      "Delete the selected profile. Protected profiles "
                      "('High Fidelity', 'Optimization Mesh') cannot be "
                      "deleted.")
        if self.profiles.error:
            for btn in (self.profile_save_btn, self.profile_saveas_btn,
                        self.profile_del_btn):
                btn.configure(state="disabled")
            ttk.Label(bar, foreground="#b8860b",
                      text="mesh_profiles.json is unreadable - shipped "
                           "profiles only, editing disabled"
                      ).pack(side="left", padx=(8, 0))
        self._refresh_profile_list()

    def _refresh_profile_list(self):
        names = [PROFILE_CUSTOM] + self.profiles.names()
        self.profile_box.configure(values=names)
        if self.profile_var.get() not in names:
            self.profile_var.set(PROFILE_CUSTOM)

    def _selected_profile(self):
        name = self.profile_var.get()
        return None if name == PROFILE_CUSTOM else name

    def _apply_selected_profile(self):
        name = self._selected_profile()
        if not name:
            return
        apply_values(self.app.state, self.profiles.values(name))
        self.reload_fields()
        self.profile_var.set(name)     # reload_fields resets the dropdown
        self.app.set_status(f"mesh profile '{name}' applied")

    def _profile_save(self):
        name = self._selected_profile()
        if name is None:
            self._profile_save_as()
            return
        self.profiles.save(name, capture_values(self.app.state))
        self._refresh_profile_list()
        self.app.set_status(f"mesh profile '{name}' saved")

    def _profile_save_as(self):
        name = simpledialog.askstring(
            "Save mesh profile", "Name for the mesh profile:",
            initialvalue=self._selected_profile() or "", parent=self)
        if name is None:
            return
        name = name.strip()
        if not name:
            return
        if self.profiles.get(name) is not None \
                and not messagebox.askyesno(
                    "Overwrite mesh profile",
                    f"Profile '{name}' already exists. Overwrite it with "
                    "the current mesh parameters?", parent=self):
            return
        self.profiles.save(name, capture_values(self.app.state))
        self._refresh_profile_list()
        self.profile_var.set(name)
        self.app.set_status(f"mesh profile '{name}' saved")

    def _profile_delete(self):
        name = self._selected_profile()
        if name is None:
            return
        if self.profiles.type_of(name) == PROTECTED:
            messagebox.showinfo(
                "Protected mesh profile",
                f"'{name}' is a protected profile - it cannot be deleted. "
                "You can overwrite its values with 'Save' if you want to "
                "change it.", parent=self)
            return
        if not messagebox.askyesno(
                "Delete mesh profile",
                f"Delete the mesh profile '{name}'?", parent=self):
            return
        self.profiles.delete(name)
        self._refresh_profile_list()
        self.app.set_status(f"mesh profile '{name}' deleted")

    # ------------------------------------------------------------ sources
    def _open_3d_view(self):
        """Open the moderngl 3D outer-surface view on a 3D mesh."""
        from tkinter import messagebox
        from .mesh3d_view import Mesh3DView, is_3d_file
        state = self.app.state
        proj = state.mesh_project_dir()
        candidates = [
            self._current_path,                                  # selection
            proj / "mesh_quad3d.su2",                            # wedge mesh
            state.case_dir() / "mesh.su2",                       # solver mesh
        ]
        path = next((p for p in candidates
                     if p is not None and p.exists() and is_3d_file(p)),
                    None)
        if path is None:
            messagebox.showinfo(
                "3D view needs a 3D mesh",
                "No 3D mesh found. Generate one first: the 3D view uses "
                "the axisymmetric3d wedge mesh (mesh_quad3d.su2) or the "
                "3D solver mesh.", parent=self)
            return
        Mesh3DView.open(self.app, path)

    def refresh_sources(self):
        state = self.app.state
        sources = []
        proj = state.mesh_project_dir()
        cases = state.case_dir()
        candidates = [
            (f"3D wedge mesh - chord units ({proj})",
             proj / "mesh_quad3d.su2"),
            (f"quad mesh - chord units ({proj})", proj / "mesh_quad.su2"),
            (f"tri mesh - chord units ({proj})", proj / "mesh.su2"),
            (f"solver mesh - scaled [m] ({cases})", cases / "mesh.su2"),
        ]
        seen = set()
        for label, path in candidates:
            key = str(path.resolve())
            if path.exists() and key not in seen:
                seen.add(key)
                sources.append((label, path))
        self._sources = dict(sources)
        values = [s[0] for s in sources]
        self.source_box.configure(values=values)
        if not values:
            self.source_var.set("")
            self.info_var.set("no mesh found - generate one with "
                              "'Generate mesh'")
            return
        chosen = self.source_var.get()
        if chosen not in values:
            # prefer the solver mesh, then the finest available
            pick = values[-1]
            for v in values:
                if "solver" in v:
                    pick = v
                    break
            self.source_var.set(pick)
        self.load_selected()

    def load_selected(self):
        path = self._sources.get(self.source_var.get())
        if not path:
            return
        self.info_var.set(f"loading {path.name} ...")
        threading.Thread(target=self._worker, args=(path,), daemon=True,
                         name="mesh-load").start()
        self.app.root.after(60, self._poll)

    def _worker(self, path):
        try:
            mesh = read_su2_mesh(path)
            self._load_queue.put(("ok", path, mesh))
        except Exception as e:
            self._load_queue.put(("error", path, f"{type(e).__name__}: {e}"))

    def _poll(self):
        try:
            status, path, payload = self._load_queue.get_nowait()
        except queue.Empty:
            self.app.root.after(60, self._poll)
            return
        if status == "error":
            self.info_var.set(f"failed to read {path.name}: {payload}")
            return
        self._current = payload
        self._current_path = path
        n_pts = len(payload["points"])
        n_quad = 0 if payload["quads"] is None else len(payload["quads"])
        n_tri = 0 if payload["tris"] is None else len(payload["tris"])
        if payload.get("ndim") == 3:
            self.info_var.set(
                f"{path.name}: {payload['hexes']} hex + {payload['prisms']} "
                f"prism ({payload['n_layers']} span layers) - showing the "
                f"mid-span layer: {n_pts} nodes, {n_quad} quads + {n_tri} "
                f"tris")
        else:
            self.info_var.set(
                f"{path.name}: {n_pts} nodes, {n_quad} quads + {n_tri} tris  "
                f"[{', '.join(payload['markers']) or 'no markers'}]")
        self._redraw()
        self._fill_stats()

    # -------------------------------------------------------------- draw
    def _redraw(self):
        mesh = self._current
        ax = self.ax
        ax.clear()
        if mesh is None:
            ax.text(0.5, 0.5, "no mesh - generate one or refresh sources",
                    transform=ax.transAxes, ha="center")
            self.canvas.draw_idle()
            return
        pts = mesh["points"]
        edges = mesh_edges(mesh["tris"], mesh["quads"])
        if len(edges):
            segs = pts[edges[:, [0, 1]]]
            ax.add_collection(LineCollection(segs, linewidths=0.2,
                                             colors="#1f4e79", alpha=0.7,
                                             zorder=1), autolim=True)
        for name, var in self.marker_vars.items():
            if not var.get() or name not in mesh["markers"]:
                continue
            e = mesh["markers"][name]
            segs = pts[e[:, [0, 1]]]
            style = dict(color=MARKER_COLORS[name], lw=1.4, zorder=3)
            if "periodic" in name:
                style.update(lw=1.1, ls="--")
            line = LineCollection(segs, **style)
            line.set_label(f" {name}")
            ax.add_collection(line, autolim=True)
        dx = float(np.ptp(pts[:, 0])) or 1.0
        dy = float(np.ptp(pts[:, 1])) or 1.0
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.set_xlim(cx - 0.55 * dx, cx + 0.55 * dx)
        ax.set_ylim(cy - 0.55 * dy, cy + 0.55 * dy)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        name = self._current_path.name if self._current_path else ""
        unit = "m" if "scaled" in self.source_var.get() else "chord units"
        layer = "  (mid-span layer)" if mesh.get("ndim") == 3 else ""
        ax.set_title(f"{name}  ({unit}){layer}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.toolbar.update()

    # ------------------------------------------------------------- stats
    def _fill_stats(self):
        mesh = self._current
        lines = [f"file: {self._current_path}"]
        pts = mesh["points"]
        if mesh.get("ndim") == 3:
            lines.append(f"cells: {mesh['hexes']} hex + {mesh['prisms']} "
                         f"prism ({mesh['n_layers']} span layers)")
            lines.append("view: mid-span node layer")
        lines.append(f"nodes (shown): {len(pts)}")
        for key, label in (("quads", "quads"), ("tris", "triangles")):
            if mesh[key] is not None:
                lines.append(f"{label}: {len(mesh[key])}")
        if mesh.get("ndim") == 3:
            for name, cnt in mesh.get("marker_face_counts", {}).items():
                lines.append(f"marker {name}: {cnt} faces")
        else:
            for name, e in mesh["markers"].items():
                lines.append(f"marker {name}: {len(e)} edges")
        bbox = (f"x [{pts[:, 0].min():.4g}, {pts[:, 0].max():.4g}]  "
                f"y [{pts[:, 1].min():.4g}, {pts[:, 1].max():.4g}]")
        lines.append(f"extent: {bbox}")

        # quality report written by the mesh pipeline, if present
        for qdir in (self._current_path.parent, self.app.state.mesh_project_dir()):
            qfile = Path(qdir) / "mesh_quality.txt"
            if qfile.exists():
                lines += ["", "quality report", "-" * 40,
                          qfile.read_text(errors="ignore").strip()]
                break
        self.stats.configure(state="normal")
        self.stats.delete("1.0", "end")
        self.stats.insert("1.0", "\n".join(lines))
        self.stats.configure(state="disabled")

    # -------------------------------------------------------------- jobs
    def on_job_event(self, kind, data):
        if kind == "done" and data["ok"] and data.get("mode") == "mesh":
            self.refresh_sources()
        elif kind == "done" and data.get("mode") in ("full", "setup"):
            self.refresh_sources()


class TooltipHelper:
    """Tiny tooltip wrapper (avoids a widget reference cycle in the bar)."""

    def __init__(self, widget, text):
        from .widgets import Tooltip
        self._t = Tooltip(widget, text)
