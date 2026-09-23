"""Solution tab: auto-populated results, field viewer, .vtk import, live monitor.

- While the solver runs, the canvas shows the live convergence (history.csv
  tail, same reader as tools/run_monitor.py).
- When post-processing writes results.json, the tab populates automatically:
  metrics on the left, latest vol_solution.vtk field on the right, plot
  thumbnails below.
- "Import .vtk..." loads any saved SU2 legacy volume file into the viewer.
"""

import json
import queue
import re
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from scipy.interpolate import griddata

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_monitor import HistoryTail, parse_cfg_totals  # noqa: E402
from su2_vtk import read_legacy_vtk                    # noqa: E402
from plot_case import (fields_from_volume, wall_points,  # noqa: E402
                       surface_distributions, volume_triangulation,
                       periodic_plot_positions)

from .widgets import Tooltip                           # noqa: E402

RESULTS_POLL_MS = 1500
LIVE_POLL_MS = 600

# 1D surface-chart options: (dropdown label, key in surface_distributions,
# y-axis label)
SURFACE_QUANTITIES = [
    ("isentropic Mach (Ma_is)", "ma_isen", "isentropic Mach number"),
    ("back-surface diffusion (DF)", "diffusion",
     "diffusion factor DF = 1 - V/V_peak"),
    ("skin friction (Cf)", "cf", "skin friction coefficient |Cf|"),
    ("pressure coefficient (Cp)", "cp", "pressure coefficient Cp"),
    ("static pressure (p)", "p", "static pressure [Pa]"),
    ("y+ (wall)", "yplus", "wall y+"),
]


class SolutionTab(ttk.Frame):
    def __init__(self, app):
        super().__init__(app.notebook)
        self.app = app
        self._queue = queue.Queue()
        self._results_mtime = None
        self._vol_mtime = None
        self._volume = None            # parsed vtk dict (case or imported)
        self._volume_kind = None       # "case" | "imported"
        self._fields = {}              # name -> (values, cmap)
        self._field_names = []
        self._live_tail = None
        self._live_cfg = None
        self._job_solving = False
        self._busy = False
        self._view_mode = None          # None | "field" | "live"
        self._live_axes = None

        # -------------------------------------------------------- top bar
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=(4, 0))
        self.status_var = tk.StringVar(value="waiting for results.json - the "
                                             "tab fills in automatically "
                                             "after a run")
        tk.Label(bar, textvariable=self.status_var, anchor="w",
                 font=("TkDefaultFont", 8, "bold")).pack(side="left")
        self.run_btn = tk.Button(bar, text="Start run", state="disabled",
                                 command=lambda: app.run_job("solve"),
                                 bg="#dff0d8")
        self.run_btn.pack(side="right", padx=2)
        Tooltip(self.run_btn, "Solve only: run SU2 on the prepared "
                              "turbine.cfg (skips meshing/case setup).")
        self.stop_btn = tk.Button(bar, text="Stop run", state="disabled",
                                  command=app.stop_job, fg="#a00")
        self.stop_btn.pack(side="right", padx=2)
        ttk.Button(bar, text="Reload", width=8, command=self.reload_all
                   ).pack(side="right", padx=2)
        ttk.Button(bar, text="Import .vtk ...", command=self.import_vtk
                   ).pack(side="right", padx=2)
        self._update_buttons()

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # --------------------------------------------------------- left
        left = ttk.Frame(pane)
        pane.add(left, weight=1)
        box = ttk.LabelFrame(left, text=" Results summary (results.json) ")
        box.pack(fill="both", expand=True, padx=4, pady=4)
        self.metrics = scrolledtext.ScrolledText(
            box, font=("Consolas", 9), width=52, state="disabled",
            wrap="word")
        self.metrics.pack(fill="both", expand=True, padx=2, pady=2)

        self.thumbs_bar = ttk.Frame(left)
        self.thumbs_bar.pack(fill="x", padx=4, pady=(0, 4))

        # -------------------------------------------------------- right
        right = ttk.Frame(pane)
        pane.add(right, weight=2)
        viewer_bar = ttk.Frame(right)
        viewer_bar.pack(fill="x")
        ttk.Label(viewer_bar, text="view:").pack(side="left", padx=(4, 2))
        self.mode_var = tk.StringVar(value="contour")
        self.mode_box = ttk.Combobox(viewer_bar, textvariable=self.mode_var,
                                     state="readonly", width=12,
                                     values=("contour", "1D surface",
                                             "streamlines", "convergence"))
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>",
                           lambda _e: self._mode_changed())
        ttk.Label(viewer_bar, text="quantity:").pack(side="left", padx=(10, 2))
        self.field_var = tk.StringVar()
        self.field_box = ttk.Combobox(viewer_bar, textvariable=self.field_var,
                                      state="readonly", width=26)
        self.field_box.pack(side="left")
        self.field_box.bind("<<ComboboxSelected>>",
                            lambda _e: self.draw_current())

        # streamlines controls (shown only in streamlines view)
        self.stream_ctl = []
        self.stream_density_lbl = ttk.Label(viewer_bar, text="density:")
        self.density_var = tk.StringVar(value="1.6")
        self.stream_density_box = ttk.Combobox(
            viewer_bar, textvariable=self.density_var, width=4,
            values=("0.8", "1.2", "1.6", "2.0", "2.6", "3.2"))
        self.stream_arrow_lbl = ttk.Label(viewer_bar, text="arrow size:")
        self.arrow_var = tk.StringVar(value="1.0")
        self.stream_arrow_box = ttk.Combobox(
            viewer_bar, textvariable=self.arrow_var, width=4,
            values=("0.7", "1.0", "1.4", "2.0"))
        for w in (self.stream_density_lbl, self.stream_density_box,
                  self.stream_arrow_lbl, self.stream_arrow_box):
            w.pack(side="left", padx=(8, 0))
            w.pack_forget()                      # revealed in streamlines view
            self.stream_ctl.append(w)
        for box in (self.stream_density_box, self.stream_arrow_box):
            box.bind("<<ComboboxSelected>>", lambda _e: self.draw_current())
            box.bind("<Return>", lambda _e: self.draw_current())
            box.bind("<FocusOut>", lambda _e: self.draw_current()
                     if self.mode_var.get() == "streamlines" else None)

        self.viewer_note = tk.Label(viewer_bar, text="", anchor="e",
                                    font=("TkDefaultFont", 8), fg="#595959")
        self.viewer_note.pack(side="right", padx=6)

        self.fig = Figure(figsize=(7.4, 5.6), dpi=96)
        self.ax = None                  # created by _enter_field_view
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right,
                                            pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        # ------------------------------------------------------ polling
        self._enter_field_view()
        self.reload_all()
        self.after(RESULTS_POLL_MS, self._poll_results)
        self.after(LIVE_POLL_MS, self._poll_live)

    # ---------------------------------------------------- view management
    def _enter_field_view(self):
        """Show exactly one full-size axes for the field plot."""
        self._view_mode = "field"
        self._live_axes = None
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

    def _enter_live_view(self):
        """Switch the canvas to the 2x2 live-convergence grid (idempotent)."""
        if self._view_mode == "live" and self._live_axes is not None:
            return
        self._view_mode = "live"
        self._live_axes = None
        self.fig.clear()
        self._live_axes = tuple(self.fig.add_subplot(n) for n in
                                (221, 222, 223, 224))

    # ------------------------------------------------------------ loading
    def reload_all(self):
        self._pitch_cache = None      # (mtime, pitch) tuple
        self._load_results(auto=True)
        self._load_case_volume()
        self._refresh_thumbnails()

    def _load_results(self, auto=False):
        path = self.app.state.case_dir() / "results.json"
        if not path.exists():
            if not auto:
                self._set_metrics("no results.json in "
                                  f"{self.app.state.case_dir()}")
            return None
        try:
            data = json.loads(path.read_text())
            self._results_mtime = path.stat().st_mtime
            self._set_metrics(self._format_results(data, path))
            self.status_var.set(
                f"results.json loaded (written "
                f"{time.strftime('%H:%M:%S', time.localtime(self._results_mtime))})"
                f" - auto-reloads when a new run finishes")
        except Exception as e:
            self._set_metrics(f"failed to read results.json: {e}")
            return None
        return data

    def _load_case_volume(self, force=False):
        case_dir = self.app.state.case_dir()
        vols = sorted(case_dir.glob("vol_solution*.vtk"))
        if not vols:
            return
        path = vols[-1]
        mtime = path.stat().st_mtime
        if not force and self._volume_kind == "case" \
                and mtime == self._vol_mtime:
            return
        if self._volume_kind == "imported" and not force:
            return                     # keep an imported file on screen
        self._vol_mtime = mtime
        self._load_volume_async(path, kind="case")

    def _load_volume_async(self, path, kind):
        if self._busy:
            return
        self._busy = True
        self.viewer_note.configure(text=f"loading {path.name} ...")
        threading.Thread(target=self._volume_worker, args=(path, kind),
                         daemon=True, name="vtk-load").start()
        self.after(80, self._poll_volume)

    def _volume_worker(self, path, kind):
        try:
            d = read_legacy_vtk(str(path))
            fields = fields_from_volume(d)
            self._queue.put(("ok", path, kind, d, fields))
        except Exception as e:
            self._queue.put(("error", path, kind, str(e), None))

    def _poll_volume(self):
        try:
            status, path, kind, payload, fields = self._queue.get_nowait()
        except queue.Empty:
            self.after(80, self._poll_volume)
            return
        self._busy = False
        if status == "error":
            self.viewer_note.configure(text=f"failed to read {path.name}: {payload}")
            return
        self._volume = payload
        self._volume_kind = kind
        self._fields = fields
        self._field_names = sorted(fields)
        self._surf_cache_for = None
        self._sync_quantity_choices()
        self.draw_current()
        n = len(payload["points"])
        self.viewer_note.configure(text=
            f"{path.name} ({n} nodes)"
            + ("  [imported]" if kind == "imported" else ""))

    # ------------------------------------------------------ quantity list
    def _mode_changed(self):
        self._sync_quantity_choices()
        self._update_stream_controls()
        self.draw_current()

    def _update_stream_controls(self):
        show = self.mode_var.get() == "streamlines"
        for w in self.stream_ctl:
            if show:
                w.pack(side="left", padx=(8, 0))
            else:
                w.pack_forget()

    def _sync_quantity_choices(self):
        """Fill the quantity dropdown for the active view mode."""
        if self.mode_var.get() == "convergence":
            # the convergence grid has no per-field quantity
            self.field_box.configure(state="disabled")
            return
        self.field_box.configure(state="readonly")
        if self.mode_var.get() == "1D surface":
            values = [q[0] for q in SURFACE_QUANTITIES]
        else:
            values = self._field_names
        self.field_box.configure(values=values)
        if values and self.field_var.get() not in values:
            if self.mode_var.get() == "1D surface":
                self.field_var.set(values[0])
            else:
                self.field_var.set("Mach" if "Mach" in values else values[0])

    def draw_current(self):
        if self.mode_var.get() == "convergence":
            self.show_convergence()
            return
        if self._volume is None:
            # no volume yet: draw_field's guard shows a placeholder while
            # still switching the canvas away from any live/convergence grid
            self.draw_field()
            return
        if self.mode_var.get() == "1D surface":
            self.draw_surface()
        elif self.mode_var.get() == "streamlines":
            self.draw_streamlines()
        else:
            self.draw_field()

    # ------------------------------------------------------------- viewer
    def draw_field(self):
        name = self.field_var.get()
        # switch the canvas first: a mode change must never leave the
        # live-convergence grid on screen when no volume is loaded
        self._enter_field_view()
        if not name or name not in self._fields:
            self.ax.text(0.5, 0.5, "no volume field loaded - run the solver "
                         "or import a .vtk",
                         transform=self.ax.transAxes, ha="center",
                         fontsize=9, color="0.4")
            self.fig.tight_layout()
            self.canvas.draw_idle()
            return
        values, cmap = self._fields[name]
        pts, conn = volume_triangulation(self._volume)
        positions = periodic_plot_positions(self._volume,
                                            self._periodic_pair())
        ax = self.ax
        tc = None
        for xy in positions:
            tc = ax.tricontourf(xy[:, 0], xy[:, 1], conn, values,
                                levels=40, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        if len(positions) > 1:
            offset = max(float(np.max(np.abs(xy[:, 1] - pts[:, 1])))
                         for xy in positions)
            ymin -= 0.45 * offset
            ymax += 0.45 * offset
        ax.set_ylim(ymin, ymax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        self.fig.colorbar(tc, ax=ax, shrink=0.9, pad=0.01)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _periodic_pair(self):
        """Periodic transform from case.json, cached until the case changes."""
        case_path = self.app.state.case_dir() / "case.json"
        try:
            mtime = case_path.stat().st_mtime
        except OSError:
            return None
        cached = getattr(self, "_periodic_cache", None)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        pair = None
        try:
            case = json.loads(case_path.read_text())
            pers = (case.get("cascade") or {}).get("periodic") or []
            if pers:
                pair = pers[0]
        except Exception:
            pass
        self._periodic_cache = (mtime, pair)
        return pair

    def _periodic_pitch(self):
        """Translation pitch for the rectilinear streamline grid."""
        pair = self._periodic_pair() or {}
        return float((pair.get("translation") or [0, 0, 0])[1] or 0.0)

    def draw_surface(self):
        """1D surface distributions (SS/PS curves over u in [0, 1])."""
        dist = self._surface_distribution()
        self._enter_field_view()
        ax = self.ax
        label, key, ylabel = next(
            (q for q in SURFACE_QUANTITIES if q[0] == self.field_var.get()),
            SURFACE_QUANTITIES[0])
        if dist is None or key not in dist["ss"]:
            reason = ("no wall nodes found in this volume" if dist is None
                      else f"{key} not present in this solution")
            ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha="center",
                    fontsize=9, color="0.4")
            ax.set_title(label, fontsize=10)
            self.canvas.draw_idle()
            return
        for side, color, name in (("ss", "tab:red", "suction side"),
                                  ("ps", "tab:blue", "pressure side")):
            s = dist[side]
            ax.plot(s["u"], s[key], color=color, lw=1.4, label=name)
        ax.set_xlabel("u  (LE -> TE)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{label} - surface distribution", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _surface_distribution(self):
        """Surface distributions of the loaded volume (cached per volume)."""
        if self._volume is None:
            return None
        if getattr(self, "_surf_cache_for", None) is self._volume:
            return self._surf_cache
        d = self._volume
        case = {}
        case_path = self.app.state.case_dir() / "case.json"
        if case_path.exists():
            try:
                case = json.loads(case_path.read_text())
            except Exception:
                case = {}
        gamma = case.get("physics", {}).get("gamma", 1.4)
        ss_upper = case.get("postprocess", {}).get("ss_upper", True)
        flds = self._fields
        if "cascade" in case:
            cas = case["cascade"]
            p0_ref = float(cas["inlet"]["total_pressure"])
            p_ref = float(cas["outlet"]["static_pressure"])
            v_ref = None
            try:
                from plot_case import cascade_refs
                v_ref = cascade_refs(cas, case["physics"])["V2"]
            except Exception:
                v_ref = None
        else:
            p0_ref = p_ref = v_ref = None
        if v_ref is None:
            v_ref = float(flds["Velocity magnitude"][0].max()) \
                if "Velocity magnitude" in flds else 1.0
        if p0_ref is None:
            p0_ref = float(flds["Total_Pressure"][0].max()) \
                if "Total_Pressure" in flds else 1.0
        if p_ref is None:
            p_ref = float(flds["Pressure"][0].mean()) \
                if "Pressure" in flds else 0.0
        dist = surface_distributions(d, v_ref, p0_ref, gamma, p_ref=p_ref,
                                     ss_upper=ss_upper)
        self._surf_cache_for = self._volume
        self._surf_cache = dist
        return dist

    def draw_streamlines(self):
        """Velocity streamlines on a regular grid, drawn for the main
        domain and both periodic copies, colored by the selected scalar
        field. The blade interior is masked and the mesh domain edges +
        blade outline are drawn as reference."""
        d = self._volume
        pts = d["points"]
        if "Velocity" not in d:
            self._enter_field_view()
            self.ax.text(0.5, 0.5, "no velocity field in this volume",
                         transform=self.ax.transAxes, ha="center",
                         fontsize=9, color="0.4")
            self.canvas.draw_idle()
            return
        vel = d["Velocity"][:, :2]
        key = id(d)
        if getattr(self, "_stream_key", None) != key:
            x, y = pts[:, 0], pts[:, 1]
            xg = np.linspace(x.min(), x.max(), 700)
            yg = np.linspace(y.min(), y.max(), 500)
            XX, YY = np.meshgrid(xg, yg)
            _, conn = volume_triangulation(d)
            from scipy.spatial import Delaunay
            from scipy.interpolate import LinearNDInterpolator
            tri = Delaunay(pts)
            # no-slip inside the blade: zero velocity there so streamlines
            # terminate on the surface instead of crossing it
            from matplotlib.path import Path
            v_ref = float(np.abs(vel).max()) or 1.0
            loop = None
            try:
                loop = wall_points(pts, vel, v_ref)
                inside = Path(np.vstack([loop, loop[0]])).contains_points(
                    np.column_stack([XX.ravel(), YY.ravel()])
                ).reshape(XX.shape)
            except Exception:
                loop, inside = None, None
            u = LinearNDInterpolator(tri, vel[:, 0])(XX, YY)
            v = LinearNDInterpolator(tri, vel[:, 1])(XX, YY)
            if inside is not None:
                u = np.where(inside, 0.0, np.nan_to_num(u))
                v = np.where(inside, 0.0, np.nan_to_num(v))
            # smooth away grid-scale interpolation noise - without this the
            # streamlines jitter and appear to cross each other
            from scipy.ndimage import gaussian_filter
            sigma = 2.0
            u_f = gaussian_filter(np.nan_to_num(u, nan=0.0), sigma)
            v_f = gaussian_filter(np.nan_to_num(v, nan=0.0), sigma)
            speed = np.hypot(u_f, v_f)
            low = speed < 0.005 * max(speed.max(), 1e-12)
            nan = np.isnan(u) | np.isnan(v)
            u = np.ma.array(u_f, mask=(nan | low))
            v = np.ma.array(v_f, mask=(nan | low))
            self._stream_key = key
            self._stream_grid = (XX, YY, u, v)
            self._stream_loop = loop
            self._stream_tri = tri
            self._stream_colors = {}
            self._edges_cache = None
        XX, YY, u, v = self._stream_grid

        # streamlines colored by the selected scalar (interpolated to the
        # same grid)
        name = self.field_var.get()
        color = cmap = None
        if name in self._fields:
            values, cmap = self._fields[name]
            ckey = (key, name)
            self._stream_colors.setdefault(ckey, None)
            if self._stream_colors[ckey] is None:
                from scipy.interpolate import LinearNDInterpolator
                c = LinearNDInterpolator(self._stream_tri, values)(XX, YY)
                self._stream_colors[ckey] = np.ma.masked_invalid(c)
            color = self._stream_colors[ckey]

        try:
            density = min(max(float(self.density_var.get()), 0.05), 10.0)
        except (ValueError, tk.TclError):
            density = 1.6
        try:
            arrowsize = min(max(float(self.arrow_var.get()), 0.2), 4.0)
        except (ValueError, tk.TclError):
            arrowsize = 1.0

        pitch = self._periodic_pitch()
        shifts = (-pitch, 0.0, pitch) if pitch else (0.0,)

        self._enter_field_view()
        ax = self.ax
        for dy in shifts:
            sp = dict(x=XX, y=YY + dy, u=u, v=v, density=density,
                      linewidth=0.9, arrowsize=arrowsize, minlength=0.3)
            if color is not None:
                ax.streamplot(color=color, cmap=cmap, **sp)
            else:
                ax.streamplot(color="0.35", **sp)
        ax.set_aspect("equal")

        # reference: mesh periodic domain edges + blade outline
        edges = self._domain_edges()
        if edges:
            for name_e, c in edges.items():
                ax.plot(c[:, 0], c[:, 1], "--", color="0.35", lw=0.9,
                        zorder=4,
                        label="domain edges" if name_e == "periodic_bottom"
                        else None)
        loop = getattr(self, "_stream_loop", None)
        if loop is not None:
            closed = np.vstack([loop, loop[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="0.2", lw=1.2,
                    zorder=5, label="airfoil")
        ax.set_title("velocity streamlines", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if edges or loop is not None:
            ax.legend(fontsize=7, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _domain_edges(self):
        """Periodic domain edge polylines from the solver mesh (meters)."""
        mesh_path = self.app.state.case_dir() / "mesh.su2"
        if not mesh_path.exists():
            return None
        try:
            stamp = (mesh_path, mesh_path.stat().st_mtime)
        except OSError:
            return None
        cached = getattr(self, "_edges_cache", None)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        edges = {}
        try:
            from foilage_gui.su2_mesh import read_su2_mesh
            m = read_su2_mesh(mesh_path)
            for name in ("periodic_bottom", "periodic_top", "farfield"):
                if name in m["markers"]:
                    e = m["markers"][name]
                    coords = m["points"][e[:, 0]]      # first node per edge
                    edges[name] = coords[np.argsort(coords[:, 0])]
        except Exception:
            edges = {}
        self._edges_cache = (stamp, edges)
        return edges

    def import_vtk(self):
        case_dir = self.app.state.case_dir()
        path = filedialog.askopenfilename(
            title="Import SU2 legacy volume .vtk",
            initialdir=str(case_dir if case_dir.exists() else Path.home()),
            filetypes=[("VTK legacy", "*.vtk"), ("All files", "*.*")])
        if path:
            self._load_volume_async(Path(path), kind="imported")

    # ------------------------------------------------------------- thumbs
    def _refresh_thumbnails(self):
        for child in self.thumbs_bar.winfo_children():
            child.destroy()
        case_dir = self.app.state.case_dir()
        plots = [("convergence.png", "convergence"),
                 ("fields.png", "fields"),
                 ("nearwall.png", "near-wall"),
                 ("bl_validation.png", "BL validation")]
        found = False
        for fname, label in plots:
            path = case_dir / fname
            if path.exists():
                found = True
                ttk.Button(self.thumbs_bar, text=label, width=13,
                           command=lambda p=path: self._open_image(p)
                           ).pack(side="left", padx=3)
        if found:
            ttk.Button(self.thumbs_bar, text="open folder", width=10,
                       command=lambda: self._open_folder(case_dir)
                       ).pack(side="right", padx=3)

    def _open_image(self, path):
        win = tk.Toplevel(self)
        win.title(str(path))
        try:
            photo = tk.PhotoImage(file=str(path))   # Tk 8.6 reads PNG natively
        except tk.TclError:
            self._open_externally(path)
            win.destroy()
            return
        w, h = photo.width(), photo.height()
        win.geometry(f"{min(w, 1100) + 24}x{min(h, 800) + 24}")
        canvas = tk.Canvas(win, highlightthickness=0)
        hs = ttk.Scrollbar(win, orient="horizontal", command=canvas.xview)
        vs = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        hs.pack(side="bottom", fill="x")
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, w, h))
        canvas.image = photo                        # keep a reference

    @staticmethod
    def _open_externally(path):
        import os
        if hasattr(os, "startfile"):
            os.startfile(str(path))
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])

    def _open_folder(self, path):
        self._open_externally(path)

    # ------------------------------------------------------------ summary
    def _set_metrics(self, text):
        self.metrics.configure(state="normal")
        self.metrics.delete("1.0", "end")
        self.metrics.insert("1.0", text)
        self.metrics.configure(state="disabled")

    def _format_results(self, r, path):
        lines = [f"case: {r.get('case', self.app.state.case_name())}",
                 f"file: {path}", "-" * 46]
        c = r.get("convergence", {})
        lines.append(f"iterations run : {c.get('iterations_run')}")
        lines.append(f"target reached : {c.get('target_reached')} "
                     f"(rms <= {c.get('target_rms')})")
        for k, v in (c.get("rms_final") or {}).items():
            lines.append(f"  {k:10s} = {v: .3f}")
        f = r.get("forces", {})
        if f:
            lines.append(f"forces         : CD = {f.get('CD', 0):.4f}  "
                         f"CL = {f.get('CL', 0):.4f}")
        w = r.get("wall", {})
        if w:
            lines.append(f"wall y+        : median {w.get('yplus_median', 0):.1f}  "
                         f"p95 {w.get('yplus_p95', 0):.1f}  "
                         f"({w.get('wall_nodes')} nodes)")
        lines.append(f"max Mach       : "
                     f"{r.get('fields', {}).get('max_mach', 0):.3f}")
        for plane in ("inlet", "outlet"):
            p = r.get(plane)
            if p:
                lines.append("")
                lines.append(f"{plane}:")
                if "mass_flow_kg_s" in p:       # 3D wedge: real sector flow
                    lines.append(f"  mass flow  : "
                                 f"{p.get('mass_flow_kg_s', 0):.4f} kg/s "
                                 f"(wedge sector)")
                    lines.append(f"  corrected  : "
                                 f"{p.get('corrected_flow_kg_s', 0):.4f} "
                                 f"kg/s  (W*sqrt(theta)/delta)")
                else:
                    lines.append(f"  mass flow  : "
                                 f"{p.get('mass_flow_kg_s_m', 0):.4f} kg/(s.m)")
                    lines.append(f"  corrected  : "
                                 f"{p.get('corrected_flow_kg_s_m', 0):.4f} "
                                 f"kg/(s.m)  (W*sqrt(theta)/delta)")
                lines.append(f"  Mach       : {p.get('mach', 0):.3f}")
                lines.append(f"  velocity   : {p.get('velocity_m_s', 0):.1f} m/s")
                lines.append(f"  flow angle : {p.get('flow_angle_deg', 0):+.2f} deg")
                lines.append(f"  static p   : {p.get('static_p_pa', 0):.1f} Pa")
                lines.append(f"  total p    : {p.get('p0_pa', 0):.1f} Pa")
        g = r.get("geometry")
        if g:
            lines.append("")
            lines.append(f"geometry (from mesh pipeline):")
            lines.append(f"  throat width : "
                         f"{g.get('throat_width_cax', 0):.4f} c_ax at "
                         f"x/c_ax {g.get('throat_x_over_cax', 0):.3f}")
            lines.append(f"  unguided turning: "
                         f"{g.get('unguided_turning_deg', 0):+.1f} deg "
                         f"(exit {g.get('angle_exit_deg', 0):.1f} deg)")
            if "pitch_to_chord" in g:
                lines.append(f"  pitch-to-chord: {g['pitch_to_chord']:.5f} "
                             f"(pitch {g.get('pitch_throat_cax', 0):.4f} "
                             f"c_ax / true chord {g.get('true_chord_cax', 0):.4f} "
                             f"c_ax)")
        if "losses" in r:
            lines.append("")
            lines.append(f"total-pressure loss Yp : "
                         f"{r['losses'].get('total_pressure_loss_coeff_Yp', 0):.4f}")
        zw = r.get("zweifel")
        if zw:
            lines.append(f"Zweifel incompressible : "
                         f"{zw.get('incompressible', 0):.4f}")
            lines.append(f"Zweifel compressible   : "
                         f"{zw.get('compressible', 0):.4f}   "
                         f"(s/bx {zw.get('pitch_over_axial_chord', 0):.3f}, "
                         f"a1 {zw.get('alpha1_deg', 0):+.1f} deg -> "
                         f"a2 {zw.get('alpha2_deg', 0):+.1f} deg)")
        bsd = r.get("back_surface_diffusion")
        if bsd:
            ss_df = bsd.get("ss", {}).get("DF", 0)
            ps_df = bsd.get("ps", {}).get("DF", 0)
            lines.append(f"back-surface DF (TE)   : ss {ss_df:.3f}   "
                         f"ps {ps_df:.3f}")
        if "mass_balance" in r:
            lines.append(f"mass-flow imbalance    : "
                         f"{r['mass_balance'].get('imbalance_pct', 0):.3f} %")
        extra = {k: v for k, v in r.items()
                 if k not in ("case", "convergence", "forces", "wall",
                              "fields", "inlet", "outlet", "losses",
                              "mass_balance")}
        if extra:
            lines += ["", "additional data:", json.dumps(extra, indent=2)]
        return "\n".join(lines)

    # --------------------------------------------------------- live mode
    def _update_buttons(self):
        running = self.app.job is not None and self.app.job.running
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")

    def on_job_event(self, kind, data):
        self._update_buttons()
        if kind == "stage":
            if data["state"] == "start" and "SU2 solve" in data["title"]:
                self._start_live()
            elif data["state"] in ("ok", "failed", "stopped") \
                    and "SU2 solve" in data["title"]:
                self._stop_live(finished=data["state"] == "ok")
        elif kind == "done":
            # the watchdog writes results.json shortly after; keep polling
            pass

    def _start_live(self):
        case_dir = self.app.state.case_dir()
        cfg_path = case_dir / "turbine.cfg"
        self._live_tail = HistoryTail(case_dir / "history.csv")
        self._live_cfg = parse_cfg_totals(cfg_path)
        self._job_solving = True
        self.status_var.set("solver running - live convergence below "
                            "(results.json will load automatically)")

    def _stop_live(self, finished):
        self._job_solving = False
        self.status_var.set("solver stopped - waiting for post-processing "
                            "to write results.json ...")

    def _poll_live(self):
        # a single bad tick must never cancel the polling loop (the live
        # plots would freeze permanently) - reschedule unconditionally
        try:
            if self._job_solving and self._live_tail:
                self._live_tail.poll()
                self._draw_live()
            elif self.mode_var.get() == "convergence":
                # static convergence view: refresh when history.csv changes
                # (e.g. a solve started outside the GUI)
                hist = self.app.state.case_dir() / "history.csv"
                try:
                    mtime = hist.stat().st_mtime
                except OSError:
                    mtime = None
                if mtime is not None and mtime != getattr(self, "_conv_mtime",
                                                          None):
                    self.show_convergence()
        except Exception:
            import traceback
            self.status_var.set("live view error - see console")
            traceback.print_exc()
        self.after(LIVE_POLL_MS, self._poll_live)

    def _draw_live(self):
        total = self._live_cfg[1] if self._live_cfg else None
        self._render_convergence(self._live_tail, total,
                                 "solver running - live convergence")

    def show_convergence(self):
        """The 'convergence' view mode: the exact same 2x2 grid the
        solver's live view shows, rendered from the case's history.csv.
        While a solve is running it keeps live-tailing the file."""
        if self._job_solving and self._live_tail:
            self._draw_live()
            return
        case_dir = self.app.state.case_dir()
        hist = case_dir / "history.csv"
        self._enter_live_view()
        if not hist.exists():
            ax_res, ax_f, ax_m, ax_b = self._live_axes
            for ax in self._live_axes:
                ax.clear()
                ax.grid(True, alpha=0.3)
            ax_res.text(0.5, 0.5, f"no history.csv in {case_dir}",
                        transform=ax_res.transAxes, ha="center",
                        fontsize=8, color="0.4")
            self.fig.tight_layout()
            self.canvas.draw_idle()
            self._conv_mtime = None
            return
        tail = HistoryTail(hist)
        tail.poll()
        self._conv_mtime = hist.stat().st_mtime
        cfg_path = case_dir / "turbine.cfg"
        total = None
        if cfg_path.exists():
            try:
                total = parse_cfg_totals(cfg_path)[1]
            except Exception:
                total = None
        self._render_convergence(tail, total, "convergence (history.csv)")

    def _render_convergence(self, tail, total, status_prefix):
        xcol = "Time_Iter" if (tail.get("Time_Iter")
                               and max(tail.get("Time_Iter")) > 0) \
            else ("Inner_Iter" if "Inner_Iter" in (tail.columns or [])
                  else None)
        x = tail.get(xcol) or []
        self._enter_live_view()
        ax_res, ax_f, ax_m, ax_b = self._live_axes

        if not x:
            for ax in self._live_axes:
                ax.clear()
                ax.grid(True, alpha=0.3)
            ax_res.text(0.5, 0.5,
                        f"waiting for history.csv ... ({tail.rows} rows)",
                        transform=ax_res.transAxes, ha="center")
            self.status_var.set(f"{status_prefix} - waiting for history.csv")
            self.fig.tight_layout()
            self.canvas.draw_idle()
            return

        x = np.asarray(x)
        self._panel_residuals(ax_res, tail, x)
        if self._live_is_freestream():
            self._panel_forces(ax_f, tail, x)
        else:
            # periodic cascade: the meaningful "force" is the total-
            # pressure loss, not CL/CD (those belong to freestream cases)
            self._panel_yloss(ax_f, tail, x)
        self._panel_massflow(ax_m, tail, x)
        self._panel_imbalance(ax_b, tail, x)
        extra = f"   iter {x[-1]}/{total}" if total else f"   iter {x[-1]}"
        self.status_var.set(f"{status_prefix}{extra}")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _panel_residuals(ax, tail, x):
        ax.clear()
        plotted = False
        for col in (tail.columns or []):
            if col.startswith("rms["):
                ax.plot(x, tail.get(col), lw=1,
                        label=col.replace("rms", "rms "))
                plotted = True
        ax.set_title("residuals (log10)", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=6, ncol=2, loc="upper right")

    def _live_is_freestream(self):
        """CL/CD are only meaningful without periodics (isolated airfoil);
        periodic cascade cases monitor the total-pressure loss instead."""
        return (self.app.state.get("domain.periodicity")
                or "axisymmetric") == "freestream"

    @staticmethod
    def _panel_forces(ax, tail, x):
        ax.clear()
        cd, cl = tail.get("CD"), tail.get("CL")
        if cd:
            ax.plot(x, cd, lw=1.1, color="tab:red", label="CD")
        if cl:
            ax.plot(x, cl, lw=1.1, color="tab:blue", label="CL")
        ax.set_title("force coefficients", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        if cd or cl:
            ax.legend(fontsize=6, loc="upper right")

    def _panel_yloss(self, ax, tail, x):
        """Total-pressure loss coefficient convergence for periodic
        cascade cases: Yp = (p01 - p02) / (p01 - p2), with p02 the
        mass-averaged outlet total pressure from the per-surface
        history columns and p01/p2 the case boundary conditions."""
        ax.clear()
        ax.set_title("total-pressure loss Yp", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="k", ls=":", lw=0.8)
        p02 = tail.get("Avg_TotalPress(outlet)")
        if not p02:
            ax.text(0.5, 0.5,
                    "no Avg_TotalPress(outlet) column\n(re-run case setup)",
                    transform=ax.transAxes, ha="center", fontsize=7,
                    color="0.4")
            return
        p01 = self.app.state.get("BCs.inlet.total pressure")
        p2 = self.app.state.get("BCs.outlet.static pressure")
        if not p01 or not p2 or p01 <= p2:
            ax.text(0.5, 0.5, "inlet total / outlet static pressure\n"
                    "not set in the Setup tab",
                    transform=ax.transAxes, ha="center", fontsize=7,
                    color="0.4")
            return
        yp = [(p01 - v) / (p01 - p2) for v in p02]
        ax.plot(x, yp, lw=1.2, color="tab:red", label="Yp")
        ax.legend(fontsize=6, loc="upper right")

    def _panel_massflow(self, ax, tail, x):
        """Mass flow at inlet and outlet (per-surface history columns)."""
        ax.clear()
        ax.set_title("mass flow  [kg/(s\u00b7m)]", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        cols = tail.columns or []
        per_surface = {}
        for col in cols:
            m = re.match(r"Avg_Massflow\((.+)\)$", col.strip())
            if m:
                per_surface[m.group(1)] = col
        if per_surface:
            for marker, col in per_surface.items():
                ax.plot(x, tail.get(col), lw=1.2, label=f"mdot {marker}")
            ax.legend(fontsize=6, loc="upper right")
            return
        # older cases: only the aggregate (first analyzed marker) exists
        mdot = tail.get("Avg_Massflow")
        if mdot is None:
            mdot = tail.get("SURFACE_MASSFLOW")
        if mdot:
            ax.plot(x, mdot, lw=1.2, color="tab:green", label="mdot inlet")
            ax.legend(fontsize=6, loc="upper right")
            ax.text(0.98, 0.05, "outlet needs FLOW_COEFF_SURF\n(re-run case "
                    "setup)", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6, color="0.4")
        else:
            ax.text(0.5, 0.5, "no mass-flow columns in history.csv",
                    transform=ax.transAxes, ha="center", fontsize=7,
                    color="0.4")

    def _panel_imbalance(self, ax, tail, x):
        """Domain imbalances in % from per-surface mass-averaged fluxes."""
        ax.clear()
        ax.set_title("domain imbalance  [%]", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="k", ls=":", lw=0.8)
        cols = tail.columns or []
        col_set = set(c.strip() for c in cols)

        def arr(prefix):
            a = tail.get(f"{prefix}(inlet)")
            b = tail.get(f"{prefix}(outlet)")
            if a is None or b is None:
                return None, None
            return np.asarray(a), np.asarray(b)

        mi, mo = arr("Avg_Massflow")
        if mi is None:
            ax.text(0.5, 0.5, "needs FLOW_COEFF_SURF history\n(re-run case "
                    "setup)", transform=ax.transAxes, ha="center",
                    fontsize=7, color="0.4")
            return

        def safe_ratio(a, b):
            ok = np.abs(a) > 1e-12
            out = np.zeros_like(np.asarray(a, dtype=float))
            out[ok] = (np.asarray(a)[ok] - np.asarray(b)[ok]) / a[ok] * 100.0
            return out

        # SU2 per-surface mass flow uses outward boundary normals: inflow is
        # positive at the inlet and NEGATIVE at the outlet, so conservation
        # means the two fluxes SUM to ~0.
        # continuity: (mdot_in + mdot_out) / mdot_in
        plotted = [ax.plot(x, safe_ratio(mi, -mo), lw=1.1, color="tab:blue",
                           label="continuity")[0]]
        # energy: (mdot*cp*T0_in + mdot*cp*T0_out) / |mdot*cp*T0_in|, cp cancels
        T0i, T0o = arr("Avg_TotalTemp")
        if T0i is not None:
            e_in = mi * T0i
            e_out = mo * T0o
            plotted.append(ax.plot(x, safe_ratio(e_in, -e_out), lw=1.1,
                                   color="tab:red", label="energy (h0)")[0])
        # momentum-flux density: rho*Vn^2 + p (mass-averaged). The passage
        # areas at inlet/outlet are equal (R1 == R2) in the 2D modes so
        # they cancel in the ratio; the 3D wedge carries a streamtube
        # contraction, so the exit flux is scaled by A2/A1 = R2 h2/(R1 h1).
        # NOTE: this settles at a constant equal to the blade axial force,
        # NOT at zero.
        rho_i, rho_o = arr("Avg_Density")
        vn_i, vn_o = arr("Avg_NormalVel")
        p_i, p_o = arr("Avg_Press")
        if rho_i is not None and vn_i is not None and p_i is not None:
            f_in = rho_i * vn_i ** 2 + p_i
            f_out = rho_o * vn_o ** 2 + p_o
            area_ratio = self.app.state.derived().get(
                "streamtube_area_ratio")
            if area_ratio:
                f_out = np.asarray(f_out, dtype=float) * area_ratio
            plotted.append(ax.plot(x, safe_ratio(f_in, f_out), lw=1.1,
                                   color="tab:green",
                                   label="momentum flux")[0])
        # startup spikes (mdot passes through 0) must not flatten the scale
        data = np.concatenate([ln.get_ydata() for ln in plotted]) \
            if plotted else np.zeros(1)
        span = float(np.percentile(np.abs(data), 90)) * 1.6
        ax.set_ylim(-max(span, 1.0), max(span, 1.0))
        ax.legend(fontsize=6, loc="upper right")

    # ------------------------------------------------------ results poll
    def _poll_results(self):
        if not self._job_solving:
            path = self.app.state.case_dir() / "results.json"
            if path.exists():
                mtime = path.stat().st_mtime
                if self._results_mtime is None or mtime > self._results_mtime:
                    self._load_results()
                    self._load_case_volume(force=True)
                    self._refresh_thumbnails()
        self.after(RESULTS_POLL_MS, self._poll_results)
