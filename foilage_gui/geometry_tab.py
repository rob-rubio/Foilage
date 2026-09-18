"""Geometry tab: edit the airfoil/passage with sliders + exact entries.

The preview recomputes the airfoil (pyturbo-aero or an imported geomTurbo
section) and the periodic passage envelope whenever a geometry field
changes (debounced), on a worker thread - the GUI stays responsive while
splines rebuild. Below the blade view, two inspection charts show the
channel width (with the throat) and the surface curvature.
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

from pipeline.airfoil import build_geometry           # noqa: E402
from pipeline.cascade_metrics import channel_widths, curvature, throat_metrics
from pipeline.mesh_tris import pitch_profile, periodic_edges  # noqa: E402

from .schema import sections_for                       # noqa: E402
from .widgets import FieldWidget, ScrolledFrame        # noqa: E402

DEBOUNCE_MS = 250


def _compute_geometry(cfg):
    """Worker: build the airfoil + passage edges + channel/curvature data.

    Returns (data dict) or ("error", message)."""
    try:
        airfoil = build_geometry(cfg)
        prof = pitch_profile(cfg.get("domain", {}), airfoil)
        x0 = float(cfg.get("domain", {}).get("x_min", -0.5))
        x1 = float(cfg.get("domain", {}).get("x_max", 2.5))
        bot, top = periodic_edges(prof, x0, x1)
        outline = airfoil["outline"]
        ghost_p = outline + np.column_stack(
            [np.zeros(len(outline)), prof["p"](outline[:, 0])])
        ghost_m = outline - np.column_stack(
            [np.zeros(len(outline)), prof["p"](outline[:, 0])])
        ss, ps = airfoil["ss"], airfoil["ps"]

        # ---- channel + throat (blade vs pitch-translated neighbor) ----
        pitch = float(prof["p_le"])         # R1 == R2 for SU2 periodicity
        s_ch, _j = channel_widths(ss, ps, pitch)
        throat = throat_metrics(ss, ps, pitch)

        # ---- surface curvature ----
        k_ss, _s_ss = curvature(ss)
        k_ps, _s_ps = curvature(ps)

        # ---- imported geomTurbo reference overlay ----
        # the section is normalized to axial chord = 1 (same frame as the
        # generated blade) - raw file coordinates may be in any unit
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
                import_pts = (ref_af["ss"], ref_af["ps"])
                import_mtime = gt_path.stat().st_mtime
            except Exception as e:
                ref_err = f"{type(e).__name__}: {e}"

        a1 = cfg.get("airfoil", {}).get("alpha1")
        a2 = cfg.get("airfoil", {}).get("alpha2")
        return {
            "ss": ss, "ps": ps, "outline": outline,
            "bot": bot, "top": top, "ghost_p": ghost_p, "ghost_m": ghost_m,
            "style": airfoil["style"], "ss_upper": airfoil["ss_upper"],
            "n_points": len(ss), "x0": x0, "x1": x1,
            "turning": (float(a1) - float(a2))
            if None not in (a1, a2) and src.get("type", "pyturbo") == "pyturbo"
            else None,
            "p_le": prof["p_le"], "p_te": prof["p_te"],
            "source": src.get("type", "pyturbo"),
            "blade_count": airfoil.get("blade_count"),
            "import_pts": import_pts,
            "import_err": ref_err,
            "import_mtime": import_mtime,
            "show_reference": bool(src.get("show_reference", True)),
            "channel_x": ss[:, 0], "channel_s": s_ch,
            "k_ss": k_ss, "k_ps": k_ps,
            "throat": throat,
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

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ------------------------------------------------------------ form
        left = ttk.Frame(pane)
        pane.add(left, weight=1)
        form = ScrolledFrame(left)
        form.pack(fill="both", expand=True)
        for section in sections_for("geometry"):
            box = ttk.LabelFrame(form.inner, text=f" {section.title} ")
            box.pack(fill="x", padx=6, pady=(8, 2))
            for spec in section.fields:
                w = FieldWidget(box, spec, self._commit)
                self.fields[spec.path] = w
            if section.title == "Airfoil source":
                ttk.Button(box, text="Save as geomTurbo ...",
                           command=self._export_geomturbo).pack(
                    anchor="e", padx=4, pady=(2, 4))

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
        self.schedule_regeneration(immediate=True)

    # ------------------------------------------------------------- fields
    def _commit(self, spec, value):
        self.app.state.set(spec.path, value)
        if spec.path == "airfoil_source.geomturbo_file":
            self._refresh_sections(commit=True)
        elif spec.path == "airfoil_source.show_reference":
            if self._last_good:                     # overlay-only change
                self._draw(self._last_good)

    def _export_geomturbo(self):
        """Save the active blade section (SS/PS) as a .geomTurbo file.

        Coordinates are exported in meters: the normalized preview arrays
        are scaled by the physical axial chord.
        """
        from tkinter import filedialog, messagebox
        g = self._last_good
        if g is None:
            messagebox.showinfo(
                "Save geomTurbo",
                "The geometry has not been built yet - wait for the "
                "preview to finish and try again.")
            return
        from pipeline.geomturbo import write_geomturbo
        scale = self.app.state.scale_m_per_chord()      # m per chord unit
        path = filedialog.asksaveasfilename(
            title="Save blade section as geomTurbo",
            defaultextension=".geomTurbo",
            initialfile=f"{self.app.state.case_name()}.geomTurbo",
            filetypes=[("geomTurbo", "*.geomTurbo"), ("All files", "*.*")])
        if not path:
            return
        write_geomturbo(path, g["ss"] * scale, g["ps"] * scale, z=0.0,
                        units="m")
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
        if g["turning"] is not None:
            parts.append(f"turning {g['turning']:.1f} deg")
        parts.append(f"pitch LE {g['p_le']:.4f}, TE {g['p_te']:.4f}")
        parts.append(f"{g['n_points']} points/side")
        if g.get("blade_count") is not None:
            parts.append(f"{g['blade_count']} blades")
        if g["import_err"]:
            parts.append(f"geomTurbo reference failed ({g['import_err']})")
        return "  |  ".join(parts)

    def _update_metrics(self, g):
        t = g["throat"]
        scale_mm = self.app.state.scale_m_per_chord() * 1000.0
        self.metrics_var.set(
            f"throat {t['width']:.4f} c_ax ({t['width'] * scale_mm:.2f} mm) "
            f"@ x/c_ax {t['x_over_cax']:.3f}   |   "
            f"\u03b1_throat {t['angle_throat_deg']:.1f}\u00b0   |   "
            f"unguided {t['unguided_turning_deg']:.1f}\u00b0")

    # -------------------------------------------------------------- draw
    def _draw(self, g):
        ax = self.ax
        ax.clear()
        outline = g["outline"]

        # passage envelope + ghost blades for context
        ax.plot(g["bot"][:, 0], g["bot"][:, 1], "--", color="0.55", lw=1,
                label="periodic edges")
        ax.plot(g["top"][:, 0], g["top"][:, 1], "--", color="0.55", lw=1)
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

    def _draw_channel(self, g):
        fig, ax, cvs = self._chart_tabs["Channel"]
        ax.clear()
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
