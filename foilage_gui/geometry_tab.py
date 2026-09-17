"""Geometry tab: edit the airfoil/passage with sliders + exact entries.

The preview recomputes the pyturbo-aero profile and the periodic passage
envelope whenever a geometry field changes (debounced), on a worker
thread - the GUI stays responsive while splines rebuild.
"""

import queue
import sys
import threading
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

from pipeline.airfoil import build_airfoil            # noqa: E402
from pipeline.mesh_tris import pitch_profile, periodic_edges  # noqa: E402

from .schema import sections_for                       # noqa: E402
from .widgets import FieldWidget, ScrolledFrame        # noqa: E402

DEBOUNCE_MS = 250


def _compute_geometry(cfg):
    """Worker: build the airfoil + passage edges from the current config.

    Returns (data dict) or ("error", message)."""
    try:
        airfoil = build_airfoil(cfg["airfoil"])
        prof = pitch_profile(cfg.get("domain", {}), airfoil)
        x0 = float(cfg.get("domain", {}).get("x_min", -0.5))
        x1 = float(cfg.get("domain", {}).get("x_max", 2.5))
        bot, top = periodic_edges(prof, x0, x1)
        outline = airfoil["outline"]
        ghost_p = outline + np.column_stack(
            [np.zeros(len(outline)), prof["p"](outline[:, 0])])
        ghost_m = outline - np.column_stack(
            [np.zeros(len(outline)), prof["p"](outline[:, 0])])
        a1 = float(cfg["airfoil"]["alpha1"])
        a2 = float(cfg["airfoil"]["alpha2"])
        return {
            "ss": airfoil["ss"], "ps": airfoil["ps"], "outline": outline,
            "bot": bot, "top": top, "ghost_p": ghost_p, "ghost_m": ghost_m,
            "style": airfoil["style"], "ss_upper": airfoil["ss_upper"],
            "n_points": len(airfoil["ss"]), "x0": x0, "x1": x1,
            "turning": a1 - a2,
            "p_le": prof["p_le"], "p_te": prof["p_te"],
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

        hint = tk.Label(left, text="Sliders update the preview on release; "
                        "type exact values in the boxes and press Enter.",
                        font=("TkDefaultFont", 8), fg="#595959", wraplength=280,
                        justify="left")
        hint.pack(fill="x", padx=8, pady=4)

        # ---------------------------------------------------------- canvas
        right = ttk.Frame(pane)
        pane.add(right, weight=3)
        self.fig = Figure(figsize=(7.2, 5.4), dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        self.status_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.status_var, anchor="w",
                 font=("TkDefaultFont", 8), fg="#595959").pack(fill="x",
                                                            side="bottom")

        self.reload_fields()
        self.schedule_regeneration(immediate=True)

    # ------------------------------------------------------------- fields
    def _commit(self, spec, value):
        self.app.state.set(spec.path, value)

    def reload_fields(self):
        for path, widget in self.fields.items():
            widget.set_value(self.app.state.get(path))

    def on_state_changed(self, paths):
        if any(p.startswith(("airfoil.", "domain.")) for p in paths):
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
            self.status_var.set(
                f"{result['style'].upper()} profile  |  turning "
                f"{result['turning']:.1f} deg  |  pitch LE {result['p_le']:.4f}, "
                f"TE {result['p_te']:.4f}  |  {result['n_points']} points/side")
        # if edits arrived while computing, run once more with the latest
        if self._dirty_while_busy:
            self._dirty_while_busy = False
            self.schedule_regeneration(immediate=True)

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

        closed = np.vstack([outline, outline[0]])
        ax.fill(closed[:, 0], closed[:, 1], color="#dbe9f6", zorder=2)
        ax.plot(closed[:, 0], closed[:, 1], color="#1f4e79", lw=1.6, zorder=3)

        ss_lbl = f"suction side ({'upper' if g['ss_upper'] else 'lower'})"
        ps_lbl = f"pressure side ({'lower' if g['ss_upper'] else 'upper'})"
        ax.plot(g["ss"][:, 0], g["ss"][:, 1], color="tab:red", lw=1, alpha=0.7,
                label=ss_lbl, zorder=4)
        ax.plot(g["ps"][:, 0], g["ps"][:, 1], color="tab:blue", lw=1,
                alpha=0.7, label=ps_lbl, zorder=4)
        ax.plot(g["ss"][0, 0], g["ss"][0, 1], "k.", ms=7, zorder=5)
        ax.annotate("LE", g["ss"][0], textcoords="offset points",
                    xytext=(8, 8), fontsize=8)
        ax.annotate("TE", g["ss"][-1], textcoords="offset points",
                    xytext=(8, 8), fontsize=8)

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
