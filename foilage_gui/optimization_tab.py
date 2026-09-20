"""Optimization tab: constrained multi-objective optimization over geometry.

Left: pick objectives (results.json postprocessed quantities) with a
minimize/maximize choice each, set output constraints (e.g. minimum throat
width, corrected-flow limits - violations are handled with Deb's
constrained domination), pick design variables with min/avg/max
exploration ranges, and set run options including SU2 max iterations /
threads local to the optimization. The design-variable catalog follows
the airfoil source: pyturbo generator parameters, extension-plugin
parameters, and the FFD morph-cage offsets for imported/generated
sections (optimizing cage offsets enables the morph for every evaluated
case).
Right: live plot - Pareto front when several objectives are selected,
best fitness vs evaluation (or generation) when a single objective is -
plus a run log. A second page ("SU2 line convergence") follows the
evaluation currently in flight: rms residual lines and the Yp (or CL/CD)
convergence, read live from that case's history.csv.

Start launches an optimizer.OptimizationRun in a worker thread: every
individual is a full pipeline case (mesh, setup, SU2 solve, post-process)
in its own case folder; all evaluations are appended to evaluations.csv in
the run folder. The run stops when the feasible front's hypervolume has
not improved for the set patience, when the generation cap is hit, or via
Stop.

Pause holds the run after the case in flight finishes and Resume
continues it. Whenever a run pauses or finishes - and on demand via Save
state - its complete parameters and search history are written to
optimizer_state.json in the run folder. Load state... opens such a file:
the form is restored, the plot shows the recorded history, and Start
continues the optimization from that point in a new run folder.
"""

import copy
import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from run_monitor import HistoryTail                    # noqa: E402

from foilage_config import resolve_su2_executable   # noqa: E402

from .optimizer import (ALGORITHMS, CAGE_DV_DEFAULT_BOUND,
                        OptimizationRun, constraint_quantities_for_case,
                        design_variables_for, front_ids, objectives_for_case,
                        sanitize_name, validate_state)  # noqa: E402
from .state import get_path                        # noqa: E402
from .widgets import ScrolledFrame, Tooltip        # noqa: E402

POLL_MS = 150
PANEL_W = 500


class OptimizationTab(ttk.Frame):
    def __init__(self, app):
        super().__init__(app.notebook)
        self.app = app
        self.run = None                # active OptimizationRun
        self.records = []              # evaluated records for the live plot
        self.plot_objectives = []      # snapshot taken when the run starts
        self._plot_dirty = False       # coalesce redraws to once per poll
        self._loaded_state = None      # optimizer state waiting to resume
        self._loaded_path = None
        self._cage_amp = CAGE_DV_DEFAULT_BOUND
        self._cage_amp_var = tk.StringVar(value=str(self._cage_amp))
        self._dv_catalog = self._current_catalog()
        self._dv_sig = self._catalog_sig(self._dv_catalog)
        # SU2 line convergence of the evaluation currently in flight
        self._conv_tail = None
        self._conv_case = None

        # ------------------------------------------------------------ bar
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=(4, 0))
        self.status_var = tk.StringVar(
            value="select objectives and design variables, then Start")
        tk.Label(bar, textvariable=self.status_var, anchor="w",
                 font=("TkDefaultFont", 8, "bold")).pack(
            side="left", fill="x", expand=True)
        self.folder_btn = tk.Button(bar, text="Run folder", width=10,
                                    state="disabled",
                                    command=self._open_run_folder)
        self.folder_btn.pack(side="right", padx=2)
        self.load_btn = tk.Button(bar, text="Load state…", width=11,
                                  command=self._load_state_dialog)
        Tooltip(self.load_btn,
                "Open a saved optimizer_state.json: its objectives, "
                "constraints, design variables, run settings and full "
                "search history are restored, and 'Start optimization' "
                "continues the search from that point in a new run folder.")
        self.load_btn.pack(side="right", padx=2)
        self.save_btn = tk.Button(bar, text="Save state", width=10,
                                  state="disabled",
                                  command=self._save_state)
        Tooltip(self.save_btn,
                "Write the parameters and full search history to "
                "optimizer_state.json in the run folder (snapshotted after "
                "the evaluation in flight). The same file is written "
                "automatically whenever the run pauses or finishes.")
        self.save_btn.pack(side="right", padx=2)
        self.stop_btn = tk.Button(bar, text="Stop optimization",
                                  state="disabled", fg="#a00",
                                  command=self._stop)
        self.stop_btn.pack(side="right", padx=2)
        self.pause_btn = tk.Button(bar, text="Pause", width=8,
                                   state="disabled",
                                   command=self._pause_toggle)
        Tooltip(self.pause_btn,
                "Hold the run after the case currently evaluating "
                "finishes; the button becomes Resume. Pausing also saves "
                "optimizer_state.json.")
        self.pause_btn.pack(side="right", padx=2)
        self.start_btn = tk.Button(bar, text="Start optimization",
                                   state="normal", bg="#dff0d8",
                                   command=self._start)
        self.start_btn.pack(side="right", padx=2)

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ----------------------------------------------------------- left
        left = ttk.Frame(pane, width=PANEL_W)
        left.pack_propagate(False)
        pane.add(left, weight=1)
        scroll = ScrolledFrame(left)
        scroll.pack(fill="both", expand=True)
        panel = scroll.inner

        self.obj_box = ttk.LabelFrame(
            panel, text=" Objectives (postprocessed results.json values) ")
        self.obj_box.pack(fill="x", padx=6, pady=(6, 3))

        self.con_box = ttk.LabelFrame(
            panel, text=" Constraints (bound on a results.json quantity) ")
        self.con_box.pack(fill="x", padx=6, pady=3)
        self.obj_vars = {}
        self.con_vars = {}
        self._build_output_quantity_panels()

        self.dv_box = ttk.LabelFrame(
            panel, text=" Design variables (geometry to explore) ")
        self.dv_box.pack(fill="x", padx=6, pady=3)
        self._build_dv_panel()

        run_box = ttk.LabelFrame(panel, text=" Run settings ")
        run_box.pack(fill="x", padx=6, pady=3)
        self.run_vars = {}
        algorithm_row = ttk.Frame(run_box)
        algorithm_row.pack(fill="x", padx=4, pady=(3, 2))
        tk.Label(algorithm_row, text="Search algorithm", width=34,
                 anchor="w").pack(side="left")
        self.algorithm_var = tk.StringVar(value="NSGA-II")
        self.algorithm_box = ttk.Combobox(
            algorithm_row, textvariable=self.algorithm_var, state="readonly",
            width=24, values=ALGORITHMS)
        self.algorithm_box.pack(side="left")
        Tooltip(self.algorithm_box,
                "NSGA-II uses non-dominated sorting and crowding distance. "
                "Adaptive Hybrid Search mixes global differential-evolution "
                "proposals, genetic proposals, and local refinement.")
        algorithm_note = tk.Label(
            run_box, justify="left", wraplength=PANEL_W - 40,
            font=("TkDefaultFont", 8), fg="#595959",
            text="Adaptive Hybrid Search uses the same objectives, "
                 "constraints, evaluations, and plots.")
        algorithm_note.pack(fill="x", padx=4, pady=(0, 3))
        for key, label, default, tip_text in (
                ("pop_size", "Population size", "8",
                 "Individuals evaluated per generation. Each one is a full "
                 "CFD case - keep it small."),
                ("patience", "Convergence patience [generations]", "10",
                 "Stop when the hypervolume of the accumulated front has "
                 "improved by less than 0.5% for this many consecutive "
                 "generations (i.e. the optimum has stopped improving)."),
                ("max_generations", "Generation cap (0 = none)", "0",
                 "Hard stop after this many generations, regardless of "
                 "convergence."),
                ("stage_timeout", "Stage timeout [min] (blank = none)", "30",
                 "Kill a geometry+mesh / setup / solve / post-process stage "
                 "still running after this many minutes and record the "
                 "evaluation as failed (penalty fitness). Guards against "
                 "gmsh/SU2 hangs on degenerate geometries."),
                ("seed", "Random seed (blank = random)", "",
                 "Seed for the initial sampling and genetic operators; set "
                 "it to repeat a run.")):
            row = ttk.Frame(run_box)
            row.pack(fill="x", padx=4, pady=1)
            tk.Label(row, text=label, width=34, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            var = tk.StringVar(value=default)
            e = tk.Entry(row, textvariable=var, width=8, justify="right",
                         font=("Consolas", 9), relief="solid", bd=1)
            e.pack(side="left")
            Tooltip(e, tip_text)
            self.run_vars[key] = var
        # solver settings local to the optimization (base case untouched)
        case_iters = self.app.state.get("solver_settings.max_iterations")
        for key, label, default, tip_text in (
                ("max_iterations", "SU2 max iterations (blank = case value)",
                 case_iters,
                 "Applied to every evaluated case only - the base case "
                 "input.json is not modified. Smaller values make each "
                 "evaluation cheaper but noisier."),
                ("threads", "SU2 threads",
                 getattr(self.app, "threads", 6),
                 "Threads per SU2 solve, local to the optimization "
                 "(independent of the Setup tab).")):
            row = ttk.Frame(run_box)
            row.pack(fill="x", padx=4, pady=1)
            tk.Label(row, text=label, width=34, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            var = tk.StringVar(
                value="" if default is None else str(default))
            e = tk.Entry(row, textvariable=var, width=8, justify="right",
                         font=("Consolas", 9), relief="solid", bd=1)
            e.pack(side="left")
            Tooltip(e, tip_text)
            self.run_vars[key] = var
        self.require_conv = tk.BooleanVar(value=True)
        rc = ttk.Checkbutton(
            run_box, text="Require residual target (penalize unconverged "
                          "solves)", variable=self.require_conv)
        rc.pack(fill="x", padx=4, pady=(3, 4))
        Tooltip(rc, "Evaluations whose SU2 run did not reach "
                    "CONV_RESIDUAL_MINVAL get a penalty fitness and are "
                    "ignored by the optimizer.")

        # ---------------------------------------------------------- right
        right = ttk.Frame(pane)
        pane.add(right, weight=3)

        self.book = ttk.Notebook(right)
        self.book.pack(fill="both", expand=True)
        page_opt = ttk.Frame(self.book)
        self.book.add(page_opt, text=" optimization ")
        page_conv = ttk.Frame(self.book)
        self.book.add(page_conv, text=" SU2 line convergence ")

        plot_bar = ttk.Frame(page_opt)
        plot_bar.pack(fill="x")
        tk.Label(plot_bar, text="x axis:").pack(side="left", padx=(4, 2))
        self.xaxis_var = tk.StringVar(value="evaluation #")
        self.xaxis_box = ttk.Combobox(plot_bar, textvariable=self.xaxis_var,
                                      state="disabled", width=12,
                                      values=("evaluation #", "generation"))
        self.xaxis_box.pack(side="left")
        Tooltip(self.xaxis_box, "x axis of the single-objective fitness "
                                "plot: sequential evaluation number or the "
                                "generation each case belongs to.")
        tk.Label(plot_bar, text="obj x:").pack(side="left", padx=(12, 2))
        self.x_var = tk.StringVar()
        self.x_box = ttk.Combobox(plot_bar, textvariable=self.x_var,
                                  state="disabled", width=26)
        self.x_box.pack(side="left")
        tk.Label(plot_bar, text="obj y:").pack(side="left", padx=(10, 2))
        self.y_var = tk.StringVar()
        self.y_box = ttk.Combobox(plot_bar, textvariable=self.y_var,
                                  state="disabled", width=26)
        self.y_box.pack(side="left")
        for box in (self.xaxis_box, self.x_box, self.y_box):
            box.bind("<<ComboboxSelected>>", lambda _e: self.redraw())

        self.fig = Figure(figsize=(7.4, 5.2), dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=page_opt)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, page_opt,
                                            pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")

        # SU2 line-convergence page: the residual lines (and the Yp /
        # CL-CD convergence) of the evaluation currently in flight
        self.conv_note = tk.Label(page_conv, text="waiting for an "
                                  "evaluation ...", anchor="w",
                                  font=("TkDefaultFont", 8), fg="#595959")
        self.conv_note.pack(fill="x")
        self.conv_fig = Figure(figsize=(7.4, 5.2), dpi=96)
        self.conv_ax_res = self.conv_fig.add_subplot(211)
        self.conv_ax_obj = self.conv_fig.add_subplot(212)
        self.conv_canvas = FigureCanvasTkAgg(self.conv_fig, master=page_conv)
        self.conv_canvas.get_tk_widget().pack(fill="both", expand=True)

        log_box = ttk.LabelFrame(right, text=" Run log ")
        log_box.pack(fill="x", padx=4, pady=(0, 4))
        # NOT named self.log: the app broadcasts job log lines to any tab
        # with a log() method, which this tab must not intercept
        self.log_text = scrolledtext.ScrolledText(
            log_box, font=("Consolas", 8), height=9, state="disabled",
            wrap="none")
        self.log_text.pack(fill="both", expand=True, padx=2, pady=2)

        self._sync_axis_choices()
        self._plot_placeholder()
        self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _config_is_freestream(config):
        return (config.get("domain", {}).get("periodicity") or
                "axisymmetric") == "freestream"

    def _is_freestream_case(self):
        if self._loaded_state:
            return self._config_is_freestream(
                self._loaded_state.get("base_config") or {})
        return (self.app.state.get("domain.periodicity") or "axisymmetric") \
            == "freestream"

    def _build_output_quantity_panels(self):
        """Build mode-specific objective and constraint rows.

        The force coefficients are intentionally added only for freestream
        cases, where CL/CD are the natural external-airfoil outputs.  Preserve
        any selections that remain valid when the mode changes.
        """
        old_obj = {
            path: (v["check"].get(), v["sense"].get())
            for path, v in getattr(self, "obj_vars", {}).items()
        }
        old_con = {
            path: (v["check"].get(), v["min"].get(), v["max"].get())
            for path, v in getattr(self, "con_vars", {}).items()
        }
        for box in (self.obj_box, self.con_box):
            for child in box.winfo_children():
                child.destroy()

        hdr = ttk.Frame(self.obj_box)
        hdr.pack(fill="x", padx=4)
        tk.Label(hdr, text="", width=2).pack(side="left")
        tk.Label(hdr, text="quantity", width=30, anchor="w").pack(side="left")
        tk.Label(hdr, text="minimize", font=("TkDefaultFont", 8)).pack(
            side="left", padx=(24, 0))
        tk.Label(hdr, text="maximize", font=("TkDefaultFont", 8)).pack(
            side="left", padx=(8, 0))
        self.obj_vars = {}
        for obj in objectives_for_case(self._is_freestream_case()):
            row = ttk.Frame(self.obj_box)
            row.pack(fill="x", padx=4)
            check = tk.BooleanVar(value=old_obj.get(obj["path"],
                                                    (False, obj["sense"]))[0])
            sense = tk.StringVar(value=old_obj.get(obj["path"],
                                                   (False, obj["sense"]))[1])
            cb = ttk.Checkbutton(row, variable=check,
                                 command=lambda o=obj: self._obj_toggled(o))
            cb.pack(side="left")
            tk.Label(row, text=obj["label"], width=30, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            rb_min = ttk.Radiobutton(row, text="", value="min",
                                     variable=sense)
            rb_max = ttk.Radiobutton(row, text="", value="max",
                                     variable=sense)
            rb_min.pack(side="left", padx=(38, 0))
            rb_max.pack(side="left", padx=(14, 0))
            if not check.get():
                rb_min.configure(state="disabled")
                rb_max.configure(state="disabled")
            Tooltip(cb, f"results.json key: {obj['path']}")
            self.obj_vars[obj["path"]] = {
                "check": check, "sense": sense,
                "radios": (rb_min, rb_max), "spec": obj}
        note = tk.Label(
            self.obj_box, justify="left", wraplength=PANEL_W - 40,
            font=("TkDefaultFont", 8), fg="#595959",
            text=("Several objectives give a Pareto-front optimization; "
                  "a single objective is plotted as fitness vs evaluation. "
                  "Freestream mode adds lift-to-drag ratio L/D."))
        note.pack(fill="x", padx=4, pady=(2, 3))

        hdr = ttk.Frame(self.con_box)
        hdr.pack(fill="x", padx=4)
        tk.Label(hdr, text="", width=2).pack(side="left")
        tk.Label(hdr, text="quantity", width=26, anchor="w").pack(side="left")
        tk.Label(hdr, text="min (>=)", font=("TkDefaultFont", 8)).pack(
            side="left", padx=(10, 0))
        tk.Label(hdr, text="max (<=)", font=("TkDefaultFont", 8)).pack(
            side="left", padx=(12, 0))
        self.con_vars = {}
        for con in constraint_quantities_for_case(self._is_freestream_case()):
            row = ttk.Frame(self.con_box)
            row.pack(fill="x", padx=4)
            previous = old_con.get(con["path"], (False, "", ""))
            check = tk.BooleanVar(value=previous[0])
            cb = ttk.Checkbutton(row, variable=check,
                                 command=lambda c=con: self._con_toggled(c))
            cb.pack(side="left")
            tk.Label(row, text=con["label"], width=26, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            entries = []
            vars = {"check": check, "spec": con}
            for value in previous[1:]:
                var = tk.StringVar(value=value)
                e = tk.Entry(row, textvariable=var, width=8, justify="right",
                             font=("Consolas", 9), relief="solid", bd=1,
                             state="normal" if check.get() else "disabled")
                e.pack(side="left", padx=(6, 0))
                entries.append(e)
                vars["min" if len(entries) == 1 else "max"] = var
            vars["min_entry"], vars["max_entry"] = entries
            Tooltip(cb, f"results.json key: {con['path']}. Keep the "
                        f"quantity within [min, max]; leave a field blank "
                        f"for no bound on that side. Evaluations outside "
                        f"the bounds stay in the search (Deb's constrained "
                        f"domination) but never enter the Pareto front.")
            self.con_vars[con["path"]] = vars
        con_note = tk.Label(
            self.con_box, justify="left", wraplength=PANEL_W - 40,
            font=("TkDefaultFont", 8), fg="#595959",
            text=("Examples: 'Throat width' min 0.25 rejects narrower "
                  "throats; a corrected-flow min AND max keeps the operating "
                  "point inside a band. Freestream mode also supports an "
                  "L/D bound."))
        con_note.pack(fill="x", padx=4, pady=(2, 3))
        n = sum(v["check"].get() for v in self.obj_vars.values())
        self.status_var.set(
            f"{n} objective(s) selected" if n else
            "select objectives and design variables, then Start")

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            s = f"{v:.6g}"
            # strip float noise ("2.00000" -> "2") without eating integer
            # trailing zeros ("500" must stay "500")
            if "." in s or "e" in s or "E" in s:
                s = s.rstrip("0").rstrip(".")
            return s or "0"
        return str(v)

    # ------------------------------------------------------ form dynamics
    def _obj_toggled(self, obj):
        on = self.obj_vars[obj["path"]]["check"].get()
        state = "normal" if on else "disabled"
        for rb in self.obj_vars[obj["path"]]["radios"]:
            rb.configure(state=state)
        self._sync_axis_choices()
        n = sum(v["check"].get() for v in self.obj_vars.values())
        self.status_var.set(
            f"{n} objective(s) selected" if n else
            "select objectives and design variables, then Start")

    def _con_toggled(self, con):
        vars = self.con_vars[con["path"]]
        state = "normal" if vars["check"].get() else "disabled"
        vars["min_entry"].configure(state=state)
        vars["max_entry"].configure(state=state)

    def reload_fields(self):
        """Refresh design-variable avg defaults after a case reload."""
        if self.run and self.run.running:
            return
        if self._loaded_state:
            return          # the form shows the loaded state; keep it
        for path, vars in self.dv_vars.items():
            cur = self._dv_current(path)
            if cur is not None:
                vars["entries"][1].set(self._fmt(cur))

    def _reset_ranges(self):
        ticks = {p: v["check"].get() for p, v in self.dv_vars.items()}
        self._build_dv_panel()
        for p, on in ticks.items():
            if p in self.dv_vars:
                self.dv_vars[p]["check"].set(on)

    # ------------------------------------------- design-variable catalog
    def _current_catalog(self):
        """Design-variable catalog for the active airfoil source."""
        src = self.app.state.get("airfoil_source.type") or "pyturbo"
        try:
            morph_n = int(self.app.state.get("airfoil_source.morph.n") or 0)
        except (TypeError, ValueError):
            morph_n = 0
        return design_variables_for(src, morph_n=morph_n if morph_n >= 2
                                    else 0)

    @staticmethod
    def _catalog_sig(catalog):
        return tuple((d["path"], d["label"], d["min"], d["max"])
                     for d in catalog)

    def on_state_changed(self, paths):
        """Rebuild the design-variable panel when the geometry source or
        the morph lattice changes (the catalog is source-specific)."""
        if self.opt_running:
            return
        if "domain.periodicity" in paths:
            self._build_output_quantity_panels()
            self._sync_axis_choices()
        if not any(p in ("airfoil_source.type", "airfoil_source.morph.n")
                   for p in paths):
            return
        catalog = self._current_catalog()
        sig = self._catalog_sig(catalog)
        if sig != self._dv_sig:
            self._dv_catalog = catalog
            self._dv_sig = sig
            ticks = {p: v["check"].get() for p, v in self.dv_vars.items()}
            self._build_dv_panel()
            for p, on in ticks.items():
                if p in self.dv_vars:
                    self.dv_vars[p]["check"].set(on)

    def _build_dv_panel(self):
        """(Re)create the design-variable rows from the active catalog."""
        for child in self.dv_box.winfo_children():
            child.destroy()
        self.dv_vars = {}
        hdr = ttk.Frame(self.dv_box)
        hdr.pack(fill="x", padx=4)
        tk.Label(hdr, text="", width=2).pack(side="left")
        tk.Label(hdr, text="parameter", width=26,
                 anchor="w").pack(side="left")
        for txt, w in (("min", 8), ("avg", 8), ("max", 8)):
            tk.Label(hdr, text=txt, width=w, anchor="e",
                     font=("TkDefaultFont", 8)).pack(side="left", padx=1)
        if any(d.get("cage") for d in self._dv_catalog):
            amp_row = ttk.Frame(self.dv_box)
            amp_row.pack(fill="x", padx=4, pady=(2, 0))
            tk.Label(amp_row, text="Cage offset bound (+/-)", width=26,
                     anchor="w", font=("TkDefaultFont", 9)).pack(side="left")
            e = tk.Entry(amp_row, textvariable=self._cage_amp_var, width=8,
                         justify="right", font=("Consolas", 9),
                         relief="solid", bd=1)
            e.pack(side="left")
            e.bind("<Return>", lambda _e: self._apply_cage_amplitude())
            e.bind("<FocusOut>", lambda _e: self._apply_cage_amplitude())
            Tooltip(e, "Symmetric min/max prefilled on every FFD cage "
                       "offset row (axial-chord units); edit individual "
                       "rows afterwards for finer control.")
        for dv in self._dv_catalog:
            row = ttk.Frame(self.dv_box)
            row.pack(fill="x", padx=4)
            check = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(row, variable=check)
            cb.pack(side="left")
            tk.Label(row, text=dv["label"][:26], width=26, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            lo, avg, hi = self._dv_defaults(dv)
            entries = []
            for val in (lo, avg, hi):
                var = tk.StringVar(value="" if val is None else self._fmt(val))
                e = tk.Entry(row, textvariable=var, width=8, justify="right",
                             font=("Consolas", 9), relief="solid", bd=1)
                e.pack(side="left", padx=1)
                entries.append(var)
            if dv.get("cage"):
                key, idx = dv["path"].split("@")
                Tooltip(cb, f"{key}[{idx}] - offset of one FFD cage control "
                            "point in axial-chord units. Optimizing cage "
                            "offsets enables the morph automatically for "
                            "every evaluated case.")
            else:
                Tooltip(cb, f"parameter: {dv['path']}")
            self.dv_vars[dv["path"]] = {"check": check, "entries": entries,
                                        "spec": dv}
        reset_row = ttk.Frame(self.dv_box)
        reset_row.pack(fill="x", padx=4, pady=(2, 3))
        ttk.Button(reset_row, text="Reset ranges from current case",
                   command=self._reset_ranges).pack(side="left")
        tip = tk.Label(self.dv_box, justify="left",
                       wraplength=PANEL_W - 40,
                       font=("TkDefaultFont", 8), fg="#595959",
                       text="avg is the starting point (individual #0 of the "
                            "first generation); the rest of the initial "
                            "population samples [min, max]. The catalog "
                            "follows the airfoil source: pyturbo parameters, "
                            "extension-plugin parameters, and the FFD "
                            "morph-cage offsets (the morph is enabled "
                            "automatically for evaluated cases; the passage "
                            "radius R2 is kept equal to R1).")
        tip.pack(fill="x", padx=4, pady=(0, 3))

    def _apply_cage_amplitude(self):
        try:
            amp = abs(float(self._cage_amp_var.get().replace(",", ".")))
        except ValueError:
            amp = self._cage_amp
        self._cage_amp = amp if amp > 0 else CAGE_DV_DEFAULT_BOUND
        self._cage_amp_var.set(self._fmt(self._cage_amp))
        self._reset_ranges()

    def _dv_current(self, path):
        """Current value of a design variable from the case state
        (list-element paths address the morph offset arrays)."""
        if "@" in path:
            base, idx = path.rsplit("@", 1)
            lst = self.app.state.get(base)
            try:
                return lst[int(idx)]
            except (TypeError, ValueError, IndexError):
                return 0.0
        return self.app.state.get(path)

    def _dv_defaults(self, dv):
        """(min, avg, max) prefilled for one catalog row; the avg comes
        from the current case (clamped into the bounds)."""
        lo, hi = float(dv["min"]), float(dv["max"])
        if dv.get("cage"):
            lo, hi = -self._cage_amp, self._cage_amp
        avg = self._dv_current(dv["path"])
        if avg is None:
            avg = dv.get("default")
        if avg is not None:
            avg = min(max(float(avg), lo), hi)
        return lo, avg, hi

    # ------------------------------------------------------------- start
    @property
    def opt_running(self):
        """True while an optimization run is active (app.run_job checks it)."""
        return self.run is not None and self.run.running

    def _start(self):
        if self.opt_running:
            messagebox.showinfo("Busy", "An optimization is already running.")
            return
        if self.app.job and self.app.job.running:
            messagebox.showerror(
                "Busy", "Another job is running in the Setup/Mesh tabs. "
                        "Stop it before starting an optimization.")
            return
        resume_state = self._loaded_state
        if resume_state is None:
            issues = [i[1] for i in self.app.state.validate()
                      if i[0] == "error"]
            if issues:
                messagebox.showerror("Cannot start", "\n".join(issues[:10]))
                return
            try:
                objectives, design_vars, options = self._collect()
            except ValueError as e:
                messagebox.showerror("Cannot start", str(e))
                return
        try:
            su2_exe = resolve_su2_executable()
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("SU2 not configured", str(e))
            return
        if not su2_exe.exists():
            messagebox.showerror("SU2 not found",
                                 f"SU2_CFD not found at {su2_exe}")
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        if resume_state is not None:
            # continue a saved run: everything comes from the state file
            tag = sanitize_name(resume_state.get("run_tag") or "opt")
            run_tag = f"{tag}_res{stamp}"
            run_dir = REPO / "runs" / run_tag
            try:
                self.run = OptimizationRun.from_state(
                    resume_state, run_dir, run_tag, su2_exe=su2_exe)
            except (ValueError, KeyError, TypeError) as e:
                messagebox.showerror(
                    "Cannot resume",
                    f"{self._loaded_path or 'The state file'} is not "
                    f"usable:\n{e}")
                return
            self.records = list(self.run.records)
            self.plot_objectives = list(resume_state["objectives"])
            status_txt = f"optimization resumed into: {run_dir}"
        else:
            base_name = sanitize_name(self.app.state.case_name())
            run_tag = f"{base_name}_opt_{stamp}"
            run_dir = REPO / "runs" / run_tag
            self.run = OptimizationRun(
                copy.deepcopy(self.app.state.config), run_dir, run_tag,
                objectives, design_vars, su2_exe=su2_exe, **options)
            self.records = []
            self.plot_objectives = objectives
            status_txt = f"optimization started: {run_dir}"
        self._loaded_state = None
        self._loaded_path = None
        self._sync_axis_choices()
        self._reset_eval_convergence()
        self.run.start()
        self._update_opt_buttons()
        self.folder_btn.configure(state="normal")
        self.app.set_status(status_txt)

    def _reset_eval_convergence(self):
        """Clear the SU2 line-convergence page for a fresh run."""
        self._conv_tail = None
        self._conv_case = None
        for ax in (self.conv_ax_res, self.conv_ax_obj):
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.conv_ax_res.text(0.5, 0.5, "waiting for an evaluation ...",
                              transform=self.conv_ax_res.transAxes,
                              ha="center", fontsize=9, color="0.4")
        self.conv_note.configure(text="SU2 line convergence - waiting for "
                                      "an evaluation ...")
        self.conv_fig.tight_layout()
        self.conv_canvas.draw_idle()

    def _collect(self):
        """Read the form into (objectives, design_vars, options).

        Raises ValueError with a user-facing message on bad input."""
        objectives = []
        for path, vars in self.obj_vars.items():
            if vars["check"].get():
                objectives.append({
                    "path": path,
                    "label": vars["spec"]["label"],
                    "sense": vars["sense"].get()})
        if not objectives:
            raise ValueError("Select at least one objective to optimize.")
        design_vars = []
        for path, vars in self.dv_vars.items():
            if not vars["check"].get():
                continue
            texts = [e.get().strip() for e in vars["entries"]]
            try:
                lo, avg, hi = (float(t.replace(",", ".")) for t in texts)
            except ValueError:
                raise ValueError(
                    f"'{vars['spec']['label']}': min/avg/max must all be "
                    "numbers.")
            if not lo < hi:
                raise ValueError(
                    f"'{vars['spec']['label']}': min must be below max.")
            if not lo <= avg <= hi:
                raise ValueError(
                    f"'{vars['spec']['label']}': avg must lie between min "
                    "and max.")
            design_vars.append({
                "path": path, "label": vars["spec"]["label"],
                "kind": vars["spec"]["kind"],
                "min": lo, "avg": avg, "max": hi})
        if not design_vars:
            raise ValueError("Select at least one design variable to "
                             "explore (tick its checkbox).")
        constraints = []
        for path, vars in self.con_vars.items():
            if not vars["check"].get():
                continue
            lo_text = vars["min"].get().strip()
            hi_text = vars["max"].get().strip()
            if not lo_text and not hi_text:
                raise ValueError(
                    f"Constraint '{vars['spec']['label']}' needs a min "
                    "and/or a max bound (leave one blank for a one-sided "
                    "constraint).")
            lo = hi = None
            try:
                if lo_text:
                    lo = float(lo_text.replace(",", "."))
                if hi_text:
                    hi = float(hi_text.replace(",", "."))
            except ValueError:
                raise ValueError(
                    f"Constraint '{vars['spec']['label']}': bounds must be "
                    "numbers.")
            if lo is not None and hi is not None and not lo < hi:
                raise ValueError(
                    f"Constraint '{vars['spec']['label']}': min must be "
                    "below max.")
            if lo is not None:
                constraints.append({
                    "path": path, "label": vars["spec"]["label"],
                    "op": ">=", "value": lo, "col": f"{path} (min)"})
            if hi is not None:
                constraints.append({
                    "path": path, "label": vars["spec"]["label"],
                    "op": "<=", "value": hi, "col": f"{path} (max)"})
        try:
            pop_size = int(self.run_vars["pop_size"].get())
            patience = int(self.run_vars["patience"].get())
            max_gen = int(self.run_vars["max_generations"].get())
        except ValueError:
            raise ValueError("Population, patience and generation cap must "
                             "be integers.")
        if pop_size < 4:
            raise ValueError("Population size must be at least 4.")
        if patience < 1:
            raise ValueError("Convergence patience must be at least 1 "
                             "generation.")
        if max_gen < 0:
            raise ValueError("Generation cap cannot be negative.")
        seed = self.run_vars["seed"].get().strip()
        seed = int(seed) if seed else None
        max_iter_text = self.run_vars["max_iterations"].get().strip()
        if max_iter_text:
            try:
                max_iterations = int(max_iter_text)
            except ValueError:
                raise ValueError("SU2 max iterations must be an integer.")
            if max_iterations < 10:
                raise ValueError("SU2 max iterations must be at least 10.")
        else:
            max_iterations = None
        threads_text = self.run_vars["threads"].get().strip()
        try:
            threads = int(threads_text) if threads_text else self.app.threads
        except ValueError:
            raise ValueError("SU2 threads must be an integer.")
        if not 1 <= threads <= 128:
            raise ValueError("SU2 threads must be between 1 and 128.")
        stage_timeout = None
        to_text = self.run_vars["stage_timeout"].get().strip()
        if to_text:
            try:
                stage_timeout = float(to_text.replace(",", "."))
            except ValueError:
                raise ValueError("Stage timeout must be a number of minutes.")
            if stage_timeout <= 0:
                raise ValueError("Stage timeout must be positive.")
            stage_timeout *= 60.0
        options = {"pop_size": pop_size, "patience": patience,
                   "max_generations": max_gen,
                   "algorithm": self.algorithm_var.get(),
                   "require_converged": self.require_conv.get(),
                   "seed": seed, "constraints": constraints,
                   "max_iterations": max_iterations, "threads": threads,
                   "stage_timeout": stage_timeout}
        return objectives, design_vars, options

    def _pause_toggle(self):
        if not (self.run and self.run.running):
            return
        # key off the button label: it only flips to "Resume" once the
        # worker actually paused, so re-clicking while pausing is harmless
        if str(self.pause_btn.cget("text")) == "Resume":
            self.run.resume()
            self.status_var.set("resuming ...")
        else:
            self.run.pause()
            self.status_var.set(
                "pausing - finishing the current case, then holding ...")

    def _save_state(self):
        if self.run:
            self.run.save_state()

    def _load_state_dialog(self):
        if self.opt_running:
            messagebox.showinfo(
                "Busy", "Stop the running optimization before loading "
                        "another state.")
            return
        if self.app.job and self.app.job.running:
            messagebox.showinfo("Busy", "A job is already running.")
            return
        initial = REPO / "runs"
        path = filedialog.askopenfilename(
            title="Open optimizer state",
            filetypes=(("Optimizer state", "optimizer_state*.json"),
                       ("JSON", "*.json"), ("All files", "*.*")),
            initialdir=str(initial) if initial.exists() else str(REPO))
        if not path:
            return
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
            validate_state(state)
        except Exception as e:
            messagebox.showerror(
                "Load failed", f"Cannot use {path} as optimizer state:\n{e}")
            return
        self._loaded_state = state
        self._loaded_path = Path(path)
        self._build_output_quantity_panels()
        self._apply_state_to_form(state)
        self.records = list(state["records"])
        self.plot_objectives = list(state["objectives"])
        self._sync_axis_choices()
        if self.records:
            self.redraw()
        else:
            self._plot_placeholder(
                "loaded state has no evaluations yet - press "
                "'Start optimization' to continue")
        self.status_var.set(
            f"loaded {len(self.records)} evaluation(s) "
            f"(generation {state['generation']}) from "
            f"{Path(path).name} - press Start optimization to continue")
        self._log_line(f"loaded optimizer state: {path}")

    def _apply_state_to_form(self, state):
        """Show the loaded state in the form (the resumed run itself uses
        the values stored in the state file)."""
        sobjs = {o["path"]: o for o in state["objectives"]}
        for path, vars in self.obj_vars.items():
            on = path in sobjs
            vars["check"].set(on)
            if on:
                vars["sense"].set(sobjs[path].get("sense", "min"))
            for rb in vars["radios"]:
                rb.configure(state="normal" if on else "disabled")
        scons = {}
        for c in state["constraints"]:
            scons.setdefault(c["path"], {})[c["op"]] = c["value"]
        for path, vars in self.con_vars.items():
            spec = scons.get(path)
            vars["check"].set(bool(spec))
            vars["min"].set(
                self._fmt(spec[">="]) if spec and ">=" in spec else "")
            vars["max"].set(
                self._fmt(spec["<="]) if spec and "<=" in spec else "")
            st = "normal" if spec else "disabled"
            vars["min_entry"].configure(state=st)
            vars["max_entry"].configure(state=st)
        sdvs = {d["path"]: d for d in state["design_vars"]}
        for path, vars in self.dv_vars.items():
            dv = sdvs.get(path)
            vars["check"].set(dv is not None)
            if dv:
                for var, key in zip(vars["entries"], ("min", "avg", "max")):
                    var.set(self._fmt(dv[key]))
        opts = state.get("options") or {}
        self.run_vars["pop_size"].set(str(opts.get("pop_size", 8)))
        self.run_vars["patience"].set(str(opts.get("patience", 10)))
        self.run_vars["max_generations"].set(
            str(opts.get("max_generations", 0)))
        to = opts.get("stage_timeout")
        self.run_vars["stage_timeout"].set(
            "" if to is None else f"{to / 60.0:g}")
        seed = opts.get("seed")
        self.run_vars["seed"].set("" if seed is None else str(seed))
        mi = opts.get("max_iterations")
        self.run_vars["max_iterations"].set(
            "" if mi is None else str(mi))
        self.run_vars["threads"].set(
            str(opts.get("threads", self.app.threads)))
        self.algorithm_var.set(state["algorithm"])
        self.require_conv.set(bool(opts.get("require_converged", True)))
        n = sum(v["check"].get() for v in self.obj_vars.values())
        self.status_var.set(f"{n} objective(s) selected (loaded state)")

    def _stop(self):
        if self.run and self.run.running:
            if messagebox.askokcancel(
                    "Stop", "Stop the optimization? Cases evaluated so far "
                            "stay on disk and in evaluations.csv, and the "
                            "parameters plus search history are saved to "
                            "optimizer_state.json in the run folder - load "
                            "that file later to continue this run."):
                self.run.stop()
                self.status_var.set("stopping ...")

    def _open_run_folder(self):
        if self.run:
            path = Path(self.run.run_dir)
            if path.exists():
                if hasattr(os, "startfile"):
                    os.startfile(str(path))
                else:
                    import subprocess
                    subprocess.Popen(["xdg-open", str(path)])

    # ------------------------------------------------------------ events
    def _poll(self):
        run = self.run
        if run:
            while True:
                try:
                    kind, data = run.queue.get_nowait()
                except Exception:
                    break
                self._on_event(kind, data)
            # a full matplotlib replot per evaluation starves the Tk
            # thread when evaluations finish faster than redraws - redraw
            # at most once per poll cycle instead
            if self._plot_dirty:
                self._plot_dirty = False
                self.redraw()
        if self._conv_tail is not None:
            try:
                added = self._conv_tail.poll()
            except Exception:
                added = 0
            if added:
                self._draw_eval_convergence()
        self.after(POLL_MS, self._poll)

    # --------------------------------------------- SU2 line convergence
    def _start_eval_convergence(self, case):
        """Follow the history.csv of the evaluation now in flight so the
        'SU2 line convergence' page shows its residual lines live."""
        self._conv_case = case
        self._conv_tail = HistoryTail(REPO / "cases" / case / "history.csv")
        self.conv_note.configure(
            text=f"SU2 line convergence - {case} (waiting for "
                 "history.csv ...)")
        for ax in (self.conv_ax_res, self.conv_ax_obj):
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.conv_ax_res.text(0.5, 0.5, "waiting for history.csv ...",
                              transform=self.conv_ax_res.transAxes,
                              ha="center", fontsize=9, color="0.4")
        self.conv_fig.tight_layout()
        self.conv_canvas.draw_idle()

    def _draw_eval_convergence(self):
        """Render the in-flight evaluation's convergence: rms residual
        lines on top, the loading convergence (Yp for periodic cases,
        CL/CD for freestream) below - from the eval's own history.csv."""
        tail = self._conv_tail
        self.conv_ax_res.clear()
        self.conv_ax_obj.clear()
        self.conv_ax_res.grid(True, alpha=0.3)
        self.conv_ax_obj.grid(True, alpha=0.3)
        xcol = "Inner_Iter" if "Inner_Iter" in (tail.columns or []) else None
        x = np.asarray(tail.get(xcol) or [], dtype=float)
        plotted = False
        for col in (tail.columns or []):
            if col.startswith("rms["):
                self.conv_ax_res.plot(x, tail.get(col), lw=1,
                                      label=col.replace("rms", "rms "))
                plotted = True
        self.conv_ax_res.set_title("SU2 residuals (log10)", fontsize=9,
                                   loc="left")
        if plotted:
            self.conv_ax_res.legend(fontsize=6, ncol=2, loc="upper right")
        base = self.run.base_config if self.run else {}
        freestream = (get_path(base, "domain.periodicity")
                      or "axisymmetric") == "freestream"
        if freestream:
            cd, cl = tail.get("CD"), tail.get("CL")
            if cd:
                self.conv_ax_obj.plot(x, cd, lw=1.1, color="tab:red",
                                      label="CD")
            if cl:
                self.conv_ax_obj.plot(x, cl, lw=1.1, color="tab:blue",
                                      label="CL")
            self.conv_ax_obj.set_title("force coefficients", fontsize=9,
                                       loc="left")
        else:
            p02 = tail.get("Avg_TotalPress(outlet)")
            p01 = get_path(base, "BCs.inlet.total pressure")
            p2 = get_path(base, "BCs.outlet.static pressure")
            if p02 and p01 and p2 and p01 > p2:
                self.conv_ax_obj.plot(
                    x, [(p01 - v) / (p01 - p2) for v in p02],
                    lw=1.2, color="tab:red", label="Yp")
                self.conv_ax_obj.legend(fontsize=6, loc="upper right")
            self.conv_ax_obj.set_title("total-pressure loss Yp",
                                       fontsize=9, loc="left")
        rows = tail.rows
        self.conv_note.configure(
            text=f"SU2 line convergence - {self._conv_case} "
                 f"({rows} iterations)")
        self.conv_fig.tight_layout()
        self.conv_canvas.draw_idle()

    def _on_event(self, kind, data):
        if kind == "log":
            self._log_line(data["line"])
        elif kind == "started":
            self.status_var.set(
                f"running - {data.get('algorithm', 'NSGA-II')} - "
                f"population {data['pop_size']}, "
                f"generation 0 in progress ...")
        elif kind == "eval_start":
            self.status_var.set(
                f"running - generation {data['gen']}, evaluating "
                f"{data['case']} ...")
            self._start_eval_convergence(data["case"])
        elif kind == "eval_done":
            self.records.append(data["record"])
            self._plot_dirty = True
        elif kind == "generation_done":
            gen = data["gen"]
            self.status_var.set(
                f"running - generation {gen} done ({len(self.records)} "
                f"cases) | {data['best_text']}")
        elif kind == "paused":
            self.pause_btn.configure(text="Resume")
            self.status_var.set(
                f"paused after {len(self.records)} evaluations - state "
                "saved, press Resume to continue")
        elif kind == "resumed":
            self.pause_btn.configure(text="Pause")
            self.status_var.set("running ...")
        elif kind == "state_saved":
            self._log_line(f"state saved: {data['path']} "
                           f"({data['count']} evaluations)")
        elif kind == "done":
            self.status_var.set(
                f"{'converged' if data['converged'] else 'finished'} - "
                f"{data['reason']} ({len(self.records)} cases)")
            self._log_line(f"=== {data['reason']} ===")
            self._update_opt_buttons()

    def _update_opt_buttons(self):
        running = self.opt_running
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.pause_btn.configure(state="normal" if running else "disabled",
                                 text="Pause")
        self.save_btn.configure(state="normal" if running else "disabled")

    def _log_line(self, line):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------- plots
    def _sync_axis_choices(self):
        """Fitness x-axis selector (single objective) and Pareto objective
        dropdowns (meaningful with 3+ plotted objectives)."""
        labels = [o["label"] for o in self.plot_objectives]
        for box, var in ((self.x_box, self.x_var), (self.y_box, self.y_var)):
            box.configure(values=labels)
        if labels:
            if self.x_var.get() not in labels:
                self.x_var.set(labels[0])
            if self.y_var.get() not in labels:
                self.y_var.set(labels[1] if len(labels) > 1 else labels[0])
        for box, enabled in (
                (self.xaxis_box, len(labels) == 1),
                (self.x_box, len(labels) > 2),
                (self.y_box, len(labels) > 2)):
            box.configure(state="readonly" if enabled else "disabled")

    def _plot_placeholder(self, message=None):
        self.ax.clear()
        self.ax.text(0.5, 0.5,
                     message or "select objectives + design variables and "
                                "press 'Start optimization'",
                     transform=self.ax.transAxes, ha="center", fontsize=10,
                     color="0.4")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def redraw(self):
        """Replot from self.records (called after every evaluation)."""
        objectives = self.plot_objectives
        if not objectives or not self.records:
            return
        self.ax.clear()
        if len(objectives) == 1:
            self._plot_fitness(objectives[0])
        else:
            self._plot_pareto(objectives)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_fitness(self, obj):
        sense = obj["sense"]
        by_gen = self.xaxis_var.get() == "generation"

        def is_failed(r):
            return r["F"] is None or r["F"][0] >= 1e11

        def is_infeasible(r):
            return not is_failed(r) and r.get("cv", 0.0) > 0.0

        valid = [r for r in self.records if not is_failed(r)]
        infeasible = [r for r in valid if is_infeasible(r)]
        feasible = [r for r in valid if not is_infeasible(r)]
        failed = [r for r in self.records if is_failed(r)]
        if not valid:
            self.ax.set_title(
                f"{obj['label']} - no valid evaluation yet "
                f"({len(failed)} failed/penalized)", fontsize=10)
            return
        xs = [(r["gen"] if by_gen else r["id"]) for r in valid]
        ys = [r["raw"][obj["path"]] for r in valid]
        if feasible:
            self.ax.scatter([r["id"] if not by_gen else r["gen"]
                             for r in feasible],
                            [r["raw"][obj["path"]] for r in feasible],
                            s=26, color="tab:blue",
                            label="evaluations (feasible)")
        if infeasible:
            self.ax.scatter([r["id"] if not by_gen else r["gen"]
                             for r in infeasible],
                            [r["raw"][obj["path"]] for r in infeasible],
                            marker="x", color="darkorange", s=34,
                            label=f"infeasible ({len(infeasible)})")
        if failed:
            span = (max(ys) - min(ys)) or (abs(min(ys)) or 1.0) * 0.1
            floor = min(ys) - 0.12 * span
            self.ax.scatter([r["id"] for r in failed], [floor] * len(failed),
                            marker="x", color="0.5", s=26,
                            label=f"failed/penalized ({len(failed)})")
        # best-so-far staircase over the feasible evaluations
        if feasible:
            best, best_xs, best_ys = None, [], []
            for r in sorted(feasible, key=lambda r: r["id"]):
                v = r["raw"][obj["path"]]
                if best is None or (sense == "min" and v < best) \
                        or (sense == "max" and v > best):
                    best = v
                best_xs.append(r["gen"] if by_gen else r["id"])
                best_ys.append(best)
            self.ax.step(best_xs, best_ys, where="post", color="tab:red",
                         lw=1.6, label="best so far")
        self.ax.set_xlabel("generation" if by_gen else "evaluation #")
        self.ax.set_ylabel(f"{obj['label']} "
                           f"({'lower is better' if sense == 'min' else 'higher is better'})")
        self.ax.set_title(f"fitness vs "
                          f"{'generation' if by_gen else 'evaluation'} - "
                          f"{obj['label']}", fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=8, loc="best")

    def _plot_pareto(self, objectives):
        labels = [o["label"] for o in objectives]
        xi = labels.index(self.x_var.get()) if self.x_var.get() in labels else 0
        yi = labels.index(self.y_var.get()) if self.y_var.get() in labels \
            else min(1, len(labels) - 1)
        valid = [r for r in self.records
                 if r["F"] is not None and max(r["F"]) < 1e11]
        if not valid:
            self.ax.set_title("Pareto front", fontsize=10)
            return
        F = [r["F"] for r in valid]
        CV = [r.get("cv", 0.0) for r in valid]
        front = set(front_ids(F, CV))
        infeasible = [r for k, r in enumerate(valid)
                      if k not in front and CV[k] > 0.0]
        others = [r for k, r in enumerate(valid)
                  if k not in front and CV[k] <= 0.0]
        if infeasible:
            self.ax.scatter(
                [r["raw"][objectives[xi]["path"]] for r in infeasible],
                [r["raw"][objectives[yi]["path"]] for r in infeasible],
                marker="x", s=30, color="darkorange",
                label=f"infeasible ({len(infeasible)})")
        if others:
            self.ax.scatter([r["raw"][objectives[xi]["path"]] for r in others],
                            [r["raw"][objectives[yi]["path"]] for r in others],
                            s=22, color="0.7", label="evaluated (feasible)")
        fr = [r for k, r in enumerate(valid) if k in front]
        fx = [r["raw"][objectives[xi]["path"]] for r in fr]
        fy = [r["raw"][objectives[yi]["path"]] for r in fr]
        self.ax.scatter(fx, fy, s=42, color="tab:red", zorder=4,
                        label=f"Pareto front ({len(fr)})")
        if len(objectives) == 2:
            order = sorted(range(len(fr)), key=lambda k: fx[k])
            self.ax.plot([fx[k] for k in order], [fy[k] for k in order],
                         color="tab:red", lw=1.0, alpha=0.6, zorder=3)
        for o, axis in ((objectives[xi], "x"), (objectives[yi], "y")):
            arrow = "(lower better)" if o["sense"] == "min" \
                else "(higher better)"
            lbl = f"{o['label']} {arrow}"
            if axis == "x":
                self.ax.set_xlabel(lbl, fontsize=9)
            else:
                self.ax.set_ylabel(lbl, fontsize=9)
        # orient the axes so right/up always means "better": a minimized
        # objective is inverted (small values towards the edge), a
        # maximized one keeps the normal direction
        if objectives[xi]["sense"] == "min":
            self.ax.invert_xaxis()
        if objectives[yi]["sense"] == "min":
            self.ax.invert_yaxis()
        self.ax.set_title("Pareto front (live)", fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=8, loc="best")
