"""ML tab: dataset building, CFD hole-filling, MLP training, prediction.

Workflow: pick the inputs X (from the design-variable catalog - pyturbo
parameters, domain values, FFD cage offsets, plugin parameters) and the
outputs y (results.json quantities). Create a dataset, then fill it with
X points: Latin-hypercube sampling over each input's exploration range
(editably [min, max] per column, saved with the dataset), a CSV import,
or both. Rows whose y is unknown are "holes" - "Fill missing Y"
runs a full CFD case per hole (in a worker thread, sequentially) and
records the results into a local SQLite database (ml_data.db). The
"Train MLP" page fits a deep multilayer-perceptron regressor (pure
numpy, see ml_net.py) on the filled rows, showing the loss curve and a
parity plot; the "Predict" panel evaluates the trained network for
hand-entered X values.
"""

import csv
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from foilage_config import resolve_su2_executable   # noqa: E402

from .ml_net import predict as mlp_predict
from .ml_net import train_mlp
from .ml_store import DEFAULT_DB, MLStore
from .ml_runner import MLSamplingRun
from .optimizer import (constraint_quantities_for_case,
                        design_variables_for, lhs_sample)
from .state import get_path
from .widgets import ScrolledFrame, Tooltip        # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DB_PATH = DEFAULT_DB
POLL_MS = 150


class MLTab(ttk.Frame):
    def __init__(self, app):
        super().__init__(app.notebook)
        self.app = app
        self.store = MLStore(DB_PATH)
        self.ds_id = None
        self.ds = None
        self._x_specs = []             # full DV catalog for the listbox
        self._ml_run = None
        self._ml_queue = queue.Queue()
        self._pending_ids = []         # sample ids behind the runner points
        self.model = None
        self.train_info = None
        self._train_queue = queue.Queue()
        self._train_busy = False
        self._predict_entries = []
        self._range_vars = {}          # x column -> (min var, max var)
        self._ranges_loading = False

        # ------------------------------------------------------------- bar
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=(4, 0))
        self.status_var = tk.StringVar(
            value="pick X and y, create a dataset, then generate or "
                  "import samples")
        tk.Label(bar, textvariable=self.status_var, anchor="w",
                 font=("TkDefaultFont", 8, "bold")).pack(
            side="left", fill="x", expand=True)
        self.train_btn = tk.Button(bar, text="Train MLP", state="disabled",
                                   bg="#dff0d8", command=self._train)
        self.train_btn.pack(side="right", padx=2)
        self.stop_btn = tk.Button(bar, text="Stop sampling", state="disabled",
                                  fg="#a00", command=self._stop_sampling)
        self.stop_btn.pack(side="right", padx=2)
        self.fill_btn = tk.Button(bar, text="Fill missing Y",
                                  state="disabled",
                                  command=self._start_fill)
        self.fill_btn.pack(side="right", padx=2)

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ------------------------------------------------------------ left
        left = ttk.Frame(pane, width=470)
        left.pack_propagate(False)
        pane.add(left, weight=1)
        scroll = ScrolledFrame(left)
        scroll.pack(fill="both", expand=True)
        panel = scroll.inner

        # ------------------------------------------------------------ data
        ds_box = ttk.LabelFrame(panel, text=" Dataset ")
        ds_box.pack(fill="x", padx=6, pady=(6, 3))
        row = ttk.Frame(ds_box)
        row.pack(fill="x", padx=4, pady=2)
        tk.Label(row, text="name", width=10, anchor="w").pack(side="left")
        self.name_var = tk.StringVar()
        tk.Entry(row, textvariable=self.name_var, width=24,
                 font=("Consolas", 9)).pack(side="left", padx=(2, 6))
        ttk.Button(row, text="Create / open", width=14,
                   command=self._create_or_open).pack(side="left")
        row2 = ttk.Frame(ds_box)
        row2.pack(fill="x", padx=4, pady=(0, 3))
        tk.Label(row2, text="existing", width=10, anchor="w").pack(side="left")
        self.ds_box_combo = ttk.Combobox(row2, state="readonly", width=26)
        self.ds_box_combo.pack(side="left", padx=(2, 4))
        self.ds_box_combo.bind("<<ComboboxSelected>>", self._on_dataset_pick)
        ttk.Button(row2, text="Delete…", width=8,
                   command=self._delete_dataset).pack(side="left")
        self._refresh_dataset_combo()
        tk.Label(ds_box, justify="left", wraplength=430,
                  font=("TkDefaultFont", 8), fg="#595959",
                  text="X = inputs to vary (design-variable catalog, per "
                       "case source); y = results.json quantities to "
                       "learn. Both are multi-select.").pack(
            fill="x", padx=4, pady=(0, 2))

        # ------------------------------------------------------------- X/y
        xy = ttk.Frame(panel)
        xy.pack(fill="x", padx=0)
        x_box = ttk.LabelFrame(xy, text=" Inputs X (ctrl-click multi) ")
        x_box.pack(side="left", fill="both", expand=True, padx=(6, 3))
        self.x_list = tk.Listbox(x_box, selectmode="multiple",
                                 exportselection=False, height=12,
                                 font=("Consolas", 8))
        self.x_list.pack(fill="both", expand=True, padx=3, pady=3)
        y_box = ttk.LabelFrame(xy, text=" Outputs y (ctrl-click multi) ")
        y_box.pack(side="left", fill="both", expand=True, padx=(3, 6))
        self.y_list = tk.Listbox(y_box, selectmode="multiple",
                                 exportselection=False, height=12,
                                 font=("Consolas", 8))
        self.y_list.pack(fill="both", expand=True, padx=3, pady=3)
        self._rebuild_pick_lists()

        # -------------------------------------------------- X ranges (LHS)
        self.range_box = ttk.LabelFrame(
            panel, text=" X ranges (LHS bounds, editable) ")
        self.range_box.pack(fill="x", padx=6, pady=3)
        self._rebuild_range_panel()

        # --------------------------------------------------------- samples
        smp_box = ttk.LabelFrame(panel, text=" Samples ")
        smp_box.pack(fill="x", padx=6, pady=3)
        row = ttk.Frame(smp_box)
        row.pack(fill="x", padx=4, pady=2)
        tk.Label(row, text="LHS samples", width=10,
                 anchor="w").pack(side="left")
        self.lhs_var = tk.StringVar(value="20")
        tk.Entry(row, textvariable=self.lhs_var, width=6, justify="right",
                 font=("Consolas", 9)).pack(side="left", padx=(2, 6))
        ttk.Button(row, text="Generate LHS samples", width=22,
                   command=self._generate_lhs).pack(side="left")
        row = ttk.Frame(smp_box)
        row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Import CSV\u2026", width=13,
                   command=self._import_csv).pack(side="left", padx=(0, 4))
        ttk.Button(row, text="Export CSV\u2026", width=13,
                   command=self._export_csv).pack(side="left")
        self.counts_var = tk.StringVar(value="no dataset open")
        tk.Label(smp_box, textvariable=self.counts_var, anchor="w",
                 font=("TkDefaultFont", 8), fg="#595959").pack(
            fill="x", padx=4, pady=(0, 3))

        # ------------------------------------------------------ run cases
        run_box = ttk.LabelFrame(panel, text=" Fill missing Y (runs CFD) ")
        run_box.pack(fill="x", padx=6, pady=3)
        self.run_vars = {}
        for key, label, default, tip in (
                ("threads", "SU2 threads", getattr(app, "threads", 4),
                 "Threads per SU2 solve."),
                ("max_iterations", "SU2 max iterations (blank = case value)",
                 app.state.get("solver_settings.max_iterations"),
                 "Applied to every sampled case only."),
                ("stage_timeout", "Stage timeout [min] (blank = none)", "30",
                 "Kill a hanging stage and mark the sample failed.")):
            row = ttk.Frame(run_box)
            row.pack(fill="x", padx=4, pady=1)
            tk.Label(row, text=label, width=32, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            var = tk.StringVar(value="" if default is None else str(default))
            e = tk.Entry(row, textvariable=var, width=8, justify="right",
                         font=("Consolas", 9), relief="solid", bd=1)
            e.pack(side="left")
            Tooltip(e, tip)
            self.run_vars[key] = var

        # ---------------------------------------------------------- train
        tr_box = ttk.LabelFrame(panel, text=" Train MLP ")
        tr_box.pack(fill="x", padx=6, pady=3)
        for key, label, default, tip in (
                ("hidden", "Hidden layers", "128,64,32",
                 "Comma-separated layer sizes - a deep stacked MLP."),
                ("epochs", "Epochs", "500",
                 "Training passes (early stopping cuts this short)."),
                ("lr", "Learning rate", "0.001", "Adam step size."),
                ("batch", "Batch size", "32", "Mini-batch size."),
                ("val_frac", "Validation fraction", "0.2",
                 "Hold-out fraction for the loss curve and parity plot.")):
            row = ttk.Frame(tr_box)
            row.pack(fill="x", padx=4, pady=1)
            tk.Label(row, text=label, width=32, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            var = tk.StringVar(value=str(default))
            e = tk.Entry(row, textvariable=var, width=10, justify="right",
                         font=("Consolas", 9), relief="solid", bd=1)
            e.pack(side="left")
            Tooltip(e, tip)
            self.run_vars.setdefault("train", {})[key] = var

        # -------------------------------------------------------- predict
        pr_box = ttk.LabelFrame(panel, text=" Predict y from X ")
        pr_box.pack(fill="x", padx=6, pady=(3, 8))
        self.predict_frame = ttk.Frame(pr_box)
        self.predict_frame.pack(fill="x", padx=4, pady=2)
        ttk.Button(pr_box, text="Predict", command=self._predict).pack(
            anchor="e", padx=4, pady=(0, 2))
        self.predict_var = tk.StringVar(value="train a model first")
        tk.Label(pr_box, textvariable=self.predict_var, anchor="w",
                 justify="left", wraplength=430,
                 font=("Consolas", 8)).pack(fill="x", padx=4, pady=(0, 3))

        # ----------------------------------------------------------- right
        right = ttk.Frame(pane)
        pane.add(right, weight=3)
        book = ttk.Notebook(right)
        book.pack(fill="both", expand=True)

        page_samples = ttk.Frame(book)
        book.add(page_samples, text=" samples ")
        cols = ("#", "status")
        self.tree = ttk.Treeview(page_samples, columns=cols, show="headings",
                                 height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=60 if c == "#" else 80, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)

        page_loss = ttk.Frame(book)
        book.add(page_loss, text=" training ")
        self.loss_fig = Figure(figsize=(7.4, 4.4), dpi=96)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, master=page_loss)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)

        page_parity = ttk.Frame(book)
        book.add(page_parity, text=" parity ")
        self.parity_fig = Figure(figsize=(7.4, 4.4), dpi=96)
        self.parity_ax = self.parity_fig.add_subplot(111)
        self.parity_canvas = FigureCanvasTkAgg(self.parity_fig,
                                               master=page_parity)
        self.parity_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.parity_note = tk.Label(page_parity, text="", anchor="w",
                                    font=("TkDefaultFont", 8), fg="#595959")
        self.parity_note.pack(fill="x")

        self._refresh_dataset_combo()
        self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------ properties
    @property
    def sampling_running(self):
        return self._ml_run is not None and self._ml_run.running

    # ------------------------------------------------------------ data setup
    def _rebuild_pick_lists(self):
        """Fill the X/Y listboxes from the mode-dependent catalogs."""
        self._x_specs = design_variables_for(
            self.app.state.get("airfoil_source.type") or "pyturbo",
            morph_n=int(self.app.state.get("airfoil_source.morph.n") or 0)
            if self.app.state.get("airfoil_source.morph.n") else 0,
            wedge=(self.app.state.get("domain.periodicity")
                   == "axisymmetric3d"))
        freestream = (self.app.state.get("domain.periodicity")
                      or "axisymmetric") == "freestream"
        self._y_specs = constraint_quantities_for_case(
            freestream,
            wedge=self.app.state.get("domain.periodicity")
            == "axisymmetric3d")
        self.x_list.delete(0, "end")
        self.y_list.delete(0, "end")
        for spec in self._x_specs:
            self.x_list.insert("end", f"{spec['label']}  [{spec['path']}]")
        for spec in self._y_specs:
            self.y_list.insert("end", f"{spec['label']}  [{spec['path']}]")

    def on_state_changed(self, paths):
        if any(p.startswith(("airfoil_source.", "domain."))
               for p in paths):
            self._rebuild_pick_lists()

    def _selected(self, listbox, specs):
        return [specs[i] for i in listbox.curselection()]

    def _create_or_open(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showinfo("Dataset", "Enter a dataset name first.")
            return
        existing = {d["name"]: d for d in self.store.datasets()}
        if name in existing:
            self._open_dataset(existing[name])
            return
        xs = self._selected(self.x_list, self._x_specs)
        ys = self._selected(self.y_list, self._y_specs)
        if not xs or not ys:
            messagebox.showerror(
                "Dataset", "Select at least one X column and one y column.")
            return
        bounds = {d["path"]: (float(d["min"]), float(d["max"]))
                  for d in xs
                  if d.get("min") is not None and d.get("max") is not None}
        ds_id = self.store.create_dataset(
            name, [d["path"] for d in xs], [d["path"] for d in ys],
            x_bounds=bounds)
        self._open_dataset(self.store.dataset(ds_id))

    def _delete_dataset(self):
        """Delete the open dataset and all of its samples (confirmed)."""
        if self.ds is None:
            messagebox.showinfo("ML", "No dataset is open.")
            return
        name = self.ds["name"]
        n = len(self.store.samples(self.ds_id))
        if not messagebox.askyesno(
                "Delete dataset",
                "Delete '" + name + f"' and its {n} sample(s)?\n"
                "This cannot be undone."):
            return
        self.store.delete_dataset(self.ds_id)
        self.ds_id, self.ds = None, None
        self._refresh_counts()
        self._refresh_tree()
        self._rebuild_range_panel()
        self._refresh_dataset_combo()
        self._build_predict_entries()
        self._update_buttons()
        self.status_var.set(f"dataset '{name}' deleted")

    def _on_dataset_pick(self, _e=None):
        sel = self.ds_box_combo.get()
        for d in self.store.datasets():
            if d["name"] == sel:
                self._open_dataset(d)
                return

    def _open_dataset(self, ds):
        self.ds_id, self.ds = ds["id"], ds
        self.name_var.set(ds["name"])
        self._refresh_counts()
        self._refresh_tree()
        self._rebuild_range_panel()
        self._build_predict_entries()
        self._update_buttons()
        self.status_var.set(f"dataset '{ds['name']}' open: "
                            f"{len(ds['x_columns'])} X, "
                            f"{len(ds['y_columns'])} y columns")

    def _refresh_dataset_combo(self):
        names = [d["name"] for d in self.store.datasets()]
        self.ds_box_combo.configure(values=names)
        if self.ds is not None and self.ds["name"] in names:
            self.ds_box_combo.set(self.ds["name"])

    # ------------------------------------------------------- x ranges (LHS)
    def _rebuild_range_panel(self):
        """(Re)create one editable min/max row per X column of the open
        dataset. Prefilled from the stored bounds, else the catalog
        defaults; edits persist with the dataset on Return/FocusOut."""
        self._range_vars = {}          # detach handlers before destroying
        self._ranges_loading = True
        try:
            for w in self.range_box.winfo_children():
                w.destroy()
            if self.ds is None:
                tk.Label(self.range_box, justify="left", wraplength=430,
                         font=("TkDefaultFont", 8), fg="#595959",
                         text="open a dataset to edit the exploration "
                              "ranges its LHS samples are drawn from.").pack(
                    fill="x", padx=4, pady=(0, 3))
                return
            specs = {s["path"]: s for s in self._x_specs}
            for col in self.ds["x_columns"]:
                spec = specs.get(col)
                b = self.ds["x_bounds"].get(col)
                if b is not None:
                    lo, hi = b
                elif spec is not None and spec.get("min") is not None \
                        and spec.get("max") is not None:
                    lo, hi = float(spec["min"]), float(spec["max"])
                else:
                    lo = hi = None
                row = ttk.Frame(self.range_box)
                row.pack(fill="x", padx=4)
                tk.Label(row, text=(spec["label"] if spec else
                                    col.split(".")[-1])[:30],
                         width=31, anchor="w",
                         font=("TkDefaultFont", 8)).pack(side="left")
                entries = []
                for val in (lo, hi):
                    var = tk.StringVar(
                        value="" if val is None else self._fmt(val))
                    e = tk.Entry(row, textvariable=var, width=10,
                                 justify="right", font=("Consolas", 8),
                                 relief="solid", bd=1)
                    e.pack(side="left", padx=1)
                    e.bind("<Return>",
                           lambda _e, c=col: self._save_bounds(c))
                    e.bind("<FocusOut>",
                           lambda _e, c=col: self._save_bounds(c))
                    entries.append(var)
                tip = col
                if spec is not None and spec.get("min") is not None:
                    tip += f"  (catalog: {spec['min']} .. {spec['max']})"
                Tooltip(row, tip)
                self._range_vars[col] = tuple(entries)
            btn_row = ttk.Frame(self.range_box)
            btn_row.pack(fill="x", padx=4, pady=(3, 2))
            ttk.Button(btn_row, text="Reset to catalog bounds", width=20,
                       command=self._reset_ranges).pack(side="left")
            tk.Label(btn_row, justify="left", font=("TkDefaultFont", 8),
                     fg="#595959",
                     text="edits save automatically - narrow the ranges "
                          "to keep sampled cases feasible").pack(
                side="left", padx=6)
        finally:
            self._ranges_loading = False

    def _save_bounds(self, col=None):
        """Validate the range entries and persist them with the dataset.
        Validates only `col` when given; returns the list of problems
        (also shown in the status bar)."""
        if self.ds is None or self._ranges_loading:
            return []
        stored = {c: tuple(b) for c, b in self.ds["x_bounds"].items()}
        problems = []
        for c, (lo_var, hi_var) in self._range_vars.items():
            if col is not None and c != col:
                continue
            lo_t, hi_t = lo_var.get().strip(), hi_var.get().strip()
            if not lo_t and not hi_t:
                stored.pop(c, None)       # cleared -> no stored range
                continue
            try:
                lo = float(lo_t.replace(",", ".")) if lo_t else None
                hi = float(hi_t.replace(",", ".")) if hi_t else None
            except ValueError:
                problems.append(f"'{c}': min/max must be numbers")
                continue
            if lo is None or hi is None:
                problems.append(f"'{c}': needs both a min and a max")
                continue
            if not lo < hi:
                problems.append(f"'{c}': min must be below max")
                continue
            stored[c] = (lo, hi)
        if problems:
            self.status_var.set("X ranges: " + problems[0])
            return problems
        self.store.set_x_bounds(self.ds_id, stored)
        self.ds["x_bounds"] = stored
        self.status_var.set("X ranges saved" +
                            (f" ({col})" if col else " for all columns"))
        return []

    def _reset_ranges(self):
        """Restore every range entry from the design-variable catalog."""
        if self.ds is None:
            return
        specs = {s["path"]: s for s in self._x_specs}
        for col, (lo_var, hi_var) in self._range_vars.items():
            spec = specs.get(col)
            if spec is not None and spec.get("min") is not None \
                    and spec.get("max") is not None:
                lo_var.set(self._fmt(float(spec["min"])))
                hi_var.set(self._fmt(float(spec["max"])))
        self._save_bounds()

    def _update_buttons(self):
        has = self.ds is not None
        pending = len(self.store.pending(self.ds_id)) if has else 0
        self.fill_btn.configure(
            state="normal" if has and pending and not self.sampling_running
            else "disabled")
        self.stop_btn.configure(
            state="normal" if self.sampling_running else "disabled")
        can_train = has and len(self.store.filled(self.ds_id)) >= 8 \
            and not self._train_busy
        self.train_btn.configure(
            state="normal" if can_train else "disabled")

    def _refresh_counts(self):
        if self.ds is None:
            self.counts_var.set("no dataset open")
            return
        samples = self.store.samples(self.ds_id)
        filled = sum(1 for s in samples if s["status"] == "filled")
        pending = sum(1 for s in samples if s["status"] == "pending")
        failed = sum(1 for s in samples if s["status"] == "failed")
        self.counts_var.set(
            f"{len(samples)} samples: {filled} filled, {pending} pending, "
            f"{failed} failed")

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        if self.ds is None:
            return
        samples = self.store.samples(self.ds_id)
        xcols, ycols = self.ds["x_columns"], self.ds["y_columns"]
        cols = ("#", "status") + tuple(xcols) + tuple(ycols)
        self.tree.configure(columns=cols)
        for i, c in enumerate(cols):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=46 if c == "#" else
                             (70 if c == "status" else 110), anchor="w")
        for s in samples:
            vals = ([f"{s['id']}", s["status"]]
                    + [self._fmt(s["x"].get(c)) for c in xcols]
                    + [self._fmt(s["y"].get(c)) for c in ycols])
            tag = "filled" if s["status"] == "filled" else s["status"]
            self.tree.insert("", "end", values=vals, tags=(tag,))
        self.tree.tag_configure("filled", background="#e7f6e7")
        self.tree.tag_configure("failed", background="#f6e0e0")

    @staticmethod
    def _fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.5g}"
        return str(v)

    # ------------------------------------------------------------- sampling
    def _generate_lhs(self):
        if self.ds is None:
            messagebox.showinfo("ML", "Create or open a dataset first.")
            return
        try:
            n = int(self.lhs_var.get())
        except ValueError:
            messagebox.showerror("ML", "LHS samples must be an integer.")
            return
        if n < 1:
            return
        problems = self._save_bounds()
        if problems:
            messagebox.showerror("ML", "Fix the X ranges first:\n"
                                 + "\n".join(problems))
            return
        missing = [c for c in self.ds["x_columns"]
                   if c not in self.ds["x_bounds"]]
        if missing:
            messagebox.showerror(
                "ML", "Every X column needs a min and max range to "
                      "generate samples; missing:\n" + "\n".join(missing))
            return
        bounds = [tuple(self.ds["x_bounds"][c])
                  for c in self.ds["x_columns"]]
        rng = np.random.default_rng()
        vectors = lhs_sample(bounds, n, rng)
        kinds = {d["path"]: d.get("kind") for d in self._x_specs}
        rows = []
        for v in vectors:
            row = {}
            for col, val in zip(self.ds["x_columns"], v):
                if kinds.get(col) == "int":
                    val = int(round(val))     # int-kind inputs stay ints
                row[col] = val
            rows.append(row)
        self.store.add_samples(self.ds_id, rows)
        self._after_data_change()
        self.status_var.set(f"generated {n} LHS samples - 'Fill missing Y' "
                            "runs the CFD for them")

    def _import_csv(self):
        if self.ds is None:
            messagebox.showinfo("ML", "Create or open a dataset first.")
            return
        path = filedialog.askopenfilename(title="Import samples CSV",
                                          defaultextension=".csv",
                                          filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            messagebox.showerror("ML", f"cannot read {path}:\n{e}")
            return
        xcols, ycols = self.ds["x_columns"], self.ds["y_columns"]
        kinds = {d["path"]: d.get("kind")
                 for d in self._x_specs if d["path"] in xcols}
        xs, ys = [], []
        for row in rows:
            xv = {}
            for c in xcols:
                val = self._num(row.get(c))
                if val is not None and kinds.get(c) == "int":
                    val = int(round(val))
                xv[c] = val
            yv = {c: self._num(row.get(c)) for c in ycols}
            xs.append(xv)
            ys.append(yv)
        self.store.add_samples(self.ds_id, xs, ys)
        self._after_data_change()
        self.status_var.set(f"imported {len(rows)} rows from {Path(path).name}")

    @staticmethod
    def _num(text):
        try:
            return float(str(text).strip().replace(",", "."))
        except (TypeError, ValueError):
            return None

    def _export_csv(self):
        if self.ds is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export samples CSV", defaultextension=".csv",
            initialfile=f"{self.ds['name']}.csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        xcols, ycols = self.ds["x_columns"], self.ds["y_columns"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(xcols + ycols)
            for s in self.store.samples(self.ds_id):
                writer.writerow([self._fmt(s["x"].get(c)) for c in xcols]
                                + [self._fmt(s["y"].get(c)) for c in ycols])
        self.status_var.set(f"exported {path}")

    def _after_data_change(self):
        self._refresh_counts()
        self._refresh_tree()
        self._update_buttons()

    # ---------------------------------------------------------- fill (CFD)
    def _start_fill(self):
        if self.sampling_running:
            return
        if self.app.job is not None and self.app.job.running:
            messagebox.showinfo("Busy", "A job is already running.")
            return
        if any(getattr(t, "opt_running", False) for t in self.app._tabs):
            messagebox.showinfo("Busy",
                                "An optimization is running - stop it first.")
            return
        try:
            su2_exe = resolve_su2_executable()
        except Exception as e:
            messagebox.showerror("SU2 not configured", str(e))
            return
        pending = self.store.pending(self.ds_id)
        if not pending:
            messagebox.showinfo("ML", "No pending samples - generate or "
                                      "import rows with missing y first.")
            return
        self._pending_ids = [s["id"] for s in pending]
        points = [s["x"] for s in pending]
        x_kinds = {d["path"]: d.get("kind")
                   for d in self._x_specs if d["path"] in
                   set(self.ds["x_columns"])}
        tag = f"{self.ds['name']}_ml_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = REPO / "runs" / tag
        timeout = self._run_var_float("stage_timeout")
        self._ml_run = MLSamplingRun(
            self.app.state.config, run_dir, tag, points,
            self.ds["x_columns"], self.ds["y_columns"],
            su2_exe=su2_exe, threads=self._run_var_int("threads", 4),
            max_iterations=self._run_var_int("max_iterations", None),
            stage_timeout=timeout * 60.0 if timeout else None,
            x_kinds=x_kinds)
        self._ml_run.start()
        self._update_buttons()
        self.status_var.set(f"filling {len(pending)} sample(s) - cases "
                            f"under {run_dir}")

    def _stop_sampling(self):
        if self._ml_run:
            self._ml_run.stop()
            self.status_var.set("stopping sampling ...")

    def _run_var_int(self, key, default):
        text = self.run_vars[key].get().strip()
        try:
            return int(text)
        except (TypeError, ValueError):
            return default

    def _run_var_float(self, key):
        text = self.run_vars[key].get().strip()
        try:
            return float(text.replace(",", "."))
        except (TypeError, ValueError):
            return None

    # ---------------------------------------------------------------- train
    def _train(self):
        samples = self.store.filled(self.ds_id)
        xcols, ycols = self.ds["x_columns"], self.ds["y_columns"]
        X = np.array([[s["x"].get(c) for c in xcols] for s in samples],
                     dtype=float)
        y = np.array([[s["y"].get(c) for c in ycols] for s in samples],
                     dtype=float)
        tr = self.run_vars.get("train", {})
        try:
            hidden = tuple(int(h) for h in
                           tr["hidden"].get().split(",") if h.strip())
            epochs = int(tr["epochs"].get())
            lr = float(tr["lr"].get())
            batch = int(tr["batch"].get())
            val_frac = float(tr["val_frac"].get())
        except ValueError as e:
            messagebox.showerror("ML", f"bad training setting: {e}")
            return
        self._train_busy = True
        self._update_buttons()
        self.status_var.set("training MLP ...")
        cfg = dict(hidden=hidden, epochs=epochs, lr=lr, batch=batch,
                   val_frac=val_frac, seed=0)

        def worker():
            try:
                model, info = train_mlp(X, y, **cfg)
                self._train_queue.put(("ok", model, info))
            except Exception as e:
                self._train_queue.put(("error", str(e), None))

        threading.Thread(target=worker, daemon=True,
                         name="mlp-train").start()

    def _predict(self):
        if self.model is None:
            messagebox.showinfo("ML", "Train an MLP first.")
            return
        xcols = self.ds["x_columns"]
        row = {}
        for col, var in zip(xcols, self._predict_entries):
            val = self._num(var.get())
            if val is None:
                messagebox.showerror("ML", f"enter a number for {col}")
                return
            row[col] = val
        pred = mlp_predict(self.model, [list(row.values())])[0]
        ycols = self.ds["y_columns"]
        self.predict_var.set("   ".join(
            f"{c.split('.')[-1]} = {p:.5g}" for c, p in zip(ycols, pred)))

    def _build_predict_entries(self):
        for w in self.predict_frame.winfo_children():
            w.destroy()
        self._predict_entries = []
        if self.ds is None:
            return
        xcols = self.ds["x_columns"]
        for i, col in enumerate(xcols):
            row = i // 2
            col_idx = (i % 2) * 2
            ttk.Label(self.predict_frame, text=col.split(".")[-1],
                      width=13, anchor="w",
                      font=("TkDefaultFont", 8)).grid(
                row=row, column=col_idx, sticky="w", padx=(0, 2))
            var = tk.StringVar()
            e = tk.Entry(self.predict_frame, textvariable=var, width=8,
                         justify="right", font=("Consolas", 8),
                         relief="solid", bd=1)
            e.grid(row=row, column=col_idx + 1, padx=(0, 8), pady=1)
            self._predict_entries.append(var)

    # ---------------------------------------------------------------- poll
    def _poll(self):
        self._drain_sampling()
        self._drain_training()
        self.after(POLL_MS, self._poll)

    def _drain_sampling(self):
        run = self._ml_run
        if run is None:
            return
        try:
            while True:
                kind, data = run.queue.get_nowait()
                if kind == "log":
                    self.status_var.set(data["line"][-120:])
                elif kind == "sample_done":
                    sid = self._pending_ids[data["index"]]
                    self.store.record_result(sid, data["y"], data["case"])
                    self._after_data_change()
                    self.status_var.set(
                        f"sample {data['index']} filled ({data['case']})")
                elif kind == "sample_failed":
                    sid = self._pending_ids[data["index"]]
                    self.store.record_failure(sid, data["error"],
                                              data["case"])
                    self._after_data_change()
                    self.status_var.set(
                        f"sample {data['index']} FAILED: "
                        f"{str(data['error'])[:100]}")
                elif kind == "done":
                    self._ml_run = None
                    self._after_data_change()
                    self.status_var.set(
                        f"sampling done: {data['filled']} filled, "
                        f"{data['failed']} failed")
        except queue.Empty:
            pass
        if not run.running and self._ml_run is run:
            self._ml_run = None
            self._after_data_change()

    def _drain_training(self):
        try:
            status, payload, info = self._train_queue.get_nowait()
        except queue.Empty:
            return
        self._train_busy = False
        self._update_buttons()
        if status == "error":
            self.status_var.set(f"training failed: {payload}")
            messagebox.showerror("ML", f"training failed:\n{payload}")
            return
        self.model, self.train_info = payload, info
        self._draw_loss()
        self._draw_parity()
        self.status_var.set(
            f"MLP trained ({info['epochs_run']} epochs, "
            f"val RMSE {info['val_rmse']:.4g}) - use the Predict panel")

    def _draw_loss(self):
        self.loss_ax.clear()
        hist = self.train_info["history"]
        if not hist:
            return
        ep = [h[0] for h in hist]
        self.loss_ax.plot(ep, [h[1] for h in hist], lw=1.2,
                          label="train MSE")
        self.loss_ax.plot(ep, [h[2] for h in hist], lw=1.2,
                          label="validation MSE")
        self.loss_ax.set_yscale("log")
        self.loss_ax.set_xlabel("epoch")
        self.loss_ax.set_ylabel("MSE (raw y units)")
        self.loss_ax.set_title("MLP training loss", fontsize=10, loc="left")
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_ax.legend(fontsize=7)
        self.loss_fig.tight_layout()
        self.loss_canvas.draw_idle()

    def _draw_parity(self):
        info = self.train_info
        self.parity_ax.clear()
        actual = np.atleast_2d(info["val_actual"].T).T
        pred = np.atleast_2d(info["val_pred"].T).T
        n_out = actual.shape[1] if actual.ndim > 1 else 1
        colors = ("tab:blue", "tab:red", "tab:green", "tab:orange")
        for k in range(n_out):
            a = actual[:, k] if actual.ndim > 1 else actual
            p = pred[:, k] if pred.ndim > 1 else pred
            self.parity_ax.scatter(a, p, s=22, alpha=0.75,
                                   color=colors[k % len(colors)],
                                   label=f"y{k + 1}")
        lo = float(min(actual.min(), pred.min()))
        hi = float(max(actual.max(), pred.max()))
        self.parity_ax.plot([lo, hi], [lo, hi], "k--", lw=0.9,
                            label="ideal")
        self.parity_ax.set_xlabel("actual")
        self.parity_ax.set_ylabel("MLP predicted")
        self.parity_ax.set_title(
            f"validation parity (RMSE {info['val_rmse']:.4g})",
            fontsize=10, loc="left")
        self.parity_ax.grid(True, alpha=0.3)
        self.parity_ax.legend(fontsize=7)
        self.parity_fig.tight_layout()
        self.parity_canvas.draw_idle()
        self.parity_note.configure(
            text=f"{info['n_train']} train / {info['n_val']} validation "
                 "samples")
