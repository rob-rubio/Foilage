"""Setup tab: edit case/BC/solver settings, run mesh + solver."""

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from . import runner
from .schema import sections_for
from .widgets import FieldWidget, ScrolledFrame, Tooltip

DERIVED_LABELS = [
    ("gamma_eff", "gamma (effective)", None),
    ("scale_m", "mesh scale", lambda v: f"1 chord unit = {v:.4g} m"),
    ("pitch_le_m", "pitch at LE", lambda v: f"{v:.4f} m"),
    ("pitch_te_m", "pitch at TE", lambda v: f"{v:.4f} m"),
    ("pr", "pressure ratio p2/p01", lambda v: f"{v:.3f}"),
    ("M2", "isentropic exit Mach", lambda v: f"{v:.3f}"),
    ("V2", "isentropic exit speed", lambda v: f"{v:.1f} m/s"),
    ("T2", "isentropic exit temperature", lambda v: f"{v:.1f} K"),
    ("Re_axial", "Reynolds number (axial chord)", lambda v: f"{v:.3g}"),
]


class SetupTab(ttk.Frame):
    def __init__(self, app):
        super().__init__(app.notebook)
        self.app = app
        self.fields = {}

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ---------------------------------------------------------- form
        left = ttk.Frame(pane)
        pane.add(left, weight=3)
        form = ScrolledFrame(left)
        form.pack(fill="both", expand=True)
        for section in sections_for("setup"):
            box = ttk.LabelFrame(form.inner, text=f" {section.title} ")
            box.pack(fill="x", padx=6, pady=(8, 2))
            for spec in section.fields:
                w = FieldWidget(box, spec, self._commit)
                self.fields[spec.path] = w

        # ------------------------------------------------- derived + run
        right = ttk.Frame(pane)
        pane.add(right, weight=0)

        box = ttk.LabelFrame(right, text=" Derived quantities ")
        box.pack(fill="x", padx=6, pady=(8, 4))
        self.derived_vars = {}
        for key, label, fmt in DERIVED_LABELS:
            row = ttk.Frame(box)
            row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=label, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            var = tk.StringVar(value="-")
            tk.Label(row, textvariable=var, anchor="e",
                     font=("Consolas", 9)).pack(side="right")
            self.derived_vars[key] = (var, fmt)

        vbox = ttk.LabelFrame(right, text=" Validation ")
        vbox.pack(fill="x", padx=6, pady=4)
        self.validation_lbl = tk.Label(vbox, text="", justify="left",
                                       anchor="w", wraplength=330,
                                       font=("TkDefaultFont", 8))
        self.validation_lbl.pack(fill="x", padx=4, pady=2)

        rbox = ttk.LabelFrame(right, text=" Run ")
        rbox.pack(fill="x", padx=6, pady=4)

        row = ttk.Frame(rbox)
        row.pack(fill="x", padx=6, pady=(4, 2))
        tk.Label(row, text="Solver threads").pack(side="left")
        self.threads_var = tk.StringVar(value=str(app.threads))
        spin = tk.Spinbox(row, from_=1, to=64, textvariable=self.threads_var,
                          width=4, command=self._threads_changed)
        spin.pack(side="left", padx=6)
        spin.bind("<Return>", lambda _e: self._threads_changed())
        Tooltip(spin, "Threads passed to SU2_CFD (-t).")

        self.run_btn = tk.Button(rbox, text="Run mesh + solver",
                                 command=lambda: app.run_job("full"),
                                 bg="#dff0d8", font=("TkDefaultFont", 10, "bold"))
        self.run_btn.pack(fill="x", padx=6, pady=(4, 2))
        row2 = ttk.Frame(rbox)
        row2.pack(fill="x", padx=6)
        ttk.Button(row2, text="Mesh + case only", width=16,
                   command=lambda: app.run_job("setup")).pack(side="left",
                                                              padx=(0, 4))
        ttk.Button(row2, text="Solve only", width=11,
                   command=lambda: app.run_job("solve")).pack(side="left")
        Tooltip(self.run_btn, "Save input.json, then run geometry + Gmsh "
                              "mesh, SU2 case setup with periodic validation, "
                              "and the SU2 solver. Post-processing starts "
                              "automatically when the solve exits.")
        Tooltip(rbox, "Run a subset: mesh + case setup without solving, or "
                      "re-solve an existing turbine.cfg.")
        self.stop_btn = tk.Button(rbox, text="Stop", state="disabled",
                                  command=app.stop_job, fg="#a00")
        self.stop_btn.pack(fill="x", padx=6, pady=(4, 4))

        self.progress_var = tk.StringVar(value="idle")
        tk.Label(rbox, textvariable=self.progress_var, anchor="w",
                 font=("Consolas", 8), fg="#4d4d4d").pack(fill="x", padx=6,
                                                      pady=(0, 4))

        # ------------------------------------------------------------ log
        lbox = ttk.LabelFrame(self, text=" Pipeline log ")
        lbox.pack(fill="both", expand=False)
        self.log_box = scrolledtext.ScrolledText(
            lbox, height=9, font=("Consolas", 8), bg="#111", fg="#9f9",
            state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=2, pady=2)

        self.reload_fields()
        self.refresh_derived()

    # ------------------------------------------------------------- fields
    def _commit(self, spec, value):
        self.app.state.set(spec.path, value)

    def reload_fields(self):
        for path, widget in self.fields.items():
            widget.set_value(self.app.state.get(path))

    def on_state_changed(self, paths):
        self.refresh_derived()

    def refresh_derived(self):
        d = self.app.state.derived()
        for key, _label, fmt in DERIVED_LABELS:
            var, _fmt = self.derived_vars[key]
            if key == "gamma_eff":
                var.set(f"{d['gamma_eff']:.4f}  ({d['gamma_source']})")
                continue
            var.set(fmt(d[key]) if key in d else "-")
        issues = self.app.state.validate()
        if not issues:
            self.validation_lbl.configure(
                text="no issues detected", fg="#2a7a2a")
        else:
            lines = []
            for sev, msg in issues:
                mark = {"error": "ERROR", "warning": "WARN ", "info": "note "}[sev]
                lines.append(f"[{mark}] {msg}")
            self.validation_lbl.configure(text="\n".join(lines), fg="#8a6d00")
        self.validation_lbl._issues = issues

    def _threads_changed(self):
        try:
            self.app.threads = max(1, int(self.threads_var.get()))
        except ValueError:
            pass

    # -------------------------------------------------------------- job
    def on_job_event(self, kind, data):
        if kind == "stage":
            state = data["state"]
            mark = {"start": ">", "ok": "OK", "failed": "XX",
                    "stopped": "--"}[state]
            self.progress_var.set(
                f"[{mark}] {data['title']}"
                + (f" - {data['message']}" if data.get("message") else ""))
            if state == "start":
                self.log(f"\n=== {data['title']} ===\n")
        elif kind == "log":
            self.log(data["line"])
        elif kind == "done":
            ok = data["ok"]
            self.progress_var.set(
                f"{'finished' if ok else 'stopped / failed'}: {data['reason']}")
        self._update_buttons()

    def _update_buttons(self):
        running = self.app.job is not None and self.app.job.running
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")

    def log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text if text.endswith("\n") else text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
