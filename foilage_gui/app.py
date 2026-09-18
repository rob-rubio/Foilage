"""Main Foilage GUI window: toolbar + tabs + shared job handling."""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foilage_config import resolve_su2_executable  # noqa: E402

from foilage_gui.state import CaseState  # noqa: E402
from foilage_gui.runner import build_job  # noqa: E402

POLL_MS = 120
REPO = Path(__file__).resolve().parent.parent


class FoilageApp:
    def __init__(self, root: tk.Tk, input_path: Path, threads: int = 6):
        self.root = root
        self.threads = threads
        self.job = None
        self._tabs = []

        self.state = CaseState(input_path)
        self.state.observe(self._broadcast_state)

        root.title("Foilage - 2D CFD pipeline")
        root.geometry("1280x860")
        root.minsize(1024, 680)
        self._set_app_icon()
        self._build_toolbar()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        from foilage_gui.setup_tab import SetupTab
        from foilage_gui.geometry_tab import GeometryTab
        from foilage_gui.mesh_tab import MeshTab
        from foilage_gui.solution_tab import SolutionTab
        for Tab in (GeometryTab, SetupTab, MeshTab, SolutionTab):
            tab = Tab(self)
            self.notebook.add(tab, text=f" {Tab.__name__[:-3]} ")
            self._tabs.append(tab)

        self.status_var = tk.StringVar(value="")
        status = ttk.Label(root, textvariable=self.status_var, anchor="w",
                           relief="sunken", font=("TkDefaultFont", 8))
        status.pack(fill="x", side="bottom")

        self.root.after(POLL_MS, self._pump)
        self._update_title()

    # ------------------------------------------------------------- brand
    def _set_app_icon(self):
        icon = REPO / "resources" / "Foilage_Icon.png"
        if not icon.exists():
            return
        try:
            self._icon_image = tk.PhotoImage(file=str(icon))
            self.root.iconphoto(True, self._icon_image)
        except Exception:
            pass                        # cosmetic only

    # ------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", side="top")

        ttk.Label(bar, text="input.json:").pack(side="left", padx=(6, 2))
        self.path_var = tk.StringVar(value=str(self.state.path))
        ttk.Entry(bar, textvariable=self.path_var, width=58).pack(side="left")
        ttk.Button(bar, text="Browse...", command=self._browse).pack(
            side="left", padx=(4, 2))
        ttk.Button(bar, text="Reload", command=self._reload).pack(
            side="left", padx=2)
        ttk.Button(bar, text="Save", command=self._save).pack(
            side="left", padx=2)
        ttk.Button(bar, text="Save as ...", command=self._save_as).pack(
            side="left", padx=2)
        self.dirty_lbl = tk.Label(bar, text="", fg="#b8860b",
                                  font=("TkDefaultFont", 9, "bold"))
        self.dirty_lbl.pack(side="left", padx=8)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Open input.json",
            initialdir=str(self.state.path.parent),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self._load(Path(path))

    def _reload(self):
        if self._confirm_discard():
            return
        self._load(self.state.path)

    def _load(self, path):
        try:
            self.state.load(path)      # notifies observers (tabs refresh)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self.path_var.set(str(path))
        for tab in self._tabs:
            if hasattr(tab, "reload_fields"):
                tab.reload_fields()
        # mesh/solution tabs rescan their files for the new case
        for tab in self._tabs:
            if hasattr(tab, "refresh_sources"):
                tab.refresh_sources()
            if hasattr(tab, "reload_all"):
                tab.reload_all()
        self._update_title()

    def _save(self, path=None):
        try:
            self.state.save(path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        self.path_var.set(str(self.state.path))
        self._update_title()
        self.set_status(f"saved {self.state.path}")

    def _save_as(self):
        path = filedialog.asksaveasfilename(
            title="Save input.json as", defaultextension=".json",
            initialfile=self.state.path.name,
            initialdir=str(self.state.path.parent))
        if path:
            self._save(Path(path))

    # -------------------------------------------------------------- state
    def _broadcast_state(self, paths):
        for tab in self._tabs:
            if hasattr(tab, "on_state_changed"):
                tab.on_state_changed(paths)
        self._update_title()

    def _update_title(self):
        name = self.state.case_name()
        dirty = "*" if self.state.dirty else ""
        self.root.title(f"Foilage - {name}{dirty}")
        self.dirty_lbl.configure(
            text="unsaved changes" if self.state.dirty else "")

    def set_status(self, text):
        self.status_var.set(text)

    # ---------------------------------------------------------------- jobs
    def run_job(self, mode):
        if self.job and self.job.running:
            messagebox.showinfo("Busy", "A job is already running.")
            return
        issues = [i for i in self.state.validate() if i[0] == "error"]
        for tab in self._tabs:                       # validate form fields
            if hasattr(tab, "fields"):
                for widget in tab.fields.values():
                    ok, err = widget.validate()
                    if not ok and err:
                        issues.append(("error", err))
        if issues:
            messagebox.showerror(
                "Cannot start",
                "Fix the following problems first:\n\n"
                + "\n".join(f"- {m or '(no details)'}"
                            for _s, m in issues[:10]))
            return

        su2_needed = mode in ("full", "setup", "solve")
        su2_exe = None
        if su2_needed:
            try:
                su2_exe = resolve_su2_executable()
            except (FileNotFoundError, ValueError) as e:
                messagebox.showerror("SU2 not configured", str(e))
                return
            if not su2_exe.exists():
                messagebox.showerror("SU2 not found",
                                     f"SU2_CFD not found at {su2_exe}")
                return

        if mode == "solve":
            if not (self.state.case_dir() / "turbine.cfg").exists():
                messagebox.showerror(
                    "No prepared case",
                    f"No turbine.cfg in {self.state.case_dir()}.\n"
                    "Run 'Run mesh + solver' or 'Mesh + case only' first.")
                return

        # auto-save so the pipeline stages read the form values
        if self.state.dirty:
            self._save()

        case_dir = self.state.case_dir()
        case_dir.mkdir(parents=True, exist_ok=True)
        if mode in ("full", "solve") and self.state.get(
                "solver_settings.restart"):
            if not (case_dir / "restart.dat").exists():
                messagebox.showerror(
                    "No restart file",
                    f"Initialize-from-solution is enabled, but "
                    f"{case_dir / 'restart.dat'} does not exist.\n"
                    "The restart file is written by every solve - run the "
                    "case once (with the option off) before warm-starting.")
                return
        self.job = build_job(self.state, mode, self.threads, su2_exe)
        self.job.restart = bool(self.state.get("solver_settings.restart"))
        self.job.start()
        for tab in self._tabs:
            if hasattr(tab, "_update_buttons"):
                tab._update_buttons()
        self.set_status(f"job '{mode}' started")

    def stop_job(self):
        if self.job and self.job.running:
            if messagebox.askokcancel("Stop", "Terminate the running job?"):
                self.job.stop()
                self.set_status("stopping ...")

    def _pump(self):
        """Drain job events onto the Tk thread and refresh tab states."""
        job = self.job
        if job:
            while True:
                try:
                    kind, data = job.queue.get_nowait()
                except Exception:
                    break
                if kind == "log":
                    for tab in self._tabs:
                        if hasattr(tab, "log"):
                            tab.log(data["line"])
                else:
                    for tab in self._tabs:
                        if hasattr(tab, "on_job_event"):
                            tab.on_job_event(kind, data)
                    if kind == "done":
                        self.set_status(f"job '{job.name}': {data['reason']}")
                        for tab in self._tabs:
                            if hasattr(tab, "_update_buttons"):
                                tab._update_buttons()
        self.root.after(POLL_MS, self._pump)

    # ------------------------------------------------------------- helpers
    def _confirm_discard(self):
        if not self.state.dirty:
            return False
        return not messagebox.askyesno(
            "Unsaved changes", "Discard unsaved changes to the current case?")

    def run(self):
        self.root.mainloop()
