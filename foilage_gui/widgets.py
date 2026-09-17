"""Smart Tk form widgets driven by schema.FieldSpec.

Each FieldWidget is one form row: label [+ unit] + an editor appropriate
to the kind (entry, slider+entry, dropdown, checkbox, array editor,
auto-or-value). Numeric editors validate input and flag invalid values in
red instead of silently accepting them.
"""

import math
import tkinter as tk
from tkinter import ttk

from .schema import FieldSpec

LABEL_W = 26
ENTRY_W = 9
INVALID_BG = "#ffd2d2"


class Tooltip:
    """Minimal hover tooltip for any widget."""

    def __init__(self, widget, text, delay=600):
        self.widget = widget
        self.text = text
        self._id = None
        self._tip = None
        if text:
            widget.bind("<Enter>", self._schedule, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<ButtonPress>", self._hide, add="+")
        self.delay = delay

    def _schedule(self, _e=None):
        self._cancel()
        self._id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None

    def _show(self):
        if self._tip:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=self.text, justify="left", wraplength=380,
                       background="#ffffe0", relief="solid", borderwidth=1,
                       font=("TkDefaultFont", 8), padx=6, pady=3)
        lbl.pack()

    def _hide(self, _e=None):
        self._cancel()
        if self._tip:
            self._tip.destroy()
            self._tip = None


class ScrolledFrame(ttk.Frame):
    """Vertical-scrollable container frame (mouse wheel + scrollbar)."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")
        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _on_configure(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)

    def _on_wheel(self, e):
        # route the wheel to whichever scrolled frame the pointer is over
        w = self.winfo_containing(e.x_root, e.y_root)
        while w is not None:
            if w is self.canvas or w is self.inner:
                delta = -1 if getattr(e, "delta", 0) > 0 or e.num == 4 else 1
                self.canvas.yview_scroll(delta, "units")
                return
            w = w.master if hasattr(w, "master") else None


def parse_float(text):
    return float(text.strip().replace(",", "."))


def parse_int(text):
    f = float(text.strip())
    if f != int(f):
        raise ValueError(f"{text!r} is not an integer")
    return int(f)


class FieldWidget:
    """One schema field rendered as a form row."""

    def __init__(self, parent, spec: FieldSpec, on_commit):
        self.spec = spec
        self.on_commit = on_commit      # callback(spec, value)
        self._slider_guard = False
        self._last_committed = None
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", padx=4, pady=1)

        lbl = tk.Label(self.frame, text=spec.label, width=LABEL_W,
                       anchor="w", font=("TkDefaultFont", 9))
        lbl.pack(side="left")
        if spec.unit:
            tk.Label(self.frame, text=f"[{spec.unit}]", font=("TkDefaultFont", 8),
                     fg="#737373").pack(side="left", padx=(0, 4))
        Tooltip(lbl, spec.tooltip)

        builder = {
            "float": self._build_numeric,
            "int": self._build_numeric,
            "str": self._build_str,
            "bool": self._build_bool,
            "choice": self._build_choice,
            "float_array": self._build_array,
            "nullable_float": self._build_nullable,
            "nullable_int": self._build_nullable,
        }[spec.kind]
        builder()

    # ------------------------------------------------------------ editors
    def _build_numeric(self):
        self.var = tk.StringVar()
        self.entry = tk.Entry(self.frame, textvariable=self.var,
                              width=ENTRY_W, font=("Consolas", 9),
                              justify="right", relief="solid", bd=1)
        self.entry.pack(side="left")
        self.entry.bind("<Return>", self._entry_commit)
        self.entry.bind("<FocusOut>", self._entry_commit)
        self.entry.bind("<KP_Enter>", self._entry_commit)
        self._slider = None
        if self.spec.slider and self.spec.min is not None \
                and self.spec.max is not None:
            self._log = (self.spec.min > 0
                         and self.spec.max / max(self.spec.min, 1e-12) > 200)
            self._slider = ttk.Scale(self.frame, from_=0.0, to=1.0,
                                     command=self._slider_moved)
            self._slider.pack(side="left", fill="x", expand=True, padx=(8, 2))
            self._slider.bind("<ButtonRelease-1>", self._slider_release)
            self._slider.bind("<KeyRelease>", self._slider_release)
        self._error = None

    def _build_str(self):
        self.var = tk.StringVar()
        self.entry = tk.Entry(self.frame, textvariable=self.var, width=24,
                              font=("Consolas", 9), relief="solid", bd=1)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self._entry_commit)
        self.entry.bind("<FocusOut>", self._entry_commit)

    def _build_bool(self):
        self.var = tk.BooleanVar()
        cb = ttk.Checkbutton(self.frame, text="", variable=self.var,
                             command=self._bool_commit)
        cb.pack(side="left")
        self.entry = None

    def _build_choice(self):
        self._labels = [lbl for lbl, _v in self.spec.choices]
        self._by_label = dict(self.spec.choices)
        self.var = tk.StringVar()
        cb = ttk.Combobox(self.frame, textvariable=self.var, values=self._labels,
                          state="readonly", width=max(24, max(map(len, self._labels))))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._bool_commit)
        self.entry = cb

    def _build_array(self):
        self.frame_inner = ttk.Frame(self.frame)
        self.frame_inner.pack(side="left", fill="x", expand=True)
        self._entries = []
        add = self.spec.array_fixed is False
        bar = ttk.Frame(self.frame)
        bar.pack(side="left", padx=(6, 0))
        if add:
            ttk.Button(bar, text="+", width=2, command=self._array_grow,
                       takefocus=0).pack(side="left", padx=1)
            ttk.Button(bar, text="\u2212", width=2, command=self._array_shrink,
                       takefocus=0).pack(side="left", padx=1)
        self.entry = None

    def _build_nullable(self):
        self.var = tk.StringVar()
        self.entry = tk.Entry(self.frame, textvariable=self.var, width=ENTRY_W,
                              font=("Consolas", 9), justify="right",
                              relief="solid", bd=1)
        self.entry.pack(side="left")
        self.entry.bind("<Return>", self._entry_commit)
        self.entry.bind("<FocusOut>", self._entry_commit)
        self.auto_var = tk.BooleanVar()
        cb = ttk.Checkbutton(self.frame, text="auto", variable=self.auto_var,
                             command=self._auto_toggle)
        cb.pack(side="left", padx=(8, 0))
        Tooltip(cb, "When ticked the value is stored as null and decided "
                    "automatically (TE metal angle).")

    # ------------------------------------------------------ value plumbing
    def get_value(self):
        """(ok, value | error message)."""
        spec = self.spec
        try:
            if spec.kind == "bool":
                return True, bool(self.var.get())
            if spec.kind == "choice":
                return True, self._by_label[self.var.get()]
            if spec.kind == "float_array":
                vals = [parse_float(e.get()) for e in self._entries]
                return True, vals
            if spec.kind in ("nullable_float", "nullable_int"):
                if self.auto_var.get():
                    return True, None
                if spec.kind == "nullable_int":
                    return True, parse_int(self.var.get())
                return True, parse_float(self.var.get())
            text = self.var.get()
            if spec.kind == "str":
                return True, text
            if spec.kind == "int":
                v = parse_int(text)
            else:
                v = parse_float(text)
            if spec.min is not None and v < spec.min:
                return False, f"{spec.label}: {v} is below the minimum {spec.min}"
            if spec.max is not None and v > spec.max:
                return False, f"{spec.label}: {v} is above the maximum {spec.max}"
            return True, v
        except (ValueError, KeyError) as e:
            return False, str(e)

    def set_value(self, value):
        """Push a programmatic value into the editor (no commit)."""
        spec = self.spec
        self._guard = True
        try:
            self._last_committed = value
            if spec.kind == "bool":
                self.var.set(bool(value))
            elif spec.kind == "choice":
                for lbl, v in spec.choices:
                    if v == value:
                        self.var.set(lbl)
                        break
            elif spec.kind == "float_array":
                vals = list(value or [])
                self._ensure_entries(len(vals))
                for e, v in zip(self._entries, vals):
                    e.delete(0, "end")
                    e.insert(0, self._fmt(v))
                for e in self._entries:
                    e.configure(background="white")
            elif spec.kind in ("nullable_float", "nullable_int"):
                auto = value is None
                self.auto_var.set(auto)
                self._auto_toggle()
                if not auto:
                    self.var.set(self._fmt(value))
            elif value is not None:
                self.var.set(self._fmt(value))
                if getattr(self, "_slider", None) is not None:
                    self._update_slider_from_value(float(value))
        finally:
            self._guard = False

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            s = f"{v:.6g}"
            # strip float noise ("2.00000" -> "2") without eating integer
            # trailing zeros ("60000" must stay "60000")
            if "." in s or "e" in s or "E" in s:
                s = s.rstrip("0").rstrip(".")
            return s or "0"
        return str(v)

    # -------------------------------------------------------- entry events
    def _entry_commit(self, _e=None):
        ok, val = self.get_value()
        self._mark(ok)
        if ok and val != self._last_committed:
            self._last_committed = val
            self.on_commit(self.spec, val)
        return None

    def _bool_commit(self, _e=None):
        ok, val = self.get_value()
        if ok:
            self._last_committed = val
            self.on_commit(self.spec, val)

    def _auto_toggle(self):
        if self.auto_var.get():
            self.entry.configure(state="disabled")
        else:
            self.entry.configure(state="normal")
        self._bool_commit()

    def _mark(self, ok):
        if self.entry is not None:
            self.entry.configure(background=INVALID_BG if not ok else "white")

    def validate(self):
        ok, val = self.get_value()
        self._mark(ok)
        return ok, val

    # ------------------------------------------------------- slider events
    def _slider_moved(self, pos):
        if self._slider_guard or self.spec.min is None:
            return
        v = self._pos_to_value(float(pos))
        self.var.set(self._fmt(v))
        self._mark(True)

    def _slider_release(self, _e=None):
        self._slider_guard = True
        try:
            ok, val = self.get_value()
            if ok:
                self._update_slider_from_value(float(val))
                if val != self._last_committed:
                    self._last_committed = val
                    self.on_commit(self.spec, val)
        finally:
            self._slider_guard = False

    def _update_slider_from_value(self, v):
        if self._slider is None or self.spec.min is None:
            return
        v = min(max(v, self.spec.min), self.spec.max)
        if self._log:
            t = (math.log(v) - math.log(self.spec.min)) / \
                (math.log(self.spec.max) - math.log(self.spec.min))
        else:
            t = (v - self.spec.min) / (self.spec.max - self.spec.min)
        self._slider_guard = True
        self._slider.set(t)
        self._slider.update_idletasks()
        self._slider_guard = False

    def _pos_to_value(self, t):
        if self._log:
            v = math.exp(math.log(self.spec.min)
                         + t * (math.log(self.spec.max) - math.log(self.spec.min)))
        else:
            v = self.spec.min + t * (self.spec.max - self.spec.min)
        if self.spec.kind == "int":
            v = int(round(v))
        return v

    # --------------------------------------------------------- array events
    def _array_entry(self, parent, value=""):
        e = tk.Entry(parent, width=8, font=("Consolas", 9), justify="right",
                     relief="solid", bd=1)
        if value:
            e.insert(0, value)
        e.pack(side="left", padx=1)
        e.bind("<Return>", self._array_commit)
        e.bind("<FocusOut>", self._array_commit)
        self._entries.append(e)
        return e

    def _ensure_entries(self, n):
        """Create array entries up to n without firing a commit."""
        while len(self._entries) < n:
            self._array_entry(self.frame_inner)

    def _array_grow(self):
        self._ensure_entries(len(self._entries) + 1)
        self._array_commit()

    def _array_shrink(self):
        if len(self._entries) > 1:
            self._entries.pop().destroy()
            self._array_commit()

    def _array_commit(self, _e=None):
        ok, val = self.get_value()
        for e in self._entries:
            e.configure(background=INVALID_BG if not ok else "white")
        if ok and val != self._last_committed:
            self._last_committed = val
            self.on_commit(self.spec, val)

    def set_committed(self, value):
        self._last_committed = value
