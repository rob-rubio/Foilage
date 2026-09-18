"""CLI shared by run_gui.py and python -m foilage_gui.

    run_gui.py [input.json] [--threads N] [--selftest] [--smoke]
"""

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pick_default_input():
    local = REPO / "input.json"
    if local.exists():
        return local
    cases = sorted((REPO / "cases").glob("*/input.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cases:
        return cases[0]
    return local


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Foilage GUI - setup, geometry, mesh, and solution")
    ap.add_argument("input", nargs="?", type=Path, default=None,
                    help="pipeline input.json (default: newest in cases/)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--selftest", action="store_true",
                    help="run logic tests and exit (no window)")
    ap.add_argument("--smoke", action="store_true",
                    help="build the window, pump the tabs once, exit")
    args = ap.parse_args(argv)

    if args.selftest:
        from foilage_gui.selftest import main as st_main
        st_main()
        return

    inp = args.input or pick_default_input()
    if not inp.is_absolute():
        inp = (Path.cwd() / inp)
    if not inp.exists():
        sys.exit(f"input not found: {inp} (pass a path to an input.json)")

    from foilage_gui.app import FoilageApp
    root = tk.Tk()
    app = FoilageApp(root, inp.resolve(), threads=args.threads)
    if args.smoke:
        smoke(app)
        return
    app.run()


def _pump(root, n=10, dt=0.05):
    """Pump the Tk event loop long enough for after() callbacks to fire."""
    for _ in range(n):
        root.update()
        root.after_idle(root.update_idletasks)
        root.update()
        time.sleep(dt)


def smoke(app):
    """Build every tab, select each one, pump the event loop, exit."""
    root = app.root
    errors = []

    def _catch(exc, val, tb):                    # noqa: ARG001
        errors.append(val)
        import traceback
        traceback.print_exception(exc, val, tb)

    root.report_callback_exception = _catch
    _pump(root, 4)
    nb = app.notebook
    for i in range(len(nb.tabs())):
        nb.select(i)
        _pump(root, 6)
        print(f"smoke: tab {nb.tab(i, 'text').strip()} OK")
    # exercise a geometry edit end-to-end (slider path)
    geom = app._tabs[0]               # Geometry is the first tab
    spec_path = "airfoil.alpha2"
    widget = geom.fields[spec_path]
    widget.var.set("-58.5")
    widget._entry_commit()
    _pump(root, 12)
    assert abs(app.state.get(spec_path) + 58.5) < 1e-9, "edit did not land"
    assert app.state.dirty, "edit should mark the case dirty"
    status = geom.status_var.get()
    assert "error" not in status.lower(), f"geometry rebuild failed: {status}"
    print(f"smoke: geometry edit OK ({spec_path} -> "
          f"{app.state.get(spec_path)}, {status[:60]}...)")
    # exercise the mesh viewer + solution field viewer data paths
    mesh_tab = app._tabs[2]
    assert mesh_tab._current is not None, "mesh tab loaded no mesh"
    assert mesh_tab.info_var.get().count(",") >= 1
    sol = app._tabs[3]
    sol.reload_all()
    _pump(root, 20)
    if any(sol.app.state.case_dir().glob("vol_solution*.vtk")):
        assert sol._volume is not None, "solution tab loaded no volume data"
        assert "Mach" in sol._fields
    else:
        print("smoke: default case has no vol_solution*.vtk - "
              "field-view assertions skipped")
    # regression: run-job validation must pass on a well-formed case
    # (empty dynamic-choice widgets used to raise KeyError -> an error
    # dialog listing an empty message)
    issues = [i for i in app.state.validate() if i[0] == "error"]
    for tab in app._tabs:
        if hasattr(tab, "fields"):
            for widget in tab.fields.values():
                ok, err = widget.validate()
                if not ok and err:
                    issues.append(("error", err))
    assert not issues, f"run validation failed on a valid case: {issues}"
    print("smoke: run validation OK")
    _pump(root, 4)
    assert not errors, f"Tk callback errors: {errors[0]!r}"
    root.destroy()
    print("smoke OK")


if __name__ == "__main__":
    main()
