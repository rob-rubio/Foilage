"""Foilage GUI - tabbed front end for the 2D CFD pipeline.

Tabs:
    Setup     input.json solver/BC/case editing with smart widgets + run controls
    Geometry  interactive airfoil/passage editing (sliders + precise entries)
    Mesh      interactive mesh viewer (zoom/pan) and mesh generation
    Solution  auto-populated results summary, field viewer, .vtk import

Run with:  python run_gui.py [input.json]   (or python -m foilage_gui)
"""

__version__ = "1.0"
