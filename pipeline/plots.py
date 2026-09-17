"""Mesh visualization: airfoil geometry, intermediate triangles, final quads."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _save(fig, path):
    path = Path(path)
    try:
        fig.savefig(path, dpi=170)
    except OSError:
        # file locked by an open viewer: write an alternate copy instead
        alt = path.with_name(path.stem + "_new" + path.suffix)
        fig.savefig(alt, dpi=170)
        print(f"note: {path.name} locked, wrote {alt.name} instead")
    finally:
        plt.close(fig)


def _load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, _ = line.split()
                verts.append((float(x), float(y)))
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.asarray(verts), faces


def _draw_cells(verts, faces, path, title, zoom=None, face_fill=None):
    fig, ax = plt.subplots(figsize=(13, 8))
    segs = []
    for f in faces:
        n = len(f)
        segs.extend(
            [(verts[f[i]], verts[f[(i + 1) % n]]) for i in range(n)]
        )
    lc = LineCollection(segs, linewidths=0.25, colors="#1f4e79")
    if face_fill:
        from matplotlib.collections import PolyCollection

        pc = PolyCollection(
            [verts[f] for f in faces], facecolors=face_fill, edgecolors="none",
            zorder=0,
        )
        ax.add_collection(pc)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x / axial chord")
    ax.set_ylabel("y / axial chord")
    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)


def plot_airfoil(airfoil, path):
    ss, ps = airfoil["ss"], airfoil["ps"]
    style = airfoil.get("style", "cup")
    ss_upper = ss[:, 1].mean() > ps[:, 1].mean()
    # after winding normalization "ss" is always the aerodynamic suction
    # side: upper surface for CAP (blade), lower surface for CUP (vane);
    # "ps" is the pressure side on the opposite surface.
    ss_lbl = f"suction side ({'upper' if ss_upper else 'lower'})"
    ps_lbl = f"pressure side ({'lower' if ss_upper else 'upper'})"
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ss[:, 0], ss[:, 1], "-", lw=2, label=ss_lbl)
    ax.plot(ps[:, 0], ps[:, 1], "-", lw=2, label=ps_lbl)
    ax.plot([ss[0, 0], ss[-1, 0]], [ss[0, 1], ss[-1, 1]], "k.", ms=8)
    ax.annotate("LE", ss[0], textcoords="offset points", xytext=(8, 8))
    ax.annotate("TE", ss[-1], textcoords="offset points", xytext=(8, 8))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x / axial chord")
    ax.set_ylabel("y / axial chord")
    ax.set_title(f"pyturbo-aero airfoil (left-to-right, {style} style, "
                 f"axial chord = 1)")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_tri_mesh(tri_obj, path, zoom=None):
    verts, tris = _load_obj(tri_obj)
    _draw_cells(verts, tris, path,
                f"Intermediate triangle mesh (gmsh) - {len(tris)} triangles",
                zoom)


def plot_quad_mesh(quad_obj, path, zoom=None, engine=""):
    verts, quads = _load_obj(quad_obj)
    n_tri = sum(1 for f in quads if len(f) == 3)
    n_quad = len(quads) - n_tri
    label = f" ({engine})" if engine else ""
    counts = f"{n_quad} quads" + (f" + {n_tri} tris" if n_tri else "")
    _draw_cells(verts, quads, path,
                f"Final quad mesh{label} - {counts}",
                zoom, face_fill="#dbe9f6")
