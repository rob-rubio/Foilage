"""2D Free-Form Deformation (FFD) cage for imported geomTurbo sections.

The blade section (normalized to axial chord = 1) is enclosed in a
rectangular control cage: an n x n lattice of control points spread over
the section's bounding box (inflated by a small margin). Every control
point (i, j) carries user offsets (dx, dy) stored in input.json under
airfoil_source.morph. A surface point at normalized cage coordinates
(u, v) is displaced by the tensor-product Bernstein blend

    d(u, v) = sum_i sum_j B_i(u) * B_j(v) * offset(i, j),

which is the standard FFD lattice used for airfoil shape deformation:
equal offsets on all points translate the blade, opposing corner offsets
rotate/shear it, symmetric offsets stretch it, and single points create
local bumps. The deformation is a pure function of position, so the
shared LE/TE points of the suction and pressure sides always receive the
same displacement - the section stays closed.

The cage box is derived deterministically from the *unmorphed* section
(never from the morphed one), so offsets keep the same meaning no matter
how often the shape is rebuilt.
"""

from math import comb

import numpy as np

# cage box inflation, as a fraction of the section's width / height
MARGIN_FRACTION = 0.08

MIN_N = 2
MAX_N = 8


def bernstein_basis(n, t):
    """(len(t), n) matrix of Bernstein polynomials B_i^(n-1)(t).

    Row k holds the n basis values at t[k]; each row sums to 1."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    coef = np.array([comb(n - 1, i) for i in range(n)], dtype=float)
    p = np.arange(n)
    return (t[:, None] ** p[None, :]) * ((1.0 - t)[:, None] **
                                         (n - 1 - p)[None, :]) * coef[None, :]


def cage_box(ss, ps, margin=MARGIN_FRACTION):
    """Deterministic cage box [x0, x1, y0, y1] around the section."""
    pts = np.vstack([np.asarray(ss, float)[:, :2],
                     np.asarray(ps, float)[:, :2]])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [float(x0 - margin * (x1 - x0)), float(x1 + margin * (x1 - x0)),
            float(y0 - margin * (y1 - y0)), float(y1 + margin * (y1 - y0))]


def _offset_matrices(morph):
    """(n, DX, DY) from the input.json morph dict; raises ValueError on
    inconsistent input. DX/DY are indexed [i, j] = [x index, y index]."""
    n = int(morph.get("n") or 0)
    if n < MIN_N:
        raise ValueError(f"morph n must be >= {MIN_N} (got {n})")
    if n > MAX_N:
        raise ValueError(f"morph n must be <= {MAX_N} (got {n})")
    dx = np.asarray(morph.get("dx") or [], dtype=float)
    dy = np.asarray(morph.get("dy") or [], dtype=float)
    if dx.size != n * n or dy.size != n * n:
        raise ValueError(
            f"morph dx/dy must each hold n*n = {n * n} values "
            f"(got {dx.size} / {dy.size})")
    return n, dx.reshape(n, n), dy.reshape(n, n)


def cage_control_points(box, n):
    """Undeformed cage control point coordinates, (n*n, 2) ordered
    [i, j] -> row-major with i the x index and j the y index."""
    x0, x1, y0, y1 = box
    us = np.linspace(0.0, 1.0, n)
    vs = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(us, vs, indexing="ij")     # (n, n)
    return np.column_stack([(x0 + uu * (x1 - x0)).ravel(),
                            (y0 + vv * (y1 - y0)).ravel()])


def displaced_control_points(box, n, dx, dy):
    """Cage control points after applying their own offsets, (n*n, 2)."""
    cp = cage_control_points(box, n)
    cp[:, 0] += np.asarray(dx, float).ravel()
    cp[:, 1] += np.asarray(dy, float).ravel()
    return cp


def _displacement(xy, box, n, offsets):
    """Bernstein-blended displacement of xy (m, 2) for one offset matrix
    indexed [i, j]."""
    x0, x1, y0, y1 = box
    u = (xy[:, 0] - x0) / max(x1 - x0, 1e-12)
    v = (xy[:, 1] - y0) / max(y1 - y0, 1e-12)
    bu = bernstein_basis(n, u)
    bv = bernstein_basis(n, v)
    # points outside the cage clamp onto its edge (FFD constant extension)
    return np.einsum("ki,ij,kj->k", bu, offsets, bv)


def is_active(morph):
    """True when the morph dict enables the cage deformation."""
    return bool(isinstance(morph, dict) and morph.get("enabled"))


def apply_morph(ss, ps, morph, margin=MARGIN_FRACTION):
    """Displace the section with the FFD cage.

    ss/ps: (m, 2) surface arrays (any shared points stay shared because
    the displacement depends only on position). Returns
    (ss, ps, cage_info) where cage_info carries everything needed to draw
    the cage (None when the morph is disabled)."""
    if not is_active(morph):
        return ss, ps, None
    n, dx, dy = _offset_matrices(morph)
    ss = np.array(ss, dtype=float, copy=True)
    ps = np.array(ps, dtype=float, copy=True)
    box = cage_box(ss, ps, margin)
    ss += np.column_stack([_displacement(ss, box, n, dx),
                           _displacement(ss, box, n, dy)])
    ps += np.column_stack([_displacement(ps, box, n, dx),
                           _displacement(ps, box, n, dy)])
    cage = {"box": box, "n": n,
            "dx": dx.ravel().tolist(), "dy": dy.ravel().tolist(),
            "cp0": cage_control_points(box, n),
            "cp": displaced_control_points(box, n, dx, dy),
            "margin": margin}
    return ss, ps, cage


def rebuild_outline(ss, ps, ss_upper):
    """Closed clockwise outline from the surface arrays (the convention
    of build_airfoil / section_to_airfoil)."""
    upper, lower = (ss, ps) if ss_upper else (ps, ss)
    return np.vstack([upper, lower[::-1][1:-1]])
