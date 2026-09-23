"""Extrude the 2D quadified passage mesh into a 3D conical wedge.

``axisymmetric3d`` mode: the quadified 2D blade-to-blade mesh (unwrapped
coordinates x, uy) is wrapped to polar coordinates around the machine axis,
``theta = uy / R(x)`` with R the linear annulus radius profile, and stacked
spanwise between the streamtube walls

    r(x, zeta) = R(x) + (zeta - 0.5) * h(x)

where h(x) is the logistic streamtube depth from
``pipeline.mesh_tris.pitch_profile`` (clamped to h1 fore of the LE and h2
aft of the TE). R(x) continues linearly upstream of the LE and downstream
of the TE. The node angle theta is held constant across span, so the
periodic faces pair by an exact rotation of 2 pi / N about the machine
axis - which is why a varying radius R1 -> R2 is
legal for the solver in this mode. The hub (zeta = 0) and shroud
(zeta = 1) surfaces use free-slip walls; every 2D marker edge sweeps into a
wall of quad faces.

Quads extrude to hexahedra (VTK/SU2 type 12), leftover triangles to prisms
(type 13). The spanwise distribution supports optional two-sided geometric
clustering toward both walls.
"""

import numpy as np


def span_fractions(n_layers, growth=1.3):
    """Spanwise fractions zeta in [0, 1] (n_layers values, n_layers - 1
    intervals) with two-sided geometric clustering toward both walls.

    ``growth`` = 1 gives uniform spacing. With growth > 1 the interval
    widths grow by ``growth`` from each wall toward mid-span (a single
    geometric sequence folded at the center), so the first intervals at
    both walls are equally small - the spanwise analogue of a two-sided
    boundary layer mesh."""
    n_layers = int(n_layers)
    if n_layers < 2:
        return np.array([0.0])
    g = float(growth)
    n_int = n_layers - 1
    if g <= 1.0 + 1e-12:
        return np.linspace(0.0, 1.0, n_layers)
    a = n_int // 2                       # intervals growing from the hub
    widths = list(g ** np.arange(a))
    if n_int % 2:                        # odd: one folded center interval
        widths.append(g ** a)
    widths += widths[-1::-1]             # mirror toward the shroud
    widths = np.asarray(widths, dtype=float)
    return np.concatenate([[0.0], np.cumsum(widths) / widths.sum()])


def read_su2_2d(path):
    """Read a 2D SU2 mesh as written by quadify.

    Returns verts (n, 2), quads (m, 4), tris (k, 3) and a
    ``{marker: edges (e, 2)}`` dict, all 0-based."""
    verts = []
    elems = []                       # (su2_type, [original node indices])
    marker_edges = {}                # marker -> [(a, b), ...] original idx
    section = None
    tag = None
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("NDIME="):
                if int(float(s.split("=", 1)[1])) != 2:
                    raise ValueError(f"{path}: expected a 2D (NDIME= 2) "
                                     "mesh")
            elif s.startswith("NELEM="):
                section = "elem"
            elif s.startswith("NPOIN="):
                section = "poin"
            elif s.startswith("NMARK="):
                section = "mark"
            elif s.startswith("MARKER_TAG="):
                tag = s.split("=", 1)[1].strip()
                marker_edges.setdefault(tag, [])
            elif s.startswith("MARKER_ELEMS="):
                pass
            elif section == "poin":
                p = s.split()
                verts.append((float(p[0]), float(p[1]), int(p[2])))
            elif section == "elem":
                p = s.split()
                elems.append((int(p[0]), [int(v) for v in p[1:]]))
            elif section == "mark":
                p = s.split()
                if int(p[0]) != 3:
                    raise ValueError(f"{path}: unexpected marker element "
                                     f"type {p[0]} (expected a 2-node "
                                     "line)")
                marker_edges[tag].append((int(p[1]), int(p[2])))

    idx_map = {orig: i for i, (_x, _y, orig) in enumerate(verts)}
    v = np.asarray([(x, y) for x, y, _ in verts], dtype=float)
    unknown = sorted({_t for _t, _e in elems} - {5, 9})
    if unknown:
        raise ValueError(f"{path}: unsupported 2D element types {unknown}")
    quads = np.asarray([idx_map[i] for t, e in elems if t == 9 for i in e],
                       dtype=np.int64).reshape(-1, 4)
    tris = np.asarray([idx_map[i] for t, e in elems if t == 5 for i in e],
                      dtype=np.int64).reshape(-1, 3)
    markers = {m: np.asarray(e, dtype=np.int64).reshape(-1, 2)
               for m, e in marker_edges.items()}
    return v, quads, tris, markers


def _snap_periodic_edges(verts, markers, prof):
    """Make the periodic faces exactly congruent under the wedge rotation.

    The 2D mesh meshes the two periodic edges independently, so their
    nodes sit at slightly different axial stations and the wrapped faces
    mismatch by up to ~1e-3 chord - small enough for the 2D translation
    pairing, but rotational matching flags it (and non-matching halos
    break conservation). Pairs the bottom/top nodes by their order along
    the edge and moves both onto the shared averaged station with the
    exact medial-axis offset. Returns the number of paired stations
    (None when the counts differ)."""
    bot = np.unique(markers["periodic_bottom"])
    top = np.unique(markers["periodic_top"])
    if len(bot) != len(top) or len(bot) < 2:
        return None
    bot = bot[np.argsort(verts[bot, 0], kind="stable")]
    top = top[np.argsort(verts[top, 0], kind="stable")]
    y_c, p = prof["y_c"], prof["p"]
    for ib, it in zip(bot, top):
        x_bar = 0.5 * (verts[ib, 0] + verts[it, 0])
        uy_mid = float(np.asarray(y_c(x_bar)))
        half = 0.5 * float(np.asarray(p(x_bar)))
        verts[ib, 0] = verts[it, 0] = x_bar
        verts[ib, 1] = uy_mid - half
        verts[it, 1] = uy_mid + half
    return len(bot)


def extrude_wedge(mesh_quad_su2, prof, out_path, n_layers=21, growth=1.3):
    """Wrap + spanwise-extrude a quadified 2D mesh into a conical wedge.

    mesh_quad_su2 : path of the quadified 2D mesh (quadify output)
    prof          : pitch_profile dict (needs the ``radius``, ``h`` and
                    ``p`` callables)
    out_path      : destination .su2 file (NDIME= 3)
    n_layers      : spanwise node layers (>= 2)
    growth        : two-sided spanwise clustering ratio (1 = uniform)

    Returns a stats dict (hexes, prisms, points, n_layers, min cell
    volume, periodic_max_mismatch in normalized units, per-marker face
    counts)."""
    verts, quads, tris, markers = read_su2_2d(mesh_quad_su2)
    n2d = len(verts)
    L = int(n_layers)
    if L < 2:
        raise ValueError("n_layers must be >= 2")
    for required in ("periodic_bottom", "periodic_top"):
        if required not in markers or len(markers[required]) == 0:
            raise ValueError("axisymmetric3d needs a periodic cascade "
                             f"mesh; marker '{required}' missing")
    zeta = span_fractions(L, growth)

    verts = verts.copy()
    snap_info = _snap_periodic_edges(verts, markers, prof)

    x, uy = verts[:, 0], verts[:, 1]
    R = np.asarray(prof["radius"](x), dtype=float)
    hgt = np.asarray(prof["h"](x), dtype=float)
    theta = uy / R
    if (R - 0.5 * hgt).min() <= 0.0:
        raise ValueError("streamtube depth h(x) exceeds twice the annulus "
                         "radius somewhere: the hub surface crosses the "
                         "machine axis")

    # 3D points, layer-major: point id = layer * n2d + i2d
    pts = np.empty((L * n2d, 3))
    for l, z in enumerate(zeta):
        r = R + (z - 0.5) * hgt
        base = l * n2d
        pts[base:base + n2d, 0] = x
        pts[base:base + n2d, 1] = r * np.sin(theta)
        pts[base:base + n2d, 2] = r * np.cos(theta)

    nq, nt = len(quads), len(tris)
    hexes = np.empty(((L - 1) * nq, 8), dtype=np.int64)
    prisms = np.empty(((L - 1) * nt, 6), dtype=np.int64)
    for l in range(L - 1):
        off, up = l * n2d, (l + 1) * n2d
        if nq:
            blk = np.empty((nq, 8), dtype=np.int64)
            blk[:, :4] = quads + off
            blk[:, 4:] = quads + up
            hexes[l * nq:(l + 1) * nq] = blk
        if nt:
            blk = np.empty((nt, 6), dtype=np.int64)
            blk[:, :3] = tris + off
            blk[:, 3:] = tris + up
            prisms[l * nt:(l + 1) * nt] = blk

    _make_positive(hexes, prisms, pts)

    # straight-edged hexes/prisms can end up twisted when a 2D cell is too
    # large for the local wrap curvature (steep R(x)/h(x) transitions with
    # long far-field cells). SU2 may still run, but a solution that NaNs
    # early on an otherwise-valid setup is usually these - reported so the
    # mesh can be refined.
    n_bad = int(((_volumes(hexes, pts, 8) <= 0.0).sum()
                 + (_volumes(prisms, pts, 6) <= 0.0).sum()))

    # boundary faces: side walls sweep the 2D marker edges through span
    # (ALWAYS quads - the swept face of a tri-owned edge spans two layers
    # and a tri face would carry a diagonal pair that is not a primal edge,
    # which crashes SU2's surface curvature pass); hub/shroud are the
    # span-extreme footprint of every cell
    quad_owner, tri_owner = _edge_owners(quads, tris)
    marker_faces = {}
    for name, edges in markers.items():
        f4, o4 = [], []
        for a, b in edges:
            key = (min(int(a), int(b)), max(int(a), int(b)))
            for l in range(L - 1):
                off, up = l * n2d, (l + 1) * n2d
                f4.append((off + int(a), off + int(b),
                           up + int(b), up + int(a)))
                if key in quad_owner:
                    o4.append((0, l * nq + quad_owner[key]))
                else:
                    o4.append((1, l * nt + tri_owner[key]))
        marker_faces[name] = _orient(f4, o4, [], [], pts, hexes, prisms)

    marker_faces["hub"] = _footprint(quads, tris, hexes, prisms,
                                     0, n2d, L, nq, nt, pts)
    marker_faces["shroud"] = _footprint(quads, tris, hexes, prisms,
                                        L - 1, n2d, L, nq, nt, pts)

    mismatch = _periodic_mismatch(pts, verts, markers, prof, n2d, L)

    _write_su2_3d(out_path, pts, hexes, prisms, marker_faces)
    return {
        "hexes": int(len(hexes)),
        "prisms": int(len(prisms)),
        "points": int(len(pts)),
        "n_layers": L,
        "min_cell_volume": float(min(_min_volume(hexes, pts, 8),
                                     _min_volume(prisms, pts, 6))),
        "periodic_max_mismatch": mismatch,
        "periodic_pairs_snapped": snap_info,
        "inverted_cells": n_bad,
        "markers": {m: int(sum(len(f) for f in fs))
                    for m, fs in marker_faces.items()},
    }


def _edge_owners(quads, tris):
    """Map each sorted 2D edge to its owning element index (edges are
    manifold in the 2D domain: each borders exactly one element)."""
    quad_owner, tri_owner = {}, {}
    for ci, cell in enumerate(quads):
        for k in range(4):
            a, b = int(cell[k]), int(cell[(k + 1) % 4])
            quad_owner[(min(a, b), max(a, b))] = ci
    for ci, cell in enumerate(tris):
        for k in range(3):
            a, b = int(cell[k]), int(cell[(k + 1) % 3])
            tri_owner[(min(a, b), max(a, b))] = ci
    return quad_owner, tri_owner


def _footprint(quads, tris, hexes, prisms, layer, n2d, L, nq, nt, pts):
    """Boundary faces at span layer ``layer`` (0 = hub, L-1 = shroud):
    every 2D element contributes its footprint face, oriented outward."""
    off = layer * n2d
    cell_l = 0 if layer == 0 else L - 2
    f4 = [tuple(off + int(n) for n in cell) for cell in quads]
    o4 = [(0, cell_l * nq + ci) for ci in range(nq)]
    f3 = [tuple(off + int(n) for n in cell) for cell in tris]
    o3 = [(1, cell_l * nt + ci) for ci in range(nt)]
    return _orient(f4, o4, f3, o3, pts, hexes, prisms)


def _orient(f4, o4, f3, o3, pts, hexes, prisms):
    """Order each face's nodes so its normal points away from its owning
    cell centroid. Owners are ``(kind, index)`` pairs (kind 0 = hex, 1 =
    prism). Returns a list of arrays ((m, 4) and/or (m, 3))."""
    out = []
    for faces, owners in ((f4, o4), (f3, o3)):
        if not faces:
            continue
        faces = np.asarray(faces, dtype=np.int64)
        fc = pts[faces].mean(axis=1)
        cc = np.empty((len(faces), pts.shape[1]))
        kinds = np.array([k for k, _i in owners], dtype=np.int64)
        idxs = np.array([i for _k, i in owners], dtype=np.int64)
        hmask = kinds == 0
        if hmask.any():
            cc[hmask] = pts[hexes[idxs[hmask]]].mean(axis=1)
        pmask = ~hmask
        if pmask.any():
            cc[pmask] = pts[prisms[idxs[pmask]]].mean(axis=1)
        if faces.shape[1] == 4:
            a, b, c, d = (pts[faces[:, i]] for i in range(4))
            nA = 0.5 * np.cross(c - a, d - b)
        else:
            a, b, c = (pts[faces[:, i]] for i in range(3))
            nA = 0.5 * np.cross(b - a, c - a)
        flip = np.einsum("ij,ij->i", nA, fc - cc) < 0.0
        faces[flip] = faces[flip][:, ::-1]
        out.append(faces)
    return out


def _volumes(cells, pts, width):
    """Signed cell volumes via the divergence theorem (positive for the
    VTK/SU2 node ordering with CCW bottom face and +span extrusion)."""
    p = pts[cells]
    v = np.zeros(len(cells))
    if width == 8:
        quad_f = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                  (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
        for f in quad_f:
            fp = p[:, list(f)]
            nA = 0.5 * np.cross(fp[:, 2] - fp[:, 0], fp[:, 3] - fp[:, 1])
            v += np.einsum("ij,ij->i", nA, fp.mean(axis=1)) / 3.0
    else:
        for f in ((0, 2, 1), (3, 4, 5)):
            fp = p[:, list(f)]
            nA = 0.5 * np.cross(fp[:, 1] - fp[:, 0], fp[:, 2] - fp[:, 0])
            v += np.einsum("ij,ij->i", nA, fp.mean(axis=1)) / 3.0
        for f in ((0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)):
            fp = p[:, list(f)]
            nA = 0.5 * np.cross(fp[:, 2] - fp[:, 0], fp[:, 3] - fp[:, 1])
            v += np.einsum("ij,ij->i", nA, fp.mean(axis=1)) / 3.0
    return v


def _min_volume(cells, pts, width):
    if not len(cells):
        return np.inf
    return float(np.abs(_volumes(cells, pts, width)).min())


def _make_positive(hexes, prisms, pts):
    """Flip the whole stack if the extrusion direction came out inverted;
    raise if orientations are mixed (must not happen for a consistent 2D
    mesh)."""
    vol = 0.0
    for cells, width in ((hexes, 8), (prisms, 6)):
        if len(cells):
            vol += float(_volumes(cells, pts, width).sum())
    if abs(vol) < 1e-14:
        raise ValueError("degenerate wedge extrusion (zero volume)")
    if vol > 0.0:
        return
    flip8 = [0, 3, 2, 1, 4, 7, 6, 5]
    flip6 = [0, 2, 1, 3, 5, 4]
    hexes[:] = hexes[:, flip8]
    prisms[:] = prisms[:, flip6]
    for cells, width in ((hexes, 8), (prisms, 6)):
        if len(cells) and (_volumes(cells, pts, width) <= 0.0).any():
            raise ValueError("mixed cell orientation after extrusion")


def _periodic_mismatch(pts, verts, markers, prof, n2d, L):
    """Max position error (normalized units) between the rotated
    periodic_top nodes and the matching periodic_bottom nodes: the wedge
    angle dtheta(x) = p(x) / R(x) = 2 pi / N should map one face exactly
    onto the other in every span layer."""
    bot = np.unique(markers["periodic_bottom"])
    top = np.unique(markers["periodic_top"])
    top = top[np.argsort(verts[top, 0])]
    worst = 0.0
    for i_b in bot:
        xi = verts[i_b, 0]
        j = int(top[np.argmin(np.abs(verts[top, 0] - xi))])
        if abs(verts[j, 0] - xi) > 1e-9:
            continue
        dth = float(np.asarray(prof["p"](xi))
                    / np.asarray(prof["radius"](xi)))
        c, s = np.cos(dth), np.sin(dth)
        for l in range(L):
            pb = pts[l * n2d + int(i_b)]
            pt = pts[l * n2d + j]
            yr = c * pt[1] - s * pt[2]
            zr = s * pt[1] + c * pt[2]
            worst = max(worst, abs(pt[0] - pb[0]),
                        abs(yr - pb[1]), abs(zr - pb[2]))
    return worst


def _write_su2_3d(path, pts, hexes, prisms, marker_faces):
    with open(path, "w") as f:
        f.write("NDIME= 3\n")
        f.write(f"NELEM= {len(hexes) + len(prisms)}\n")
        for c in hexes:
            f.write("12 %d %d %d %d %d %d %d %d\n" % tuple(c))
        for c in prisms:
            f.write("13 %d %d %d %d %d %d\n" % tuple(c))
        f.write(f"NPOIN= {len(pts)}\n")
        for i, (x, y, z) in enumerate(pts):
            f.write("%.17g %.17g %.17g %d\n" % (x, y, z, i))
        f.write(f"NMARK= {len(marker_faces)}\n")
        for name, groups in marker_faces.items():
            n = sum(len(g) for g in groups)
            f.write(f"MARKER_TAG= {name}\n")
            f.write(f"MARKER_ELEMS= {n}\n")
            for group in groups:
                if group.ndim == 2 and group.shape[1] == 4:
                    for c in group:
                        f.write("9 %d %d %d %d\n" % tuple(c))
                else:
                    for c in group:
                        f.write("5 %d %d %d\n" % tuple(c))
