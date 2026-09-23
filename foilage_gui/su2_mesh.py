"""Reader for SU2 ASCII mesh files (.su2): points, tris/quads, markers.

Used by the Mesh tab to display solver and mesh-project meshes. Element
VTK types: 5 = triangle, 9 = quad, 12 = hexahedron, 13 = prism; marker
elements are 2-node lines (type 3) in 2D output and quads/tris (9/5) in
3D wedge output.

3D wedge meshes (axisymmetric3d mode) are displayed as their mid-span
node layer: the spanwise rank of every node is derived geometrically
(nodes sharing (x, theta) form one stack, ranked hub -> shroud by
radius), the footprint faces of the cells whose bottom rank is the
middle one become the displayed quads/tris, and each side-wall marker is
traced by the edges of its faces that cut the mid-span layer.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np


def _span_ranks(hexes, prisms, n_points):
    """Spanwise rank (0 = hub) of every node plus the layer count, from the
    volume connectivity: a hex's first four nodes sit on the lower span
    face, the last four one layer above (prisms 3+3). Faces never appearing
    as an upper face are hub-layer faces; ranks follow by chaining."""
    from collections import defaultdict

    def key(nodes):
        return tuple(sorted(int(n) for n in nodes))

    lower_of = {}
    ups_of = defaultdict(list)
    for c in hexes:
        lo, up = key(c[:4]), key(c[4:])
        lower_of[up] = lo
        ups_of[lo].append(up)
    for c in prisms:
        lo, up = key(c[:3]), key(c[3:])
        lower_of[up] = lo
        ups_of[lo].append(up)

    rank = {k: 0 for k in ups_of if k not in lower_of}   # hub faces
    if not rank:
        raise ValueError("3D mesh has no spanwise stacking")
    stack = list(rank)
    while stack:
        k = stack.pop()
        for up in ups_of.get(k, ()):
            rank[up] = rank[k] + 1
            stack.append(up)
    n_layers = max(rank.values()) + 1

    node_rank = np.full(n_points, -1, dtype=np.int64)
    for k, r in rank.items():
        for n in k:
            node_rank[n] = r
    if (node_rank < 0).any():
        raise ValueError("nodes not covered by any span face")
    return node_rank, n_layers


def read_su2_mesh(path):
    """Parse an SU2 mesh into points/tris/quads/markers.

    2D:  {"ndim": 2, "points": (n,2), "tris", "quads",
          "markers": OrderedDict[name -> (edges,2)]}
    3D:  the same dict shape for the mid-span layer plus "ndim": 3,
         "n_layers", "hexes", "prisms" (cell counts) and
         "marker_face_counts". ``markers`` hold the mid-layer trace edges.
    """
    path = Path(path)
    with open(path) as f:
        lines = f.read().splitlines()

    ndim = 2
    points3d, hexes, prisms = None, [], []
    tris, quads, markers = [], [], OrderedDict()
    marker_faces = OrderedDict()
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].strip()
        if ln.startswith("NDIME="):
            ndim = int(float(ln.split("=")[1].split()[0]))
            i += 1
            continue
        if ln.startswith("NPOIN="):
            count = int(ln.split("=")[1].split()[0])
            points3d = np.zeros((count, 3))
            for j in range(count):
                parts = lines[i + 1 + j].split()
                for k in range(ndim):
                    points3d[j, k] = float(parts[k])
            i += count + 1
            continue
        if ln.startswith("NELEM="):
            count = int(ln.split("=")[1].split()[0])
            for j in range(count):
                parts = [int(x) for x in lines[i + 1 + j].split()]
                etype, nodes = parts[0], parts[1:]
                if etype == 5:
                    tris.append(nodes[:3])
                elif etype == 9:
                    quads.append(nodes[:4])
                elif etype == 12:
                    hexes.append(nodes[:8])
                elif etype == 13:
                    prisms.append(nodes[:6])
            i += count + 1
            continue
        if ln.startswith("MARKER_TAG="):
            name = ln.split("=", 1)[1].strip()
            count = int(lines[i + 1].split("=")[1].split()[0])
            faces, edges = [], []
            for j in range(count):
                parts = [int(x) for x in lines[i + 2 + j].split()]
                if ndim == 3:
                    if parts[0] == 9:
                        faces.append(parts[1:5])
                    elif parts[0] == 5:
                        faces.append(parts[1:4])
                    else:
                        faces.append(parts[1:])
                else:
                    edges.append((parts[-2], parts[-1]))
            if ndim == 3:
                marker_faces[name] = faces
            else:
                markers[name] = np.array(edges, dtype=np.int64)
            i += count + 2
            continue
        i += 1

    if points3d is None:
        points3d = np.zeros((0, 3))
    points = points3d[:, :2]

    out = {
        "ndim": ndim,
        "points": points,
        "tris": np.array(tris, dtype=np.int64) if tris else None,
        "quads": np.array(quads, dtype=np.int64) if quads else None,
        "markers": markers,
    }

    if ndim == 3:
        out["hexes"] = len(hexes)
        out["prisms"] = len(prisms)
        out["marker_face_counts"] = {k: len(v)
                                     for k, v in marker_faces.items()}
        if len(hexes) or len(prisms):
            rank, n_layers = _span_ranks(hexes, prisms, len(points3d))
            mid = (n_layers - 1) // 2
            keep = rank == mid
            idx = np.where(keep)[0]
            remap = np.full(len(points3d), -1, dtype=np.int64)
            remap[idx] = np.arange(len(idx))
            out["n_layers"] = n_layers
            out["points"] = points3d[idx][:, :2]

            if hexes:
                hx = np.asarray(hexes, dtype=np.int64)
                fq = hx[rank[hx[:, 0]] == mid]
                out["quads"] = remap[fq[:, :4]]
            else:
                fq = None
            if prisms:
                pr = np.asarray(prisms, dtype=np.int64)
                ft = pr[rank[pr[:, 0]] == mid]
                out["tris"] = remap[ft[:, :3]]
            else:
                ft = None

            # trace each side-wall marker where it cuts the mid-span layer:
            # exactly two ring-adjacent nodes of a face lie at rank mid and
            # their edge is part of the mid-layer boundary
            trace = {}
            for name, faces in marker_faces.items():
                keys = set()
                for face in faces:
                    if len(face) != 4:
                        continue                  # hub/shroud footprints
                    fr = rank[np.asarray(face, dtype=np.int64)]
                    at = np.where(fr == mid)[0]
                    if len(at) != 2 or (at[1] - at[0]) % 4 not in (1, 3):
                        continue
                    na, nb = face[at[0]], face[at[1]]
                    keys.add(_edge_key(points3d[na], points3d[nb]))
                trace[name] = keys

            # mid-layer boundary edges (used once), classified by trace
            # key; edge bookkeeping stays in ORIGINAL node ids
            counts = {}
            for cells, nc in ((ft, 3), (fq, 4)):
                if cells is None or not len(cells):
                    continue
                for k in range(nc):
                    e = np.sort(cells[:, [k, (k + 1) % nc]], axis=1)
                    for a, b in e:
                        counts[(int(a), int(b))] = \
                            counts.get((int(a), int(b)), 0) + 1
            boundary = [e for e, c in counts.items() if c == 1]
            by_name = {name: [] for name in marker_faces}
            for a, b in boundary:
                key = _edge_key(points3d[a], points3d[b])
                name = next((nm for nm, ks in trace.items() if key in ks),
                            None)
                if name is not None:
                    by_name[name].append((int(remap[a]), int(remap[b])))
            for name, edge_list in by_name.items():
                markers[name] = np.asarray(edge_list, dtype=np.int64).reshape(
                    -1, 2)
    return out


def _edge_key(pa, pb):
    a = (round(float(pa[0]), 7), round(float(pa[1]), 7))
    b = (round(float(pb[0]), 7), round(float(pb[1]), 7))
    return (a, b) if a <= b else (b, a)


def mesh_edges(tris, quads):
    """Unique edges of the cell set as an (e,2) int array (for drawing)."""
    parts = []
    for cells, n in ((tris, 3), (quads, 4)):
        if cells is None or len(cells) == 0:
            continue
        for k in range(n):
            parts.append(np.sort(cells[:, [k, (k + 1) % n]], axis=1))
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    all_edges = np.vstack(parts)
    return np.unique(all_edges, axis=0)
