"""Reader for SU2 ASCII mesh files (.su2): points, tris/quads, marker edges.

Used by the Mesh tab to display solver and mesh-project meshes. Element
VTK types: 5 = triangle, 9 = quad; marker elements are 3-node lines with
type 3 in SU2's 2D output.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np


def read_su2_mesh(path):
    """Parse an SU2 mesh into points/tris/quads/markers.

    Returns {"points": (n,2) float64, "tris": (m,3)|None, "quads": (k,4)|None,
    "markers": OrderedDict[name -> (edges,2) int]}.
    """
    path = Path(path)
    with open(path) as f:
        lines = f.read().splitlines()

    points, tris, quads, markers = None, [], [], OrderedDict()
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].strip()
        if ln.startswith("NPOIN="):
            count = int(ln.split("=")[1].split()[0])
            points = np.zeros((count, 2))
            for j in range(count):
                parts = lines[i + 1 + j].split()
                points[j, 0] = float(parts[0])
                points[j, 1] = float(parts[1])
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
            i += count + 1
            continue
        if ln.startswith("MARKER_TAG="):
            name = ln.split("=", 1)[1].strip()
            count = int(lines[i + 1].split("=")[1].split()[0])
            edges = []
            for j in range(count):
                parts = [int(x) for x in lines[i + 2 + j].split()]
                edges.append((parts[-2], parts[-1]))
            markers[name] = np.array(edges, dtype=np.int64)
            i += count + 2
            continue
        i += 1

    return {
        "points": points if points is not None else np.zeros((0, 2)),
        "tris": np.array(tris, dtype=np.int64) if tris else None,
        "quads": np.array(quads, dtype=np.int64) if quads else None,
        "markers": markers,
    }


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
