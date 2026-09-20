"""Quad-mesh utilities for Gmsh Blossom output.

The active pipeline extracts Gmsh's recombined elements, restores the physical
markers, checks element orientation and quality, and writes the solver meshes.

Marker curves are supplied as sampled polylines (the periodic edges are the
passage medial axes, not straight lines). Marker orientation follows SU2's
2D convention (fluid on the left of the edge direction): airfoil loop
clockwise (fluid outside), outer border counter-clockwise (fluid inside).

Outputs: <prefix>_quad.su2 (QUAD type 9 / LINE type 3) and <prefix>_quad.msh
(gmsh 2.2 ASCII with physical groups).
"""

from pathlib import Path

import numpy as np

PERIODIC_MARKERS = ("airfoil", "inlet", "outlet", "periodic_bottom",
                    "periodic_top")
FREESTREAM_MARKERS = ("airfoil", "inlet", "outlet", "farfield")

# Keep the historical name for callers that need the periodic layout.  The
# active layout is selected from the physical groups in the Gmsh mesh below.
MARKERS = PERIODIC_MARKERS


def _marker_layout(physical_names):
    """Return the solver marker layout represented by Gmsh physical groups."""
    if "farfield" in set(physical_names):
        return FREESTREAM_MARKERS
    return PERIODIC_MARKERS


def extract_gmsh_recombined(prefix):
    """Extract a Blossom-recombined gmsh mesh (structured BL quads + interior
    quads + leftover tris) into mesh_quad.su2 / mesh_quad.msh / mesh_quad.obj.

    Markers come straight from the gmsh physical groups; line elements are
    re-oriented to SU2's 2D convention (fluid on the left): airfoil loop
    clockwise, inlet/outlet and outer boundaries follow the counter-clockwise
    border (inlet down, outlet up, lower +x, upper -x).
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(prefix) + ".msh")

    physical_names = [
        gmsh.model.getPhysicalName(dim, tag)
        for dim, tag in gmsh.model.getPhysicalGroups(1)
    ]
    marker_names = _marker_layout(physical_names)
    marker_index = {name: i for i, name in enumerate(marker_names)}

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    verts = coords[:, :2]
    order = {int(t): i for i, t in enumerate(node_tags)}

    quads, tris = [], []
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    for t, conn in zip(etypes, enodes):
        nn = 4 if int(t) == 3 else 3
        cells = [[order[int(tag)] for tag in row]
                 for row in np.array(conn).reshape(-1, nn)]
        (quads if nn == 4 else tris).extend(cells)

    marker_lines = []
    wanted = set(marker_names)
    for dim, ptag in gmsh.model.getPhysicalGroups(1):
        if dim != 1:
            continue
        name = gmsh.model.getPhysicalName(dim, ptag)
        if name not in wanted:
            continue
        mi = marker_index[name]
        edges = []
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
            et1, _, en1 = gmsh.model.mesh.getElements(1, ent)
            for t, conn in zip(et1, en1):
                edges += [(order[int(conn[k])], order[int(conn[k + 1])])
                          for k in range(0, len(conn), 2)]
        for a, b in _orient_marker(verts, name, edges):
            marker_lines.append((mi, a, b))
    gmsh.finalize()

    # compact vertex set, orient cells CCW
    used = np.unique(np.concatenate(
        [np.array(quads, dtype=np.int64).ravel(),
         np.array(tris, dtype=np.int64).ravel(),
         np.array([(a, b) for _, a, b in marker_lines],
                  dtype=np.int64).ravel()]))
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts = verts[used]
    quads = remap[np.array(quads, dtype=np.int64)]
    tris = remap[np.array(tris, dtype=np.int64)]
    marker_lines = [(k, int(remap[a]), int(remap[b]))
                    for k, a, b in marker_lines]

    q = quads.copy()
    ang, area, _nc, _me, _ma = _quad_metrics(verts, q)
    flip = area < 0
    if flip.any():
        q[flip] = q[flip][:, [0, 3, 2, 1]]
    if len(tris):
        v3 = verts[tris]
        a3 = 0.5 * ((v3[:, 0, 0] * v3[:, 1, 1] + v3[:, 1, 0] * v3[:, 2, 1]
                     + v3[:, 2, 0] * v3[:, 0, 1])
                    - (v3[:, 1, 0] * v3[:, 0, 1] + v3[:, 2, 0] * v3[:, 1, 1]
                       + v3[:, 0, 0] * v3[:, 2, 1]))
        tris[a3 < 0] = tris[a3 < 0][:, [0, 2, 1]]

    ang, _a, _nc, _me, _ma = _quad_metrics(verts, q)

    stats = {
        "quads": int(len(q)),
        "leftover_tris": int(len(tris)),
        "min_corner_angle_deg": float(np.degrees(ang.min())),
    }
    marker_counts = {
        m: sum(1 for ml in marker_lines if ml[0] == i)
        for i, m in enumerate(marker_names)
    }
    stats.update(marker_counts)
    stats["markers"] = marker_counts

    _write_su2(verts, q, tris, marker_lines, str(prefix) + "_quad.su2",
               marker_names)
    _write_msh2(verts, q, tris, marker_lines, str(prefix) + "_quad.msh",
                marker_names)
    _write_obj(verts, q, tris, str(prefix) + "_quad.obj")
    return stats


def _orient_marker(verts, name, edges):
    """Chain marker edges into node paths, re-oriented to SU2's 2D convention
    (fluid on the left of the edge direction):
        airfoil (closed loop) -> clockwise
        inlet (left border)   -> top to bottom
        outlet (right border) -> bottom to top
        periodic_bottom       -> +x     periodic_top -> -x
        farfield            -> lower +x, upper -x
    (the outer boundary paths keep the fluid on their left)
    """
    from collections import defaultdict

    adj = defaultdict(list)
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i))
        adj[b].append((a, i))

    used = set()
    chains = []
    for i0, (a0, b0) in enumerate(edges):
        if i0 in used:
            continue
        used.add(i0)
        chain = [a0, b0]
        while True:  # extend forward
            cand = [(nb, j) for nb, j in adj[chain[-1]] if j not in used]
            if not cand:
                break
            nb, j = cand[0]
            used.add(j)
            chain.append(nb)
        while True:  # extend backward
            cand = [(nb, j) for nb, j in adj[chain[0]] if j not in used]
            if not cand:
                break
            nb, j = cand[0]
            used.add(j)
            chain.insert(0, nb)
        chains.append(chain)

    farfield_y = None
    if name == "farfield":
        farfield_y = [float(np.mean(verts[chain[:-1], 1]))
                      for chain in chains if len(chain) > 1]
        y_mid = 0.5 * (min(farfield_y) + max(farfield_y)) \
            if farfield_y else 0.0

    out = []
    for chain in chains:
        if len(chain) < 2:
            continue
        closed = chain[0] == chain[-1]
        if closed:
            if _signed_area(verts[chain[:-1]]) > 0:
                chain = chain[::-1]
        else:
            first, last = verts[chain[0]], verts[chain[-1]]
            if name == "inlet" and first[1] < last[1]:
                chain = chain[::-1]
            elif name == "outlet" and first[1] > last[1]:
                chain = chain[::-1]
            elif name == "periodic_bottom" and first[0] > last[0]:
                chain = chain[::-1]
            elif name == "periodic_top" and first[0] < last[0]:
                chain = chain[::-1]
            elif name == "farfield":
                # Gmsh stores both outer chains in one physical group. SU2's
                # positive-fluid-side convention requires lower +x and upper
                # -x, just like the two separate periodic markers.
                lower = float(np.mean(verts[chain[:-1], 1])) <= y_mid
                wrong_way = ((lower and first[0] > last[0]) or
                             (not lower and first[0] < last[0]))
                if wrong_way:
                    chain = chain[::-1]
        seq = (list(zip(chain[:-1], chain[1:])) if not closed
               else list(zip(chain[:-1], chain[1:])))
        out += seq
    return out


def _signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _quad_metrics(verts, quads):
    """Per-quad min/max corner angle (rad), signed area, convexity, min edge."""
    v = verts[quads][:, :, :2]
    min_ang = np.full(len(quads), np.pi)
    max_ang = np.zeros(len(quads))
    for i in range(4):
        a, b, c = v[:, (i - 1) % 4], v[:, i], v[:, (i + 1) % 4]
        u, w = a - b, c - b
        cs = np.clip(
            (u * w).sum(1)
            / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1)),
            -1, 1,
        )
        a_ = np.arccos(cs)
        min_ang = np.minimum(min_ang, a_)
        max_ang = np.maximum(max_ang, a_)
    x, y = v[:, :, 0], v[:, :, 1]
    area = 0.5 * ((x * np.roll(y, -1, axis=1)).sum(1)
                  - (y * np.roll(x, -1, axis=1)).sum(1))
    cr_pos = np.zeros(len(quads), dtype=bool)
    cr_neg = np.zeros(len(quads), dtype=bool)
    edges = []
    for i in range(4):
        a = v[:, i] - v[:, (i - 1) % 4]
        b = v[:, (i + 1) % 4] - v[:, i]
        cr = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        cr_pos |= cr > 0
        cr_neg |= cr < 0
        edges.append(np.linalg.norm(b, axis=1))
    nonconvex = cr_pos & cr_neg
    min_edge = np.min(edges, axis=0)
    return min_ang, area, nonconvex, min_edge, max_ang


def _write_su2(verts, quads, fill_tris, marker_lines, path,
               marker_names=MARKERS):
    with open(path, "w") as f:
        f.write("NDIME= 2\n")
        f.write(f"NELEM= {len(quads) + len(fill_tris)}\n")
        for q in quads:
            f.write(f"9 {q[0]} {q[1]} {q[2]} {q[3]}\n")
        for t in fill_tris:
            f.write(f"5 {t[0]} {t[1]} {t[2]}\n")
        f.write(f"NPOIN= {len(verts)}\n")
        for i, (x, y) in enumerate(verts):
            f.write(f"{x:.17g} {y:.17g} {i}\n")
        f.write(f"NMARK= {len(marker_names)}\n")
        for mi, marker in enumerate(marker_names):
            lines = [ml for ml in marker_lines if ml[0] == mi]
            f.write(f"MARKER_TAG= {marker}\n")
            f.write(f"MARKER_ELEMS= {len(lines)}\n")
            for _, a, b in lines:
                f.write(f"3 {a} {b}\n")


def _write_msh2(verts, quads, fill_tris, marker_lines, path,
                marker_names=MARKERS):
    phys = [(2, 1, "fluid")] + [
        (1, 2 + i, m) for i, m in enumerate(marker_names)]
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$PhysicalNames\n{len(phys)}\n")
        for dim, tag, name in phys:
            f.write(f'{dim} {tag} "{name}"\n')
        f.write("$EndPhysicalNames\n")
        f.write(f"$Nodes\n{len(verts)}\n")
        for i, (x, y) in enumerate(verts, 1):
            f.write(f"{i} {x:.17g} {y:.17g} 0\n")
        f.write("$EndNodes\n")
        n = len(quads) + len(fill_tris) + len(marker_lines)
        f.write(f"$Elements\n{n}\n")
        eid = 1
        for q in quads:
            f.write(f"{eid} 3 2 1 1 {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
            eid += 1
        for t in fill_tris:
            f.write(f"{eid} 2 2 1 1 {t[0]+1} {t[1]+1} {t[2]+1}\n")
            eid += 1
        for mi, a, b in marker_lines:
            f.write(f"{eid} 1 2 {2+mi} {2+mi} {a+1} {b+1}\n")
            eid += 1
        f.write("$EndElements\n")


def _write_obj(verts, quads, tris, path):
    with open(path, "w") as f:
        f.write("# quad/tri mesh\n")
        for x, y in verts:
            f.write(f"v {x:.17g} {y:.17g} 0\n")
        for q in quads:
            f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
        for t in tris:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
