"""Minimal parser for binary legacy VTK (UNSTRUCTURED_GRID) as written by SU2.

Returns point coordinates, triangle/quad connectivity, and all point-data
fields, with automatic endianness detection (VTK spec is big-endian but SU2
may write little-endian).

3D wedge support: when the file holds a 3D volume mesh (hexes VTK 12 /
prisms VTK 13 from the axisymmetric3d extrusion), ``read_legacy_vtk``
slices it at the mid-span node layer and returns the same 2D-style dict as
a 2D case (``points`` (n, 2), ``tris``/``quads`` + per-node fields), so
every downstream 2D consumer keeps working. Span layers are identified
from the volume connectivity (a hex's lower/upper footprint faces chain
hub -> shroud), and the slice keeps the footprint faces whose layer rank
is the middle one (the interior slice is representative of the passage
away from the streamtube walls). Pass
``slice3d=False`` for the raw 3D dict (``points3d``, ``hexes``,
``prisms`` + full fields) - used for the inlet/outlet surface audits.
"""

import numpy as np

GAMMA = 1.4
R_AIR = 287.058

_VOLUME_NPP = {3: "tris", 4: "quads", 6: "prisms", 8: "hexes"}


def _as_float64(values):
    """Convert a candidate field without warning on invalid endian probes."""
    # read_legacy_vtk tries both byte orders.  The rejected interpretation can
    # contain signaling NaNs, which NumPy warns about while casting even
    # though that candidate is discarded by the plausibility checks.
    with np.errstate(invalid="ignore"):
        return values.astype(np.float64)


def _scan(path, dtype):
    """Single pass over the file with a given float dtype."""
    data = {"dtype": dtype}
    with open(path, "rb") as f:
        assert f.readline().decode("ascii").startswith("# vtk DataFile")
        f.readline()
        assert f.readline().decode("ascii").strip() == "BINARY"
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("ascii").strip()
            if s.startswith("POINTS"):
                n = int(s.split()[1])
                pts = _as_float64(
                    np.frombuffer(f.read(3 * n * 4), dtype=dtype)
                    .reshape(n, 3))
                data["points3d"] = pts
                data["points"] = pts[:, :2]
            elif s.startswith("CELLS"):
                n, size = int(s.split()[1]), int(s.split()[2])
                buf = np.frombuffer(f.read(size * 4), dtype=np.dtype(">i4" if dtype.byteorder == ">" else "<i4"))
                cells = {k: [] for k in _VOLUME_NPP.values()}
                i = 0
                for _ in range(n):
                    npp = int(buf[i])
                    conn = buf[i + 1: i + 1 + npp]
                    cells[_VOLUME_NPP.get(npp, "tris")].append(conn)
                    i += 1 + npp
                for key, lst in cells.items():
                    data[key] = (np.array(lst, dtype=np.int64)
                                 if lst else None)
            elif s.startswith("CELL_TYPES"):
                f.read(int(s.split()[1]) * 4)
            elif s.startswith("POINT_DATA"):
                pass
            elif s.startswith("SCALARS"):
                name, ncomp = s.split()[1], int(s.split()[3])
                f.readline()  # LOOKUP_TABLE
                data[name] = _as_float64(
                    np.frombuffer(
                        f.read(len(data["points"]) * ncomp * 4), dtype=dtype)
                    .reshape(-1, ncomp))
            elif s.startswith("VECTORS"):
                name = s.split()[1]
                n = len(data["points"])
                data[name] = _as_float64(
                    np.frombuffer(f.read(n * 3 * 4), dtype=dtype)
                    .reshape(n, 3))
    return data


def _is_3d(d):
    return (d.get("hexes") is not None and len(d["hexes"]) > 0) or \
        (d.get("prisms") is not None and len(d["prisms"]) > 0)


def _span_ranks(hexes, prisms, n_points):
    """Spanwise rank (0 = hub) of every node plus the layer count, derived
    from the volume connectivity: a hex's first four nodes sit on the lower
    span face and the last four on the face one layer above (prisms: 3+3).
    Faces that never appear as an upper face are hub-layer faces, so ranks
    follow by chaining - exact integer bookkeeping, immune to the float32
    coordinate noise of legacy VTK."""
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
        raise ValueError("wedge connectivity has no hub faces")
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


def _midspan_slice(d):
    """Reduce a parsed 3D wedge volume to the mid-span 2D-style dict."""
    pts3 = d["points3d"]
    hexes = d.get("hexes")
    hexes = hexes if hexes is not None and len(hexes) else []
    prisms = d.get("prisms")
    prisms = prisms if prisms is not None and len(prisms) else []
    rank, n_layers = _span_ranks(hexes, prisms, len(pts3))
    mid = (n_layers - 1) // 2

    keep = rank == mid
    idx = np.where(keep)[0]
    remap = np.full(len(pts3), -1, dtype=np.int64)
    remap[idx] = np.arange(len(idx))

    quads, tris = None, None
    hexes, prisms = d.get("hexes"), d.get("prisms")
    if hexes is not None and len(hexes):
        sel = hexes[rank[hexes[:, 0]] == mid]
        quads = remap[sel[:, :4]]
    if prisms is not None and len(prisms):
        sel = prisms[rank[prisms[:, 0]] == mid]
        tris = remap[sel[:, :3]]

    out = {"points": pts3[idx][:, :2], "points3d": pts3[idx],
           "tris": tris, "quads": quads,
           "dtype": d.get("dtype"), "_was3d": True,
           "_n_layers": n_layers, "_mid_layer": mid}
    n = len(pts3)
    for name, arr in d.items():
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            continue
        if name in ("points", "points3d", "tris", "quads", "hexes",
                    "prisms"):
            continue
        if len(arr) == n:
            out[name] = arr[idx]
    return out


def read_legacy_vtk(path, slice3d=True):
    candidates = []
    for endian in (">", "<"):
        try:
            d = _scan(path, np.dtype(endian + "f4"))
        except Exception:
            continue
        p = d.get("points")
        ok_pts = (
            p is not None
            and np.isfinite(p).all()
            and p[:, 0].min() > -1e4 and p[:, 0].max() < 1e4
            and p[:, 1].min() > -1e4 and p[:, 1].max() < 1e4
        )
        candidates.append((ok_pts, endian, d))
    # prefer parse with plausible points, then plausible pressure
    for ok_pts, endian, d in candidates:
        if ok_pts:
            chosen = d
            break
    else:
        # fall back: choose by pressure plausibility
        best = None
        for ok_pts, endian, d in candidates:
            pr = d.get("Pressure")
            score = int(pr is not None and np.isfinite(pr).all())
            if best is None or score > best[0]:
                best = (score, d)
        if best is None:
            raise RuntimeError(f"could not parse {path}")
        chosen = best[1]
    if _is_3d(chosen) and slice3d:
        return _midspan_slice(chosen)
    return chosen


def mach_number(d, gamma=GAMMA, r=R_AIR):
    v = d["Velocity"]
    speed = np.linalg.norm(v[:, :2], axis=1)
    return speed / np.sqrt(gamma * r * d["Temperature"].ravel())


if __name__ == "__main__":
    import sys

    d = read_legacy_vtk(sys.argv[1] if len(sys.argv) > 1 else "vol_solution.vtk")
    print("endian:", d["dtype"].byteorder)
    print("points:", d["points"].shape)
    if d.get("_was3d"):
        print(f"3D wedge volume: sliced at mid-span layer "
              f"{d['_mid_layer']} of {d['_n_layers']}")
    print("tris:", None if d["tris"] is None else d["tris"].shape,
          "quads:", None if d["quads"] is None else d["quads"].shape)
    for k, v in sorted(d.items()):
        if isinstance(v, np.ndarray) and v.ndim == 2 and k not in ("points", "tris", "quads"):
            print(f"  {k}: {v.shape}  min={v.min():.5g}  max={v.max():.5g}")
