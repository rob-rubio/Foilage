"""Minimal parser for binary legacy VTK (UNSTRUCTURED_GRID) as written by SU2.

Returns point coordinates, triangle/quad connectivity, and all point-data
fields, with automatic endianness detection (VTK spec is big-endian but SU2
may write little-endian).
"""

import numpy as np

GAMMA = 1.4
R_AIR = 287.058


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
                data["points"] = np.frombuffer(f.read(3 * n * 4), dtype=dtype).reshape(n, 3)[:, :2].astype(np.float64)
            elif s.startswith("CELLS"):
                n, size = int(s.split()[1]), int(s.split()[2])
                buf = np.frombuffer(f.read(size * 4), dtype=np.dtype(">i4" if dtype.byteorder == ">" else "<i4"))
                tris, quads = [], []
                i = 0
                for _ in range(n):
                    npp = int(buf[i])
                    conn = buf[i + 1: i + 1 + npp]
                    (tris if npp == 3 else quads).append(conn)
                    i += 1 + npp
                data["tris"] = np.array(tris, dtype=np.int64) if tris else None
                data["quads"] = np.array(quads, dtype=np.int64) if quads else None
            elif s.startswith("CELL_TYPES"):
                f.read(int(s.split()[1]) * 4)
            elif s.startswith("POINT_DATA"):
                pass
            elif s.startswith("SCALARS"):
                name, ncomp = s.split()[1], int(s.split()[3])
                f.readline()  # LOOKUP_TABLE
                data[name] = np.frombuffer(f.read(len(data["points"]) * ncomp * 4), dtype=dtype).reshape(-1, ncomp).astype(np.float64)
            elif s.startswith("VECTORS"):
                name = s.split()[1]
                n = len(data["points"])
                data[name] = np.frombuffer(f.read(n * 3 * 4), dtype=dtype).reshape(n, 3).astype(np.float64)
    return data


def read_legacy_vtk(path):
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
            return d
    # fall back: choose by pressure plausibility
    best = None
    for ok_pts, endian, d in candidates:
        pr = d.get("Pressure")
        score = int(pr is not None and np.isfinite(pr).all())
        if best is None or score > best[0]:
            best = (score, d)
    if best is None:
        raise RuntimeError(f"could not parse {path}")
    return best[1]


def mach_number(d, gamma=GAMMA, r=R_AIR):
    v = d["Velocity"]
    speed = np.linalg.norm(v[:, :2], axis=1)
    return speed / np.sqrt(gamma * r * d["Temperature"].ravel())


if __name__ == "__main__":
    import sys

    d = read_legacy_vtk(sys.argv[1] if len(sys.argv) > 1 else "vol_solution.vtk")
    print("endian:", d["dtype"].byteorder)
    print("points:", d["points"].shape)
    print("tris:", None if d["tris"] is None else d["tris"].shape,
          "quads:", None if d["quads"] is None else d["quads"].shape)
    for k, v in sorted(d.items()):
        if isinstance(v, np.ndarray) and v.ndim == 2 and k not in ("points", "tris", "quads"):
            print(f"  {k}: {v.shape}  min={v.min():.5g}  max={v.max():.5g}")
