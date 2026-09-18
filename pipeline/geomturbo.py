"""NUMECA .geomTurbo reader + section -> airfoil conversion.

Parser handles the common AutoGrid ASCII layout: GLOBAL/BLADE/HUB/SHROUD
blocks, blade sections delimited by BEGINSECTION/ENDSECTION with
"NUMBEROFPOINTSNUMBEROFPOINTS= n" followed by n "x y z" lines. Only the
blade sections are used; HUB/SHROUD/GLOBAL are skipped.

A section polyline typically runs TE -> PS -> LE -> SS -> TE (or the
reverse); the parser is order-agnostic: it locates the LE as the point
farthest from the chord line (first-to-last point), splits the two
surfaces, orients them LE -> TE, and classifies suction/pressure by
enclosed area (the suction side is the more convex one).
"""

from pathlib import Path

import numpy as np


def parse_geomturbo(path):
    """Parse blade sections from a .geomTurbo file.

    Returns [{"z": float, "points": (n,3) ndarray}, ...] in file order.
    """
    path = Path(path)
    sections = []
    current = None          # inside BEGINSECTION..ENDSECTION
    in_blade = True         # only blade blocks define sections we use
    collecting = False
    count = 0
    pts = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        u = line.upper()
        if u.startswith("BEGINBLADE"):
            in_blade = True
            continue
        if u.startswith("ENDBLADE"):
            in_blade = False
            continue
        if u.startswith("BEGINSECTION"):
            zval = 0.0
            if "Z=" in u:
                try:
                    zval = float(u.split("Z=")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            current = {"z": zval, "points": None}
            continue
        if u.startswith("ENDSECTION"):
            if current is not None and pts:
                current["points"] = np.asarray(pts, dtype=float)
                sections.append(current)
            current, collecting, pts = None, False, []
            continue
        if current is None or not in_blade:
            continue
        if "NUMBEROFPOINTS" in u and "=" in u:
            try:
                count = int(float(u.split("=")[1].split()[0]))
            except (ValueError, IndexError):
                count = 0
            collecting = count > 0
            pts = []
            continue
        if collecting:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                vals = [float(p) for p in parts[:3]]
            except ValueError:
                continue
            while len(vals) < 3:
                vals.append(0.0)
            pts.append(vals)
            if count and len(pts) >= count:
                collecting = False
    if not sections:
        raise ValueError(f"no blade sections found in {path.name}")
    return sections


def _resample(side, n):
    """Resample an open polyline to n points, evenly in arc length."""
    seg = np.linalg.norm(np.diff(side, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(side[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)     # in arc-length units, NOT 0..1
    return np.column_stack([np.interp(u, s, side[:, 0]),
                            np.interp(u, s, side[:, 1])])


def _poly_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def write_geomturbo(path, ss, ps, z=0.0, units="m"):
    """Write a single-section .geomTurbo file from SS/PS point arrays.

    The section polyline runs TE -> PS -> LE -> SS -> TE (the common
    AutoGrid convention and exactly what parse_geomturbo expects).
    Coordinates are written in the given units - pass physical values
    (e.g. meters).
    """
    ss = np.atleast_2d(np.asarray(ss, dtype=float))
    ps = np.atleast_2d(np.asarray(ps, dtype=float))
    pts = np.vstack([ps[::-1], ss[1:]])
    lines = ["================= GLOBAL =================",
             f"UNITS= {units}",
             "================= BLADE =================",
             "NK= 1",
             "BEGINBLADE",
             f"BEGINSECTION Z= {z:g}",
             f"NUMBEROFPOINTSNUMBEROFPOINTS= {len(pts)}"]
    lines += [f"{x:.9f} {y:.9f} 0.0" for x, y in pts[:, :2]]
    lines += ["ENDSECTION", "ENDBLADE"]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def section_to_airfoil(points, n_points=401):
    """Convert one section polyline to the pipeline airfoil dict.

    Normalizes like the generated-blade view: LE at x = 0, scale set by
    the axial distance LE -> suction-side trailing point, blade centered
    vertically. Accepts open polylines (TE -> PS -> LE -> SS -> TE; the
    two ends are the TE arc ends) and closed loops (first point repeats:
    the seam is the TE, and the LE is the sharpest point away from it).
    Returns {"ss", "ps", "outline", "ss_upper", "style"} (no airfoil2d
    object - that is pyturbo-specific).
    """
    pts = np.asarray(points, dtype=float)[:, :2]
    if len(pts) < 10:
        raise ValueError("section has too few points")

    # dedupe consecutive repeats (TE arc seams often duplicate a point)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-12:
            keep.append(i)
    pts = pts[keep]

    e1, e2 = pts[0], pts[-1]
    if np.linalg.norm(e2 - e1) <= 1e-9:
        # closed loop (sharp TE written as a single repeated point): the
        # seam is the TE, and the LE is the sharpest point away from the
        # seam neighborhood (max turning angle per arc length)
        n_all = len(pts)
        nxt = np.roll(pts, -1, axis=0)
        prv = np.roll(pts, 1, axis=0)
        v1, v2 = pts - prv, nxt - pts
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        ang = np.abs(np.arctan2(cross, (v1 * v2).sum(axis=1)))
        seg = np.linalg.norm(v1, axis=1)
        curv = ang / (0.5 * (seg + np.roll(seg, 1)) + 1e-18)
        guard = max(3, n_all // 20)          # keep clear of the TE seam
        curv[:guard] = 0.0
        curv[n_all - guard:] = 0.0

        def arc(i_from, i_to):
            k = (i_to - i_from) % n_all
            return pts[(i_from + np.arange(k + 1)) % n_all]

        le = int(np.argmax(curv))
        side_a = arc(le, 0)                  # LE -> TE seam
        side_b = arc(0, le)[::-1]            # reversed: LE -> TE seam
    else:
        chord = e2 - e1
        L = np.linalg.norm(chord)
        # LE = point farthest from the chord line (first -> last point)
        d = np.abs(np.cross(chord / L, pts - e1))
        i_le = int(np.argmax(d))
        side_a = pts[:i_le + 1][::-1]        # LE -> E1
        side_b = pts[i_le:]                  # LE -> E2
        for s in (side_a, side_b):
            if s[0, 0] > s[-1, 0]:           # orient each side LE -> TE (+x)
                s[:] = s[::-1]

    # classify the suction side as the more convex one (larger area
    # enclosed between the side and its chord segment - the shoelace
    # formula closes the polyline automatically); near-symmetric sections
    # fall back to the mean-y rule. A mislabel only swaps the SS/PS
    # names in plots and ma_af.json - the mesh itself is unaffected.
    area_a = _poly_area(side_a)
    area_b = _poly_area(side_b)
    if abs(area_a - area_b) <= 1e-9 * max(area_a, area_b, 1e-12):
        ss_is_a = np.mean(side_a[:, 1]) > np.mean(side_b[:, 1])
    else:
        ss_is_a = area_a > area_b
    ss_raw, ps_raw = (side_a, side_b) if ss_is_a else (side_b, side_a)

    ss = _resample(ss_raw, n_points)
    ps = _resample(ps_raw, n_points)

    # normalize exactly like the generated-blade view: LE at x = 0 and the
    # scale set by the axial distance to the suction-side trailing point.
    # (Normalizing by the overall extent would compress the reference by
    # the TE-overshoot fraction and make overlays drift toward the TE.)
    ss[0] = ps[0] = 0.5 * (ss[0] + ps[0])
    if np.linalg.norm(ss[-1] - ps[-1]) > 1e-9:
        ss = np.vstack([ss, ps[-1]])

    x_le = float(min(ss[:, 0].min(), ps[:, 0].min()))
    x_te = float(ss[-1, 0])
    ax = max(x_te - x_le, 1e-12)
    ss[:, 0] = (ss[:, 0] - x_le) / ax
    ps[:, 0] = (ps[:, 0] - x_le) / ax
    y_mid = 0.5 * (min(ss[:, 1].min(), ps[:, 1].min())
                   + max(ss[:, 1].max(), ps[:, 1].max()))
    ss[:, 1] = (ss[:, 1] - y_mid) / ax
    ps[:, 1] = (ps[:, 1] - y_mid) / ax

    ss_upper = bool(ss[:, 1].mean() > ps[:, 1].mean())
    upper, lower = (ss, ps) if ss_upper else (ps, ss)
    outline = np.vstack([upper, lower[::-1][1:-1]])
    return {"ss": ss, "ps": ps, "outline": outline,
            "ss_upper": ss_upper, "style": "geomTurbo import"}
