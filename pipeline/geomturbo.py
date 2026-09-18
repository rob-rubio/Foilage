"""Read and write the native ASCII NUMECA ``.geomTurbo`` format.

The native blade geometry is stored as two explicit ``SECTIONAL`` blocks:
``suction`` and ``pressure``.  Each contains one or more sections, and each
section contains an ``XYZ`` point count followed by coordinates.  In the
files examined by this project, NUMECA stores those coordinates as ``Z Y X``;
the public Python reader used as a cross-check also converts them to internal
``X Y Z`` order.  This module returns canonical ``X Y Z`` arrays so the rest
of the pipeline can work with the same coordinates as the generated
airfoils.

The older simplified ``BEGINSECTION`` layout is retained as a read-only
fallback for files produced by earlier Foilage versions.  It cannot provide
explicit SS/PS arrays, so callers should prefer native files.
"""

import re
from pathlib import Path

import numpy as np


_BEGIN_GEOMETRY = re.compile(
    r"^\s*#?\s*NI_BEGIN\s+nibladegeometry\b", re.IGNORECASE)
_END_GEOMETRY = re.compile(
    r"^\s*#?\s*NI_END\s+nibladegeometry\b", re.IGNORECASE)
_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$")


def _float_token(token):
    """Parse NUMECA/Fortran-style numeric tokens."""
    return float(token.strip().replace("−", "-").replace("D", "E")
                .replace("d", "e"))


def _numeric_row(line, n=3):
    parts = line.replace(",", " ").replace("−", "-").split()
    if len(parts) < n or not all(_NUMBER.fullmatch(p) for p in parts[:n]):
        return None
    try:
        return [_float_token(p) for p in parts[:n]]
    except ValueError:
        return None


def _integer_line(line):
    """Return an integer from a standalone count line, or ``None``."""
    token = line.split("#", 1)[0].strip()
    if not re.fullmatch(r"[+-]?\d+", token):
        return None
    return int(token)


def _canonical_points(raw):
    """Convert file-order ``Z Y X`` coordinates to canonical ``X Y Z``."""
    raw = np.asarray(raw, dtype=float)
    return raw[:, [2, 1, 0]]


def _skip_comments(lines, index):
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        break
    return index


def _parse_sectional_block(lines, side_name):
    """Parse one native ``suction`` or ``pressure`` block."""
    sectional = next(
        (i for i, line in enumerate(lines)
         if line.strip().upper().startswith("SECTIONAL")), None)
    if sectional is None:
        raise ValueError(f"{side_name} block has no SECTIONAL marker")

    index = _skip_comments(lines, sectional + 1)
    if index >= len(lines):
        raise ValueError(f"{side_name} block has no section count")
    section_count = _integer_line(lines[index].strip())
    if not section_count or section_count < 1:
        raise ValueError(f"invalid {side_name} section count")
    index += 1
    sections = []

    for section_index in range(section_count):
        index = _skip_comments(lines, index)
        point_count = None
        while index < len(lines):
            line = lines[index].strip()
            if not line or line.startswith("#"):
                index += 1
                continue

            # Native files normally put XYZ and the count on separate lines;
            # accepting both forms makes the reader tolerant of exporters
            # that collapse the two lines.
            inline = re.match(
                r"^(?:XYZ|X\s+Y\s+Z)\s+([+-]?\d+)\s*$",
                line, re.IGNORECASE)
            if inline:
                point_count = int(inline.group(1))
                index += 1
                break

            count = _integer_line(line)
            if count is not None:
                point_count = count
                index += 1
                break

            # XYZ, X Y Z, or an exporter-specific section label.
            index += 1

        if not point_count or point_count < 2:
            raise ValueError(
                f"invalid point count for {side_name} section {section_index + 1}")

        raw_points = []
        while len(raw_points) < point_count:
            if index >= len(lines):
                raise ValueError(
                    f"{side_name} section {section_index + 1} ends before "
                    f"{point_count} points are read")
            line = lines[index].strip()
            index += 1
            if not line or line.startswith("#"):
                continue
            values = _numeric_row(line, 3)
            if values is None:
                raise ValueError(
                    f"invalid point in {side_name} section {section_index + 1}: "
                    f"{line!r}")
            raw_points.append(values)

        points = _canonical_points(raw_points)
        sections.append({
            "index": section_index,
            "z": float(np.mean(points[:, 2])),
            "points": points,
        })

    return sections


def _blade_count(lines):
    """Extract the periodic blade count used by native geomTurbo files."""
    patterns = (
        r"\bnumber_of_blades\s*=?\s*(\d+)",
        r"\bperiodicity\s*=?\s*(\d+)",
        # A few exporters use ``number_of 24`` instead of the longer key.
        r"\bnumber_of\s*=?\s*(\d+)",
    )
    for pattern in patterns:
        for line in lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                count = int(match.group(1))
                if count > 0:
                    return count
    return None


def _units_scale(lines):
    for line in lines:
        match = re.match(r"^\s*units\s+([^#\s]+)", line, re.IGNORECASE)
        if not match:
            continue
        token = match.group(1)
        try:
            return _float_token(token)
        except ValueError:
            return {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3,
                    "in": 0.0254}.get(token.lower())
    return None


def _native_geometry_blocks(lines):
    blocks = []
    start = None
    for index, line in enumerate(lines):
        if start is None and _BEGIN_GEOMETRY.match(line):
            start = index
        elif start is not None and _END_GEOMETRY.match(line):
            blocks.append(lines[start:index + 1])
            start = None
    if start is not None:
        blocks.append(lines[start:])
    return blocks


def _parse_native(lines, path):
    geometries = []
    for block in _native_geometry_blocks(lines):
        side_positions = [
            (i, line.strip().lower()) for i, line in enumerate(block)
            if line.strip().lower() in {"suction", "pressure"}
        ]
        if not side_positions:
            continue
        parsed = {}
        for position, side in side_positions:
            end = next((i for i, _ in side_positions if i > position),
                       len(block))
            parsed[side] = _parse_sectional_block(
                block[position + 1:end], side)
        if "suction" not in parsed or "pressure" not in parsed:
            continue
        if len(parsed["suction"]) != len(parsed["pressure"]):
            raise ValueError(
                f"{path.name}: suction and pressure section counts differ")
        blade_count = _blade_count(block) or _blade_count(lines)
        if blade_count is None:
            raise ValueError(
                f"{path.name}: no number_of_blades/periodicity value found")

        sections = []
        for index, (ss, ps) in enumerate(
                zip(parsed["suction"], parsed["pressure"])):
            if len(ss["points"]) < 2 or len(ps["points"]) < 2:
                raise ValueError(f"{path.name}: section {index + 1} has too few points")
            # Keep the legacy combined representation as a compatibility aid,
            # but expose the authoritative SS and PS arrays explicitly.
            sections.append({
                "index": index,
                "z": float(np.mean([ss["z"], ps["z"]])),
                "ss": ss["points"],
                "ps": ps["points"],
                "points": np.vstack([ps["points"][::-1], ss["points"][1:]]),
            })
        geometries.append({"blade_count": blade_count, "sections": sections})

    if not geometries:
        return None
    result = geometries[0]
    return {
        "blade_count": result["blade_count"],
        "sections": result["sections"],
        "blade_geometries": geometries,
        "units": _units_scale(lines),
    }


def _parse_legacy(lines, path):
    """Read Foilage's former combined-polyline format."""
    sections = []
    current = None
    in_blade = True
    collecting = False
    count = 0
    points = []
    for raw in lines:
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("BEGINBLADE"):
            in_blade = True
            continue
        if upper.startswith("ENDBLADE"):
            in_blade = False
            continue
        if upper.startswith("BEGINSECTION"):
            z_value = 0.0
            match = re.search(r"\bZ\s*=\s*([^\s]+)", line, re.IGNORECASE)
            if match:
                try:
                    z_value = _float_token(match.group(1))
                except ValueError:
                    pass
            current = {"z": z_value, "points": None}
            continue
        if upper.startswith("ENDSECTION"):
            if current is not None and points:
                current["points"] = np.asarray(points, dtype=float)
                sections.append(current)
            current, collecting, points = None, False, []
            continue
        if current is None or not in_blade:
            continue
        if "NUMBEROFPOINTS" in upper and "=" in line:
            try:
                count = int(float(line.split("=", 1)[1].split()[0]))
            except (ValueError, IndexError):
                count = 0
            collecting = count > 0
            points = []
            continue
        if collecting:
            values = _numeric_row(line, 2)
            if values is None:
                continue
            points.append(values + [0.0] if len(values) == 2 else values)
            if count and len(points) >= count:
                collecting = False

    if not sections:
        return None
    blade_count = _blade_count(lines)
    if blade_count is None:
        match = re.search(r"\bNK\s*=\s*(\d+)", "\n".join(lines), re.IGNORECASE)
        blade_count = int(match.group(1)) if match else 1
    return {"blade_count": blade_count, "sections": sections,
            "blade_geometries": [], "units": _units_scale(lines)}


def parse_geomturbo(path):
    """Parse a NUMECA ``.geomTurbo`` file.

    Returns a dictionary with ``blade_count`` and ``sections``.  Native
    sections contain ``ss`` and ``ps`` arrays in canonical ``X Y Z`` order,
    plus ``z`` and a compatibility ``points`` polyline.  ``blade_geometries``
    contains all native blade-geometry blocks when a file has more than one.
    """
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()
    parsed = _parse_native(lines, path)
    if parsed is not None:
        return parsed
    parsed = _parse_legacy(lines, path)
    if parsed is not None:
        return parsed
    raise ValueError(f"no native blade sections found in {path.name}")


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

    ``ss`` and ``ps`` are canonical ``X Y`` arrays ordered LE -> TE.  Native
    NUMECA point rows are written as ``Z Y X``.  ``units`` may be a NUMECA
    scale factor or a common unit name such as ``"m"`` or ``"mm"``.
    """
    ss = np.atleast_2d(np.asarray(ss, dtype=float))
    ps = np.atleast_2d(np.asarray(ps, dtype=float))
    if ss.shape[1] < 2 or ps.shape[1] < 2:
        raise ValueError("SS and PS arrays must contain at least X and Y")
    if len(ss) < 2 or len(ps) < 2:
        raise ValueError("SS and PS arrays must each contain at least two points")

    if isinstance(units, str):
        units = {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3,
                 "in": 0.0254}.get(units.lower(), units)
    try:
        units = float(units)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported geomTurbo units value: {units!r}") from exc

    lines = [
        "TYPE GEOMTURBO",
        "GEOMETRY_MODIFIED 0",
        "GEOMETRY TURBO VERSION 5",
        f"units {units:g}",
        "NI_BEGIN GEOMTURBO",
        "NI_BEGIN nirow",
        "NAME Foilage",
        "PERIODICITY 1",
        "NI_BEGIN NIBlade",
        "NAME Foilage",
        "NI_BEGIN nibladegeometry",
        "number_of_blades 1",
    ]

    def add_side(name, points):
        lines.extend([name, "SECTIONAL", "1", "# SECTION 1", "XYZ",
                      str(len(points))])
        lines.extend(f"{z:.9f} {y:.9f} {x:.9f}"
                     for x, y in points[:, :2])

    add_side("suction", ss)
    add_side("pressure", ps)
    lines += [
        "NI_END nibladegeometry",
        "NI_END NIBlade",
        "NI_END nirow",
        "NI_END GEOMTURBO",
    ]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def section_to_airfoil(points, n_points=401):
    """Convert one section polyline to the pipeline airfoil dict.

    Normalizes like the generated-blade view: LE at x = 0, scale set by
    the axial distance LE -> suction-side trailing point, blade centered
    vertically. Native parser section dictionaries use their explicit
    ``ss``/``ps`` arrays; legacy callers may still pass open polylines
    (TE -> PS -> LE -> SS -> TE) or closed loops.
    Returns {"ss", "ps", "outline", "ss_upper", "style"} (no airfoil2d
    object - that is pyturbo-specific).
    """
    explicit_sides = (isinstance(points, dict) and
                      "ss" in points and "ps" in points)

    def clean_side(side):
        side = np.asarray(side, dtype=float)[:, :2]
        keep = [0]
        for i in range(1, len(side)):
            if np.linalg.norm(side[i] - side[keep[-1]]) > 1e-12:
                keep.append(i)
        side = side[keep]
        if len(side) > 1 and np.linalg.norm(side[0] - side[-1]) <= 1e-12:
            side = side[:-1]
        if len(side) < 2:
            raise ValueError("section side has too few points")
        # Native exporters differ on whether each side is written LE -> TE
        # or TE -> LE.  The axial coordinate identifies the orientation for
        # the 2-D sections handled by this pipeline.
        if side[0, 0] > side[-1, 0]:
            side = side[::-1]
        return side

    if explicit_sides:
        side_a = clean_side(points["ss"])
        side_b = clean_side(points["ps"])
        ss_raw, ps_raw = side_a, side_b
    else:
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
            guard = max(3, n_all // 20)          # keep clear of TE seam
            curv[:guard] = 0.0
            curv[n_all - guard:] = 0.0

            def arc(i_from, i_to):
                k = (i_to - i_from) % n_all
                return pts[(i_from + np.arange(k + 1)) % n_all]

            le = int(np.argmax(curv))
            side_a = arc(le, 0)                  # LE -> TE seam
            side_b = arc(0, le)[::-1]            # LE -> TE seam
        else:
            chord = e2 - e1
            L = np.linalg.norm(chord)
            # LE = point farthest from the chord line (first -> last point)
            d = np.abs(np.cross(chord / L, pts - e1))
            i_le = int(np.argmax(d))
            side_a = pts[:i_le + 1][::-1]        # LE -> E1
            side_b = pts[i_le:]                  # LE -> E2
            for s in (side_a, side_b):
                if s[0, 0] > s[-1, 0]:           # orient each side LE -> TE
                    s[:] = s[::-1]

        # For a legacy combined polyline the surface roles are not encoded;
        # infer them from convexity as the previous implementation did.
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
