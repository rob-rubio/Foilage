"""Gmsh stage: periodic cascade / freestream domain, boundary layers,
all-triangle mesh.

``domain.periodicity`` selects the domain (see pipeline/unwrap.py):

    axisymmetric   - unwrapped annular cascade sector: pitch varies with
                     the annulus radius, p(x) = 2 pi R(x) / N (R linearly
                     interpolated R1 -> R2 over the blade, constant
                     fore/aft). Top/bottom are a translation-periodic pair.
    axisymmetric3d - the same unwrapped sector extruded into a true
                     conical wedge (pipeline/extrude3d.py): the 2D mesh
                     is wrapped to polar coordinates and stacked spanwise
                     between the streamtube walls r = R(x) -/+ h(x)/2,
                     with h(x) the logistic streamtube depth (domain.h1
                     -> domain.h2 over the blade, clamped fore/aft). R(x)
                     continues linearly beyond the LE and TE. The
                     periodic pair is rotational, so R1 != R2 is legal.
    offset         - linear cascade: constant pitch p = 2 pi R1 / N and
                     straight periodic lines through y = 0 -/+ p/2 (no
                     tangential offset); y is Cartesian.
    freestream     - no periodics: the upper/lower boundaries are
                     far-field lines at domain.y_min / domain.y_max
                     (isolated airfoil calculations).

The blade sits in the middle of the passage; in the periodic modes the
two domain edges are the passage medial axes  y = y_c(x) -/+ p(x)/2
(y_c = blade circumferential centerline from the outline, held constant
fore of LE / aft of TE; identically 0 in offset mode). The top edge is
the bottom edge offset by the local pitch, so both are tagged as the
periodic pair. Each periodic edge is built from short transfinite
segments sharing the same x-stations on both edges, which keeps the
periodic node correspondence tight (verified and reported downstream).

Physical groups: inlet, outlet, periodic_bottom + periodic_top
(axisymmetric / offset) or farfield (freestream), airfoil, fluid.
"""

import numpy as np

try:
    from unwrap import periodicity_of, UNWRAPPED_MODES
except ImportError:                                  # package-style import
    from pipeline.unwrap import periodicity_of, UNWRAPPED_MODES

TYPE_NAMES = {1: "line", 2: "triangle", 3: "quad"}
NSEG_PERIODIC = 60  # transfinite segments per periodic edge


def pitch_profile(dom_cfg, airfoil):
    """Pitch p(x), radius R(x) and blade centerline y_c(x) per the
    selected periodicity mode.

    ``domain.R1`` / ``domain.R2`` are the annulus radii at the LE/TE in
    *actual units* (the same units as ``airfoil.axial_chord``, e.g. mm).
    They are normalized by the axial chord here so the passage is built
    in normalized coordinates (blade axial chord = 1). ``domain.h1`` /
    ``domain.h2`` are the streamtube depths at the LE/TE in the same
    units; h(x) transitions between them with a logistic (steepness
    ``domain.h_steepness``, default 10) over the blade and is clamped to
    h1 fore of the LE / h2 aft of the TE. In axisymmetric3d mode, R(x)
    extrapolates the LE-to-TE linear transition beyond both edges; other
    modes keep R constant outside the blade. In offset mode
    R2 is ignored (constant pitch through R1) and the centerline is
    straight; in freestream mode the pitch values are unused and the
    domain extends to domain.y_min / domain.y_max instead."""
    mode = periodicity_of(dom_cfg)
    N = float(dom_cfg["airfoil_count"])
    ac = float(airfoil.get("axial_chord") or 0.0)
    r1 = float(dom_cfg["R1"])
    r2 = float(dom_cfg.get("R2", dom_cfg["R1"]))
    h1 = dom_cfg.get("h1")
    h2 = dom_cfg.get("h2")
    if h1 is None:
        h1 = h2 if h2 is not None else 2.0 * np.pi * r1 / N
    if h2 is None:
        h2 = h1
    h1, h2 = float(h1), float(h2)
    h_s = float(dom_cfg.get("h_steepness") or 10.0)
    if ac > 0:                       # actual units -> normalized
        r1 /= ac
        r2 /= ac
        h1 /= ac
        h2 /= ac
    if mode == "offset":
        r2 = r1                      # no tangential offset: constant pitch

    p_le = 2.0 * np.pi * r1 / N
    p_te = 2.0 * np.pi * r2 / N

    ss, ps = airfoil["ss"], airfoil["ps"]
    x_le = float(min(ss[0, 0], ps[0, 0]))
    x_te = float(max(ss[-1, 0], ps[-1, 0]))

    # vertical mid-line: interpolate BOTH surfaces onto a common x grid.
    # The SS/PS spline samples never share x values, so a per-slice
    # min/max midpoint would just alternate between the two surfaces and
    # leave the periodic edges jagged after smoothing.
    uniq_x = np.unique(np.round(np.concatenate([ss[:, 0], ps[:, 0]]), 12))

    def _surf_interp(nodes):
        o = np.argsort(nodes[:, 0], kind="stable")
        xp, fp = nodes[o, 0], nodes[o, 1]
        keep = np.insert(np.diff(xp) > 1e-12, 0, True)
        return np.interp(uniq_x, xp[keep], fp[keep])

    y_mid = 0.5 * (_surf_interp(ss) + _surf_interp(ps))
    if len(y_mid) > 5:  # smooth outline discretization noise
        kernel = np.ones(5) / 5.0
        y_mid = np.convolve(np.pad(y_mid, 2, mode="edge"), kernel,
                            mode="valid")
    if mode not in UNWRAPPED_MODES:
        # offset / freestream: no tangential offset, y is Cartesian and
        # the domain is centered on the blade row (y_c identically 0)
        y_mid = np.zeros_like(y_mid)

    # assemble the centerline over the whole domain (constant asymptotes fore
    # of the LE / aft of the TE) and smooth it with a ~0.3-chord moving
    # average: rounds the C0 clamp corners at the LE/TE and the steep TE dive
    # so the periodic edges are kink-free, while the far field stays straight
    x0 = float(dom_cfg.get("x_min", -0.5))
    x1 = float(dom_cfg.get("x_max", 2.5))
    grid = np.linspace(x0, x1, 401)
    raw = np.interp(grid, uniq_x, y_mid, left=y_mid[0], right=y_mid[-1])
    w = max(5, int(round(0.30 / (x1 - x0) * len(grid))) | 1)
    smooth = raw
    for _ in range(2):
        pad = np.pad(smooth, w // 2, mode="edge")
        smooth = np.convolve(pad, np.ones(w) / w, mode="valid")

    def y_c(x):
        return np.interp(x, grid, smooth)

    def radius(x):
        t = (np.asarray(x, dtype=float) - x_le) / max(x_te - x_le, 1e-30)
        if mode != "axisymmetric3d":
            t = np.clip(t, 0.0, 1.0)
        return r1 + (r2 - r1) * t

    def h(x):
        """Streamtube depth: logistic h1 -> h2 over the blade, clamped to
        h1 fore of the LE and h2 aft of the TE (normalized units)."""
        t = (np.asarray(x, dtype=float) - x_le) / max(x_te - x_le, 1e-30)
        z = np.clip(h_s * (t - 0.5), -50.0, 50.0)
        sig = 1.0 / (1.0 + np.exp(-z))
        mid = h1 + (h2 - h1) * sig
        return np.where(t <= 0.0, h1, np.where(t >= 1.0, h2, mid))

    def p(x):
        return 2.0 * np.pi * radius(x) / N

    prof = {"p": p, "radius": radius, "h": h, "y_c": y_c,
            "e": lambda x: p(x) / 2.0,
            "p_le": p_le, "p_te": p_te, "x_le": x_le, "x_te": x_te,
            "h1": h1, "h2": h2, "h_steepness": h_s,
            "mode": mode, "periodic": mode != "freestream"}
    if mode == "freestream":
        prof["y_min"] = float(dom_cfg.get("y_min", -1.5))
        prof["y_max"] = float(dom_cfg.get("y_max", 1.5))
    return prof


def periodic_edges(prof, x_min, x_max, n=200):
    """Sample points of the bottom/top domain boundary lines: the
    periodic (medial axis) edges in the periodic modes, or the straight
    far-field boundaries of the freestream box."""
    xs = np.linspace(x_min, x_max, n)
    if not prof.get("periodic", True):
        lo = np.full(n, float(prof.get("y_min", -1.5)))
        hi = np.full(n, float(prof.get("y_max", 1.5)))
        return np.column_stack([xs, lo]), np.column_stack([xs, hi])
    e = np.asarray(prof["e"](xs))
    yc = np.asarray(prof["y_c"](xs))
    return np.column_stack([xs, yc - e]), np.column_stack([xs, yc + e])


def mesh_domain(airfoil, dom_cfg, mesh_cfg, prefix,
                bl_quads=False, recombine=False):
    import gmsh          # lazy: only the meshing stage needs it
    prof = pitch_profile(dom_cfg, airfoil)
    x0 = dom_cfg.get("x_min", -0.5)
    x1 = dom_cfg.get("x_max", 2.5)
    ss, ps = airfoil["ss"], airfoil["ps"]
    bot, top = periodic_edges(prof, x0, x1)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("airfoil_periodic_passage")
    geo = gmsh.model.geo

    # ---- blade wall: SS LE->TE, PS TE->LE, shared LE/TE point tags ----
    le_pt = geo.addPoint(ss[0, 0], ss[0, 1], 0)
    te_pt = geo.addPoint(ss[-1, 0], ss[-1, 1], 0)
    ss_int = [geo.addPoint(p[0], p[1], 0) for p in ss[1:-1]]
    ps_rev = ps[::-1]
    ps_int = [geo.addPoint(p[0], p[1], 0) for p in ps_rev[1:-1]]
    c_ss = geo.addSpline([le_pt] + ss_int + [te_pt])
    c_ps = geo.addSpline([te_pt] + ps_int + [le_pt])
    geo.synchronize()

    # optional: force the wall discretization to a fixed node count per
    # surface side (mesh.airfoil_points); without it Gmsh derives the wall
    # spacing from the size fields
    n_af = int(mesh_cfg.get("airfoil_points", 0) or 0)
    if n_af > 10:
        geo.mesh.setTransfiniteCurve(c_ss, n_af)
        geo.mesh.setTransfiniteCurve(c_ps, n_af)

    # hole loop wound clockwise (solid on the left of travel): traverse the
    # upper surface LE->TE first, then the lower one TE->LE
    wall_loop = (geo.addCurveLoop([c_ss, c_ps]) if airfoil["ss_upper"]
                 else geo.addCurveLoop([c_ps, c_ss]))

    # ---- outer boundary: split periodic edges into matched segments ----
    # both periodic edges use the same x-stations; equal transfinite counts
    # per segment keep node x-positions aligned between the two edges.
    # Segment count: explicit mesh.periodic_segments (integer), else derived
    # from mesh.periodic_size (target x-spacing, chord units), else 60.
    periodic = prof.get("periodic", True)
    if periodic:
        nseg = int(mesh_cfg.get("periodic_segments", 0) or 0)
        if nseg <= 0:
            psize = float(mesh_cfg.get("periodic_size", 0.0) or 0.0)
            nseg = int(np.ceil((x1 - x0) / psize)) if psize > 0.0 else NSEG_PERIODIC
        nseg = max(nseg, 1)
        bot_y = lambda x: prof["y_c"](x) - float(prof["e"](x))
        top_y = lambda x: prof["y_c"](x) + float(prof["e"](x))
    else:
        # freestream box: two straight segments per far-field boundary
        stations3 = np.linspace(x0, x1, 3)
        bot_y = lambda x: prof.get("y_min", -1.5)
        top_y = lambda x: prof.get("y_max", 1.5)
    stations = np.linspace(x0, x1, (nseg if periodic else 2) + 1)
    bot_pts = [bot_y(x) for x in stations]
    top_pts = [top_y(x) for x in stations]

    bot_lines, top_lines = [], []
    prev_b = geo.addPoint(stations[0], bot_pts[0], 0)
    prev_t = geo.addPoint(stations[0], top_pts[0], 0)
    b_first, t_first = prev_b, prev_t
    for i in range(1, len(stations)):
        b = geo.addPoint(stations[i], bot_pts[i], 0)
        t = geo.addPoint(stations[i], top_pts[i], 0)
        bot_lines.append(geo.addLine(prev_b, b))
        top_lines.append(geo.addLine(prev_t, t))
        prev_b, prev_t = b, t
    b_last, t_last = prev_b, prev_t
    if periodic:
        for ln in bot_lines + top_lines:
            geo.mesh.setTransfiniteCurve(ln, 2)  # 1 element per segment

    # inlet (x = x0) and outlet (x = x1) closures; loop order: up the inlet,
    # +x along the top periodic edge, down the outlet, -x along the bottom
    c_in = geo.addLine(b_first, t_first)
    c_out = geo.addLine(t_last, b_last)

    outer_loop = geo.addCurveLoop(
        [c_in] + top_lines + [c_out] + [-l for l in bot_lines]
    )
    surf = geo.addPlaneSurface([outer_loop, wall_loop])
    geo.synchronize()

    # ---- physical groups = solver markers ----
    gmsh.model.addPhysicalGroup(2, [surf], name="fluid")
    gmsh.model.addPhysicalGroup(1, [c_ss, c_ps], name="airfoil")
    gmsh.model.addPhysicalGroup(1, [c_in], name="inlet")
    gmsh.model.addPhysicalGroup(1, [c_out], name="outlet")
    if periodic:
        gmsh.model.addPhysicalGroup(1, bot_lines, name="periodic_bottom")
        gmsh.model.addPhysicalGroup(1, top_lines, name="periodic_top")
    else:
        gmsh.model.addPhysicalGroup(1, bot_lines + top_lines,
                                    name="farfield")

    # ---- size fields ----
    h_far = mesh_cfg.get("max_size", 0.3)
    bl = mesh_cfg.get("boundary_layer")
    fields = []
    f_bl = None
    if bl:
        h1 = bl["first_layer_height"]
        ratio = bl.get("growth_rate", 1.2)
        if "n_layers" in bl:
            nl = int(bl["n_layers"])
            thickness = h1 * (ratio ** nl - 1) / (ratio - 1)
        else:
            thickness = float(bl["thickness"])

        f_bl = 1
        gmsh.model.mesh.field.add("BoundaryLayer", f_bl)
        gmsh.model.mesh.field.setNumbers(f_bl, "CurvesList", [c_ss, c_ps])
        gmsh.model.mesh.field.setNumber(f_bl, "Size", h1)
        gmsh.model.mesh.field.setNumber(f_bl, "Ratio", ratio)
        gmsh.model.mesh.field.setNumber(f_bl, "Thickness", thickness)
        gmsh.model.mesh.field.setNumber(f_bl, "Quads", 1 if bl_quads else 0)
        gmsh.model.mesh.field.setNumbers(f_bl, "FanPointsList", [te_pt])
        gmsh.model.mesh.field.setNumbers(
            f_bl, "FanPointsSizesList", [float(bl.get("te_fan_points", 12))]
        )

        h_wall = mesh_cfg.get("near_wall_size", 4.0 * h1)
        f_dist, f_thr = 2, 3
        gmsh.model.mesh.field.add("Distance", f_dist)
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [c_ss, c_ps])
        gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 600)

        gmsh.model.mesh.field.add("Threshold", f_thr)
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", h_wall)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", h_far)
        dist_min = 1.5 * thickness
        # Gmsh's Threshold degenerates to SizeMin everywhere when the ramp
        # is inverted (DistMax <= DistMin), pinning the whole domain to
        # near_wall_size and making mesh.max_size a no-op - so a
        # refine_dist that ends inside the boundary-layer stack is
        # stretched just past it
        dist_max = max(float(mesh_cfg.get("refine_dist", 0.6)),
                       dist_min * 1.05)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", dist_max)
        fields.append(f_thr)

    wake = mesh_cfg.get("wake")
    wake_angle = None
    if wake:
        # wake band leaves the TE along the metal angle (camberline tangent
        # at the TE), not axially; optionally overridden from input
        te = airfoil["ss"][-1]
        x_le, x_te = prof["x_le"], prof["x_te"]
        if wake.get("metal_angle_deg") is not None:
            theta = np.radians(float(wake["metal_angle_deg"]))
        else:
            d_up = 0.12 * (x_te - x_le)
            theta = np.arctan2(te[1] - float(prof["y_c"](x_te - d_up)),
                               x_te - (x_te - d_up))
        wake_angle = float(np.degrees(theta))
        cx, sy = np.cos(theta), np.sin(theta)

        # length of the band: march from the TE along d until leaving the
        # domain (through the outlet or the top periodic edge)
        bot_s, top_s = periodic_edges(prof, x0, x1, n=400)
        y_top_max = top_s[:, 1].max()
        L, t = 0.0, 0.0
        while t < 20.0:
            t += 0.02
            px_, py_ = te[0] + cx * t, te[1] + sy * t
            if px_ > x1 or py_ > y_top_max or py_ < bot_s[:, 1].min():
                break
            L = t

        # nested cylinders give a stepped transition (core -> ring -> far)
        half_w = float(wake.get("half_width", 0.15))
        trans = float(wake.get("transition", 0.4))
        h_far_local = mesh_cfg.get("max_size", 0.15)
        h_mid = wake["size"] + 0.5 * (h_far_local - wake["size"])
        for tag, radius, v_in in ((4, half_w, wake["size"]),
                                  (6, half_w + trans, h_mid)):
            gmsh.model.mesh.field.add("Cylinder", tag)
            gmsh.model.mesh.field.setNumber(tag, "XCenter", te[0] + 0.5 * L * cx)
            gmsh.model.mesh.field.setNumber(tag, "YCenter", te[1] + 0.5 * L * sy)
            gmsh.model.mesh.field.setNumber(tag, "ZCenter", 0.0)
            gmsh.model.mesh.field.setNumber(tag, "XAxis", L * cx)
            gmsh.model.mesh.field.setNumber(tag, "YAxis", L * sy)
            gmsh.model.mesh.field.setNumber(tag, "ZAxis", 0.0)
            gmsh.model.mesh.field.setNumber(tag, "Radius", radius)
            gmsh.model.mesh.field.setNumber(tag, "VIn", v_in)
            gmsh.model.mesh.field.setNumber(tag, "VOut", 1e22)
            fields.append(tag)

    if fields:
        f_min = 5
        gmsh.model.mesh.field.add("Min", f_min)
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
    if f_bl is not None:
        gmsh.model.mesh.field.setAsBoundaryLayer(f_bl)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_far)
    gmsh.option.setNumber("Mesh.Algorithm", mesh_cfg.get("algorithm", 6))
    if recombine:
        # Blossom recombination of the interior triangles; the BL layers are
        # already structured quads, so the full mesh keeps the exact size
        # field (boundary-layer spacing, wake band, near-wall grading)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

    gmsh.model.mesh.generate(2)
    stats = _stats()
    stats["pitch_le"] = prof["p_le"]
    stats["wake_metal_angle_deg"] = wake_angle
    stats["pitch_te"] = prof["p_te"]

    # verify periodic node correspondence on the tri mesh: both periodic
    # edges were built from the same x-stations, so the match must be exact
    if periodic:
        bot_tag = _marker_tag("periodic_bottom")
        top_tag = _marker_tag("periodic_top")
        bn, bc, _ = gmsh.model.mesh.getNodes(1, bot_tag, includeBoundary=True)
        tn, tc, _ = gmsh.model.mesh.getNodes(1, top_tag, includeBoundary=True)
        bc = bc.reshape(-1, 3)
        tc = tc.reshape(-1, 3)
        bo = np.argsort(bc[:, 0])
        to = np.argsort(tc[:, 0])
        if len(bn) == len(tn):
            dx = np.abs(bc[bo, 0] - tc[to, 0])
            dy = np.abs(tc[to, 1] - bc[bo, 1])
            p_st = prof["p"](bc[bo, 0])
            stats["periodic_tri_max_mismatch"] = float(
                np.max(np.hypot(dx, dy - p_st)))
        else:
            stats["periodic_tri_max_mismatch"] = float("inf")

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(prefix) + ".msh")
    gmsh.write(str(prefix) + ".su2")
    _write_obj(str(prefix) + ".obj")

    gmsh.finalize()
    return stats


def _marker_tag(name):
    import gmsh          # lazy: only the meshing stage needs it
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        if gmsh.model.getPhysicalName(dim, tag) == name:
            return tag
    raise RuntimeError(f"physical group {name} not found")


def _stats():
    import gmsh          # lazy: only the meshing stage needs it
    etypes, etags, _ = gmsh.model.mesh.getElements(2)
    counts = {TYPE_NAMES.get(t, t): len(tags) for t, tags in zip(etypes, etags)}
    stats = {"counts": counts, "total": sum(counts.values())}
    tris = gmsh.model.mesh.getElementsByType(2)[0]
    if len(tris):
        q = gmsh.model.mesh.getElementQualities(tris, "minSICN")
        stats["min_sicn"] = float(min(q))
        stats["mean_sicn"] = float(sum(q) / len(q))
    return stats


NNODES = {1: 2, 2: 3, 3: 4}


def _write_obj(path):
    import gmsh          # lazy: only the meshing stage needs it
    """Write the 2D mesh (triangles and/or quads) as a planar OBJ."""
    etypes, etags, enodes = gmsh.model.mesh.getElements(2)
    node_ids, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    x, y = coords[:, 0], coords[:, 1]
    order = {int(g): i for i, g in enumerate(node_ids)}

    with open(path, "w") as f:
        f.write("# planar 2D mesh (z=0)\n")
        for xi, yi in zip(x, y):
            f.write(f"v {xi:.17g} {yi:.17g} 0\n")
        f.write("s off\n")
        for i, t in enumerate(etypes):
            nn = NNODES[int(t)]
            for c in np.array(enodes[i]).reshape(-1, nn):
                f.write("f " + " ".join(str(order[int(v)] + 1) for v in c)
                        + "\n")
