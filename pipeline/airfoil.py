"""Build a 2D turbine airfoil with pyturbo-aero (left-to-right orientation).

The airfoil is always built with left_to_right=True: flow goes left-to-right,
angles measured from the +X axis. The profile is extracted with get_points(),
placed with the leading edge at the origin, and normalized so that the axial
chord equals 1.0. All downstream sizing (domain extents, mesh sizes, BL
heights) is expressed in these axial-chord units.
"""

import numpy as np


def build_geometry(cfg):
    """Dispatch on airfoil_source.type: pyturbo-aero generation (default),
    an imported geomTurbo section, or a plugin generator from the
    extensions/ directory (the plugin's CLI is run and its section is
    normalized exactly like a geomTurbo import).

    Returns the same airfoil dict for all sources so meshing and
    post-processing are source-agnostic. The dict carries the axial chord
    in actual units (mm when axial_chord >= 1, else meters) because the
    domain radii R1/R2 are also specified in actual units and must be
    normalized by it.

    Imported/plugin sections may additionally carry airfoil_source.morph:
    an FFD control cage (see pipeline/ffd.py) applied to the normalized
    section before it is used for meshing. The returned dict then includes
    "morph_cage" (cage box + deformed control points) for drawing.
    """
    src = cfg.get("airfoil_source") or {}
    stype = src.get("type") or "pyturbo"
    if stype == "pyturbo":
        af = build_airfoil(cfg["airfoil"])
        af["morph_cage"] = None
    else:
        try:
            from pipeline.geomturbo import parse_geomturbo, section_to_airfoil
        except ImportError:
            from geomturbo import parse_geomturbo, section_to_airfoil
        n = int(cfg.get("airfoil", {}).get("n_points", 401))
        if stype == "geomturbo":
            file_path = src.get("geomturbo_file")
            if not file_path:
                raise ValueError("geomTurbo source selected but "
                                 "airfoil_source.geomturbo_file is empty")
            parsed = parse_geomturbo(file_path)
            sections = parsed["sections"]
            idx = int(src.get("section") or 0)
            idx = max(0, min(idx, len(sections) - 1))
            sec = sections[idx]
            af = section_to_airfoil(sec, n_points=n)
            af["section_z"] = sec["z"]
            af["blade_count"] = parsed["blade_count"]
        else:
            try:
                from pipeline.plugins import run_plugin_generator
            except ImportError:
                from plugins import run_plugin_generator
            result = run_plugin_generator(stype, cfg)
            af = section_to_airfoil({"ss": result["ss"],
                                     "ps": result["ps"]}, n_points=n)
            af["section_z"] = None
            af["blade_count"] = result.get("blade_count")
        af["airfoil2d"] = None
        # imported and generated-plugin sections share the morph stage
        morph = src.get("morph")
        if isinstance(morph, dict) and morph.get("enabled"):
            try:
                from pipeline.ffd import apply_morph, rebuild_outline
            except ImportError:
                from ffd import apply_morph, rebuild_outline
            af["ss"], af["ps"], cage = apply_morph(af["ss"], af["ps"], morph)
            af["outline"] = rebuild_outline(af["ss"], af["ps"],
                                            af["ss_upper"])
            af["morph_cage"] = cage
        else:
            af["morph_cage"] = None
    af["axial_chord"] = float(cfg.get("airfoil", {}).get("axial_chord")
                              or 1.0)
    return af


def build_airfoil(af_cfg):
    """Return dict with normalized suction/pressure side arrays and metadata.

    af_cfg keys (all in pyturbo-aero units, see NASA 2D design tutorial):
        alpha1, alpha2, axial_chord, stagger          -- camberline
        le_thickness                                  -- leading edge
        ps_thickness, ss_thickness                    -- side thickness arrays
        camber_percent, expansion_ratio               -- shared by both sides
        te_radius, wedge_ss, wedge_ps                 -- trailing edge
        ss_flow_guidance: {s_c, n}                    -- SS throat straightener
        n_points                                      -- points per side

    Style handling (surface roles follow the turning direction):
        cup: counter-clockwise turning (alpha2 > alpha1), suction side lower
        cap: clockwise turning (alpha2 < alpha1), suction side upper
    pyturbo places its geometric "ss" on the lower side for left-to-right
    airfoils regardless of style, which would put CAP inputs on the wrong
    surface. CAP airfoils are therefore built in a mirrored CUP construction
    (negated alpha1/alpha2/stagger, y mirrored back) so every "ss_*" input
    always shapes the aerodynamic suction side. The returned arrays satisfy
    ss = suction side, ps = pressure side, outline wound clockwise.
    """
    from pyturbo.aero import Airfoil2D      # lazy: only the pyturbo path
    style = "cap" if (af_cfg["alpha2"] - af_cfg["alpha1"]) < 0 else "cup"
    sign = -1.0 if style == "cap" else 1.0
    af = Airfoil2D(
        alpha1=sign * af_cfg["alpha1"],
        alpha2=sign * af_cfg["alpha2"],
        axial_chord=af_cfg["axial_chord"],
        stagger=sign * af_cfg["stagger"],
        left_to_right=True,
    )
    af.add_le_thickness(af_cfg["le_thickness"])
    af.add_ps_thickness(
        thicknessArray=af_cfg["ps_thickness"],
        camberPercent=af_cfg.get("camber_percent", 0.95),
        expansion_ratio=af_cfg.get("expansion_ratio", 1.2),
    )
    af.add_ss_thickness(
        thicknessArray=af_cfg["ss_thickness"],
        camberPercent=af_cfg.get("camber_percent", 0.8),
        expansion_ratio=af_cfg.get("expansion_ratio", 1.2),
    )
    af.match_le_thickness()
    af.te_create(
        radius=af_cfg["te_radius"],
        wedge_ss=af_cfg.get("wedge_ss", 2.5),
        wedge_ps=af_cfg.get("wedge_ps", 2.4),
    )
    fg = af_cfg.get("ss_flow_guidance")
    if fg:
        af.add_ss_flow_guidance_2(s_c=fg["s_c"], n=fg["n"])
    af.center_le()

    n = int(af_cfg.get("n_points", 160))
    ss, ps = af.get_points(n)  # each (n(+TE arc), 2); row 0 = LE, row -1 = TE

    # normalize: axial chord = 1.0, LE at origin, blade centered vertically;
    # mirror CAP constructions back to the requested orientation
    scale = af_cfg["axial_chord"]
    ss = np.asarray(ss, dtype=float) / scale
    ps = np.asarray(ps, dtype=float) / scale
    if style == "cap":
        ss[:, 1] *= -1.0
        ps[:, 1] *= -1.0
    ss[0] = ps[0] = (ss[0] + ps[0]) / 2.0  # pin shared LE point
    ss[-1] = ps[-1] = (ss[-1] + ps[-1]) / 2.0  # pin shared TE point
    y_mid = 0.5 * (min(ss[:, 1].min(), ps[:, 1].min())
                   + max(ss[:, 1].max(), ps[:, 1].max()))
    ss[:, 1] -= y_mid
    ps[:, 1] -= y_mid

    # deduplicate consecutive points (splines reject zero-length segments)
    ss = _dedupe(ss)
    ps = _dedupe(ps)

    ss_upper = bool(ss[:, 1].mean() > ps[:, 1].mean())
    # closed outline wound clockwise: upper surface LE->TE, lower TE->LE --
    # the convention assumed by the wall hole loop, marker orientation and
    # wake placement
    upper, lower = (ss, ps) if ss_upper else (ps, ss)
    outline = np.vstack([upper, lower[::-1][1:-1]])

    return {
        "ss": ss,
        "ps": ps,
        "outline": outline,
        "style": style,
        "ss_upper": ss_upper,
        "airfoil2d": af,
    }


def _dedupe(points, eps=1e-12):
    keep = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - keep[-1]) > eps:
            keep.append(p)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= eps:
        keep.pop()
    return np.asarray(keep)
