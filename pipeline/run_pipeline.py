"""Airfoil -> Gmsh triangles -> Blossom-recombined quads, driven by input.json.

Usage:
    python pipeline\\run_pipeline.py [input.json]

Stages:
    1. airfoil   : pyturbo-aero Airfoil2D, left_to_right=True, axial chord = 1
    2. tri mesh  : periodic cascade domain (medial-axis edges) + BL fields
                   -> .msh/.su2/.obj
    3. quad mesh : Gmsh Blossom output, repair + markers -> _quad.su2/_quad.msh
    4. quality   : SICN / angle / inverted-element report -> mesh_quality.txt
    5. pictures  : airfoil, tri mesh, quad mesh (full + zoom)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from airfoil import build_geometry
from cascade_metrics import (metal_angles, throat_metrics, true_chord,
                             zweifel_geometric)
from mesh_tris import mesh_domain, pitch_profile
from plots import plot_airfoil, plot_tri_mesh, plot_quad_mesh
import quadify

ROOT = Path(__file__).resolve().parent.parent


def quality_report(case_dir, prefix):
    """Compute quality statistics for the quad (and tri) meshes with gmsh.

    Writes mesh_quality.txt and enforces: no negative-SICN elements in the
    quad mesh. Returns the report text.
    """
    import gmsh

    lines = ["mesh quality report", "=" * 60]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)

        quad_msh = str(prefix) + "_quad.msh"
        gmsh.open(quad_msh)
        etypes, _, _ = gmsh.model.mesh.getElements(2)
        lines.append(f"quad mesh ({Path(quad_msh).name}):")
        for t, name in [(3, "quad"), (2, "triangle")]:
            if t not in etypes:
                continue
            elems = gmsh.model.mesh.getElementsByType(t)[0]
            sicn = np.asarray(gmsh.model.mesh.getElementQualities(elems, "minSICN"))
            neg = int((sicn <= 0).sum())
            lines.append(
                f"  {name:9s} count {len(elems):7d}   SICN min {sicn.min():8.5f}   "
                f"mean {sicn.mean():7.4f}   negative {neg}"
            )
            if neg > 0:
                lines.append(f"  FAIL: {neg} negative {name} elements")
        n1 = len(gmsh.model.mesh.getElementsByType(1)[0])
        lines.append(f"  line      count {n1:7d}")
        gmsh.finalize()

        tri_msh = str(prefix) + ".msh"
        gmsh.initialize()
        gmsh.open(tri_msh)
        tris = gmsh.model.mesh.getElementsByType(2)[0]
        lines.append(f"triangle mesh ({Path(tri_msh).name}):")
        if len(tris):
            sicn = np.asarray(
                gmsh.model.mesh.getElementQualities(tris, "minSICN"))
            lines.append(
                f"  triangle  count {len(tris):7d}   SICN min {sicn.min():8.5f}   "
                f"mean {sicn.mean():7.4f}   negative {int((sicn <= 0).sum())}"
            )
        else:
            lines.append("  triangle  count       0   (fully recombined)")
        gmsh.finalize()
    finally:
        pass

    text = "\n".join(lines) + "\n"
    (Path(case_dir) / "mesh_quality.txt").write_text(text)
    return text


def resolve_config():
    """Config lookup: explicit argument (json file, or a directory holding
    input.json) > ./input.json in the current working directory > the repo's
    own input.json. The case output directory is resolved relative to the
    config file's location, so a local input.json keeps all outputs local.
    """
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        p = p if p.is_absolute() else Path.cwd() / p
        p = p.resolve()
        if p.is_dir():
            p = p / "input.json"
        return p
    local = Path.cwd() / "input.json"
    if local.exists():
        return local
    return ROOT / "input.json"


def main():
    cfg_path = resolve_config()
    if not cfg_path.exists():
        sys.exit(f"config not found: {cfg_path}\n"
                 "run from a directory containing input.json, or pass the "
                 "path to one")
    cfg = json.loads(cfg_path.read_text())
    base_dir = cfg_path.parent
    print(f"config: {cfg_path}")

    case_name = cfg.get("case", {}).get("name", "case")
    out_dir = Path(cfg.get("case", {}).get("output_dir", "cases"))
    case_dir = (out_dir if out_dir.is_absolute() else base_dir / out_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    prefix = case_dir / "mesh"

    t_all = time.time()

    src = (cfg.get("airfoil_source") or {}).get("type", "pyturbo")
    print(f"[1/5] airfoil (source: {src}) ...")
    airfoil = build_geometry(cfg)
    plot_airfoil(airfoil, case_dir / "airfoil_geometry.png")
    print(
        f"      outline points: {len(airfoil['outline'])}, "
        f"x range [{airfoil['outline'][:,0].min():.3f}, "
        f"{airfoil['outline'][:,0].max():.3f}]"
    )

    dom_cfg = cfg.get("domain", {})
    prof = pitch_profile(dom_cfg, airfoil)
    mode = prof.get("mode", "axisymmetric")
    if mode == "freestream":
        print("      freestream domain: no periodics, far-field "
              f"boundaries at y = {prof.get('y_min', -1.5):.3f} / "
              f"{prof.get('y_max', 1.5):.3f}")
    else:
        print(
            f"      periodic passage ({mode}): N={dom_cfg['airfoil_count']} "
            f"R1={dom_cfg['R1']} R2={dom_cfg['R2']} -> "
            f"pitch_le={prof['p_le']:.4f}, pitch_te={prof['p_te']:.4f}"
        )

    # 0D geometric metrics (axial-chord units) - copied into the SU2 case
    # by setup_cascade_case.py and merged into results.json by plot_case.py
    metrics = {
        "periodicity": mode,
        "pitch_le_cax": float(prof["p_le"]),
        "pitch_te_cax": float(prof["p_te"]),
        "true_chord_cax": true_chord(airfoil["ss"], airfoil["ps"]),
    }
    if mode != "freestream":
        # cascade-only: the throat lives in the channel to the
        # pitch-translated neighbor, which a freestream domain has none of
        throat = throat_metrics(airfoil["ss"], airfoil["ps"], prof["p"],
                                airfoil["ss_upper"])
        # geometric Zweifel predictor: metal angles from the section
        # tangents stand in for the flow angles of the classic criterion
        a1g, a2g = metal_angles(airfoil["ss"], airfoil["ps"])
        metrics.update({
            "throat_width_cax": throat["width"],
            "throat_x_over_cax": throat["x_over_cax"],
            "angle_throat_deg": throat["angle_throat_deg"],
            "angle_exit_deg": throat["angle_exit_deg"],
            "unguided_turning_deg": throat["unguided_turning_deg"],
            "radius_throat_cax": float(prof["radius"](throat["x_over_cax"])),
            "pitch_throat_cax": float(throat["pitch_throat"]),
            "pitch_to_chord": float(throat["pitch_throat"] /
                                    max(metrics["true_chord_cax"], 1e-30)),
            "angle_metal_inlet_deg": a1g,
            "angle_metal_exit_deg": a2g,
            "zweifel_geometric": zweifel_geometric(
                float(throat["pitch_throat"]), a1g, a2g),
        })
        print(
            f"      throat: width {throat['width']:.4f} c_ax at "
            f"x/c_ax = {throat['x_over_cax']:.3f}, unguided turning "
            f"{throat['unguided_turning_deg']:.1f} deg, geometric Zweifel "
            f"{metrics['zweifel_geometric']:.3f}"
        )
    (case_dir / "geometry_metrics.json").write_text(json.dumps(
        metrics, indent=2))

    wake_cfg = cfg["mesh"].get("wake") or {}
    print(
        f"      wake: TE metal angle "
        f"{(wake_cfg.get('metal_angle_deg') if wake_cfg.get('metal_angle_deg') is not None else 'auto')}"
    )
    tri_obj = Path(str(prefix) + ".obj")

    print("[2/5] gmsh mesh: structured BL quads + Blossom-recombined "
          "interior (density preserved) ...")
    stats = mesh_domain(airfoil, dom_cfg, cfg["mesh"], prefix,
                        bl_quads=True, recombine=True)
    print(f"      elements: {stats['counts']}, total {stats['total']}")
    if "min_sicn" in stats:
        print(f"      quality: min SICN {stats['min_sicn']:.3f}, "
              f"mean {stats['mean_sicn']:.3f}")

    print("[3/5] extract quad mesh (BL spacing preserved) ...")
    qstats = quadify.extract_gmsh_recombined(prefix)
    quad_obj = Path(str(prefix) + "_quad.obj")
    print(f"      quads: {qstats['quads']}, leftover tris: "
          f"{qstats['leftover_tris']}")
    print(f"      min quad corner angle: "
          f"{qstats['min_corner_angle_deg']:.2f} deg")
    print(
        f"      periodic: exact 1:1 node pairing on the periodic edges, "
        f"mismatch {stats.get('periodic_tri_max_mismatch', 0.0):.2e}"
        if stats.get("periodic_tri_max_mismatch") is not None else
        "      freestream: no periodic edges (far-field top/bottom)"
    )

    print(f"      markers: {qstats['markers']}")
    if stats.get("wake_metal_angle_deg") is not None:
        print(f"      wake band oriented at TE metal angle: "
              f"{stats['wake_metal_angle_deg']:.2f} deg")
    print("[4/5] quality report ...")
    report = quality_report(case_dir, prefix)
    print("\n".join("      " + ln for ln in report.splitlines()[2:]))
    if "FAIL" in report:
        sys.exit("mesh quality gate failed: negative elements present")

    print("[5/5] pictures ...")
    zoom = cfg.get("plot", {}).get("zoom_window")
    plot_tri_mesh(tri_obj, case_dir / "mesh_tris_full.png")
    plot_tri_mesh(tri_obj, case_dir / "mesh_tris_zoom.png", zoom)
    label = "structured BL + Gmsh Blossom recombine"
    plot_quad_mesh(quad_obj, case_dir / "mesh_quad_full.png", engine=label)
    plot_quad_mesh(quad_obj, case_dir / "mesh_quad_zoom.png", zoom,
                   engine=label)

    print(f"done in {time.time() - t_all:.1f}s -> {case_dir}")
    if cfg.get("case", {}).get("show_images", True) and os.name == "nt":
        for png in ("airfoil_geometry.png", "mesh_tris_full.png",
                    "mesh_tris_zoom.png", "mesh_quad_full.png",
                    "mesh_quad_zoom.png"):
            # os.startfile(str(case_dir / png))
            pass


if __name__ == "__main__":
    main()
