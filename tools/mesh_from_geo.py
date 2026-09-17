"""Generate a solver-ready mesh from a self-contained Gmsh .geo file.

The .geo file owns geometry, size fields, and physical groups. This driver
only: opens it, meshes it, prints statistics, and exports.

Usage:
    python mesh_from_geo.py <geometry.geo> [mesh_prefix]

Outputs: <prefix>.su2 (solver) and <prefix>.msh (Gmsh GUI inspection).
Default prefix = .geo file stem.
"""

import sys
from pathlib import Path

import gmsh

TYPE_NAMES = {1: "line", 2: "triangle", 3: "quad"}


def main():
    geo = Path(sys.argv[1])
    prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else geo.with_suffix("")
    if not geo.exists():
        sys.exit(f"not found: {geo}")

    gmsh.initialize()
    gmsh.open(str(geo))
    gmsh.model.mesh.generate(2)

    # --- statistics
    etypes, etags, _ = gmsh.model.mesh.getElements(2)
    print("--- mesh statistics ---")
    total = 0
    for t, tags in zip(etypes, etags):
        total += len(tags)
        print(f"  {TYPE_NAMES.get(t, t)}: {len(tags)}")
    print(f"  total 2D elements: {total}")

    tris = gmsh.model.mesh.getElementsByType(2)[0]
    if len(tris):
        q = gmsh.model.mesh.getElementQualities(tris, "minSICN")
        print(f"  tri min SICN: {min(q):.4f}  mean SICN: {sum(q)/len(q):.4f}")

    # --- export
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(str(prefix) + ".msh")
    gmsh.write(str(prefix) + ".su2")
    gmsh.finalize()

    # --- marker report from the .su2 file (validates physical groups)
    print("--- SU2 markers ---")
    name, n = None, None
    with open(str(prefix) + ".su2") as f:
        for line in f:
            if line.startswith("MARKER_TAG="):
                name = line.split("=", 1)[1].strip()
            elif line.startswith("MARKER_ELEMS=") and name:
                n = int(line.split("=", 1)[1])
                print(f"  {name}: {n} elements")
                name = None
    print(f"wrote {prefix}.su2 and {prefix}.msh")


if __name__ == "__main__":
    main()
