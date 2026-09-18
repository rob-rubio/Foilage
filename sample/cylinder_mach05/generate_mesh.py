"""2D channel-with-cylinder mesh for SU2 (Mach 0.5 demo case).

Geometry (D = 1 cylinder diameter, cylinder at origin):
    domain : x in [-5, 15], y in [-3, 3]   (blockage D/H = 1/6)
    BCs    : inlet (left), outlet (right), freestream (top+bottom),
             cylinder (no-slip wall)

Mesh strategy (per hpc-gmsh skill):
    - OpenCASCADE kernel, BooleanDifference for the hole
    - BoundaryLayer field on the circle: 8 quad layers, first cell 0.004D
    - Distance+Threshold field around cylinder + wake Box field, Min-combined
    - Frontal-Delaunay triangles off the BL, exported to .su2 with markers

Outputs: mesh.su2 (solver), mesh.msh (GUI inspection)
"""

import gmsh

gmsh.initialize()

# ----------------------------- parameters ---------------------------------
R = 0.5                  # cylinder radius (D = 1)
X_MIN, X_MAX = -5.0, 15.0
Y_MIN, Y_MAX = -3.0, 3.0

H_FAR = 0.40             # far-field cell size
H_WALL = 0.015           # target size at the wall (tangential / off-BL)
H_WAKE = 0.03            # wake refinement size
BL_FIRST = 0.004         # first boundary-layer cell height
BL_RATIO = 1.3           # BL growth ratio
BL_THICK = 0.10          # total BL thickness (~0.1D, laminar Re regime)

MSH_FILE = "mesh.msh"
SU2_FILE = "mesh.su2"

# ----------------------------- geometry -----------------------------------
gmsh.model.add("cylinder_channel")
occ = gmsh.model.occ

rect = occ.addRectangle(X_MIN, Y_MIN, 0.0, X_MAX - X_MIN, Y_MAX - Y_MIN)
disk = occ.addDisk(0.0, 0.0, 0.0, R, R)
out, _ = occ.cut([(2, rect)], [(2, disk)])
occ.synchronize()

surf = out[0][1]
curves = [c[1] for c in gmsh.model.getBoundary([(2, surf)], oriented=False)]

# classify boundary curves by bounding box
inlet_c, outlet_c, far_c, cyl_c = [], [], [], []
for tag in curves:
    xmin, ymin, _, xmax, ymax, _ = occ.getBoundingBox(1, tag)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    ext_x, ext_y = xmax - xmin, ymax - ymin
    if ext_x < 2.0 * R * 1.001 and ext_y < 2.0 * R * 1.001:
        cyl_c.append(tag)
    elif abs(cx - X_MIN) < 1e-6:
        inlet_c.append(tag)
    elif abs(cx - X_MAX) < 1e-6:
        outlet_c.append(tag)
    else:
        far_c.append(tag)

assert len(cyl_c) == 1 and len(inlet_c) == 1 and len(outlet_c) == 1, \
    f"curve classification failed: cyl={cyl_c} in={inlet_c} out={outlet_c}"

# ------------------------- physical groups --------------------------------
gmsh.model.addPhysicalGroup(2, [surf], name="fluid")
gmsh.model.addPhysicalGroup(1, inlet_c, name="inlet")
gmsh.model.addPhysicalGroup(1, outlet_c, name="outlet")
gmsh.model.addPhysicalGroup(1, far_c, name="freestream")
gmsh.model.addPhysicalGroup(1, cyl_c, name="cylinder")

# --------------------------- size fields ----------------------------------
f_bl = 1
gmsh.model.mesh.field.add("BoundaryLayer", f_bl)
gmsh.model.mesh.field.setNumbers(f_bl, "CurvesList", cyl_c)
gmsh.model.mesh.field.setNumber(f_bl, "Size", BL_FIRST)
gmsh.model.mesh.field.setNumber(f_bl, "Ratio", BL_RATIO)
gmsh.model.mesh.field.setNumber(f_bl, "Thickness", BL_THICK)
gmsh.model.mesh.field.setNumber(f_bl, "Quads", 1)

f_dist, f_thr, f_box, f_min = 2, 3, 4, 5
gmsh.model.mesh.field.add("Distance", f_dist)
gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", cyl_c)
gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 400)

gmsh.model.mesh.field.add("Threshold", f_thr)
gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", H_WALL)
gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", H_FAR)
gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.30)
gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 5.0)

gmsh.model.mesh.field.add("Box", f_box)
gmsh.model.mesh.field.setNumber(f_box, "VIn", H_WAKE)
gmsh.model.mesh.field.setNumber(f_box, "VOut", 1e22)
gmsh.model.mesh.field.setNumber(f_box, "XMin", 0.4)
gmsh.model.mesh.field.setNumber(f_box, "XMax", 8.0)
gmsh.model.mesh.field.setNumber(f_box, "YMin", -1.5)
gmsh.model.mesh.field.setNumber(f_box, "YMax", 1.5)
gmsh.model.mesh.field.setNumber(f_box, "Thickness", 1.0)

gmsh.model.mesh.field.add("Min", f_min)
gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thr, f_box])
gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
gmsh.model.mesh.field.setAsBoundaryLayer(f_bl)

# fields are authoritative
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeMax", H_FAR)

gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

# ----------------------------- generate -----------------------------------
gmsh.model.mesh.generate(2)

# ----------------------------- statistics ---------------------------------
etypes, etags, _ = gmsh.model.mesh.getElements(2)
names = {2: "triangle", 3: "quad"}
print("--- mesh statistics ---")
total = 0
for t, tags_ in zip(etypes, etags):
    total += len(tags_)
    print(f"  {names.get(t, t)}: {len(tags_)}")
print(f"  total 2D elements: {total}")

quals = gmsh.model.mesh.getElementQualities(gmsh.model.mesh.getElementsByType(2)[0], "minSICN")
print(f"  tri min SICN: {min(quals):.4f}  mean SICN: {sum(quals)/len(quals):.4f}")

# ------------------------------ export ------------------------------------
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write(MSH_FILE)
gmsh.write(SU2_FILE)
print(f"wrote {MSH_FILE} and {SU2_FILE}")

gmsh.finalize()
