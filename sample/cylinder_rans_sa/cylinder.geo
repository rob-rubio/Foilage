// =============================================================================
// 2D channel with circular cylinder - RANS case (Re_D = 1e5, wall-resolved SA)
//
// Self-contained: geometry + size fields + physical groups. Mesh and export
// via:  python tools/mesh_from_geo.py cases/cylinder_rans_sa/cylinder.geo
//
// Physical groups -> SU2 markers:
//   inlet (left), outlet (right), freestream (top+bottom), cylinder (no-slip)
//
// Boundary-layer sizing (y+ ~ 1 wall-resolved):
//   y+ = rho * u_tau * y1 / mu  =>  y1 = nu / u_tau
//   with u_tau/U ~ 0.05 (typical cylinder wall shear):  y1 = 20*D/Re
//   Re = 1e5, D = 1  =>  y1 = 2e-4 D, 18 layers, ratio 1.25, to 0.04 D
// =============================================================================

// ----------------------------- parameters -----------------------------------
D    = 1.0;
R    = D/2.0;
XMIN = -5.0;  XMAX = 15.0;      // domain: 5D upstream, 15D downstream
YMIN = -3.0;  YMAX = 3.0;       // blockage D/H = 1/6

Re       = DefineNumber(1e5, Name "Re");
Y_PLUS   = 1.0;
UTAU_OU  = 0.05;                // u_tau/U_inf estimate
Y_FIRST  = Y_PLUS / (Re * UTAU_OU) * D;   // first-layer height
N_RATIO  = 1.25;                // BL growth ratio
BL_THICK = 0.04 * D;            // total BL thickness

H_FAR   = 0.40 * D;             // far-field size
H_WALL  = 0.010 * D;            // near-wall tangential/off-BL size
H_WAKE  = 0.020 * D;            // wake refinement size

// ----------------------------- geometry -------------------------------------
// Built-in (GEO) kernel: stable explicit tags, no booleans.
Point(1) = {XMIN, YMIN, 0, H_FAR};   // corners, CCW
Point(2) = {XMAX, YMIN, 0, H_FAR};
Point(3) = {XMAX, YMAX, 0, H_FAR};
Point(4) = {XMIN, YMAX, 0, H_FAR};
Point(5) = {0, 0, 0, 0};             // circle center (arc definition only)

Line(1) = {1, 2};                    // bottom  -> freestream
Line(2) = {2, 3};                    // right   -> outlet
Line(3) = {3, 4};                    // top     -> freestream
Line(4) = {4, 1};                    // left    -> inlet

Point(6) = {R, 0, 0, H_WALL};        // circle quadrisection points
Point(7) = {0, R, 0, H_WALL};
Point(8) = {-R, 0, 0, H_WALL};
Point(9) = {0, -R, 0, H_WALL};
// Arcs run CW (E -> S -> W -> N) so that the hole loop is clockwise and the
// fluid (outside the circle) lies on a consistent side of the boundary lines.
Circle(5) = {6, 5, 9};               // E -> S
Circle(6) = {9, 5, 8};               // S -> W
Circle(7) = {8, 5, 7};               // W -> N
Circle(8) = {7, 5, 6};               // N -> E

Curve Loop(1) = {1, 2, 3, 4};        // outer boundary (CCW)
Curve Loop(2) = {5, 6, 7, 8};        // hole (CCW)
Plane Surface(1) = {1, 2};           // loop 2 cuts the hole

// ------------------------- physical groups (SU2 markers) --------------------
Physical Curve("inlet")      = {4};
Physical Curve("outlet")     = {2};
Physical Curve("freestream") = {1, 3};
Physical Curve("cylinder")   = {5, 6, 7, 8};
Physical Surface("fluid")    = {1};

// --------------------------- size fields ------------------------------------
// 1: boundary layer on the cylinder (wall-resolved, quad layers)
Field[1] = BoundaryLayer;
Field[1].CurvesList = {5, 6, 7, 8};
Field[1].Size       = Y_FIRST;
Field[1].Ratio      = N_RATIO;
Field[1].Thickness  = BL_THICK;
Field[1].Quads      = 1;
BoundaryLayer Field = 1;

// 2+3: distance-to-wall threshold refinement around the cylinder
Field[2] = Distance;
Field[2].CurvesList = {5, 6, 7, 8};
Field[2].Sampling   = 400;

Field[3] = Threshold;
Field[3].InField = 2;
Field[3].SizeMin = H_WALL;
Field[3].SizeMax = H_FAR;
Field[3].DistMin = 0.30 * D;
Field[3].DistMax = 5.0 * D;

// 4: wake refinement
Field[4] = Box;
Field[4].VIn       = H_WAKE;
Field[4].VOut      = 1e22;
Field[4].XMin      = 0.4 * D;
Field[4].XMax      = 8.0 * D;
Field[4].YMin      = -1.5 * D;
Field[4].YMax      =  1.5 * D;
Field[4].Thickness = 1.0;

// 5: combine (min), fields are authoritative
Field[5] = Min;
Field[5].FieldsList = {3, 4};
Background Field    = 5;

Mesh.MeshSizeFromPoints          = 0;
Mesh.MeshSizeFromCurvature       = 0;
Mesh.MeshSizeExtendFromBoundary  = 0;
Mesh.MeshSizeMax                 = H_FAR;
Mesh.Algorithm                   = 6;   // Frontal-Delaunay
