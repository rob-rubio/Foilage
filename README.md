<p align="center">
  <img src="resources/Foilage_Logo.png" alt="Foilage logo" width="240">
</p>

<h1 align="center">Foilage</h1>

<p align="center">A Windows-first 2D computational-fluid-dynamics workflow from parametric geometry to checked, post-processed SU2 results.</p>

Foilage turns an airfoil or external-flow definition into a solver-ready CFD case. The main workflow builds a turbine-vane passage, generates a boundary-layer-aware mesh with Gmsh, configures SU2 from JSON, runs the solver with a live convergence monitor, and produces plots plus a machine-readable `results.json` summary.

The repository also contains a standalone `.geo` workflow for external-flow cases such as the circular-cylinder examples.

## At a glance

```text
input.json
    │
    ├─ pyturbo-aero      Generate and normalize the airfoil
    ├─ Gmsh              Build periodic domain, BL, wake, and mesh
    ├─ Blossom/quadify    Recombine and orient the mesh for SU2
    ├─ SU2 case setup     Scale units, map BCs, render turbine.cfg
    ├─ SU2_CFD            Solve while the Tk monitor tails history.csv
    └─ watchdog           Run plots and results.json after solver exit
```

## What it does

- Generates a left-to-right 2D turbine airfoil from the `airfoil` block in `input.json` and normalizes its axial chord to `1.0` mesh unit.
- Builds a periodic cascade passage from the domain radii, blade count, and axial limits.
- Applies Gmsh size fields for structured wall layers, near-wall refinement, and a trailing-edge wake band. The interior is recombined with Gmsh Blossom where possible.
- Preserves solver-facing physical markers: `airfoil`, `inlet`, `outlet`, `periodic_bottom`, and `periodic_top`.
- Checks element quality, marker counts, and periodic node correspondence before the solver run.
- Converts the mesh into an SU2 case, scales normalized coordinates to meters, and derives inlet/outlet reference quantities from the requested boundary conditions.
- Renders an SU2 v8 configuration from `case.json` and a template instead of requiring manual edits to the generated `.cfg`.
- Launches SU2 in the background and displays residuals, force coefficients, mass flow, progress, and the recent solver log in a live Tk window.
- Post-processes legacy VTK output into convergence, field, near-wall, and boundary-layer plots, then writes convergence, force, flow, loss, mass-balance, and wall `y+` metrics to `results.json`.
- Ships a tabbed desktop GUI (`run_gui.py`) that edits `input.json` with smart widgets, previews the geometry interactively, views and generates meshes, runs the pipeline, and browses solutions - including importing saved `.vtk` files.

## Requirements

The complete one-command workflow currently targets Windows because it invokes `SU2_CFD.exe`, uses the Windows SU2 directory layout, and opens a Tk desktop monitor. SU2 is an external runtime dependency: Foilage does not ship the solver distribution.

- Python 3.12 or newer
- An SU2 installation containing `bin\SU2_CFD.exe`; official downloads are listed at <https://su2code.github.io/download.html>
- Python packages listed in `requirements.txt`: `numpy`, `scipy`, `matplotlib`, `gmsh`, and `pyturbo-aero`
- A desktop Python installation with Tk support for `tools\run_monitor.py`

Create the environment in a fresh checkout with either `uv`:

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
```

or standard `pip`:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Foilage does not include a Python virtual environment. Activate the environment before running the commands below, or use its interpreter explicitly.

Configure SU2 by copying the example and editing its path:

```powershell
Copy-Item foilage.cfg.example foilage.cfg
notepad foilage.cfg
```

The current developer checkout uses the ignored local directory `SU2-v8.5.0-win64-omp\`; end users should download SU2 separately and set `[su2] root` in `foilage.cfg` to their installation directory.

## Quick start: full turbine-cascade run

From the repository root, pass one of the available case inputs to the orchestrator:

```powershell
python run_full_pipeline.py cases\turbine_blade_4\input.json --threads 6
```

The command performs three stages:

1. **Geometry and mesh** — `pipeline\run_pipeline.py` generates the airfoil, builds the periodic passage, creates the Gmsh mesh, recombines it, validates quality, and writes mesh images.
2. **SU2 case setup** — `tools\setup_cascade_case.py` reads the same input, scales the mesh to physical units, writes `case.json`, renders `turbine.cfg`, and runs a two-iteration periodic-pairing validation.
3. **Solve and post-process** — `tools\run_monitor.py` starts SU2 and the live monitor. A detached watchdog runs `tools\plot_case.py` when SU2 exits, regardless of whether the run finishes naturally or is stopped from the UI.

Useful options:

```powershell
# Build and validate the case, but do not start SU2.
python run_full_pipeline.py cases\turbine_blade_4\input.json --no-run

# Reuse an existing generated mesh and start at SU2 setup.
python run_full_pipeline.py cases\turbine_blade_4\input.json --skip-mesh

# Use a different output case name and thread count.
python run_full_pipeline.py cases\turbine_blade_4\input.json --name blade_trial --threads 8
```

If no input path is given, the runner looks for `input.json` at the repository root. In this checkout, the runnable examples are under `cases\turbine_blade_*\input.json`, so passing the path explicitly is recommended.

## GUI workflow

The same workflow is available in a tabbed desktop GUI:

```powershell
python run_gui.py cases\turbine_blade_4\input.json
```

Without an argument the GUI loads the most recently modified `cases\*\input.json`. The top bar loads, reloads, and saves `input.json`; unsaved changes are flagged and the run auto-saves first. The tabs, in order:

- **Geometry** - shows the normalized airfoil in its periodic passage (periodic edges and neighbouring blades included) and regenerates it live while you drag sliders or type exact values for the camberline, thickness, trailing-edge, flow-guidance, and domain parameters. Invalid combinations are reported instead of crashing.
- **Setup** - edits the `case`, `BCs`, and `solver_settings`/`numerics` blocks with widgets that know their allowed values: turbulence model (SA/SST), slope limiter, gradient method, and mesh algorithm are dropdowns; boundary conditions get sliders plus exact entry boxes; arrays get expandable editors; gamma is either set explicitly (air: 1.4) or left on *auto*, which computes it from the inlet total temperature using the temperature-dependent specific heats of air (NASA Glenn's calorically-imperfect model, Eggers NACA Report 959: 1.400 at 300 K, 1.364 at 700 K). A live panel derives quantities from the current values (effective gamma and its source, pitch, pressure ratio, isentropic exit Mach/speed/temperature, axial Reynolds number) and a validation panel flags problems such as `R1 != R2` or an outlet pressure above the inlet total pressure before any solver time is wasted. The **Run mesh + solver** button saves the input, then runs geometry + Gmsh meshing, SU2 case setup with periodic validation, and the SU2 solver, streaming all stage output into the pipeline log. *Mesh + case only* and *Solve only* rerun subsets, and *Stop* terminates the current stage.
- **Mesh** - edits the `mesh` block (far-field/near-wall/periodic sizes, mesh algorithm, wall node count, boundary-layer stack, wake refinement) with the same slider + exact-entry widgets, and views the result interactively with zoom, pan, and home (matplotlib navigation toolbar) over any mesh found for the case: the Gmsh tri mesh, the Blossom-recombined quad mesh (both in axial-chord units), and the scaled solver mesh in meters. Boundary markers overlay in color with toggles, the side panel reports node/element/marker counts and the `mesh_quality.txt` quality gate, and **Generate mesh** reruns the meshing pipeline on the current input.
- **Solution** - while the solver runs, the canvas shows a 2x2 live view: residuals, force coefficients, mass flow at inlet and outlet, and the domain imbalance (continuity, energy, momentum flux in %, computed from SU2's per-surface `Avg_*(inlet)`/`Avg_*(outlet)` history columns; continuity and energy converge to ~0, momentum settles at the blade axial force). Once post-processing writes `results.json`, the tab populates automatically: metrics on the left, an interactive viewer on the right with two modes - **contour** (Mach, pressure, temperature, total pressure, total temperature, velocity magnitude, Cp, y+, skin friction magnitude, ... over the latest `vol_solution.vtk`, all in rainbow coloring with blue at the low end) and **1D surface** (isentropic Mach, skin friction, pressure coefficient, static pressure, and y+ as suction/pressure-side curves over the surface coordinate u, LE -> TE; the same data is exported to `ma_af.json` by post-processing) - filling the whole plot area, plus buttons for the generated PNGs. **Import .vtk ...** loads any saved SU2 legacy volume file into the same viewer.

Developer checks that require no interaction:

```powershell
python run_gui.py --selftest   # schema/state/mesh/post-processing logic tests
python run_gui.py --smoke      # builds the window, exercises every tab, exits
```


## How the pipeline works

### 1. Input-driven geometry and meshing

The parametric input is divided into a few responsibilities:

| Block | Role |
| --- | --- |
| `case` | Output name, output directory, and image behavior |
| `airfoil` | Camberline angles, stagger, thickness, trailing edge, and point count |
| `domain` | Passage extent, annulus radii, blade count, and periodic pitch |
| `mesh` | Far-field size, near-wall size, boundary-layer growth, and wake refinement |
| `BCs` | Total pressure/temperature/angle at the inlet and static pressure at the outlet |
| `solver_settings` / `numerics` | Optional turbulence model, gamma, iteration count, CFL, limiter, and gradient settings |

`pipeline\airfoil.py` uses `pyturbo-aero` with left-to-right flow orientation, handles CUP/CAP turning conventions, pins the shared leading/trailing-edge points, and returns suction-side, pressure-side, and closed-outline coordinates.

`pipeline\mesh_tris.py` then:

- creates a passage whose upper and lower edges share the same axial stations;
- tags the fluid and all SU2 boundary markers as Gmsh physical groups;
- adds a Gmsh `BoundaryLayer` field on the airfoil, a distance/threshold field near the wall, and a directed wake refinement field;
- combines sizing fields with a `Min` field and generates a 2D Frontal-Delaunay mesh;
- verifies the periodic edge node correspondence; and
- writes the initial `.msh`, `.su2`, and `.obj` files.

`pipeline\quadify.py` extracts the recombined quads, retains any leftover triangles, fixes element orientation, re-orients marker edges to SU2's 2D convention, and writes the final solver mesh.

`airfoil.n_points` sets how finely the pyturbo-aero splines sample the geometry. The number of mesh nodes along the wall is normally derived by Gmsh from the size fields; set `mesh.airfoil_points` (GUI: Mesh tab, "Nodes per surface side") to force an exact node count on each surface.

The airfoil is normalized to one axial-chord unit during meshing. The setup stage later converts it to meters. By default, an `axial_chord` of `100.0` means `100 mm`, while a value below `1.0` is interpreted as meters; `tools\setup_cascade_case.py --scale` overrides this rule.

### 2. SU2 case generation

The setup stage writes the solver case into `cases\<name>\`:

- copies and scales `mesh_quad.su2` (or `mesh.su2` if no quad mesh exists);
- translates the input BCs into a total-condition inlet and static-pressure outlet;
- computes the periodic translation from the passage pitch;
- estimates reference exit state, Reynolds number, and post-processing probe locations;
- writes `case.json`; and
- renders `turbine.cfg` from `templates\su2_cascade.cfg`.

The mesh marker names and SU2 marker names are a strict contract:

| Mesh marker | SU2 meaning |
| --- | --- |
| `airfoil` | Adiabatic wall and monitored surface |
| `inlet` | Total temperature, total pressure, and flow direction |
| `outlet` | Static-pressure outlet |
| `periodic_bottom`, `periodic_top` | Translational periodic pair |
| `fluid` | Gmsh 2D region group; not a boundary condition |

The inlet and outlet are both added to SU2's `MARKER_ANALYZE`, which makes the solver write per-surface `Avg_*(inlet)`/`Avg_*(outlet)` history columns; the GUI's live mass-flow and imbalance plots read them.

For SU2 periodicity to be a single rigid translation, the input requires `R1 == R2`. A varying-pitch unwrapped annulus is rejected because its two periodic edges cannot be paired by one constant translation.

### 3. Solve and monitor

The monitor runs SU2 with the requested thread count and incrementally reads `history.csv`. It displays:

- log10 residual histories;
- `CD` and `CL`, including a running mean for unsteady cases;
- mass flow per unit depth when SU2 writes it;
- iteration progress and ETA; and
- the tail of `su2_run.log`.

The detached watchdog waits on the SU2 process handle, then invokes `tools\plot_case.py` once. A lock file prevents duplicate post-processing when more than one process notices the solver exit.

To run an already-prepared case directly:

```powershell
python tools\run_monitor.py cases\turbine_blade_4 turbine.cfg -t 6
```

Residuals in `history.csv` are already log10 values. For steady cases, inspect residual decrease and stable force coefficients. For shedding flows or unsteady cases, use the statistically stable force mean rather than residual depth alone.

### 4. Post-processing and validation

After a run, `tools\plot_case.py` reads SU2's legacy VTK volume output and the case's post-processing hints. It writes:

- `convergence.png` — residuals and force coefficients;
- `fields.png` — Mach number and pressure coefficient;
- `nearwall.png` — near-wall flow and mesh view;
- `bl_validation.png` — boundary-layer profile and wall distributions;
- `ma_af.json` — suction/pressure-side surface distributions over the normalized surface coordinate `u` (0 at the LE, 1 at the TE): `ma` is the isentropic Mach number `Ma_is = sqrt(2/(gamma-1) * ((p01/p)^((gamma-1)/gamma) - 1))` built from the inlet total pressure and the local wall static pressure, and `cf` is the skin-friction coefficient magnitude. Schema: `{"ss": {"u": [...], "ma": [...], "cf": [...]}, "ps": {...}}`; and
- `results.json` — convergence status, forces, maximum Mach, wall `y+`, and cascade mass-flow/loss audits when applicable.

Run it manually when the solver has already finished:

```powershell
python tools\plot_case.py cases\turbine_blade_4
```

For interactive field inspection, open the generated `.vtu` files in ParaView. The legacy `.vtk` outputs are kept because the post-processing parser uses them reliably with SU2 v8 output.

## Standalone `.geo` external-flow cases

The cylinder cases use a separate self-contained Gmsh path. A `.geo` file owns the geometry, physical groups, mesh-size fields, boundary layer, and meshing algorithm. For example, `cases\cylinder_rans_sa\cylinder.geo` defines `inlet`, `outlet`, `freestream`, `cylinder`, and `fluid` groups.

Generate a solver mesh and the Gmsh inspection mesh with:

```powershell
python tools\mesh_from_geo.py `
    cases\cylinder_rans_sa\cylinder.geo `
    cases\cylinder_rans_sa\mesh
```

Render an SU2 configuration from a case description:

```powershell
python tools\render_config.py `
    cases\cylinder_rans_sa\case.json `
    templates\su2_steady_viscous.cfg `
    cases\cylinder_rans_sa\cylinder.cfg
```

Then use the same live monitor:

```powershell
python tools\run_monitor.py cases\cylinder_rans_sa cylinder.cfg -t 6
```

This path is useful when geometry is easier to express directly in Gmsh GEO syntax, especially for domains with explicit curves, holes, physical groups, and wall-resolved boundary layers.

## Repository layout

```text
run_gui.py                  Launch the tabbed GUI (setup/geometry/mesh/solution)
run_full_pipeline.py        One-command geometry -> mesh -> SU2 -> results driver
pipeline/                   Parametric airfoil and periodic cascade mesh pipeline
tools/                      Case setup, SU2 rendering, monitoring, VTK parsing, plots
foilage_gui/                GUI package: schema-driven forms, tabs, runner, viewers
templates/                  SU2 configuration templates
cases/                      Example inputs and generated local case artifacts
resources/                  Project assets, including the Foilage logo
future_features.md          Agreed, not-yet-implemented feature designs
foilage.cfg.example         Template for the local SU2 path
foilage_config.py           Runtime configuration and SU2 resolution
requirements.txt            Python runtime dependencies
THIRD_PARTY_NOTICES.md      Dependency license and attribution notices
```

## Common failure checks

- **SU2 not found:** confirm `[su2] root` in `foilage.cfg` points to an installation containing `bin\SU2_CFD.exe`.
- **Missing Python module:** activate the environment created from `requirements.txt` and reinstall the listed packages.
- **Periodic validation fails:** check that `domain.R1` equals `domain.R2` and that the marker names are unchanged.
- **No wall-layer quads:** verify the Gmsh boundary-layer field is enabled and that the generated mesh passed `mesh_quality.txt`.
- **Unexpected forces or wall fluxes:** check marker orientation. SU2 expects the fluid to remain on the left side of each oriented boundary edge.
- **No post-processing plots:** confirm SU2 wrote `vol_solution.vtk` and that `history.csv` is newer than any existing `results.json`.

## License and third-party software

Foilage's original source code is licensed under the GNU General Public License, version 3 or later. See [`LICENSE`](LICENSE).
Copyright (C) 2026 Foilage contributors.

Gmsh, PyTurbo-Aero, NumPy, SciPy, Matplotlib, and SU2 remain under their own licenses. Foilage does not include the SU2 distribution or a Python virtual environment. Runtime dependency versions, upstream links, license information, and redistribution notes are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

