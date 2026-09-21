<p align="center">
  <img src="resources/Foilage_Logo.png" alt="Foilage logo" width="240">
</p>

<p align="center">A 2D computational-fluid-dynamics workflow from parametric geometry to checked, post-processed SU2 results.</p>

Foilage turns an airfoil or external-flow definition into a solver-ready CFD case. The main workflow builds a turbine-vane passage, generates a boundary-layer-aware mesh with Gmsh, configures SU2 from JSON, runs the solver with a live convergence monitor, and produces plots plus a machine-readable `results.json` summary.

The repository also contains a standalone `.geo` workflow for external-flow cases such as the circular-cylinder examples.

## At a glance

```text
input.json
    │
    ├─ pyturbo-aero      Generate and normalize the airfoil
    ├─ Gmsh              Build cascade/freestream domain, BL, wake, and mesh
    ├─ Blossom/quadify    Recombine and orient the mesh for SU2
    ├─ SU2 case setup     Scale units, map BCs, render turbine.cfg
    ├─ SU2_CFD            Solve while the Tk monitor tails history.csv
    └─ watchdog           Run plots and results.json after solver exit
```

## What it does

- Generates a left-to-right 2D turbine airfoil from the `airfoil` block in `input.json` and normalizes its axial chord to `1.0` mesh unit.
- Builds a periodic cascade passage from the domain radii, blade count, and axial limits.
- Applies Gmsh size fields for structured wall layers, near-wall refinement, and a trailing-edge wake band. The interior is recombined with Gmsh Blossom where possible.
- Preserves solver-facing physical markers: `airfoil`, `inlet`, `outlet`, plus either the periodic pair (`periodic_bottom`/`periodic_top`) or the isolated-airfoil `farfield` boundary.
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
python run_full_pipeline.py sample\turbine_sample\input.json --threads 6
```

The command performs three stages:

1. **Geometry and mesh** — `pipeline\run_pipeline.py` generates the airfoil, builds the periodic passage, creates the Gmsh mesh, recombines it, validates quality, and writes mesh images.
2. **SU2 case setup** — `tools\setup_cascade_case.py` reads the same input, scales the mesh to physical units, writes `case.json`, renders `turbine.cfg`, and runs a two-iteration periodic-pairing validation.
3. **Solve and post-process** — `tools\run_monitor.py` starts SU2 and the live monitor. A detached watchdog runs `tools\plot_case.py` when SU2 exits, regardless of whether the run finishes naturally or is stopped from the UI.

Useful options:

```powershell
# Build and validate the case, but do not start SU2.
python run_full_pipeline.py sample\turbine_sample\input.json --no-run

# Reuse an existing generated mesh and start at SU2 setup.
python run_full_pipeline.py sample\turbine_sample\input.json --skip-mesh

# Use a different output case name and thread count.
python run_full_pipeline.py sample\turbine_sample\input.json --name blade_trial --threads 8
```

If no input path is given, the runner looks for `input.json` at the repository root. In this checkout, the runnable examples are under `sample\turbine_sample*\input.json`, so passing the path explicitly is recommended.

## GUI workflow

The same workflow is available in a tabbed desktop GUI:

```powershell
python run_gui.py sample\turbine_sample\input.json
```

Without an argument the GUI loads the most recently modified `sample\turbine_sample\input.json`. The top bar loads, reloads, and saves `input.json`; unsaved changes are flagged and the run auto-saves first. The tabs, in order:

- **Geometry** - shows the normalized airfoil in its periodic passage (periodic edges and neighbouring blades included) and regenerates it live while you drag sliders or type exact values for the camberline, thickness, trailing-edge, flow-guidance, and domain parameters. Invalid combinations are reported instead of crashing. The blade comes from a selectable **airfoil source**: the pyturbo-aero generator, an imported NUMECA `.geomTurbo` file (choose the file, pick one of its blade sections, and that section is normalized to axial chord = 1 and used for meshing as-is; with pyturbo-aero active the imported section can be shown as a gray reference so the generated blade can be matched to it manually), or any **extension plugin** from the `extensions\` directory (see [Geometry generator extensions](#geometry-generator-extensions) below). The form panels follow the selected source: with pyturbo-aero you get the generator panels (camberline, thickness, trailing edge, suction-side flow guidance); with a geomTurbo import or a plugin those are hidden and replaced by the source-specific panels - file/section fields and the **FFD morph cage** for a geomTurbo import, the plugin's own parameter panels (plus the morph cage) for a plugin. The Discretisation and Domain (periodic passage) panels apply to every source and are always visible. The **Cascade periodicity** dropdown in the Domain panel selects how the passage is built (see [Periodicity modes](#periodicity-modes) below): axisymmetric periodics, offset periodics (linear cascade), or freestream boundaries for isolated-airfoil calculations. The tab also reports derived geometry: throat width and location, suction-side metal angle at the throat, true chord, and pitch-to-chord (PTC) - cascade metrics that are computed in the two periodic modes - plus two inspection charts - channel width vs x and surface curvature of both sides.

  **geomTurbo morph (FFD cage)** - the imported section can be reshaped with a free-form deformation lattice: **N_morph** selects the number of control points per direction (the cage is an N_morph x N_morph grid around the section's bounding box, +8% margin), and each cage point gets **dx** and **dy** offsets in axial-chord units, typed into a grid that mirrors the cage layout (u across, v upwards). The surface is displaced by the tensor-product Bernstein blend of the offsets, so equal offsets translate the blade, corner offsets rotate/shear it, symmetric offsets stretch it, and single points make local bumps; the shared LE/TE points of both surfaces always move together, keeping the section closed. Enable the morph to see the deformed cage (orange) over the live preview; the deformation is stored in `input.json` under `airfoil_source.morph` and is applied before meshing, so the mesh, the solver case, and any downstream runs use exactly the morphed shape. The same morph panel is offered for extension-plugin sources and is applied after the plugin's script runs.

### Geometry generator extensions

Custom geometry generators plug in through the `extensions\` directory - one folder per plugin, no code changes in Foilage needed. A plugin folder contains a `plugin.json` manifest and a CLI script. The manifest declares the dropdown label, a unique id (stored as `airfoil_source.type` in `input.json`), an optional description, and the parameters: each parameter becomes a form row (slider + exact entry, dropdown, checkbox, array editor - same widgets as the built-in panels) with label, unit, tooltip, default, and range, optionally grouped into several panels. Parameter values are stored under `airfoil_source.params.<name>` and ride along with every save, reload, and run.

Whenever the geometry is needed - live preview, mesh + solver run, or an optimization evaluation - Foilage runs the plugin's script as

```
python <cli> --config <config.json> --output <result.json>
```

(from the plugin folder; `config.json` carries the parameter values plus `n_points`, `axial_chord`, and the case name). The script writes `result.json` with suction/pressure-side polylines LE -> TE in any consistent unit (`{"ss": [[x, y], ...], "ps": [[x, y], ...], "blade_count": 45}`), which Foilage normalizes exactly like a `.geomTurbo` import and feeds into the normal meshing pipeline; results are cached per parameter set so the preview only re-runs the script when its inputs change. Failures (non-zero exit, missing output, timeout, invalid JSON) are reported in the Geometry status line together with the script's stderr. Plugin parameters with numeric `min`/`max` in the manifest are also offered automatically as **optimization design variables** in the Optimization tab. `extensions\README.md` documents the full manifest reference and `extensions\sample_naca\` is a complete working example (NACA 4-digit generator) to copy as a starting point.
- **Setup** - edits the `case`, `BCs`, and `solver_settings`/`numerics` blocks with widgets that know their allowed values: turbulence model (SA/SST), slope limiter, gradient method, and mesh algorithm are dropdowns; boundary conditions get sliders plus exact entry boxes; arrays get expandable editors; gamma is either set explicitly (air: 1.4) or left on *auto*, which computes it from the inlet total temperature using the temperature-dependent specific heats of air (NASA Glenn's calorically-imperfect model, Eggers NACA Report 959: 1.400 at 300 K, 1.364 at 700 K). A live panel derives quantities from the current values (effective gamma and its source, pitch, pressure ratio, isentropic exit Mach/speed/temperature, axial Reynolds number) and a validation panel flags problems such as `R1 != R2` or an outlet pressure above the inlet total pressure before any solver time is wasted. The **Run mesh + solver** button saves the input, then runs geometry + Gmsh meshing, SU2 case setup with periodic validation, and the SU2 solver, streaming all stage output into the pipeline log. *Mesh + case only* and *Solve only* rerun subsets, and *Stop* terminates the current stage (the Solution tab has its own Start/Stop buttons for solve-only runs). With **Initialize from previous solution** enabled, the solve warm-starts from `restart.dat` in the case folder (`RESTART_SOL= YES`) - the restart file is written by every solve, so run the case once before warm-starting. The case template pins SU2's restart filenames to `restart.dat` (`RESTART_FILENAME`/`SOLUTION_FILENAME` - SU2 v8 otherwise defaults them to `solution.dat`, which no Foilage solve writes), the validation step preserves `restart.dat` across full (mesh + setup + solve) runs, and a restart written for a different mesh is detected and reported before SU2 starts.
- **Mesh** - edits the `mesh` block (far-field/near-wall/periodic sizes, mesh algorithm, wall node count, boundary-layer stack, wake refinement) with the same slider + exact-entry widgets, and views the result interactively with zoom, pan, and home (matplotlib navigation toolbar) over any mesh found for the case: the Gmsh tri mesh, the Blossom-recombined quad mesh (both in axial-chord units), and the scaled solver mesh in meters. Boundary markers overlay in color with toggles, the side panel reports node/element/marker counts and the `mesh_quality.txt` quality gate, and **Generate mesh** reruns the meshing pipeline on the current input.
- **Solution** - while the solver runs, the canvas shows a 2x2 live view: residuals, total-pressure loss Yp (periodic modes) or force coefficients CL/CD (freestream mode), mass flow at inlet and outlet, and the domain imbalance (continuity, energy, momentum flux in %, computed from SU2's per-surface `Avg_*(inlet)`/`Avg_*(outlet)` history columns; continuity and energy converge to ~0, momentum settles at the blade axial force). The tab also has **Start run** (solve-only) and **Stop run** buttons of its own. The right viewer's **view** dropdown has a **convergence** entry that shows the same 2x2 grid on demand - rendered from the case's history.csv, live while the solver runs, and refreshable anytime afterwards. Once post-processing writes `results.json`, the tab populates automatically: metrics on the left, an interactive viewer on the right with three modes - **contour** (Mach, pressure, temperature, total pressure, total temperature, velocity magnitude, Cp, y+, skin friction magnitude, ... over the latest `vol_solution.vtk`, all in rainbow coloring with blue at the low end, drawn with the upper and lower periodic copies of the passage), **streamlines** (velocity streamlines over a regular grid with the blade interior masked, colored by any scalar field), and **1D surface** (isentropic Mach, skin friction, pressure coefficient, static pressure, and y+ as suction/pressure-side curves over the surface coordinate u, LE -> TE; the same data is exported to `ma_af.json` by post-processing) - filling the whole plot area, plus buttons for the generated PNGs. The left metrics panel prints the total-pressure loss Yp, the **Zweifel loading coefficients** (the classic incompressible criterion and the density-corrected compressible form, periodic cases only), the geometric Zweifel predictor from the mesh pipeline, throat/PTC geometry metrics, y+, and the mass-flow imbalance. **Import .vtk ...** loads any saved SU2 legacy volume file into the same viewer.
- **Optimization** - runs repeated full CFD evaluations while exploring selected geometry variables. Choose one or more `results.json` quantities as objectives, set each to minimize or maximize, add bounds on output quantities as constraints, and select **NSGA-II** or **Adaptive Hybrid Search** from the Search algorithm dropdown. NSGA-II is the default non-dominated genetic search. Adaptive Hybrid Search combines global differential-evolution proposals, genetic proposals, and local refinement while adapting the global/local balance as the search progresses. Both modes use the same constrained Pareto selection and live fitness/Pareto plots. The design-variable catalog follows the airfoil source: with pyturbo-aero it offers the generator parameters (metal angles, stagger, thickness, TE, flow guidance, plus `domain.R1` and blade count); with a geomTurbo import or an extension plugin it offers the plugin's declared parameters (bounded by their manifest min/max), the same domain variables, and the **FFD morph-cage offsets** - one variable per cage `dx`/`dy` with a symmetric bound prefilled from the *Cage offset bound* field, and the morph is enabled automatically for every evaluated case. Periodic cascades can additionally optimize or band the **Zweifel loading coefficients** (incompressible and density-corrected compressible) as objectives/constraints, with the geometric Zweifel predictor from the mesh pipeline available as a CFD-free constraint; all three are cascade-only and hidden for freestream cases.

### Optimization runs

The Optimization tab evaluates every candidate as a complete case: geometry generation, Gmsh meshing, SU2 setup, SU2 solve, and `results.json` post-processing. A single evaluation can therefore take as long as an ordinary CFD run. Start with a small **Population size**, use **Generation cap** when you need a fixed evaluation budget, and set **Stage timeout** to stop degenerate geometries or solver stages from holding the run indefinitely. **Convergence patience** stops a run after the accumulated feasible front's hypervolume has stopped improving; set a **Random seed** when you want repeatable sampling and search proposals.

The design-variable fields are `min`, `avg`, and `max`. The `avg` value is evaluated as the nominal individual in the initial population, while the remaining initial individuals sample the selected ranges. If `domain.R1` is explored, the optimizer keeps `domain.R2` equal to it so the SU2 periodic pair remains a rigid translation. Enable **Require residual target** when unconverged SU2 solutions should be penalized and excluded from the usable front.

The reported PTC is dimensionless and uses the local pitch at the detected throat: `PTC = (2*pi*R_throat / blade_count) / true_chord`. The true chord is the straight-line LE-to-TE distance, while the throat radius is obtained from the radius profile at the throat's axial position.

Multiple objectives produce a live Pareto-front plot; one objective produces a best-fitness history. While the run is in flight, the Optimization tab's **SU2 line convergence** page follows the case currently being evaluated: its rms residual lines plus the Yp (periodic modes) or CL/CD (freestream) convergence, read live from that eval's `history.csv`. Output is written under `runs\<case>_opt_<timestamp>\`, including `settings.json`, `evaluations.csv`, per-generation evaluation folders, and a log. The generated CFD cases are also kept under `cases\` with their own `input.json` copies so that promising designs can be inspected or rerun.

#### Pausing, stopping, and resuming an optimization

**Pause** holds the run as soon as the case currently being evaluated finishes; **Resume** continues it, keeping the full population, history, and search state in memory. Whenever a run pauses or finishes - including when it is stopped or hits an error - it writes `optimizer_state.json` into its run folder, and **Save state** requests the same snapshot on demand after the evaluation in flight. That file contains everything needed to continue later: objectives, senses, constraints, design-variable ranges, run options, the base case configuration, the complete evaluation history, the current population, any unevaluated proposals of a partially evaluated generation, the convergence counters, and the random-number generator state.

**Load state…** opens such a file: the form is filled in for review, the live plot shows the recorded history, and **Start optimization** continues the search from that exact point in a new run folder (`..._res<timestamp>`) whose `evaluations.csv` includes the carried-over history rows. Note two behaviors when resuming: the run uses the configuration stored in the state file (form edits are display-only until a fresh run is started), and the **Generation cap** is absolute - a resumed run keeps whatever budget remains of the original cap.

### Periodicity modes

The **Cascade periodicity** dropdown in the Geometry tab's Domain panel selects how the passage and its boundaries are built. It drives the geometry view, the meshed domain, and the SU2 boundary configuration together:
- **ML** - builds a surrogate model of the CFD: pick the inputs X (any design variables from the mode-dependent catalog - pyturbo parameters, domain values, FFD cage offsets, plugin parameters) and the outputs y (any results.json quantity), then create a dataset. Fill it with X points via Latin-hypercube sampling over each input's exploration range - every X column gets an editable `[min, max]` row (prefilled from the catalog, saved with the dataset, so the sampled design space can be narrowed when cases fail) - or a CSV import (rows that already carry y are kept), and **Fill missing Y** runs a complete CFD case per hole - sequentially in a worker thread, mirroring the optimization eval flow - recording results into a local SQLite database (`ml_data.db`) that accumulates across sessions. **Train MLP** fits a deep multilayer perceptron (pure numpy: standardized inputs/outputs, ReLU hidden layers, mini-batch Adam, validation split with early stopping and best-weights restore) to the filled rows, drawing the loss curve and a validation parity plot; the **Predict** panel evaluates the trained network for hand-entered X values.

- **Axisymmetric periodics (unwrapped annulus)** - the classic turbomachinery view and the default. The mesh plane is the unwrapped blade-to-blade surface, so the tangential coordinate is the arc-length "unwrapped y" (`uy = R*theta`); the pitch follows the annulus radius (`p = 2*pi*R/N`, R interpolated R1 -> R2 over the blade; keep R1 = R2 for SU2's single-translation periodic pair). The upper/lower mesh boundaries are the `periodic_top`/`periodic_bottom` pair.
- **Offset periodics (linear cascade)** - constant pitch (`p = 2*pi*R1/N`, R2 is ignored) with straight periodic lines through `y = -/+ p/2`: no tangential offset, y is Cartesian and nothing is unwrapped. Still a translation-periodic pair for SU2.
- **Freestream boundaries (isolated airfoil)** - no periodics at all. The upper/lower boundaries are straight far-field lines at `domain.y_min`/`domain.y_max` (axial-chord units), tagged `farfield` and rendered as SU2 `MARKER_FAR` (Riemann boundary condition) instead of a periodic pair - the mode for aircraft-airfoil calculations, where the lift-to-drag ratio `L/D = CL/CD` is available as an optimization objective or constraint. Cascade-only metrics (throat width, unguided turning, PTC, and the Zweifel loading coefficients - the classic incompressible `Zw = 2 (s/bx) cos^2(a2) (tan a1 - tan a2)` and its density-corrected compressible form `Zw_c = Zw * rho1/rho2`, both from the mass-averaged inlet/outlet plane audit -) are computed in the two periodic modes and are not computed in freestream mode.

Unwrapped-y bookkeeping: in **axisymmetric** mode the design and mesh plane uses `uy`, while `.geomTurbo` files (and plugin output) carry **Cartesian y** (`y = R*sin(theta)` around the machine axis). Sections are therefore unwrapped (`uy = R*asin(y/R)`, about the blade's tangential center, with R = R1 in axial-chord units) when loaded by any method, and re-wrapped (`y = R*sin(uy/R)`) when saved via *Save as geomTurbo...* - the round trip is lossless. In **offset** and **freestream** modes no conversion takes place (periodics are either offset-free or absent).


## How the pipeline works

### 1. Input-driven geometry and meshing

The parametric input is divided into a few responsibilities:

| Block | Role |
| --- | --- |
| `case` | Output name, output directory, and image behavior |
| `airfoil_source` | Geometry source: `pyturbo` (default) or `geomturbo` with `geomturbo_file` + `section` index |
| `airfoil` | Camberline angles, stagger, thickness, trailing edge, and point count |
| `domain` | Passage extent, annulus radii (actual units, e.g. mm), blade count, and periodic pitch |
| `mesh` | Far-field size, near-wall size, boundary-layer growth, and wake refinement |
| `BCs` | Total pressure/temperature/angle at the inlet and static pressure at the outlet |
| `solver_settings` / `numerics` | Optional turbulence model, gamma, iteration count, CFL, limiter, and gradient settings |

When `airfoil_source.type` is `geomturbo`, the named section replaces the
pyturbo-aero blade entirely (normalized to axial chord = 1) and the
`airfoil` block only supplies the discretization (`n_points`) and the
`axial_chord` used for physical scaling. The axial chord, blade count, and
LE/TE annulus radii (`domain.R1`/`R2`, mid hub/shroud radii at the LE/TE
stations) are inferred from the geomTurbo file and written into the input
automatically whenever the geomTurbo source is selected - the radii are
always specified in actual units (mm/m, like the axial chord) and are
normalized internally.

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
python tools\run_monitor.py sample\turbine_sample turbine.cfg -t 6
```

Residuals in `history.csv` are already log10 values. For steady cases, inspect residual decrease and stable force coefficients. For shedding flows or unsteady cases, use the statistically stable force mean rather than residual depth alone.

### 4. Post-processing and validation

After a run, `tools\plot_case.py` reads SU2's legacy VTK volume output and the case's post-processing hints. It writes:

- `convergence.png` — residuals and force coefficients;
- `fields.png` — Mach number and pressure coefficient;
- `nearwall.png` — near-wall flow and mesh view;
- `bl_validation.png` — boundary-layer profile and wall distributions;
- `ma_af.json` — suction/pressure-side surface distributions over the normalized surface coordinate `u` (0 at the LE, 1 at the TE): `ma` is the isentropic Mach number `Ma_is = sqrt(2/(gamma-1) * ((p01/p)^((gamma-1)/gamma) - 1))` built from the inlet total pressure and the local wall static pressure, and `cf` is the skin-friction coefficient magnitude. Schema: `{"ss": {"u": [...], "ma": [...], "cf": [...]}, "ps": {...}}`; and
- `results.json` — convergence status, forces, maximum Mach, wall `y+`, cascade mass-flow/loss audits when applicable, and geometry metrics including local throat pitch, true chord, and pitch-to-chord (PTC).

Run it manually when the solver has already finished:

```powershell
python tools\plot_case.py sample\turbine_sample
```

For interactive field inspection, open the generated `.vtu` files in ParaView. The legacy `.vtk` outputs are kept because the post-processing parser uses them reliably with SU2 v8 output.

## Standalone `.geo` external-flow cases

The cylinder samples use a separate self-contained Gmsh path. A `.geo` file owns the geometry, physical groups, mesh-size fields, boundary layer, and meshing algorithm. For example, `sample\cylinder_rans_sa\cylinder.geo` defines `inlet`, `outlet`, `freestream`, `cylinder`, and `fluid` groups.

Generate a solver mesh and the Gmsh inspection mesh with:

```powershell
python tools\mesh_from_geo.py `
    sample\cylinder_rans_sa\cylinder.geo `
    sample\cylinder_rans_sa\mesh
```

Render an SU2 configuration from a case description:

```powershell
python tools\render_config.py `
    sample\cylinder_rans_sa\case.json `
    templates\su2_steady_viscous.cfg `
    sample\cylinder_rans_sa\cylinder.cfg
```

Then use the same live monitor:

```powershell
python tools\run_monitor.py sample\cylinder_rans_sa cylinder.cfg -t 6
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
sample/                     Example inputs and generated local sample artifacts
resources/                  Project assets, including the Foilage logo
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

### `.geomTurbo` interoperability

Foilage provides independent interoperability with a limited point-based subset of the `.geomTurbo` format. Native files are read from their `suction` and `pressure` `SECTIONAL` blocks; the parser returns the periodic blade count plus each section's SS and PS point arrays, and the GUI can select a section for meshing. Export writes the same native point-block structure. Coordinates are normalized internally to canonical X/Y/Z order, while NUMECA's file-order `Z -Y X` rows are handled at the format boundary: the stored transverse coordinate is the negative of the Cartesian y and is flipped on read and re-applied on write. On export the spanwise `Z` of every point is computed from the annulus radius and the Cartesian y (`Z = sqrt(R^2 - y^2)`; in axisymmetric mode the section is re-wrapped from unwrapped-y to Cartesian first), so exported sections sit on the machine surface instead of a flat plane.

NUMECA, AutoGrid, and `.geomTurbo` are trademarks or proprietary technology of their respective owners. Foilage is not affiliated with, endorsed by, or sponsored by NUMECA or Cadence. No NUMECA software, documentation, or sample geometry files are distributed with Foilage.
