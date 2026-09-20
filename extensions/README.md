# Geometry generator extensions

Every subdirectory of `extensions/` that contains a `plugin.json`
manifest becomes an airfoil source: it appears automatically in the
Geometry tab's **Geometry source** dropdown, gets its own parameter
panels (sliders, dropdowns, checkboxes, arrays - the same widgets the
built-in forms use), and its CLI script is run automatically whenever
Foilage needs the geometry (live preview, meshing, solver runs).

Copy `sample_naca/` to start a new plugin; it is a complete, working
example.

## Plugin layout

```
extensions/
  my_generator/
    plugin.json        <- manifest (required)
    generate.py        <- your CLI script (any language runnable via
                          the interpreter; Python is used by default)
    ...                <- any helper files your script needs
```

## Manifest reference (`plugin.json`)

| key | type | meaning |
| --- | ---- | ------- |
| `id` | string | Source value stored in `input.json` (default: folder name, lowercased). Must not be `pyturbo` or `geomturbo`; must be unique. |
| `name` | string | Label shown in the Geometry source dropdown and as the default panel title. |
| `description` | string | Shown as gray helper text under the parameter panels. |
| `cli` | string | **Required.** Script file inside this folder that generates the geometry. |
| `timeout_s` | number | Kill the script after this many seconds (default 120). |
| `parameters` | list | Parameter definitions - one entry per form row (see below). |
| `groups` | list | Optional panel grouping: `[{"title": "Camber", "params": ["camber", ...]}]`. Without it one panel titled with `name` holds all parameters. |

Each **parameter** is one form row. Keys: `path` (required, stored under
`airfoil_source.params.<path>` in `input.json`), `label`, `kind`
(`float`, `int`, `bool`, `str`, `choice`, `float_array`,
`nullable_float`, `nullable_int`, `file`), `default`, `min`, `max`,
`step`, `slider` (pair the entry with a slider), `unit`, `tooltip`,
`choices` (`[["label", value], ...]`, for `kind: "choice"`),
`array_fixed`.

## CLI contract

Foilage runs, with the plugin folder as the working directory:

```
<python> <cli> --config <config.json> --output <result.json>
```

`config.json` contains:

```json
{
  "parameters": {"max_camber": 0.02, "...": "..."},
  "n_points": 401,
  "axial_chord": 100.0,
  "case_name": "my_case"
}
```

Your script must write `result.json`:

```json
{
  "ss": [[0.0, 0.0], [0.01, 0.012], "... LE -> TE"],
  "ps": [[0.0, 0.0], [0.01, -0.010], "... LE -> TE"],
  "blade_count": 45
}
```

- `ss` / `ps` are the suction- and pressure-side polylines, LE -> TE, in
  **any consistent unit** - Foilage normalizes the section to axial
  chord = 1 and centers it, exactly like a `.geomTurbo` import.
- `blade_count` is optional; when given it fills the Domain panel's
  blade count on import.
- Progress/debug output goes to stderr and is shown in the GUI status if
  the script fails (non-zero exit, missing output, timeout, or invalid
  JSON).

The generated section can additionally be reshaped with the **FFD morph
cage** panel (the morph is applied after your script runs). Plugin
parameters declared with numeric `min`/`max` and `float`/`int` kind are
also offered automatically in the **Optimization tab** as design
variables, so the optimizer can explore your generator's inputs (each
evaluated case re-runs your CLI with the optimizer's parameter values).

## Testing a plugin

Run the script by hand to iterate quickly:

```
echo {"parameters": {"max_camber": 0.04}} > config.json
python generate.py --config config.json --output result.json
```

or just edit the parameters in the Geometry tab - the preview (and its
status line) shows the result after every change. A plugin whose
manifest is broken does not appear in the dropdown; the Geometry status
line reports how many extensions were skipped and why.
