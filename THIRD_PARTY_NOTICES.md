# Third-party software notices

Foilage's original source code is licensed under the GNU General Public
License, version 3 or later. This file records the licenses of software that
Foilage uses at runtime. These dependencies remain under their own licenses;
the Foilage license does not relicense them.

## Python runtime dependencies

The versions below correspond to `requirements.txt`:

| Dependency | Version | License | Upstream |
| --- | --- | --- | --- |
| Gmsh | 4.15.2 | GNU GPL v2 or later, with the Gmsh exception | [gmsh.info](https://gmsh.info/) |
| PyTurbo-Aero | 1.3.8 | NASA Open Source Agreement v1.3 | [NASA pyturbo-aero](https://github.com/nasa/pyturbo-aero) |
| NumPy | 2.5.3 | BSD 3-Clause and included notices | [numpy.org](https://numpy.org/) |
| SciPy | 1.18.1 | BSD 3-Clause and included notices | [scipy.org](https://scipy.org/) |
| Matplotlib | 3.11.2 | Matplotlib License (PSF-based) and included notices | [matplotlib.org](https://matplotlib.org/) |

Users who install dependencies with `uv` or `pip` receive the dependency
license files from the package distributions. If Foilage is later distributed
as a bundled application, installer, wheel, or archive containing these
dependencies, the distributor must include the applicable license texts,
copyright notices, and any source-code or written-offer materials required by
each dependency's license.

## SU2

SU2 is not included in the Foilage source distribution. The user downloads an
SU2 release separately and points `foilage.cfg` at that installation. SU2 is
available under the GNU Lesser General Public License, version 2.1. See the
[official SU2 download and license page](https://su2code.github.io/download.html).

The local `SU2-v8.5.0-win64-omp/` directory is ignored by Git and is retained
only as a developer-local installation. It must not be treated as part of the
Foilage GPL-covered source tree.

## `.geomTurbo` interoperability

Foilage includes an independently written reader and writer for a limited,
point-based subset of the `.geomTurbo` format. This is an interoperability
feature, not a distribution of NUMECA software, documentation, or sample
geometry files.

NUMECA, AutoGrid, and `.geomTurbo` are trademarks or proprietary technology
of their respective owners. Foilage is not affiliated with, endorsed by, or
sponsored by NUMECA or Cadence. This notice does not claim or grant any
license to NUMECA or Cadence intellectual property.

## Important redistribution note

If a future release bundles Gmsh, PyTurbo-Aero, SU2, or their binaries, update
this notice and perform a fresh license review before distributing that bundle.
