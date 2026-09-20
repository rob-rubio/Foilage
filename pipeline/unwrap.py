"""Cascade periodicity modes and the Cartesian-y <-> unwrapped-y mapping.

``domain.periodicity`` selects how the passage is built:

- ``axisymmetric``  - the annular cascade: the mesh plane is the unwrapped
  blade-to-blade surface, so the tangential coordinate shown everywhere is
  the arc-length ``uy = R * theta`` ("unwrapped y"); the pitch follows the
  annulus radius (p = 2 pi R / N). Imported/exported geomTurbo sections and
  plugin sections are in *Cartesian* y (y = R * sin(theta) around the
  machine axis), so they are unwrapped on import and re-wrapped on export
  - see cartesian_to_uy / uy_to_cartesian.
- ``offset``        - a linear cascade: constant pitch, straight periodic
  lines with no tangential offset; y is Cartesian, nothing is unwrapped.
- ``freestream``    - no periodics at all: the upper/lower domain
  boundaries are far-field lines (isolated airfoil calculations).

All conversions operate on the normalized design arrays (axial chord = 1,
blade centered at y = 0) with ``radius`` the annulus radius in the same
units (radius / axial_chord for the normalized arrays).
"""

import numpy as np

PERIODICITY_MODES = ("axisymmetric", "offset", "freestream")
DEFAULT_MODE = "axisymmetric"


def periodicity_of(dom_cfg):
    """The selected periodicity mode (validated, defaults axisymmetric)."""
    mode = (dom_cfg or {}).get("periodicity", DEFAULT_MODE)
    return mode if mode in PERIODICITY_MODES else DEFAULT_MODE


def is_periodic(dom_cfg):
    """True for the two periodic modes (False only for freestream)."""
    return periodicity_of(dom_cfg) != "freestream"


def _clip_ratio(y, radius):
    return np.clip(np.asarray(y, dtype=float) / radius, -0.999, 0.999)


def cartesian_to_uy(ss, ps, radius):
    """Unwrap Cartesian y -> arc-length uy: uy = R * asin(y / R).

    ss/ps are (m, 2) arrays in the normalized frame (blade centered at
    y = 0); `radius` is the annulus radius in the same units. Returns
    new arrays; the input is unchanged. With no usable radius (None or
    <= 0) the arrays are returned unchanged - Cartesian y and uy
    coincide for a vanishing unwrap."""
    radius = float(radius or 0.0)
    if radius <= 0.0:
        return ss, ps
    out = []
    for side in (ss, ps):
        side = np.array(side, dtype=float, copy=True)
        side[:, 1] = radius * np.arcsin(_clip_ratio(side[:, 1], radius))
        out.append(side)
    return out


def uy_to_cartesian(ss, ps, radius):
    """Wrap arc-length uy -> Cartesian y: y = R * sin(uy / R) (the exact
    inverse of cartesian_to_uy)."""
    radius = float(radius or 0.0)
    if radius <= 0.0:
        return ss, ps
    out = []
    for side in (ss, ps):
        side = np.array(side, dtype=float, copy=True)
        side[:, 1] = radius * np.sin(np.asarray(side[:, 1], dtype=float)
                                     / radius)
        out.append(side)
    return out
