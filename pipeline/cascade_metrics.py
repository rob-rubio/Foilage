"""Cascade geometry metrics: channel width, throat, unguided turning,
surface curvature. Pure numpy so pipeline and GUI can both use it.

All inputs are normalized airfoil arrays as produced by build_airfoil /
section_to_airfoil (axial chord = 1, LE at x = 0, left-to-right).
"""

import math

import numpy as np


def channel_widths(ss, ps, pitch, ss_upper):
    """Width of the blade-to-blade channel.

    The passage adjacent to the suction side is bounded by that surface
    and the neighboring blade's pressure side: the neighbor sits at
    +pitch for CAP blades (suction side up) and at -pitch for CUP blades
    (suction side down). For every suction-side point, the distance to
    the nearest point of that neighbor pressure side. Returns (s_min per
    SS point, index into ps of the nearest point).
    """
    sgn = 1.0 if ss_upper else -1.0
    nb_ps = np.asarray(ps) + np.array([0.0, sgn * float(pitch)])
    d = np.linalg.norm(ss[:, None, :] - nb_ps[None, :, :], axis=2)
    j = np.argmin(d, axis=1)
    return d[np.arange(len(ss)), j], j


def _surface_angle(xy, i, half=3):
    """Local surface tangent angle [deg from +x] around point index i."""
    a = max(0, i - half)
    b = min(len(xy) - 1, i + half)
    dx, dy = xy[b] - xy[a]
    return float(np.degrees(np.arctan2(dy, dx)))


def _exit_angle(xy, u0=0.85, u1=0.95, half=3):
    """Mean surface tangent angle [deg] over an aft window (u in [u0, u1]).

    The window stays clear of the TE arc curl and blunt-TE face, so with a
    straightened aft suction side this is the exit metal angle.
    """
    n = len(xy)
    i0, i1 = int(u0 * (n - 1)), int(u1 * (n - 1))
    vx = vy = 0.0
    for i in range(i0, i1 + 1):
        a = max(0, i - half)
        b = min(n - 1, i + half)
        dx, dy = xy[b] - xy[a]
        L = math.hypot(dx, dy)
        if L > 0:
            vx += dx / L
            vy += dy / L
    return float(np.degrees(np.arctan2(vy, vx)))


def throat_metrics(ss, ps, pitch, ss_upper, half=4):
    """Throat and unguided-turning metrics, measured on the suction side
    (the surface guiding the flow through the throat).

    Unguided turning is the metal-angle change from the throat to the
    trailing edge: alpha_throat - alpha_exit.
    """
    s_ch, _j = channel_widths(ss, ps, pitch, ss_upper)
    ki = int(np.argmin(s_ch))
    width = float(s_ch[ki])

    a_throat = _surface_angle(ss, ki, half)
    a_exit = _exit_angle(ss)
    return {
        "width": width,                    # axial-chord units
        "x_over_cax": float(ss[ki, 0]),    # throat location on the SS
        "u": float(ki) / (len(ss) - 1),
        "angle_throat_deg": a_throat,
        "angle_exit_deg": a_exit,
        "unguided_turning_deg": a_throat - a_exit,
    }


def curvature(xy, smooth=5):
    """Signed-magnitude curvature of an open polyline.

    Returns (kappa, s) with kappa in 1/(axial-chord units) at each point
    and s the arc length coordinate. Coordinates are lightly smoothed
    first so imported (noisy) polylines stay readable.
    """
    xy = np.asarray(xy, dtype=float)
    if smooth and smooth > 1 and len(xy) > smooth:
        kernel = np.ones(smooth) / smooth
        pad = smooth // 2
        xp = np.convolve(np.pad(xy[:, 0], pad, mode="edge"), kernel,
                         mode="valid")
        yp = np.convolve(np.pad(xy[:, 1], pad, mode="edge"), kernel,
                         mode="valid")
        xy = np.column_stack([xp, yp])
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s[s == 0] = 1e-12
    xs, ys = np.gradient(xy[:, 0], s), np.gradient(xy[:, 1], s)
    xss, yss = np.gradient(xs, s), np.gradient(ys, s)
    kappa = np.abs(xs * yss - ys * xss) / (xs ** 2 + ys ** 2) ** 1.5
    return kappa, s
