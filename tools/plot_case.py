"""Generic SU2 case post-processing driven by case.json.

Usage (from a case directory, or pass the directory as argument):
    python tools/plot_case.py [case_dir]

Expects in the case directory:
    case.json         case settings (physics + optional "postprocess" hints)
    history.csv       SU2 history
    vol_solution.vtk  SU2 legacy VTK volume output

Produces: convergence.png, fields.png, nearwall.png, bl_validation.png
Optional postprocess hints:
    wall_hint : [x, y] point inside the body (identifies the wall node set)
    probe     : {"point": [x,y] on wall, "normal": [nx,ny], "length": L}
                boundary-layer profile ray
    zoom      : {"xmin","xmax","ymin","ymax"} streamlines/mesh zoom window
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freestream import state  # noqa: E402
from su2_vtk import read_legacy_vtk  # noqa: E402

GAMMA, R_AIR = 1.4, 287.058
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": False})


# ----------------------------------------------------------------- history
def plot_convergence(case_dir, case_name, unsteady=None):
    cols = {}
    with open(case_dir / "history.csv") as f:
        reader = csv.reader(f)
        header = [h.strip().strip('"').strip("'") for h in next(reader)]
        arrays = [[] for _ in header]
        for row in reader:
            try:
                for i, v in enumerate(row):
                    arrays[i].append(float(v))
            except ValueError:
                continue
    for name, arr in zip(header, arrays):
        cols[name] = np.array(arr)

    rms = {k: v for k, v in cols.items() if k.startswith("rms[")}
    cd = cols.get("CD")
    cl = cols.get("CL")
    stats = {}

    if unsteady and cols.get("Time_Iter") is not None and cols["Time_Iter"].max() > 0 and cd is not None:
        # one sample per physical time step (last inner iteration of the step)
        ti = cols["Time_Iter"]
        steps = np.unique(ti)
        last = np.array([np.where(ti == s)[0][-1] for s in steps])
        t, cd_s, cl_s = steps, cd[last], cl[last]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.4))
        ax1.plot(t, cd_s, lw=0.5, color="tab:red", alpha=0.6, label="CD (per step)")
        win = max(1, len(cd_s) // 12)
        running = np.convolve(cd_s, np.ones(win) / win, mode="valid")
        ax1.plot(t[win - 1:], running, lw=2, color="black", label="CD running mean")
        w0 = unsteady.get("window_start_iter", 300)
        ax1.axvline(w0, color="k", ls=":", lw=1)
        ax1.text(w0, ax1.get_ylim()[1], " statistics window", fontsize=8, va="top")
        ax1.set_xlabel("time step")
        ax1.set_ylabel("CD")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"URANS limit cycle - {case_name} ({len(t)} steps)")

        ax2.plot(t, cl_s, lw=0.5, color="tab:blue", alpha=0.6, label="CL (per step)")
        ax2.set_xlabel("time step")
        ax2.set_ylabel("CL")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(case_dir / "convergence.png")
        plt.close(fig)

        sel = t >= max(w0, int(0.6 * t.max()))
        stats = {
            "CD_mean": float(np.mean(cd_s[sel])),
            "CD_std": float(np.std(cd_s[sel])),
            "CL_amp": float(np.max(np.abs(cl_s[sel]))),
        }
        mdot = cols.get("Avg_Massflow")
        if mdot is not None:
            stats["mdot_mean"] = float(np.mean(mdot[last][sel]))
        print(f"convergence.png  (steps={len(t)}, CD mean {stats['CD_mean']:.3f} "
              f"+/- {stats['CD_std']:.3f}, max|CL| {stats['CL_amp']:.3f})")
        return stats

    # ---- steady path (as before)
    it = cols.get("Inner_Iter", np.arange(len(next(iter(cols.values())))))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for k, v in rms.items():
        ax1.plot(it, v, lw=1.2, label=k.replace("rms", "rms "))
    ax1.set_ylabel("log10 residual")
    ax1.set_title(f"SU2 convergence - {case_name} ({len(it)} iterations)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    if cd is not None:
        ax2.plot(it, cd, color="tab:red", lw=1.2, label="$C_D$")
    if cl is not None:
        ax2.plot(it, cl, color="tab:blue", lw=1.2, label="$C_L$")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("coefficient")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir / "convergence.png")
    plt.close(fig)

    last = {k: v[-1] for k, v in rms.items()}
    msg = "  ".join(f"{k}={v:.2f}" for k, v in last.items())
    extra = f"  CD={cd[-1]:.4f}" if cd is not None else ""
    print(f"convergence.png  ({msg}{extra})")
    return stats


# ------------------------------------------------------------------- fields
def load_volume(case_dir):
    # unsteady runs write numbered snapshots; take the latest
    vols = sorted(case_dir.glob("vol_solution*.vtk"))
    if not vols:
        raise FileNotFoundError(f"no vol_solution*.vtk in {case_dir}")
    return read_legacy_vtk(str(vols[-1]))


def fields_from_volume(d, gamma=GAMMA, r=R_AIR):
    """Display-ready scalar fields from a parsed SU2 volume VTK dict.

    Returns {display_name: (array_1d, cmap)} with one value per node:
    vector fields (Skin_Friction_Coefficient, Momentum, ...) become their
    magnitudes, and total pressure/temperature are derived from the
    primitive fields. All contours use the classic rainbow (jet) map,
    blue at the low end.
    """
    cmap = "jet"
    out = {}
    for name, arr in d.items():
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            continue
        if name in ("points", "tris", "quads"):
            continue
        if name == "Velocity":
            out["Velocity magnitude"] = (
                np.linalg.norm(arr[:, :2], axis=1), cmap)
        elif arr.shape[1] == 1:
            out[name] = (arr.ravel(), cmap)
        else:
            out[name] = (np.linalg.norm(arr, axis=1), cmap)

    mach = d["Mach"].ravel() if "Mach" in d else None
    if mach is None and "Velocity" in d and "Temperature" in d:
        speed = np.linalg.norm(d["Velocity"][:, :2], axis=1)
        mach = speed / np.sqrt(gamma * r * d["Temperature"].ravel())
        d["Mach"] = mach.reshape(-1, 1)
        out["Mach"] = (mach, cmap)
    if mach is not None and "Pressure" in d:
        fac = 1.0 + 0.5 * (gamma - 1.0) * mach ** 2
        p = d["Pressure"].ravel()
        out["Total_Pressure"] = (p * fac ** (gamma / (gamma - 1.0)), cmap)
        if "Temperature" in d:
            out["Total_Temperature"] = (d["Temperature"].ravel() * fac, cmap)
    return out


def volume_triangulation(d):
    """(points, triangles) with quads split, ready for tricontourf/tripcolor."""
    pts = d["points"]
    parts = [d["tris"]]
    if d["quads"] is not None:
        q = d["quads"]
        parts += [q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
    conn = np.vstack([p for p in parts if p is not None])
    return pts, conn


def _volume_arrays(d):
    """(pts, conn, mach, cp, vel) from a parsed volume dict (legacy shape)."""
    pts, conn = volume_triangulation(d)
    return (pts, conn, d["Mach"].ravel(), d["Pressure_Coefficient"].ravel(),
            d["Velocity"][:, :2])


def plot_fields(case_dir, pts, conn, mach, cp, pitch=0.0):
    """Mach + Cp contour pictures.

    With a periodic pitch (cascade cases) the contours are drawn three
    times - main domain plus the upper and lower periodic copies - so the
    neighbouring blades and the periodicity are visible.
    """
    shifts = (-pitch, 0.0, pitch) if pitch else (0.0,)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6))
    for ax, (fld, title, cmap) in zip(
        axes,
        [(mach, "Mach number", "viridis"), (cp, r"pressure coefficient $C_p$", "coolwarm")],
    ):
        for dy in shifts:
            tc = ax.tricontourf(pts[:, 0], pts[:, 1] + dy, conn, fld,
                                levels=60, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        if pitch:
            ymin -= 0.45 * pitch
            ymax += 0.45 * pitch
        ax.set_ylim(ymin, ymax)
        ax.set_title(title + ("  (with periodic copies)" if pitch else ""),
                     fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(tc, ax=ax, shrink=0.9, pad=0.01)
    fig.tight_layout()
    fig.savefig(case_dir / "fields.png")
    plt.close(fig)
    print(f"fields.png  (max Mach = {mach.max():.3f})"
          + (f"  [+/- pitch periodic copies]" if pitch else ""))


# ---------------------------------------------------------------- near wall
def plot_nearwall(case_dir, pp, pts, conn, vel, U_inf, R=0.5):
    x0, x1 = pp.get("zoom", {}).get("xmin", -1.6), pp.get("zoom", {}).get("xmax", 3.4)
    y0, y1 = pp.get("zoom", {}).get("ymin", -1.4), pp.get("zoom", {}).get("ymax", 1.4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    gx, gy = np.meshgrid(np.linspace(x0, x1, 420), np.linspace(y0, y1, 300))
    u = griddata(pts, vel[:, 0], (gx, gy), method="linear")
    v = griddata(pts, vel[:, 1], (gx, gy), method="linear")
    inside = griddata(pts, np.hypot(pts[:, 0], pts[:, 1]), (gx, gy), method="linear") < R * 1.02
    u[inside] = np.nan
    v[inside] = np.nan
    ax1.contourf(gx, gy, np.nan_to_num(np.hypot(u, v) / U_inf, nan=0.0), levels=40, cmap="viridis")
    ax1.streamplot(gx, gy, u, v, density=1.6, color="k", linewidth=0.6, arrowsize=0.7)
    ax1.add_patch(Circle((0, 0), R, fc="0.25", zorder=5))
    ax1.set_aspect("equal")
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y0, y1)
    ax1.set_title("near-cylinder velocity + streamlines", fontsize=9)
    ax1.set_xlabel("x / D")
    ax1.set_ylabel("y / D")

    ax2.triplot(pts[:, 0], pts[:, 1], conn, lw=0.15, color="0.4")
    ax2.add_patch(Circle((0, 0), R, fc="k", zorder=5))
    ax2.set_aspect("equal")
    ax2.set_xlim(-0.9, 0.9)
    ax2.set_ylim(-0.9, 0.9)
    ax2.set_title("mesh near wall - BL layers", fontsize=9)
    ax2.set_xlabel("x / D")
    fig.tight_layout()
    fig.savefig(case_dir / "nearwall.png")
    plt.close(fig)
    print("nearwall.png")


# ------------------------------------------------------------ BL validation
def plot_bl_validation(case_dir, pp, d, pts, vel, fs, R=0.5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # --- probe-ray BL profile
    probe = pp.get("probe", {"point": [0.0, R], "normal": [0.0, 1.0], "length": 1.2})
    p0 = np.array(probe["point"])
    nrm = np.array(probe["normal"], dtype=float)
    nrm /= np.linalg.norm(nrm)
    rel = pts - p0
    t = rel @ nrm
    dist = np.abs(rel[:, 0] * nrm[1] - rel[:, 1] * nrm[0])
    tol = 0.008 * fs["reynolds_length"] if fs["reynolds_length"] else 0.008
    col = (t > 1e-6) & (t < probe.get("length", 1.2)) & (dist < tol)
    speed = np.linalg.norm(vel, axis=1)
    tt, mm = t[col], speed[col] / np.sqrt(fs["gamma"] * fs["R"] * fs["T_inf"])
    srt = np.argsort(tt)
    tt, mm = tt[srt], mm[srt]
    ax1.plot(tt / fs["reynolds_length"], mm, "o-", ms=3, lw=1)
    ax1.set_xlabel("distance from wall / L")
    ax1.set_ylabel("local Mach number")
    ax1.set_title(f"BL profile along probe ray\n({len(tt)} nodes)", fontsize=9)
    ax1.grid(alpha=0.3)

    # --- wall Cp / Cf distributions
    hint = np.array(pp.get("wall_hint", [0.0, 0.0]))
    speed0 = speed
    wall = speed0 < 1e-6 * fs["U_inf"]
    if wall.sum() < 5:
        print("  warning: wall nodes not found for Cp plot")
        fig.savefig(case_dir / "bl_validation.png")
        plt.close(fig)
        return
    dc = np.hypot(pts[wall, 0] - hint[0], pts[wall, 1] - hint[1])
    keep = dc < dc.min() + 1e-3 * (dc.max() + 1e-9)
    idx = np.where(wall)[0][keep]
    wall = np.zeros(len(pts), dtype=bool)
    wall[idx] = True
    theta = np.degrees(np.arctan2(*(pts[wall] - hint).T[::-1]))
    cpw = (d["Pressure"].ravel()[wall] - fs["p_inf"]) / fs["q_inf"]
    cfw = np.linalg.norm(d["Skin_Friction_Coefficient"][wall], axis=1)
    srt = np.argsort(theta)
    ax2.plot(theta[srt], cpw[srt], "o-", ms=2.5, lw=1, label=r"$C_p$", color="tab:blue")
    ax2.set_xlabel(r"angle $\theta$ (deg, 0 = downstream)")
    ax2.set_ylabel(r"$C_p$", color="tab:blue")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(theta[srt], cfw[srt], "s-", ms=2, lw=0.8, color="tab:red", alpha=0.7)
    ax3.set_ylabel(r"$|C_f|$", color="tab:red")
    ax2.set_title("wall distributions", fontsize=9)
    fig.tight_layout()
    fig.savefig(case_dir / "bl_validation.png")
    plt.close(fig)
    print(f"bl_validation.png  (wall nodes: {int(wall.sum())}, first probe node at {tt[0]/fs['reynolds_length']:.4f} L)")


# ------------------------------------------------------------------ cascade
def cascade_refs(cas, physics):
    """Isentropic reference state from the cascade BCs + SU2 init q (for Cf)."""
    g = physics.get("gamma", GAMMA)
    R = physics.get("gas_constant", R_AIR)
    p01 = cas["inlet"]["total_pressure"]
    T01 = cas["inlet"]["total_temperature"]
    p2 = cas["outlet"]["static_pressure"]
    T2 = T01 * (p2 / p01) ** ((g - 1) / g)
    V2 = (2 * g / (g - 1) * R * (T01 - T2)) ** 0.5
    rho_i = physics.get("init_pressure", 1e5) / (R * physics.get("init_temperature", 660.0))
    a_i = (g * R * physics.get("init_temperature", 660.0)) ** 0.5
    q_init = 0.5 * rho_i * (physics["mach"] * a_i) ** 2
    return {"p01": p01, "T01": T01, "p2": p2, "T2": T2, "V2": V2,
            "M2": V2 / (g * R * T2) ** 0.5, "q_init": q_init}


def wall_points(pts, vel, V_ref):
    """Airfoil surface nodes as a closed loop, ordered by walking nearest
    neighbors (angle-sorting breaks on cambered/swept surfaces)."""
    wall = np.linalg.norm(vel, axis=1) < 1e-6 * V_ref
    wp = pts[wall]
    out = [np.argmin(np.linalg.norm(wp, axis=1))]   # start near origin = LE
    used = {out[0]}
    for _ in range(len(wp) - 1):
        last = wp[out[-1]]
        d = np.linalg.norm(wp - last, axis=1)
        d[np.array(sorted(used))] = np.inf
        nxt = int(np.argmin(d))
        used.add(nxt)
        out.append(nxt)
    return wp[np.array(out)]


def _wall_loop_indices(pts, vel, V_ref):
    """Wall nodes as global indices, ordered LE -> TE -> LE by nearest
    neighbor walk (same ordering as wall_points)."""
    wall = np.linalg.norm(vel, axis=1) < 1e-6 * V_ref
    wp = pts[wall]
    if len(wp) < 10:
        return None
    walk = [int(np.argmin(np.linalg.norm(wp, axis=1)))]
    used = {walk[0]}
    for _ in range(len(wp) - 1):
        last = wp[walk[-1]]
        d = np.linalg.norm(wp - last, axis=1)
        d[np.array(sorted(used))] = np.inf
        nxt = int(np.argmin(d))
        used.add(nxt)
        walk.append(nxt)
    return np.where(wall)[0][np.array(walk)]


def surface_distributions(d, v_ref, p0_ref, gamma=GAMMA, p_ref=None,
                          ss_upper=True):
    """1D wall distributions split into suction/pressure sides.

    Splits the ordered wall loop at the LE (x-min) and TE (x-max) into two
    sides and normalizes surface arc length to u in [0, 1] (LE -> TE).
    Isentropic Mach uses the classic turbomachinery definition with the
    reference total pressure p0_ref (inlet p01) and the local wall static
    pressure:  Ma_is = sqrt(2/(gamma-1) * ((p0/p)^((gamma-1)/gamma) - 1)).

    Returns {"ss": {...}, "ps": {...}} or None when no wall nodes are
    found. Per side: u, x, y, p, cf, ma_isen, cp, yplus (when available).
    """
    pts = d["points"]
    vel = d["Velocity"][:, :2]
    loop = _wall_loop_indices(pts, vel, v_ref)
    if loop is None:
        return None

    x = pts[loop, 0]
    i_le = int(np.argmin(x))
    loop = np.roll(loop, -i_le)              # start at the LE
    x = pts[loop, 0]
    i_te = int(np.argmax(x))
    side_a = loop[:i_te + 1]                     # LE -> TE along one side
    side_b = np.concatenate([loop[i_te:], loop[:1]])  # TE -> LE (wrap)
    side_b = side_b[::-1]                        # reversed: LE -> TE

    upper = side_a if np.mean(pts[side_a, 1]) > np.mean(pts[side_b, 1]) \
        else side_b
    lower = side_b if upper is side_a else side_a
    ss, ps = (upper, lower) if ss_upper else (lower, upper)

    p_all = d["Pressure"].ravel()
    cf_all = (np.linalg.norm(d["Skin_Friction_Coefficient"][:, :2], axis=1)
              if "Skin_Friction_Coefficient" in d else None)
    yp_all = d["Y_Plus"].ravel() if "Y_Plus" in d else None
    if p_ref is None:
        p_ref = float(np.median(p_all[loop]))

    out = {}
    for name, idx in (("ss", ss), ("ps", ps)):
        xy = pts[idx]
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        u = s / s[-1] if s[-1] > 0 else s
        p = p_all[idx]
        ma_is = np.sqrt(np.clip(
            2.0 / (gamma - 1.0) * ((p0_ref / p) ** ((gamma - 1.0) / gamma)
                                   - 1.0), 0.0, None))
        # back-surface diffusion factor (Lieblein): the fractional
        # deceleration downstream of the surface velocity peak,
        # DF(u) = 1 - V(u)/V_peak, with V proportional to Ma_is at
        # constant stagnation temperature. Upstream of the peak DF = 0.
        peak = int(np.argmax(ma_is))
        ma_peak = float(np.max(ma_is))
        diffusion = np.zeros_like(ma_is)
        diffusion[peak:] = 1.0 - ma_is[peak:] / max(ma_peak, 1e-12)
        side = {"u": u, "x": xy[:, 0], "y": xy[:, 1], "p": p,
                "ma_isen": ma_is,
                "cp": (p - p_ref) / max(p0_ref - p_ref, 1e-12),
                "diffusion": diffusion,
                "cf": cf_all[idx] if cf_all is not None else np.zeros(len(idx))}
        if yp_all is not None:
            side["yplus"] = yp_all[idx]
        out[name] = side
        # 0-D back-surface diffusion factor at the trailing edge
        out.setdefault("summary", {})[name] = {
            "DF": float(1.0 - ma_is[-1] / max(ma_peak, 1e-12)),
            "ma_peak": ma_peak,
            "ma_te": float(ma_is[-1]),
            "peak_u": float(u[peak]),
        }
    return out


def write_ma_af(case_dir, dist):
    """Export the surface distributions used by the GUI 1D charts.

    Schema: {"ss": {"u": [...], "ma": [...], "cf": [...]},
             "ps": {"u": [...], "ma": [...], "cf": [...]}}
    with u the surface coordinate from 0 (LE) to 1 (TE), ma the isentropic
    Mach number and cf the skin friction coefficient magnitude.
    """
    out = {}
    for side in ("ss", "ps"):
        s = dist[side]
        out[side] = {"u": [float(v) for v in s["u"]],
                     "ma": [float(v) for v in s["ma_isen"]],
                     "cf": [float(v) for v in s["cf"]]}
    path = Path(case_dir) / "ma_af.json"
    path.write_text(json.dumps(out))
    print(f"ma_af.json written ({len(out['ss']['u'])} ss / "
          f"{len(out['ps']['u'])} ps points)")


def plot_nearwall_cascade(case_dir, pp, pts, conn, vel, wp, V_ref):
    zoom = pp.get("zoom", {})
    x0, x1 = zoom.get("xmin", -0.01), zoom.get("xmax", 0.05)
    y0, y1 = zoom.get("ymin", -0.04), zoom.get("ymax", 0.04)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    gx, gy = np.meshgrid(np.linspace(x0, x1, 420), np.linspace(y0, y1, 300))
    u = griddata(pts, vel[:, 0], (gx, gy), method="linear")
    v = griddata(pts, vel[:, 1], (gx, gy), method="linear")
    inside = griddata(pts, np.hypot(pts[:, 0], pts[:, 1]), (gx, gy),
                      method="linear") < 0  # placeholder, real mask below
    from matplotlib.path import Path
    mask = Path(np.vstack([wp, wp[0]])).contains_points(
        np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    u[mask] = np.nan
    v[mask] = np.nan
    ax1.contourf(gx, gy, np.nan_to_num(np.hypot(u, v), nan=0.0), levels=40, cmap="viridis")
    ax1.streamplot(gx, gy, u, v, density=1.6, color="k", linewidth=0.6, arrowsize=0.7)
    ax1.plot(np.append(wp[:, 0], wp[0, 0]), np.append(wp[:, 1], wp[0, 1]),
             color="0.2", lw=1.4)
    ax1.set_aspect("equal")
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y0, y1)
    ax1.set_title("near-vane velocity + streamlines [m/s]", fontsize=9)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")

    ax2.triplot(pts[:, 0], pts[:, 1], conn, lw=0.15, color="0.4")
    ax2.plot(np.append(wp[:, 0], wp[0, 0]), np.append(wp[:, 1], wp[0, 1]),
             color="k", lw=1.2)
    ax2.set_aspect("equal")
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y0, y1)
    ax2.set_title("mesh near wall - BL layers", fontsize=9)
    ax2.set_xlabel("x [m]")
    fig.tight_layout()
    fig.savefig(case_dir / "nearwall.png")
    plt.close(fig)
    print("nearwall.png")


def plot_wall_cascade(case_dir, pp, d, pts, vel, refs):
    """BL probe profile + wall Cp (cascade refs) / skin friction vs x/c."""
    from matplotlib.path import Path  # noqa: F401  (consistent import point)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    probe = pp.get("probe", {"point": [0.0, 0.0], "normal": [0.0, 1.0], "length": 0.02})
    p0 = np.array(probe["point"])
    nrm = np.array(probe["normal"], dtype=float)
    nrm /= np.linalg.norm(nrm)
    rel = pts - p0
    t = rel @ nrm
    dist = np.abs(rel[:, 0] * nrm[1] - rel[:, 1] * nrm[0])
    tol = 0.008 * 0.038
    col = (t > 1e-9) & (t < probe.get("length", 0.02)) & (dist < tol)
    speed = np.linalg.norm(vel, axis=1)
    tt, vv = t[col], speed[col]
    srt = np.argsort(tt)
    ax1.plot(tt[srt] * 1e3, vv[srt], "o-", ms=3, lw=1)
    ax1.axhline(refs["V2"], color="k", ls=":", lw=1)
    ax1.annotate(f"isentropic $V_2$ = {refs['V2']:.0f} m/s", xy=(0.98, refs["V2"]),
                 xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8)
    ax1.set_xlabel("distance from wall [mm]")
    ax1.set_ylabel("velocity [m/s]")
    ax1.set_title(f"BL profile along probe ray\n({len(tt)} nodes)", fontsize=9)
    ax1.grid(alpha=0.3)

    wall = speed < 1e-6 * refs["V2"]
    wp = pts[wall]
    xle = wp[:, 0].min()
    c_ax = wp[:, 0].max() - xle
    cpw = (d["Pressure"].ravel()[wall] - refs["p2"]) / (refs["p01"] - refs["p2"])
    tau = np.linalg.norm(d["Skin_Friction_Coefficient"][wall], axis=1) * refs["q_init"]
    xx = (wp[:, 0] - xle) / c_ax
    ax2.plot(xx, cpw, ".", ms=3, color="tab:blue")
    ax2.set_xlabel("$x/c_{axial}$")
    ax2.set_ylabel(r"$C_p = (p-p_2)/(p_{01}-p_2)$", color="tab:blue")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(xx, tau, ".", ms=2, color="tab:red", alpha=0.6)
    ax3.set_ylabel(r"$\tau_w$ [Pa]", color="tab:red")
    ax2.set_title("wall distributions (both surfaces)", fontsize=9)
    fig.tight_layout()
    fig.savefig(case_dir / "bl_validation.png")
    plt.close(fig)
    print(f"bl_validation.png  (wall nodes: {int(wall.sum())}, "
          f"first probe node at {tt[srt][0]*1e3:.3f} mm)")


# ------------------------------------------------------------- results.json
def read_history(path):
    """Parse an SU2 history.csv into {column: np.ndarray} (GUI-reusable)."""
    with open(path) as f:
        reader = csv.reader(f)
        header = [h.strip().strip('"').strip("'") for h in next(reader)]
        arrays = [[] for _ in header]
        for row in reader:
            try:
                for i, v in enumerate(row):
                    arrays[i].append(float(v))
            except ValueError:
                continue
    return {name: np.array(arr) for name, arr in zip(header, arrays)}


# keep the historical private name working for any existing callers
_read_history = read_history


def _first_layer_height(case_dir):
    """min/median first-cell height above the airfoil surface (mesh units)."""
    path = case_dir / "mesh.su2"
    if not path.exists():
        return None
    with open(path) as f:
        lines = f.read().splitlines()
    i, pts, quads, tris, edges = 0, None, [], [], []
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("NPOIN="):
            n = int(ln.split("=")[1].split()[0])
            pts = np.zeros((n, 2))
            for j in range(n):
                p = lines[i + 1 + j].split()
                pts[j, 0], pts[j, 1] = float(p[0]), float(p[1])
            i += n + 1
            continue
        if ln.startswith("NELEM="):
            ne = int(ln.split("=")[1].split()[0])
            for j in range(ne):
                p = [int(x) for x in lines[i + 1 + j].split()]
                if p[0] == 9:
                    quads.append(p[1:5])
                elif p[0] == 5:
                    tris.append(p[1:4])
            i += ne + 1
            continue
        if ln.startswith("MARKER_TAG= airfoil"):
            cnt = int(lines[i + 1].split("=")[1].split()[0])
            for j in range(cnt):
                p = [int(x) for x in lines[i + 2 + j].split()]
                edges.append((p[1], p[2]))
            break
        i += 1
    if pts is None or not edges:
        return None
    from collections import defaultdict
    e2e = defaultdict(list)
    for el in [np.array(q) for q in quads] + [np.array(t) for t in tris]:
        n = len(el)
        for k in range(n):
            a, b = int(el[k]), int(el[(k + 1) % n])
            e2e[(min(a, b), max(a, b))].append(el)
    heights = []
    for a, b in edges:
        pa, pb = pts[a], pts[b]
        e = pa - pb
        L = np.hypot(*e)
        elems = e2e.get((min(a, b), max(a, b)), [])
        hs = [abs(e[0] * (pts[o] - pb)[1] - e[1] * (pts[o] - pb)[0]) / L
              for el in elems for o in el if o != a and o != b]
        if hs:
            heights.append(min(hs))
    if not heights:
        return None
    return float(np.min(heights)), float(np.median(heights))


def write_results(case_dir, case, fs, dist=None):
    """Write results.json: convergence, forces, cascade plane audit, y+."""
    out = {"case": case_dir.resolve().name}
    hist = read_history(case_dir / "history.csv")

    rms = {k: v for k, v in hist.items() if k.startswith("rms[")}
    minval = case.get("convergence", {}).get("minval", -6.0)
    iters = int(hist["Inner_Iter"][-1])
    out["convergence"] = {
        "iterations_run": iters,
        "rms_final": {k: float(v[-1]) for k, v in rms.items()},
        "rms_min": {k: float(v.min()) for k, v in rms.items()},
        "target_rms": minval,
        "target_reached": bool(all(v[-1] <= minval for v in rms.values())),
    }
    out["forces"] = {
        "CD": float(hist["CD"][-1]),
        "CL": float(hist["CL"][-1]),
    }

    d = load_volume(case_dir)
    pts, conn, mach, cp, vel = _volume_arrays(d)
    out["fields"] = {"max_mach": float(mach.max())}

    # wall y+ from skin friction
    speed = np.linalg.norm(vel, axis=1)
    cascade = case.get("cascade")
    if cascade:
        refs = cascade_refs(cascade, case["physics"])
        V_ref, q_ref = refs["V2"], refs["q_init"]
    else:
        V_ref, q_ref = fs["U_inf"], fs["q_inf"]
    g = case["physics"].get("gamma", GAMMA)
    R = case["physics"].get("gas_constant", R_AIR)
    wall = speed < 1e-6 * V_ref
    if wall.any():
        rho_w = d["Pressure"].ravel()[wall] / (R * d["Temperature"].ravel()[wall])
        mu_w = 1.716e-5 * (d["Temperature"].ravel()[wall] / 273.15) ** 1.5 \
            * (383.55 / (d["Temperature"].ravel()[wall] + 110.4))
        tau = np.linalg.norm(d["Skin_Friction_Coefficient"][wall, :2], axis=1) * q_ref
        fl = _first_layer_height(case_dir)
        if fl:
            yplus = np.sqrt(tau / rho_w) * fl[1] * rho_w / mu_w
            out["wall"] = {"first_layer_height_units": fl,
                           "yplus_median": float(np.median(yplus)),
                           "yplus_p95": float(np.percentile(yplus, 95)),
                           "wall_nodes": int(wall.sum())}

    if cascade:
        for name, xf in (("inlet", pts[:, 0].min()), ("outlet", pts[:, 0].max())):
            m = np.isclose(pts[:, 0], xf, atol=1e-7)
            idx = np.where(m)[0]
            y = pts[idx, 1]
            srt = np.argsort(y)
            y = y[srt]
            pi, Ti = d["Pressure"].ravel()[idx][srt], d["Temperature"].ravel()[idx][srt]
            vi, Mi = vel[idx][srt], mach[idx][srt]
            ri = pi / (R * Ti)
            u = vi[:, 0]
            md = np.trapezoid(ri * u, y)
            p0l = pi * (1 + 0.2 * Mi ** 2) ** 3.5
            out[name] = {
                "mass_flow_kg_s_m": float(md),
                "mach": float(np.trapezoid(ri * u * Mi, y) / md),
                "velocity_m_s": float(np.trapezoid(ri * u * np.linalg.norm(vi, axis=1), y) / md),
                "flow_angle_deg": float(np.degrees(np.arctan2(
                    np.trapezoid(ri * u * vi[:, 1], y), np.trapezoid(ri * u * u, y)))),
                "static_p_pa": float(np.trapezoid(ri * u * pi, y) / md),
                "p0_pa": float(np.trapezoid(ri * u * p0l, y) / md),
            }
        p01 = cascade["inlet"]["total_pressure"]
        p2 = cascade["outlet"]["static_pressure"]
        p02 = out["outlet"]["p0_pa"]
        out["losses"] = {"total_pressure_loss_coeff_Yp": float((p01 - p02) / (p01 - p2))}
        md_i = out["inlet"]["mass_flow_kg_s_m"]
        md_o = out["outlet"]["mass_flow_kg_s_m"]
        out["mass_balance"] = {"imbalance_pct": float(abs(md_i - md_o) / md_i * 100)}

    if dist and dist.get("summary"):
        out["back_surface_diffusion"] = {
            side: {"DF": round(vals["DF"], 4),
                   "ma_peak": round(vals["ma_peak"], 4),
                   "peak_u": round(vals["peak_u"], 4)}
            for side, vals in dist["summary"].items()}
    (case_dir / "results.json").write_text(json.dumps(out, indent=2))
    print("results.json written")


def main():
    case_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    case = json.loads((case_dir / "case.json").read_text())
    case_name = case_dir.resolve().name
    fs = state(case["physics"]["mach"], case["physics"]["reynolds"],
               case["physics"].get("reynolds_length", 1.0),
               case["physics"].get("freestream_temperature", 288.15),
               case["physics"].get("gamma", 1.4), case["physics"].get("gas_constant", 287.058))
    pp = case.get("postprocess", {})

    d = load_volume(case_dir)
    pts, conn, mach, cp, vel = _volume_arrays(d)
    plot_convergence(case_dir, case_name, case.get("unsteady"))
    pitch = 0.0
    if "cascade" in case:
        pers = case["cascade"].get("periodic") or []
        if pers:
            pitch = float((pers[0].get("translation") or [0, 0, 0])[1] or 0.0)
    plot_fields(case_dir, pts, conn, mach, cp, pitch=pitch)
    gamma = case["physics"].get("gamma", GAMMA)
    ss_upper = case.get("postprocess", {}).get("ss_upper", True)
    if "cascade" in case:
        refs = cascade_refs(case["cascade"], case["physics"])
        wp = wall_points(pts, vel, refs["V2"])
        plot_nearwall_cascade(case_dir, pp, pts, conn, vel, wp, refs["V2"])
        plot_wall_cascade(case_dir, pp, d, pts, vel, refs)
        dist = surface_distributions(d, refs["V2"], refs["p01"], gamma,
                                     p_ref=refs["p2"], ss_upper=ss_upper)
        if dist:
            write_ma_af(case_dir, dist)
        print(f"cascade refs: M2_isen={refs['M2']:.3f}  V2={refs['V2']:.1f} m/s  "
              f"T2={refs['T2']:.1f} K")
    else:
        plot_nearwall(case_dir, pp, pts, conn, vel, fs["U_inf"])
        plot_bl_validation(case_dir, pp, d, pts, vel, fs)
        dist = surface_distributions(d, fs["U_inf"], fs["p0"], gamma,
                                     p_ref=fs["p_inf"], ss_upper=ss_upper)
        if dist:
            write_ma_af(case_dir, dist)

    write_results(case_dir, case, fs, dist=dist)


if __name__ == "__main__":
    main()
