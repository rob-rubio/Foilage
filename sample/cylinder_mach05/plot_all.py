"""Post-processing for the SU2 Mach-0.5 cylinder demo.

Produces:
    convergence.png   residual history + drag/lift history
    fields.png        Mach number and pressure coefficient, full domain
    nearwall.png      near-wall zoom: streamlines/twin vortices + BL mesh
    bl_validation.png BL velocity profile + Cp/Cf around the cylinder
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import griddata

from read_su2_vtk import read_legacy_vtk

GAMMA, R_AIR = 1.4, 287.058
R = 0.5
P_INF, RHO_INF, U_INF = 0.34797, 4.2069e-6, 170.147

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": False})


# ----------------------------------------------------------------- history
def plot_convergence():
    it, rms_r, rms_e, cd, cl = [], [], [], [], []
    with open("history.csv") as f:
        for row in csv.reader(f):
            if row and row[0].strip().startswith(("Time_Iter",)):
                continue
            try:
                vals = [float(x) for x in row]
            except ValueError:
                continue
            it.append(vals[2])
            rms_r.append(vals[3])
            rms_e.append(vals[6])
            cd.append(vals[7])
            cl.append(vals[8])
    it, rms_r, rms_e, cd, cl = map(np.array, (it, rms_r, rms_e, cd, cl))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(it, rms_r, label=r"rms$\,\rho$ (continuity)", lw=1.2)
    ax1.plot(it, rms_e, label=r"rms$\,\rho E$ (energy)", lw=1.2, alpha=0.8)
    ax1.axhline(-8, color="k", ls=":", lw=0.8)
    ax1.set_ylabel("log10 residual")
    ax1.set_title("SU2 convergence - M=0.5 laminar cylinder, Re$_D$=40")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(it, cd, color="tab:red", label="C$_D$ (cylinder)")
    ax2.plot(it, cl, color="tab:blue", label="C$_L$ (cylinder)")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("coefficient")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("convergence.png")
    plt.close(fig)
    print(f"convergence.png  (final rms_rho={rms_r[-1]:.2f}, CD={cd[-1]:.4f}, CL={cl[-1]:.2e})")


# ------------------------------------------------------------------- fields
def load_volume():
    d = read_legacy_vtk("vol_solution.vtk")
    pts = d["points"]
    tris = [d["tris"]]
    if d["quads"] is not None:  # split quads for plotting
        q = d["quads"]
        tris.append(q[:, [0, 1, 2]])
        tris.append(q[:, [0, 2, 3]])
    conn = np.vstack([t for t in tris if t is not None])
    mach = d["Mach"].ravel()
    cp = d["Pressure_Coefficient"].ravel()
    vel = d["Velocity"][:, :2]
    return pts, conn, mach, cp, vel


def plot_fields(pts, conn, mach, cp):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6))
    labels = ["inlet", "outlet", "freestream (top/bottom)"]
    for ax, (fld, title, cmap) in zip(
        axes,
        [(mach, "Mach number", "viridis"), (cp, r"pressure coefficient $C_p$", "coolwarm")],
    ):
        tc = ax.tricontourf(pts[:, 0], pts[:, 1], conn, fld, levels=60, cmap=cmap)
        ax.add_patch(Circle((0, 0), R, fc="k", zorder=5))
        ax.set_aspect("equal")
        ax.set_xlim(-5, 15)
        ax.set_ylim(-3, 3)
        ax.set_title(title, fontsize=10)
        ax.text(-4.7, 0, "INLET", rotation=90, va="center", fontsize=8, color="k")
        ax.text(14.3, 0, "OUTLET", rotation=270, va="center", fontsize=8, color="k")
        ax.text(5, 2.7, "freestream", ha="center", fontsize=8)
        ax.text(5, -2.85, "freestream", ha="center", fontsize=8)
        fig.colorbar(tc, ax=ax, shrink=0.9, pad=0.01)
        ax.set_xlabel("x / D")
        ax.set_ylabel("y / D")
    fig.tight_layout()
    fig.savefig("fields.png")
    plt.close(fig)
    print(f"fields.png  (max Mach = {mach.max():.3f})")


def plot_nearwall(pts, conn, mach, vel):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # --- streamlines on regular grid
    x0, x1, y0, y1 = -1.6, 3.4, -1.4, 1.4
    gx, gy = np.meshgrid(np.linspace(x0, x1, 420), np.linspace(y0, y1, 300))
    u = griddata(pts, vel[:, 0], (gx, gy), method="linear")
    v = griddata(pts, vel[:, 1], (gx, gy), method="linear")
    speed = np.hypot(u, v)
    inside = np.hypot(gx, gy) < R * 1.02
    u[inside] = np.nan
    v[inside] = np.nan
    ax1.contourf(gx, gy, np.nan_to_num(speed / U_INF, nan=0.0), levels=40, cmap="viridis")
    ax1.streamplot(gx, gy, u, v, density=1.6, color="k", linewidth=0.6, arrowsize=0.7)
    ax1.add_patch(Circle((0, 0), R, fc="0.25", zorder=5))
    ax1.set_aspect("equal")
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y0, y1)
    ax1.set_title("near-cylinder velocity + streamlines (twin recirculation)", fontsize=9)
    ax1.set_xlabel("x / D")
    ax1.set_ylabel("y / D")

    # --- mesh zoom at the wall (boundary-layer capture)
    x0m, x1m, y0m, y1m = -0.9, 0.9, -0.9, 0.9
    ax2.triplot(pts[:, 0], pts[:, 1], conn, lw=0.15, color="0.4")
    ax2.add_patch(Circle((0, 0), R, fc="k", zorder=5))
    ax2.set_aspect("equal")
    ax2.set_xlim(x0m, x1m)
    ax2.set_ylim(y0m, y1m)
    n_cells_near = int(np.sum((np.hypot(pts[conn[:, 0], 0], pts[conn[:, 0], 1]) < 1.3)))
    ax2.set_title(f"mesh near wall - BL quad layers visible\n(~{n_cells_near} cells within 1.6D of cylinder)", fontsize=9)
    ax2.set_xlabel("x / D")
    fig.tight_layout()
    fig.savefig("nearwall.png")
    plt.close(fig)
    print("nearwall.png")


def plot_bl_validation(pts, mach):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # --- velocity profile along vertical ray at top of cylinder (theta=90deg)
    col = (np.abs(pts[:, 0]) < 0.008) & (pts[:, 1] > R * 1.001) & (pts[:, 1] < 1.6)
    ys = pts[col, 1]
    order = np.argsort(ys)
    ys, prof = ys[order], mach[col][order]
    _, uniq = np.unique(np.round(ys, 4), return_index=True)
    ys, prof = ys[uniq], prof[uniq]
    ax1.plot(ys - R, prof / 1.0, "o-", ms=3, lw=1)
    ax1.axhline(1.0, color="k", ls=":", lw=0.8)
    ax1.text(0.05, 1.02, "local edge Mach", fontsize=8)
    ax1.set_xlabel("(y - R) / D   above cylinder top")
    ax1.set_ylabel("local Mach number")
    ax1.set_title(f"BL profile at $\\theta$=90$^\\circ$\n({len(ys)} nodes across first 1.1 D)", fontsize=9)
    ax1.set_xlim(-0.01, 1.1)
    ax1.grid(alpha=0.3)

    # --- Cp and Cf around cylinder
    r = np.hypot(pts[:, 0], pts[:, 1])
    wall = np.abs(r - R) < 2e-3
    tw = np.degrees(np.arctan2(pts[wall, 1], pts[wall, 0]))  # 0=downstream, 180=upstream
    d = read_legacy_vtk("vol_solution.vtk")
    cp_w = (d["Pressure"].ravel()[wall] - P_INF) / (0.5 * RHO_INF * U_INF**2)
    cf_w = np.linalg.norm(d["Skin_Friction_Coefficient"][wall], axis=1)
    srt = np.argsort(tw)
    ax2.plot(tw[srt], cp_w[srt], "o-", ms=2.5, lw=1, label=r"$C_p$", color="tab:blue")
    ax2.set_xlabel(r"angle $\theta$ (deg, 0 = downstream stagnation)")
    ax2.set_ylabel(r"$C_p$", color="tab:blue")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(tw[srt], np.abs(cf_w[srt]), "s-", ms=2, lw=0.8, label=r"$|C_f|$", color="tab:red", alpha=0.7)
    ax3.set_ylabel(r"$|C_f|$", color="tab:red")
    ax2.set_title("wall distributions around cylinder", fontsize=9)
    fig.tight_layout()
    fig.savefig("bl_validation.png")
    plt.close(fig)
    print(f"bl_validation.png  (first-node offset = {ys[1]-ys[0]:.4f} D)")


if __name__ == "__main__":
    pts, conn, mach, cp, vel = load_volume()
    plot_convergence()
    plot_fields(pts, conn, mach, cp)
    plot_nearwall(pts, conn, mach, vel)
    plot_bl_validation(pts, mach)
