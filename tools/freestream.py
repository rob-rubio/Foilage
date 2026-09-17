"""Consistent compressible freestream state for SU2 configs and post-processing.

Computes a thermodynamic state (INIT_OPTION= TD_CONDITIONS) whose Reynolds
number matches the requested Re: density is derived from Re, everything else
follows from the ideal-gas relations and Sutherland viscosity.

Import as a module (render_config.py, plot_case.py) or run directly to print
a state table.
"""

import math


def sutherland_mu(T, T_ref=273.15, mu_ref=1.71604e-5, S=110.4):
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


# thermal constant of the diatomic vibrational mode of air, SI
# (NASA Glenn beginner's guide: theta = 5500 degrees Rankine)
THETA_VIB_AIR_K = 5500.0 / 1.8
GAMMA_PERFECT_AIR = 1.4


def gamma_of_air(T):
    """Ratio of specific heats of air at temperature T [K].

    Calorically-imperfect harmonic-vibrator model, NASA Glenn
    (equation of Eggers, NACA Report 959 / 1135):

        gam = 1 + (gamp - 1) / (1 + (gamp - 1)
                                * (th/T)^2 e^(th/T) / (e^(th/T) - 1)^2)

    with gamp = 1.4 and th = 3055.6 K. Returns 1.400 at 300 K,
    1.364 at 700 K, 1.335 at 1000 K.
    """
    th = THETA_VIB_AIR_K / float(T)
    e = math.exp(th)
    f = th * th * e / ((e - 1.0) ** 2)
    g = GAMMA_PERFECT_AIR
    return 1.0 + (g - 1.0) / (1.0 + (g - 1.0) * f)


def state(mach, reynolds, reynolds_length=1.0, T_inf=288.15,
          gamma=1.4, R=287.058, reynolds_target=None):
    """Freestream dict consistent with (M, Re, L) at temperature T_inf.

    rho = Re * mu(T_inf) / (U_inf * L);  p = rho * R * T_inf;
    total conditions from isentropic relations.
    """
    a = math.sqrt(gamma * R * T_inf)
    U = mach * a
    mu = sutherland_mu(T_inf)
    rho = reynolds * mu / (U * reynolds_length)
    p = rho * R * T_inf
    fac = 1.0 + 0.5 * (gamma - 1.0) * mach ** 2
    return {
        "mach": mach,
        "reynolds": reynolds,
        "reynolds_length": reynolds_length,
        "gamma": gamma,
        "R": R,
        "T_inf": T_inf,
        "a_inf": a,
        "U_inf": U,
        "mu_inf": mu,
        "rho_inf": rho,
        "p_inf": p,
        "q_inf": 0.5 * rho * U ** 2,
        "T0": T_inf * fac,
        "p0": p * fac ** (gamma / (gamma - 1.0)),
    }


def fmt(x):
    return f"{x:.6g}"


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--mach", type=float, required=True)
    ap.add_argument("--reynolds", type=float, required=True)
    ap.add_argument("--length", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=288.15)
    args = ap.parse_args()

    s = state(args.mach, args.reynolds, args.length, args.T)
    print(json.dumps(s, indent=2))
