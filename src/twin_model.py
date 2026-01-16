import numpy as np
from scipy.optimize import curve_fit

R = 8.314
T_REF = 303


def ph_decay_model(t, pH_inf, k, pH0):
    return pH_inf + (pH0 - pH_inf) * np.exp(-k * t)


def compute_k_eff(k0, OD, T, PD, Ea=25000, alpha=1.5, Ks=0.3):

    monod = OD / (Ks + OD + 1e-6)
    arrhenius = np.exp((Ea / R) * (1 / T_REF - 1 / T))
    pd_factor = 1 / (1 + alpha * PD)

    return k0 * monod * arrhenius * pd_factor


# =========================================================
# CONSTRAINED FIT — FORCE FURTHER ACIDIFICATION
# =========================================================

def fit_ph_model(time_days, pH_values):

    mask = ~np.isnan(pH_values)
    t = time_days[mask]
    pH = pH_values[mask]

    pH0_guess = pH[0]

    # force pH_inf to be LOWER than observed minimum
    pHinf_guess = np.min(pH) - 0.5

    k_guess = 0.3

    bounds = (
        [0, 0.0001, 0],      # lower bounds
        [pH[0], 5.0, 14]     # upper bounds
    )

    params, _ = curve_fit(
        ph_decay_model,
        t,
        pH,
        p0=[pHinf_guess, k_guess, pH0_guess],
        bounds=bounds,
        maxfev=8000
    )

    return params  # pH_inf, k0, pH0
