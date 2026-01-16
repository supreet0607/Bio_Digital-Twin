import numpy as np


def predict_metal_recovery_dynamic(
    time_days,
    pH_values,
    OD,
    R_max=95,
    beta=1.2
):
    """
    Recovery driven by acidification rate and biomass
    """

    dpH_dt = np.gradient(pH_values, time_days)
    acid_rate = np.clip(-dpH_dt, 0, None)

    k_eff = beta * OD * acid_rate

    recovery = np.zeros_like(time_days)

    for i in range(1, len(time_days)):
        dt = time_days[i] - time_days[i-1]
        recovery[i] = recovery[i-1] + (R_max - recovery[i-1]) * k_eff[i] * dt

    return np.clip(recovery, 0, R_max)
