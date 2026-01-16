import numpy as np
from src.twin_model import compute_k_eff


def predict_future_ph_state(
    time_days,
    pH_obs,
    params,
    OD,
    T,
    PD,
    future_days=10,
    dt=0.1
):
    """
    State-aware pH prediction with bio-kinetic effects
    """

    pH_inf, k0, pH0 = params

    t_last = time_days.max()
    pH_last = pH_obs[-1]

    # effective kinetic constant
    k_eff = compute_k_eff(k0, OD, T, PD)

    t_future = np.arange(0, future_days + dt, dt)
    pH_future = pH_inf + (pH_last - pH_inf) * np.exp(-k_eff * t_future)

    return t_last + t_future, pH_future
