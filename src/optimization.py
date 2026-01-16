import numpy as np
from src.prediction_model import predict_future_ph_state
from src.recovery_model import predict_metal_recovery_dynamic


def optimize_operating_conditions(
    t_obs, pH_obs, params, OD,
    T_range, PD_range, future_days
):
    """
    Grid search optimization to maximize final recovery
    """

    best_score = -1
    best_T = None
    best_PD = None

    results = []

    for T in T_range:
        for PD in PD_range:

            t_pred, pH_pred = predict_future_ph_state(
                t_obs, pH_obs, params, OD, T, PD, future_days
            )

            rec = predict_metal_recovery_dynamic(t_pred, pH_pred, OD)

            final_rec = rec[-1]

            results.append((T, PD, final_rec))

            if final_rec > best_score:
                best_score = final_rec
                best_T = T
                best_PD = PD

    return best_T, best_PD, best_score, results
