import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_validation_metrics(y_true, y_pred):
    """
    Compute RMSE and R2 score
    """

    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))

    y_t = y_true[mask]
    y_p = y_pred[mask]

    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)

    return rmse, r2
