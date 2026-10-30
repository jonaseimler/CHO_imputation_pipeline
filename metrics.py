"""
metrics.py
─────────────────────────────────────────────────────────────────────────────
Imputation metrics used for the hyperparameter tuning (optimize_*.py) and as
building blocks of evaluation_metrics.py.
"""

import numpy as np
import pandas as pd

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray,
           full_series: np.ndarray) -> float:
    """
    NRMSE scaled by the local variation within the gap (not the std of the
    full series), so large local offsets are not masked.
    """
    local_std = float(np.std(y_true))
    rmse = _rmse(y_true, y_pred)
    if local_std > 1e-6:
        return rmse / local_std

    # Fallback to mean if the piece is almost completely flat
    local_mean = float(np.mean(np.abs(y_true)))
    return rmse / local_mean if local_mean > 1e-6 else rmse


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-10
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _var_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vt = float(np.var(y_true))
    vp = float(np.var(y_pred))
    return vp / vt if vt > 1e-10 else np.nan


def _sim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    T = len(y_true)
    if T == 0:
        return np.nan
    range_x = float(np.max(y_true) - np.min(y_true))
    if range_x < 1e-10:
        return float(np.mean(1.0 / (1.0 + np.abs(y_true - y_pred))))
    return float(np.mean(1.0 / (1.0 + np.abs(y_true - y_pred) / range_x)))


def _nmae(y_true: np.ndarray, y_pred: np.ndarray,
          full_series: np.ndarray) -> float:
    # Changed to local boundary normalization to match NRMSE fixes
    vmax = float(np.max(y_true))
    vmin = float(np.min(y_true))
    denom = vmax - vmin
    if denom < 1e-10:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def _fsd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sd_true = float(np.std(y_true))
    sd_pred = float(np.std(y_pred))
    denom = sd_true + sd_pred
    if denom < 1e-10:
        return np.nan
    return float(2.0 * abs(sd_pred - sd_true) / denom)


def _dtw_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard DTW distance between two equal- or unequal-length 1-D sequences,
    using absolute difference as the local cost.  O(n·m) DP.
    """
    n, m = len(y_true), len(y_pred)
    cost = np.abs(y_true[:, None] - y_pred[None, :])   # (n, m)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j],
                                                   acc[i, j - 1],
                                                   acc[i - 1, j - 1])
    return float(acc[n, m])


def _ndtw(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized DTW: DTW(y_true, y_pred) / (n * local_std).
    Dimensionless and comparable across columns and scales,
    analogous to NRMSE.  Lower is better; 0 = perfect match.
    """
    n = len(y_true)
    if n == 0:
        return np.nan
    raw = _dtw_distance(y_true, y_pred)
    local_std = float(np.std(y_true))
    denom = n * local_std
    if denom > 1e-6:
        return raw / denom
    # Fallback: normalize by n * mean magnitude (flat reference signal)
    local_mean = float(np.mean(np.abs(y_true)))
    denom2 = n * local_mean
    return raw / denom2 if denom2 > 1e-6 else raw / n


def _nrmsd_deriv(y_true: np.ndarray, y_pred: np.ndarray, order: int = 1) -> float:
    """
    Normalized RMSD of the first (order=1) or second (order=2) derivative.

    NRMSDderiv = sqrt(1/n * sum((dy_true_i - dy_pred_i)^2)) / (max(dy_true) - min(dy_true))

    Uses numpy.gradient for central-difference numeric differentiation.
    Normalization by the range of the true derivative makes the metric
    dimensionless and comparable across sensors with different magnitudes
    and sampling rates.

    - order=1: evaluates whether the imputer preserves physiological
      *rates of change* (biological dynamics).
    - order=2: evaluates curvature and smoothness of the reconstruction
      (noise / oscillation fidelity).
    """
    if len(y_true) < order + 1:
        return np.nan

    dy_true = y_true.astype(float).copy()
    dy_pred = y_pred.astype(float).copy()
    for _ in range(order):
        dy_true = np.gradient(dy_true)
        dy_pred = np.gradient(dy_pred)

    rmsd = float(np.sqrt(np.mean((dy_true - dy_pred) ** 2)))

    true_range = float(np.max(dy_true) - np.min(dy_true))
    if true_range > 1e-10:
        return rmsd / true_range

    # Fallback: normalize by mean absolute magnitude of the true derivative
    mean_abs = float(np.mean(np.abs(dy_true)))
    return rmsd / mean_abs if mean_abs > 1e-10 else rmsd


def evaluate_imputation(X_true: pd.DataFrame,
                        X_imputed: pd.DataFrame,
                        mask: pd.DataFrame) -> dict:
    """
    Computes all metrics over all columns that have gaps.
    Returns the average metrics across all affected columns.
    """
    rmses, nrmses, mapes, var_ratios, sims, nmaes, fsds, ndtws = [], [], [], [], [], [], [], []
    nrmsd_d1s, nrmsd_d2s = [], []

    for col in X_true.columns:
        col_mask = mask[col].values
        if col_mask.sum() == 0:
            continue

        yt = X_true[col].values[col_mask]
        yp = X_imputed[col].values[col_mask]
        full = X_true[col].values

        if np.isnan(yp).any():
            continue  # Skip failed imputations

        rmses.append(_rmse(yt, yp))
        nrmses.append(_nrmse(yt, yp, full))
        mapes.append(_mape(yt, yp))
        var_ratios.append(_var_ratio(yt, yp))
        sims.append(_sim(yt, yp))
        nmaes.append(_nmae(yt, yp, full))
        fsds.append(_fsd(yt, yp))
        ndtws.append(_ndtw(yt, yp))
        nrmsd_d1s.append(_nrmsd_deriv(yt, yp, order=1))
        nrmsd_d2s.append(_nrmsd_deriv(yt, yp, order=2))

    if not rmses:
        return {"RMSE": np.nan, "NRMSE": np.nan, "MAPE": np.nan,
                "Var_Ratio": np.nan, "Sim": np.nan, "NMAE": np.nan,
                "FSD": np.nan, "NDTW": np.nan,
                "NRMSD_Deriv1": np.nan, "NRMSD_Deriv2": np.nan}

    return {
        "RMSE":          float(np.nanmean(rmses)),
        "NRMSE":         float(np.nanmean(nrmses)),
        "MAPE":          float(np.nanmean(mapes)),
        "Var_Ratio":     float(np.nanmean(var_ratios)),
        "Sim":           float(np.nanmean(sims)),
        "NMAE":          float(np.nanmean(nmaes)),
        "FSD":           float(np.nanmean(fsds)),
        "NDTW":          float(np.nanmean(ndtws)),
        "NRMSD_Deriv1":  float(np.nanmean(nrmsd_d1s)),
        "NRMSD_Deriv2":  float(np.nanmean(nrmsd_d2s)),
    }

