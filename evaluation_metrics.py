"""
evaluation_metrics.py
─────────────────────────────────────────────────────────────────────────────
Per-gap evaluation metrics for the stored imputations
(run_imputation_multiseed.py):

  • univariate:  one row per (unit, gap, column)
  • correlation: one row per (unit, gap, cluster)

  • All metrics are oriented "lower = better":
        AbsVarRatioDev = |Var ratio − 1|,  OneMinusSim = 1 − Sim,
        FSD (0 = equal spread).
  • Run-normalized errors NRMSE_run, NDTW_run: normalized by the standard
    deviation of the whole true reactor series instead of the local gap std.
    RMSE stays in physical units.
  • Correlations only within each cluster, plus the mean |r| of the imputed
    matrix (linear interpolation forces |r| = 1 inside the gap) and the number
    of undefined correlations (flat channels).
  • DTW identical to metrics._dtw_distance, but vectorized per row
    (≈100× faster; equality tested).
"""

import numpy as np
import pandas as pd

import metrics as m_v1   # metric definitions shared with the hyperparameter tuning

# Faulty channels in EXP001/EXP002 R02 (constant or invalid signals), excluded
# from the evaluation and, for FLk-NN/Hankel, from the imputation inputs.
def is_excluded_sensor(exp: str, reactor: str, col: str) -> bool:
    if exp in ("EXP001", "EXP002") and reactor == "R02":
        if "fc(kHz)" in col or "Alpha" in col or "DeltaEps" in col or col.startswith("C("):
            return True
    return False


# ── Metric catalog ────────────────────────────────────────────────────────────
# All values: lower = better. Family → one primary metric per family.
UNIVARIATE_METRICS = {
    # error
    "RMSE":           "error",      # physical units (comparable within a column only)
    "NRMSE_local":    "error",      # RMSE / std of the true values in the gap (factor, not %)
    "NRMSE_run":      "error",      # RMSE / std of the whole true reactor series
    "NMAE":           "error",      # MAE / range of the true values in the gap
    "MAPE":           "error",
    # variance
    "AbsVarRatioDev": "variance",   # |Var(pred)/Var(true) − 1|
    "FSD":            "variance",   # 2·|sd_pred − sd_true| / (sd_pred + sd_true)
    # shape / dynamics
    "NDTW_local":     "shape",      # DTW / (n · local std)
    "NDTW_run":       "shape",      # DTW / (n · std of the whole reactor series)
    "OneMinusSim":    "shape",
    "NRMSD_Deriv1":   "shape",
    "NRMSD_Deriv2":   "shape",
}
CORRELATION_METRICS = {
    "Frob_Abs":       "correlation",
    "Frob_Rel":       "correlation",
    "MAD_OffDiag":    "correlation",
    "MaxAD_OffDiag":  "correlation",
}

# One primary metric per family, chosen from the measured redundancy
# (metric_redundancy_spearman.csv):
#   • NRMSE_run / NDTW_run instead of the local variants: same ranking within a
#     gap × column (constant factor), but not inflated in quiet phases.
#   • FSD instead of |Var ratio − 1| (ρ = 0.95): FSD is symmetric and bounded [0, 2];
#     |VR − 1| penalizes overestimated spread more than underestimated spread.
#   • MAD_OffDiag: in 2-channel clusters all four correlation metrics coincide;
#     MAD is the easiest to interpret. Descriptive only (|r| = 1 for Linear).
PRIMARY_METRICS = {
    "error":       "NRMSE_run",
    "shape":       "NDTW_run",
    "variance":    "FSD",
    "correlation": "MAD_OffDiag",
}
# Sensitivity analysis "smoothed reference": the primary metrics against the
# smoothed true series instead of the noisy raw values (imputations unchanged).
# Checks whether the ranking is an artifact of measurement noise, which mainly
# affects FSD and correlations (smooth reconstructions vs. noisy truth).
SMOOTH_REF_WINDOW = 100             # points (≈ 50 min), as the smoothed baselines
SMOOTH_METRICS = {                  # variant → primary raw-data metric
    "NRMSE_run_smooth":   "NRMSE_run",
    "NDTW_run_smooth":    "NDTW_run",
    "FSD_smooth":         "FSD",
    "MAD_OffDiag_smooth": "MAD_OffDiag",
}
# NoiseFloor_run = RMS(true − smoothed true) / run std: noise share no method
# can reproduce (descriptive).


def smooth_reference(X: pd.DataFrame) -> pd.DataFrame:
    """Smoothed reference: centered rolling median over the whole true series."""
    return X.rolling(SMOOTH_REF_WINDOW, center=True, min_periods=1).median()


# Reported in addition (not in the method comparison):
#   NRMSE_local → share ≥ 1 ("not better than the local mean")
#   RMSE        → physical units for key variables
# Kept in the records but not reported (redundant or not interpretable):
#   NMAE, OneMinusSim (ρ = 1.00 with NMAE), MAPE, AbsVarRatioDev, NDTW_local,
#   NRMSD_Deriv1/2 (dominated by measurement noise at 30-s sampling; Deriv2 does not
#   separate the methods without FLk-NN, Kendall's W ≈ 0.001), Frob_Abs, Frob_Rel,
#   MaxAD_OffDiag.


# ── Fast exact DTW ───────────────────────────────────────────────────────
def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    DTW with local cost |x_i − y_j|, identical to metrics._dtw_distance.

    Row recursion acc[i, j] = c_ij + min(acc[i−1, j−1], acc[i−1, j], acc[i, j−1]).
    With b_j = c_ij + min(acc[i−1, j−1], acc[i−1, j]) and S_j = Σ_{l≤j} c_il:
        acc[i, j] = min(b_j, c_ij + acc[i, j−1]) = S_j + min_{k≤j}(b_k − S_k),
    i.e. each row is a cumulative minimum (vectorized).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    cur = np.empty(m + 1)
    cur[0] = np.inf
    for xi in x:
        c = np.abs(xi - y)
        b = c + np.minimum(prev[:-1], prev[1:])
        s = np.cumsum(c)
        cur[1:] = s + np.minimum.accumulate(b - s)
        prev, cur = cur, prev
        cur[0] = np.inf
    return float(prev[m])


def _norm_dtw(raw: float, n: int, scale: float, y_true: np.ndarray) -> float:
    """Normalization as in metrics._ndtw, with a free scale."""
    if n * scale > 1e-6:
        return raw / (n * scale)
    mean_abs = float(np.mean(np.abs(y_true)))
    return raw / (n * mean_abs) if n * mean_abs > 1e-6 else raw / n


# ── Univariate metrics of one gap ───────────────────────────────────────────
def univariate_metrics(y_true: np.ndarray, y_pred: np.ndarray, run_std: float,
                       y_true_smooth: np.ndarray | None = None) -> dict:
    """
    All univariate metrics for one column in one gap (lower = better).
    With y_true_smooth, also the smoothed-reference variants (SMOOTH_METRICS)
    and the noise floor NoiseFloor_run.
    """
    rmse = m_v1._rmse(y_true, y_pred)
    vr = m_v1._var_ratio(y_true, y_pred)
    sim = m_v1._sim(y_true, y_pred)
    raw_dtw = dtw_distance(y_true, y_pred)
    n = len(y_true)
    local_std = float(np.std(y_true))
    smooth = {}
    if y_true_smooth is not None:
        ok = run_std > 1e-12
        smooth = {
            "NRMSE_run_smooth": m_v1._rmse(y_true_smooth, y_pred) / run_std if ok else np.nan,
            "NDTW_run_smooth":  _norm_dtw(dtw_distance(y_true_smooth, y_pred), n, run_std, y_true_smooth),
            "FSD_smooth":       m_v1._fsd(y_true_smooth, y_pred),
            "NoiseFloor_run":   m_v1._rmse(y_true, y_true_smooth) / run_std if ok else np.nan,
        }
    return {
        **smooth,
        "RMSE":           rmse,
        "NRMSE_local":    m_v1._nrmse(y_true, y_pred, None),
        "NRMSE_run":      rmse / run_std if run_std > 1e-12 else np.nan,
        "NMAE":           m_v1._nmae(y_true, y_pred, None),
        "MAPE":           m_v1._mape(y_true, y_pred),
        "AbsVarRatioDev": abs(vr - 1.0) if np.isfinite(vr) else np.nan,
        "FSD":            m_v1._fsd(y_true, y_pred),
        "NDTW_local":     _norm_dtw(raw_dtw, n, local_std, y_true),
        "NDTW_run":       _norm_dtw(raw_dtw, n, run_std, y_true),
        "OneMinusSim":    1.0 - sim if np.isfinite(sim) else np.nan,
        "NRMSD_Deriv1":   m_v1._nrmsd_deriv(y_true, y_pred, order=1),
        "NRMSD_Deriv2":   m_v1._nrmsd_deriv(y_true, y_pred, order=2),
    }


# ── Correlation metrics of one gap × cluster ────────────────────────────────
def _corr(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Pearson matrix (undefined → 0, diagonal 1) and number of undefined pairs."""
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X, rowvar=False)
    C = np.atleast_2d(C)
    off = ~np.eye(C.shape[0], dtype=bool)
    n_undef = int(np.isnan(C[off]).sum() // 2)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C, n_undef


def correlation_metrics(X_true: np.ndarray, X_pred: np.ndarray,
                        X_true_smooth: np.ndarray | None = None) -> dict:
    """
    Deviation of the correlation matrices (n_rows × n_cols) within one cluster.
    With X_true_smooth, also MAD_OffDiag_smooth and MeanAbsR_true_smooth.
    """
    C_true, und_true = _corr(X_true)
    C_pred, und_pred = _corr(X_pred)
    D = C_pred - C_true
    off = ~np.eye(D.shape[0], dtype=bool)
    frob_true = float(np.linalg.norm(C_true, "fro"))
    frob_abs = float(np.linalg.norm(D, "fro"))
    smooth = {}
    if X_true_smooth is not None:
        C_s, und_s = _corr(X_true_smooth)
        smooth = {"MAD_OffDiag_smooth": float(np.mean(np.abs(C_pred - C_s)[off])),
                  "MeanAbsR_true_smooth": float(np.mean(np.abs(C_s[off]))),
                  "n_undefined_true_smooth": und_s}
    return {
        **smooth,
        "Frob_Abs":        frob_abs,
        "Frob_Rel":        frob_abs / frob_true if frob_true > 1e-10 else np.nan,
        "MAD_OffDiag":     float(np.mean(np.abs(D[off]))),
        "MaxAD_OffDiag":   float(np.max(np.abs(D[off]))),
        "MeanAbsR_true":   float(np.mean(np.abs(C_true[off]))),
        "MeanAbsR_pred":   float(np.mean(np.abs(C_pred[off]))),
        "n_undefined_true": und_true,
        "n_undefined_pred": und_pred,
    }


# ── Process phase ─────────────────────────────────────────────────────────────
PHASE_SIGNAL = "Permittivity(pF/cm)-VCD"   # proxy for viable biomass
PHASE_SMOOTH_POINTS = 240                   # 2 h at 30 s sampling
PHASE_EARLY_FRACTION = 0.2                  # end of early phase: 20 % of the rise (start → maximum)
PHASE_PLATEAU_FRACTION = 0.9                # plateau: 90 % of the rise first reached
PHASE_REFERENCE_QUANTILE = 0.95             # reference "maximum" (robust to end spikes)
PHASES = ["early", "exponential", "stationary/decline"]


def phase_boundaries(X_true: pd.DataFrame) -> dict:
    """
    Phase boundaries of a cultivation from the smoothed permittivity (proxy for
    viable biomass): first time the smoothed signal reaches a share of the rise
    (start value → maximum):
      early               – < 20 % of the rise
      exponential         – 20–90 %
      stationary/decline  – from the first time 90 % is reached
    The "maximum" is the 95th percentile of the smoothed signal, because several
    runs show a transient spike shortly before the end (~185–195 h).
    Returns hours since the start.
    """
    s = X_true[PHASE_SIGNAL]
    smooth = np.asarray(s.rolling(PHASE_SMOOTH_POINTS, center=True, min_periods=1).median()
                        .rolling(PHASE_SMOOTH_POINTS, center=True, min_periods=1).mean(), dtype=float)
    t_h = (X_true.index - X_true.index[0]).total_seconds().values / 3600
    start = float(np.median(smooth[:PHASE_SMOOTH_POINTS]))
    rise = float(np.quantile(smooth, PHASE_REFERENCE_QUANTILE)) - start

    def first_reach(fraction: float) -> float:
        return float(t_h[np.flatnonzero(smooth >= start + fraction * rise)[0]])

    return {"t_early_end_h": first_reach(PHASE_EARLY_FRACTION),
            "t_plateau_h": first_reach(PHASE_PLATEAU_FRACTION),
            "t_end_h": float(t_h[-1])}


def assign_phase(mid_h: float, bounds: dict) -> str:
    """Phase of a gap from its temporal midpoint."""
    if mid_h < bounds["t_early_end_h"]:
        return "early"
    if mid_h < bounds["t_plateau_h"]:
        return "exponential"
    return "stationary/decline"
