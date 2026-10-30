"""
revision_config.py
─────────────────────────────────────────────────────────────────────────────
Shared configuration for imputation, evaluation and downstream analysis.

  • Paths to data, gap positions, optimization logs and results.
  • Method list and display names.
  • Cluster mapping (CSV IDs ↔ manuscript numbering).
  • Optimized hyperparameters, read directly from the Optuna logs
    (optimization_results.csv). There are no fallback values: a missing
    entry raises an error.

Selection rule for all methods: the trial with the minimal objective
NRMSE + NDTW (Optuna best value).
"""

import os
import glob

import pandas as pd

from impute_methods import (
    BaselineImputerLinear, BaselineImputerLinearSmoothed,
    BaselineImputerSpline, BaselineImputerSplineSmoothed,
    FLkNNImputer, HankelImputer,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
import paths

ROOT = str(paths.ROOT)

SIM_DIR       = str(paths.MULTISEED_DIR)
CLUSTER_CSV   = str(paths.CLUSTER_CSV)
FLKNN_OPT_DIR = str(paths.FLKNN_OPT_DIR)
HANKEL_OPT_DIR = str(paths.HANKEL_OPT_DIR)
RESULTS_DIR   = str(paths.IMPUTED_DIR)

# ── Methods ───────────────────────────────────────────────────────────────────
# key = internal name (file names), value = name in the manuscript
METHODS = {
    "Linear":          "Linear Baseline",
    "Linear_Smoothed": "Smoothed Linear Baseline",
    "Spline":          "Spline Baseline",
    "Spline_Smoothed": "Smoothed Spline Baseline",
    "FLk-NN":          "FLk-NN",
    "Hankel":          "Hankel",
}
CLUSTER_WISE_METHODS = {"FLk-NN", "Hankel"}   # all others: column-wise on the full matrix
SMOOTHING_WINDOW = 100                        # samples

# ── Clusters ──────────────────────────────────────────────────────────────────
# cluster_assignments.csv uses other IDs than the manuscript (Sec. 2.6):
#   CSV 1 = 3 temperatures          → manuscript Cluster 4
#   CSV 2 = 23 capacitance/VCD/TCD  → manuscript Cluster 1
#   CSV 3 = conductivity, fc        → manuscript Cluster 2
#   CSV 4 = alpha, CO2              → manuscript Cluster 3
# CSV IDs are used internally (optimization, imputation);
# PAPER_CLUSTER converts them for tables and figures.
PAPER_CLUSTER = {1: 4, 2: 1, 3: 2, 4: 3}


def _csv_name_to_col(name: str) -> str:
    """'Feature [MODULE]' (CSV) → 'Feature-MODULE' (DataFrame column)."""
    name = name.replace("\n", " ")
    if " [" in name and name.endswith("]"):
        base, module = name.rsplit(" [", 1)
        return f"{base}-{module[:-1]}"
    return name


def load_cluster_map(csv_path: str | None = None) -> dict[int, list[str]]:
    """{csv_cluster_id: [column names]}"""
    df = pd.read_csv(csv_path or CLUSTER_CSV)
    result: dict[int, list[str]] = {}
    for _, row in df.iterrows():
        result.setdefault(int(row["cluster"]), []).append(_csv_name_to_col(row["feature"]))
    return result


# ── Hyperparameters from the Optuna logs ──────────────────────────────────────
def _best_trial(csv_path: str) -> pd.Series:
    """Trial with minimal NRMSE + NDTW (Optuna objective) from optimization_results.csv."""
    df = pd.read_csv(csv_path)
    obj = df["NRMSE"] + df["NDTW"]
    if obj.notna().sum() == 0:
        raise ValueError(f"No valid trials in {csv_path}")
    return df.loc[obj.idxmin()]


def _collect(opt_dir: str) -> dict[tuple[int, int], str]:
    """{(n_gaps, cluster_id): path to optimization_results.csv}"""
    found = {}
    for path in glob.glob(os.path.join(opt_dir, "*_gaps", "Cluster_*", "optimization_results.csv")):
        parts = os.path.normpath(path).split(os.sep)
        n_gaps = int(parts[-3].split("_")[0])
        cluster_id = int(parts[-2].split("_")[1])
        found[(n_gaps, cluster_id)] = path
    return found


def load_flknn_params(opt_dir: str | None = None) -> dict[tuple[int, int], dict]:
    """{(n_gaps, cluster_id): FLkNNImputer kwargs}"""
    params = {}
    for key, path in _collect(opt_dir or FLKNN_OPT_DIR).items():
        b = _best_trial(path)
        params[key] = {
            "k": int(b["k"]),
            "num_lags": int(b["num_lags"]),
            "max_lag": int(b["max_lag"]),
            "fourier_descriptor_percent": int(b["fd_percent"]),
        }
    return params


def load_hankel_params(opt_dir: str | None = None) -> dict[tuple[int, int], dict]:
    """{(n_gaps, cluster_id): HankelImputer kwargs} – relative parametrization."""
    params = {}
    for key, path in _collect(opt_dir or HANKEL_OPT_DIR).items():
        b = _best_trial(path)
        params[key] = {
            "lag": None, "max_lag": None, "max_window": None,
            "lag_buffer": int(b["lag_buffer"]),
            "context_factor": float(b["context_factor"]),
            "max_iter": int(b["max_iter"]),
            "tol": float(b["tol"]),
            "rank": int(b["rank"]),
        }
    return params


def check_complete(params: dict, n_gaps_list, cluster_ids, method: str):
    """Raise if parameters are missing for a required (n_gaps, cluster) combination."""
    missing = [(n, c) for n in n_gaps_list for c in cluster_ids if (n, c) not in params]
    if missing:
        raise RuntimeError(f"{method}: no optimized parameters for (n_gaps, cluster) = {missing}")


# ── Imputer factory ───────────────────────────────────────────────────────────
def make_baseline(method: str):
    if method == "Linear":
        return BaselineImputerLinear()
    if method == "Linear_Smoothed":
        return BaselineImputerLinearSmoothed(window_size=SMOOTHING_WINDOW)
    if method == "Spline":
        return BaselineImputerSpline()
    if method == "Spline_Smoothed":
        return BaselineImputerSplineSmoothed(window_size=SMOOTHING_WINDOW)
    raise ValueError(f"Not a baseline method: {method}")


def make_cluster_imputer(method: str, params: dict):
    if method == "FLk-NN":
        return FLkNNImputer(**params)
    if method == "Hankel":
        return HankelImputer(**params)
    raise ValueError(f"Not a cluster-wise method: {method}")
