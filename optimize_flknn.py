"""
optimize_flknn.py
─────────────────────────────────────────────────────────────────────────────
Bayesian optimization (Optuna, TPE) of the FLkNNImputer on the tuning
experiment EXP007 (simulate_gaps.py, seed 42).

  • Each sensor cluster (cluster_assignments.csv) is optimized separately for
    each gap count; the imputer only sees the columns of that cluster.
  • Metrics are averaged over both reactors.
  • Objective: NRMSE + NDTW (minimize; implemented as 0.5·NRMSE + 0.5·NDTW).
    RMSE, MAPE, Var_Ratio, Sim, NMAE and FSD are logged as well.

Search space:
  • k          – number of nearest neighbors
  • num_lags   – number of lag matrices
  • max_lag    – maximum lag in samples, [max gap length + 60, + 600]
  • fd_percent – share of retained Fourier coefficients (%)

Output (OUT_DIR):
  {n}_gaps/Cluster_{id}/optimization_results.csv   all trials
  {n}_gaps/Cluster_{id}/best_params.csv            best trial per metric
  summary_best_params.csv
"""

import os
import pickle
import time
import warnings
import gc
import numpy as np
import pandas as pd
import optuna

from impute_methods import FLkNNImputer


def _csv_name_to_col(name: str) -> str:
    """
    Convert a cluster_assignments.csv feature name to the actual DataFrame column name.
    CSV format:  'Feature Name [MODULE]'  (spaces, brackets)
    Pickle format: 'Feature Name-MODULE'  (dash, no brackets)
    Also strips newlines that may appear in the raw correlation_matrix.csv source.
    """
    name = name.replace("\n", " ")
    if " [" in name and name.endswith("]"):
        base, module = name.rsplit(" [", 1)
        return f"{base}-{module[:-1]}"
    return name


def _load_cluster_map(csv_path: str) -> dict:
    """
    Returns {cluster_id: [list of column names]} from cluster_assignments.csv.
    Feature names are converted to the DataFrame column format used in the pickle files
    (e.g. 'CO2(%-sat) [CO2]' → 'CO2(%-sat)-CO2').
    """
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        result.setdefault(cid, []).append(_csv_name_to_col(row["feature"]))
    return result

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

import paths
SIM_DIR      = str(paths.SIM_DIR)
OUT_DIR      = str(paths.FLKNN_OPT_DIR)
CLUSTER_CSV  = str(paths.CLUSTER_CSV)
EXP_NAME     = "EXP007"

GAP_COUNTS = [1, 2, 3, 4]
N_JOBS = -1                   # parallel trials (-1 = all CPU cores)
N_TRIALS = 50

# ── Search space (min, max) ───────────────────────────────────────────────────
PARAM_BOUNDS = {
    "k":          (1, 15),
    "num_lags":   (1, 5),
    "fd_percent": (1, 50),
}
# max_lag is bounded by the maximum gap length (see objective).

RANDOM_STATE = 42   # candidate sampling in FLkNNImputer

# ══════════════════════════════════════════════════════════════════════════════


# ── Metrics ────────────────────────────────────────────────────

from metrics import evaluate_imputation


# ── Combined score (logged only; not the Optuna objective) ───────────────────

def combined_score(nrmse, var_ratio, ndtw):
    """NRMSE + |Var_Ratio − 1| + NDTW (lower = better)."""
    terms = []
    if not np.isnan(nrmse):
        terms.append(nrmse)
    if not np.isnan(var_ratio):
        terms.append(abs(var_ratio - 1.0))
    if not np.isnan(ndtw):
        terms.append(ndtw)
    return float(np.sum(terms)) if terms else np.nan

# ── Evaluation of one parameter combination ──

def _evaluate_combo(params: dict, datasets: list, trial_number: int = None) -> dict:
    """
    Evaluate one parameter combination on all datasets.
    Returns a dict with the parameters, mean metrics and error count.
    """
    imputer = FLkNNImputer(
        k                          = params["k"],
        num_lags                   = params["num_lags"],
        max_lag                    = params["max_lag"],
        fourier_descriptor_percent = params["fd_percent"],
        random_state               = RANDOM_STATE,
    )

    all_metrics = []
    errors = 0
    total_ds = len(datasets)
    print_interval = max(1, total_ds // 5)
    for i, ds in enumerate(datasets):
        if trial_number is not None and (i + 1) % print_interval == 0:
            print(f"      [Trial {trial_number}] dataset {i+1}/{total_ds} done...", flush=True)

        X_imp = None
        try:
            X_imp = imputer.impute(ds["X_missing"])
            m = evaluate_imputation(ds["X_true"], X_imp, ds["mask"])
            if not np.isnan(m["RMSE"]):
                all_metrics.append(m)
        except Exception:
            errors = errors + 1
        finally:
            if X_imp is not None:
                del X_imp
            gc.collect()

    if all_metrics:
        mean_rmse      = float(np.nanmean([m["RMSE"]      for m in all_metrics]))
        mean_nrmse     = float(np.nanmean([m["NRMSE"]     for m in all_metrics]))
        mean_mape      = float(np.nanmean([m["MAPE"]      for m in all_metrics]))
        mean_var_ratio = float(np.nanmean([m["Var_Ratio"] for m in all_metrics]))
        mean_sim       = float(np.nanmean([m["Sim"]       for m in all_metrics]))
        mean_nmae      = float(np.nanmean([m["NMAE"]      for m in all_metrics]))
        mean_fsd       = float(np.nanmean([m["FSD"]       for m in all_metrics]))
        mean_ndtw      = float(np.nanmean([m["NDTW"]      for m in all_metrics]))
    else:
        mean_rmse = mean_nrmse = mean_mape = mean_var_ratio = np.nan
        mean_sim  = mean_nmae  = mean_fsd  = mean_ndtw      = np.nan

    score = combined_score(mean_nrmse, mean_var_ratio, mean_ndtw)

    return {
        **params,
        "RMSE":       mean_rmse,
        "NRMSE":      mean_nrmse,
        "MAPE":       mean_mape,
        "Var_Ratio":  mean_var_ratio,
        "Sim":        mean_sim,
        "NMAE":       mean_nmae,
        "FSD":        mean_fsd,
        "NDTW":       mean_ndtw,
        "Score":      score,
        "Errors":     errors,
        "N_Datasets": len(all_metrics),
    }


# ── Optimization ─────────────────────────────────────────────────────────────

def _get_max_gap_length(datasets: list) -> int:
    max_len = 0
    for ds in datasets:
        for col in ds["mask"].columns:
            is_gap = ds["mask"][col].values
            if not is_gap.any():
                continue
            padded = np.pad(is_gap.astype(int), (1, 1), mode='constant')
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            if len(starts) > 0:
                max_len = max(max_len, np.max(ends - starts))
    return int(max_len)

def optimize_for_gap_count(n_gaps: int, datasets: list) -> pd.DataFrame:
    """
    Optuna study for one gap count (n_gaps); trials run on N_JOBS cores.
    Returns (DataFrame of all completed trials with mean metrics, study).
    """

    if not datasets:
        print(f"  [WARNING] No data for {n_gaps} gaps.")
        return pd.DataFrame()

    print(f"  → {len(datasets)} datasets (reactors) loaded.")

    gap_length = _get_max_gap_length(datasets)
    print(f"  → Maximum gap length = {gap_length}")

    # ── Bayesian Optimization (Optuna) ────────────────────────────────────────
    print(f"  → Bayesian optimization: {N_TRIALS} trials on {N_JOBS} cores …")

    def objective(trial):
        print(f"    -> [Trial {trial.number}] started...", flush=True)
        params = {
            "k": trial.suggest_int("k", PARAM_BOUNDS["k"][0], PARAM_BOUNDS["k"][1]),
            "num_lags": trial.suggest_int("num_lags", PARAM_BOUNDS["num_lags"][0], PARAM_BOUNDS["num_lags"][1]),
            "max_lag": trial.suggest_int("max_lag", gap_length + 60, gap_length + 600),
            "fd_percent": trial.suggest_int("fd_percent", PARAM_BOUNDS["fd_percent"][0], PARAM_BOUNDS["fd_percent"][1]),
        }
        res = _evaluate_combo(params, datasets, trial_number=trial.number)

        # metrics as trial user attributes
        for key, val in res.items():
            if key not in params:
                trial.set_user_attr(key, val)

        # objective: NRMSE + NDTW
        nrmse = res.get("NRMSE", float('inf'))
        ndtw  = res.get("NDTW",  float('inf'))

        del res
        gc.collect()

        if np.isnan(nrmse) or np.isnan(ndtw):
            return float('inf')
        return 0.5*nrmse + 0.5*ndtw

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")

    t_start = time.time()

    def optuna_callback(study, trial):
        elapsed = time.time() - t_start
        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        if completed > 0:
            avg_time = elapsed / completed
            eta_min = (avg_time * (N_TRIALS - completed)) / 60
            try:
                best_val = study.best_value
            except ValueError:
                best_val = float('inf')

            t_val = trial.value if trial.value is not None else float('inf')
            print(f"  [Progress] trial {completed}/{N_TRIALS} done | "
                  f"NRMSE+NDTW (Trial {trial.number}): {t_val:.4f} | "
                  f"best NRMSE+NDTW: {best_val:.4f} | "
                  f"elapsed: {elapsed/60:.1f} m | ETA: {eta_min:.1f} m", flush=True)

    print(f"  → Starting optimization (update after each trial)...", flush=True)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, callbacks=[optuna_callback])
    elapsed = time.time() - t_start
    print(f"  → Done in {elapsed:.1f}s ({elapsed/max(1, len(study.trials)):.1f}s per trial)")

    # ── Collect results ────────────────────────────────────────────────
    results = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = t.params.copy()
            row.update(t.user_attrs)
            row["Trial"] = t.number
            results.append(row)

    df = pd.DataFrame(results)

    # ── Compact overview ────────────────────────────────────────────
    total_complete = len(df)
    for i, row in df.iterrows():
        print(f"  [{i+1:>3}/{total_complete}]  "
              f"k={int(row['k']):>2}  num_lags={int(row['num_lags'])}  "
              f"max_lag={int(row['max_lag']):>3}  fd={int(row['fd_percent']):>3}%  →  "
              f"NRMSE={row['NRMSE']:.4f}  NDTW={row['NDTW']:.4f}  "
              f"MAPE={row['MAPE']:.4f}  VarR={row['Var_Ratio']:.3f}  "
              f"Sim={row['Sim']:.4f}  Score={row['Score']:.4f}")

    return df, study


def print_best(df: pd.DataFrame, n_gaps: int, cluster_id: int = None):
    """Print and return the best configuration per metric."""
    metrics = {
        "RMSE":      ("min",          "RMSE"),
        "NRMSE":     ("min",          "NRMSE"),
        "MAPE":      ("min",          "MAPE"),
        "Var_Ratio": ("closest_to_1", "Var_Ratio"),
        "Sim":       ("max",          "Sim"),
        "NMAE":      ("min",          "NMAE"),
        "FSD":       ("min",          "FSD"),
        "NDTW":      ("min",          "NDTW"),
        "Score":     ("min",          "Score"),
    }
    cluster_str = f" | Cluster {cluster_id}" if cluster_id is not None else ""
    print(f"\n{'─'*70}")
    print(f"  BEST CONFIGURATIONS for {n_gaps} gap(s){cluster_str}")
    print(f"{'─'*70}")

    best_rows = []
    for metric_name, (mode, col) in metrics.items():
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        if mode == "min":
            best_idx = sub[col].idxmin()
        elif mode == "max":
            best_idx = sub[col].idxmax()
        else:  # closest_to_1
            best_idx = (sub[col] - 1.0).abs().idxmin()

        best_row = sub.loc[best_idx]
        best_rows.append({
            "Optimized_for": metric_name,
            "k":         int(best_row["k"]),
            "num_lags":  int(best_row["num_lags"]),
            "max_lag":   int(best_row["max_lag"]),
            "fd_percent":int(best_row["fd_percent"]),
            "RMSE":      round(best_row["RMSE"],      5),
            "NRMSE":     round(best_row["NRMSE"],     5),
            "MAPE":      round(best_row["MAPE"],      5),
            "Var_Ratio": round(best_row["Var_Ratio"], 5),
            "Sim":       round(best_row["Sim"],       5),
            "NMAE":      round(best_row["NMAE"],      5),
            "FSD":       round(best_row["FSD"],       5),
            "NDTW":      round(best_row["NDTW"],      5),
            "Score":     round(best_row["Score"],     5),
        })

    df_best = pd.DataFrame(best_rows)
    print(df_best.to_string(index=False))
    return df_best


def _save_optuna_plots(study, out_dir: str):
    """Saves optimization history and slice plot for an Optuna study."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history as mpl_opt_history,
        plot_slice as mpl_slice,
    )

    try:
        ax = mpl_opt_history(study)
        fig = ax.get_figure()
        path = os.path.join(out_dir, "optimization_history.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved → {path}")
    except Exception as e:
        print(f"  [WARNING] Optimization history plot failed: {e}")

    try:
        axes = mpl_slice(study)
        fig = (axes.flatten()[0] if hasattr(axes, "flatten") else axes).get_figure()
        path = os.path.join(out_dir, "slice_plot.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved → {path}")
    except Exception as e:
        print(f"  [WARNING] Slice plot failed: {e}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Cluster assignment ─────────────────────────────────────────────
    cluster_map = _load_cluster_map(CLUSTER_CSV)
    cluster_ids = sorted(cluster_map.keys())
    print(f"\n  {len(cluster_ids)} clusters loaded from {CLUSTER_CSV}")
    for cid in cluster_ids:
        print(f"    Cluster {cid}: {len(cluster_map[cid])} Features")

    all_best = []

    for n_gaps in GAP_COUNTS:
        pkl_path = os.path.join(SIM_DIR, f"simulated_data_{n_gaps}_gaps.pkl")
        if not os.path.exists(pkl_path):
            print(f"\n[SKIP] File not found: {pkl_path}")
            continue

        print(f"\n{'═'*70}")
        print(f"  GAP COUNT: {n_gaps}")
        print(f"{'═'*70}")

        print(f"  Loading {pkl_path} …")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if EXP_NAME not in data:
            print(f"  [SKIP] {EXP_NAME} not in the pickle file.")
            del data
            gc.collect()
            continue

        datasets = []
        for reactor_name, enc in data[EXP_NAME].items():
            datasets.append({
                "reactor_name": reactor_name,
                "X_missing":  enc["X_missing"].copy(),
                "X_true":     enc["X_true"].copy(),
                "mask":       enc["mask"].copy(),
            })

        del data
        gc.collect()

        # ── Clusters ──────────────────────────────────────────────────
        for cluster_id in cluster_ids:
            cluster_features = cluster_map[cluster_id]

            print(f"\n  {'─'*65}")
            print(f"  CLUSTER {cluster_id} ({len(cluster_features)} Features)")
            print(f"  Features: {', '.join(cluster_features)}")
            print(f"  {'─'*65}")

            # restrict to the cluster columns
            cluster_datasets = []
            for ds in datasets:
                available_cols = [c for c in cluster_features if c in ds["X_missing"].columns]
                if not available_cols:
                    continue
                cluster_datasets.append({
                    "reactor_name": ds["reactor_name"],
                    "X_missing":   ds["X_missing"][available_cols].copy(),
                    "X_true":      ds["X_true"][available_cols].copy(),
                    "mask":        ds["mask"][available_cols].copy(),
                })

            if not cluster_datasets:
                print(f"  [SKIP] No datasets with cluster {cluster_id} columns.")
                continue

            df_results, study = optimize_for_gap_count(n_gaps, cluster_datasets)

            if df_results.empty:
                continue

            out_dir_cluster = os.path.join(OUT_DIR, f"{n_gaps}_gaps", f"Cluster_{cluster_id}")
            os.makedirs(out_dir_cluster, exist_ok=True)

            optuna_csv = os.path.join(out_dir_cluster, "optimization_results.csv")
            df_results.to_csv(optuna_csv, index=False)
            print(f"\n  All results saved → {optuna_csv}")

            _save_optuna_plots(study, out_dir_cluster)

            df_best = print_best(df_results, n_gaps, cluster_id=cluster_id)
            df_best.insert(0, "Cluster_ID", cluster_id)
            df_best.insert(0, "N_Gaps", n_gaps)

            best_csv = os.path.join(out_dir_cluster, "best_params.csv")
            df_best.to_csv(best_csv, index=False)
            print(f"  Best parameters saved → {best_csv}")

            all_best.append(df_best)

    # ── Summary ────────────────────────────────────────────────
    if all_best:
        df_summary = pd.concat(all_best, ignore_index=True)
        summary_path = os.path.join(OUT_DIR, "summary_best_params.csv")
        df_summary.to_csv(summary_path, index=False)

        print(f"\n{'═'*70}")
        print(f"  SUMMARY: best parameters for all gap counts and clusters")
        print(f"{'═'*70}")
        print(df_summary.to_string(index=False))
        print(f"\n  Summary saved → {summary_path}")
    else:
        print("\nNo results – check the pickle files and EXP_NAME.")


if __name__ == "__main__":
    main()
