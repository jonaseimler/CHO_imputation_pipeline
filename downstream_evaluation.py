"""
downstream_evaluation.py
─────────────────────────────────────────────────────────────────────────────
Downstream evaluation: does the imputation method affect subsequent tasks?

Two tasks, both computed from the stored gap values of the multi-seed
imputation (run_imputation_multiseed.py), without re-imputing:

  A) PLS soft sensor for the viable cell density (VCD)
     • Inputs: the 23 variables of manuscript Cluster 1 (capacitance spectrum,
       permittivity, turbidity …); target: offline "Live Cell Conc. Mean"
       (R0x_offline, 12–16 samples per run); inline values = median ±15 min.
     • One pooled model over all valid runs (incl. EXP007 and EXP003/005/006
       outside the real outages); number of components by leave-one-experiment-
       out CV. No LOEO model per test experiment, because spectral soft sensors
       transfer poorly between experiments (probe-specific levels; EXP004:
       R² < 0). This does not bias the method comparison: the model sees no
       imputed data and all methods are scored with the same model.
     • In each gap: VCD prediction from imputed vs. true inputs → VCD error
       (10^6 cells/mL), error of the specific growth rate µ and of the
       integral of viable cells (IVCD) over the gap. The reference is the soft
       sensor on the true signals (offline samples inside gaps are too few).

  B) PCA-based multivariate statistical process monitoring (MSPC)
     • PCA on all 30 variables (autoscaled), calibrated on the complete runs of
       the other experiments; components up to 95 % variance; control limits =
       99th percentile of T² and Q (SPE) of the calibration data.
     • In each gap: T² and Q from imputed vs. true data → log error and share
       of time points with a different alarm state.

  In addition: distance of each method from linear interpolation relative to
  the distance of the truth from linear, RMS(pred − lin) / RMS(true − lin)
  (0 = identical to Linear, 1 = as far from Linear as the truth).

Runs with faulty sensors (EXP001/002 R02: capacitance spectra, fc, alpha,
DeltaEps; evaluation_metrics.is_excluded_sensor) are excluded from PLS and
MSPC (calibration and test) → 4 test runs.

Steps:
  python downstream_evaluation.py records --jobs 3    # per-gap downstream metrics
  python downstream_evaluation.py stats               # method comparisons (as evaluate_multiseed)
  python downstream_evaluation.py real                # PLS on real outages vs. offline samples
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import evaluate_multiseed as ems
import evaluation_metrics as em
import revision_config as cfg
from run_imputation_multiseed import load_unit
from simulate_gaps import DATA_DIR
from simulate_gaps_multiseed import load_scenarios

OUT_DIR = str(cfg.paths.DOWNSTREAM_DIR)
REAL_DIR = str(cfg.paths.REAL_OUTAGE_DIR)
TEST_EXPERIMENTS = ["EXP001", "EXP002", "EXP004"]
ALL_EXPERIMENTS = ["EXP001", "EXP002", "EXP003", "EXP004", "EXP005", "EXP006", "EXP007"]
OFFLINE_TARGET = "Live Cell Conc. Mean"
VCD_UNIT = 1e6                      # cells/mL → 10^6 cells/mL
OFFLINE_WINDOW_MIN = 15             # inline median ±15 min around the sample
PLS_MAX_COMPONENTS = 10
PCA_VARIANCE = 0.95
PCA_SUBSAMPLE = 10                  # every 10th point for the PCA calibration
CONTROL_QUANTILE = 0.99
MU_SMOOTH_H = 2.0                   # smoothing of ln(VCD) before differentiation (h)
VCD_FLOOR = 0.05                    # 10^6 cells/mL, lower bound for ln(VCD)
PLS_CLUSTER_CSV = 2                 # CSV cluster 2 = manuscript Cluster 1 (23 variables)

DOWNSTREAM_METRICS = {              # all: lower = better
    "VCD_RMSE":          "pls",     # 10^6 cells/mL
    "VCD_relRMSE":       "pls",     # RMSE / mean reference VCD in the gap
    "IVCD_relErr":       "pls",     # |ΔIVCD| / IVCD over the gap
    "mu_RMSE":           "pls",     # specific growth rate, 1/h
    "T2_logRMSE":        "mspc",    # RMSE of log10 T²
    "Q_logRMSE":         "mspc",    # RMSE of log10 Q
    "T2_alarm_disagree": "mspc",    # share of time points with an alarm state different from the truth
    "Q_alarm_disagree":  "mspc",
}
PRIMARY_DOWNSTREAM = ["VCD_RMSE", "mu_RMSE", "Q_logRMSE", "T2_logRMSE"]


# ══════════════════════════════════════════════════════════════════════════════
# Data and models
# ══════════════════════════════════════════════════════════════════════════════
def load_offline(exp: str) -> pd.DataFrame:
    xls = pd.ExcelFile(os.path.join(DATA_DIR, f"{exp}_FLAGGED.xlsx"))
    parts = []
    for reactor in ["R01", "R02"]:
        df = pd.read_excel(xls, sheet_name=f"{reactor}_offline")
        parts.append(pd.DataFrame({"experiment": exp, "reactor": reactor, "time_h": df["Time (h)"],
                                   "vcd": df[OFFLINE_TARGET] / VCD_UNIT}).dropna())
    return pd.concat(parts, ignore_index=True)


def inline_runs(clean: dict) -> dict:
    """{(exp, reactor): DataFrame} – EXP003/005/006 with NaN in the real outages (real_outages.py)."""
    runs = {(e, r): df for e in clean for r, df in clean[e].items()}
    for p in glob.glob(os.path.join(REAL_DIR, "imputed", "*.pkl")):
        exp, reactor = os.path.basename(p)[:-4].split("_")
        with open(p, "rb") as f:
            runs[(exp, reactor)] = pickle.load(f)["X_missing"]
    return runs


def run_is_valid(exp: str, reactor: str, columns) -> bool:
    return not any(em.is_excluded_sensor(exp, reactor, c) for c in columns)


def calibration_samples(runs: dict, offline: pd.DataFrame, inputs: list[str]) -> pd.DataFrame:
    """Offline samples with the inline median (±OFFLINE_WINDOW_MIN) of the inputs; samples in outages are dropped."""
    rows = []
    for (exp, reactor), X in runs.items():
        if not run_is_valid(exp, reactor, inputs):
            continue
        t = (X.index - X.index[0]).total_seconds().values / 3600
        for s in offline[(offline.experiment == exp) & (offline.reactor == reactor)].itertuples():
            sel = np.abs(t - s.time_h) <= OFFLINE_WINDOW_MIN / 60
            if not sel.any():
                continue
            x = X[inputs].values[sel]
            if np.isnan(x).any():
                continue
            rows.append({"experiment": exp, "reactor": reactor, "time_h": s.time_h, "vcd": s.vcd,
                         **dict(zip(inputs, np.median(x, axis=0)))})
    return pd.DataFrame(rows)


def fit_pls(samples: pd.DataFrame, inputs: list[str]) -> tuple[PLSRegression, int, pd.DataFrame]:
    """PLS with the number of components from leave-one-experiment-out CV on the calibration data."""
    X, y, groups = samples[inputs].values, samples["vcd"].values, samples["experiment"].values
    cv = []
    for n in range(1, min(PLS_MAX_COMPONENTS, len(inputs)) + 1):
        err = []
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            m = PLSRegression(n_components=n, scale=True).fit(X[tr], y[tr])
            err.append((m.predict(X[te]).ravel() - y[te]) ** 2)
        cv.append({"n_components": n, "rmsecv": float(np.sqrt(np.mean(np.concatenate(err))))})
    cv = pd.DataFrame(cv)
    n_best = int(cv.loc[cv.rmsecv.idxmin(), "n_components"])
    return PLSRegression(n_components=n_best, scale=True).fit(X, y), n_best, cv


class PCAMonitor:
    """Observation-wise PCA-MSPC with empirical control limits."""

    def __init__(self, X_cal: np.ndarray):
        self.scaler = StandardScaler().fit(X_cal)
        Z = self.scaler.transform(X_cal)
        full = PCA().fit(Z)
        self.n_components = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), PCA_VARIANCE) + 1)
        self.pca = PCA(n_components=self.n_components).fit(Z)
        t2, q = self.stats(X_cal)
        self.t2_lim = float(np.quantile(t2, CONTROL_QUANTILE))
        self.q_lim = float(np.quantile(q, CONTROL_QUANTILE))

    def stats(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = self.scaler.transform(X)
        T = self.pca.transform(Z)
        t2 = np.sum(T ** 2 / self.pca.explained_variance_, axis=1)
        q = np.sum((Z - self.pca.inverse_transform(T)) ** 2, axis=1)
        return t2, q


def build_models(clean: dict, out_dir: str) -> dict:
    """Pooled PLS model and, per test experiment, a PCA model calibrated without that experiment."""
    cluster_map = cfg.load_cluster_map()
    inputs = cluster_map[PLS_CLUSTER_CSV]
    all_cols = list(next(iter(clean.values()))["R01"].columns)
    runs = inline_runs(clean)
    offline = pd.concat([load_offline(e) for e in ALL_EXPERIMENTS], ignore_index=True)
    samples = calibration_samples(runs, offline, inputs)

    # PLS: one pooled model over all valid runs (see module docstring); the LOEO CV
    # error is reported for transparency.
    pls, n_comp, cv = fit_pls(samples, inputs)
    fitted = samples.assign(pred=pls.predict(samples[inputs].values).ravel())
    run_r2 = fitted.groupby(["experiment", "reactor"]).apply(
        lambda g: 1 - np.sum((g.pred - g.vcd) ** 2) / np.sum((g.vcd - g.vcd.mean()) ** 2))

    models, calib_rows = {}, []
    for exp in TEST_EXPERIMENTS:
        calib_rows.append({"test_experiment": exp, "pls_n_samples": len(samples),
                           "pls_experiments": "+".join(sorted(samples.experiment.unique())),
                           "pls_components": n_comp, "pls_rmsecv_loeo": float(cv.rmsecv.min()),
                           "pls_r2_pooled": float(1 - np.sum((fitted.pred - fitted.vcd) ** 2)
                                                  / np.sum((fitted.vcd - fitted.vcd.mean()) ** 2)),
                           "pls_r2_test_runs": "; ".join(f"{r}: {v:.2f}" for (e, r), v in run_r2.items() if e == exp)})

        # PCA calibration: complete runs of the other experiments (clean_data holds EXP001/002/004/007)
        X_cal = np.vstack([clean[e][r][all_cols].values[::PCA_SUBSAMPLE] for e in clean for r in clean[e]
                           if e != exp and run_is_valid(e, r, all_cols)])
        mon = PCAMonitor(X_cal)
        calib_rows[-1].update({"pca_components": mon.n_components, "pca_cal_points": len(X_cal),
                               "t2_limit": mon.t2_lim, "q_limit": mon.q_lim})
        models[exp] = {"pls": pls, "pca": mon}

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(os.path.join(out_dir, "model_calibration.csv"), index=False)
    fitted.to_csv(os.path.join(out_dir, "pls_calibration_samples.csv"), index=False)
    cv.to_csv(os.path.join(out_dir, "pls_cv_components.csv"), index=False)
    return {"models": models, "inputs": inputs, "all_cols": all_cols}


# ══════════════════════════════════════════════════════════════════════════════
# Per-gap metrics
# ══════════════════════════════════════════════════════════════════════════════
def _mu(vcd: np.ndarray, t_h: np.ndarray) -> np.ndarray:
    """Specific growth rate d ln(VCD)/dt (1/h) after smoothing over MU_SMOOTH_H."""
    dt = np.median(np.diff(t_h)) if len(t_h) > 1 else 1.0
    w = max(1, int(round(MU_SMOOTH_H / dt)))
    s = pd.Series(np.log(np.maximum(vcd, VCD_FLOOR))).rolling(w, center=True, min_periods=1).mean().values
    return np.gradient(s, t_h) if len(t_h) > 1 else np.zeros_like(s)


def gap_metrics(Xt: np.ndarray, Xp: np.ndarray, t_h: np.ndarray, M: dict, in_idx: list[int]) -> dict:
    """Downstream metrics of one gap (Xt/Xp: n × all columns, true/imputed values)."""
    out = {}
    v_t = M["pls"].predict(Xt[:, in_idx]).ravel()
    v_p = M["pls"].predict(Xp[:, in_idx]).ravel()
    rmse = float(np.sqrt(np.mean((v_p - v_t) ** 2)))
    ivcd_t = float(np.trapezoid(v_t, t_h)) if len(t_h) > 1 else np.nan
    ivcd_p = float(np.trapezoid(v_p, t_h)) if len(t_h) > 1 else np.nan
    out.update({"VCD_RMSE": rmse,
                "VCD_relRMSE": rmse / abs(np.mean(v_t)) if abs(np.mean(v_t)) > 1e-9 else np.nan,
                "IVCD_relErr": abs(ivcd_p - ivcd_t) / abs(ivcd_t) if ivcd_t and abs(ivcd_t) > 1e-9 else np.nan,
                "mu_RMSE": float(np.sqrt(np.mean((_mu(v_p, t_h) - _mu(v_t, t_h)) ** 2))),
                "VCD_ref_mean": float(np.mean(v_t))})

    pca = M["pca"]
    t2_t, q_t = pca.stats(Xt)
    t2_p, q_p = pca.stats(Xp)
    lg = lambda x: np.log10(np.maximum(x, 1e-12))
    out.update({"T2_logRMSE": float(np.sqrt(np.mean((lg(t2_p) - lg(t2_t)) ** 2))),
                "Q_logRMSE": float(np.sqrt(np.mean((lg(q_p) - lg(q_t)) ** 2))),
                "T2_alarm_disagree": float(np.mean((t2_p > pca.t2_lim) != (t2_t > pca.t2_lim))),
                "Q_alarm_disagree": float(np.mean((q_p > pca.q_lim) != (q_t > pca.q_lim))),
                "T2_alarm_rate_true": float(np.mean(t2_t > pca.t2_lim)),
                "T2_alarm_rate_pred": float(np.mean(t2_p > pca.t2_lim)),
                "Q_alarm_rate_true": float(np.mean(q_t > pca.q_lim)),
                "Q_alarm_rate_pred": float(np.mean(q_p > pca.q_lim)),
                "Q_median_ratio": float(np.median(q_p) / np.median(q_t)) if np.median(q_t) > 0 else np.nan})
    return out


_W: dict = {}


def _init_worker(sim_dir: str, model_path: str):
    clean, positions = load_scenarios(sim_dir)
    _W["clean"], _W["positions"] = clean, positions
    with open(model_path, "rb") as f:
        _W["m"] = pickle.load(f)
    _W["col_cluster"] = {c: cid for cid, cols in cfg.load_cluster_map().items() for c in cols}


def _unit_dir_records(unit_dir: str) -> dict:
    """All methods of one unit (design/seed/n_gaps/experiment/reactor)."""
    units = {}
    for p in sorted(glob.glob(os.path.join(unit_dir, "*.npz"))):
        if not p.endswith(".tmp.npz"):
            u = load_unit(p, _W["clean"])
            units[u["meta"]["method"]] = u
    any_u = next(iter(units.values()))
    meta = any_u["meta"]
    exp, reactor = meta["experiment"], meta["reactor"]
    base = {"design": meta["design"], "seed": meta["seed"], "n_gaps": meta["n_gaps"], "experiment": exp,
            "reactor": reactor, "run": f"{exp}_{reactor}"}
    X_full = _W["clean"][exp][reactor]
    t_all = (X_full.index - X_full.index[0]).total_seconds().values / 3600
    cols = list(any_u["y_true"].columns)
    all_cols, inputs = _W["m"]["all_cols"], _W["m"]["inputs"]
    col_pos = [cols.index(c) for c in all_cols]
    in_idx = [all_cols.index(c) for c in inputs]
    valid = run_is_valid(exp, reactor, all_cols) and exp in _W["m"]["models"]
    rows_pos = {g: np.flatnonzero((any_u["gap_id"] == g).values) for g in np.unique(any_u["gap_id"])}
    t_rows = X_full.index.get_indexer(any_u["y_true"].index)

    down, dev = [], []
    for gid, sel in rows_pos.items():
        Xt = any_u["y_true"].values[sel][:, col_pos]
        t_h = t_all[t_rows[sel]]
        ginfo = {**base, "gap_id": int(gid), "duration_h": float(t_h[-1] - t_h[0]), "start_h": float(t_h[0])}
        lin = units["Linear"]["y_pred"].values[sel] if "Linear" in units else None
        for method, u in units.items():
            Xp = u["y_pred"].values[sel][:, col_pos]
            if valid:
                down.append({**ginfo, "method": method, **gap_metrics(Xt, Xp, t_h, _W["m"]["models"][exp], in_idx)})
            if lin is not None and method != "Linear":
                yt, yp = any_u["y_true"].values[sel], u["y_pred"].values[sel]
                for j, c in enumerate(cols):
                    if em.is_excluded_sensor(exp, reactor, c):
                        continue
                    d_true = float(np.sqrt(np.mean((yt[:, j] - lin[:, j]) ** 2)))
                    d_pred = float(np.sqrt(np.mean((yp[:, j] - lin[:, j]) ** 2)))
                    dev.append({**ginfo, "method": method, "column": c,
                                "cluster": cfg.PAPER_CLUSTER[_W["col_cluster"][c]],
                                "rms_pred_minus_linear": d_pred, "rms_true_minus_linear": d_true,
                                "ratio": d_pred / d_true if d_true > 0 else np.nan})
    return {"downstream": down, "linear_deviation": dev}


def records(args):
    t0 = time.time()
    clean, _ = load_scenarios(args.sim_dir)
    model_bundle = build_models(clean, args.out)
    model_path = os.path.join(args.out, "models.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(pd.read_csv(os.path.join(args.out, "model_calibration.csv")).round(3).to_string(index=False), flush=True)

    unit_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(args.results, "**", "*.npz"), recursive=True)})
    print(f"{len(unit_dirs)} units (jobs={args.jobs}).", flush=True)
    down, dev, errors = [], [], []
    if args.jobs == 1:
        _init_worker(args.sim_dir, model_path)
        results = ((d, _safe(d)) for d in unit_dirs)
    else:
        ex = ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker, initargs=(args.sim_dir, model_path))
        futs = {ex.submit(_safe, d): d for d in unit_dirs}
        results = ((futs[f], f.result()) for f in as_completed(futs))
    for i, (d, res) in enumerate(results, 1):
        if "error" in res:
            errors.append((d, res["error"]))
        else:
            down += res["downstream"]
            dev += res["linear_deviation"]
        if i % 50 == 0 or i == len(unit_dirs):
            print(f"  [{i:>5}/{len(unit_dirs)}]  {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.jobs != 1:
        ex.shutdown()

    pd.DataFrame(down).to_csv(os.path.join(args.out, "records_downstream.csv.gz"), index=False)
    pd.DataFrame(dev).to_csv(os.path.join(args.out, "records_linear_deviation.csv.gz"), index=False)
    print(f"Records: {len(down)} downstream, {len(dev)} linear deviation. {len(errors)} errors.", flush=True)
    for d, e in errors:
        print(f"  ERROR {d}: {e}")
    return 1 if errors else 0


def _safe(unit_dir: str) -> dict:
    try:
        return _unit_dir_records(unit_dir)
    except Exception as e:
        import traceback
        return {"error": f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)!r}"}


# ══════════════════════════════════════════════════════════════════════════════
# Statistics (same design as evaluate_multiseed)
# ══════════════════════════════════════════════════════════════════════════════
def stats(args):
    tab = os.path.join(args.out, "tables")
    os.makedirs(tab, exist_ok=True)
    rec = pd.read_csv(os.path.join(args.out, "records_downstream.csv.gz"))
    rec["cluster"] = 0                                  # one observation per gap (all variables)
    long = rec.melt(id_vars=ems.UNIT_KEYS + ["cluster", "method", "duration_h"],
                    value_vars=list(DOWNSTREAM_METRICS), var_name="metric", value_name="value")
    rng = np.random.default_rng(ems.BOOT_SEED)
    summaries, pairs = [], []
    for design, d in long.groupby("design"):
        strata = [("global", "all", d)] + [("n_gaps", f"{n} gap(s)", g) for n, g in d.groupby("n_gaps")]
        for level, name, g in strata:
            boot_cache = {}
            for metric in DOWNSTREAM_METRICS:
                piv, methods = ems._matched(g[g.metric == metric])
                if len(methods) < 2 or len(piv) < 5:
                    continue
                key = tuple(map(tuple, piv[ems.UNIT_KEYS].values.tolist()))
                if key not in boot_cache:
                    boot_cache[key] = ems._boot_index(piv, args.n_boot, rng)
                summary, pair_df = ems.compare(piv, methods, boot_cache[key])
                primary = metric in PRIMARY_DOWNSTREAM
                role = ("sensitivity" if design != ems.PRIMARY_DESIGN else
                        "confirmatory" if (level == "global" and primary) else "exploratory")
                info = {"role": role, "design": design, "level": level, "stratum": name, "metric": metric,
                        "task": DOWNSTREAM_METRICS[metric], "primary": primary}
                summaries.append({**info, **summary})
                pairs.append(pair_df.assign(**info))
    pd.DataFrame(summaries).to_csv(os.path.join(tab, "downstream_comparisons_summary.csv"), index=False)
    pd.concat(pairs).to_csv(os.path.join(tab, "downstream_comparisons_pairwise.csv"), index=False)

    # descriptive: medians per method, alarm rates, Q ratio
    desc_cols = list(DOWNSTREAM_METRICS) + ["T2_alarm_rate_true", "T2_alarm_rate_pred", "Q_alarm_rate_true",
                                            "Q_alarm_rate_pred", "Q_median_ratio", "VCD_ref_mean"]
    rec.groupby(["design", "method"])[desc_cols].median().reset_index().to_csv(
        os.path.join(tab, "downstream_medians.csv"), index=False)
    rec.groupby(["design", "run", "method"])[PRIMARY_DOWNSTREAM].median().reset_index().to_csv(
        os.path.join(tab, "downstream_per_run_medians.csv"), index=False)

    # distance from Linear per cluster
    dev = pd.read_csv(os.path.join(args.out, "records_linear_deviation.csv.gz"))
    # also normalized by the reactor-run std (z): how far does the truth deviate from the
    # straight line at all? Same for all methods of a cluster.
    clean, _ = load_scenarios(args.sim_dir)
    run_std = {(e, r, c): s for e in clean for r in clean[e] for c, s in clean[e][r].std().items()}
    sd = np.array([run_std[k] for k in zip(dev.experiment, dev.reactor, dev.column)])
    dev = dev.assign(true_minus_linear_z=dev.rms_true_minus_linear / np.where(sd > 0, sd, np.nan),
                     pred_minus_linear_z=dev.rms_pred_minus_linear / np.where(sd > 0, sd, np.nan))
    dev.groupby(["design", "cluster", "method"]).agg(
        median_ratio=("ratio", "median"), q25_ratio=("ratio", lambda x: x.quantile(.25)),
        q75_ratio=("ratio", lambda x: x.quantile(.75)),
        median_true_minus_linear_z=("true_minus_linear_z", "median"),
        median_pred_minus_linear_z=("pred_minus_linear_z", "median"), n=("ratio", "size")).reset_index().to_csv(
        os.path.join(tab, "deviation_from_linear.csv"), index=False)

    s = pd.DataFrame(summaries)
    cols = ["stratum", "metric", "winner", "runner_up", "winner_clear", "winner_vs_runner_p_holm",
            "winner_vs_runner_runs_consistent", "winner_vs_runner_seeds_consistent", "n_units", "n_runs"]
    print(s[s.role == "confirmatory"][cols].to_string(index=False))
    with open(os.path.join(tab, "downstream_config.json"), "w", encoding="utf-8") as f:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": args.n_boot,
                   "primary_metrics": PRIMARY_DOWNSTREAM, "pca_variance": PCA_VARIANCE,
                   "control_quantile": CONTROL_QUANTILE, "offline_window_min": OFFLINE_WINDOW_MIN,
                   "mu_smooth_h": MU_SMOOTH_H, "vcd_unit": VCD_UNIT, "pls_inputs": "paper cluster 1 (23 variables)"},
                  f, indent=2)
    print(f"Tables → {tab}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Real outages: PLS prediction vs. offline samples inside the outages
# ══════════════════════════════════════════════════════════════════════════════
def real(args):
    """
    EXP003/005/006: offline samples inside a real outage are compared with the PLS
    prediction from the imputed inline signals of each method. Same pooled PLS
    model as in records(); samples inside outages cannot be used for calibration
    (inputs missing there).
    """
    clean, _ = load_scenarios(args.sim_dir)
    inputs = cfg.load_cluster_map()[PLS_CLUSTER_CSV]
    runs = inline_runs(clean)
    offline = pd.concat([load_offline(e) for e in ALL_EXPERIMENTS], ignore_index=True)
    samples = calibration_samples(runs, offline, inputs)
    pls, n_comp, _ = fit_pls(samples, inputs)
    rows = []
    for p in sorted(glob.glob(os.path.join(REAL_DIR, "imputed", "*.pkl"))):
        exp, reactor = os.path.basename(p)[:-4].split("_")
        with open(p, "rb") as f:
            d = pickle.load(f)
        X = d["X_missing"]
        t = (X.index - X.index[0]).total_seconds().values / 3600
        for ev in d["events"].itertuples():
            in_ev = offline[(offline.experiment == exp) & (offline.reactor == reactor) &
                            (offline.time_h >= ev.start_h) & (offline.time_h <= ev.end_h)]
            for s in in_ev.itertuples():
                sel = np.abs(t - s.time_h) <= OFFLINE_WINDOW_MIN / 60
                row = {"experiment": exp, "reactor": reactor, "event_id": ev.event_id,
                       "modules_missing": ev.modules_missing, "time_h": s.time_h, "vcd_offline": s.vcd,
                       "pls_components": n_comp}
                for method, r in d["results"].items():
                    Xi = X.copy()
                    Xi.loc[r["values"].index, r["values"].columns] = r["values"].values
                    row[f"pred_{method}"] = float(pls.predict(np.median(Xi[inputs].values[sel], axis=0)[None, :]).ravel()[0])
                rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, "real_outages_pls_vs_offline.csv"), index=False)
    print(out.round(3).to_string(index=False) if len(out) else "No offline samples inside the real outages.")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("records")
    r.add_argument("--jobs", type=int, default=1)
    s = sub.add_parser("stats")
    s.add_argument("--n-boot", type=int, default=ems.N_BOOT)
    q = sub.add_parser("real")
    for p in (r, s, q):
        p.add_argument("--results", default=cfg.RESULTS_DIR)
        p.add_argument("--sim-dir", default=cfg.SIM_DIR)
        p.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")         # "→" in messages (Windows console)
    return {"records": records, "stats": stats, "real": real}[args.step](args)


if __name__ == "__main__":
    # pickle the models under the module name (not __main__), so other scripts
    # (paper_outputs.py) can load models.pkl
    import downstream_evaluation as _module
    PCAMonitor = _module.PCAMonitor
    sys.exit(main())
