"""
revision_diagnostics.py
─────────────────────────────────────────────────────────────────────────────
Diagnostics on the mechanisms behind the results. paper_outputs.py builds the
corresponding tables and key numbers from the files written here.

  A. FLk-NN decomposition (Sec. 3.8): FLk-NN is re-run for a subset of seeds,
     logging the Fourier and k-NN components and the endpoint-stitching
     correction separately. Check: the result must match the stored imputation
     from run_imputation_multiseed.py.
  B. Hankel and the periodicity of dissolved CO2 (Sec. 3.3, 3.7):
       B1 simulated gaps (all seeds, random placement, stored imputations):
          does Hankel deviate from the straight line in the same direction as
          the truth? Does it match the period of periodic signals?
       B2 real outages (real_outages/imputed): period of the Hankel oscillation
          vs. the observed context before and after the outage.

Output (OUT_DIR = evaluation_multiseed/diagnostics):
  flknn_components.csv          per gap × column (A)
  flknn_lags.csv                per run × cluster × variable pair × lag rank (A)
  hankel_co2_coherence.csv      per simulated gap, CO2 (B1)
  hankel_co2_real_outages.csv   per real outage, CO2 (B2)

  python revision_diagnostics.py                       # all (A: seeds 1–3)
  python revision_diagnostics.py --skip-flknn          # B only (fast)
  python revision_diagnostics.py --flknn-seeds 1 --jobs 3
"""

import argparse
import glob
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import correlate, detrend

import evaluation_metrics as em
import revision_config as cfg
from impute_methods import BaselineImputerLinear, FLkNNImputer
from run_imputation_multiseed import load_unit, result_path
from simulate_gaps_multiseed import apply_gaps, load_scenarios

OUT_DIR = str(cfg.paths.DIAG_DIR)
REAL_DIR = str(cfg.paths.REAL_OUTAGE_DIR)
CO2 = "CO2(%-sat)-CO2"
CO2_CLUSTER_CSV = next(c for c, p in cfg.PAPER_CLUSTER.items() if p == 3)   # cluster with alpha and CO2
EDGE = 3                       # edge points excluded from the line fit inside the gap
PERIOD_MIN_H, PERIOD_MAX_H = 0.6, 6.0
PERIODIC_ACF = 0.3             # ACF peak above which a signal counts as periodic


# ── Helpers ───────────────────────────────────────────────────────────
def acf_period(x: np.ndarray, dt_h: float) -> tuple[float, float]:
    """
    Dominant period (h) and ACF peak height of a (detrended) signal.
    Searches a true local maximum between PERIOD_MIN_H and min(PERIOD_MAX_H, half the
    length); a maximum at the search border counts as not periodic (NaN).
    """
    x = np.asarray(x, float) - np.mean(x)
    hi_h = min(PERIOD_MAX_H, len(x) * dt_h / 2)
    lo, hi = int(round(PERIOD_MIN_H / dt_h)), int(round(hi_h / dt_h))
    if x.std() == 0 or hi - lo < 3:
        return np.nan, np.nan
    ac = correlate(x, x, mode="full", method="fft")[len(x) - 1:]
    ac = ac / ac[0]
    seg = ac[lo:hi + 1]
    i = int(np.argmax(seg))
    if i == 0 or i == len(seg) - 1:
        return np.nan, np.nan
    return (lo + i) * dt_h, float(seg[i])


def _hours(index) -> np.ndarray:
    return (index - index[0]).total_seconds().values / 3600


# ── A. FLk-NN decomposition ───────────────────────────────────────────────────────
class _TracedFLkNN(FLkNNImputer):
    """FLk-NN that keeps both components (computation unchanged)."""

    def _fourier_impute(self, X_in):
        self.fourier_ = super()._fourier_impute(X_in)
        return self.fourier_

    def _knn_lag_impute(self, X_in):
        self.knn_ = super()._knn_lag_impute(X_in)
        return self.knn_


def _lags(X: pd.DataFrame, num_lags: int, max_lag: int) -> list[dict]:
    """Selected lags per variable pair, computed as in FLkNNImputer._knn_lag_impute (step 1)."""
    v = (pd.DataFrame(X.values.astype(float)).interpolate(method="linear", limit_direction="both")
         .bfill().ffill().fillna(0.0).values)
    rows = []
    for i in range(v.shape[1]):
        for j in range(i + 1, v.shape[1]):
            cc = correlate(v[:, i], v[:, j], method="fft")
            center = v.shape[0] - 1
            st, en = max(0, center - max_lag), min(len(cc), center + max_lag + 1)
            w = np.abs(cc[st:en])
            k = min(num_lags, len(w))
            top = np.argpartition(w, -k)[-k:]
            top = top[np.argsort(w[top])[::-1]]
            for rank, t in enumerate(top):
                rows.append({"col_a": X.columns[i], "col_b": X.columns[j], "lag_rank": rank + 1,
                             "lag": int(t + st - center)})
    return rows


_A: dict = {}


def _init_flknn(sim_dir: str):
    _A["clean"], _A["positions"] = load_scenarios(sim_dir)
    _A["cluster_map"] = cfg.load_cluster_map()
    _A["params"] = cfg.load_flknn_params()


def flknn_unit(unit: dict, results_dir: str) -> tuple[list[dict], list[dict], str]:
    exp, reactor, n_gaps = unit["experiment"], unit["reactor"], unit["n_gaps"]
    X_true = _A["clean"][exp][reactor]
    p = _A["positions"]
    gaps = p[(p.design == unit["design"]) & (p.seed == unit["seed"]) & (p.n_gaps == n_gaps)
             & (p.experiment == exp)]
    X_missing, _ = apply_gaps(X_true, gaps)
    stored = load_unit(result_path(results_dir, {**unit, "method": "FLk-NN"}), _A["clean"])
    run_std = X_true.std().replace(0, 1.0)
    t_idx = X_true.index
    rows, lag_rows, msgs = [], [], []

    for cid, features in sorted(_A["cluster_map"].items()):
        cols = [c for c in features if not em.is_excluded_sensor(exp, reactor, c)]
        if not cols:
            continue
        params = _A["params"][(n_gaps, cid)]
        imp = _TracedFLkNN(**params)
        Xc = X_missing[cols]
        full = imp.impute(Xc)[cols].values
        F, K = imp.fourier_[cols].values, imp.knn_[cols].values
        with np.errstate(all="ignore"):
            pre = np.nanmean(np.dstack((F, K)), axis=2)          # ensemble before stitching
        lin = BaselineImputerLinear().impute(Xc)[cols].values
        T = X_true[cols].values
        miss = Xc.isna().values
        base = {**unit, "run": f"{exp}_{reactor}", "cluster": cfg.PAPER_CLUSTER[cid], "n_cols": len(cols)}
        if len(cols) > 1:
            lag_rows += [{**base, **r} for r in _lags(Xc, imp.num_lags, imp.max_lag)]

        # check against the stored imputation (same computation → identical)
        sp = stored["y_pred"][cols]
        dev = np.abs(full[t_idx.get_indexer(sp.index)] - sp.values) / run_std[cols].values
        if np.nanmax(dev) > 1e-6:
            msgs.append(f"{exp} {reactor} {n_gaps} gaps cluster {base['cluster']}: "
                        f"deviation from the stored imputation {np.nanmax(dev):.2e} std")

        for g in gaps.itertuples():
            st, en = int(g.start_idx), int(g.end_idx)
            dur = (t_idx[en] - t_idx[st]).total_seconds() / 3600 if en < len(t_idx) else np.nan
            for j, c in enumerate(cols):
                if not miss[st:en, j].all():
                    continue
                s = float(run_std[c])
                f, k, y, pr, li, tr = (a[st:en, j] / s for a in (F, K, full, pre, lin, T))
                cov = ~np.isnan(k)
                stitch = y - pr                                  # stitching correction
                d = (y - f)[EDGE:-EDGE]                          # deviation from the Fourier component, interior
                tt = np.arange(len(d))
                r2 = np.nan
                if len(d) > 2 and d.std() > 0:
                    fit = np.polyval(np.polyfit(tt, d, 1), tt)
                    r2 = 1 - np.sum((d - fit) ** 2) / np.sum((d - d.mean()) ** 2)
                outside = np.ones(len(X_true), bool)
                outside[miss[:, j]] = False
                d_out = np.diff(T[:, j])[outside[1:] & outside[:-1]]
                rms = lambda a: float(np.sqrt(np.mean(a ** 2))) if len(a) else np.nan
                rows.append({**base, "gap_id": int(g.gap_id), "gap_len": en - st, "duration_h": dur,
                             "column": c,
                             "knn_coverage": float(cov.mean()),
                             "rms_fourier_minus_linear": rms(f - li),
                             "rms_knn_minus_linear": rms((k - li)[cov]),
                             "rms_flknn_minus_linear": rms(y - li),
                             "rms_stitch_correction": rms(stitch),
                             "r2_line_flknn_minus_fourier": r2,
                             "roughness_ratio": float(np.std(np.diff(y)) / (np.std(d_out) / s))
                             if len(d_out) and np.std(d_out) > 0 else np.nan,
                             "nrmse_linear": rms(li - tr), "nrmse_fourier": rms(f - tr),
                             "nrmse_flknn": rms(y - tr)})
    return rows, lag_rows, "; ".join(msgs)


def run_flknn(a) -> None:
    clean, pos = load_scenarios(a.sim)
    scen = (pos[(pos.design == "unfixed") & pos.seed.isin(a.flknn_seeds)]
            [["design", "seed", "n_gaps", "experiment"]].drop_duplicates())
    units = [{"design": s.design, "seed": int(s.seed), "n_gaps": int(s.n_gaps), "experiment": s.experiment,
              "reactor": r} for s in scen.itertuples() for r in sorted(clean[s.experiment])]
    print(f"A. FLk-NN decomposition: {len(units)} units (seeds {a.flknn_seeds}), {a.jobs} processes", flush=True)
    rows, lag_rows, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init_flknn, initargs=(a.sim,)) as ex:
        futs = {ex.submit(flknn_unit, u, a.results): u for u in units}
        for i, fut in enumerate(as_completed(futs), 1):
            r, lr, msg = fut.result()
            rows += r
            lag_rows += lr
            if msg:
                print("   WARNING:", msg, flush=True)
            print(f"   {i}/{len(units)}  {time.time() - t0:6.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "flknn_components.csv"), index=False)
    pd.DataFrame(lag_rows).to_csv(os.path.join(a.out, "flknn_lags.csv"), index=False)


# ── B1. Hankel and CO2 periodicity, simulated gaps ────────────────────────
def run_hankel_sim(a) -> None:
    clean, pos = load_scenarios(a.sim)
    bounds = {(e, r): em.phase_boundaries(clean[e][r]) for e in clean for r in clean[e]}
    smooth = {(e, r): em.smooth_reference(clean[e][r][[CO2]])[CO2] for e in clean for r in clean[e]}
    rows = []
    paths = sorted(glob.glob(os.path.join(a.results, "unfixed", "seed_*", "*_gaps", "*", "*", "Hankel.npz")))
    print(f"B1. Hankel/CO2, simulated gaps: {len(paths)} units", flush=True)
    for hp in paths:
        H = load_unit(hp, clean)
        L = load_unit(hp.replace("Hankel.npz", "Linear.npz"), clean)
        m = H["meta"]
        exp, reactor = m["experiment"], m["reactor"]
        t_idx = clean[exp][reactor].index
        gaps = pos[(pos.design == m["design"]) & (pos.seed == m["seed"]) & (pos.n_gaps == m["n_gaps"])
                   & (pos.experiment == exp)].set_index("gap_id")
        for gid, g in gaps.iterrows():
            sel = (H["gap_id"] == gid).values
            idx = H["y_true"].index[sel]
            h, li = H["y_pred"][CO2].values[sel], L["y_pred"][CO2].values[sel]
            ys = smooth[(exp, reactor)].loc[idx].values
            hours = lambda i: (t_idx[int(i)] - t_idx[0]).total_seconds() / 3600
            start_h, end_h = hours(g.start_idx), hours(g.end_idx)      # as in evaluate_multiseed.py
            dt = float(np.median(np.diff(_hours(idx))))
            dh, dt_true = h - li, ys - li
            per_t, acf_t = acf_period(detrend(ys), dt)
            per_h, acf_h = acf_period(detrend(h), dt)
            rows.append({"seed": m["seed"], "n_gaps": m["n_gaps"], "experiment": exp, "reactor": reactor,
                         "gap_id": int(gid), "duration_h": end_h - start_h,
                         "phase": em.assign_phase((start_h + end_h) / 2, bounds[(exp, reactor)]),
                         "r_hankel_vs_truth_deviation": float(np.corrcoef(dh, dt_true)[0, 1])
                         if dh.std() > 0 and dt_true.std() > 0 else np.nan,
                         "period_truth_h": per_t, "acf_truth": acf_t,
                         "period_hankel_h": per_h, "acf_hankel": acf_h,
                         "sd_detrended_truth": float(np.std(detrend(ys))),
                         "sd_detrended_hankel": float(np.std(detrend(h)))})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "hankel_co2_coherence.csv"), index=False)


# ── B2. Hankel and CO2 periodicity, real outages ───────────────────────────
def run_hankel_real(a) -> None:
    hparams = cfg.load_hankel_params()
    rows = []
    paths = sorted(glob.glob(os.path.join(a.real, "imputed", "*.pkl")))
    print(f"B2. Hankel/CO2, real outages: {len(paths)} runs", flush=True)
    for path in paths:
        with open(path, "rb") as f:
            d = pickle.load(f)
        X, ev, res = d["X_missing"], d["events"], d["results"]
        exp, reactor = os.path.basename(path)[:-4].split("_")
        dt = float(np.median(np.diff(_hours(X.index))))
        diag = [g for g in res["Hankel"]["diagnostics"] if g.get("cluster_id") == CO2_CLUSTER_CSV]
        ctx = hparams[(len(ev), CO2_CLUSTER_CSV)].get("context_factor", 2)
        for e in ev.itertuples():
            st, en = int(e.start_idx), int(e.end_idx)
            L = en - st
            idx = X.index[st:en]
            h = res["Hankel"]["values"][CO2].loc[idx].values
            li = res["Linear"]["values"][CO2].loc[idx].values
            left = X[CO2].iloc[max(0, st - int(ctx * L)):st].dropna().values
            right = X[CO2].iloc[en:en + int(ctx * L)].dropna().values
            pl = acf_period(detrend(left), dt) if len(left) > 3 else (np.nan, np.nan)
            pr = acf_period(detrend(right), dt) if len(right) > 3 else (np.nan, np.nan)
            ph = acf_period(detrend(h - li), dt)
            dg = next((g for g in diag if g.get("gap_start") == st), {})
            rows.append({"experiment": exp, "reactor": reactor, "event_id": int(e.event_id),
                         "start_h": e.start_h, "duration_h": e.duration_h,
                         "context_left_h": len(left) * dt, "context_right_h": len(right) * dt,
                         "period_context_left_h": pl[0], "acf_context_left": pl[1],
                         "period_context_right_h": pr[0], "acf_context_right": pr[1],
                         "period_hankel_deviation_h": ph[0], "acf_hankel_deviation": ph[1],
                         "hankel_iterations": dg.get("n_iter"), "hankel_converged": dg.get("converged"),
                         "hankel_rank": dg.get("rank_used")})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "hankel_co2_real_outages.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", default=cfg.SIM_DIR)
    ap.add_argument("--results", default=cfg.RESULTS_DIR)
    ap.add_argument("--real", default=REAL_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--flknn-seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--skip-flknn", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    run_hankel_real(a)
    run_hankel_sim(a)
    if not a.skip_flknn:
        run_flknn(a)
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
