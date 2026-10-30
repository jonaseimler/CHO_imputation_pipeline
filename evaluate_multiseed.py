"""
evaluate_multiseed.py
─────────────────────────────────────────────────────────────────────────────
Evaluation of the multi-seed imputations.

Two steps (run separately):

  1) records – per-gap metrics (expensive, incremental, parallel)
       python evaluate_multiseed.py records --jobs 3
     One cache per result file (run_imputation_multiseed.py); only missing,
     outdated or METRICS_VERSION-mismatched caches are recomputed. All caches
     are then merged into:
       records_univariate.csv.gz   per (unit, gap, column)
       records_correlation.csv.gz  per (unit, gap, cluster)
       hankel_diagnostics.csv      per (Hankel unit, gap, cluster)
       phase_boundaries.csv        phase boundaries per reactor run

  2) stats – statistics and tables (fast, repeatable)
       python evaluate_multiseed.py stats
     Output in OUT_DIR/tables/ (see stats()).

Statistical design:
  • Observation unit = gap × cluster: univariate metrics are aggregated over the
    channels of a cluster (median), so highly correlated channels (e.g. 17
    capacitance frequencies) do not count as independent observations.
  • Independent unit = reactor run (6 test runs). Uncertainty from a
    hierarchical bootstrap: (1) runs with replacement, (2) gap events within
    each drawn run with replacement (all clusters of a gap stay together).
    Yields 95 % CIs of the paired median differences and bootstrap p-values,
    Holm-corrected over all method pairs.
  • "Clear win" only for Holm p < 0.05. Winner = most clear pairwise wins,
    then lower median.
  • Consistency: for each method pair, the number of independent reactor runs
    (k/6) and seeds (k/10) whose median difference has the overall sign. With
    only 6 independent runs this is the most robust statement; p-values are
    supplementary (an exact run-level test reaches at best p = 2/64 and does
    not survive Holm correction).
  • Omnibus: Friedman test at run level (median per run and method, n = 6)
    and Kendall's W as agreement between runs.
  • Role of each analysis (column "role"), fixed in advance:
      confirmatory – unfixed, per cluster, primary metrics (main result)
      summary      – unfixed, global (summary over clusters)
      exploratory  – unfixed, per gap count; all non-primary metrics
      sensitivity  – fixed design (all strata), smoothed reference
  • Method comparisons only for the primary metrics (one per family,
    evaluation_metrics.PRIMARY_METRICS); --all-metrics adds all others
    (marked exploratory).
  • All metrics are "lower = better" (evaluation_metrics.py).
  • EXP007 (tuning) is not included – only EXP001/002/004.
"""

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

import evaluation_metrics as em
import revision_config as cfg
from run_imputation_multiseed import load_unit
from simulate_gaps_multiseed import load_scenarios

METRICS_VERSION = 3          # increase when evaluation_metrics changes → caches are recomputed
OUT_DIR = str(cfg.paths.EVAL_DIR)
N_BOOT = 2000
BOOT_SEED = 20261020
ALPHA = 0.05
PRIMARY_DESIGN = "unfixed"       # main analysis; "fixed" is a sensitivity analysis
DURATION_BINS = [3, 6, 9, 12, 15]
REFERENCE_METHOD = "Linear"      # reference for paired differences by duration/phase
KEY_VARIABLES = ["Permittivity(pF/cm)-VCD", "CO2(%-sat)-CO2",
                 "Conductivity(mS/cm)-VCD", "Temperature(°C)-VCD"]   # RMSE in physical units
UNIT_KEYS = ["design", "seed", "n_gaps", "experiment", "reactor", "run", "gap_id"]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: records
# ══════════════════════════════════════════════════════════════════════════════
_W: dict = {}


def _init_worker(sim_dir: str):
    clean, positions = load_scenarios(sim_dir)
    _W["clean"] = clean
    _W["positions"] = positions
    _W["col_cluster"] = {c: cid for cid, cols in cfg.load_cluster_map().items() for c in cols}
    _W["bounds"] = {(e, r): em.phase_boundaries(clean[e][r]) for e in clean for r in clean[e]}
    _W["run_std"] = {(e, r): clean[e][r].std() for e in clean for r in clean[e]}
    _W["smooth"] = {(e, r): em.smooth_reference(clean[e][r]) for e in clean for r in clean[e]}


def _unit_records(npz_path: str) -> dict:
    u = load_unit(npz_path, _W["clean"])
    meta = u["meta"]
    exp, reactor = meta["experiment"], meta["reactor"]
    p = _W["positions"]
    gaps = p[(p.design == meta["design"]) & (p.seed == meta["seed"]) &
             (p.n_gaps == meta["n_gaps"]) & (p.experiment == exp)].set_index("gap_id")
    bounds = _W["bounds"][(exp, reactor)]
    run_std = _W["run_std"][(exp, reactor)]
    cols = [c for c in u["y_true"].columns if not em.is_excluded_sensor(exp, reactor, c)]
    # times from the real timestamps (sampling is slightly irregular; idx · 30 s
    # drifts by up to ~1 h at the end of a run) – consistent with phase_boundaries()
    t_idx = _W["clean"][exp][reactor].index
    hours = lambda i: (t_idx[int(i)] - t_idx[0]).total_seconds() / 3600
    # smoothed reference at the same rows (column order as y_true)
    ys_all = _W["smooth"][(exp, reactor)][list(u["y_true"].columns)].values[t_idx.get_indexer(u["y_true"].index)]

    base ={"design": meta["design"], "seed": meta["seed"], "n_gaps": meta["n_gaps"],
            "experiment": exp, "reactor": reactor, "run": f"{exp}_{reactor}", "method": meta["method"]}
    uni, cor = [], []
    for gid, g in gaps.iterrows():
        sel = (u["gap_id"] == gid).values
        yt, yp, ys = u["y_true"].values[sel], u["y_pred"].values[sel], ys_all[sel]
        start_h, end_h = hours(g.start_idx), hours(g.end_idx)   # end_idx = first point after the gap
        mid_h = (start_h + end_h) / 2
        ginfo = {**base, "gap_id": int(gid), "gap_len": int(g.n_points), "duration_h": end_h - start_h,
                 "start_h": start_h, "mid_h": mid_h, "phase": em.assign_phase(mid_h, bounds)}
        col_idx = {c: i for i, c in enumerate(u["y_true"].columns)}

        for c in cols:
            i = col_idx[c]
            cid = _W["col_cluster"][c]
            uni.append({**ginfo, "column": c, "cluster_csv": cid, "cluster": cfg.PAPER_CLUSTER[cid],
                        **em.univariate_metrics(yt[:, i], yp[:, i], float(run_std[c]), ys[:, i])})

        for cid in sorted(set(_W["col_cluster"][c] for c in cols)):
            ccols = [col_idx[c] for c in cols if _W["col_cluster"][c] == cid]
            if len(ccols) < 2 or sel.sum() < max(5, len(ccols) + 1):   # too few points for a correlation
                continue
            cor.append({**ginfo, "cluster_csv": cid, "cluster": cfg.PAPER_CLUSTER[cid], "n_cols": len(ccols),
                        **em.correlation_metrics(yt[:, ccols], yp[:, ccols], ys[:, ccols])})

    diag = []
    if u["diagnostics"]:
        start_to_gid = {int(g.start_idx): int(gid) for gid, g in gaps.iterrows()}
        for d in u["diagnostics"]:
            diag.append({**base, "gap_id": start_to_gid.get(int(d["gap_start"]), -1),
                         "cluster": cfg.PAPER_CLUSTER[int(d["cluster_id"])], **d})
    return {"version": METRICS_VERSION, "univariate": pd.DataFrame(uni),
            "correlation": pd.DataFrame(cor), "diagnostics": pd.DataFrame(diag)}


def _cache_path(npz_path: str, results_dir: str, cache_dir: str) -> str:
    rel = os.path.relpath(npz_path, results_dir)
    return os.path.join(cache_dir, rel[:-4] + ".pkl")


def _needs_update(npz_path: str, cache: str, force: bool) -> bool:
    if force or not os.path.exists(cache) or os.path.getmtime(cache) < os.path.getmtime(npz_path):
        return True
    try:
        return pd.read_pickle(cache)["version"] != METRICS_VERSION
    except Exception:
        return True


def _process(npz_path: str, results_dir: str, cache_dir: str) -> tuple[str, str]:
    try:
        rec = _unit_records(npz_path)
        cache = _cache_path(npz_path, results_dir, cache_dir)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + ".tmp"
        pd.to_pickle(rec, tmp)
        os.replace(tmp, cache)
        return npz_path, ""
    except Exception as e:
        return npz_path, f"{type(e).__name__}: {e}"


def records(args):
    cache_dir = os.path.join(args.out, "records_cache")
    files = sorted(f for f in glob.glob(os.path.join(args.results, "**", "*.npz"), recursive=True)
                   if not f.endswith(".tmp.npz"))
    todo = [f for f in files if _needs_update(f, _cache_path(f, args.results, cache_dir), args.force)]
    print(f"{len(files)} result files, {len(todo)} to evaluate (jobs={args.jobs}).", flush=True)

    errors, t0 = [], time.time()

    def _report(i: int, f: str, err: str):
        if err:
            errors.append((f, err))
        if i % 50 == 0 or i == len(todo) or err:
            eta = (time.time() - t0) / i * (len(todo) - i) / 60
            print(f"  [{i:>5}/{len(todo)}] {'ERR ' + err if err else 'ok'}  ETA {eta:.1f} min", flush=True)

    if todo and args.jobs == 1:
        _init_worker(args.sim_dir)
        for i, f in enumerate(todo, 1):
            _report(i, *_process(f, args.results, cache_dir))
    elif todo:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                                 initargs=(args.sim_dir,)) as ex:
            futures = [ex.submit(_process, f, args.results, cache_dir) for f in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                _report(i, *fut.result())

    # merge all caches
    parts = {"univariate": [], "correlation": [], "diagnostics": []}
    for f in files:
        cache = _cache_path(f, args.results, cache_dir)
        if os.path.exists(cache):
            rec = pd.read_pickle(cache)
            for k in parts:
                if len(rec[k]):
                    parts[k].append(rec[k])
    os.makedirs(args.out, exist_ok=True)
    uni = pd.concat(parts["univariate"], ignore_index=True)
    cor = pd.concat(parts["correlation"], ignore_index=True)
    uni.to_csv(os.path.join(args.out, "records_univariate.csv.gz"), index=False)
    cor.to_csv(os.path.join(args.out, "records_correlation.csv.gz"), index=False)
    if parts["diagnostics"]:
        pd.concat(parts["diagnostics"], ignore_index=True).to_csv(
            os.path.join(args.out, "hankel_diagnostics.csv"), index=False)

    clean, _ = load_scenarios(args.sim_dir)
    pd.DataFrame([{"experiment": e, "reactor": r, **em.phase_boundaries(clean[e][r])}
                  for e in clean for r in clean[e]]).to_csv(os.path.join(args.out, "phase_boundaries.csv"), index=False)

    print(f"\nRecords: {len(uni)} univariate, {len(cor)} correlation – "
          f"methods: {sorted(uni.method.unique())}. {len(errors)} errors. "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    for f, e in errors:
        print(f"  ERROR {f}: {e}")
    return 1 if errors else 0


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: statistics
# ══════════════════════════════════════════════════════════════════════════════
def load_units(out_dir: str) -> pd.DataFrame:
    """
    Long table of observation units (gap × cluster × method × metric).
    Univariate metrics: median over the channels of the cluster. RMSE (physical
    units) is not aggregated over channels and is evaluated separately.
    """
    uni = pd.read_csv(os.path.join(out_dir, "records_univariate.csv.gz"))
    cor = pd.read_csv(os.path.join(out_dir, "records_correlation.csv.gz"))
    keys = UNIT_KEYS + ["cluster", "method", "duration_h", "mid_h", "phase"]
    uni_metrics = [m for m in em.UNIVARIATE_METRICS if m != "RMSE"]
    uni_metrics += [m for m in em.SMOOTH_METRICS if m in uni.columns]
    cor_metrics = list(em.CORRELATION_METRICS) + [m for m in em.SMOOTH_METRICS if m in cor.columns]
    agg = uni.groupby(keys)[uni_metrics].median().reset_index()
    long_uni = agg.melt(id_vars=keys, var_name="metric", value_name="value")
    long_cor = cor[keys + cor_metrics].melt(id_vars=keys, var_name="metric", value_name="value")
    return pd.concat([long_uni, long_cor], ignore_index=True)


def _family(metric: str) -> str:
    """Metric family; smoothed-reference variants inherit the family of their raw metric."""
    families = {**em.UNIVARIATE_METRICS, **em.CORRELATION_METRICS}
    return families[em.SMOOTH_METRICS.get(metric, metric)]


def _matched(df: pd.DataFrame, extra: tuple = ()) -> tuple[pd.DataFrame, list[str]]:
    """
    Pivot to methods; only units with a value for every method present.
    extra: additional per-unit columns to keep (e.g. duration_h, phase).
    """
    idx = UNIT_KEYS + ["cluster"] + list(extra)
    piv = df.pivot_table(index=idx, columns="method", values="value", aggfunc="first")
    methods = [m for m in cfg.METHODS if m in piv.columns]
    piv = piv[methods].dropna()
    return piv.reset_index(), methods


def _boot_index(piv: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Hierarchical bootstrap: runs, then gap events within each run."""
    event = piv[["design", "seed", "n_gaps", "run", "gap_id"]].astype(str).agg("|".join, axis=1).values
    runs = piv["run"].values
    by_run = {}
    for r in np.unique(runs):
        ev = event[runs == r]
        rows = np.flatnonzero(runs == r)
        by_run[r] = [rows[ev == e] for e in np.unique(ev)]
    run_list = list(by_run)
    out = []
    for _ in range(n_boot):
        parts = []
        for r in rng.choice(run_list, size=len(run_list), replace=True):
            evs = by_run[r]
            for k in rng.integers(0, len(evs), size=len(evs)):
                parts.append(evs[k])
        out.append(np.concatenate(parts))
    return out


def _rank_biserial(d: np.ndarray) -> float:
    """Paired rank-biserial correlation; positive = first method better (lower values)."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    w_pos, w_neg = r[d > 0].sum(), r[d < 0].sum()
    return float((w_neg - w_pos) / (w_pos + w_neg))


def compare(piv: pd.DataFrame, methods: list[str], boot_idx: list[np.ndarray]) -> tuple[dict, pd.DataFrame]:
    """Compare all methods in one stratum for one metric."""
    V = piv[methods].values
    pairs = list(combinations(range(len(methods)), 2))
    D = np.column_stack([V[:, a] - V[:, b] for a, b in pairs])
    obs = np.median(D, axis=0)
    boot = np.vstack([np.median(D[i], axis=0) for i in boot_idx])           # (B, n_pairs)
    B = len(boot_idx)
    p = np.minimum(1.0, 2 * np.minimum(((boot <= 0).sum(0) + 1) / (B + 1), ((boot >= 0).sum(0) + 1) / (B + 1)))
    p_holm = multipletests(p, method="holm")[1]

    # sign consistency over independent runs and over seeds
    D_df = pd.DataFrame(D)
    by_run = np.sign(D_df.groupby(piv["run"].values).median().values)     # (n_runs, n_pairs)
    by_seed = np.sign(D_df.groupby(piv["seed"].values).median().values)   # (n_seeds, n_pairs)

    rows, clear_wins = [], {m: 0 for m in methods}
    for k, (a, b) in enumerate(pairs):
        lo, hi = np.percentile(boot[:, k], [2.5, 97.5])
        clear = p_holm[k] < ALPHA
        better = methods[a] if obs[k] < 0 else methods[b] if obs[k] > 0 else ""
        if clear and better:
            clear_wins[better] += 1
        direction = np.sign(obs[k])
        rows.append({"method_a": methods[a], "method_b": methods[b], "median_diff_a_minus_b": obs[k],
                     "ci_low": lo, "ci_high": hi, "p_boot": p[k], "p_holm": p_holm[k],
                     "rank_biserial_a_better": _rank_biserial(D[:, k]),
                     "better": better, "clear": bool(clear and better),
                     "runs_consistent": int((by_run[:, k] == direction).sum()) if direction else 0,
                     "n_runs": len(by_run),
                     "seeds_consistent": int((by_seed[:, k] == direction).sum()) if direction else 0,
                     "n_seeds": len(by_seed)})
    pair_df = pd.DataFrame(rows)

    med = dict(zip(methods, np.median(V, axis=0)))
    q1, q3 = np.percentile(V, 25, axis=0), np.percentile(V, 75, axis=0)
    ranking = sorted(methods, key=lambda m: (-clear_wins[m], med[m]))
    winner, runner = ranking[0], ranking[1]
    wr = pair_df[((pair_df.method_a == winner) & (pair_df.method_b == runner)) |
                 ((pair_df.method_a == runner) & (pair_df.method_b == winner))].iloc[0]

    run_med = piv.groupby("run")[methods].median()
    if len(run_med) >= 2 and len(methods) >= 3:
        chi2, p_fr = friedmanchisquare(*[run_med[m].values for m in methods])
        kendall_w = chi2 / (len(run_med) * (len(methods) - 1))
    else:
        chi2 = p_fr = kendall_w = np.nan
    summary = {
        "n_units": len(piv), "n_runs": piv["run"].nunique(), "n_gap_events": len(_events(piv)),
        "winner": winner, "runner_up": runner, "winner_clear": bool(wr["clear"]),
        "winner_vs_runner_p_holm": wr["p_holm"], "winner_vs_runner_rank_biserial":
            wr["rank_biserial_a_better"] if wr["method_a"] == winner else -wr["rank_biserial_a_better"],
        "winner_vs_runner_runs_consistent": f"{wr['runs_consistent']}/{wr['n_runs']}",
        "winner_vs_runner_seeds_consistent": f"{wr['seeds_consistent']}/{wr['n_seeds']}",
        "friedman_runs_chi2": chi2, "friedman_runs_p": p_fr, "kendall_w_runs": kendall_w,
        **{f"median_{m}": med[m] for m in methods},
        **{f"iqr_{m}": f"{q1[i]:.3g}–{q3[i]:.3g}" for i, m in enumerate(methods)},
        **{f"clear_wins_{m}": clear_wins[m] for m in methods},
    }
    return summary, pair_df


def _events(piv: pd.DataFrame) -> pd.DataFrame:
    return piv[["design", "seed", "n_gaps", "run", "gap_id"]].drop_duplicates()


def _role(design: str, level: str, primary: bool, smooth: bool = False) -> str:
    """Pre-specified role of an analysis (see module docstring)."""
    if design != PRIMARY_DESIGN or smooth:
        return "sensitivity"
    if not primary or level == "n_gaps":
        return "exploratory"
    return "confirmatory" if level == "cluster" else "summary"


def metric_redundancy(out_dir: str, tab: str):
    """
    Basis for the metric selection (supplement): Spearman correlations between the
    metrics and discrimination per metric (Kendall's W over gap × column or
    gap × cluster: 0 = methods indistinguishable, 1 = always the same ranking).
    Main design only.
    """
    uni = pd.read_csv(os.path.join(out_dir, "records_univariate.csv.gz"))
    cor = pd.read_csv(os.path.join(out_dir, "records_correlation.csv.gz"))
    uni, cor = uni[uni.design == PRIMARY_DESIGN], cor[cor.design == PRIMARY_DESIGN]
    uni_m = [m for m in em.UNIVARIATE_METRICS if m != "RMSE"]
    cor_m = list(em.CORRELATION_METRICS)
    uni[uni_m].corr(method="spearman").to_csv(os.path.join(tab, "metric_redundancy_spearman.csv"))
    cor[cor_m].corr(method="spearman").to_csv(os.path.join(tab, "metric_redundancy_spearman_correlation.csv"))

    def kendall_w(piv: pd.DataFrame) -> float:
        n, k = piv.shape
        R = piv.rank(axis=1).values
        return 12 * ((R.sum(0) - n * (k + 1) / 2) ** 2).sum() / (n ** 2 * (k ** 3 - k))

    rows = []
    for df, ms, key in [(uni, uni_m, UNIT_KEYS + ["column"]), (cor, cor_m, UNIT_KEYS + ["cluster"])]:
        for m in ms:
            piv = df.pivot_table(index=key, columns="method", values=m, aggfunc="first").dropna()
            n, k = piv.shape
            if n < 2 or k < 2:
                continue
            # FLk-NN is almost always last and inflates W; W without FLk-NN shows whether
            # a metric separates the competitive methods.
            rest = piv.drop(columns=[c for c in ["FLk-NN"] if c in piv.columns])
            rows.append({"metric": m, "family": {**em.UNIVARIATE_METRICS, **em.CORRELATION_METRICS}[m],
                         "primary": m in em.PRIMARY_METRICS.values(), "kendall_w": kendall_w(piv),
                         "kendall_w_without_flknn": kendall_w(rest) if rest.shape[1] >= 2 else np.nan,
                         "n_units": n,
                         **{f"mean_rank_{c}": v for c, v in zip(piv.columns, piv.rank(axis=1).values.mean(0))}})
    pd.DataFrame(rows).to_csv(os.path.join(tab, "metric_discrimination.csv"), index=False)


def stats(args):
    t0 = time.time()
    tab = os.path.join(args.out, "tables")
    os.makedirs(tab, exist_ok=True)
    units = load_units(args.out)
    rng = np.random.default_rng(BOOT_SEED)
    primary = list(em.PRIMARY_METRICS.values())
    smooth_metrics = [m for m in em.SMOOTH_METRICS if m in set(units.metric)]
    metrics = primary + smooth_metrics if not args.all_metrics else sorted(units.metric.unique())

    # ── A. Method comparisons per stratum (global, cluster, gap count) ────────
    summaries, pair_tables = [], []
    for design in sorted(units.design.unique()):
        d_units = units[units.design == design]
        strata = [("global", "all", d_units)]
        strata += [("cluster", f"Cluster {c}", g) for c, g in d_units.groupby("cluster")]
        strata += [("n_gaps", f"{n} gap(s)", g) for n, g in d_units.groupby("n_gaps")]
        for level, name, g in strata:
            boot_cache = {}
            for metric in metrics:
                piv, methods = _matched(g[g.metric == metric])
                if len(methods) < 2 or len(piv) < 5:
                    continue
                key = tuple(map(tuple, piv[UNIT_KEYS + ["cluster"]].values.tolist()))
                if key not in boot_cache:   # same resamples for all metrics with identical units
                    boot_cache[key] = _boot_index(piv, args.n_boot, rng)
                summary, pairs = compare(piv, methods, boot_cache[key])
                info = {"role": _role(design, level, metric in primary, metric in em.SMOOTH_METRICS),
                        "design": design, "level": level, "stratum": name, "metric": metric,
                        "family": _family(metric), "primary": metric in primary}
                summaries.append({**info, **summary})
                pair_tables.append(pairs.assign(**info))
            print(f"  {design:<7} {level:<7} {name:<10} done ({time.time() - t0:.0f}s)", flush=True)
    if summaries:
        pd.DataFrame(summaries).to_csv(os.path.join(tab, "comparisons_summary.csv"), index=False)
        pd.concat(pair_tables).to_csv(os.path.join(tab, "comparisons_pairwise.csv"), index=False)
    else:
        print("  WARNING: no stratum with ≥ 5 complete units – no method comparisons.", flush=True)

    prim = units[units.metric.isin(primary)]

    # ── A1. Sensitivity: smoothed reference instead of raw data (measurement noise) ──
    #     Per stratum: same winner? rank correlation of the method medians?
    #     same set of clear pairwise decisions?
    if summaries and smooth_metrics:
        s_df = pd.DataFrame(summaries).set_index(["design", "level", "stratum", "metric"])
        p_df = pd.concat(pair_tables)
        sens = []
        for sm in smooth_metrics:
            raw = em.SMOOTH_METRICS[sm]
            for (design, level, stratum, _), s in s_df.xs(sm, level="metric", drop_level=False).iterrows():
                if (design, level, stratum, raw) not in s_df.index:
                    continue
                r = s_df.loc[(design, level, stratum, raw)]
                meths = [m for m in cfg.METHODS if f"median_{m}" in s and pd.notna(s[f"median_{m}"])]
                rho = spearmanr([r[f"median_{m}"] for m in meths], [s[f"median_{m}"] for m in meths])[0]
                key = ["method_a", "method_b"]
                pr = p_df[(p_df.design == design) & (p_df.stratum == stratum) & (p_df.metric == raw)].set_index(key)
                ps = p_df[(p_df.design == design) & (p_df.stratum == stratum) & (p_df.metric == sm)].set_index(key)
                both = pr.join(ps, lsuffix="_raw", rsuffix="_smooth", how="inner")
                same_dir = (np.sign(both.median_diff_a_minus_b_raw) == np.sign(both.median_diff_a_minus_b_smooth))
                sens.append({"design": design, "level": level, "stratum": stratum, "metric_raw": raw,
                             "metric_smooth": sm, "winner_raw": r["winner"], "winner_smooth": s["winner"],
                             "same_winner": r["winner"] == s["winner"], "spearman_method_medians": rho,
                             "pairs_same_direction": f"{int(same_dir.sum())}/{len(both)}",
                             "clear_pairs_raw": int(both.clear_raw.sum()),
                             "clear_pairs_smooth": int(both.clear_smooth.sum()),
                             "clear_pairs_flipped": int((both.clear_raw & both.clear_smooth & ~same_dir).sum())})
        pd.DataFrame(sens).to_csv(os.path.join(tab, "smooth_reference_sensitivity.csv"), index=False)

    # ── A2. Redundancy and discrimination of the metrics ──
    metric_redundancy(args.out, tab)

    # Matched units per (design, metric): all methods on the same gap × cluster units, so
    # rankings per seed/run and paired differences are not biased by missing values.
    matched = {}
    for (design, metric), g in prim.groupby(["design", "metric"]):
        piv, methods = _matched(g, extra=("duration_h", "phase"))
        if len(methods) >= 2 and len(piv):
            matched[(design, metric)] = (piv, methods)

    # ── B. Seed stability of the ranking ─────────────────────────────────────
    seed_rows = []
    for (design, metric), (piv, methods) in matched.items():
        ranks = piv.groupby("seed")[methods].median().rank(axis=1, method="min")
        for m in methods:
            seed_rows.append({"design": design, "metric": metric, "method": m,
                              "mean_rank": ranks[m].mean(), "sd_rank": ranks[m].std(),
                              "n_seeds_best": int((ranks[m] == 1).sum()), "n_seeds": len(ranks)})
    pd.DataFrame(seed_rows).to_csv(os.path.join(tab, "seed_rank_stability.csv"), index=False)

    # ── C. Results per independent reactor run ───────────────────────────────
    run_tabs = []
    for (design, metric), (piv, methods) in matched.items():
        t = piv.groupby("run")[methods].median()
        t["best"] = t.idxmin(axis=1)
        run_tabs.append(t.reset_index().assign(design=design, metric=metric, n_units=piv.groupby("run").size().values))
    pd.concat(run_tabs).to_csv(os.path.join(tab, "per_run_medians.csv"), index=False)

    # ── D. Gap duration and process phase ────────────────────────────────────
    prim = prim.assign(duration_bin=pd.cut(prim.duration_h, DURATION_BINS, include_lowest=True).astype(str))
    for col, fname in [("duration_bin", "by_duration.csv"), ("phase", "by_phase.csv")]:
        t = prim.groupby(["design", "metric", col, "method"]).value.agg(["median", "size"]).reset_index()
        t.to_csv(os.path.join(tab, fname), index=False)
    rho = []
    for (design, metric, method), g in prim.groupby(["design", "metric", "method"]):
        r, p = spearmanr(g.duration_h, g.value, nan_policy="omit")
        rho.append({"design": design, "metric": metric, "method": method, "spearman_rho_duration": r,
                    "p_naive": p, "n": len(g)})
    pd.DataFrame(rho).to_csv(os.path.join(tab, "duration_spearman.csv"), index=False)

    # D2. Does the ranking change with duration/phase? Paired difference (method − reference)
    #     per unit; < 0 = method better than the reference. Consistency of the trend over runs.
    diff_bins, diff_rho = [], []
    for (design, metric), (piv, methods) in matched.items():
        if REFERENCE_METHOD not in methods:
            continue
        bins = pd.cut(piv.duration_h, DURATION_BINS, include_lowest=True).astype(str)
        for m in methods:
            if m == REFERENCE_METHOD:
                continue
            d = piv[m] - piv[REFERENCE_METHOD]
            for col, key in [("duration_bin", bins), ("phase", piv.phase)]:
                for level, dd in d.groupby(key.values):
                    diff_bins.append({"design": design, "metric": metric, "method": m,
                                      "reference": REFERENCE_METHOD, "by": col, "level": level,
                                      "median_diff": dd.median(), "frac_method_better": float((dd < 0).mean()),
                                      "n": len(dd)})
            r, p = spearmanr(piv.duration_h, d)
            run_r = [spearmanr(g.duration_h, d.loc[g.index])[0] for _, g in piv.groupby("run")]
            diff_rho.append({"design": design, "metric": metric, "method": m, "reference": REFERENCE_METHOD,
                             "spearman_rho_duration": r, "p_naive": p, "n": len(d),
                             "runs_same_sign": int(np.sum(np.sign(run_r) == np.sign(r))), "n_runs": len(run_r)})
    pd.DataFrame(diff_bins).to_csv(os.path.join(tab, "paired_diff_by_duration_phase.csv"), index=False)
    pd.DataFrame(diff_rho).to_csv(os.path.join(tab, "paired_diff_duration_spearman.csv"), index=False)

    # D3. Breakdown per cluster: where does a difference to the reference come from?
    #     Paired difference (method − reference) per cluster by variable (gap × channel),
    #     experiment, duration class and phase (gap × cluster). rel_pct relative to the
    #     reference median of the same subgroup; frac_method_better = share of units where
    #     the method is better.
    uni_metrics = [m for m in primary if m in em.UNIVARIATE_METRICS]
    uni_rec = pd.read_csv(os.path.join(args.out, "records_univariate.csv.gz"),
                          usecols=UNIT_KEYS + ["cluster", "column", "method"] + uni_metrics)
    detail = []

    def _detail_rows(design, metric, cluster, piv, methods, by, levels):
        for m in methods:
            if m == REFERENCE_METHOD:
                continue
            d = piv[m] - piv[REFERENCE_METHOD]
            for level, idx in d.groupby(levels).groups.items():
                dd, ref = d.loc[idx], piv[REFERENCE_METHOD].loc[idx]
                detail.append({"design": design, "metric": metric, "cluster": cluster, "method": m,
                               "reference": REFERENCE_METHOD, "by": by, "level": level,
                               "median_diff": dd.median(), "reference_median": ref.median(),
                               "rel_pct": 100 * dd.median() / ref.median() if ref.median() else np.nan,
                               "frac_method_better": float((dd < 0).mean()), "n": len(dd)})

    for (design, metric), (piv, methods) in matched.items():
        if metric not in uni_metrics or REFERENCE_METHOD not in methods:
            continue
        piv = piv.assign(duration_bin=pd.cut(piv.duration_h, DURATION_BINS, include_lowest=True).astype(str))
        for cluster, g in piv.groupby("cluster"):
            for by in ["experiment", "duration_bin", "phase"]:
                _detail_rows(design, metric, cluster, g, methods, by, g[by].values)
        # per variable: unit = gap × channel (same method set, complete units only)
        col_piv = (uni_rec[uni_rec.design == design]
                   .pivot_table(index=UNIT_KEYS + ["cluster", "column"], columns="method", values=metric,
                                aggfunc="first")[methods].dropna().reset_index())
        for cluster, g in col_piv.groupby("cluster"):
            _detail_rows(design, metric, cluster, g, methods, "column", g["column"].values)
    pd.DataFrame(detail).to_csv(os.path.join(tab, "paired_diff_by_cluster_detail.csv"), index=False)

    # ── E. NRMSE scale and physical units ────────────────────────────────────
    uni = pd.read_csv(os.path.join(args.out, "records_univariate.csv.gz"))
    nr = uni.groupby(["design", "method"]).agg(
        median_NRMSE_local=("NRMSE_local", "median"),
        frac_NRMSE_local_ge_1=("NRMSE_local", lambda x: float((x >= 1).mean())),
        median_NRMSE_run=("NRMSE_run", "median")).reset_index()
    nr.to_csv(os.path.join(tab, "nrmse_scale.csv"), index=False)
    if "NoiseFloor_run" in uni.columns:
        # Noise floor: error share no method can reproduce (RMS of true − smoothed true).
        # Per gap × channel; median per cluster and method. ratio = NRMSE_run / NoiseFloor_run (1 = noise only).
        nf = uni.assign(ratio=uni.NRMSE_run / uni.NoiseFloor_run.replace(0, np.nan))
        nf.groupby(["design", "cluster", "method"]).agg(
            median_noise_floor=("NoiseFloor_run", "median"), median_NRMSE_run=("NRMSE_run", "median"),
            median_NRMSE_run_smooth=("NRMSE_run_smooth", "median"),
            median_ratio_to_noise=("ratio", "median")).reset_index().to_csv(
            os.path.join(tab, "noise_floor.csv"), index=False)
    phys = (uni[uni.column.isin(KEY_VARIABLES)].groupby(["design", "column", "method"])
            .RMSE.agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
            .reset_index())
    phys.to_csv(os.path.join(tab, "rmse_physical_units.csv"), index=False)

    # ── F. Correlation structure of the imputations ──────────────────────────
    cor = pd.read_csv(os.path.join(args.out, "records_correlation.csv.gz"))
    cor_tab = cor.groupby(["design", "cluster", "n_cols", "method"]).agg(
        median_mean_abs_r_true=("MeanAbsR_true", "median"),
        median_mean_abs_r_pred=("MeanAbsR_pred", "median"),
        frac_pred_abs_r_eq_1=("MeanAbsR_pred", lambda x: float((x > 0.999).mean())),
        frac_pred_with_undefined_r=("n_undefined_pred", lambda x: float((x > 0).mean())),
        frac_true_with_undefined_r=("n_undefined_true", lambda x: float((x > 0).mean())),
        n_segments=("MeanAbsR_pred", "size")).reset_index()
    cor_tab.to_csv(os.path.join(tab, "correlation_structure.csv"), index=False)

    # ── G. Hankel diagnostics ────────────────────────────────────────────────
    diag_path = os.path.join(args.out, "hankel_diagnostics.csv")
    if os.path.exists(diag_path):
        dg = pd.read_csv(diag_path)
        dg = dg[~dg.skipped.astype(bool)]
        hd = dg.groupby(["n_gaps", "cluster"]).agg(
            n=("n_iter", "size"), iter_median=("n_iter", "median"), iter_max=("n_iter", "max"),
            converged_frac=("converged", "mean"),
            rms_change_from_linear_z_median=("rms_change_from_linear_z", "median"),
            rms_change_from_linear_z_q90=("rms_change_from_linear_z", lambda x: x.quantile(.9)),
            hankel_rows_min=("hankel_rows", "min"), hankel_rows_max=("hankel_rows", "max"),
            hankel_cols_min=("hankel_cols", "min"), hankel_cols_max=("hankel_cols", "max"),
            rows_ge_k_all=("hankel_rows", lambda x: bool((x >= dg.loc[x.index, "k"]).all()))).reset_index()
        hd.to_csv(os.path.join(tab, "hankel_diagnostics_summary.csv"), index=False)

    with open(os.path.join(tab, "stats_config.json"), "w", encoding="utf-8") as f:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_boot": args.n_boot,
                   "boot_seed": BOOT_SEED, "alpha": ALPHA, "metrics": metrics,
                   "primary_design": PRIMARY_DESIGN,
                   "roles": {"confirmatory": f"{PRIMARY_DESIGN}, per cluster, primary metrics",
                             "summary": f"{PRIMARY_DESIGN}, global, primary metrics",
                             "exploratory": f"{PRIMARY_DESIGN}, per gap count or non-primary metrics",
                             "sensitivity": "other design(s)"},
                   "primary_metrics": em.PRIMARY_METRICS, "duration_bins": DURATION_BINS,
                   "methods_present": sorted(units.method.unique()),
                   "phase_definition": {"signal": em.PHASE_SIGNAL, "early_fraction": em.PHASE_EARLY_FRACTION,
                                        "plateau_fraction": em.PHASE_PLATEAU_FRACTION,
                                        "reference_quantile": em.PHASE_REFERENCE_QUANTILE}},
                  f, indent=2)

    # ── Overview ─────────────────────────────────────────────────────────
    if summaries:
        s = pd.DataFrame(summaries)
        cols = ["stratum", "metric", "winner", "runner_up", "winner_clear", "winner_vs_runner_p_holm",
                "winner_vs_runner_runs_consistent", "winner_vs_runner_seeds_consistent",
                "kendall_w_runs", "n_units", "n_runs"]
        for role in ["confirmatory", "summary"]:
            g = s[s.role == role]
            if len(g):
                print(f"\n{role.upper()} ({PRIMARY_DESIGN}):")
                print(g[cols].to_string(index=False))
    print(f"\nTables → {tab}  ({(time.time() - t0) / 60:.1f} min)")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("records", help="compute per-gap metrics")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("stats", help="statistics and tables")
    s.add_argument("--n-boot", type=int, default=N_BOOT)
    s.add_argument("--all-metrics", action="store_true", help="all metrics instead of the primary ones only")
    for p in (r, s):
        p.add_argument("--results", default=cfg.RESULTS_DIR)
        p.add_argument("--sim-dir", default=cfg.SIM_DIR)
        p.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")         # "→" in messages (Windows console)
    return records(args) if args.step == "records" else stats(args)


if __name__ == "__main__":
    sys.exit(main())
