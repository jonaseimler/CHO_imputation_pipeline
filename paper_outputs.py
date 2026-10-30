"""
paper_outputs.py
─────────────────────────────────────────────────────────────────────────────
Figures and tables of the manuscript. Reads only the finished results of
  evaluate_multiseed.py      (results/evaluation_multiseed/)
  downstream_evaluation.py   (results/evaluation_downstream/)
  real_outages.py            (results/real_outages/)
  revision_diagnostics.py    (results/evaluation_multiseed/diagnostics/)
  Optuna logs                (results/optimization/)
  imputations                (results/imputed_multiseed/, example trajectories and runtimes only)
and does not recompute any statistics.

Output (OUT_DIR = results/paper):
  figures/   F*.png (300 dpi) + F*.pdf
  tables/    T*.csv, S*.csv           – tables of the main text and supplement
  repository/ *.csv                   – tables not cited in the manuscript

Main text:  F3 trajectories, F4 main result, F5 duration/phase, F6 real outages,
            F7 downstream; T1 real outages. The numbers behind F4/F7 are in S3a/S11e.
Supplement: S1–S16 (tables), SF1 results per run.
(F1 correlation heatmap = cluster_columns.py; F2 flow chart is part of the manuscript.)

  python paper_outputs.py
  python paper_outputs.py --eval <dir> --down <dir> --results <dir>   # e.g. test data
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

import evaluation_metrics as em
import revision_config as cfg
from revision_diagnostics import PERIOD_MAX_H, PERIOD_MIN_H, PERIODIC_ACF
from run_imputation_multiseed import load_unit, result_path
from simulate_gaps_multiseed import load_scenarios

OUT_DIR = str(cfg.paths.PAPER_DIR)
EVAL_DIR = str(cfg.paths.EVAL_DIR)
DIAG_DIR = str(cfg.paths.DIAG_DIR)
DOWN_DIR = str(cfg.paths.DOWNSTREAM_DIR)
REAL_DIR = str(cfg.paths.REAL_OUTAGE_DIR)

# ── Style ────────────────────────────
DPI = 300
MAX_WIDTH_INCH = 170 / 25.4
MAX_HEIGHT_INCH = 225 / 25.4
FONT_FAMILY = "Times New Roman"
METHOD_COLORS = {
    "Linear": "#4878CF", "Linear_Smoothed": "#6ACC65", "Spline": "#D65F5F",
    "Spline_Smoothed": "#B47CC7", "FLk-NN": "#C4AD66", "Hankel": "#77BEDB",
}
REFERENCE = "Linear"
PRIMARY_DESIGN = "unfixed"

METRIC_LABELS = {
    "NRMSE_run": "NRMSE", "NDTW_run": "NDTW", "FSD": "FSD", "MAD_OffDiag": "MAD of correlations",
    "NRMSE_run_smooth": "NRMSE (smoothed ref.)", "NDTW_run_smooth": "NDTW (smoothed ref.)",
    "FSD_smooth": "FSD (smoothed ref.)", "MAD_OffDiag_smooth": "MAD of correlations (smoothed ref.)",
    "VCD_RMSE": "VCD RMSE (10$^6$ cells/mL)", "mu_RMSE": "$\\mu$ RMSE (1/h)",
    "T2_logRMSE": "RMSE of log$_{10}$ T$^2$", "Q_logRMSE": "RMSE of log$_{10}$ Q",
    "VCD_relRMSE": "relative VCD RMSE", "IVCD_relErr": "relative IVCD error",
    "T2_alarm_disagree": "T$^2$ alarm disagreement", "Q_alarm_disagree": "Q alarm disagreement",
}
METRIC_SHORT = {"MAD_OffDiag": "MAD (corr.)", "VCD_RMSE": "VCD RMSE", "mu_RMSE": "$\\mu$ RMSE",
                "T2_logRMSE": "log$_{10}$ T$^2$ RMSE", "Q_logRMSE": "log$_{10}$ Q RMSE"}
METRIC_PLAIN = {k: (v.replace("$^6$", "^6").replace("$\\mu$", "mu").replace("log$_{10}$", "log10")
                     .replace("T$^2$", "T2")) for k, v in METRIC_LABELS.items()}
VAR_LABELS = {
    "Permittivity(pF/cm)-VCD": "Permittivity (pF/cm)",
    "CO2(%-sat)-CO2": "Dissolved CO$_2$ (%-sat)",
    "TCD(AU)-TCD": "TCD (AU)",
    "Conductivity(mS/cm)-VCD": "Conductivity (mS/cm)",
    "Alpha-VCD": "Cole–Cole α",
    "Temperature(°C)-VCD": "Temperature (°C)",
}
PHASE_ORDER = ["early", "exponential", "stationary/decline"]
TRAJ_VARS = ["Permittivity(pF/cm)-VCD", "CO2(%-sat)-CO2"]


def name(method: str) -> str:
    return cfg.METHODS.get(method, method)


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": DPI,
        "savefig.dpi": DPI, "savefig.bbox": "tight", "mathtext.fontset": "stix",
    })
    return plt


# ── Output helpers ────────────────────────────────────────────────────────────
class Out:
    def __init__(self, root: str):
        self.root = root
        for sub in ("figures", "tables", "repository"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        self.written = []
        self.notes = []          # key numbers from tables and figures (→ key_numbers.md)

    def note(self, section: str, key: str, value, source: str):
        self.notes.append({"section": section, "key": key, "value": value, "source": source})

    def figure(self, fig, key: str):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(self.root, "figures", f"{key}.{ext}"))
        import matplotlib.pyplot as plt
        plt.close(fig)
        self.written.append(f"figures/{key}")

    def table(self, df: pd.DataFrame, key: str, description: str = "", repo: bool = False):
        """repo=True: not cited in the manuscript → repository/ only (description as header line)."""
        if repo:
            path = os.path.join(self.root, "repository", f"{key}.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(f"# {' '.join(description.split())}\n")
                df.to_csv(f, index=False)
            self.written.append(f"repository/{key}")
            return
        df.to_csv(os.path.join(self.root, "tables", f"{key}.csv"), index=False, encoding="utf-8")
        self.written.append(f"tables/{key}")


def fmt(x, sig: int = 3) -> str:
    """Number with `sig` significant digits; NaN → '–'."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 10 ** sig or abs(x) < 1e-3:
        return f"{x:.{sig - 1}e}"
    return f"{x:.{sig}g}"


def fmt_ci(est, lo, hi, sig: int = 3) -> str:
    return f"{fmt(est, sig)} [{fmt(lo, sig)}, {fmt(hi, sig)}]"


def fmt_p(p) -> str:
    if not np.isfinite(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def panel_label(ax, i: int):
    ax.text(-0.01, 1.02, f"({chr(97 + i)})", transform=ax.transAxes, fontweight="bold",
            ha="right", va="bottom", fontsize=10)


def read(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


def vs_reference(pairs: pd.DataFrame, ref: str = REFERENCE) -> pd.DataFrame:
    """Pairwise comparisons as difference method − reference (negative = method better)."""
    rows = []
    for r in pairs.itertuples():
        if ref not in (r.method_a, r.method_b):
            continue
        sign = 1 if r.method_b == ref else -1          # median_diff = a − b
        m = r.method_a if sign == 1 else r.method_b
        lo, hi = (r.ci_low, r.ci_high) if sign == 1 else (-r.ci_high, -r.ci_low)
        rows.append({"stratum": r.stratum, "metric": r.metric, "method": m,
                     "diff": sign * r.median_diff_a_minus_b, "ci_low": lo, "ci_high": hi,
                     "p_holm": r.p_holm, "clear": bool(r.clear), "runs": r.runs_consistent,
                     "n_runs": r.n_runs, "seeds": r.seeds_consistent, "n_seeds": r.n_seeds,
                     "rank_biserial_method_better": sign * r.rank_biserial_a_better})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Tables
# ══════════════════════════════════════════════════════════════════════════════
def with_relative(d: pd.DataFrame, ref_median: float) -> pd.DataFrame:
    """Relative effect size: difference (and CI) in % of the reference median (practical relevance)."""
    scale = 100 / ref_median if ref_median and np.isfinite(ref_median) else np.nan
    return d.assign(rel=d["diff"] * scale, rel_low=d.ci_low * scale, rel_high=d.ci_high * scale)


def fmt_rel(x) -> str:
    return "–" if not np.isfinite(x) else f"{x:+.1f}%".replace("+0.0%", "0.0%").replace("-0.0%", "0.0%")


def reference_table(summary: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Result table per stratum × metric: reference median (Linear) and paired difference of each
    other method to the reference [95 % CI], * = clear, with relative difference (% of the reference
    median) and runs with the same sign."""
    rows = []
    for _, s in summary.iterrows():      # not itertuples: "median_FLk-NN" is not a valid attribute name
        p = pairs[(pairs.design == s.design) & (pairs.level == s.level) & (pairs.stratum == s.stratum)
                  & (pairs.metric == s.metric)]
        d = with_relative(vs_reference(p), s[f"median_{REFERENCE}"]).set_index("method")
        row = {"Stratum": s.stratum, "Metric": METRIC_PLAIN.get(s.metric, s.metric),
               f"{name(REFERENCE)}, median (IQR)": f"{fmt(s[f'median_{REFERENCE}'])} ({s[f'iqr_{REFERENCE}']})"}
        for m in cfg.METHODS:
            if m == REFERENCE:
                continue
            if m not in d.index:
                row[name(m)] = "–"
                continue
            r = d.loc[m]
            row[name(m)] = (f"{fmt_ci(r['diff'], r.ci_low, r.ci_high)}{'*' if r.clear else ''} "
                            f"({fmt_rel(r.rel)}; {int(r.runs)}/{int(r.n_runs)})")
        row.update({"Runs": int(s.n_runs), "Gap events": int(s.n_gap_events), "Units": int(s.n_units)})
        rows.append(row)
    return pd.DataFrame(rows)


def medians_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in summary.iterrows():
        row = {"Stratum": s.stratum, "Metric": METRIC_PLAIN.get(s.metric, s.metric)}
        for m in cfg.METHODS:
            med = s.get(f"median_{m}", np.nan)
            iqr = s.get(f"iqr_{m}", "")
            row[name(m)] = f"{fmt(med)} ({iqr})" if pd.notna(med) else "–"
        rows.append(row)
    return pd.DataFrame(rows)


def tables(out: Out, a) -> dict:
    info = {}
    summ = read(os.path.join(a.eval, "tables", "comparisons_summary.csv"))
    pairs = read(os.path.join(a.eval, "tables", "comparisons_pairwise.csv"))

    # T1 – real outages
    ev = read(os.path.join(a.real, "outage_events.csv"))
    if ev is not None:
        # duration = points × sampling interval (as gap_length_detection.py)
        ev = ev.assign(duration_h=ev.n_points * ev.dt_s / 3600)
        # both reactors of an experiment fail together → one row per experiment × outage
        g = ev.groupby(["experiment", "event_id"])
        spread = g.agg(s=("start_h", np.ptp), e=("end_h", np.ptp), d=("duration_h", np.ptp)).max().max()
        g = g.agg(reactors=("reactor", lambda x: ", ".join(sorted(set(x)))), start=("start_h", "min"),
                  end=("end_h", "max"), dur=("duration_h", "max"), modules=("modules_missing", "first"),
                  n_cols=("n_cols_missing", "first")).reset_index()
        assert (g.reactors == "R01, R02").all(), "Outage in one reactor only – T1 assumes both reactors"
        out.note("1 Data and design", "Real outages: max. difference of start, end or duration between reactors, min",
                 f"{60 * spread:.0f}", "T1")
        out.note("1 Data and design", "Real outages: missing variables, full / partial outage",
                 f"{g.n_cols.max()} / {g.n_cols.min()}", "T1")
        t1 = pd.DataFrame({
            "Experiment": g.experiment, "Outage": g.event_id + 1,
            "Start–end (h)": [f"{r.start:.1f}–{r.end:.1f}" for r in g.itertuples()],
            "Duration (h)": g.dur.round(1)})
        out.table(t1, "T1_real_outages")

    if summ is not None:
        conf = summ[summ.role == "confirmatory"]
        # S3a, S3c, S5a, S6 – comparisons with the Linear Baseline
        out.table(reference_table(conf, pairs), "S3a_main_results")
        out.table(medians_table(conf), "medians_all_methods",
                  "Median (interquartile range) of each primary metric per method and sensor cluster (random "
                  "placement, held-out experiments).", repo=True)

        s = summ[(summ.role == "summary")]
        out.table(reference_table(s, pairs), "S3c_global_summary")
        ex = summ[(summ.design == PRIMARY_DESIGN) & (summ.level == "n_gaps") & summ.primary]
        out.table(reference_table(ex, pairs), "S6_by_gap_count")
        fx = summ[(summ.design == "fixed") & (summ.level == "cluster") & summ.primary]
        out.table(reference_table(fx, pairs), "S5a_fixed_placement")

        # Repository – all pairwise comparisons (main analysis)
        p = pairs[pairs.role == "confirmatory"].copy()
        s3 = pd.DataFrame({
            "Stratum": p.stratum, "Metric": p.metric.map(METRIC_PLAIN),
            "Method A": p.method_a.map(name), "Method B": p.method_b.map(name),
            "Median difference A - B [95% CI]": [fmt_ci(r.median_diff_a_minus_b, r.ci_low, r.ci_high) for r in p.itertuples()],
            "Holm-adjusted p": p.p_holm.map(fmt_p),
            "Rank-biserial r (A better)": p.rank_biserial_a_better.map(lambda x: f"{x:.2f}"),
            "Consistent runs": [f"{r.runs_consistent}/{r.n_runs}" for r in p.itertuples()],
            "Consistent seeds": [f"{r.seeds_consistent}/{r.n_seeds}" for r in p.itertuples()],
            "Clear": p.clear.map({True: "yes", False: "no"})})
        out.table(s3, "all_pairwise",
                  "All pairwise method comparisons of the confirmatory analysis (random placement, per sensor "
                  "cluster and primary metric). Negative differences favour method A.", repo=True)

    # S5b – sensitivity, smoothed reference
    sm = read(os.path.join(a.eval, "tables", "smooth_reference_sensitivity.csv"))
    if sm is not None:
        sm = sm[(sm.design == PRIMARY_DESIGN) & (sm.level == "cluster")]
        out.table(pd.DataFrame({
            "Stratum": sm.stratum, "Metric": sm.metric_raw.map(METRIC_PLAIN),
            "Best method (raw reference)": sm.winner_raw.map(name),
            "Best method (smoothed reference)": sm.winner_smooth.map(name),
            "Spearman rho of method medians": sm.spearman_method_medians.map(lambda x: f"{x:.2f}"),
            "Pairs with same direction": sm.pairs_same_direction,
            "Clear pairs (raw)": sm.clear_pairs_raw, "Clear pairs (smoothed)": sm.clear_pairs_smooth,
            "Clear pairs reversed": sm.clear_pairs_flipped}), "S5b_smoothed_reference")

    # S4 – per run and per seed
    pr = read(os.path.join(a.eval, "tables", "per_run_medians.csv"))
    if pr is not None:
        pr = pr[pr.design == PRIMARY_DESIGN]
        cols = [m for m in cfg.METHODS if m in pr.columns]
        s4 = pr[["run", "metric"] + cols + ["best"]].copy()
        s4["metric"] = s4.metric.map(METRIC_PLAIN)
        for m in cols:
            s4[m] = s4[m].map(fmt)
        s4["best"] = s4.best.map(name)
        s4 = s4.rename(columns={"run": "Run", "metric": "Metric", "best": "Best method", **{m: name(m) for m in cols}})
        out.table(s4, "S4a_per_run")
    sr = read(os.path.join(a.eval, "tables", "seed_rank_stability.csv"))
    if sr is not None:
        sr = sr[sr.design == PRIMARY_DESIGN]
        out.table(pd.DataFrame({"Metric": sr.metric.map(METRIC_PLAIN), "Method": sr.method.map(name),
                                "Mean rank": sr.mean_rank.map(lambda x: f"{x:.2f}"),
                                "SD of rank": sr.sd_rank.map(lambda x: f"{x:.2f}"),
                                "Seeds ranked first": [f"{r.n_seeds_best}/{r.n_seeds}" for r in sr.itertuples()]}),
                  "S4b_seed_rank_stability")

    # S7 – metric selection
    md = read(os.path.join(a.eval, "tables", "metric_discrimination.csv"))
    if md is not None:
        out.table(pd.DataFrame({"Metric": md.metric, "Family": md.family,
                                "Primary": md.primary.map({True: "yes", False: "no"}),
                                "Kendall's W (all methods)": md.kendall_w.map(lambda x: f"{x:.3f}"),
                                "Kendall's W (without FLk-NN)": md.kendall_w_without_flknn.map(lambda x: f"{x:.3f}"),
                                "Units": md.n_units}),
                  "S7a_metric_discrimination")
    rs = read(os.path.join(a.eval, "tables", "metric_redundancy_spearman.csv"))
    if rs is not None:
        rs = rs.rename(columns={rs.columns[0]: "Metric"})
        for c in rs.columns[1:]:
            rs[c] = rs[c].map(lambda x: f"{x:.2f}")
        out.table(rs, "S7b_metric_redundancy")

    # S8 – error scale
    ns = read(os.path.join(a.eval, "tables", "nrmse_scale.csv"))
    nf = read(os.path.join(a.eval, "tables", "noise_floor.csv"))
    if ns is not None:
        ns = ns[ns.design == PRIMARY_DESIGN].assign(cluster="all")
        bc = _error_scale_by_cluster(a.eval)
        if bc is not None:
            ns = pd.concat([ns, bc], ignore_index=True)
        out.table(pd.DataFrame({"Cluster": ns.cluster, "Method": ns.method.map(name),
                                "Median NRMSE (gap-normalized)": ns.median_NRMSE_local.map(fmt),
                                "Share of units with gap-normalized NRMSE >= 1": ns.frac_NRMSE_local_ge_1.map(lambda x: f"{100 * x:.1f}%"),
                                "Median NRMSE (run-normalized)": ns.median_NRMSE_run.map(fmt)}),
                  "S8a_error_scale")
    if nf is not None:
        nf = nf[nf.design == PRIMARY_DESIGN]
        out.table(pd.DataFrame({"Cluster": nf.cluster, "Method": nf.method.map(name),
                                "Median noise floor": nf.median_noise_floor.map(fmt),
                                "Median NRMSE": nf.median_NRMSE_run.map(fmt),
                                "Median NRMSE (smoothed ref.)": nf.median_NRMSE_run_smooth.map(fmt),
                                "NRMSE / noise floor": nf.median_ratio_to_noise.map(fmt)}),
                  "S8b_noise_floor")
    ph = read(os.path.join(a.eval, "tables", "rmse_physical_units.csv"))
    if ph is not None:
        ph = ph[ph.design == PRIMARY_DESIGN]
        out.table(pd.DataFrame({"Variable": ph.column.map(lambda c: VAR_LABELS.get(c, c).replace("$_2$", "2")),
                                "Method": ph.method.map(name),
                                "Median RMSE (IQR)": [f"{fmt(r.median)} ({fmt(r.q25)}–{fmt(r.q75)})" for r in ph.itertuples()]}),
                  "S8c_rmse_physical_units")

    # S9 – Hankel diagnostics, distance from Linear
    hd = read(os.path.join(a.eval, "tables", "hankel_diagnostics_summary.csv"))
    if hd is not None:
        out.table(pd.DataFrame({
            "Gaps": hd.n_gaps, "Cluster": hd.cluster, "Gap events": hd.n,
            "Iterations, median (max)": [f"{r.iter_median:.0f} ({r.iter_max:.0f})" for r in hd.itertuples()],
            "Converged": hd.converged_frac.map(lambda x: f"{100 * x:.0f}%"),
            "RMS change from linear initialization (z), median (90th pct.)":
                [f"{fmt(r.rms_change_from_linear_z_median)} ({fmt(r.rms_change_from_linear_z_q90)})" for r in hd.itertuples()],
            "Hankel rows": [f"{r.hankel_rows_min}–{r.hankel_rows_max}" for r in hd.itertuples()],
            "Hankel columns": [f"{r.hankel_cols_min}–{r.hankel_cols_max}" for r in hd.itertuples()]}),
            "S9a_hankel_diagnostics")
    dv = read(os.path.join(a.down, "tables", "deviation_from_linear.csv"))
    if dv is not None:
        dv = dv[dv.design == PRIMARY_DESIGN]
        out.table(pd.DataFrame({"Cluster": dv.cluster, "Method": dv.method.map(name),
                                "Median ratio (IQR)": [f"{fmt(r.median_ratio)} ({fmt(r.q25_ratio)}–{fmt(r.q75_ratio)})" for r in dv.itertuples()],
                                "RMS(truth − linear), run-normalized": dv.median_true_minus_linear_z.map(fmt),
                                "RMS(prediction − linear), run-normalized": dv.median_pred_minus_linear_z.map(fmt),
                                "Gap × variable units": dv.n}),
                  "S9b_deviation_from_linear")

    # S12 – breakdown Hankel vs. Linear in Cluster 3 (Sec. 3.3)
    det = read(os.path.join(a.eval, "tables", "paired_diff_by_cluster_detail.csv"))
    if det is not None:
        det = det[(det.design == PRIMARY_DESIGN) & (det.method == "Hankel") & (det.cluster == 3)
                  & det.metric.isin(["NRMSE_run", "NDTW_run", "FSD"])].copy()
        by_label = {"column": "Variable", "experiment": "Experiment", "duration_bin": "Gap duration",
                    "phase": "Process phase"}
        det["order"] = det.by.map({k: i for i, k in enumerate(by_label)})
        det["lvl"] = [VAR_LABELS.get(l, l).replace("$_2$", "2") if b == "column" else
                      (f"{float(l.strip('([').split(',')[0]):.0f}–{float(l.strip(')]').split(',')[1]):.0f} h"
                       if b == "duration_bin" else l) for b, l in zip(det.by, det.level)]
        det["lvl_sort"] = [float(l.strip("([").split(",")[0]) if b == "duration_bin" else
                           (PHASE_ORDER.index(l) if b == "phase" else 0) for b, l in zip(det.by, det.level)]
        det = det.sort_values(["metric", "order", "lvl_sort", "lvl"],
                              key=lambda s: s.map({"NRMSE_run": 0, "NDTW_run": 1, "FSD": 2}) if s.name == "metric" else s)
        out.table(pd.DataFrame({"Metric": det.metric.map(METRIC_PLAIN), "Breakdown": det.by.map(by_label),
                                "Level": det.lvl, "Linear Baseline, median": det.reference_median.map(fmt),
                                "Median difference Hankel − Linear": det.median_diff.map(fmt),
                                "Relative difference": det.rel_pct.map(fmt_rel),
                                "Share of units with Hankel better": det.frac_method_better.map(lambda x: f"{100 * x:.0f}%"),
                                "Units": det.n}),
                  "S12_hankel_cluster3_breakdown")

    # S10 – correlation structure
    cs = read(os.path.join(a.eval, "tables", "correlation_structure.csv"))
    if cs is not None:
        cs = cs[cs.design == PRIMARY_DESIGN]
        # which runs are behind a (cluster, number of variables) pair?
        rc = pd.read_csv(os.path.join(a.eval, "records_correlation.csv.gz"),
                         usecols=["design", "cluster", "n_cols", "run"])
        runs = (rc[rc.design == PRIMARY_DESIGN].groupby(["cluster", "n_cols"]).run
                .agg(lambda x: ", ".join(r.replace("_", " ") for r in sorted(set(x)))))
        full = cs.groupby("cluster").n_cols.transform("max")
        variable_set = ["complete" if n == f else "reduced (no capacitance spectrum)"
                        for n, f in zip(cs.n_cols, full)]
        out.table(pd.DataFrame({"Cluster": cs.cluster, "Variables": cs.n_cols, "Variable set": variable_set,
                                "Runs": [runs.get((c, n), "") for c, n in zip(cs.cluster, cs.n_cols)],
                                "Method": cs.method.map(name),
                                "Median mean |r| (true)": cs.median_mean_abs_r_true.map(lambda x: f"{x:.2f}"),
                                "Median mean |r| (reconstruction)": cs.median_mean_abs_r_pred.map(lambda x: f"{x:.2f}"),
                                "Share with |r| = 1": cs.frac_pred_abs_r_eq_1.map(lambda x: f"{100 * x:.0f}%"),
                                "Share with undefined r (reconstruction)": cs.frac_pred_with_undefined_r.map(lambda x: f"{100 * x:.0f}%")}),
                  "S10_correlation_structure")

    # S1 – hyperparameters
    info.update(hyperparameter_tables(out))

    # S2 – simulated gaps
    pos = read(os.path.join(a.sim, "gap_positions.csv"))
    if pos is not None:
        g = pos.groupby(["design", "n_gaps"]).agg(
            n=("duration_h", "size"), dmin=("duration_h", "min"), dmed=("duration_h", "median"),
            dmax=("duration_h", "max"), smin=("start_h", "min"), emax=("end_h", "max")).reset_index()
        out.table(pd.DataFrame({"Placement": g.design, "Gaps per run": g.n_gaps, "Simulated gaps": g.n,
                                "Duration (h), median (min–max)": [f"{r.dmed:.1f} ({r.dmin:.1f}–{r.dmax:.1f})" for r in g.itertuples()],
                                "Covered time span (h)": [f"{r.smin:.1f}–{r.emax:.1f}" for r in g.itertuples()]}),
                  "S2_simulated_gaps")

    # Downstream: S11
    dsum = read(os.path.join(a.down, "tables", "downstream_comparisons_summary.csv"))
    dpairs = read(os.path.join(a.down, "tables", "downstream_comparisons_pairwise.csv"))
    if dsum is not None:
        conf = dsum[dsum.role == "confirmatory"]
        dmed = read(os.path.join(a.down, "tables", "downstream_medians.csv"))
        out.table(reference_table(conf, dpairs).drop(columns=["Stratum"]), "S11e_downstream")
        if dmed is not None:
            dmed = dmed[dmed.design == PRIMARY_DESIGN].set_index("method")
            mets = list(METRIC_PLAIN)[8:16]
            # alarm disagreement: median per gap is 0 (most gaps without alarm) → report the mean
            alarm = _alarm_means(a.down)
            for k in ("T2_alarm_disagree", "Q_alarm_disagree"):
                if alarm is not None and k in dmed:
                    dmed[k] = alarm[k]
            out.table(pd.DataFrame([{"Method": name(m), **{METRIC_PLAIN[k]: fmt(dmed.loc[m, k]) for k in mets if k in dmed}}
                                    for m in cfg.METHODS if m in dmed.index]),
                      "S11a_downstream_medians")
    mc = read(os.path.join(a.down, "model_calibration.csv"))
    if mc is not None:
        r = mc.iloc[0]
        info["pls"] = r
        rmsec = _pls_rmse_calibration(a.down)
        out.table(pd.DataFrame({"Test experiment": mc.test_experiment,
                                "PLS components": mc.pls_components,
                                "PLS RMSE, calibration (10^6 cells/mL)": f"{rmsec:.2f}" if rmsec is not None else "–",
                                "PLS RMSECV, leave-one-experiment-out (10^6 cells/mL)": mc.pls_rmsecv_loeo.map(lambda x: f"{x:.2f}"),
                                "PLS R2 per test run": mc.pls_r2_test_runs,
                                "PCA components": mc.pca_components,
                                "T2 limit (99%)": mc.t2_limit.map(lambda x: f"{x:.1f}"),
                                "Q limit (99%)": mc.q_limit.map(lambda x: f"{x:.2f}")}),
                  "S11b_downstream_models")
    pl = read(os.path.join(a.real, "plausibility_summary.csv"))
    if pl is not None:
        out.table(pd.DataFrame({"Cluster": pl.cluster, "Method": pl.method.map(name),
                                "Outside context range": pl.out_of_range_frac.map(lambda x: f"{100 * x:.1f}%"),
                                "Roughness ratio": pl.roughness_ratio.map(fmt),
                                "Boundary jump": pl.boundary_jump.map(fmt),
                                "RMS deviation from linear": pl.rms_dev_linear.map(fmt)}),
                  "S11c_real_outage_plausibility")
    pv = read(os.path.join(a.real, "plausibility.csv"))
    if pv is not None:
        pv = pv[pv.column.isin(REAL_KEY_VARS)]
        g = (pv.groupby(["column", "method"])[["out_of_range_frac", "roughness_ratio", "boundary_jump", "rms_dev_linear"]]
             .median().reset_index())
        g["order"] = g.column.map(REAL_KEY_VARS.index) * 10 + g.method.map(list(cfg.METHODS).index)
        g = g.sort_values("order")
        out.table(pd.DataFrame({"Variable": g.column.map(lambda c: VAR_LABELS.get(c, c).replace("$_2$", "2")),
                                "Method": g.method.map(name),
                                "Outside context range": g.out_of_range_frac.map(lambda x: f"{100 * x:.1f}%"),
                                "Roughness ratio": g.roughness_ratio.map(fmt),
                                "Boundary jump": g.boundary_jump.map(fmt),
                                "RMS deviation from linear": g.rms_dev_linear.map(fmt)}),
                  "S11d_real_outage_plausibility_variables")
        out.note("8 Real outages (plausibility, no ground truth)", "Outage events (reactors × outages) behind S11d",
                 pv.groupby(["experiment", "reactor", "event_id"]).ngroups, "S11d")
    info.update(diagnostic_tables(out, a))
    return info


# ── Supplementary tables from records and diagnostics (Sec. 3.4, 3.5, 3.7–3.9) ───────
REAL_KEY_VARS = ["Permittivity(pF/cm)-VCD", "TCD(AU)-TCD", "CO2(%-sat)-CO2"]


def _error_scale_by_cluster(eval_dir: str) -> pd.DataFrame | None:
    """S8a per sensor cluster (gap × variable), same indicators as nrmse_scale.csv."""
    path = os.path.join(eval_dir, "records_univariate.csv.gz")
    if not os.path.exists(path):
        return None
    u = pd.read_csv(path, usecols=["design", "method", "cluster", "NRMSE_local", "NRMSE_run"])
    u = u[u.design == PRIMARY_DESIGN]
    return (u.groupby(["cluster", "method"])
            .agg(median_NRMSE_local=("NRMSE_local", "median"),
                 frac_NRMSE_local_ge_1=("NRMSE_local", lambda x: float((x >= 1).mean())),
                 median_NRMSE_run=("NRMSE_run", "median"), n=("NRMSE_local", "size"))
            .reset_index().assign(cluster=lambda d: d.cluster.astype(str)))


def _pls_rmse_calibration(down_dir: str) -> float | None:
    """RMSE of the PLS model on its own calibration samples (fitting error)."""
    p = read(os.path.join(down_dir, "pls_calibration_samples.csv"))
    return None if p is None else float(np.sqrt(np.mean((p.pred - p.vcd) ** 2)))


def _runtimes(results_dir: str) -> pd.DataFrame | None:
    """Runtime per reactor run (all clusters, all gaps) from the imputation log."""
    log = read(os.path.join(results_dir, "run_log.csv"))
    if log is None:
        return None
    log = log[log.status == "ok"].drop_duplicates(
        subset=["design", "seed", "n_gaps", "experiment", "reactor", "method"], keep="last")
    return log[log.design == PRIMARY_DESIGN]


def _flknn_by_cluster(diag_dir: str) -> pd.DataFrame | None:
    comp = read(os.path.join(diag_dir, "flknn_components.csv"))
    lags = read(os.path.join(diag_dir, "flknn_lags.csv"))
    if comp is None:
        return None
    g = comp.groupby("cluster").agg(
        units=("knn_coverage", "size"), runs=("run", "nunique"),
        seeds=("seed", lambda x: ", ".join(map(str, sorted(set(x))))),
        knn_cov=("knn_coverage", "median"), knn_full=("knn_coverage", lambda x: float((x > 0.99).mean())),
        fourier_med=("rms_fourier_minus_linear", "median"), fourier_max=("rms_fourier_minus_linear", "max"),
        flknn_lin=("rms_flknn_minus_linear", "median"), stitch=("rms_stitch_correction", "median"),
        r2_line=("r2_line_flknn_minus_fourier", "median"),
        r2_line_099=("r2_line_flknn_minus_fourier", lambda x: float((x.dropna() > 0.99).mean())),
        rough=("roughness_ratio", "median"))
    if lags is not None and len(lags):
        lags = lags.assign(abs_lag=lags.lag.abs())
        lg = lags.groupby("cluster").agg(lag_le2=("abs_lag", lambda x: float((x <= 2).mean())),
                                          lag_med=("abs_lag", "median"), lag_max=("abs_lag", "max"))
        g = g.join(lg)
    return g.reset_index()


def diagnostic_tables(out: Out, a) -> dict:
    # S13 – paired difference by duration and phase (data behind F5)
    pd_ = read(os.path.join(a.eval, "tables", "paired_diff_by_duration_phase.csv"))
    if pd_ is not None:
        pd_ = pd_[(pd_.design == PRIMARY_DESIGN) & pd_.metric.isin(["NRMSE_run", "NDTW_run"])].copy()
        dur = pd_.by == "duration_bin"
        pd_["lvl"] = [f"{float(l.strip('([').split(',')[0]):.0f}–{float(l.strip(')]').split(',')[1]):.0f} h"
                      if d else l for l, d in zip(pd_.level, dur)]
        pd_["lvl_sort"] = [float(l.strip("([").split(",")[0]) if d else PHASE_ORDER.index(l)
                           for l, d in zip(pd_.level, dur)]
        rows = []
        for (met, by, lvl), g in pd_.sort_values(["metric", "by", "lvl_sort"]).groupby(
                ["metric", "by", "lvl"], sort=False):
            g = g.set_index("method")
            row = {"Metric": METRIC_PLAIN[met], "Breakdown": "Gap duration" if by == "duration_bin" else "Process phase",
                   "Level": lvl, "Gap × cluster units": int(g.n.iloc[0])}
            for m in cfg.METHODS:
                if m in g.index:
                    row[name(m)] = f"{fmt(g.loc[m, 'median_diff'])} ({100 * g.loc[m, 'frac_method_better']:.0f}%)"
            rows.append(row)
        out.table(pd.DataFrame(rows), "S13_duration_phase")

    # S14 – FLk-NN decomposition (Sec. 3.8)
    fk = _flknn_by_cluster(a.diag)
    if fk is not None:
        out.table(pd.DataFrame({
            "Cluster": fk.cluster, "Gap × variable units": fk.units,
            "Share of lags with |lag| ≤ 2 samples": fk.lag_le2.map(lambda x: f"{100 * x:.0f}%") if "lag_le2" in fk else "–",
            "Max |lag| (samples)": fk.lag_max.map(lambda x: f"{x:.0f}") if "lag_max" in fk else "–",
            "k-NN coverage of gap points, median": fk.knn_cov.map(lambda x: f"{100 * x:.1f}%"),
            "Fourier − linear, RMS (z), median (max)": [f"{fmt(r.fourier_med)} ({fmt(r.fourier_max)})" for r in fk.itertuples()],
            "Stitching correction, RMS (z), median": fk.stitch.map(fmt),
            "FLk-NN − linear, RMS (z), median": fk.flknn_lin.map(fmt),
            "R² of straight line to FLk-NN − Fourier, median": fk.r2_line.map(lambda x: f"{x:.3f}" if np.isfinite(x) else "–"),
            "Roughness ratio, median": fk.rough.map(fmt)}),
            "S14_flknn_components")
        out.note("9 FLk-NN mechanism (diagnostics)", "Seeds of the FLk-NN decomposition", fk.seeds.iloc[0], "S14")

    # S15 – Hankel and the periodicity of dissolved CO2 (Sec. 3.3, 3.7)
    co = read(os.path.join(a.diag, "hankel_co2_coherence.csv"))
    if co is not None:
        per = co.acf_truth > PERIODIC_ACF

        def _row(label, d):
            both = d[d.acf_truth.gt(PERIODIC_ACF) & d.acf_hankel.gt(PERIODIC_ACF)]
            return {"Subset": label, "Gaps": len(d),
                    "Median r (Hankel − linear vs. truth − linear)": f"{d.r_hankel_vs_truth_deviation.median():.2f}",
                    "Share of gaps with r > 0": f"{100 * (d.r_hankel_vs_truth_deviation > 0).mean():.0f}%",
                    "Periodic truth (gaps)": int((d.acf_truth > PERIODIC_ACF).sum()),
                    "Hankel also periodic (gaps)": len(both),
                    "Period error, median": f"{100 * np.median(np.abs(both.period_hankel_h - both.period_truth_h) / both.period_truth_h):.1f}%" if len(both) else "–",
                    "Amplitude ratio Hankel / truth, median": f"{np.median(d.sd_detrended_hankel / d.sd_detrended_truth):.2f}"}

        rows = [_row("all gaps", co)] + [_row(f"phase: {p}", co[co.phase == p]) for p in PHASE_ORDER]
        rows.append(_row("gaps with periodic truth", co[per]))
        out.table(pd.DataFrame(rows), "S15a_hankel_co2_coherence")
    cr = read(os.path.join(a.diag, "hankel_co2_real_outages.csv"))
    ev = read(os.path.join(a.real, "outage_events.csv"))
    if cr is not None and ev is not None:
        # duration as in T1 (points × sampling interval)
        ev = ev.assign(duration_t1=ev.n_points * ev.dt_s / 3600)[["experiment", "reactor", "event_id", "duration_t1"]]
        cr = cr.merge(ev, on=["experiment", "reactor", "event_id"], how="left").assign(duration_h=lambda d: d.duration_t1)
        p = lambda v, acf: f"{v:.2f} ({acf:.2f})" if np.isfinite(v) else "–"
        out.table(pd.DataFrame({
            "Experiment": cr.experiment, "Reactor": cr.reactor, "Outage": cr.event_id + 1,
            "Duration (h)": cr.duration_h.round(1),
            "Context before / after (h)": [f"{l:.1f} / {r:.1f}" for l, r in zip(cr.context_left_h, cr.context_right_h)],
            "Period of CO2 before outage, h (ACF)": [p(v, x) for v, x in zip(cr.period_context_left_h, cr.acf_context_left)],
            "Period of CO2 after outage, h (ACF)": [p(v, x) for v, x in zip(cr.period_context_right_h, cr.acf_context_right)],
            "Period of Hankel − linear, h (ACF)": [p(v, x) for v, x in zip(cr.period_hankel_deviation_h, cr.acf_hankel_deviation)],
            "Hankel iterations (converged)": [f"{int(i)} ({'yes' if c else 'no'})" for i, c in zip(cr.hankel_iterations, cr.hankel_converged)]}),
            "S15b_hankel_co2_real_outages")

    # S16 – runtime
    rt = _runtimes(a.results)
    if rt is not None:
        g = rt.groupby(["method", "n_gaps"]).seconds.median().unstack()
        tot = rt.groupby("method").seconds.agg(["median", "min", "max"])
        out.table(pd.DataFrame([{"Method": name(m), **{f"{n} gap(s)": fmt(g.loc[m, n]) for n in g.columns},
                                 "All, median (min–max)": f"{fmt(tot.loc[m, 'median'])} ({fmt(tot.loc[m, 'min'])}–{fmt(tot.loc[m, 'max'])})"}
                                for m in cfg.METHODS if m in g.index]),
                  "S16_runtime")
    return {}


def hyperparameter_tables(out: Out) -> dict:
    """S1: search spaces and selected parameters from the Optuna logs (manuscript cluster numbering)."""
    rows_h, rows_f = [], []
    for (n, cid), path in sorted(cfg._collect(cfg.HANKEL_OPT_DIR).items()):
        b = cfg._best_trial(path)
        rows_h.append({"Gaps": n, "Cluster": cfg.PAPER_CLUSTER[cid], "Lag buffer b": int(b.lag_buffer),
                       "Rank": int(b["rank"]), "max_iter": int(b.max_iter), "tol": f"{b.tol:.1e}",
                       "Objective (NRMSE + NDTW)": fmt(b.NRMSE + b.NDTW),
                       "Mean iterations": f"{b.Iter_mean:.1f}", "Converged": f"{100 * b.Converged_frac:.0f}%",
                       "RMS change from linear (z)": fmt(b.RMS_change_lin_z)})
    for (n, cid), path in sorted(cfg._collect(cfg.FLKNN_OPT_DIR).items()):
        b = cfg._best_trial(path)
        rows_f.append({"Gaps": n, "Cluster": cfg.PAPER_CLUSTER[cid], "k": int(b.k), "num_lags": int(b.num_lags),
                       "max_lag": int(b.max_lag), "fd_percent": int(b.fd_percent),
                       "Objective (NRMSE + NDTW)": fmt(b.NRMSE + b.NDTW)})
    key = lambda d: (d["Gaps"], d["Cluster"])
    out.table(pd.DataFrame(sorted(rows_f, key=key)), "S1a_hyperparameters_flknn")
    out.table(pd.DataFrame(sorted(rows_h, key=key)), "S1b_hyperparameters_hankel")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════
def _context(n: int, st: int, en: int) -> tuple[int, int]:
    c = en - st
    return max(0, st - c), min(n, en + c)


def _hours(idx) -> np.ndarray:
    return (idx - idx[0]).total_seconds().values / 3600


def _median_gap(rec: pd.DataFrame, value: str) -> pd.Series:
    """Gap whose value is closest to the median (selection rule against cherry-picking)."""
    med = rec[value].median()
    return rec.loc[(rec[value] - med).abs().idxmin()]


def fig_trajectories(out: Out, a, plt):
    uni = read(os.path.join(a.eval, "records_univariate.csv.gz"))
    if uni is None:
        return
    rec = uni[(uni.design == PRIMARY_DESIGN) & (uni.method == REFERENCE)]
    # Selection rule: per variable the gap whose Linear NRMSE is closest to the median, only from runs
    # with the full variable set (EXP001/EXP002 R02 lack alpha and the spectrum, so FLk-NN and Hankel
    # work on CO2 alone in Cluster 3 there).
    ncol = rec.groupby("run").column.nunique()
    rec = rec[rec.run.isin(ncol[ncol == ncol.max()].index)]
    clean, positions = load_scenarios(a.sim)
    panels = []
    for col in TRAJ_VARS:
        g = _median_gap(rec[rec.column == col], "NRMSE_run")
        unit = {"design": g.design, "seed": int(g.seed), "n_gaps": int(g.n_gaps), "experiment": g.experiment,
                "reactor": g.reactor}
        X = clean[g.experiment][g.reactor]
        pos = positions[(positions.design == g.design) & (positions.seed == g.seed)
                        & (positions.n_gaps == g.n_gaps) & (positions.experiment == g.experiment)
                        & (positions.gap_id == g.gap_id)].iloc[0]
        st, en = int(pos.start_idx), int(pos.end_idx)
        panels.append({"col": col, "g": g, "unit": unit, "X": X, "t": _hours(X.index), "st": st, "en": en,
                       "ctx": _context(len(X), st, en)})
    methods = [m for m in cfg.METHODS
               if all(os.path.exists(result_path(a.results, {**p["unit"], "method": m})) for p in panels)]
    for p in panels:
        p["units"] = {m: load_unit(result_path(a.results, {**p["unit"], "method": m}), clean) for m in methods}

    fig, axes = plt.subplots(len(methods), len(TRAJ_VARS), figsize=(MAX_WIDTH_INCH, 1.35 * len(methods) + 0.6),
                             sharex="col", squeeze=False)
    for i, m in enumerate(methods):
        for j, p in enumerate(panels):
            ax, X, t, col, g = axes[i, j], p["X"], p["t"], p["col"], p["g"]
            (lo, hi), st, en, u = p["ctx"], p["st"], p["en"], p["units"][m]
            ax.axvspan(t[st], t[en - 1], color="gray", alpha=0.15, lw=0)
            ax.plot(t[lo:hi], X[col].values[lo:hi], color="black", lw=1.0, zorder=3)
            sel = (u["gap_id"] == g.gap_id).values
            ax.plot(t[X.index.get_indexer(u["y_pred"].index[sel])], u["y_pred"][col].values[sel],
                    color=METHOD_COLORS[m], lw=1.4, zorder=4)
            ax.set_xlim(t[lo], t[hi - 1])
            panel_label(ax, i * len(TRAJ_VARS) + j)
            if i == len(methods) - 1:
                ax.set_xlabel("Cultivation time (h)")
            if j == 0:
                ax.set_ylabel(VAR_LABELS[TRAJ_VARS[0]].split(" (")[0])
            if j == 1:
                ax.set_ylabel(VAR_LABELS[TRAJ_VARS[1]].split(" (")[0])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([0], [0], color="black", lw=1.0, label="True signal"),
               Patch(color="gray", alpha=0.15, label="Simulated gap")] + \
              [Line2D([0], [0], color=METHOD_COLORS[m], lw=1.4, label=name(m)) for m in methods]
    leg = fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False)
    for txt in leg.get_texts()[2:]:
        txt.set_fontstyle("italic")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (1.35 * len(methods) + 0.6)))
    for p in panels:
        g = p["g"]
        out.note("12 Example gaps in the figures", f"F3 example gap, {VAR_LABELS[p['col']].split(' (')[0].replace('$_2$', '2')}",
                 f"{g.experiment} {g.reactor}, seed {int(g.seed)}, {int(g.n_gaps)}-gap scenario, "
                 f"{g.duration_h:.1f} h from {g.start_h:.1f} h ({g.phase})", "F3")
    out.figure(fig, "F3_trajectories")


# Methods that do not set the axis scale (otherwise all other differences shrink to 0);
# values outside the axis are shown as an arrow at the border with the number.
OFFSCALE_METHODS = {"FLk-NN"}


def _limits(values, pad: float = 0.12) -> tuple[float, float]:
    """Axis limits around `values` and the zero line, with margin."""
    v = np.asarray([0.0, *values], dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = v.min(), v.max()
    span = hi - lo if hi > lo else max(abs(hi), 1e-6)
    return lo - pad * span, hi + pad * span


def _sci_axis(axis, lo: float, hi: float) -> str:
    """Scale the ticks; returns the factor for the axis label (instead of an offset text, which would
    collide with the label or the panel letters)."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    axis.set_major_locator(MaxNLocator(nbins=3))
    e = int(np.floor(np.log10(max(abs(lo), abs(hi)))))
    e = e if (e < -2 or e > 3) else 0
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 10 ** e:g}".replace("-", "−")))
    return f"$\\times 10^{{{e}}}$" if e else ""


def _with_scale(label: str, scale: str, sep: str = " ") -> str:
    return f"{label}{sep}({scale})" if scale else label


def as_percent(d: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Difference and CI in % of the reference median of the same stratum × metric (as the percentages
    in the text and in Table S3a)."""
    ref = summary.set_index(["stratum", "metric"])[f"median_{REFERENCE}"]
    scale = 100 / ref.reindex(pd.MultiIndex.from_frame(d[["stratum", "metric"]])).to_numpy()
    return d.assign(diff=d["diff"] * scale, ci_low=d.ci_low * scale, ci_high=d.ci_high * scale)


def _forest(ax, d: pd.DataFrame, methods: list[str], show_names: bool, show_runs: bool = True,
            pct: bool = False):
    y = np.arange(len(methods))[::-1]
    rows = {m: d[d.method == m].iloc[0] for m in methods if (d.method == m).any()}
    core = [r for m, r in rows.items() if m not in OFFSCALE_METHODS]
    xlo, xhi = _limits([v for r in (core or rows.values()) for v in (r.ci_low, r.ci_high)])
    for yi, m in zip(y, methods):
        if m not in rows:
            continue
        r = rows[m]
        c = METHOD_COLORS[m]
        mfc = c if r.clear else "white"
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        if xlo <= r["diff"] <= xhi:
            ax.plot(r["diff"], yi, "o", ms=5.5, mec=c, mfc=mfc, mew=1.4, zorder=3)
        else:                                        # outside the scale: arrow at the border + value
            right = r["diff"] > xhi
            edge = xhi - 0.07 * (xhi - xlo) if right else xlo + 0.07 * (xhi - xlo)
            ax.plot(edge, yi, ">" if right else "<", ms=6.5, mec=c, mfc=mfc, mew=1.4, zorder=3, clip_on=False)
            val = f"{r['diff']:+.0f}%" if pct else fmt(r["diff"], 2)
            ax.annotate(val.replace("-", "−"), xy=(edge, yi), xytext=(-8 if right else 8, 0),
                        textcoords="offset points", ha="right" if right else "left", va="center",
                        fontsize=7, color=c, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        if show_runs:
            ax.annotate(f"{int(r.runs)}/{int(r.n_runs)}", xy=(1.0, yi), xycoords=("axes fraction", "data"),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color="#555555")
    ax.axvline(0, color="black", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(xlo, xhi)
    ax.set_yticks(y)
    ax.set_yticklabels([name(m) for m in methods] if show_names else [], fontstyle="italic")
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.tick_params(axis="x", labelsize=8)
    return _sci_axis(ax.xaxis, xlo, xhi)


def _forest_legend(fig, methods: list[str], y: float = 1.0, loc: str = "upper center"):
    """Shared legend: method colors and meaning of filled/open markers."""
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=METHOD_COLORS[m], lw=1.6, marker="o", ms=5, mfc=METHOD_COLORS[m], label=name(m))
         for m in methods]
    h += [Line2D([0], [0], color="none", marker="o", ms=5, mfc="#555555", mec="#555555", label="clear (Holm p < 0.05)"),
          Line2D([0], [0], color="none", marker="o", ms=5, mfc="white", mec="#555555", mew=1.4, label="not clear")]
    leg = fig.legend(handles=h, loc=loc, ncol=4, bbox_to_anchor=(0.5, y), frameon=False, fontsize=8.5)
    for txt in leg.get_texts()[:len(methods)]:
        txt.set_fontstyle("italic")


def fig_effects(out: Out, a, plt):
    pairs = read(os.path.join(a.eval, "tables", "comparisons_pairwise.csv"))
    if pairs is None:
        return
    summ = read(os.path.join(a.eval, "tables", "comparisons_summary.csv"))
    if summ is None:
        return
    d = as_percent(vs_reference(pairs[pairs.role == "confirmatory"]), summ[summ.role == "confirmatory"])
    metrics = [m for m in em.PRIMARY_METRICS.values() if m in set(d.metric)]
    clusters = sorted(d.stratum.unique(), key=lambda s: int(s.split()[-1]))
    methods = [m for m in cfg.METHODS if m != REFERENCE and m in set(d.method)]
    height = 1.5 * len(clusters) + 0.9
    fig, axes = plt.subplots(len(clusters), len(metrics), figsize=(MAX_WIDTH_INCH, height), squeeze=False)
    for i, cl in enumerate(clusters):
        for j, met in enumerate(metrics):
            ax = axes[i, j]
            scale = _forest(ax, d[(d.stratum == cl) & (d.metric == met)], methods, show_names=False, pct=True)
            panel_label(ax, i * len(metrics) + j)
            if j == 0:
                ax.set_ylabel(cl)
            if i == len(clusters) - 1:
                ax.set_xlabel(_with_scale(f"Δ {METRIC_SHORT.get(met, METRIC_LABELS[met])} (%)", scale))
            elif scale:
                ax.set_xlabel(scale, fontsize=8, labelpad=1)
    fig.tight_layout(w_pad=2.4, h_pad=1.2)
    # legend directly above the top panel row (space for the panel letters)
    _forest_legend(fig, methods, y=axes[0, 0].get_position().y1 + 0.32 / height, loc="lower center")
    out.figure(fig, "F4_main_effects")


def fig_duration_phase(out: Out, a, plt):
    pdp = read(os.path.join(a.eval, "tables", "paired_diff_by_duration_phase.csv"))
    if pdp is None:
        return
    pdp = pdp[pdp.design == PRIMARY_DESIGN]
    metrics = ["NRMSE_run", "NDTW_run"]
    methods = [m for m in cfg.METHODS if m != REFERENCE and m in set(pdp.method)]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(MAX_WIDTH_INCH, 4.6), squeeze=False)
    for i, met in enumerate(metrics):
        for j, by in enumerate(["duration_bin", "phase"]):
            ax = axes[i, j]
            d = pdp[(pdp.metric == met) & (pdp.by == by)]
            levels = (sorted(d.level.unique(), key=lambda s: float(s.strip("([").split(",")[0]))
                      if by == "duration_bin" else [p for p in PHASE_ORDER if p in set(d.level)])
            x = np.arange(len(levels))
            ylo, yhi = _limits(d[~d.method.isin(OFFSCALE_METHODS)].median_diff, pad=0.15)
            for k, m in enumerate(methods):
                dm = d[d.method == m].set_index("level").reindex(levels)
                off = (k - (len(methods) - 1) / 2) * 0.07
                yv = dm.median_diff.values
                if m in OFFSCALE_METHODS and np.nanmax(np.abs(yv)) > max(abs(ylo), abs(yhi)):
                    # outside the scale: arrows at the border with value
                    edge = np.where(yv > 0, yhi, ylo)
                    ax.plot(x + off, edge, ":", color=METHOD_COLORS[m], lw=1.0, label=name(m),
                            marker="^", ms=5, clip_on=False)
                    for xi, v, e in zip(x + off, yv, edge):
                        ax.annotate(fmt(v, 2).replace("-", "−"), xy=(xi, e), xytext=(4, -3 if v > 0 else 3),
                                    textcoords="offset points", ha="left", va="top" if v > 0 else "bottom",
                                    fontsize=7, color=METHOD_COLORS[m])
                    continue
                ax.plot(x + off, yv, "o-", color=METHOD_COLORS[m], lw=1.2, ms=4, label=name(m))
            ax.set_ylim(ylo, yhi)
            scale = _sci_axis(ax.yaxis, ylo, yhi)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{float(l.strip('([').split(',')[0]):.0f}–{float(l.strip(')]').split(',')[1]):.0f} h"
                                if by == "duration_bin" else l for l in levels])
            ax.set_ylabel(_with_scale(f"Δ {METRIC_LABELS[met]}", scale))
            if i == len(metrics) - 1:
                ax.set_xlabel("Gap duration" if by == "duration_bin" else "Process phase at gap centre")
            panel_label(ax, i * 2 + j)
    h, l = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.03), frameon=False)
    for txt in leg.get_texts():
        txt.set_fontstyle("italic")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.figure(fig, "F5_duration_phase")


def fig_real_outages(out: Out, a, plt):
    import real_outages as ro
    from matplotlib.lines import Line2D
    events = read(os.path.join(a.real, "outage_events.csv"))
    if events is None:
        return
    sel, data = [], {}
    for exp in ro.EXPERIMENTS:
        ev = events[(events.experiment == exp) & (events.reactor == "R01")]
        p = os.path.join(a.real, "imputed", f"{exp}_R01.pkl")
        if len(ev) and os.path.exists(p):
            with open(p, "rb") as f:
                data[exp] = pickle.load(f)
            sel.append((exp, ev.loc[ev.duration_h.idxmax()]))
    fig, axes = plt.subplots(len(ro.PLOT_VARS), len(sel), figsize=(MAX_WIDTH_INCH, 1.75 * len(ro.PLOT_VARS) + 0.5),
                             squeeze=False)
    for j, (exp, e) in enumerate(sel):
        for i, col in enumerate(ro.PLOT_VARS):
            ax = axes[i, j]
            ro._plot_panel(ax, data[exp]["X_missing"], data[exp]["results"], e, col)
            for ln in ax.get_lines():
                ln.set_linewidth(1.0 if ln.get_color() == "black" else 1.3)
            panel_label(ax, i * len(sel) + j)
            if j == 0:
                ax.set_ylabel(VAR_LABELS[col].split(" (")[0])
            if i == len(ro.PLOT_VARS) - 1:
                ax.set_xlabel("Cultivation time (h)")
    present = [m for m in cfg.METHODS if any(m in d["results"] for d in data.values())]
    handles = [Line2D([0], [0], color="black", lw=1.0, label="Observed signal")] + \
              [Line2D([0], [0], color=METHOD_COLORS[m], lw=1.3, label=name(m)) for m in present]
    leg = fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), frameon=False)
    for txt in leg.get_texts()[1:]:
        txt.set_fontstyle("italic")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.figure(fig, "F6_real_outages")


def fig_downstream(out: Out, a, plt):
    dp = read(os.path.join(a.down, "tables", "downstream_comparisons_pairwise.csv"))
    rec = read(os.path.join(a.down, "records_downstream.csv.gz"))
    if dp is None or rec is None:
        return
    ds = read(os.path.join(a.down, "tables", "downstream_comparisons_summary.csv"))
    if ds is None:
        return
    d = as_percent(vs_reference(dp[dp.role == "confirmatory"]), ds[ds.role == "confirmatory"])
    metrics = [m for m in ["VCD_RMSE", "mu_RMSE", "T2_logRMSE", "Q_logRMSE"] if m in set(d.metric)]
    methods = [m for m in cfg.METHODS if m != REFERENCE and m in set(d.method)]
    fig = plt.figure(figsize=(MAX_WIDTH_INCH, 5.6))
    gs = fig.add_gridspec(2, len(metrics), height_ratios=[1, 1.05], hspace=0.55, wspace=0.45, top=0.86)
    for j, met in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        scale = _forest(ax, d[d.metric == met], methods, show_names=False, pct=True)
        ax.set_xlabel(_with_scale(f"Δ {METRIC_SHORT.get(met, METRIC_LABELS[met])} (%)", scale, sep="\n"))
        panel_label(ax, j)
    _forest_legend(fig, methods, y=1.0)

    # example: PLS VCD in the gap with the median VCD error of Linear
    r = rec[(rec.design == PRIMARY_DESIGN) & (rec.method == REFERENCE)]
    g = _median_gap(r, "VCD_RMSE")
    import __main__
    import downstream_evaluation
    __main__.PCAMonitor = downstream_evaluation.PCAMonitor   # older models.pkl (pickled under __main__)
    with open(os.path.join(a.down, "models.pkl"), "rb") as f:
        mb = pickle.load(f)
    clean, positions = load_scenarios(a.sim)
    X = clean[g.experiment][g.reactor][mb["all_cols"]]
    t = _hours(X.index)
    pos = positions[(positions.design == g.design) & (positions.seed == g.seed) & (positions.n_gaps == g.n_gaps)
                     & (positions.experiment == g.experiment) & (positions.gap_id == g.gap_id)].iloc[0]
    st, en = int(pos.start_idx), int(pos.end_idx)
    lo, hi = _context(len(X), st, en)
    pls = mb["models"][g.experiment]["pls"]
    ax = fig.add_subplot(gs[1, :])
    ax.axvspan(t[st], t[en - 1], color="gray", alpha=0.15, lw=0)
    ax.plot(t[lo:hi], pls.predict(X[mb["inputs"]].values[lo:hi]).ravel(), color="black", lw=1.0, zorder=3,
            label="Soft sensor on true signals")
    unit = {"design": g.design, "seed": int(g.seed), "n_gaps": int(g.n_gaps), "experiment": g.experiment, "reactor": g.reactor}
    for m in cfg.METHODS:
        pth = result_path(a.results, {**unit, "method": m})
        if not os.path.exists(pth):
            continue
        u = load_unit(pth, clean)
        sel = (u["gap_id"] == g.gap_id).values
        Xi = u["y_pred"][mb["inputs"]].values[sel]
        ax.plot(t[X.index.get_indexer(u["y_pred"].index[sel])], pls.predict(Xi).ravel(), color=METHOD_COLORS[m],
                lw=1.3, zorder=4, label=name(m))
    ax.set_xlim(t[lo], t[hi - 1])
    ax.set_xlabel("Cultivation time (h)")
    ax.set_ylabel("Predicted VCD (10$^6$ cells/mL)")
    panel_label(ax, len(metrics))
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, frameon=False)
    for txt in leg.get_texts()[1:]:
        txt.set_fontstyle("italic")
    out.note("12 Example gaps in the figures", "F7 example gap (median VCD error of the Linear Baseline)",
             f"{g.experiment} {g.reactor}, seed {int(g.seed)}, {int(g.n_gaps)}-gap scenario, "
             f"{g.duration_h:.1f} h; {pls.n_components} PLS components", "F7")
    out.figure(fig, "F7_downstream")


def fig_per_run(out: Out, a, plt):
    pr = read(os.path.join(a.eval, "tables", "per_run_medians.csv"))
    if pr is None:
        return
    pr = pr[pr.design == PRIMARY_DESIGN]
    metrics = [m for m in em.PRIMARY_METRICS.values() if m in set(pr.metric)]
    methods = [m for m in cfg.METHODS if m in pr.columns]
    fig, axes = plt.subplots(2, 2, figsize=(MAX_WIDTH_INCH, 5.2), squeeze=False)
    for j, met in enumerate(metrics):
        ax = axes.flat[j]
        d = pr[pr.metric == met].sort_values("run")
        x = np.arange(len(d))
        for k, m in enumerate(methods):
            ax.plot(x + (k - (len(methods) - 1) / 2) * 0.11, d[m].values, "o", ms=4, color=METHOD_COLORS[m], label=name(m))
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in d.run], fontsize=8)
        ax.set_ylabel(METRIC_LABELS[met])
        panel_label(ax, j)
    h, l = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False)
    for txt in leg.get_texts():
        txt.set_fontstyle("italic")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.figure(fig, "SF1_per_run")


# ══════════════════════════════════════════════════════════════════════════════
# Key numbers for the text
# ══════════════════════════════════════════════════════════════════════════════
def _ref_rows(summary: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """All comparisons method − reference for the strata in `summary` (long format)."""
    out = []
    for _, s in summary.iterrows():
        p = pairs[(pairs.design == s.design) & (pairs.level == s.level) & (pairs.stratum == s.stratum)
                  & (pairs.metric == s.metric)]
        out.append(with_relative(vs_reference(p), s[f"median_{REFERENCE}"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _ref_text(r) -> str:
    return (f"Δ = {fmt_ci(r['diff'], r.ci_low, r.ci_high)} ({fmt_rel(r.rel)} of Linear median, "
            f"[{fmt_rel(r.rel_low)}, {fmt_rel(r.rel_high)}]); same sign in {int(r.runs)}/{int(r.n_runs)} runs, "
            f"{int(r.seeds)}/{int(r.n_seeds)} seeds; {'clear' if r.clear else 'not clear'} (Holm p = {fmt_p(r.p_holm)})")


def _clear_counts(d: pd.DataFrame, metrics=None) -> dict:
    """Per method: number of clearly better / clearly worse comparisons vs. the reference."""
    if metrics is not None:
        d = d[d.metric.isin(metrics)]
    res = {}
    for m in cfg.METHODS:
        x = d[d.method == m]
        if len(x):
            res[m] = (int((x.clear & (x["diff"] < 0)).sum()), int((x.clear & (x["diff"] > 0)).sum()), len(x))
    return res


def _alarm_means(down_dir: str) -> pd.DataFrame | None:
    """Mean alarm disagreement per method and share of gaps with a true alarm (main design)."""
    path = os.path.join(down_dir, "records_downstream.csv.gz")
    if not os.path.exists(path):
        return None
    r = pd.read_csv(path)
    r = r[r.design == PRIMARY_DESIGN]
    return r.groupby("method").agg(
        T2_alarm_disagree=("T2_alarm_disagree", "mean"), Q_alarm_disagree=("Q_alarm_disagree", "mean"),
        T2_gap_with_alarm=("T2_alarm_rate_true", lambda x: float((x > 0).mean())),
        Q_gap_with_alarm=("Q_alarm_rate_true", lambda x: float((x > 0).mean())))


def key_numbers(out: Out, a):
    """Key numbers for the text → key_numbers.md (readable) and key_numbers.csv (lookup).
    Summarizes the finished tables; no statistics are recomputed."""
    rows = []

    def add(section, key, value, source):
        rows.append({"section": section, "key": key, "value": value, "source": source})

    nonmad = ["NRMSE_run", "NDTW_run", "FSD"]
    summ = read(os.path.join(a.eval, "tables", "comparisons_summary.csv"))
    pairs = read(os.path.join(a.eval, "tables", "comparisons_pairwise.csv"))

    # ── Data and design ──────────────────────────────────────────────────────
    sec = "1 Data and design"
    if summ is not None:
        conf = summ[summ.role == "confirmatory"]
        add(sec, "Held-out runs (confirmatory)", int(conf.n_runs.max()), "comparisons_summary")
        add(sec, "Gap events per cluster (random placement)", int(conf.n_gap_events.max()), "comparisons_summary")
    pos = read(os.path.join(a.sim, "gap_positions.csv"))
    if pos is not None:
        for des, g in pos.groupby("design"):
            add(sec, f"Simulated gaps ({des}): number over the 3 held-out experiments (R01/R02 share positions)",
                len(g), "S2")
            add(sec, f"Simulated gaps ({des}): duration median (min–max), h",
                f"{g.duration_h.median():.1f} ({g.duration_h.min():.1f}–{g.duration_h.max():.1f})", "S2")
    uni_path = os.path.join(a.eval, "records_univariate.csv.gz")
    if os.path.exists(uni_path):
        u = pd.read_csv(uni_path, usecols=["design", "method", "run", "column"])
        ncol = u[(u.design == PRIMARY_DESIGN) & (u.method == REFERENCE)].groupby("run").column.nunique()
        red = ncol[ncol < ncol.max()]
        add(sec, "Runs with reduced variable set (no capacitance spectrum)",
            f"{', '.join(r.replace('_', ' ') for r in red.index)} ({int(red.max())} of {int(ncol.max())} variables)"
            if len(red) else "none", "records_univariate")
    ev = read(os.path.join(a.real, "outage_events.csv"))
    if ev is not None:
        dur = ev.n_points * ev.dt_s / 3600
        add(sec, "Real outages: events (reactors × outages)", len(ev), "T1")
        add(sec, "Real outages: duration range, h", f"{dur.min():.1f}–{dur.max():.1f}", "T1")
        add(sec, "Real outages: partial (module still recording)",
            ", ".join(sorted(set(ev[ev.n_cols_missing < ev.n_cols_missing.max()].experiment))) or "none", "T1")

    # ── Main result ─────────────────────────────────────────────────────────
    if summ is not None and pairs is not None:
        conf = summ[summ.role == "confirmatory"]
        d = _ref_rows(conf, pairs)
        sec = "2 Main results vs. Linear Baseline (confirmatory, random placement)"
        for m, (b, w, n) in _clear_counts(d, nonmad).items():
            add(sec, f"{name(m)}: clearly better / clearly worse than Linear (NRMSE, NDTW, FSD; 12 comparisons)",
                f"{b} / {w}", "T2")
        for m, (b, w, n) in _clear_counts(d, ["MAD_OffDiag"]).items():
            add(sec, f"{name(m)}: clearly better / clearly worse than Linear (MAD, descriptive; 4 comparisons)",
                f"{b} / {w}", "T2")
        # practical relevance: all clear differences (without MAD), sorted by relative size
        sec = "2b Clear differences vs. Linear by relative size (NRMSE, NDTW, FSD; % of Linear median)"
        cl = d[d.clear & d.metric.isin(nonmad)].sort_values("rel")
        for r in cl.itertuples():
            add(sec, f"{r.stratum}, {METRIC_PLAIN[r.metric]}, {name(r.method)}",
                f"{fmt_rel(r.rel)} [{fmt_rel(r.rel_low)}, {fmt_rel(r.rel_high)}]", "T2")
        sec = "3 Main results per cluster and metric (confirmatory)"
        for _, s in conf.iterrows():
            lab = f"{s.stratum}, {METRIC_PLAIN[s.metric]}"
            add(sec, f"{lab}: {name(REFERENCE)} median (IQR)",
                f"{fmt(s[f'median_{REFERENCE}'])} ({s[f'iqr_{REFERENCE}']})", "T2/S3b")
            dd = d[(d.stratum == s.stratum) & (d.metric == s.metric)].set_index("method")
            for m in cfg.METHODS:
                if m != REFERENCE and m in dd.index:
                    add(sec, f"{lab}: {name(m)} (median {fmt(s[f'median_{m}'])})", _ref_text(dd.loc[m]), "T2/F4")

        sec = "4 Robustness"
        fx = summ[(summ.design == "fixed") & (summ.level == "cluster") & summ.primary]
        dfx = _ref_rows(fx, pairs)
        for m, (b, w, n) in _clear_counts(dfx, nonmad).items():
            add(sec, f"Fixed placement – {name(m)}: clearly better / worse than Linear (12 comparisons)", f"{b} / {w}", "S5a")
        h3 = dfx[(dfx.stratum == "Cluster 3") & (dfx.method == "Hankel") & dfx.metric.isin(nonmad)]
        for r in h3.itertuples():
            add(sec, f"Fixed placement – Hankel, Cluster 3, {METRIC_PLAIN[r.metric]}", _ref_text(pd.Series(r._asdict())), "S5a")
    sm = read(os.path.join(a.eval, "tables", "smooth_reference_sensitivity.csv"))
    if sm is not None:
        sm = sm[(sm.design == PRIMARY_DESIGN) & (sm.level == "cluster")]
        add("4 Robustness", "Smoothed reference: same best method (cluster × metric)",
            f"{int(sm.same_winner.sum())}/{len(sm)}", "S5b")
        add("4 Robustness", "Smoothed reference: Spearman ρ of method medians, range",
            f"{sm.spearman_method_medians.min():.2f}–{sm.spearman_method_medians.max():.2f}", "S5b")
        for met, g in sm.groupby("metric_raw"):
            add("4 Robustness", f"Smoothed reference: clear decisions reversed, {METRIC_PLAIN[met]}",
                f"{int(g.clear_pairs_flipped.sum())} of {int(g.clear_pairs_raw.sum())}", "S5b")
    sr = read(os.path.join(a.eval, "tables", "seed_rank_stability.csv"))
    if sr is not None:
        sr = sr[sr.design == PRIMARY_DESIGN]
        for met, g in sr.groupby("metric"):
            top = g.sort_values(["n_seeds_best", "mean_rank"], ascending=[False, True]).iloc[0]
            last = g.sort_values("mean_rank").iloc[-1]
            add("4 Robustness", f"Seeds: most often first, {METRIC_PLAIN[met]}",
                f"{name(top.method)} ({int(top.n_seeds_best)}/{int(top.n_seeds)} seeds, mean rank {top.mean_rank:.1f})", "S4b")
            add("4 Robustness", f"Seeds: last, {METRIC_PLAIN[met]}",
                f"{name(last.method)} (mean rank {last.mean_rank:.1f}, SD {last.sd_rank:.2f})", "S4b")
    rho = read(os.path.join(a.eval, "tables", "paired_diff_duration_spearman.csv"))
    if rho is not None:
        rho = rho[(rho.design == PRIMARY_DESIGN) & rho.metric.isin(["NRMSE_run", "NDTW_run"])]
        for r in rho.itertuples():
            add("4 Robustness", f"Duration vs. paired difference to Linear, {METRIC_PLAIN[r.metric]}, {name(r.method)}",
                f"ρ = {r.spearman_rho_duration:.2f}; same sign in {int(r.runs_same_sign)}/{int(r.n_runs)} runs", "F5")

    # ── Error scale ───────────────────────────────────────────────────────────
    sec = "5 Error scale"
    ns = read(os.path.join(a.eval, "tables", "nrmse_scale.csv"))
    if ns is not None:
        for r in ns[ns.design == PRIMARY_DESIGN].itertuples():
            add(sec, f"{name(r.method)}: gap-normalized NRMSE median; share ≥ 1",
                f"{fmt(r.median_NRMSE_local)}; {100 * r.frac_NRMSE_local_ge_1:.1f}%", "S8a")
    nf = read(os.path.join(a.eval, "tables", "noise_floor.csv"))
    if nf is not None:
        nf = nf[(nf.design == PRIMARY_DESIGN) & ~nf.method.isin(OFFSCALE_METHODS)]
        for cl, g in nf.groupby("cluster"):
            add(sec, f"Cluster {cl}: NRMSE / noise floor, range over methods (without FLk-NN)",
                f"{g.median_ratio_to_noise.min():.1f}–{g.median_ratio_to_noise.max():.1f}", "S8b")
        add(sec, "NRMSE reduction against smoothed reference, range (without FLk-NN)",
            f"{100 * (1 - nf.median_NRMSE_run_smooth / nf.median_NRMSE_run).min():.0f}–"
            f"{100 * (1 - nf.median_NRMSE_run_smooth / nf.median_NRMSE_run).max():.0f}%", "S8b")
    md = read(os.path.join(a.eval, "tables", "metric_discrimination.csv"))
    if md is not None:
        for r in md[md.primary].itertuples():
            add(sec, f"Kendall's W (method discrimination), {r.metric}", f"{r.kendall_w:.3f}", "S7a")

    # ── Correlations and Hankel ──────────────────────────────────────────────
    sec = "6 Correlation structure and Hankel diagnostics"
    cs = read(os.path.join(a.eval, "tables", "correlation_structure.csv"))
    if cs is not None:
        cs = cs[cs.design == PRIMARY_DESIGN]
        cs = cs[cs.n_cols == cs.groupby("cluster").n_cols.transform("max")].set_index(["cluster", "method"])
        for cl in sorted(cs.index.get_level_values(0).unique()):
            t = cs.loc[(cl, REFERENCE)]
            h = cs.loc[(cl, "Hankel")] if (cl, "Hankel") in cs.index else None
            add(sec, f"Cluster {cl}: mean |r| true; Linear share |r| = 1; Hankel mean |r|",
                f"{t.median_mean_abs_r_true:.2f}; {100 * t.frac_pred_abs_r_eq_1:.0f}%; "
                f"{h.median_mean_abs_r_pred:.2f}" if h is not None else "–", "S10")
    hd = read(os.path.join(a.eval, "tables", "hankel_diagnostics_summary.csv"))
    if hd is not None:
        add(sec, "Hankel: median iterations per stratum, range (max)",
            f"{hd.iter_median.min():.0f}–{hd.iter_median.max():.0f} ({hd.iter_max.max():.0f})", "S9a")
        add(sec, "Hankel: share of gaps converged before max_iter, overall (strata range)",
            f"{100 * (hd.converged_frac * hd.n).sum() / hd.n.sum():.0f}% "
            f"({100 * hd.converged_frac.min():.0f}–{100 * hd.converged_frac.max():.0f}%)", "S9a")
        for cl, g in hd.groupby("cluster"):
            add(sec, f"Hankel, cluster {cl}: RMS change from linear initialization (z), median per stratum, range",
                f"{fmt(g.rms_change_from_linear_z_median.min())}–{fmt(g.rms_change_from_linear_z_median.max())}", "S9a")
        add(sec, "Hankel matrix dimensions: rows; columns",
            f"{hd.hankel_rows_min.min()}–{hd.hankel_rows_max.max()}; {hd.hankel_cols_min.min()}–{hd.hankel_cols_max.max()}",
            "S9a")
        add(sec, "Note", "S9a pools random and fixed placement", "S9a")
    dv = read(os.path.join(a.down, "tables", "deviation_from_linear.csv"))
    if dv is not None:
        for r in dv[(dv.design == PRIMARY_DESIGN) & (dv.method == "Hankel")].itertuples():
            add(sec, f"Hankel, cluster {r.cluster}: distance from linear relative to truth, median (IQR)",
                f"{fmt(r.median_ratio)} ({fmt(r.q25_ratio)}–{fmt(r.q75_ratio)})", "S9b")
            add(sec, f"Cluster {r.cluster}: RMS(truth − linear) / RMS(Hankel − linear), run-normalized (z)",
                f"{fmt(r.median_true_minus_linear_z)} / {fmt(r.median_pred_minus_linear_z)}", "S9b")
    det = read(os.path.join(a.eval, "tables", "paired_diff_by_cluster_detail.csv"))
    if det is not None:
        det = det[(det.design == PRIMARY_DESIGN) & (det.method == "Hankel") & (det.cluster == 3)
                  & (det.metric == "NRMSE_run")]
        for r in det.itertuples():
            add("6b Hankel vs. Linear in cluster 3, NRMSE (S12)", f"{r.by}: {r.level}",
                f"Δ = {fmt(r.median_diff)} ({fmt_rel(r.rel_pct)}); Hankel better in {100 * r.frac_method_better:.0f}% "
                f"of {int(r.n)} units", "S12")

    # ── Downstream ────────────────────────────────────────────────────────────
    sec = "7 Downstream"
    dsum = read(os.path.join(a.down, "tables", "downstream_comparisons_summary.csv"))
    dpairs = read(os.path.join(a.down, "tables", "downstream_comparisons_pairwise.csv"))
    if dsum is not None and dpairs is not None:
        dconf = dsum[dsum.role == "confirmatory"]
        add(sec, "Runs with valid capacitance spectra", int(dconf.n_runs.max()), "T3")
        dd = _ref_rows(dconf, dpairs)
        for m, (b, w, n) in _clear_counts(dd).items():
            add(sec, f"{name(m)}: clearly better / worse than Linear ({n} metrics)", f"{b} / {w}", "T3")
        for _, s in dconf.iterrows():
            lab = METRIC_PLAIN[s.metric]
            add(sec, f"{lab}: {name(REFERENCE)} median (IQR)", f"{fmt(s[f'median_{REFERENCE}'])} ({s[f'iqr_{REFERENCE}']})", "T3")
            for r in dd[dd.metric == s.metric].itertuples():
                add(sec, f"{lab}: {name(r.method)}", _ref_text(pd.Series(r._asdict())), "T3/F7")
    dmed = read(os.path.join(a.down, "tables", "downstream_medians.csv"))
    if dmed is not None:
        dmed = dmed[dmed.design == PRIMARY_DESIGN]
    alarm = _alarm_means(a.down)
    if alarm is not None:
        for m, r in alarm.iterrows():
            add(sec, f"{name(m)}: alarm disagreement T2 / Q, mean share of time points per gap",
                f"{100 * r.T2_alarm_disagree:.2f}% / {100 * r.Q_alarm_disagree:.2f}% "
                f"(gaps with a true alarm: T2 {100 * r.T2_gap_with_alarm:.0f}%, Q {100 * r.Q_gap_with_alarm:.0f}%)",
                "S11a")
    mc = read(os.path.join(a.down, "model_calibration.csv"))
    if mc is not None:
        r = mc.iloc[0]
        add(sec, "PLS: components; RMSECV (LOEO), 10^6 cells/mL; calibration samples",
            f"{int(r.pls_components)}; {r.pls_rmsecv_loeo:.2f}; {int(r.pls_n_samples)}", "S11b")
        add(sec, "PLS: R² per test run", "; ".join(mc.pls_r2_test_runs), "S11b")
        if dmed is not None:
            lin = dmed.set_index("method").loc[REFERENCE, "VCD_RMSE"]
            add(sec, "Linear: imputation-induced VCD RMSE relative to PLS RMSECV",
                f"{fmt(lin)} vs. {r.pls_rmsecv_loeo:.2f} (≈ 1/{r.pls_rmsecv_loeo / lin:.0f})", "S11a/S11b")

    # ── Real outages ────────────────────────────────────────────────────────
    sec = "8 Real outages (plausibility, no ground truth)"
    pl = read(os.path.join(a.real, "plausibility_summary.csv"))
    if pl is not None:
        for m, g in pl.groupby("method"):
            add(sec, f"{name(m)}: boundary jump (noise units), range over clusters",
                f"{fmt(g.boundary_jump.min())}–{fmt(g.boundary_jump.max())}", "S11c")
            add(sec, f"{name(m)}: roughness ratio, range over clusters",
                f"{fmt(g.roughness_ratio.min())}–{fmt(g.roughness_ratio.max())}", "S11c")
    _diagnostic_key_numbers(add, a)

    df = pd.DataFrame(rows + out.notes)
    df.to_csv(os.path.join(out.root, "key_numbers.csv"), index=False, encoding="utf-8")
    with open(os.path.join(out.root, "key_numbers.md"), "w", encoding="utf-8") as f:
        f.write("# Key numbers\n\nGenerated by `paper_outputs.py` from the finished tables – do not edit "
                "by hand. Δ = method − Linear Baseline (negative = method better), 95 % CI from the "
                "hierarchical bootstrap.\n")
        for sec, g in df.groupby("section", sort=False):
            f.write(f"\n## {sec}\n\n| Key | Value | Source |\n|---|---|---|\n")
            for r in g.itertuples():
                cell = lambda x: str(x).replace("|", "\\|")
                f.write(f"| {cell(r.key)} | {cell(r.value)} | {r.source} |\n")
    out.written += ["key_numbers.md", "key_numbers.csv"]


def _diagnostic_key_numbers(add, a):
    """Key numbers for S8a (per cluster), S11b/d, S13–S16."""
    bc = _error_scale_by_cluster(a.eval)
    if bc is not None:
        for m in ("Linear", "Hankel"):
            g = bc[bc.method == m].set_index("cluster")
            add("5 Error scale", f"{name(m)}: share of gap × variable units with gap-normalized NRMSE ≥ 1, per cluster",
                "; ".join(f"C{c} {100 * r.frac_NRMSE_local_ge_1:.0f}% (median {fmt(r.median_NRMSE_local)})"
                          for c, r in g.iterrows()), "S8a")
        n = bc[bc.method == REFERENCE].set_index("cluster").n
        add("5 Error scale", "Share of gap × variable units from cluster 1", f"{100 * n['1'] / n.sum():.0f}%", "S8a")

    rmsec = _pls_rmse_calibration(a.down)
    dmed = read(os.path.join(a.down, "tables", "downstream_medians.csv"))
    if rmsec is not None and dmed is not None:
        lin = dmed[dmed.design == PRIMARY_DESIGN].set_index("method").loc[REFERENCE, "VCD_RMSE"]
        add("7 Downstream", "PLS: RMSE on calibration samples, 10^6 cells/mL", f"{rmsec:.2f}", "S11b")
        add("7 Downstream", "Linear: imputation-induced VCD RMSE relative to PLS calibration RMSE",
            f"{fmt(lin)} vs. {rmsec:.2f} (≈ 1/{rmsec / lin:.0f})", "S11a/S11b")

    sec = "8 Real outages (plausibility, no ground truth)"
    pv = read(os.path.join(a.real, "plausibility.csv"))
    if pv is not None:
        for col in REAL_KEY_VARS:
            g = pv[pv.column == col].groupby("method")
            lab = VAR_LABELS.get(col, col).replace("$_2$", "2")
            add(sec, f"{lab}: boundary jump (noise units), median over outages, Linear / Smoothed Linear / Hankel / FLk-NN",
                " / ".join(fmt(g.boundary_jump.median().get(m, np.nan)) for m in ("Linear", "Linear_Smoothed", "Hankel", "FLk-NN")),
                "S11d")
        co2 = pv[pv.column == "CO2(%-sat)-CO2"]
        add(sec, "Dissolved CO2: FLk-NN share of values outside context range, median over outages",
            f"{100 * co2[co2.method == 'FLk-NN'].out_of_range_frac.median():.1f}%", "S11d")
        ev = co2[(co2.experiment == "EXP003") & (co2.reactor == "R01") & (co2.event_id == 0)].set_index("method")
        if len(ev):
            add(sec, "EXP003 R01 outage 1 (Fig. 6g), dissolved CO2: boundary jump Hankel / Smoothed Linear / Linear",
                f"{fmt(ev.boundary_jump['Hankel'])} / {fmt(ev.boundary_jump['Linear_Smoothed'])} / "
                f"{fmt(ev.boundary_jump['Linear'])}", "plausibility.csv")

    pdp = read(os.path.join(a.eval, "tables", "paired_diff_by_duration_phase.csv"))
    if pdp is not None:
        pdp = pdp[pdp.design == PRIMARY_DESIGN]
        for met, m in (("NRMSE_run", "Spline"), ("NDTW_run", "Hankel")):
            for by in ("duration_bin", "phase"):
                g = pdp[(pdp.metric == met) & (pdp.method == m) & (pdp.by == by)]
                if by == "duration_bin":
                    g = g.assign(s=g.level.map(lambda l: float(l.strip("([").split(",")[0]))).sort_values("s")
                else:
                    g = g.assign(s=g.level.map(PHASE_ORDER.index)).sort_values("s")
                add("4 Robustness", f"{name(m)}, {METRIC_PLAIN[met]}: paired difference to Linear by "
                    f"{'duration (3–6, 6–9, 9–12, 12–15 h)' if by == 'duration_bin' else 'phase (early, exponential, stationary)'}",
                    " / ".join(fmt(x) for x in g.median_diff), "S13")

    sec = "9 FLk-NN mechanism (diagnostics)"
    fk = _flknn_by_cluster(a.diag)
    if fk is not None:
        for r in fk.itertuples():
            add(sec, f"Cluster {r.cluster}: share of lags ≤ 2 samples; max |lag|; k-NN coverage of gap points (median)",
                f"{100 * r.lag_le2:.0f}%; {r.lag_max:.0f}; {100 * r.knn_cov:.1f}%" if hasattr(r, "lag_le2")
                else f"–; –; {100 * r.knn_cov:.1f}%", "S14")
            add(sec, f"Cluster {r.cluster}: RMS Fourier − linear (median, max); stitching correction; FLk-NN − linear (z)",
                f"{fmt(r.fourier_med)} ({fmt(r.fourier_max)}); {fmt(r.stitch)}; {fmt(r.flknn_lin)}", "S14")
            add(sec, f"Cluster {r.cluster}: R² straight line to FLk-NN − Fourier (median; share > 0.99); roughness ratio",
                f"{r.r2_line:.3f}; {100 * r.r2_line_099:.0f}%; {fmt(r.rough)}", "S14")

    sec = "10 Hankel and the periodicity of dissolved CO2 (diagnostics)"
    co = read(os.path.join(a.diag, "hankel_co2_coherence.csv"))
    if co is not None:
        for p in PHASE_ORDER:
            d = co[co.phase == p].r_hankel_vs_truth_deviation
            add(sec, f"Simulated gaps, {p}: share with r > 0; median r", f"{100 * (d > 0).mean():.0f}%; {d.median():.2f}", "S15a")
        per = co[co.acf_truth > PERIODIC_ACF]
        both = per[per.acf_hankel > PERIODIC_ACF]
        add(sec, "Simulated gaps with periodic truth: n (early); share r > 0; median r",
            f"{len(per)} of {len(co)} ({int((per.phase == 'early').sum())}); "
            f"{100 * (per.r_hankel_vs_truth_deviation > 0).mean():.0f}%; {per.r_hankel_vs_truth_deviation.median():.2f}", "S15a")
        add(sec, "Periodic truth: period of truth, median (IQR), h",
            f"{per.period_truth_h.median():.2f} ({per.period_truth_h.quantile(0.25):.2f}–{per.period_truth_h.quantile(0.75):.2f})", "S15a")
        if len(both):
            add(sec, "Hankel also periodic: n; median relative period error; amplitude ratio Hankel / truth (periodic truth)",
                f"{len(both)}; {100 * np.median(np.abs(both.period_hankel_h - both.period_truth_h) / both.period_truth_h):.1f}%; "
                f"{np.median(per.sd_detrended_hankel / per.sd_detrended_truth):.2f}", "S15a")
    cr = read(os.path.join(a.diag, "hankel_co2_real_outages.csv"))
    if cr is not None:
        for r in cr[cr.period_hankel_deviation_h.notna()].itertuples():
            add(sec, f"Real outage {r.experiment} {r.reactor} #{r.event_id + 1}: period Hankel − linear vs. CO2 before / after, h",
                f"{r.period_hankel_deviation_h:.2f} vs. {fmt(r.period_context_left_h)} / {fmt(r.period_context_right_h)}", "S15b")
        add(sec, "Real outages with periodic Hankel deviation", f"{int(cr.period_hankel_deviation_h.notna().sum())} of {len(cr)}", "S15b")

    rt = _runtimes(a.results)
    if rt is not None:
        for m, g in rt.groupby("method"):
            add("11 Runtime", f"{name(m)}: seconds per reactor run, median (min–max)",
                f"{fmt(g.seconds.median())} ({fmt(g.seconds.min())}–{fmt(g.seconds.max())})", "S16")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=EVAL_DIR)
    ap.add_argument("--down", default=DOWN_DIR)
    ap.add_argument("--real", default=REAL_DIR)
    ap.add_argument("--results", default=cfg.RESULTS_DIR)
    ap.add_argument("--sim", default=cfg.SIM_DIR)
    ap.add_argument("--diag", default=DIAG_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")         # "→" in the final message (Windows console)
    plt = setup_style()
    out = Out(a.out)
    tables(out, a)
    for f in (fig_trajectories, fig_effects, fig_duration_phase, fig_real_outages, fig_downstream, fig_per_run):
        try:
            f(out, a, plt)
        except Exception as e:                       # a single figure fails → still produce the rest
            import traceback
            print(f"ERROR in {f.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    key_numbers(out, a)                              # after the figures: includes their notes
    print(f"{len(out.written)} outputs → {a.out}")
    for w in out.written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
