"""
real_outages.py
─────────────────────────────────────────────────────────────────────────────
Application of the tuned methods to the real outages in EXP003, EXP005 and EXP006.

Real outages (flag_gap_global ≥ MIN_OUTAGE_H, per sensor module):
  EXP003  14.6 h + 3.5 h   all three ArcAir modules (CO2, TCD, VCD)
  EXP005  10.7 h           all modules; sampling 60 s instead of 30 s
  EXP006  15.4 h + 10.4 h  CO2 and TCD modules only; the VCD data of the outages
                            were restored from an ArcAir log file (starting
                            72–86 min after outage onset, linearly interpolated before)
                            → partial outage: Hankel/FLk-NN can use the
                              VCD channels in the same cluster, the baselines cannot.

Procedure:
  1. Load and clean as in simulate_gaps.load_and_clean_data (outliers → NaN,
     local interpolation, modules merged per reactor). The columns of the failed
     module are then reset to NaN within the outage (the standard loader would
     interpolate them).
  2. Impute with all methods; parameters as in the main evaluation
     (revision_config, n_gaps = number of outage events of the reactor).
  3. Without ground truth: plausibility indicators per outage × column × method
     (range violation, noise level, boundary jump, deviation from Linear) and
     figures. The offline VCD samples in the outages are used in
     downstream_evaluation.py (PLS soft sensor).

Note:
  • EXP005 (60 s): Hankel is parametrized relative to the gap length and
    transfers; the FLk-NN max_lag and the smoothing window (100 points) are
    defined in samples and correspond to twice the time span here.

Output (OUT_DIR):
  outage_events.csv            outage events per reactor (and modules)
  imputed/{exp}_{reactor}.pkl  X_missing + imputed values per method
  plausibility.csv             indicators per event × column × method
  plausibility_summary.csv     median per method × cluster
  fig_*.png                    figures

  python real_outages.py                    # all methods (resumable)
  python real_outages.py --methods Hankel   # add Hankel only (memory: 15-h outages in the 23-variable cluster)
  python real_outages.py --plots            # figures from stored results only
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import revision_config as cfg
import run_imputation_multiseed as rim
from simulate_gaps import DATA_DIR, NUMERIC_COLS, RELEVANT_SHEETS

OUT_DIR = str(cfg.paths.REAL_OUTAGE_DIR)
EXPERIMENTS = ["EXP003", "EXP005", "EXP006"]
MIN_OUTAGE_H = 2.0          # as gap_length_detection.py (threshold_hours)
METHOD_COLORS = {           # as paper_outputs.py
    "Linear": "#4878CF", "Linear_Smoothed": "#6ACC65", "Spline": "#D65F5F",
    "Spline_Smoothed": "#B47CC7", "FLk-NN": "#C4AD66", "Hankel": "#77BEDB",
}
PLOT_VARS = ["Permittivity(pF/cm)-VCD", "TCD(AU)-TCD", "CO2(%-sat)-CO2"]
CONTEXT_FRACTION = 1.0      # context per side in figures/indicators: 1 · outage duration


# ── 1. Loading ──────────────────────────────────────────────────────────────────
def _blocks(flag: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True blocks as (start, end_exclusive)."""
    d = np.diff(flag.astype(int), prepend=0, append=0)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def load_outage_run(exp: str) -> tuple[dict, pd.DataFrame]:
    """
    Returns ({reactor: X_missing}, outage intervals per reactor × module).
    Cleaning as in simulate_gaps.load_and_clean_data; the columns of a module
    are NaN within its outage interval.
    """
    xls = pd.ExcelFile(os.path.join(DATA_DIR, f"{exp}_FLAGGED.xlsx"))
    per_reactor, intervals = {}, []
    for sheet in RELEVANT_SHEETS:
        df = pd.read_excel(xls, sheet_name=sheet)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        _, reactor, module = sheet.split("_")
        cols = [c for c in NUMERIC_COLS if c in df.columns]
        clean = df[cols].copy()
        for c in cols:
            flag_col = f"flag_outlier__{c}"
            if flag_col in df.columns:
                clean.loc[df[flag_col].astype(bool).values, c] = np.nan

        dt = df.index.to_series().diff().median().total_seconds()
        for st, en in _blocks(df["flag_gap_global"].astype(bool).values):
            if (en - st) * dt / 3600 >= MIN_OUTAGE_H:
                intervals.append({"experiment": exp, "reactor": reactor, "module": module,
                                  "t_start": df.index[st], "t_end": df.index[en - 1],
                                  "n_points_sheet": en - st, "dt_s": dt})

        clean = clean.interpolate(method="linear", limit_direction="both").bfill().ffill()
        per_reactor.setdefault(reactor, []).append(clean.add_suffix(f"-{module}"))

    iv = pd.DataFrame(intervals)
    runs = {}
    for reactor, dfs in per_reactor.items():
        merged = pd.concat(dfs, axis=1, join="outer", sort=True)
        merged = merged.interpolate(method="linear", limit_direction="both").bfill().ffill()
        for r in iv[iv.reactor == reactor].itertuples():
            mod_cols = [c for c in merged.columns if c.endswith(f"-{r.module}")]
            merged.loc[r.t_start:r.t_end, mod_cols] = np.nan
        runs[reactor] = merged
    return runs, iv


def outage_events(X_missing: pd.DataFrame) -> pd.DataFrame:
    """Outage events = contiguous row blocks with at least one missing column."""
    rows = []
    any_missing = X_missing.isna().any(axis=1).values
    t = X_missing.index
    for eid, (st, en) in enumerate(_blocks(any_missing)):
        miss_cols = X_missing.columns[X_missing.iloc[st:en].isna().any()].tolist()
        modules = sorted({c.rsplit("-", 1)[1] for c in miss_cols})
        rows.append({"event_id": eid, "start_idx": st, "end_idx": en, "n_points": en - st,
                     "start_h": (t[st] - t[0]).total_seconds() / 3600,
                     "end_h": (t[en - 1] - t[0]).total_seconds() / 3600,
                     "duration_h": (t[en - 1] - t[st]).total_seconds() / 3600,
                     "modules_missing": "+".join(modules), "n_cols_missing": len(miss_cols)})
    return pd.DataFrame(rows)


# ── 2. Imputation ─────────────────────────────────────────────────────────────
def impute_all(X_missing: pd.DataFrame, n_gaps: int, methods: list[str]) -> dict:
    """Returns {method: (imputed values at the missing rows, parameters, diagnostics)}."""
    rows = np.flatnonzero(X_missing.isna().any(axis=1).values)
    mask = X_missing.isna()
    out = {}
    for method in methods:
        t0 = time.time()
        X_imp, params, diag = rim._impute(method, X_missing, n_gaps)
        rim._validate(X_missing.fillna(X_imp), X_imp, mask)   # observed values unchanged, no NaN
        out[method] = {"values": X_imp.iloc[rows], "params": params, "diagnostics": diag,
                       "seconds": time.time() - t0}
        print(f"    {method:<16} {time.time() - t0:6.1f}s", flush=True)
    return out


# ── 3. Plausibility without ground truth ────────────────────────────────────────
def plausibility(X_missing: pd.DataFrame, events: pd.DataFrame, results: dict) -> list[dict]:
    """
    Per event × missing column × method (context = CONTEXT_FRACTION · duration per side):
      out_of_range_frac  share of imputed values outside [min, max] of the context
      roughness_ratio    std of 1st differences in the gap / in the context
                         (1 = realistic measurement noise, 0 = smooth curve)
      boundary_jump      mean jump at both edges / std of 1st differences in the context
      rms_dev_linear     RMS(method − Linear) / std of the context
    """
    rows = []
    n = len(X_missing)
    for ev in events.itertuples():
        st, en = ev.start_idx, ev.end_idx
        c = int(round(CONTEXT_FRACTION * (en - st)))
        left, right = X_missing.iloc[max(0, st - c):st], X_missing.iloc[en:min(n, en + c)]
        for col in X_missing.columns:
            gap_nan = X_missing[col].iloc[st:en].isna().values
            if not gap_nan.any():
                continue
            ctx = pd.concat([left[col], right[col]]).dropna().values
            noise = np.nanstd(np.concatenate([np.diff(left[col].dropna().values),
                                              np.diff(right[col].dropna().values)]))
            lin = results["Linear"]["values"][col].loc[X_missing.index[st:en]].values
            lo, hi = ctx.min(), ctx.max()
            for method, res in results.items():
                y = res["values"][col].loc[X_missing.index[st:en]].values
                jumps = []
                if st > 0:
                    jumps.append(abs(y[0] - X_missing[col].iloc[st - 1]))
                if en < n:
                    jumps.append(abs(y[-1] - X_missing[col].iloc[en]))
                rows.append({
                    "event_id": ev.event_id, "column": col, "method": method,
                    "out_of_range_frac": float(((y < lo) | (y > hi)).mean()),
                    "roughness_ratio": float(np.std(np.diff(y)) / noise) if noise > 0 else np.nan,
                    "boundary_jump": float(np.mean(jumps) / noise) if noise > 0 and jumps else np.nan,
                    "rms_dev_linear": float(np.sqrt(np.mean((y - lin) ** 2)) / ctx.std()) if ctx.std() > 0 else np.nan,
                })
    return rows


# ── 4. Figures ────────────────────────────────────────────────────────────
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#888888"})
    return plt


def _plot_panel(ax, X_missing, res, ev, col, show_methods=True):
    st, en = ev.start_idx, ev.end_idx
    c = int(round(CONTEXT_FRACTION * (en - st)))
    lo, hi = max(0, st - c), min(len(X_missing), en + c)
    t = (X_missing.index - X_missing.index[0]).total_seconds() / 3600
    ax.axvspan(t[st], t[en - 1], color="#888888", alpha=0.12, lw=0)
    ax.plot(t[lo:hi], X_missing[col].values[lo:hi], color="black", lw=0.8, label="Observed")
    if show_methods and X_missing[col].iloc[st:en].isna().any():
        idx = X_missing.index[st:en]
        for method, r in res.items():
            ax.plot(t[st:en], r["values"][col].loc[idx].values, color=METHOD_COLORS[method],
                    lw=1.2, label=cfg.METHODS[method])
    elif not X_missing[col].iloc[st:en].isna().any():
        ax.text(0.97, 0.05, "module running\n(observed)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="#555555")
    ax.set_xlim(t[lo], t[hi - 1])


def make_plots(out_dir: str):
    plt = _setup_mpl()
    from matplotlib.lines import Line2D
    events = pd.read_csv(os.path.join(out_dir, "outage_events.csv"))
    data = {}
    for exp in EXPERIMENTS:
        for reactor in ["R01", "R02"]:
            p = os.path.join(out_dir, "imputed", f"{exp}_{reactor}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data[(exp, reactor)] = pickle.load(f)

    present = [m for m in cfg.METHODS if any(m in d["results"] for d in data.values())]
    handles = [Line2D([0], [0], color="black", lw=0.8, label="Observed")] + \
              [Line2D([0], [0], color=METHOD_COLORS[m], lw=1.2, label=cfg.METHODS[m]) for m in present]

    # detail: all events and plot variables per experiment × reactor
    for (exp, reactor), d in data.items():
        ev = events[(events.experiment == exp) & (events.reactor == reactor)]
        fig, axes = plt.subplots(len(PLOT_VARS), len(ev), figsize=(3.3 * len(ev) + 1.2, 1.9 * len(PLOT_VARS)),
                                 squeeze=False)
        for j, e in enumerate(ev.itertuples()):
            for i, col in enumerate(PLOT_VARS):
                _plot_panel(axes[i, j], d["X_missing"], d["results"], e, col)
                if j == 0:
                    axes[i, j].set_ylabel(col.rsplit("-", 1)[0])
                if i == 0:
                    axes[i, j].set_title(f"{exp} {reactor}, outage {e.event_id + 1} "
                                         f"({e.duration_h:.1f} h; {e.modules_missing})")
                if i == len(PLOT_VARS) - 1:
                    axes[i, j].set_xlabel("Cultivation time (h)")
        fig.legend(handles=handles, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.02), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out_dir, f"fig_{exp}_{reactor}.png"))
        plt.close(fig)

    # overview: longest event in R01 per experiment
    sel = []
    for exp in EXPERIMENTS:
        ev = events[(events.experiment == exp) & (events.reactor == "R01")]
        if len(ev) and (exp, "R01") in data:
            sel.append((exp, ev.loc[ev.duration_h.idxmax()]))
    fig, axes = plt.subplots(len(PLOT_VARS), len(sel), figsize=(6.7, 1.75 * len(PLOT_VARS)), squeeze=False)
    for j, (exp, e) in enumerate(sel):
        d = data[(exp, "R01")]
        for i, col in enumerate(PLOT_VARS):
            _plot_panel(axes[i, j], d["X_missing"], d["results"], e, col)
            if j == 0:
                axes[i, j].set_ylabel(col.rsplit("-", 1)[0])
            if i == 0:
                axes[i, j].set_title(f"{exp} R01 ({e.duration_h:.1f} h)")
            if i == len(PLOT_VARS) - 1:
                axes[i, j].set_xlabel("Cultivation time (h)")
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "fig_real_outages_overview.png"))
    plt.close(fig)
    print(f"Figures → {out_dir}")


# ── Control ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots", action="store_true", help="figures from stored results only")
    ap.add_argument("--methods", nargs="+", default=list(cfg.METHODS), choices=list(cfg.METHODS),
                    help="methods to compute; stored results of other methods are kept")
    ap.add_argument("--force", action="store_true", help="recompute stored results of the selected methods")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    if args.plots:
        make_plots(args.out)
        return 0

    os.makedirs(os.path.join(args.out, "imputed"), exist_ok=True)
    rim._STATE["cluster_map"] = cfg.load_cluster_map()
    rim._STATE["params"] = {}
    if "FLk-NN" in args.methods:
        rim._STATE["params"]["FLk-NN"] = cfg.load_flknn_params()
    if "Hankel" in args.methods:
        rim._STATE["params"]["Hankel"] = cfg.load_hankel_params()
    expected_cols = sorted(c for cols in rim._STATE["cluster_map"].values() for c in cols)

    all_events, all_intervals, plaus = [], [], []
    for exp in EXPERIMENTS:
        print(f"{exp}: loading …", flush=True)
        runs, intervals = load_outage_run(exp)
        all_intervals.append(intervals)
        for reactor, X_missing in sorted(runs.items()):
            if sorted(X_missing.columns) != expected_cols:
                raise RuntimeError(f"{exp} {reactor}: columns differ from the cluster assignment: "
                                   f"{sorted(set(expected_cols) ^ set(X_missing.columns))}")
            events = outage_events(X_missing)
            n_gaps = len(events)
            print(f"  {reactor}: {n_gaps} outage(s) "
                  f"{[f'{d:.1f} h ({m})' for d, m in zip(events.duration_h, events.modules_missing)]}", flush=True)
            for method in cfg.CLUSTER_WISE_METHODS & set(args.methods):
                cfg.check_complete(rim._STATE["params"][method], [n_gaps], sorted(rim._STATE["cluster_map"]), method)

            path = os.path.join(args.out, "imputed", f"{exp}_{reactor}.pkl")
            results = {}
            if os.path.exists(path):
                with open(path, "rb") as f:
                    old = pickle.load(f)
                if not old["X_missing"].equals(X_missing):
                    raise RuntimeError(f"{path}: stored input data differ – rerun with --force.")
                results = old["results"]
            todo = [m for m in args.methods if args.force or m not in results]
            results.update(impute_all(X_missing, n_gaps, todo))
            results = {m: results[m] for m in cfg.METHODS if m in results}   # fixed order
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"X_missing": X_missing, "events": events, "results": results}, f)
            os.replace(tmp, path)
            all_events.append(events.assign(experiment=exp, reactor=reactor,
                                            dt_s=X_missing.index.to_series().diff().median().total_seconds()))
            plaus += [{"experiment": exp, "reactor": reactor, **r} for r in plausibility(X_missing, events, results)]

    pd.concat(all_events).to_csv(os.path.join(args.out, "outage_events.csv"), index=False)
    pd.concat(all_intervals).to_csv(os.path.join(args.out, "outage_intervals_by_module.csv"), index=False)
    pl = pd.DataFrame(plaus)
    col_cluster = {c: cfg.PAPER_CLUSTER[cid] for cid, cols in rim._STATE["cluster_map"].items() for c in cols}
    pl["cluster"] = pl.column.map(col_cluster)
    pl.to_csv(os.path.join(args.out, "plausibility.csv"), index=False)
    summ = (pl.groupby(["cluster", "method"])[["out_of_range_frac", "roughness_ratio", "boundary_jump", "rms_dev_linear"]]
            .median().reset_index())
    summ.to_csv(os.path.join(args.out, "plausibility_summary.csv"), index=False)
    print(summ.round(3).to_string(index=False))
    make_plots(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
