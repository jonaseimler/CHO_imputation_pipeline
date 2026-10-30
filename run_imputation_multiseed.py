"""
run_imputation_multiseed.py
─────────────────────────────────────────────────────────────────────────────
Imputation of all methods on all multi-seed scenarios (simulate_gaps_multiseed.py).
The raw gap values are stored once, so all metrics, statistics and the downstream
evaluation can be computed without re-imputing.

A "unit" = (design, seed, n_gaps, experiment, reactor, method).

Per unit:
  1. Load cleaned reactor data and gap positions, apply the gaps.
  2. Impute – baselines column-wise on the full matrix, FLk-NN / Hankel
     cluster-wise with the optimized parameters (from the Optuna logs,
     see revision_config.py). Faulty channels are not used as inputs.
  3. Validate (the unit fails otherwise):
       • shape, index and columns unchanged,
       • all gap values finite,
       • observed values outside the gaps unchanged (< 1e-6 column std).
  4. Save atomically as .npz:
       rows          row indices of the gap points
       gap_id        gap ID per row (as in gap_positions.csv)
       columns       column names
       y_pred        imputed values (len(rows) × n_columns, float64)
       meta_json     unit, parameters used, runtime, validation values
       diag_json     Hankel diagnostics per gap and cluster (else "[]")
     True values are not duplicated; load_unit() joins them from clean_data.pkl.

Resumable: existing result files are skipped (unless --force).
Logs: run_log.csv (one row per unit) and run_manifest_*.json (configuration,
parameters, input checksums).

Examples:
  python run_imputation_multiseed.py --methods Linear Linear_Smoothed Spline Spline_Smoothed FLk-NN --jobs 3
  python run_imputation_multiseed.py --methods Hankel --jobs 1
  python run_imputation_multiseed.py --seeds 1 --gaps 1 --experiments EXP001 --limit 4   # quick test
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import evaluation_metrics as em
import revision_config as cfg
from simulate_gaps_multiseed import apply_gaps, load_scenarios

OBS_TOLERANCE = 1e-6   # allowed change of observed values (column std)


# ── Paths / loading ─────────────────────────────────────────────────────────────
def result_path(out_dir: str, unit: dict) -> str:
    return os.path.join(out_dir, unit["design"], f"seed_{unit['seed']:02d}",
                        f"{unit['n_gaps']}_gaps", unit["experiment"], unit["reactor"],
                        f"{unit['method']}.npz")


def load_unit(path: str, clean: dict) -> dict:
    """
    Load a result file and add the true values from clean_data.
    Returns a dict with y_true, y_pred (DataFrames indexed by the gap
    timestamps), gap_id (Series), meta, diagnostics.
    """
    with np.load(path, allow_pickle=False) as z:
        rows, gap_id = z["rows"], z["gap_id"]
        columns = [str(c) for c in z["columns"]]
        y_pred = z["y_pred"]
        meta = json.loads(str(z["meta_json"]))
        diag = json.loads(str(z["diag_json"]))
    X_true = clean[meta["experiment"]][meta["reactor"]]
    index = X_true.index[rows]
    return {
        "y_true": X_true[columns].iloc[rows].set_axis(index),
        "y_pred": pd.DataFrame(y_pred, index=index, columns=columns),
        "gap_id": pd.Series(gap_id, index=index, name="gap_id"),
        "meta": meta,
        "diagnostics": diag,
    }


# ── Worker state (loaded once per process) ───────────────────────────────
_STATE: dict = {}


def _init_worker(sim_dir: str, methods: list[str]):
    clean, positions = load_scenarios(sim_dir)
    _STATE["clean"] = clean
    _STATE["positions"] = positions
    _STATE["cluster_map"] = cfg.load_cluster_map()
    _STATE["params"] = {}
    if "FLk-NN" in methods:
        _STATE["params"]["FLk-NN"] = cfg.load_flknn_params()
    if "Hankel" in methods:
        _STATE["params"]["Hankel"] = cfg.load_hankel_params()


# ── Imputation of one unit ──────────────────────────────────────────────────
def _impute(method: str, X_missing: pd.DataFrame, n_gaps: int, exclude=()):
    """
    Returns (X_imputed, parameters used, diagnostics list).
    exclude: faulty channels (evaluation_metrics.is_excluded_sensor). For the cluster-wise
    methods they are not used as inputs (constant channels would distort e.g. the FLk-NN
    neighbor search) and are filled linearly; they are not evaluated. The baselines work
    column-wise and are unaffected.
    """
    if method not in cfg.CLUSTER_WISE_METHODS:
        X_imp = cfg.make_baseline(method).impute(X_missing)
        return X_imp, {"smoothing_window": cfg.SMOOTHING_WINDOW} if "Smoothed" in method else {}, []

    parts, used, diag = [], {}, []
    covered = []
    for cid, features in sorted(_STATE["cluster_map"].items()):
        cols = [c for c in features if c in X_missing.columns]
        if len(cols) != len(features):
            raise RuntimeError(f"Cluster {cid}: columns missing in the data: "
                               f"{sorted(set(features) - set(cols))}")
        params = _STATE["params"][method][(n_gaps, cid)]
        use = [c for c in cols if c not in exclude]
        dropped = [c for c in cols if c in exclude]
        imputer = cfg.make_cluster_imputer(method, params)
        part = [imputer.impute(X_missing[use])[use]] if use else []
        if dropped:
            part.append(cfg.make_baseline("Linear").impute(X_missing[dropped]))
        parts.append(pd.concat(part, axis=1)[cols])
        covered += cols
        used[str(cid)] = {**params, "excluded_inputs": dropped} if dropped else params
        for d in getattr(imputer, "diagnostics_", []):
            diag.append({"cluster_id": cid, **{k: (v.item() if hasattr(v, "item") else v)
                                               for k, v in d.items()}})

    if sorted(covered) != sorted(X_missing.columns):
        raise RuntimeError(f"Clusters do not cover all columns: "
                           f"{sorted(set(X_missing.columns) - set(covered))}")
    X_imp = pd.concat(parts, axis=1)[X_missing.columns]
    return X_imp, used, diag


def _validate(X_true: pd.DataFrame, X_imp: pd.DataFrame, mask: pd.DataFrame) -> float:
    if X_imp.shape != X_true.shape or not X_imp.index.equals(X_true.index) \
            or not X_imp.columns.equals(X_true.columns):
        raise AssertionError("Shape, index or columns of the imputation result differ.")
    m = mask.values
    if not np.isfinite(X_imp.values[m]).all():
        raise AssertionError("Imputed gap values contain NaN/Inf.")
    std = X_true.std().replace(0, 1.0).values
    dev = np.abs(X_imp.values - X_true.values) / std
    max_obs_dev = float(dev[~m].max()) if (~m).any() else 0.0
    if not np.isfinite(max_obs_dev) or max_obs_dev > OBS_TOLERANCE:
        raise AssertionError(f"Observed values changed (max. {max_obs_dev:.2e} std).")
    return max_obs_dev


def run_unit(unit: dict, out_dir: str) -> dict:
    t0 = time.time()
    log = {**unit, "status": "ok", "seconds": np.nan, "n_gap_rows": 0,
           "max_obs_dev_std": np.nan, "error": ""}
    try:
        X_true = _STATE["clean"][unit["experiment"]][unit["reactor"]]
        p = _STATE["positions"]
        gaps = p[(p.design == unit["design"]) & (p.seed == unit["seed"]) &
                 (p.n_gaps == unit["n_gaps"]) & (p.experiment == unit["experiment"])]
        if len(gaps) != unit["n_gaps"]:
            raise RuntimeError(f"{len(gaps)} instead of {unit['n_gaps']} gaps in gap_positions.csv")

        X_missing, mask = apply_gaps(X_true, gaps)
        exclude = [c for c in X_true.columns if em.is_excluded_sensor(unit["experiment"], unit["reactor"], c)]
        X_imp, used_params, diag = _impute(unit["method"], X_missing, unit["n_gaps"], exclude)
        max_obs_dev = _validate(X_true, X_imp, mask)

        rows = np.flatnonzero(mask.values.any(axis=1))
        gap_id = np.full(len(rows), -1, dtype=np.int64)
        for gid, st, en in zip(gaps.gap_id, gaps.start_idx, gaps.end_idx):
            gap_id[(rows >= st) & (rows < en)] = gid
        if (gap_id < 0).any():
            raise AssertionError("Gap rows without a gap_id.")

        seconds = time.time() - t0
        meta = {**unit, "params": used_params, "seconds": round(seconds, 3),
                "max_obs_dev_std": max_obs_dev, "n_gap_rows": int(len(rows)),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}

        path = result_path(out_dir, unit)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(
            tmp, rows=rows, gap_id=gap_id,
            columns=np.array(X_true.columns, dtype=str),
            y_pred=X_imp.values[rows].astype(np.float64),
            meta_json=json.dumps(meta), diag_json=json.dumps(diag),
        )
        os.replace(tmp, path)   # atomic: no partially written result files

        log.update(seconds=round(seconds, 3), n_gap_rows=int(len(rows)),
                   max_obs_dev_std=max_obs_dev)
    except Exception as e:
        log.update(status="error", seconds=round(time.time() - t0, 3),
                   error=f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)!r}")
    return log


# ── Control ─────────────────────────────────────────────────────────────────
def build_units(positions: pd.DataFrame, clean: dict, args) -> list[dict]:
    scen = (positions[["design", "seed", "n_gaps", "experiment"]].drop_duplicates()
            .sort_values(["design", "seed", "n_gaps", "experiment"]))
    scen = scen[scen.design.isin(args.designs) & scen.seed.isin(args.seeds) &
                scen.n_gaps.isin(args.gaps) & scen.experiment.isin(args.experiments)]
    units = []
    for s in scen.itertuples(index=False):
        for reactor in [r for r in sorted(clean[s.experiment]) if r in args.reactors]:
            for method in args.methods:
                units.append({"design": s.design, "seed": int(s.seed), "n_gaps": int(s.n_gaps),
                              "experiment": s.experiment, "reactor": reactor, "method": method})
    return units


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: str, args, n_units: int):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    params = {}
    if "FLk-NN" in args.methods:
        params["FLk-NN"] = {f"{n}_gaps/cluster_{c}": p for (n, c), p in sorted(cfg.load_flknn_params().items())}
    if "Hankel" in args.methods:
        params["Hankel"] = {f"{n}_gaps/cluster_{c}": p for (n, c), p in sorted(cfg.load_hankel_params().items())}
    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit, "git_dirty": dirty,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "sim_dir": args.sim_dir,
        "sha256": {f: _sha256(os.path.join(args.sim_dir, f))
                   for f in ["clean_data.pkl", "gap_positions.csv"]},
        "cluster_csv": cfg.CLUSTER_CSV, "paper_cluster": cfg.PAPER_CLUSTER,
        "selection": {k: v for k, v in vars(args).items() if k not in ("sim_dir", "out")},
        "n_units": n_units,
        "obs_tolerance_std": OBS_TOLERANCE,
        "smoothing_window": cfg.SMOOTHING_WINDOW,
        "params": params,
    }
    path = os.path.join(out_dir, f"run_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=list(cfg.METHODS), choices=list(cfg.METHODS))
    ap.add_argument("--designs", nargs="+", default=["unfixed", "fixed"], choices=["unfixed", "fixed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--gaps", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--experiments", nargs="+", default=["EXP001", "EXP002", "EXP004"])
    ap.add_argument("--reactors", nargs="+", default=["R01", "R02"])
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes (Hankel: 1 recommended, memory)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N units (test)")
    ap.add_argument("--force", action="store_true", help="recompute existing results")
    ap.add_argument("--sim-dir", default=cfg.SIM_DIR)
    ap.add_argument("--out", default=cfg.RESULTS_DIR)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")         # "→" in messages (Windows console)

    # check all parameters up front rather than failing after hours
    cluster_ids = sorted(cfg.load_cluster_map())
    if "FLk-NN" in args.methods:
        cfg.check_complete(cfg.load_flknn_params(), args.gaps, cluster_ids, "FLk-NN")
    if "Hankel" in args.methods:
        cfg.check_complete(cfg.load_hankel_params(), args.gaps, cluster_ids, "Hankel")

    clean, positions = load_scenarios(args.sim_dir)
    units = build_units(positions, clean, args)
    if args.limit:
        units = units[:args.limit]
    todo = [u for u in units if args.force or not os.path.exists(result_path(args.out, u))]
    print(f"{len(units)} units selected, {len(units) - len(todo)} already done, "
          f"{len(todo)} to compute (jobs={args.jobs}).", flush=True)
    if not todo:
        return 0

    os.makedirs(args.out, exist_ok=True)
    manifest = write_manifest(args.out, args, len(todo))
    print(f"Manifest → {manifest}", flush=True)
    log_path = os.path.join(args.out, "run_log.csv")

    def _record(entry: dict, i: int, t_start: float):
        pd.DataFrame([entry]).to_csv(log_path, mode="a", index=False,
                                     header=not os.path.exists(log_path))
        eta = (time.time() - t_start) / i * (len(todo) - i) / 60
        flag = "OK " if entry["status"] == "ok" else "ERR"
        print(f"[{i:>5}/{len(todo)}] {flag} {entry['design']:<7} s{entry['seed']:02d} "
              f"{entry['n_gaps']}g {entry['experiment']} {entry['reactor']} {entry['method']:<15} "
              f"{entry['seconds']:7.1f}s  ETA {eta:6.1f} min"
              + (f"  → {entry['error'][:200]}" if entry["status"] != "ok" else ""), flush=True)

    n_err, t_start = 0, time.time()
    if args.jobs == 1:
        _init_worker(args.sim_dir, args.methods)
        for i, u in enumerate(todo, 1):
            entry = run_unit(u, args.out)
            n_err += entry["status"] != "ok"
            _record(entry, i, t_start)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                                 initargs=(args.sim_dir, args.methods)) as ex:
            futures = [ex.submit(run_unit, u, args.out) for u in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                entry = fut.result()
                n_err += entry["status"] != "ok"
                _record(entry, i, t_start)

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min – {len(todo) - n_err} ok, {n_err} errors. "
          f"Log: {log_path}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
