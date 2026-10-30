"""
simulate_gaps_multiseed.py
─────────────────────────────────────────────────────────────────────────────
Multi-seed gap simulation on the test experiments.

  • 10 independent realizations (SEEDS) per gap count and design.
  • Independent random streams per (design, seed, gap count, experiment) via
    numpy SeedSequence, so the 1–4-gap scenarios are not nested.
  • Test experiments only (EXP001, EXP002, EXP004); EXP007 is the tuning set.
  • Two designs:
      unfixed – positions drawn per experiment ("random placement").
      fixed   – one position set per seed, applied to all three test experiments.
  • The cleaned data are stored once (clean_data.pkl), the gaps only as a
    position table (gap_positions.csv); apply_gaps() builds X_missing / mask.
  • Durations are stored as realized (points · sampling interval).

Output (OUTPUT_DIR):
  clean_data.pkl        {exp: {reactor: DataFrame}}  – incl. EXP007
  gap_positions.csv     one row per gap
  simulation_config.json
"""

import os
import json
import pickle
import subprocess
import time

import numpy as np
import pandas as pd

from simulate_gaps import load_and_clean_data

# ─── Configuration ────────────────────────────────────────────────────────────
import paths
OUTPUT_DIR = str(paths.MULTISEED_DIR)

TEST_EXPERIMENTS = ["EXP001", "EXP002", "EXP004"]
GAP_COUNTS       = [1, 2, 3, 4]
SEEDS            = list(range(1, 11))          # 10 realizations
DESIGNS          = ["unfixed", "fixed"]

DURATION_H = (3.0, 15.0)   # uniform gap duration (as in simulate_gaps.py)
BORDER     = 10            # min. distance to series border (points)
MIN_SEP    = 10            # min. distance between gaps (points)
MAX_TRIES  = 500

_DESIGN_ID = {"unfixed": 0, "fixed": 1}
_EXP_ID    = {"EXP001": 1, "EXP002": 2, "EXP004": 4, "ALL": 0}


# ──────────────────────────────────────────────────────────────────────────────
def _rng(design: str, seed: int, n_gaps: int, exp: str) -> np.random.Generator:
    """Independent, reproducible random stream per combination."""
    return np.random.default_rng([seed, n_gaps, _DESIGN_ID[design], _EXP_ID[exp]])


def generate_gap_positions(num_rows: int, dt_seconds: float, num_gaps: int,
                           rng: np.random.Generator) -> list[tuple[int, int]]:
    """
    Draw `num_gaps` non-overlapping gaps (start_idx, end_idx).
    Same rules as simulate_gaps._generate_gap_positions: duration ~ U(DURATION_H),
    length ≤ num_rows // (num_gaps + 1), border and separation of 10 points.
    """
    gaps = []
    tries = 0
    while len(gaps) < num_gaps and tries < MAX_TRIES:
        tries += 1
        duration_h  = rng.uniform(*DURATION_H)
        window_size = int((duration_h * 3600) / dt_seconds)
        window_size = min(window_size, num_rows // (num_gaps + 1))

        if num_rows <= window_size + 2 * BORDER:
            break

        start = int(rng.integers(BORDER, num_rows - window_size - BORDER, endpoint=True))
        end   = start + window_size

        overlap = any(
            not (end + MIN_SEP <= g_start or start >= g_end + MIN_SEP)
            for g_start, g_end in gaps
        )
        if not overlap:
            gaps.append((start, end))

    if len(gaps) < num_gaps:
        raise RuntimeError(f"Only {len(gaps)}/{num_gaps} gaps could be placed (num_rows={num_rows}).")

    return sorted(gaps)


def apply_gaps(df: pd.DataFrame, gaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the gaps of one scenario (subset of gap_positions.csv) to a cleaned
    reactor DataFrame. Returns (X_missing, mask).
    """
    X_missing = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for st, en in zip(gaps["start_idx"], gaps["end_idx"]):
        X_missing.iloc[st:en] = np.nan
        mask.iloc[st:en] = True
    return X_missing, mask


def load_scenarios(sim_dir: str = OUTPUT_DIR):
    """Load clean_data.pkl and gap_positions.csv."""
    with open(os.path.join(sim_dir, "clean_data.pkl"), "rb") as f:
        clean = pickle.load(f)
    positions = pd.read_csv(os.path.join(sim_dir, "gap_positions.csv"))
    return clean, positions


# ──────────────────────────────────────────────────────────────────────────────
def _series_info(reactors: dict) -> tuple[int, float, pd.Timestamp]:
    """Shortest reactor length, median sampling interval and start time (first reactor)."""
    first = next(iter(reactors.values()))
    dt = first.index.to_series().diff().median().total_seconds()
    min_len = min(len(df) for df in reactors.values())
    return min_len, dt, first.index[0]


def simulate(clean: dict) -> pd.DataFrame:
    rows = []
    info = {exp: _series_info(clean[exp]) for exp in TEST_EXPERIMENTS}

    for design in DESIGNS:
        for seed in SEEDS:
            for n_gaps in GAP_COUNTS:
                if design == "fixed":
                    # one position set for all test experiments (shortest series, common dt)
                    min_len = min(info[e][0] for e in TEST_EXPERIMENTS)
                    dts = {info[e][1] for e in TEST_EXPERIMENTS}
                    if len(dts) != 1:
                        raise RuntimeError(f"Different sampling intervals: {dts}")
                    shared = generate_gap_positions(min_len, dts.pop(), n_gaps,
                                                    _rng(design, seed, n_gaps, "ALL"))

                for exp in TEST_EXPERIMENTS:
                    min_len, dt, t0 = info[exp]
                    if design == "fixed":
                        gaps = shared
                    else:
                        gaps = generate_gap_positions(min_len, dt, n_gaps,
                                                      _rng(design, seed, n_gaps, exp))

                    for gap_id, (st, en) in enumerate(gaps):
                        rows.append({
                            "design": design, "seed": seed, "n_gaps": n_gaps,
                            "experiment": exp, "gap_id": gap_id,
                            "start_idx": st, "end_idx": en, "n_points": en - st,
                            "start_h": st * dt / 3600, "end_h": en * dt / 3600,
                            "duration_h": (en - st) * dt / 3600,
                            "sampling_rate_sec": dt,
                        })
    return pd.DataFrame(rows)


def _save_config(path: str):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    config = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "test_experiments": TEST_EXPERIMENTS,
        "tuning_experiment_excluded": "EXP007",
        "gap_counts": GAP_COUNTS,
        "seeds": SEEDS,
        "designs": DESIGNS,
        "duration_h_uniform": list(DURATION_H),
        "border_points": BORDER,
        "min_separation_points": MIN_SEP,
        "max_gap_points": "num_rows // (n_gaps + 1)",
        "rng": "numpy.default_rng([seed, n_gaps, design_id, exp_id]); "
               "design_id unfixed=0/fixed=1; exp_id EXP001=1, EXP002=2, EXP004=4, fixed=0",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clean = load_and_clean_data()
    with open(os.path.join(OUTPUT_DIR, "clean_data.pkl"), "wb") as f:
        pickle.dump(clean, f)

    positions = simulate(clean)
    positions.to_csv(os.path.join(OUTPUT_DIR, "gap_positions.csv"), index=False)
    _save_config(os.path.join(OUTPUT_DIR, "simulation_config.json"))
    # tracked copy of the gap positions
    paths.GAP_MASK_DIR.mkdir(parents=True, exist_ok=True)
    positions.to_csv(paths.GAP_MASK_DIR / "test_gaps_multiseed.csv", index=False)
    _save_config(str(paths.GAP_MASK_DIR / "test_gaps_multiseed_config.json"))

    print(f"\n{len(positions)} gaps in "
          f"{positions.groupby(['design', 'seed', 'n_gaps', 'experiment']).ngroups} scenarios saved.")
    print(positions.groupby(["design", "n_gaps"])["duration_h"]
          .describe()[["count", "min", "50%", "max"]].round(2).to_string())
