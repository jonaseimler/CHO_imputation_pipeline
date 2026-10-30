"""
simulate_gaps.py
─────────────────────────────────────────────────────────────────────────────
Cleaning of the flagged data and simulation of the tuning gaps (seed 42).

  load_and_clean_data()     outliers → NaN, local gaps interpolated, the three
                            sensor modules merged per reactor (also used by
                            simulate_gaps_multiseed.py)
  simulate_artificial_gaps  1–4 contiguous blackouts per experiment, applied to
                            all sensors of both reactors

Output: data/simulated/simulated_data_{1..4}_gaps.pkl (EXP007 is used by
optimize_*.py), results/gap_masks/tuning_gaps_seed42.csv
"""

import pandas as pd
import numpy as np
import os
import random
import pickle

# ─── Configuration ────────────────────────────────────────────────────────────
import paths
DATA_DIR   = str(paths.FLAGGED_DIR)
OUTPUT_DIR = str(paths.SIM_DIR)      # tuning gaps (EXP007, seed 42) for optimize_*.py
FILES = {
    "EXP001": "EXP001_FLAGGED.xlsx",
    "EXP002": "EXP002_FLAGGED.xlsx",
    "EXP004": "EXP004_FLAGGED.xlsx",
    "EXP007": "EXP007_FLAGGED.xlsx"
}
RELEVANT_SHEETS = [
    "FLAGGED_R01_CO2", "FLAGGED_R01_TCD", "FLAGGED_R01_VCD",
    "FLAGGED_R02_CO2", "FLAGGED_R02_TCD", "FLAGGED_R02_VCD"
]
NUMERIC_COLS = [
    "Cell Density(pF/cm)", "Permittivity(pF/cm)", "DeltaEps(pF/cm)", "Conductivity(mS/cm)",
    "fc(kHz)", "Alpha", "CO2(%-sat)", "Transmission(arb. Unit)", "Reflection(arb. Unit)",
    "C(300kHz)", "C(374kHz)", "C(466kHz)", "C(580kHz)", "C(720kHz)", "C(896kHz)",
    "C(1118kHz)", "C(1392kHz)", "C(1729kHz)", "C(2155kHz)", "C(2689kHz)", "C(3347kHz)",
    "C(4158kHz)", "C(5188kHz)", "C(6447kHz)", "C(8030kHz)", "C(9995kHz)",
    "TCD(AU)", "Temperature(°C)"
]

# ─── How many artificial gaps to simulate per experiment ──────────────────────
# Each gap represents a separate "power outage" scenario and will be applied
# identically to ALL sheets of the experiment (all sensors fail together).
NUM_GAPS    = 4          # scenarios with 1..NUM_GAPS gaps
RANDOM_SEED = 42         # Fixed seed → reproducible gap positions


# ──────────────────────────────────────────────────────────────────────────────
def load_and_clean_data():
    """
    Loads data, cleans outliers and local gaps, and merges sheets per reactor.
    Returns a dictionary: data[experiment][reactor] = DataFrame
    """
    print("Loading and cleaning data...")
    data = {}

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for exp_name, filename in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        print(f"Processing {exp_name}...")
        data[exp_name] = {}
        reactor_dfs = {}

        try:
            xls = pd.ExcelFile(filepath)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        for sheet in RELEVANT_SHEETS:
            if sheet not in xls.sheet_names:
                continue

            df = pd.read_excel(xls, sheet_name=sheet)

            # 1. Filter columns
            valid_cols = [c for c in NUMERIC_COLS if c in df.columns]
            if not valid_cols:
                print(f"  Warning: No valid numeric columns in {sheet}")
                continue

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")

            df_clean = df[valid_cols].copy()

            # 2. Handle Outliers
            for col in valid_cols:
                flag_col = f"flag_outlier__{col}"
                if flag_col in df.columns:
                    mask_outlier = df[flag_col].astype(bool)
                    if mask_outlier.any():
                        df_clean.loc[mask_outlier, col] = np.nan

            # 3. Interpolate local gaps
            df_clean = df_clean.interpolate(method='linear', limit_direction='both')
            df_clean = df_clean.bfill().ffill()

            if df_clean.isna().any().any():
                cols_before = df_clean.shape[1]
                df_clean = df_clean.dropna(axis=1, how='all')
                df_clean = df_clean.dropna()
                if df_clean.shape[1] < cols_before:
                    print(f"  Dropped {cols_before - df_clean.shape[1]} columns due to all-NaNs.")

            if df_clean.empty:
                print(f"  Skipping {sheet} (empty after cleaning)")
                continue

            # Determine reactor and module
            parts = sheet.split('_')
            reactor = parts[1] if len(parts) > 1 else 'Unknown'
            module = parts[2] if len(parts) > 2 else 'Unknown'

            # Append module suffix to all variables to avoid collision
            df_clean = df_clean.add_suffix(f"-{module}")

            if reactor not in reactor_dfs:
                reactor_dfs[reactor] = []
            reactor_dfs[reactor].append(df_clean)

        # Merge dataframes by reactor
        for reactor, dfs in reactor_dfs.items():
            merged = pd.concat(dfs, axis=1, join='outer')
            # Outer join might introduce NaNs due to minor jitter, so we do one final baseline fill
            merged = merged.interpolate(method='linear', limit_direction='both').bfill().ffill()
            data[exp_name][reactor] = merged
            print(f"  Processed Reactor {reactor} (Merged shape: {merged.shape})")

    return data


# ──────────────────────────────────────────────────────────────────────────────
def _generate_gap_positions(num_rows: int, avg_diff_seconds: float,
                             num_gaps: int, rng: random.Random) -> list[tuple]:
    """
    Generates `num_gaps` non-overlapping (start_idx, end_idx, duration_hours)
    tuples for a time series of length `num_rows`.

    Gap durations are uniformly drawn from [3, 15] hours.
    Gaps are separated by at least 10 data points from each other and
    from the series edges to avoid border effects.
    """
    BORDER   = 10          # min. distance from start/end of the series
    MIN_SEP  = 10          # min. distance between two consecutive gaps
    MAX_TRIES = 500        # give up after this many placement attempts

    gaps = []
    tries = 0

    while len(gaps) < num_gaps and tries < MAX_TRIES:
        tries += 1

        duration_h   = rng.uniform(3, 15)
        window_size  = int((duration_h * 3600) / avg_diff_seconds)

        # Safety: each gap may take at most 1/(num_gaps+1) of the series
        window_size = min(window_size, num_rows // (num_gaps + 1))

        if num_rows <= window_size + 2 * BORDER:
            print("  Warning: time series too short for requested gap size.")
            break

        start = rng.randint(BORDER, num_rows - window_size - BORDER)
        end   = start + window_size

        # Check for overlap (with buffer) against already placed gaps
        overlap = any(
            not (end + MIN_SEP <= g_start or start >= g_end + MIN_SEP)
            for g_start, g_end, _ in gaps
        )
        if not overlap:
            gaps.append((start, end, duration_h))

    if len(gaps) < num_gaps:
        print(f"  Warning: only {len(gaps)}/{num_gaps} non-overlapping gaps could be placed.")

    # Return sorted by position so gap_00 is always the earliest
    gaps.sort(key=lambda x: x[0])
    return gaps


# ──────────────────────────────────────────────────────────────────────────────
def simulate_artificial_gaps(data, num_gaps: int = NUM_GAPS):
    """
    Creates `num_gaps` artificial power-outage scenarios per experiment within the SAME time series.

    KEY DESIGN DECISION:
    - Gap positions are determined ONCE per experiment (based on the first sheet).
    - All identical positions are applied to the SAME DataFrame for EACH sheet.
    - This simulates a realistic scenario where an experiment has multiple outages over time.

    Output structure:
        stored_data[exp_name][sheet_name] = {
            "X_true":    DataFrame (ground truth),
            "X_missing": DataFrame (with all NaN gaps),
            "mask":      DataFrame (True where ANY gap values are missing),
            "gap_infos": list of dicts with start/end indices, duration, sampling rate
        }
    """
    print(f"\nSimulating {num_gaps} artificial gap(s) per experiment...")
    stored_data = {}

    rng = random.Random(RANDOM_SEED)   # isolated RNG – reproducible

    for exp_name, reactors in data.items():
        if not reactors:
            continue

        stored_data[exp_name] = {}

        # ── Determine common reference from first reactor ───────────────────────
        first_df = next(iter(reactors.values()))

        if len(first_df) < 2:
            print(f"  Skipping {exp_name}: too few data points.")
            continue

        time_diffs       = first_df.index.to_series().diff().dropna()
        avg_diff_seconds = time_diffs.median().total_seconds()

        # Use the shortest reactor so indices are valid for ALL reactors
        min_len = min(len(df) for df in reactors.values())

        # ── Generate gap positions once for this experiment ───────────────────
        gap_positions = _generate_gap_positions(
            num_rows=min_len,
            avg_diff_seconds=avg_diff_seconds,
            num_gaps=num_gaps,
            rng=rng
        )

        print(f"  {exp_name}: {len(gap_positions)} gap(s) placed "
              f"(sampling interval ≈ {avg_diff_seconds:.0f}s)")

        # ── Apply all gaps to ALL reactors ──────────────────────────────────────
        for reactor_name, df in reactors.items():
            X_missing = df.copy()
            mask = pd.DataFrame(False, index=df.index, columns=df.columns)
            gap_infos = []

            for gap_idx, (start_idx, end_idx, duration_h) in enumerate(gap_positions):
                X_missing.iloc[start_idx:end_idx] = np.nan
                mask.iloc[start_idx:end_idx] = True

                gap_infos.append({
                    "start_idx":        start_idx,
                    "end_idx":          end_idx,
                    "duration_hours":   duration_h,
                    "sampling_rate_sec": avg_diff_seconds
                })

                n_pts = end_idx - start_idx
                if reactor_name == list(reactors.keys())[0]: # Print gap info only once per experiment
                    print(f"    Gap {gap_idx:02d}: {duration_h:.2f}h ({n_pts} pts, "
                          f"idx {start_idx}–{end_idx})")

            stored_data[exp_name][reactor_name] = {
                "X_true":    df,
                "X_missing": X_missing,
                "mask":      mask,
                "gap_infos": gap_infos
            }

    return stored_data


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # each gap count starts with a fresh RNG (seed 42)
    data = load_and_clean_data()
    rows = []
    for num_gaps in range(1, NUM_GAPS + 1):
        simulated_data = simulate_artificial_gaps(data, num_gaps=num_gaps)
        output_file = os.path.join(OUTPUT_DIR, f"simulated_data_{num_gaps}_gaps.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(simulated_data, f)
        print(f"Data saved to {output_file}")
        for exp, reactors in simulated_data.items():
            first = next(iter(reactors.values()))
            for gid, g in enumerate(first["gap_infos"]):
                rows.append({"n_gaps": num_gaps, "experiment": exp, "gap_id": gid, **g})

    # gap positions as a table (EXP007 = tuning gaps)
    paths.GAP_MASK_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.GAP_MASK_DIR / "tuning_gaps_seed42.csv", index=False)
    print("\nSimulation complete.")
