from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class ProcessingConfig:
    date_col: str = "Date"
    time_h_col: str = "Time (h)"              # optional, only kept if present
    sampling_period: str = "60s"              # pandas offset alias
    global_gap_min_sheets: int = 2            # gap is global if it appears in >= N sheets per reactor
    zero_isolated_window: int = 10            # +-10 neighbors (10 before, 10 after)

    # --- Outlier detection (rolling median + MAD) ---
    outlier_set_to_nan: bool = False          # default: keep values, only flag
    outlier_default_mode: str = "value"        # "diff" or "value"
    outlier_default_window: int = 21           # rolling window size (samples)
    outlier_default_k: float = 6.0            # threshold in (scaled) MAD units
    # Gap-edge skipping: explicit buffer instead of deriving from window
    # None => buffer = window
    outlier_default_gap_edge_buffer: Optional[int] = None
    outlier_per_col_gap_edge_buffer: Optional[Dict[str, int]] = None

    # Scale floor: prevent exploding scores when rolling MAD is tiny
    # If None or <= 0: disabled
    outlier_min_scale_rel_default: Optional[float] = 0.2
    outlier_min_scale_rel_per_col: Optional[Dict[str, float]] = None

    # Rolling-statistics completeness requirement
    # Rolling median/MAD at time t is computed only if at least ceil(window * frac) valid points exist.
    outlier_min_valid_frac_default: float = 0.1
    outlier_min_valid_frac_per_col: Optional[Dict[str, float]] = None

    outlier_default_center: bool = True
    outlier_per_col_center: Optional[Dict[str, bool]] = field(default_factory=lambda: {
        "TCD(AU)": False,
        "Reflection(arb. Unit)": False,
        "Cell Density(pF/cm)": False,
        "fc(kHz)": False,
        "Temperature(°C)": False
    })

    # Optional per-column overrides (by column name)
    outlier_per_col_mode: Optional[Dict[str, str]] = field(default_factory=lambda: {
        "Cell Density(pF/cm)": "value",
        "Permittivity(pF/cm)": "value",
        "DeltaEps": "value",
        "Conductivity(mS/cm)": "value",
        "fc(kHz)": "value",
        "Alpha": "value",
        "CO2(%-sat)": "value",
        "Transmission(arb. Unit)": "value",
        "Reflection(arb. Unit)": "value",
        # "C(300kHz)": "diff",
        # "C(374kHz)": "diff",
        # "C(466kHz)": "diff",
        # "C(580kHz)": "diff",
        # "C(720kHz)": "diff",
        # "C(896kHz)": "diff",
        # "C(1118kHz)": "diff",
        # "C(1392kHz)": "diff",
        # "C(1729kHz)": "diff",
        # "C(2155kHz)": "diff",
        # "C(2689kHz)": "diff",
        # "C(3347kHz)": "diff",
        # "C(4158kHz)": "diff",
        # "C(5188kHz)": "diff",
        # "C(6447kHz)": "diff",
        # "C(8030kHz)": "diff",
        # "C(9995kHz)": "diff",
        "TCD(AU)": "value",
        "Temperature(°C)": "value",
    })
    outlier_per_col_window: Optional[Dict[str, int]] = field(default_factory=lambda: {
        "Cell Density(pF/cm)": 41,
        "Permittivity(pF/cm)": 41,
        "Conductivity(mS/cm)": 41,
        "DeltaEps(pF/cm)": 41,
        "fc(kHz)": 41,
        "Alpha": 41,
        "CO2(%-sat)": 25,
        "Transmission(arb. Unit)": 41,
        "Reflection(arb. Unit)": 41,
        "C(300kHz)": 41,
        "C(374kHz)": 41,
        "C(466kHz)": 41,
        "C(580kHz)": 41,
        "C(720kHz)": 41,
        "C(896kHz)": 41,
        "C(1118kHz)": 41,
        "C(1392kHz)": 41,
        "C(1729kHz)": 41,
        "C(2155kHz)": 41,
        "C(2689kHz)": 41,
        "C(3347kHz)": 41,
        "C(4158kHz)": 41,
        "C(5188kHz)": 41,
        "C(6447kHz)": 41,
        "C(8030kHz)": 41,
        "C(9995kHz)": 41,
        "TCD(AU)": 41,
        "Temperature(°C)": 101,
    })
    outlier_per_col_k: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "Transmission(arb. Unit)": 10.0,
        "Reflection(arb. Unit)": 10.0,
        "Cell Density(pF/cm)": 2.0,
        "Permittivity(pF/cm)": 5.0,
        "Conductivity(mS/cm)": 5.0,
        "DeltaEps(pF/cm)": 10.0,
        "fc(kHz)": 8.0,
        "Alpha": 10.0,
        "CO2(%-sat)": 7.0,
        "C(300kHz)": 5.0,
        "C(374kHz)": 5.0,
        "C(466kHz)": 5.0,
        "C(580kHz)": 5.0,
        "C(720kHz)": 5.0,
        "C(896kHz)": 5.0,
        "C(1118kHz)": 5.0,
        "C(1392kHz)": 5.0,
        "C(1729kHz)": 5.0,
        "C(2155kHz)": 5.0,
        "C(2689kHz)": 5.0,
        "C(3347kHz)": 5.0,
        "C(4158kHz)": 5.0,
        "C(5188kHz)": 5.0,
        "C(6447kHz)": 5.0,
        "C(8030kHz)": 5.0,
        "C(9995kHz)": 5.0,
        "TCD(AU)": 1.5,
        "Temperature(°C)": 10.5,
    })

    # Robust scale factor: scaled_MAD = 1.4826 * MAD (common convention)
    outlier_mad_scale: float = 1.4826

    # Avoid flagging near gap edges (derived buffer = window)
    outlier_skip_near_gap_edges: bool = False


# -----------------------------
# Helpers: column role detection
# -----------------------------

def _detect_numeric_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _reactor_id_from_sheet(sheet_name: str) -> Optional[str]:
    """
    Extract reactor prefix like 'R01' from 'R01_CO2'.
    Returns None if no match.
    """
    m = re.match(r"^(R\d+)_", sheet_name)
    return m.group(1) if m else None


# -----------------------------
# Core: per-sheet processing
# -----------------------------

def parse_and_sort_sheet(df: pd.DataFrame, cfg: ProcessingConfig) -> pd.DataFrame:
    if cfg.date_col not in df.columns:
        raise ValueError(f"Missing required column '{cfg.date_col}'")

    out = df.copy()

    # 1. Parse Date to datetime (robust); keep original column name
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col], errors="coerce")
    out = out.dropna(subset=[cfg.date_col])

    # 2. Force numeric types for all columns except Date and Time
    exclude_cols = [cfg.date_col]
    if cfg.time_h_col:
        exclude_cols.append(cfg.time_h_col)

    for col in out.columns:
        if col in exclude_cols:
            continue

        # text entries such as "Error" become NaN, so is_numeric_dtype holds later
        out[col] = pd.to_numeric(out[col], errors='coerce')

    out = out.sort_values(cfg.date_col).reset_index(drop=True)
    return out

def insert_time_gaps_as_nan_rows(df: pd.DataFrame, cfg: ProcessingConfig) -> pd.DataFrame:
    """
    Reindex to a complete grid using 'merge_asof' with tolerance.
    This handles jitter (e.g. 31s instead of 30s) without creating fake gaps.
    """
    df = df.copy()

    # 1. Auto-detect frequency (Median)
    dt_series = cast(pd.Series, df[cfg.date_col]).sort_values()
    if len(dt_series) < 2:
        return df # Not enough data

    median_diff = dt_series.diff().median()
    if not isinstance(median_diff, pd.Timedelta):
        raise ValueError(
            f"Could not infer sampling period from '{cfg.date_col}': median diff is {type(median_diff)}"
        )
    seconds = round(median_diff.total_seconds())
    detected_freq = f"{int(seconds)}s"

    print(f"Auto-detected sampling period: {detected_freq}")

    # 2. Build the regular grid, starting at the first timestamp rounded to the grid
    start_raw = df[cfg.date_col].min()
    end_raw = df[cfg.date_col].max()
    if not isinstance(start_raw, pd.Timestamp) or not isinstance(end_raw, pd.Timestamp):
        raise ValueError(
            f"Could not determine valid start/end timestamps from '{cfg.date_col}'"
        )
    start_time = start_raw.round(freq=detected_freq)
    end_time = end_raw.round(freq=detected_freq)

    full_grid = pd.date_range(start=start_time, end=end_time, freq=detected_freq)
    grid_df = pd.DataFrame({cfg.date_col: full_grid}) # This is our target structure

    # 3. Sort original data (required for merge_asof)
    df = df.sort_values(cfg.date_col)

    # 4. Match each grid point to the nearest sample within half a period
    #    (±15 s at 30 s: catches jitter, but not a missing sample)
    tol = cast(pd.Timedelta, pd.Timedelta(seconds=seconds * 0.5))

    merged = pd.merge_asof(
        grid_df,
        df,
        on=cfg.date_col,
        direction='nearest',
        tolerance=tol
    )

    # 5. Identify gaps: grid points without a matched sample (tracked by a helper column)
    df['_match_found'] = True

    merged = pd.merge_asof(
        grid_df,
        df,
        on=cfg.date_col,
        direction='nearest',
        tolerance=tol
    )

    # _match_found is NaN -> no sample within tolerance -> gap
    merged["flag_gap_inserted"] = merged["_match_found"].isna()

    # Clean up helper
    merged = merged.drop(columns=["_match_found"])

    # 6. Keep the original (jittered) timestamps: merge_asof keeps the grid dates,
    #    so the original date column is carried along under another name.
    df_orig = df.rename(columns={cfg.date_col: "Original_Date"}).drop(columns=["_match_found"])

    merged_final = pd.merge_asof(
        grid_df, # Left: Grid Dates
        df_orig, # Right: Real Data with Original_Date
        left_on=cfg.date_col,
        right_on="Original_Date",
        direction='nearest',
        tolerance=tol
    )

    merged_final["flag_gap_inserted"] = merged_final["Original_Date"].isna()

    # real sample -> original date; gap -> grid date
    merged_final["Final_Date"] = merged_final["Original_Date"].fillna(merged_final[cfg.date_col])

    # Cleanup
    merged_final = merged_final.drop(columns=[cfg.date_col, "Original_Date"])
    merged_final = merged_final.rename(columns={"Final_Date": cfg.date_col})

    # Date as first column
    cols = [cfg.date_col] + [c for c in merged_final.columns if c != cfg.date_col]
    merged_final = merged_final[cols]

    return cast(pd.DataFrame, merged_final)


def flag_isolated_zeros(df: pd.DataFrame, cfg: ProcessingConfig) -> pd.DataFrame:
    """
    Marks isolated zeros for each numeric column:
    a zero at t is invalid if in the window of +-10 samples (excluding itself),
    there are no other zeros.
    """
    out = df.copy()

    exclude_cols = [cfg.date_col, "flag_gap_inserted"]
    if cfg.time_h_col in out.columns:
        exclude_cols.append(cfg.time_h_col)

    numeric_cols = _detect_numeric_columns(out, exclude=exclude_cols)

    # Create per-column flags
    for col in numeric_cols:
        s = out[col]

        # Only evaluate where s is exactly 0 (and not NaN)
        is_zero = s.eq(0) & s.notna()

        if not is_zero.any():
            out[f"flag_invalid_zero__{col}"] = False
            continue

        # Rolling count of zeros in a window of size (2w+1)
        w = cfg.zero_isolated_window
        zero_int = is_zero.astype(int)

        # center=True gives a symmetric window; includes the center point
        # So isolated means rolling_sum == 1 at positions where is_zero is True.
        rolling_sum = zero_int.rolling(window=2 * w + 1, center=True, min_periods=1).sum()

        invalid_zero = is_zero & rolling_sum.eq(1)

        out[f"flag_invalid_zero__{col}"] = invalid_zero.fillna(False)

        # Optional: turn invalid zeros into NaN now (so "local gaps" exist for later interpolation)
        out.loc[out[f"flag_invalid_zero__{col}"], col] = np.nan

    # Add a convenience "any invalid zero in row" flag
    invalid_cols = [c for c in out.columns if c.startswith("flag_invalid_zero__")]
    if invalid_cols:
        out["flag_invalid_zero_any"] = out[invalid_cols].any(axis=1)
    else:
        out["flag_invalid_zero_any"] = False

    return out


def flag_outliers_rolling_mad(df: pd.DataFrame, cfg: ProcessingConfig) -> pd.DataFrame:
    """
    Flags outliers per numeric column using a rolling median + MAD detector.

    Default operates on raw values ("value"). Mode "diff" operates on first differences
    to avoid flagging trends as outliers.
    Excludes:
      - inserted gap rows (flag_gap_inserted == True)
      - NaNs
      - (optionally) samples near gap edges (buffer derived from window)

    Adds:
      - flag_outlier__{col}
      - outlier_score__{col}  (robust z-like score; NaN where not defined)
      - flag_outlier_any
    """
    out = df.copy()

    exclude_cols = [cfg.date_col, "flag_gap_inserted", "flag_gap_global", "flag_gap_local", "flag_invalid_zero_any"]
    if cfg.time_h_col in out.columns:
        exclude_cols.append(cfg.time_h_col)

    # Candidate numeric signal columns (avoid picking up flag/score helper columns)
    numeric_cols = _detect_numeric_columns(out, exclude=exclude_cols)
    numeric_cols = [
        c for c in numeric_cols
        if not c.startswith("flag_") and not c.startswith("outlier_score__")
    ]

    inserted = out.get("flag_gap_inserted", pd.Series(False, index=out.index))
    if inserted is not None:
        inserted = inserted.fillna(False).astype(bool)

    per_mode = cfg.outlier_per_col_mode or {}
    per_window = cfg.outlier_per_col_window or {}
    per_k = cfg.outlier_per_col_k or {}
    per_gap_buffer = cfg.outlier_per_col_gap_edge_buffer or {}
    per_min_scale_rel = cfg.outlier_min_scale_rel_per_col or {}
    per_min_valid_frac = cfg.outlier_min_valid_frac_per_col or {}
    per_center = cfg.outlier_per_col_center or {}

    outlier_flag_cols: List[str] = []

    for col in numeric_cols:
        mode = per_mode.get(col, cfg.outlier_default_mode)
        center_ = bool(per_center.get(col, cfg.outlier_default_center))
        window = int(per_window.get(col, cfg.outlier_default_window))
        k = float(per_k.get(col, cfg.outlier_default_k))
        min_valid_frac = float(per_min_valid_frac.get(col, cfg.outlier_min_valid_frac_default))
        if not (0.0 < min_valid_frac <= 1.0):
            raise ValueError(
                f"outlier_min_valid_frac must be in (0,1], got {min_valid_frac} for column '{col}'"
            )

        min_periods = int(np.ceil(window * min_valid_frac))
        min_periods = max(3, min_periods)      # safety
        min_periods = min(window, min_periods) # safety

        flag_col = f"flag_outlier__{col}"
        score_col = f"outlier_score__{col}"
        outlier_flag_cols.append(flag_col)

        s = out[col]

        if mode not in {"diff", "value"}:
            raise ValueError(f"Unknown outlier mode '{mode}' for column '{col}'. Use 'diff' or 'value'.")

        x = s.diff() if mode == "diff" else s

        # Build eligibility mask:
        # - exclude inserted gap rows
        # - exclude NaNs
        eligible = (~inserted.fillna(False).astype(bool)) & x.notna() if inserted is not None else x.notna()

        # Avoid flagging near gap edges by excluding a neighborhood around inserted rows.
        # Buffer is derived from the chosen window to avoid introducing another parameter.
        if cfg.outlier_skip_near_gap_edges and window > 1 and inserted is not None:
                    gap_buffer = per_gap_buffer.get(col, cfg.outlier_default_gap_edge_buffer)
                    if gap_buffer is None:
                        gap_buffer = window  # default: one window on each side of a gap
                    gap_buffer = int(gap_buffer)

                    if gap_buffer > 0:
                        near_gap = (
                            inserted.astype(int)
                            .rolling(window=2 * gap_buffer + 1, center=True, min_periods=1)
                            .max()
                            .fillna(0)
                            .astype(bool)
                        )
                        eligible = eligible & (~near_gap)

        # Initialize outputs
        out[flag_col] = False
        out[score_col] = np.nan

        # If window is too small or no eligible points, skip safely
        if window < 3 or not bool(eligible.any()):
            continue

        x_eligible = x.where(eligible)

        # Rolling median
        roll_med = x_eligible.rolling(window=window, center=center_, min_periods=min_periods).median()

        # Rolling MAD: median(|x - median(window)|) computed within each window
        def _mad_raw(arr: np.ndarray) -> float:
            med = np.nanmedian(arr)
            return float(np.nanmedian(np.abs(arr - med)))

        roll_mad = x_eligible.rolling(window=window, center=center_, min_periods=min_periods).apply(
            _mad_raw, raw=True
        )
        # Global robust scale (scalar) computed on eligible points
        x_vals = x_eligible.dropna().to_numpy()
        if x_vals.size > 0:
            g_med = float(np.nanmedian(x_vals))
            g_mad = float(np.nanmedian(np.abs(x_vals - g_med)))
            global_scale = cfg.outlier_mad_scale * g_mad
        else:
            global_scale = np.nan

        min_scale_rel = per_min_scale_rel.get(col, cfg.outlier_min_scale_rel_default)
        if min_scale_rel is None:
            min_scale = 0.0
        else:
            min_scale = float(min_scale_rel) * (0.0 if not np.isfinite(global_scale) else global_scale)


        # Robust score

        abs_dev = (x_eligible - roll_med).abs()
        scale = cfg.outlier_mad_scale * roll_mad


        if np.isfinite(min_scale) and min_scale > 0:
            scale = scale.clip(lower=min_scale)

        score = (x_eligible - roll_med).abs() / scale
        # IMPORTANT: do NOT replace zeros with NaN
# division by 0 will yield inf when abs_dev > 0 (good), and NaN when 0/0 (we fix to 0 below)
        score = abs_dev / scale

        zero_scale = scale.eq(0)
        score = score.mask(zero_scale & abs_dev.eq(0), 0.0)

        out[score_col] = score

        # Flag outliers where eligible and score > k
        out[flag_col] = eligible & score.gt(k)

        # Optional: convert outliers to NaN in the ORIGINAL series (not the diff)
        if cfg.outlier_set_to_nan:
            out.loc[out[flag_col], col] = np.nan

    if outlier_flag_cols:
        out["flag_outlier_any"] = out[outlier_flag_cols].any(axis=1)
    else:
        out["flag_outlier_any"] = False

    return out

# -----------------------------
# Core: reactor-level global gap classification
# -----------------------------

def _gap_intervals_from_flagged_sheet(df: pd.DataFrame, cfg: ProcessingConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Compute contiguous intervals of inserted gaps based on flag_gap_inserted.
    Intervals are inclusive start/end timestamps.
    """
    if "flag_gap_inserted" not in df.columns:
        return []

    gap = df["flag_gap_inserted"].fillna(False).to_numpy(dtype=bool)
    times = pd.to_datetime(df[cfg.date_col]).to_numpy()

    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    i = 0
    n = len(gap)
    while i < n:
        if not gap[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and gap[j + 1]:
            j += 1
        intervals.append((pd.Timestamp(times[i].item()), pd.Timestamp(times[j].item())))
        i = j + 1
    return intervals

from collections import Counter

def classify_global_gaps_per_reactor(
    flagged_sheets: Dict[str, pd.DataFrame],
    cfg: ProcessingConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Robust Global Gap Detection using Timestamp Voting.

    Instead of matching exact start/end intervals (which fails if sensors
    drop out slightly differently), we count how many sheets have a gap
    at each specific timestamp.
    """
    # 1. Group sheets by reactor
    reactor_to_sheets: Dict[str, List[str]] = {}
    for sh in flagged_sheets:
        rid = _reactor_id_from_sheet(sh)
        if rid is None:
            continue
        reactor_to_sheets.setdefault(rid, []).append(sh)

    out = {k: v.copy() for k, v in flagged_sheets.items()}

    for rid, sheets in reactor_to_sheets.items():
        # --- Step 1: Collect all gap timestamps across this reactor ---
        all_gap_timestamps = []

        for sh in sheets:
            df = out[sh]
            if "flag_gap_inserted" not in df.columns:
                continue

            # Get the timestamps where a gap was inserted
            # We assume df[cfg.date_col] holds the aligned grid time for gap rows
            gap_times = df.loc[df["flag_gap_inserted"], cfg.date_col]
            all_gap_timestamps.extend(gap_times.tolist())

        # --- Step 2: Count occurrences of each timestamp ---
        # If a timestamp appears in >= N sheets, it's a global gap moment
        time_counts = Counter(all_gap_timestamps)

        # Create a set of timestamps that are considered global
        global_time_set = {
            t for t, count in time_counts.items()
            if count >= cfg.global_gap_min_sheets
        }

        # --- Step 3: Apply flags back to the sheets ---
        for sh in sheets:
            df = out[sh]

            # Initialize columns
            df["flag_gap_global"] = False
            df["flag_gap_local"] = False

            if "flag_gap_inserted" not in df.columns or not bool(df["flag_gap_inserted"].any()):
                out[sh] = df
                continue

            # Boolean masks
            is_gap = df["flag_gap_inserted"].fillna(False)

            # Check which timestamps are in the global set
            # We use .isin() which is very fast
            # pandas' typing expects a Sequence-like; list() keeps behavior identical
            is_global_time = df[cfg.date_col].isin(list(global_time_set))

            # Logic:
            # Global Gap = It is a gap AND the timestamp is global
            # Local Gap  = It is a gap AND the timestamp is NOT global
            df.loc[is_gap & is_global_time, "flag_gap_global"] = True
            df.loc[is_gap & ~is_global_time, "flag_gap_local"] = True

            out[sh] = df

    # Ensure columns exist for non-reactor sheets too
    for sh, df in out.items():
        if "flag_gap_global" not in df.columns:
            df["flag_gap_global"] = False
        if "flag_gap_local" not in df.columns:
            df["flag_gap_local"] = False
        out[sh] = df

    return out


# -----------------------------
# I/O: Excel pipeline
# -----------------------------

def load_workbook_sheets(
    xlsx_path: str,
) -> Dict[str, pd.DataFrame]:
    """
    Loads all sheets as DataFrames.
    """
    xls = pd.ExcelFile(xlsx_path)
    sheets = {}
    for sh in xls.sheet_names:
        sheets[sh] = pd.read_excel(xlsx_path, sheet_name=sh)
    return sheets


def process_workbook(
    xlsx_path: str,
    process_sheet_names: List[str],
    cfg: ProcessingConfig = ProcessingConfig(),
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Returns:
      - raw_sheets: all original sheets (unmodified)
      - flagged_sheets: dict mapping original sheet name -> processed DataFrame
    Only sheets listed in process_sheet_names are processed.
    """
    raw_sheets = load_workbook_sheets(xlsx_path)

    missing = [s for s in process_sheet_names if s not in raw_sheets]
    if missing:
        raise ValueError(f"These requested sheets were not found in workbook: {missing}")

    flagged_sheets: Dict[str, pd.DataFrame] = {}

    for sh in process_sheet_names:
        df = raw_sheets[sh]
        df = parse_and_sort_sheet(df, cfg)
        df = insert_time_gaps_as_nan_rows(df, cfg)
        df = flag_isolated_zeros(df, cfg)
        df = flag_outliers_rolling_mad(df, cfg)
        flagged_sheets[sh] = df

    flagged_sheets = classify_global_gaps_per_reactor(flagged_sheets, cfg)

    return raw_sheets, flagged_sheets


def apply_stored_outlier_flags(
    flagged_sheets: Dict[str, pd.DataFrame],
    stored: pd.DataFrame,
    col: str,
    cfg: ProcessingConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Replaces flag_outlier__{col} with stored flags (columns: sheet, Date) and
    recomputes flag_outlier_any. Scores are left unchanged.
    """
    flag_col = f"flag_outlier__{col}"
    for sh, df in flagged_sheets.items():
        if flag_col not in df.columns:
            continue
        dates = pd.to_datetime(stored.loc[stored["sheet"] == sh, "Date"])
        df[flag_col] = pd.to_datetime(df[cfg.date_col]).isin(dates)
        flag_cols = [c for c in df.columns if c.startswith("flag_outlier__")]
        df["flag_outlier_any"] = df[flag_cols].any(axis=1)
    return flagged_sheets


def write_flagged_workbook(
    out_xlsx_path: str,
    raw_sheets: Dict[str, pd.DataFrame],
    flagged_sheets: Dict[str, pd.DataFrame],
):
    """
    Writes a workbook with:
      - all original sheets unchanged
      - additional sheets FLAGGED_<name> for each processed sheet
    """
    with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
        # Original sheets
        for sh, df in raw_sheets.items():
            df.to_excel(writer, sheet_name=sh, index=False)

        # Flagged sheets
        for sh, df in flagged_sheets.items():
            flagged_name = f"FLAGGED_{sh}"
            # Excel sheet name limit is 31 chars; handle if needed
            if len(flagged_name) > 31:
                flagged_name = flagged_name[:31]
            df.to_excel(writer, sheet_name=flagged_name, index=False)


# -----------------------------
# Run on all experiments
# -----------------------------
EXPERIMENTS = ["EXP001", "EXP002", "EXP003", "EXP004", "EXP005", "EXP006", "EXP007"]
SAMPLING_PERIOD = {"EXP005": "60s"}      # EXP005 was recorded at 60 s, all others at 30 s
PROCESS_SHEETS = ["R01_CO2", "R01_TCD", "R01_VCD", "R02_CO2", "R02_TCD", "R02_VCD"]
# EXP001–EXP004 were flagged with an earlier, unrecorded temperature setting.
# The flags used in the study are stored and applied instead (see README).
STORED_FLAG_COL = "Temperature(°C)"
STORED_FLAG_FILE = "temperature_outlier_flags.csv"

if __name__ == "__main__":
    import paths
    paths.FLAGGED_DIR.mkdir(parents=True, exist_ok=True)
    stored_flags = pd.read_csv(paths.RAW_DIR / STORED_FLAG_FILE)
    for exp in EXPERIMENTS:
        cfg = ProcessingConfig(
            sampling_period=SAMPLING_PERIOD.get(exp, "30s"),
            global_gap_min_sheets=2,
            zero_isolated_window=10,
        )
        raw_sheets, flagged_sheets = process_workbook(
            xlsx_path=str(paths.RAW_DIR / f"{exp}_input.xlsx"),
            process_sheet_names=PROCESS_SHEETS,
            cfg=cfg,
        )
        stored_exp = stored_flags[stored_flags["experiment"] == exp]
        if not stored_exp.empty:
            flagged_sheets = apply_stored_outlier_flags(flagged_sheets, stored_exp, STORED_FLAG_COL, cfg)
        out = paths.FLAGGED_DIR / f"{exp}_FLAGGED.xlsx"
        write_flagged_workbook(out_xlsx_path=str(out), raw_sheets=raw_sheets, flagged_sheets=flagged_sheets)
        print(f"{exp} -> {out}")
