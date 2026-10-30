"""
gap_length_detection.py
─────────────────────────────────────────────────────────────────────────────
Durations of the real outages (flag_gap_global) and outlier counts per column
in the flagged workbooks (gap_outlier_detect.py). The outage durations set the
bounds for the gap simulation.
"""

import pandas as pd
import numpy as np
import os

# --- Configuration ---
threshold_hours = 2
threshold_minutes = threshold_hours * 60

def detect_sampling_interval(df, date_col='Date'):
    """
    Sampling interval in minutes (median of the differences, robust to gaps).
    """
    if date_col not in df.columns:
        return 1.0
    
    dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
    
    if len(dates) < 2:
        return 1.0
    
    diffs = dates.diff().dropna()
    median_diff = diffs.median()
    minutes = median_diff.total_seconds() / 60.0
    
    if abs(minutes - 0.5) < 0.05: return 0.5
    if abs(minutes - 1.0) < 0.05: return 1.0
    
    return minutes

def get_gap_details(df, sampling_rate):
    """
    Identifies gaps where flag_gap_global is True.
    Returns a list of dicts: {'start_time': ..., 'duration_min': ...}
    """
    if 'flag_gap_global' not in df.columns or 'Date' not in df.columns:
        return []

    # Ensure boolean
    is_gap = df['flag_gap_global'].astype(bool)
    
    if not is_gap.any():
        return []

    # Create groups for consecutive values to identify blocks
    # (is_gap != is_gap.shift()).cumsum() creates a new ID every time the value changes (True->False or False->True)
    # We need to work on a copy to avoid SettingWithCopy warnings if df is a slice
    df_temp = df.copy()
    df_temp['gap_group'] = (is_gap != is_gap.shift()).cumsum()
    
    gap_details = []
    
    # Filter only the rows that are gaps
    gap_rows = df_temp[is_gap]
    
    if gap_rows.empty:
        return []

    # Group by the continuous blocks
    for _, group in gap_rows.groupby('gap_group'):
        start_time = group['Date'].iloc[0]
        length_rows = len(group)
        duration_min = length_rows * sampling_rate
        
        gap_details.append({
            'start_time': start_time,
            'duration_min': duration_min,
            'length_rows': length_rows
        })
        
    return gap_details

def count_outliers(df):
    """
    Count True values in all 'flag_outlier__' columns.
    Returns {column: count, 'TOTAL_SHEET': sum}.
    """
    outlier_cols = [col for col in df.columns if col.startswith("flag_outlier__")]
    
    stats = {}
    total_sheet = 0
    
    for col in outlier_cols:
        count = df[col].astype(bool).sum()
        stats[col] = count
        total_sheet += count
        
    stats['TOTAL_SHEET'] = total_sheet
    return stats

def calculate_metrics_from_minutes(gap_durations_min, context_name):
    """
    Summary statistics of gap durations (minutes).
    """
    if not gap_durations_min:
        return {
            "Name": context_name,
            "Count Gaps": 0,
            "Avg Length (Min)": 0,
            "Min Length (Min)": 0,
            "Max Length (Min)": 0,
            "Gaps > 2h": 0
        }
    
    arr = np.array(gap_durations_min)
    
    return {
        "Name": context_name,
        "Count Gaps": len(arr),
        "Avg Length (Min)": np.mean(arr),
        "Min Length (Min)": np.min(arr),
        "Max Length (Min)": np.max(arr),
        "Gaps > 2h": np.sum(arr > threshold_minutes)
    }

def analyze_gaps_and_outliers(folder_path):
    target_sheets = [
        "FLAGGED_R01_CO2", "FLAGGED_R01_TCD", "FLAGGED_R01_VCD",
        "FLAGGED_R02_CO2", "FLAGGED_R02_TCD", "FLAGGED_R02_VCD"
    ]

    file_metrics = []
    total_gaps_minutes = []
    all_detailed_gaps = []
    
    all_outlier_stats = []

    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]
    
    if not files:
        print("No files found.")
        return

    print(f"Analyzing {len(files)} files...")
    print(f"Searching for gaps (flag_gap_global) and outliers (flag_outlier__*).\n")

    total_outliers_all_files = 0

    for file in files:
        file_path = os.path.join(folder_path, file)
        file_gaps_minutes = [] 
        file_outlier_count = 0
        
        try:
            xls = pd.ExcelFile(file_path)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        sheets_in_file = [s for s in xls.sheet_names if s in target_sheets]

        for sheet_name in sheets_in_file:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # --- Gaps ---
            required_cols = ['flag_gap_global', 'Date']
            if not all(col in df.columns for col in required_cols):
                continue

            sampling_rate = detect_sampling_interval(df, date_col='Date')
            gaps = get_gap_details(df, sampling_rate)
            
            lengths_minutes = []
            for g in gaps:
                lengths_minutes.append(g['duration_min'])
                all_detailed_gaps.append({
                    'File': file,
                    'Sheet': sheet_name,
                    'Start Time': g['start_time'],
                    'Duration (Min)': g['duration_min']
                })
            
            file_gaps_minutes.extend(lengths_minutes)
            total_gaps_minutes.extend(lengths_minutes)

            # --- Outliers ---
            outlier_stats = count_outliers(df)
            file_outlier_count += outlier_stats['TOTAL_SHEET']
            
            for col_name, count in outlier_stats.items():
                if col_name == 'TOTAL_SHEET': continue
                if count > 0:
                    all_outlier_stats.append({
                        'File': file,
                        'Sheet': sheet_name,
                        'Column': col_name.replace('flag_outlier__', ''),
                        'Count': count
                    })

        # per-file gap metrics
        fm = calculate_metrics_from_minutes(file_gaps_minutes, file)
        fm['Total Outliers'] = file_outlier_count
        file_metrics.append(fm)
        
        total_outliers_all_files += file_outlier_count

    # overall gap metrics
    total_metrics = calculate_metrics_from_minutes(total_gaps_minutes, "TOTAL (All Files)")
    total_metrics['Total Outliers'] = total_outliers_all_files

    # --- Report ---
    
    print("--- 1. All global gaps ---")
    if all_detailed_gaps:
        all_detailed_gaps.sort(key=lambda x: (x['File'], x['Start Time']))
        for gap in all_detailed_gaps:
            print(f"{gap['File']} ({gap['Sheet']}): Gap of {gap['Duration (Min)']:.1f} min at {gap['Start Time']}")
    else:
        print("No global gaps found.")

    print("\n--- 2. Outliers per column ---")
    if all_outlier_stats:
        df_outliers = pd.DataFrame(all_outlier_stats)
        print(df_outliers.to_string(index=False))
    else:
        print("No outliers found.")

    print("\n--- 3. Summary per file ---")
    df_file = pd.DataFrame(file_metrics)
    if not df_file.empty:
        cols = ["Name", "Count Gaps", "Avg Length (Min)", "Gaps > 2h", "Total Outliers"]
        print(df_file[cols].to_string(index=False, float_format="%.2f"))
    
    print("\n--- 4. Overall (all files) ---")
    print(f"Total gaps: {total_metrics['Count Gaps']}")
    print(f"Mean duration: {total_metrics['Avg Length (Min)']:.2f} min")
    print(f"Gaps > 2 h: {total_metrics['Gaps > 2h']}")
    print(f"Total outliers: {total_metrics['Total Outliers']}")

# --- Run ---
if __name__ == "__main__":
    import paths
    analyze_gaps_and_outliers(str(paths.FLAGGED_DIR))


