"""
paths.py – central paths, relative to the repository root.

data/      input data (tracked) and intermediate files (not tracked)
  raw/                  EXP00X_input.xlsx                  (tracked; see README)
                        temperature_outlier_flags.csv      (tracked; see README)
  flagged/              EXP00X_FLAGGED.xlsx                (gap_outlier_detect.py)
  simulated/            tuning gaps (seed 42)              (simulate_gaps.py)
  simulated_multiseed/  test gaps, 10 seeds                (simulate_gaps_multiseed.py)
results/   results
  clustering/           correlation matrix, cluster assignment (tracked)
  optimization/         Optuna logs for Hankel and FLk-NN      (tracked)
  gap_masks/            gap positions of all scenarios         (tracked)
  paper/tables/         tables of the manuscript and supplement (tracked)
  imputed_multiseed/ …  imputations and evaluations            (not tracked)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FLAGGED_DIR = DATA_DIR / "flagged"
SIM_DIR = DATA_DIR / "simulated"
MULTISEED_DIR = DATA_DIR / "simulated_multiseed"

RESULTS_DIR = ROOT / "results"
CLUSTER_DIR = RESULTS_DIR / "clustering"
CLUSTER_CSV = CLUSTER_DIR / "cluster_assignments.csv"
OPT_DIR = RESULTS_DIR / "optimization"
HANKEL_OPT_DIR = OPT_DIR / "hankel"
FLKNN_OPT_DIR = OPT_DIR / "flknn"
GAP_MASK_DIR = RESULTS_DIR / "gap_masks"
IMPUTED_DIR = RESULTS_DIR / "imputed_multiseed"
EVAL_DIR = RESULTS_DIR / "evaluation_multiseed"
DIAG_DIR = EVAL_DIR / "diagnostics"
REAL_OUTAGE_DIR = RESULTS_DIR / "real_outages"
DOWNSTREAM_DIR = RESULTS_DIR / "evaluation_downstream"
PAPER_DIR = RESULTS_DIR / "paper"
