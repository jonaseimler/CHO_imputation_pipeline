"""
Correlation matrix for the ground-truth sensor data.

Source : data/simulated/simulated_data_1_gaps.pkl   (simulate_gaps.py)
Subset : all experiments, reactor R01, X_true, all 30 sensor columns

Output : results/clustering/correlation_matrix.csv
         results/clustering/correlation_matrix.png
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────
import paths
PKL_FILE  = paths.SIM_DIR / "simulated_data_1_gaps.pkl"
OUT_CSV   = paths.CLUSTER_DIR / "correlation_matrix.csv"
OUT_PNG   = paths.CLUSTER_DIR / "correlation_matrix.png"


# ── short, readable column labels ────────────────────────────────────────────
def short_label(col: str) -> str:
    """Strip the module suffix for display; keep the measurement part."""
    name, module = col.rsplit("-", 1)
    return f"{name}\n[{module}]"


def main():
    paths.CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

    # ── load ─────────────────────────────────────────────────────────────────
    print("Loading data …")
    with open(PKL_FILE, "rb") as f:
        data = pickle.load(f)

    # Collect X_true / R01 for every experiment and concatenate
    frames = []
    for exp_id, exp_dict in data.items():
        df = exp_dict["R01"]["X_true"]
        frames.append(df)

    combined: pd.DataFrame = pd.concat(frames, axis=0)
    print(f"Combined shape (all experiments, R01): {combined.shape}")

    labels = [short_label(c) for c in combined.columns]

    # ── compute correlation matrix ───────────────────────────────────────────
    print("Computing Pearson correlations …")
    corr: pd.DataFrame = combined.corr(method="pearson")
    corr.index   = labels
    corr.columns = labels

    # ── save CSV ─────────────────────────────────────────────────────────────
    corr.to_csv(OUT_CSV)
    print(f"Saved CSV  -> {OUT_CSV}")

    # ── plot ─────────────────────────────────────────────────────────────────
    print("Plotting …")
    N = len(corr)
    fig, ax = plt.subplots(figsize=(16, 14))

    # custom diverging palette
    cmap = sns.diverging_palette(220, 20, s=80, l=45, as_cmap=True)

    sns.heatmap(
        corr,
        ax=ax,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.4,
        linecolor="#1e2130",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6, "color": "white"},
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )

    # ── style ────────────────────────────────────────────────────────────────
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    ax.tick_params(axis="both", colors="white", labelsize=7, length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=8)
    cbar.ax.yaxis.label.set_color("white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("#ffffff30")

    ax.set_title(
        "Pearson Correlation Matrix — CHO Sensor Data\n"
        "Source: simulated_data_1_gaps.pkl | R01 | X_true | 30 features",
        color="white", fontsize=13, pad=14, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved PNG  -> {OUT_PNG}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
