"""
Cluster the 30 sensor columns into 4 groups based on absolute Pearson correlation.

Method : Agglomerative hierarchical clustering on the distance matrix
         distance = 1 - |corr|   (so perfectly correlated/anti-correlated = 0)

Inputs  : results/clustering/correlation_matrix.csv   (correlation_matrix.py)
Outputs : results/clustering/cluster_assignments.csv
          results/clustering/cluster_dendrogram.png
          results/clustering/cluster_heatmap.png
          printed cluster list in the console
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path

# ── Utility: load cluster assignments ─────────────────────────────────────────

def load_cluster_map(csv_path) -> dict:
    """
    Load cluster assignments from a CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to cluster_assignments.csv (columns: feature, cluster).

    Returns
    -------
    dict
        {cluster_id (int): [list of feature name strings]}
    """
    df = pd.read_csv(csv_path)
    result: dict = {}
    for _, row in df.iterrows():
        cid = int(row["cluster"])
        result.setdefault(cid, []).append(row["feature"])
    return result

# ── paths ─────────────────────────────────────────────────────────────────────
import paths
CSV_IN      = paths.CLUSTER_DIR / "correlation_matrix.csv"
OUT_CSV     = paths.CLUSTER_CSV
OUT_DENDRO  = paths.CLUSTER_DIR / "cluster_dendrogram.png"
OUT_HEATMAP = paths.CLUSTER_DIR / "cluster_heatmap.png"

N_CLUSTERS = 4

# ── palette ───────────────────────────────────────────────────────────────────
BG_COLOR = "white"
TEXT_COLOR = "black"
CLUSTER_COLORS = ["#4fc3f7", "#ff8a65", "#a5d6a7", "#ce93d8"]   # blue, orange, green, purple


# ── helper: format tick label for plots ───────────────────────────────────────
def format_label(label: str) -> str:
    """
    Format feature label for plot ticks (heatmap and dendrogram):
    - Subscript 2 for CO2 -> CO₂
    - Remove suffixes like [VCD], [CO₂], [TCD] for normal columns
    - For Temperature columns, keep suffix with replacements:
      [CO2] -> [DCO₂], [TCD] -> [Turb.], [VCD] -> [Perm.]
    """
    label_clean = label.replace("CO2", "CO₂")
    if label_clean.startswith("Temperature"):
        label_clean = label_clean.replace("[CO₂]", "[DCO₂]")
        label_clean = label_clean.replace("[TCD]", "[Turb.]")
        label_clean = label_clean.replace("[VCD]", "[Perm.]")
        return label_clean
    else:
        parts = label_clean.rsplit(" [", 1)
        return parts[0]


def main():
    # ── load correlation matrix ───────────────────────────────────────────────
    corr_raw = pd.read_csv(CSV_IN, index_col=0)
    # column/index names include newlines from the plot labels → strip them
    corr_raw.columns = [c.replace("\n", " ") for c in corr_raw.columns]
    corr_raw.index   = [c.replace("\n", " ") for c in corr_raw.index]

    # ── build absolute-correlation distance matrix ────────────────────────────
    abs_dist = 1.0 - corr_raw.abs()
    np.fill_diagonal(abs_dist.values, 0.0)  # ensure exact zeros on diagonal

    # scipy linkage expects a condensed distance vector
    condensed = squareform(abs_dist.values, checks=False)

    # ── hierarchical clustering (Ward linkage) ────────────────────────────────
    Z = linkage(condensed, method="ward")
    labels_raw = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")   # 1-indexed cluster ids

    features = list(corr_raw.columns)
    cluster_map = {feat: int(lbl) for feat, lbl in zip(features, labels_raw)}

    # ── print cluster assignments ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Clustering into {N_CLUSTERS} groups  (distance = 1 - |Pearson r|)")
    print(f"{'='*55}")
    for cid in sorted(set(cluster_map.values())):
        members = [f for f, c in cluster_map.items() if c == cid]
        print(f"\n  Cluster {cid}  ({len(members)} features)")
        print(f"  {'-'*50}")
        for m in members:
            print(f"    {m}")
    print()

    # ── save CSV ──────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(
        {"feature": features, "cluster": labels_raw}
    ).sort_values(["cluster", "feature"]).reset_index(drop=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Cluster assignments saved -> {OUT_CSV}")

    # ── 1. dendrogram ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # colour each leaf according to its final cluster
    leaf_order = dendrogram(Z, no_plot=True)["leaves"]
    leaf_clusters = [cluster_map[features[i]] for i in leaf_order]

    def leaf_color_fn(leaf_id):
        # leaf_id here is 0-indexed position in the *original* order
        return CLUSTER_COLORS[cluster_map[features[leaf_id]] - 1]

    dn = dendrogram(
        Z,
        ax=ax,
        labels=[format_label(f) for f in features],
        leaf_rotation=45,
        leaf_font_size=8,
        color_threshold=0.75,          # disable scipy auto-colouring
        link_color_func=lambda k: "gray",
    )

    # colour leaf labels by cluster
    xcoords = [t.get_position()[0] for t in ax.get_xticklabels()]
    for tick, leaf_idx in zip(ax.get_xticklabels(), dn["leaves"]):
        tick.set_color(CLUSTER_COLORS[cluster_map[features[leaf_idx]] - 1])

    # ax.set_title(
    #     "Hierarchical Clustering Dendrogram — 30 CHO Sensor Features\n"
    #     "Distance = 1 - |Pearson r|,  Ward linkage,  4 clusters",
    #     color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=12,
    # )
    ax.set_ylabel("Distance (1 - |r|)", color=TEXT_COLOR, fontsize=11)
    ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=11)
    ax.tick_params(axis="x", colors=TEXT_COLOR, length=0)
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DENDRO, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Dendrogram saved        -> {OUT_DENDRO}")
    plt.close()

    # ── 2. reordered heatmap with cluster bands ───────────────────────────────
    # reorder rows/cols so same-cluster features are adjacent
    order = sorted(range(len(features)), key=lambda i: (cluster_map[features[i]], features[i]))
    feat_ordered = [features[i] for i in order]
    corr_ord = corr_raw.loc[feat_ordered, feat_ordered]

    fig, ax = plt.subplots(figsize=(17, 15))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    cmap = sns.diverging_palette(220, 20, s=80, l=45, as_cmap=True)
    sns.heatmap(
        corr_ord,
        ax=ax,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        linecolor="lightgrey",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "color": "black"},
        cbar_kws={"shrink": 0.65, "label": "Pearson r"},
        xticklabels=[format_label(f) for f in feat_ordered],
        yticklabels=[format_label(f) for f in feat_ordered],
    )

    ax.tick_params(axis="both", colors=TEXT_COLOR, labelsize=9, length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelsize=9)
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.yaxis.label.set_size(9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cbar.outline.set_edgecolor("black")

    # ax.set_title(
    #     "Pearson Correlation Matrix — Reordered by Cluster\n"
    #     "30 CHO Sensor Features  |  4 Clusters  |  Distance = 1 - |r|",
    #     color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=14,
    # )

    plt.tight_layout()
    plt.savefig(OUT_HEATMAP, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Cluster heatmap saved   -> {OUT_HEATMAP}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
