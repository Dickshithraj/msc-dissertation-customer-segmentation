"""
Phase 6: Cluster profiling and marketing segment naming.

For the best algorithm (default: HDBSCAN) and K-Means (for comparison),
computes per-segment un-scaled feature means, assigns a marketing-friendly
segment name via a rule-based decision tree, and produces a heatmap and a
radar/spider chart for the dissertation.

Outputs
-------
outputs/tables/segment_profiles.csv   -- mean features + name for each segment
outputs/figures/segment_profiles.png  -- heatmap (z-score coloured, raw values annotated)
outputs/figures/radar_profiles.png    -- radar chart (min-max normalised)
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CLUSTER_PARQUET,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
)

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [
    "Recency", "Frequency", "Monetary",
    "Tenure", "AvgOrderValue", "AvgInterPurchaseDays", "DistinctProducts",
]

# Canonical column names for each algorithm
_LABEL_COL: dict[str, str] = {
    "HDBSCAN":  "HDBSCAN_Cluster",
    "K-Means":  "KMeans_Cluster",
    "GMM":      "GMM_Cluster",
    "DBSCAN":   "DBSCAN_Cluster",
    "Agglomerative": "Agglomerative_Cluster",
    "Spectral":      "Spectral_Cluster",
}

PROFILE_CSV = OUTPUTS_TABLES_DIR / "segment_profiles.csv"
HEATMAP_PNG = OUTPUTS_FIGURES_DIR / "segment_profiles.png"
RADAR_PNG   = OUTPUTS_FIGURES_DIR / "radar_profiles.png"


# ---------------------------------------------------------------------------
# Rule-based segment naming
# ---------------------------------------------------------------------------

def _assign_name(row: pd.Series, overall: pd.Series) -> str:
    """Map a cluster mean vector to a marketing segment name.

    Rules are applied in order; the first matching rule wins.

    Directional conventions (lower/higher = better):
    - Recency: lower = more recent purchase = better
    - Frequency, Monetary, AvgOrderValue, DistinctProducts: higher = better
    - AvgInterPurchaseDays: lower = buys more often = better

    All thresholds are expressed as multiples of the overall customer mean so
    no absolute monetary values are hard-coded.
    """
    r   = row["Recency"]
    f   = row["Frequency"]
    m   = row["Monetary"]

    or_ = overall["Recency"]
    of  = overall["Frequency"]
    om  = overall["Monetary"]

    if r < or_ * 0.55 and f > of * 1.5 and m > om * 1.5:
        return "Champions"
    if f > of * 1.2 and m > om * 1.2:
        return "Loyal Customers"
    if m > om * 2.0:
        return "Big Spenders"
    if r > or_ * 1.5 and f > of * 0.8:
        return "At Risk"
    if r > or_ * 1.6 and f < of * 0.8:
        return "Lost Customers"
    if r < or_ * 0.8 and f < of * 0.8:
        return "Potential Loyalists"
    return "General Customers"


def _name_clusters(
    profiles: pd.DataFrame,
    overall: pd.Series,
) -> dict[int, str]:
    """Return {cluster_id: segment_name} for every cluster label."""
    names: dict[int, str] = {}
    for cid, row in profiles.iterrows():
        if cid == -1:
            names[int(cid)] = "Noise / Uncategorised"
        else:
            names[int(cid)] = _assign_name(row, overall)
    return names


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

def _build_profiles(
    cluster_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """Compute per-cluster mean feature values (un-scaled, raw units).

    Parameters
    ----------
    cluster_df:
        DataFrame with raw feature columns and a cluster-label column.
    label_col:
        Name of the cluster label column (e.g. ``"HDBSCAN_Cluster"``).

    Returns
    -------
    pd.DataFrame
        Indexed by cluster label; columns are FEATURE_COLS plus ``size``
        and ``pct_of_total``.
    """
    grp = cluster_df.groupby(label_col)[FEATURE_COLS].mean()
    grp["size"] = cluster_df.groupby(label_col).size()
    grp["pct_of_total"] = (grp["size"] / len(cluster_df) * 100).round(2)
    return grp


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_heatmap(
    profiles: pd.DataFrame,
    names: dict[int, str],
    algo: str,
) -> None:
    """Save a z-score heatmap of cluster mean features.

    Colour encodes how far each cluster's mean deviates from the cross-cluster
    mean (red = below average, green = above average).  Raw values are
    annotated in each cell for exact readability.  Noise clusters are shown
    but excluded from the z-score normalisation reference.
    """
    feat_data = profiles[FEATURE_COLS].copy()
    non_noise = feat_data[feat_data.index != -1]
    ref_mean = non_noise.mean()
    ref_std  = non_noise.std() + 1e-9
    z = (feat_data - ref_mean) / ref_std

    row_labels = [
        f"C{cid}: {names.get(int(cid), str(cid))}\n(n={int(profiles.loc[cid,'size']):,},  "
        f"{profiles.loc[cid,'pct_of_total']:.1f}%)"
        for cid in profiles.index
    ]
    col_labels = [
        "Recency", "Frequency", "Monetary",
        "Tenure", "Avg\nOrder\nValue", "Avg Inter-\nPurchase\nDays", "Distinct\nProducts",
    ]

    nrows = len(row_labels)
    fig, ax = plt.subplots(figsize=(12, max(3.5, nrows * 1.6 + 2.0)))
    im = ax.imshow(z.values, cmap="RdYlGn", aspect="auto", vmin=-2.5, vmax=2.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Z-score vs cluster mean", fontsize=9)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels, fontsize=9)

    for ri, cid in enumerate(profiles.index):
        for ci, feat in enumerate(FEATURE_COLS):
            val = profiles.loc[cid, feat]
            fmt = f"{val:,.0f}" if abs(val) >= 100 else f"{val:.1f}"
            zval = float(z.loc[cid, feat])
            txt_colour = "white" if abs(zval) > 1.8 else "black"
            ax.text(ci, ri, fmt, ha="center", va="center",
                    fontsize=8.5, color=txt_colour, fontweight="bold")

    ax.set_title(
        f"Segment Profile Heatmap — {algo}\n"
        "(colour = z-score vs cluster means;  cell text = raw un-scaled mean)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(HEATMAP_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to %s", HEATMAP_PNG)


def _plot_radar(
    profiles: pd.DataFrame,
    names: dict[int, str],
    algo: str,
    population: pd.DataFrame,
) -> None:
    """Save a radar/spider chart comparing non-noise cluster profiles.

    Each feature is expressed as the **percentile rank of the segment's mean
    within the full customer population** (0-1): an axis value of 0.8 means the
    segment's average sits above 80% of customers on that feature.  Recency is
    inverted (lower = better -> outward = better).

    ``AvgInterPurchaseDays`` is intentionally omitted from the radar: a one-time
    buyer has a value of 0 (no repeat-purchase interval), which under a
    "lower = better" inversion would wrongly rank them as having excellent
    cadence.  Frequency already captures purchase intensity without this
    ambiguity, and the raw value is still shown in the heatmap.

    Percentile rank is used instead of min-max across the cluster means because
    the latter is degenerate when there are only two non-noise segments (every
    axis is forced to exactly 0 or 1, collapsing a segment to a single spike).
    Ranking against the population is robust to the number of clusters and stays
    interpretable.
    """
    non_noise = profiles[profiles.index != -1].copy()
    if len(non_noise) < 2:
        logger.warning(
            "Only %d non-noise cluster(s) for %s; skipping radar chart.",
            len(non_noise), algo,
        )
        return

    # Exclude AvgInterPurchaseDays (see docstring: 0 for one-time buyers would
    # invert to "best cadence"). The remaining axes are cleanly directional.
    radar_cols = [c for c in FEATURE_COLS if c != "AvgInterPurchaseDays"]
    inverted = {"Recency"}  # lower = better
    norm = pd.DataFrame(index=non_noise.index, columns=radar_cols, dtype=float)
    for col in radar_cols:
        pop = population[col].to_numpy()
        n_pop = len(pop)
        for cid in non_noise.index:
            pct = float((pop <= non_noise.loc[cid, col]).sum()) / n_pop
            norm.loc[cid, col] = (1.0 - pct) if col in inverted else pct

    categories = [
        "Recency\n(inv)", "Frequency", "Monetary",
        "Tenure", "Avg\nOrder\nValue", "Distinct\nProducts",
    ]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles_closed = angles + angles[:1]

    _palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                "#8172B2", "#937860", "#DA8BC3"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (cid, row) in enumerate(norm.iterrows()):
        values = row.tolist() + [row.iloc[0]]
        colour = _palette[i % len(_palette)]
        seg_name = names.get(int(cid), str(cid))
        n_cust = int(profiles.loc[cid, "size"])
        ax.plot(angles_closed, values, "o-", linewidth=2, color=colour,
                label=f"C{cid}: {seg_name} (n={n_cust:,})")
        ax.fill(angles_closed, values, alpha=0.15, color=colour)

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#888")
    ax.set_title(
        f"Cluster Radar Chart — {algo}\n(axis = percentile vs all customers; outward = better)",
        fontsize=11, fontweight="bold", pad=22,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.18), fontsize=9)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="#ccc", linewidth=0.6)

    plt.tight_layout()
    fig.savefig(RADAR_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Radar chart saved to %s", RADAR_PNG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_clusters(
    algo: str = "HDBSCAN",
    label_col: str | None = None,
) -> pd.DataFrame:
    """Build and save cluster profiles for the selected algorithm.

    Parameters
    ----------
    algo:
        Algorithm name — used for plot titles and as the CSV index label.
        Must be one of ``"HDBSCAN"``, ``"K-Means"``, ``"GMM"``, ``"DBSCAN"``.
    label_col:
        Override for the cluster label column name.  Inferred from ``algo``
        if ``None`` (e.g. ``"HDBSCAN"`` -> ``"HDBSCAN_Cluster"``).

    Returns
    -------
    pd.DataFrame
        Profile table: cluster means + size + pct_of_total + segment_name.
    """
    if label_col is None:
        label_col = _LABEL_COL.get(algo)
        if label_col is None:
            raise ValueError(
                f"Unknown algo '{algo}'. Pass label_col explicitly or use one of "
                f"{list(_LABEL_COL)}."
            )

    logger.info("Loading cluster data from %s", CLUSTER_PARQUET)
    cluster_df = pd.read_parquet(CLUSTER_PARQUET)

    if label_col not in cluster_df.columns:
        raise ValueError(
            f"Column '{label_col}' not found. Available: {list(cluster_df.columns)}"
        )

    profiles = _build_profiles(cluster_df, label_col)
    overall  = cluster_df[FEATURE_COLS].mean()
    names    = _name_clusters(profiles, overall)
    profiles["segment_name"] = [names[int(cid)] for cid in profiles.index]

    logger.info("Segment profiles (%s):", algo)
    for cid, row in profiles.iterrows():
        logger.info(
            "  C%s [%s] n=%d (%.1f%%) | R=%.0f F=%.1f M=%.0f",
            cid, row["segment_name"], int(row["size"]),
            row["pct_of_total"], row["Recency"],
            row["Frequency"], row["Monetary"],
        )

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(PROFILE_CSV)
    logger.info("Segment profiles saved to %s", PROFILE_CSV)

    _plot_heatmap(profiles, names, algo)
    _plot_radar(profiles, names, algo, population=cluster_df[FEATURE_COLS])

    return profiles
