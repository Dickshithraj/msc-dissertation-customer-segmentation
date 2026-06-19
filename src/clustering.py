"""
Clustering module: K-Means sweep + HDBSCAN on pre-scaled customer features.

Algorithm choice rationale
--------------------------
Two complementary algorithms are applied and compared:

K-Means
    Partitions all customers into exactly *k* spherical, equal-variance
    clusters by minimising total within-cluster squared Euclidean distance.
    It is the most widely used baseline in RFM segmentation literature, is
    computationally efficient at n ≈ 6 000, and produces hard, mutually
    exclusive labels that map cleanly onto distinct marketing personas.
    The optimal k is chosen objectively using two criteria measured across a
    sweep of k = 2 … 10:

    * **Elbow / inertia** — the rate at which adding another cluster reduces
      total within-cluster variance.  The "elbow" marks the point of
      diminishing returns.
    * **Silhouette score** — measures how much closer each point is to its
      own cluster centroid than to the nearest rival centroid (range −1 to 1;
      higher = better separation).  Unlike the elbow, silhouette gives a
      single scalar that can be compared objectively across k values.
    * **Davies-Bouldin index** — ratio of within-cluster scatter to between-
      cluster distance (lower = better).  Provides a third independent view
      that helps resolve ties between silhouette-equivalent k values.

    The k with the highest silhouette score is selected automatically;
    ``config.KMEANS_BEST_K`` can override this for manual control.

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
         with Noise)
    Discovers clusters of arbitrary shape and automatically marks low-density
    points as noise (label −1).  Unlike K-Means it does not require k to be
    specified in advance, making it well-suited for exploratory analysis
    where the true number of segments is unknown.  Comparing HDBSCAN output
    to K-Means helps validate whether the K-Means partition reflects genuine
    density structure or is an artefact of the spherical assumption.

    Noise points (label −1) are retained in the output table with their
    original label rather than being reassigned, as they often represent a
    strategically interesting segment of infrequent, low-spend customers
    that any targeted campaign should handle separately.

PCA projection
--------------
Both sets of labels are visualised in the first two principal components of
the scaled feature space.  The PCA is computed purely for visualisation; it
is not used as input to either clustering algorithm.
"""

from __future__ import annotations

import logging

import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import (
    CLUSTER_PARQUET,
    CUSTOMER_FEATURES_PARQUET,
    DATA_PROCESSED_DIR,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    KMEANS_BEST_K,
    KMEANS_K_MAX,
    KMEANS_K_MIN,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    RANDOM_STATE,
    SCALED_FEATURES_PARQUET,
)
from src.preprocessing import FEATURE_COLS, PreprocessingResult

logger = logging.getLogger(__name__)

KMEANS_METRICS_CSV = OUTPUTS_TABLES_DIR / "kmeans_metrics.csv"
KMEANS_SELECTION_PNG = OUTPUTS_FIGURES_DIR / "kmeans_selection.png"
CLUSTER_PCA_PNG = OUTPUTS_FIGURES_DIR / "cluster_pca_projection.png"

# Palette used for cluster scatter plots (up to 12 clusters + noise).
_CLUSTER_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F",
]
_NOISE_COLOUR = "#CCCCCC"


# ---------------------------------------------------------------------------
# K-Means sweep
# ---------------------------------------------------------------------------

def _kmeans_sweep(X: np.ndarray) -> pd.DataFrame:
    """Fit K-Means for every k in [KMEANS_K_MIN, KMEANS_K_MAX] and record metrics.

    Three metrics are collected at each k so the dissertation can triangulate
    the optimal number of clusters without relying on a single criterion:

    * ``inertia`` (within-cluster sum of squares) for the elbow plot.
    * ``silhouette`` (mean inter- vs intra-cluster distance ratio).
    * ``davies_bouldin`` (mean ratio of scatter to separation; lower = better).

    Parameters
    ----------
    X:
        Scaled feature matrix of shape ``(n_customers, n_features)``.

    Returns
    -------
    pd.DataFrame
        One row per k with columns k, inertia, silhouette, davies_bouldin.
    """
    rows: list[dict] = []
    k_range = range(KMEANS_K_MIN, KMEANS_K_MAX + 1)
    logger.info("K-Means sweep: k = %d … %d", KMEANS_K_MIN, KMEANS_K_MAX)

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=20,
                    random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(3000, len(X)),
                               random_state=RANDOM_STATE)
        db = davies_bouldin_score(X, labels)
        rows.append({
            "k": k,
            "inertia": round(km.inertia_, 2),
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
        })
        logger.info("  k=%2d | inertia=%10.1f | silhouette=%.4f | DB=%.4f",
                    k, km.inertia_, sil, db)

    return pd.DataFrame(rows)


def _pick_best_k(metrics: pd.DataFrame) -> int:
    """Return the k with the highest silhouette score, or the config override.

    Silhouette is preferred over inertia for automatic selection because it is
    normalised (−1 to 1) and does not merely reward adding more clusters.
    The config constant ``KMEANS_BEST_K`` allows a researcher to override the
    automatic choice after inspecting the elbow plot.

    Parameters
    ----------
    metrics:
        DataFrame produced by :func:`_kmeans_sweep`.
    """
    if KMEANS_BEST_K is not None:
        logger.info("Using manually configured k = %d (KMEANS_BEST_K).", KMEANS_BEST_K)
        return KMEANS_BEST_K
    best = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    logger.info("Auto-selected k = %d (highest silhouette = %.4f).",
                best, metrics["silhouette"].max())
    return best


def _plot_kmeans_selection(metrics: pd.DataFrame, best_k: int) -> None:
    """Save a three-panel figure for the K-Means model-selection step.

    Panels (left to right):
    1. Elbow curve — inertia vs k.
    2. Silhouette score vs k (higher = better).
    3. Davies-Bouldin index vs k (lower = better).

    A vertical dashed red line marks the chosen k so the figure is
    self-documenting when included in the dissertation appendix.

    Parameters
    ----------
    metrics:
        DataFrame from :func:`_kmeans_sweep`.
    best_k:
        The chosen number of clusters (marked with a vertical line).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("K-Means model selection metrics", fontsize=12,
                 fontweight="bold", y=1.01)

    panels = [
        ("inertia",        "Inertia (WCSS)",         "steelblue",  False),
        ("silhouette",     "Silhouette score",        "darkorange", True),
        ("davies_bouldin", "Davies-Bouldin index",    "forestgreen", False),
    ]
    for ax, (col, ylabel, colour, higher_better) in zip(axes, panels):
        ax.plot(metrics["k"], metrics[col], marker="o", color=colour,
                linewidth=2, markersize=6)
        ax.axvline(best_k, color="red", linestyle="--", linewidth=1.5,
                   label=f"k = {best_k} (chosen)")
        ax.set_xlabel("Number of clusters (k)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.set_xticks(metrics["k"])
        direction = "↑ better" if higher_better else "↓ better"
        ax.set_title(f"{ylabel}\n({direction})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(KMEANS_SELECTION_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("K-Means selection plot saved to %s", KMEANS_SELECTION_PNG)


def _fit_final_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """Fit K-Means with the chosen k and return integer cluster labels.

    Uses ``n_init=50`` for the final fit (more restarts than the sweep) to
    reduce sensitivity to centroid initialisation, at the cost of extra
    compute.  ``k-means++`` initialisation further reduces the risk of a
    degenerate local minimum.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    k:
        Number of clusters chosen by :func:`_pick_best_k`.
    """
    logger.info("Fitting final K-Means with k = %d (n_init=50).", k)
    km = KMeans(n_clusters=k, init="k-means++", n_init=50,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    logger.info("Final K-Means silhouette = %.4f", sil)
    return labels


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def _fit_hdbscan(X: np.ndarray) -> np.ndarray:
    """Fit HDBSCAN and return cluster labels (−1 = noise / unclustered).

    ``min_cluster_size`` is the primary parameter: it sets the smallest group
    the algorithm will consider a genuine cluster.  Set to 50 in config,
    which means any density peak containing fewer than 50 customers is treated
    as noise.  ``min_samples`` controls how conservative the noise labelling
    is — higher values produce more noise points but more robust core clusters.

    The noise label (−1) is **not** reassigned to the nearest cluster.
    Noise points in HDBSCAN indicate customers who do not belong to any
    coherent density group and are analytically meaningful in their own right
    (often the least active or most erratic buyers).

    Parameters
    ----------
    X:
        Scaled feature matrix.

    Returns
    -------
    np.ndarray
        Integer array of cluster labels; −1 denotes noise.
    """
    logger.info(
        "Fitting HDBSCAN (min_cluster_size=%d, min_samples=%d).",
        HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",  # Excess of Mass — more stable than leaf
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info(
        "HDBSCAN: %d clusters found, %d noise points (%.1f%%).",
        n_clusters, n_noise, n_noise / len(labels) * 100,
    )
    return labels


# ---------------------------------------------------------------------------
# PCA visualisation
# ---------------------------------------------------------------------------

def _plot_pca_clusters(
    X: np.ndarray,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
) -> None:
    """Save a side-by-side PCA scatter coloured by K-Means and HDBSCAN labels.

    The PCA is fitted on the full 7-dimensional scaled space and projected
    onto the first two principal components for visualisation only.  The
    explained variance ratio of PC1 and PC2 is annotated on the axes so the
    reader can assess how well the 2D projection captures the clustering
    structure.

    Noise points (HDBSCAN label −1) are rendered in light grey and drawn
    first so genuine cluster points appear on top.

    Parameters
    ----------
    X:
        Scaled feature matrix of shape ``(n_customers, n_features)``.
    kmeans_labels:
        Integer cluster labels from K-Means (0-indexed, no noise).
    hdbscan_labels:
        Integer cluster labels from HDBSCAN (−1 = noise).
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    logger.info("PCA: PC1=%.1f%%, PC2=%.1f%% variance explained.",
                ev[0] * 100, ev[1] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Cluster assignments — PCA projection\n"
        f"(PC1 {ev[0]*100:.1f}% + PC2 {ev[1]*100:.1f}% = "
        f"{(ev[0]+ev[1])*100:.1f}% variance explained)",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: K-Means ───────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title(f"K-Means (k = {len(np.unique(kmeans_labels))})",
                 fontsize=11, fontweight="bold")
    for cid in sorted(np.unique(kmeans_labels)):
        mask = kmeans_labels == cid
        colour = _CLUSTER_PALETTE[cid % len(_CLUSTER_PALETTE)]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, s=8, alpha=0.5, label=f"Cluster {cid} (n={mask.sum():,})")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
    ax.legend(fontsize=8, markerscale=2, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: HDBSCAN ──────────────────────────────────────────────────
    ax = axes[1]
    unique_hdb = sorted(np.unique(hdbscan_labels))
    n_real = len([c for c in unique_hdb if c != -1])
    ax.set_title(f"HDBSCAN ({n_real} clusters)", fontsize=11, fontweight="bold")

    # Draw noise first so cluster points render on top.
    if -1 in unique_hdb:
        mask = hdbscan_labels == -1
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=_NOISE_COLOUR, s=6, alpha=0.3, label=f"Noise (n={mask.sum():,})")

    colour_idx = 0
    for cid in unique_hdb:
        if cid == -1:
            continue
        mask = hdbscan_labels == cid
        colour = _CLUSTER_PALETTE[colour_idx % len(_CLUSTER_PALETTE)]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, s=8, alpha=0.5, label=f"Cluster {cid} (n={mask.sum():,})")
        colour_idx += 1

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
    ax.legend(fontsize=8, markerscale=2, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CLUSTER_PCA_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PCA cluster projection saved to %s", CLUSTER_PCA_PNG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_clustering(
    preprocessing_result: PreprocessingResult | None = None,
) -> pd.DataFrame:
    """Run K-Means sweep + HDBSCAN and save cluster assignments.

    If ``preprocessing_result`` is not supplied the function loads the
    pre-computed scaled features from ``data/processed/scaled_features.parquet``
    so Stage 3 can be called independently of Stage 2b.

    Parameters
    ----------
    preprocessing_result:
        Optional output of :func:`src.preprocessing.preprocess_features`.
        When provided, the scaler and ``log1p_cols`` are available for
        later inverse-transforming centroids.  Pass ``None`` to load from
        the saved parquet.

    Returns
    -------
    pd.DataFrame
        Customer-level cluster assignment table with columns:

        * ``Customer ID``
        * ``KMeans_Cluster``  — integer 0 … k−1
        * ``HDBSCAN_Cluster`` — integer ≥ 0, or −1 for noise

        Saved to ``data/processed/clustered_customers.parquet``.

    Side effects
    ------------
    * Writes ``outputs/tables/kmeans_metrics.csv``
    * Writes ``outputs/figures/kmeans_selection.png``
    * Writes ``outputs/figures/cluster_pca_projection.png``
    * Writes ``data/processed/clustered_customers.parquet``
    """
    # ── Load data ──────────────────────────────────────────────────────────
    if preprocessing_result is not None:
        X = preprocessing_result.X_scaled
        scaled_df = pd.read_parquet(SCALED_FEATURES_PARQUET)
        customer_ids = scaled_df["Customer ID"].values
    else:
        logger.info("Loading scaled features from %s", SCALED_FEATURES_PARQUET)
        scaled_df = pd.read_parquet(SCALED_FEATURES_PARQUET)
        customer_ids = scaled_df["Customer ID"].values
        X = scaled_df[FEATURE_COLS].values

    logger.info("Clustering %d customers on %d features.", *X.shape)

    # ── K-Means sweep ──────────────────────────────────────────────────────
    metrics = _kmeans_sweep(X)
    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(KMEANS_METRICS_CSV, index=False)
    logger.info("K-Means metrics saved to %s", KMEANS_METRICS_CSV)

    best_k = _pick_best_k(metrics)
    _plot_kmeans_selection(metrics, best_k)

    kmeans_labels = _fit_final_kmeans(X, best_k)

    # ── HDBSCAN ────────────────────────────────────────────────────────────
    hdbscan_labels = _fit_hdbscan(X)

    # ── PCA visualisation ──────────────────────────────────────────────────
    _plot_pca_clusters(X, kmeans_labels, hdbscan_labels)

    # ── Persist cluster assignments ────────────────────────────────────────
    cluster_df = pd.DataFrame({
        "Customer ID": customer_ids,
        "KMeans_Cluster": kmeans_labels,
        "HDBSCAN_Cluster": hdbscan_labels,
    })

    # Merge in raw features so the table is self-contained for profiling.
    features_df = pd.read_parquet(CUSTOMER_FEATURES_PARQUET)
    cluster_df = cluster_df.merge(features_df, on="Customer ID", how="left")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cluster_df.to_parquet(CLUSTER_PARQUET, index=False)
    logger.info("Cluster assignments saved to %s", CLUSTER_PARQUET)

    # ── Summary log ────────────────────────────────────────────────────────
    logger.info("\nK-Means cluster sizes:")
    for cid, count in pd.Series(kmeans_labels).value_counts().sort_index().items():
        logger.info("  Cluster %d: %d customers", cid, count)

    logger.info("\nHDBSCAN cluster sizes (incl. noise = −1):")
    for cid, count in pd.Series(hdbscan_labels).value_counts().sort_index().items():
        logger.info("  Cluster %d: %d customers", cid, count)

    return cluster_df
