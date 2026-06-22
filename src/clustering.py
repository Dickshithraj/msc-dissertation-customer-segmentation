"""
Clustering module: K-Means, DBSCAN, GMM, and HDBSCAN on pre-scaled features.

Four algorithms are benchmarked so that the dissertation can argue for the
chosen solution on the basis of a systematic comparison rather than a single
arbitrary choice.  Each algorithm makes different assumptions about cluster
geometry and density, which is documented in its function docstring.

K-Means
    Assumes spherical, equally-sized clusters.  Chosen as the standard RFM
    baseline; interpretable centroids map directly to marketing personas.
    Tuned via elbow (inertia) + silhouette + Davies-Bouldin sweep over k=2..10.

DBSCAN
    Density-based; discovers arbitrary shapes; labels outliers as noise (-1).
    No k required, but sensitive to eps (neighbourhood radius).  eps is chosen
    objectively from the "knee" of the sorted k-distance graph (k = min_samples).

Gaussian Mixture Model (GMM)
    Probabilistic; assigns soft membership probabilities; allows elliptical
    clusters.  More flexible than K-Means for non-spherical RFM distributions.
    Number of components chosen by minimising BIC over n=2..10.

HDBSCAN
    Hierarchical density-based; handles variable-density clusters; labels
    low-density points as noise.  No parameters need tuning for small n.

PCA projection
--------------
All four label sets are visualised in a 2x2 PCA scatter grid (PC1/PC2).
The PCA is fitted once on the full scaled space and reused for all panels.
"""

from __future__ import annotations

import logging

import hdbscan as hdbscan_lib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from src.config import (
    CLUSTER_PARQUET,
    CUSTOMER_FEATURES_PARQUET,
    DATA_PROCESSED_DIR,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    GMM_BEST_N,
    GMM_N_MAX,
    GMM_N_MIN,
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
from src.preprocessing import FEATURE_COLS

logger = logging.getLogger(__name__)

KMEANS_METRICS_CSV   = OUTPUTS_TABLES_DIR / "kmeans_metrics.csv"
GMM_BIC_CSV          = OUTPUTS_TABLES_DIR / "gmm_bic.csv"
KMEANS_SELECTION_PNG = OUTPUTS_FIGURES_DIR / "kmeans_selection.png"
DBSCAN_KDIST_PNG     = OUTPUTS_FIGURES_DIR / "dbscan_kdistance.png"
GMM_BIC_PNG          = OUTPUTS_FIGURES_DIR / "gmm_bic.png"
CLUSTER_PCA_PNG      = OUTPUTS_FIGURES_DIR / "cluster_pca_projection.png"

_CLUSTER_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F",
]
_NOISE_COLOUR = "#CCCCCC"


# ===========================================================================
# K-Means
# ===========================================================================

def _kmeans_sweep(X: np.ndarray) -> pd.DataFrame:
    """Fit K-Means for every k in [KMEANS_K_MIN, KMEANS_K_MAX] and record metrics.

    Three metrics collected per k:
    * inertia (WCSS) — elbow plot
    * silhouette — normalised inter/intra cluster ratio (higher = better)
    * davies_bouldin — scatter-to-separation ratio (lower = better)
    """
    rows: list[dict] = []
    logger.info("K-Means sweep: k = %d to %d", KMEANS_K_MIN, KMEANS_K_MAX)
    for k in range(KMEANS_K_MIN, KMEANS_K_MAX + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20,
                    random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(3000, len(X)),
                               random_state=RANDOM_STATE)
        db = davies_bouldin_score(X, labels)
        rows.append({"k": k, "inertia": round(km.inertia_, 2),
                     "silhouette": round(sil, 4), "davies_bouldin": round(db, 4)})
        logger.info("  k=%2d | inertia=%10.1f | silhouette=%.4f | DB=%.4f",
                    k, km.inertia_, sil, db)
    return pd.DataFrame(rows)


def _pick_best_k(metrics: pd.DataFrame) -> int:
    """Return k with highest silhouette, or the KMEANS_BEST_K config override."""
    if KMEANS_BEST_K is not None:
        logger.info("Using manually configured k = %d (KMEANS_BEST_K).", KMEANS_BEST_K)
        return KMEANS_BEST_K
    best = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    logger.info("Auto-selected k = %d (silhouette = %.4f).", best,
                metrics["silhouette"].max())
    return best


def _plot_kmeans_selection(metrics: pd.DataFrame, best_k: int) -> None:
    """Three-panel figure: inertia (elbow), silhouette, Davies-Bouldin vs k."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("K-Means model selection", fontsize=12, fontweight="bold", y=1.01)
    panels = [
        ("inertia",        "Inertia (WCSS)",      "steelblue",   False),
        ("silhouette",     "Silhouette score",     "darkorange",  True),
        ("davies_bouldin", "Davies-Bouldin index", "forestgreen", False),
    ]
    for ax, (col, ylabel, colour, higher) in zip(axes, panels):
        ax.plot(metrics["k"], metrics[col], marker="o", color=colour,
                linewidth=2, markersize=6)
        ax.axvline(best_k, color="red", linestyle="--", linewidth=1.5,
                   label=f"k={best_k} chosen")
        ax.set_xlabel("k", fontsize=10)
        ax.set_title(f"{ylabel}\n({'higher' if higher else 'lower'} = better)",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(metrics["k"])
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(KMEANS_SELECTION_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("K-Means selection plot saved to %s", KMEANS_SELECTION_PNG)


def _fit_final_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    """Fit final K-Means (n_init=50) and return (labels, fitted model)."""
    logger.info("Fitting final K-Means with k=%d (n_init=50).", k)
    km = KMeans(n_clusters=k, init="k-means++", n_init=50,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    logger.info("Final K-Means silhouette = %.4f", silhouette_score(X, labels))
    return labels, km


# ===========================================================================
# DBSCAN
# ===========================================================================

def _find_knee(distances: np.ndarray) -> float:
    """Return the knee point of a sorted distance array using perpendicular distance.

    The knee is found geometrically: for each point on the sorted-distance
    curve, compute its perpendicular distance from the straight line connecting
    the first and last points.  The index with the maximum perpendicular
    distance is the knee — the point where the curve bends most sharply from
    flat (core region) to steep (outlier region).

    This is equivalent to the standard Kneedle algorithm (Satopaa et al., 2011)
    without requiring an external dependency.

    Parameters
    ----------
    distances:
        1-D array of sorted k-th-nearest-neighbour distances.
    """
    n = len(distances)
    x = np.linspace(0, 1, n)
    y = (distances - distances.min()) / (distances.max() - distances.min() + 1e-12)
    # Direction vector of the line from (x[0],y[0]) to (x[-1],y[-1])
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    # Perpendicular distances from each point to the line
    perp = np.abs(dx * (y - y[0]) - dy * (x - x[0])) / norm
    return float(distances[np.argmax(perp)])


def _kdistance_plot(X: np.ndarray, k: int, eps: float) -> None:
    """Save the sorted k-distance plot used to choose DBSCAN eps.

    The k-distance graph (Ester et al., 1996) plots, for each point, the
    distance to its k-th nearest neighbour, sorted in ascending order.  The
    point where the curve transitions sharply from flat to steep marks the
    natural neighbourhood radius (eps).  Points beyond this threshold are
    outliers; a cluster must contain at least ``min_samples`` core points
    within distance eps.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    k:
        Neighbourhood size; set to ``DBSCAN_MIN_SAMPLES`` so the plot
        directly reflects the algorithm's core-point criterion.
    eps:
        Auto-detected knee value, marked with a horizontal red line.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nbrs.kneighbors(X)
    kd = np.sort(dists[:, -1])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(kd, color="steelblue", linewidth=1.2)
    ax.axhline(eps, color="red", linestyle="--", linewidth=1.5,
               label=f"Knee / eps = {eps:.3f}")
    ax.set_xlabel("Points sorted by distance", fontsize=10)
    ax.set_ylabel(f"{k}-th nearest neighbour distance", fontsize=10)
    ax.set_title(f"DBSCAN k-distance plot  (k={k})\n"
                 f"Knee detection selects eps = {eps:.3f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(DBSCAN_KDIST_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("DBSCAN k-distance plot saved to %s", DBSCAN_KDIST_PNG)


def _fit_dbscan(X: np.ndarray) -> tuple[np.ndarray, DBSCAN, float]:
    """Fit DBSCAN, auto-selecting eps from the k-distance knee if not configured.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise,
    Ester et al., 1996) groups points that are density-reachable within
    radius eps, requiring at least min_samples core points per cluster.

    Assumptions and suitability for RFM data
    -----------------------------------------
    * DBSCAN does not assume spherical clusters, making it appropriate for
      the irregular shapes that emerge in RFM space where one dimension
      (Monetary) is heavily right-skewed even after log-compression.
    * It automatically identifies noise points (label -1), which correspond
      to customers with unusual purchase behaviour that does not fit any
      coherent segment.
    * The main limitation is sensitivity to eps: a value too small produces
      many noise points; too large merges distinct segments.  The k-distance
      knee provides an objective, data-driven eps choice.

    Parameters
    ----------
    X:
        Scaled feature matrix.

    Returns
    -------
    tuple
        (labels, fitted DBSCAN, eps_used)
    """
    k = DBSCAN_MIN_SAMPLES
    if DBSCAN_EPS is not None:
        eps = DBSCAN_EPS
        logger.info("Using configured DBSCAN eps = %.4f.", eps)
    else:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nbrs.kneighbors(X)
        kd = np.sort(dists[:, -1])
        eps = _find_knee(kd)
        logger.info("DBSCAN auto eps from k-distance knee = %.4f.", eps)

    _kdistance_plot(X, k, eps)

    db = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean")
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("DBSCAN: %d clusters, %d noise points (%.1f%%), eps=%.4f.",
                n_clusters, n_noise, n_noise / len(labels) * 100, eps)
    return labels, db, eps


# ===========================================================================
# Gaussian Mixture Model
# ===========================================================================

def _gmm_bic_sweep(X: np.ndarray) -> pd.DataFrame:
    """Fit GMM for n=GMM_N_MIN..GMM_N_MAX and return BIC + AIC per component count.

    Gaussian Mixture Models (Dempster et al., 1977) represent the data as a
    weighted sum of multivariate Gaussian distributions.  Unlike K-Means, GMM:
    * allows elliptical cluster shapes via the full covariance matrix
    * assigns soft probabilities (posterior responsibilities) rather than hard
      labels, capturing uncertainty in borderline customers
    * is penalised for complexity via BIC/AIC, providing a principled criterion
      for choosing the number of components

    BIC (Bayesian Information Criterion) is preferred over AIC here because it
    applies a stronger penalty for model complexity, guarding against over-
    fitting the number of segments to the training sample.  The component count
    with the lowest BIC is selected automatically.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    """
    rows = []
    logger.info("GMM BIC sweep: n = %d to %d", GMM_N_MIN, GMM_N_MAX)
    for n in range(GMM_N_MIN, GMM_N_MAX + 1):
        gmm = GaussianMixture(n_components=n, covariance_type="full",
                              random_state=RANDOM_STATE, n_init=5, max_iter=300)
        gmm.fit(X)
        rows.append({"n": n, "bic": round(gmm.bic(X), 2),
                     "aic": round(gmm.aic(X), 2)})
        logger.info("  n=%2d | BIC=%10.1f | AIC=%10.1f", n, gmm.bic(X), gmm.aic(X))
    return pd.DataFrame(rows)


def _pick_best_n(bic_df: pd.DataFrame) -> int:
    """Return n with lowest BIC, or GMM_BEST_N config override."""
    if GMM_BEST_N is not None:
        logger.info("Using manually configured GMM n = %d.", GMM_BEST_N)
        return GMM_BEST_N
    best = int(bic_df.loc[bic_df["bic"].idxmin(), "n"])
    logger.info("Auto-selected GMM n = %d (BIC = %.1f).", best,
                bic_df["bic"].min())
    return best


def _plot_gmm_bic(bic_df: pd.DataFrame, best_n: int) -> None:
    """Save BIC and AIC curves vs number of GMM components."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bic_df["n"], bic_df["bic"], marker="o", color="steelblue",
            linewidth=2, label="BIC (used for selection)")
    ax.plot(bic_df["n"], bic_df["aic"], marker="s", color="darkorange",
            linewidth=2, linestyle="--", label="AIC")
    ax.axvline(best_n, color="red", linestyle="--", linewidth=1.5,
               label=f"n={best_n} chosen (min BIC)")
    ax.set_xlabel("Number of GMM components", fontsize=10)
    ax.set_ylabel("Information criterion (lower = better)", fontsize=10)
    ax.set_title("GMM model selection via BIC / AIC", fontsize=11,
                 fontweight="bold")
    ax.set_xticks(bic_df["n"])
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GMM_BIC_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("GMM BIC plot saved to %s", GMM_BIC_PNG)


def _fit_final_gmm(X: np.ndarray, n: int) -> tuple[np.ndarray, GaussianMixture]:
    """Fit final GMM with chosen n and return (hard labels, fitted model).

    Hard labels are the argmax of the posterior responsibility matrix so that
    downstream profiling and validation work with integer segment IDs.  The
    full soft probabilities are available via ``model.predict_proba(X)``.
    """
    logger.info("Fitting final GMM with n=%d (n_init=10).", n)
    gmm = GaussianMixture(n_components=n, covariance_type="full",
                          random_state=RANDOM_STATE, n_init=10, max_iter=500)
    gmm.fit(X)
    labels = gmm.predict(X)
    sil = silhouette_score(X, labels, sample_size=min(3000, len(X)),
                           random_state=RANDOM_STATE)
    logger.info("Final GMM silhouette = %.4f.", sil)
    return labels, gmm


# ===========================================================================
# HDBSCAN
# ===========================================================================

def _fit_hdbscan(X: np.ndarray) -> tuple[np.ndarray, object]:
    """Fit HDBSCAN and return (labels, clusterer). Label -1 = noise.

    HDBSCAN (Campello et al., 2013) extends DBSCAN by building a cluster
    hierarchy and extracting stable flat clusters at the level of maximum
    persistence.  Key advantages over DBSCAN for RFM data:

    * No eps parameter — stability is assessed across all density thresholds.
    * Handles clusters of varying density, which is common in RFM space where
      high-value customers are sparse but form tight groups.
    * ``cluster_selection_method="eom"`` (Excess of Mass) produces fewer,
      more robust clusters than the leaf method.

    Noise points (-1) are retained; they typically correspond to customers
    with highly irregular purchase histories unsuitable for targeted campaigns.
    """
    logger.info("Fitting HDBSCAN (min_cluster_size=%d, min_samples=%d).",
                HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES)
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("HDBSCAN: %d clusters, %d noise points (%.1f%%).",
                n_clusters, n_noise, n_noise / len(labels) * 100)
    return labels, clusterer


# ===========================================================================
# PCA visualisation (2x2 grid — all four algorithms)
# ===========================================================================

def _plot_pca_all(
    X: np.ndarray,
    all_labels: dict[str, np.ndarray],
) -> None:
    """Save a 2x2 PCA scatter grid showing all four algorithm partitions.

    One PCA is fitted on the full scaled space and shared across all panels,
    so differences between panels reflect genuine algorithmic variation rather
    than projection differences.  Explained variance is annotated on the axis
    labels.  Noise points (label -1) are drawn first in light grey so genuine
    cluster points render on top.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    all_labels:
        Dict mapping algorithm name to integer label array.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    logger.info("PCA: PC1=%.1f%%, PC2=%.1f%% explained.", ev[0]*100, ev[1]*100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"All-algorithm PCA projection\n"
        f"PC1 {ev[0]*100:.1f}% + PC2 {ev[1]*100:.1f}% = "
        f"{(ev[0]+ev[1])*100:.1f}% variance",
        fontsize=12, fontweight="bold",
    )

    for ax, (algo, labels) in zip(axes.flat, all_labels.items()):
        unique = sorted(np.unique(labels))
        n_real = len([c for c in unique if c != -1])
        ax.set_title(f"{algo}  ({n_real} clusters)", fontsize=11,
                     fontweight="bold")

        if -1 in unique:
            mask = labels == -1
            ax.scatter(coords[mask, 0], coords[mask, 1], c=_NOISE_COLOUR,
                       s=6, alpha=0.3, label=f"Noise (n={mask.sum():,})")

        cidx = 0
        for cid in unique:
            if cid == -1:
                continue
            mask = labels == cid
            colour = _CLUSTER_PALETTE[cidx % len(_CLUSTER_PALETTE)]
            ax.scatter(coords[mask, 0], coords[mask, 1], c=colour,
                       s=8, alpha=0.5, label=f"C{cid} (n={mask.sum():,})")
            cidx += 1

        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
        ax.legend(fontsize=7, markerscale=2, framealpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CLUSTER_PCA_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PCA 2x2 projection saved to %s", CLUSTER_PCA_PNG)


# ===========================================================================
# Public API
# ===========================================================================

def run_all_clustering(X: np.ndarray | None = None) -> dict:
    """Run K-Means, DBSCAN, GMM, and HDBSCAN and save all results.

    Loads scaled features from ``data/processed/scaled_features.parquet`` if
    ``X`` is not supplied.  All four label arrays are merged with the raw
    customer features and saved to ``data/processed/clustered_customers.parquet``.

    Parameters
    ----------
    X:
        Optional pre-loaded scaled feature matrix.

    Returns
    -------
    dict
        Keys: ``"kmeans"``, ``"dbscan"``, ``"gmm"``, ``"hdbscan"``.
        Each value is a dict with ``"labels"`` (np.ndarray) and ``"model"``.
        Also includes ``"customer_ids"`` and ``"cluster_df"``.

    Side effects
    ------------
    * ``outputs/tables/kmeans_metrics.csv``
    * ``outputs/tables/gmm_bic.csv``
    * ``outputs/figures/kmeans_selection.png``
    * ``outputs/figures/dbscan_kdistance.png``
    * ``outputs/figures/gmm_bic.png``
    * ``outputs/figures/cluster_pca_projection.png``
    * ``data/processed/clustered_customers.parquet``
    """
    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading scaled features from %s", SCALED_FEATURES_PARQUET)
    scaled_df = pd.read_parquet(SCALED_FEATURES_PARQUET)
    customer_ids = scaled_df["Customer ID"].values
    if X is None:
        X = scaled_df[FEATURE_COLS].values
    logger.info("Clustering %d customers on %d features.", *X.shape)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ── K-Means ────────────────────────────────────────────────────────────
    km_metrics = _kmeans_sweep(X)
    km_metrics.to_csv(KMEANS_METRICS_CSV, index=False)
    best_k = _pick_best_k(km_metrics)
    _plot_kmeans_selection(km_metrics, best_k)
    km_labels, km_model = _fit_final_kmeans(X, best_k)

    # ── DBSCAN ────────────────────────────────────────────────────────────
    db_labels, db_model, db_eps = _fit_dbscan(X)

    # ── GMM ───────────────────────────────────────────────────────────────
    bic_df = _gmm_bic_sweep(X)
    bic_df.to_csv(GMM_BIC_CSV, index=False)
    best_n = _pick_best_n(bic_df)
    _plot_gmm_bic(bic_df, best_n)
    gmm_labels, gmm_model = _fit_final_gmm(X, best_n)

    # ── HDBSCAN ───────────────────────────────────────────────────────────
    hdb_labels, hdb_model = _fit_hdbscan(X)

    # ── PCA (2x2) ─────────────────────────────────────────────────────────
    all_labels = {
        "K-Means":  km_labels,
        "DBSCAN":   db_labels,
        "GMM":      gmm_labels,
        "HDBSCAN":  hdb_labels,
    }
    _plot_pca_all(X, all_labels)

    # ── Persist cluster assignments ────────────────────────────────────────
    cluster_df = pd.DataFrame({
        "Customer ID":    customer_ids,
        "KMeans_Cluster": km_labels,
        "DBSCAN_Cluster": db_labels,
        "GMM_Cluster":    gmm_labels,
        "HDBSCAN_Cluster": hdb_labels,
    })
    features_df = pd.read_parquet(CUSTOMER_FEATURES_PARQUET)
    cluster_df = cluster_df.merge(features_df, on="Customer ID", how="left")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cluster_df.to_parquet(CLUSTER_PARQUET, index=False)
    logger.info("Cluster assignments saved to %s", CLUSTER_PARQUET)

    # ── Summary log ────────────────────────────────────────────────────────
    for algo, lbl in all_labels.items():
        sizes = pd.Series(lbl).value_counts().sort_index()
        parts = [f"C{c}:{n}" for c, n in sizes.items()]
        logger.info("%s: %s", algo, "  ".join(parts))

    return {
        "kmeans":       {"labels": km_labels,  "model": km_model},
        "dbscan":       {"labels": db_labels,  "model": db_model, "eps": db_eps},
        "gmm":          {"labels": gmm_labels, "model": gmm_model},
        "hdbscan":      {"labels": hdb_labels, "model": hdb_model},
        "customer_ids": customer_ids,
        "cluster_df":   cluster_df,
    }


# Backward-compatible alias kept so any cached imports still work.
def run_clustering(*args, **kwargs) -> pd.DataFrame:
    """Alias for run_all_clustering(); returns cluster_df for compatibility."""
    result = run_all_clustering(*args, **kwargs)
    return result["cluster_df"]
