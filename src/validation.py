"""
Cluster validation and stability analysis.

Internal validity metrics
-------------------------
Three complementary internal metrics are computed for each algorithm.
They are called *internal* because they require only the data and labels —
no external ground truth — making them appropriate here where true customer
segments are unknown:

Silhouette Coefficient (Rousseeuw, 1987)
    For each point i, measures how similar it is to its own cluster (a)
    versus the nearest rival cluster (b): s(i) = (b-a) / max(a,b).
    The mean over all points lies in [-1, 1]; higher is better.
    It is the most intuitive metric and the most widely reported in the RFM
    segmentation literature, so it is used as the primary selection criterion.

Davies-Bouldin Index (Davies & Bouldin, 1979)
    Average ratio of within-cluster scatter to between-cluster distance.
    Lower is better; no absolute scale.  Tends to favour compact, well-
    separated clusters and is complementary to silhouette.

Calinski-Harabasz Index (Calinski & Harabasz, 1974)
    Ratio of between-cluster dispersion to within-cluster dispersion,
    scaled by cluster and sample counts.  Higher is better.  It rewards
    both tight clusters and large inter-cluster gaps, but tends to favour
    higher k because adding clusters always increases between-cluster
    variance.  Useful as a third independent vote.

HDBSCAN noise handling
----------------------
HDBSCAN labels points it cannot assign to any dense region as -1 (noise).
Including these in metric calculations would be misleading: sklearn treats
-1 as a valid cluster label, artificially inflating Davies-Bouldin (a
scattered noise "cluster" is far from its centroid) and deflating silhouette
(noise points are poorly separated from everything).  All three metrics for
HDBSCAN are therefore computed only on the subset of points with a genuine
cluster label (>= 0).  The noise fraction is reported separately.

Bootstrap stability analysis
----------------------------
A clustering solution is *stable* if it consistently recovers the same
partition when fitted on different random subsamples of the data.  Instability
signals that the cluster structure is an artefact of the particular sample
rather than a genuine feature of the population.

For each algorithm and each of ``N_BOOTSTRAP_ITERATIONS`` rounds:

1.  Draw a bootstrap sample (n_customers indices, *with* replacement) from
    the scaled feature matrix.
2.  Deduplicate the sample indices to obtain a unique subsample of roughly
    63.2% of customers on average (the expected fraction of distinct items in
    a bootstrap sample).
3.  Fit the algorithm on the subsample.
4.  For K-Means, predict labels for *all* customers (K-Means exposes a
    ``predict`` method, so out-of-sample assignment is trivial).  Compare
    the full predicted labels with the reference labels using ARI.
5.  For HDBSCAN, labels are available only for the unique subsample.
    Compare those subsample labels to the reference HDBSCAN labels at the
    same indices.  ARI treats -1 (noise) as just another label, so noise
    agreement contributes to — rather than inflating — the score.

Adjusted Rand Index (ARI) — Hubert & Arabie, 1985
    Measures pairwise label agreement between two partitions, corrected for
    chance.  Range: [-1, 1]; 1 = perfect agreement, 0 = random, negative =
    worse than random.  ARI is invariant to label permutations, so swapping
    "Cluster 0" and "Cluster 1" between runs does not penalise the score.

Best algorithm selection rule
------------------------------
:func:`select_best_algorithm` applies the following decision tree:

1.  Disqualify HDBSCAN if its noise fraction exceeds
    ``MAX_HDBSCAN_NOISE_FRACTION`` (default 30%).  A solution that cannot
    assign nearly a third of customers to any meaningful group is not useful
    for targeted marketing, regardless of its metric scores.
2.  Among the remaining candidates, retain only those with
    mean ARI >= ``ARI_STABILITY_THRESHOLD`` (default 0.70).  A solution that
    is not reproducible on fresh subsamples cannot be trusted, even if it
    scores well on a single run.
3.  Among stable candidates, return the one with the highest Silhouette
    Coefficient, as it is the most interpretable metric and the standard
    reported in the RFM literature.
4.  If no candidate is stable (rare with this dataset size), fall back to
    highest silhouette across all candidates and emit a warning.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import hdbscan as hdbscan_lib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.config import (
    ARI_STABILITY_THRESHOLD,
    CLUSTER_PARQUET,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    KMEANS_BEST_K,
    MAX_HDBSCAN_NOISE_FRACTION,
    N_BOOTSTRAP_ITERATIONS,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    RANDOM_STATE,
    SCALED_FEATURES_PARQUET,
)
from src.preprocessing import FEATURE_COLS

logger = logging.getLogger(__name__)

VALIDATION_CSV = OUTPUTS_TABLES_DIR / "cluster_validation.csv"
STABILITY_PNG = OUTPUTS_FIGURES_DIR / "stability_ari.png"


class ValidationResult(NamedTuple):
    """Container returned by :func:`run_validation`.

    Attributes
    ----------
    metrics_df:
        Internal validity metrics table (one row per algorithm).
    stability_df:
        Per-algorithm ARI summary (mean, std, min, max).
    ari_scores:
        Raw ARI lists keyed by algorithm name, for custom analysis.
    best_algorithm:
        Name of the algorithm chosen by :func:`select_best_algorithm`.
    """
    metrics_df: pd.DataFrame
    stability_df: pd.DataFrame
    ari_scores: dict[str, list[float]]
    best_algorithm: str


# ---------------------------------------------------------------------------
# Internal validity metrics
# ---------------------------------------------------------------------------

def _compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
) -> dict:
    """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz for one set of labels.

    For HDBSCAN, noise points (label == -1) are excluded before metric
    calculation.  See module docstring for the full justification.

    Parameters
    ----------
    X:
        Scaled feature matrix, shape ``(n_customers, n_features)``.
    labels:
        Integer cluster labels; -1 is treated as noise and excluded.
    algorithm:
        Name string written into the output row.

    Returns
    -------
    dict
        Keys: Algorithm, n_clusters, noise_fraction, silhouette,
              davies_bouldin, calinski_harabasz.
    """
    is_noise = labels == -1
    noise_fraction = float(is_noise.mean())
    n_all = len(labels)

    # Restrict metrics to non-noise points.
    mask = ~is_noise
    X_valid = X[mask]
    labels_valid = labels[mask]

    n_clusters = len(np.unique(labels_valid))

    if n_clusters < 2 or len(X_valid) < n_clusters + 1:
        logger.warning("%s: not enough clusters/points for metrics.", algorithm)
        return {
            "Algorithm": algorithm,
            "n_clusters": n_clusters,
            "noise_fraction": round(noise_fraction, 4),
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    sil = silhouette_score(X_valid, labels_valid,
                           sample_size=min(3000, len(X_valid)),
                           random_state=RANDOM_STATE)
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)

    return {
        "Algorithm": algorithm,
        "n_clusters": n_clusters,
        "noise_fraction": round(noise_fraction, 4),
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
    }


def _build_metrics_table(
    X: np.ndarray,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
) -> pd.DataFrame:
    """Assemble the internal validity comparison table.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    kmeans_labels:
        Labels from the final K-Means fit.
    hdbscan_labels:
        Labels from the HDBSCAN fit (-1 = noise).

    Returns
    -------
    pd.DataFrame
        One row per algorithm with all three metric values.
    """
    rows = [
        _compute_internal_metrics(X, kmeans_labels, "K-Means"),
        _compute_internal_metrics(X, hdbscan_labels, "HDBSCAN"),
    ]
    df = pd.DataFrame(rows).set_index("Algorithm")
    logger.info("Validation metrics:\n%s", df.to_string())
    return df


# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

def _bootstrap_ari_kmeans(
    X: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> float:
    """One bootstrap round for K-Means.

    Draws a bootstrap sample, fits KMeans on it, predicts labels for *all*
    customers, and returns ARI against the reference partition.

    K-Means exposes ``predict``, so full-dataset assignment requires no
    approximation.  Using ``n_init=10`` (vs 50 in the main fit) keeps the
    stability loop fast; the goal here is comparative reproducibility, not
    finding the global optimum.

    Parameters
    ----------
    X:
        Full scaled feature matrix.
    reference_labels:
        Labels from the original K-Means fit (the target to compare against).
    k:
        Number of clusters (same as the original fit).
    rng:
        Seeded numpy random generator for reproducibility.
    """
    idx = rng.choice(len(X), size=len(X), replace=True)
    unique_idx = np.unique(idx)
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=int(rng.integers(1e6)))
    km.fit(X[unique_idx])
    boot_labels = km.predict(X)
    return adjusted_rand_score(reference_labels, boot_labels)


def _bootstrap_ari_hdbscan(
    X: np.ndarray,
    reference_labels: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """One bootstrap round for HDBSCAN.

    HDBSCAN does not support out-of-sample ``predict`` for arbitrary points,
    so this function fits on the unique bootstrap subsample and computes ARI
    only on those in-sample points.  ARI treats -1 (noise) as a valid label,
    so noise agreement is included in the score.

    The expected subsample size is ~63.2% of n_customers (the mean fraction of
    distinct items in a bootstrap draw), ensuring a meaningful coverage.

    Parameters
    ----------
    X:
        Full scaled feature matrix.
    reference_labels:
        Labels from the original HDBSCAN fit (the target to compare against).
    rng:
        Seeded numpy random generator for reproducibility.
    """
    idx = rng.choice(len(X), size=len(X), replace=True)
    unique_idx = np.unique(idx)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    boot_labels = clusterer.fit_predict(X[unique_idx])
    ref_sub = reference_labels[unique_idx]
    return adjusted_rand_score(ref_sub, boot_labels)


def _run_stability_analysis(
    X: np.ndarray,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
    k: int,
) -> dict[str, list[float]]:
    """Run N_BOOTSTRAP_ITERATIONS rounds for each algorithm, collect ARI scores.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    kmeans_labels:
        Reference labels from the final K-Means fit.
    hdbscan_labels:
        Reference labels from the final HDBSCAN fit.
    k:
        K-Means cluster count.

    Returns
    -------
    dict
        ``{"K-Means": [ari_0, …, ari_N], "HDBSCAN": [ari_0, …, ari_N]}``
    """
    rng = np.random.default_rng(RANDOM_STATE)
    ari_scores: dict[str, list[float]] = {"K-Means": [], "HDBSCAN": []}

    logger.info("Bootstrap stability: %d iterations …", N_BOOTSTRAP_ITERATIONS)
    for i in range(N_BOOTSTRAP_ITERATIONS):
        ari_km = _bootstrap_ari_kmeans(X, kmeans_labels, k, rng)
        ari_hdb = _bootstrap_ari_hdbscan(X, hdbscan_labels, rng)
        ari_scores["K-Means"].append(ari_km)
        ari_scores["HDBSCAN"].append(ari_hdb)
        if (i + 1) % 10 == 0:
            logger.info(
                "  Iteration %2d/%d | K-Means ARI=%.3f | HDBSCAN ARI=%.3f",
                i + 1, N_BOOTSTRAP_ITERATIONS, ari_km, ari_hdb,
            )

    return ari_scores


def _summarise_stability(ari_scores: dict[str, list[float]]) -> pd.DataFrame:
    """Convert raw ARI lists into a mean / std / min / max summary table."""
    rows = []
    for algo, scores in ari_scores.items():
        arr = np.array(scores)
        rows.append({
            "Algorithm": algo,
            "ARI_mean": round(arr.mean(), 4),
            "ARI_std": round(arr.std(), 4),
            "ARI_min": round(arr.min(), 4),
            "ARI_max": round(arr.max(), 4),
            "Stable": arr.mean() >= ARI_STABILITY_THRESHOLD,
        })
    df = pd.DataFrame(rows).set_index("Algorithm")
    logger.info("Stability summary:\n%s", df.to_string())
    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_stability_boxplot(ari_scores: dict[str, list[float]]) -> None:
    """Save a boxplot comparing ARI distributions across algorithms.

    The horizontal dashed line marks ``ARI_STABILITY_THRESHOLD`` so the
    figure is self-documenting: algorithms whose boxes sit above the line are
    considered stable by the criterion used in :func:`select_best_algorithm`.

    Parameters
    ----------
    ari_scores:
        Raw ARI lists keyed by algorithm name.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = list(ari_scores.keys())
    data = [ari_scores[k] for k in labels]
    colours = ["#4C72B0", "#DD8452"]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        widths=0.45,
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    # Scatter jittered raw points for transparency.
    rng = np.random.default_rng(RANDOM_STATE)
    for i, scores in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(scores))
        ax.scatter(np.full(len(scores), i) + jitter, scores,
                   color=colours[i - 1], s=12, alpha=0.4, zorder=3)

    ax.axhline(ARI_STABILITY_THRESHOLD, color="red", linestyle="--",
               linewidth=1.4, label=f"Stability threshold (ARI={ARI_STABILITY_THRESHOLD})")

    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=11)
    ax.set_title(
        f"Clustering stability — {N_BOOTSTRAP_ITERATIONS} bootstrap iterations\n"
        "(ARI vs reference partition; higher = more stable)",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate mean ARI on each box.
    for i, (algo, scores) in enumerate(ari_scores.items(), start=1):
        mean_ari = np.mean(scores)
        ax.text(i, mean_ari + 0.03, f"mean={mean_ari:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(STABILITY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Stability boxplot saved to %s", STABILITY_PNG)


# ---------------------------------------------------------------------------
# Algorithm selection
# ---------------------------------------------------------------------------

def select_best_algorithm(
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> str:
    """Choose the best clustering algorithm by a transparent, documented rule.

    Decision tree (applied in order; the first matching rule wins):

    1. **Noise disqualification** — if HDBSCAN's noise fraction exceeds
       ``MAX_HDBSCAN_NOISE_FRACTION`` (default 0.30), HDBSCAN is removed from
       consideration.  A solution that cannot assign ≥ 30% of customers to any
       cohesive group has limited marketing utility: those customers cannot be
       assigned to a campaign strategy.

    2. **Stability filter** — retain only algorithms whose mean bootstrap ARI
       is >= ``ARI_STABILITY_THRESHOLD`` (default 0.70).  An ARI below 0.70
       indicates that refitting on a random subsample produces a substantially
       different partition, meaning the labels are sensitive to the exact
       sample and should not be trusted as a stable segment definition.

    3. **Primary criterion: silhouette** — among the stable candidates, select
       the algorithm with the highest Silhouette Coefficient.  Silhouette is
       chosen as the primary metric because it is normalised (interpretable
       without a baseline), widely reported in the RFM literature, and measures
       the property that matters most for marketing: that customers within a
       segment are more similar to each other than to customers in other
       segments.

    4. **Fallback** — if no algorithm passes the stability filter (rare at
       n > 5 000), all candidates are reinstated and step 3 is applied with a
       logged warning so the researcher is alerted.

    Parameters
    ----------
    metrics_df:
        DataFrame produced by :func:`_build_metrics_table`, indexed by
        algorithm name.  Must contain a ``silhouette`` column and a
        ``noise_fraction`` column.
    stability_df:
        DataFrame produced by :func:`_summarise_stability`, indexed by
        algorithm name.  Must contain ``ARI_mean`` and ``Stable`` columns.

    Returns
    -------
    str
        Name of the selected algorithm (matches the index of ``metrics_df``).
    """
    combined = metrics_df.join(stability_df, how="inner")

    # Step 1: disqualify noisy HDBSCAN.
    too_noisy = combined["noise_fraction"] > MAX_HDBSCAN_NOISE_FRACTION
    if too_noisy.any():
        noisy_algos = combined.index[too_noisy].tolist()
        logger.warning(
            "Disqualifying %s: noise fraction exceeds %.0f%%.",
            noisy_algos, MAX_HDBSCAN_NOISE_FRACTION * 100,
        )
        combined = combined[~too_noisy]

    if combined.empty:
        logger.warning("All algorithms disqualified by noise criterion; "
                       "re-instating all candidates.")
        combined = metrics_df.join(stability_df, how="inner")

    # Step 2: stability filter.
    stable = combined[combined["Stable"]]
    if stable.empty:
        logger.warning(
            "No algorithm reached ARI stability threshold (%.2f). "
            "Falling back to highest silhouette across all candidates.",
            ARI_STABILITY_THRESHOLD,
        )
        stable = combined  # fallback: all candidates

    # Step 3: highest silhouette among stable candidates.
    best = stable["silhouette"].idxmax()
    best_sil = stable.loc[best, "silhouette"]
    best_ari = stable.loc[best, "ARI_mean"]
    logger.info(
        "Selected algorithm: %s  (silhouette=%.4f, mean ARI=%.4f).",
        best, best_sil, best_ari,
    )
    return str(best)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_validation(
    X: np.ndarray | None = None,
    kmeans_labels: np.ndarray | None = None,
    hdbscan_labels: np.ndarray | None = None,
) -> ValidationResult:
    """Run the full validation and stability suite.

    If no arguments are supplied, loads data from the saved parquet files
    produced by Stages 2b and 3.

    Parameters
    ----------
    X:
        Scaled feature matrix.  Pass ``None`` to load from
        ``data/processed/scaled_features.parquet``.
    kmeans_labels:
        K-Means cluster labels.  Pass ``None`` to load from
        ``data/processed/clustered_customers.parquet``.
    hdbscan_labels:
        HDBSCAN cluster labels.  Pass ``None`` to load from the same parquet.

    Returns
    -------
    ValidationResult
        Named tuple with fields ``metrics_df``, ``stability_df``,
        ``ari_scores``, and ``best_algorithm``.

    Side effects
    ------------
    * Writes ``outputs/tables/cluster_validation.csv``.
    * Writes ``outputs/figures/stability_ari.png``.
    """
    # ── Load inputs ────────────────────────────────────────────────────────
    if X is None:
        logger.info("Loading scaled features from %s", SCALED_FEATURES_PARQUET)
        scaled_df = pd.read_parquet(SCALED_FEATURES_PARQUET)
        X = scaled_df[FEATURE_COLS].values

    if kmeans_labels is None or hdbscan_labels is None:
        logger.info("Loading cluster labels from %s", CLUSTER_PARQUET)
        cluster_df = pd.read_parquet(CLUSTER_PARQUET)
        kmeans_labels = cluster_df["KMeans_Cluster"].values
        hdbscan_labels = cluster_df["HDBSCAN_Cluster"].values

    k = len(np.unique(kmeans_labels))
    logger.info(
        "Validating: %d customers, K-Means k=%d, HDBSCAN clusters=%d + noise.",
        len(X), k,
        len([c for c in np.unique(hdbscan_labels) if c != -1]),
    )

    # ── Internal validity metrics ──────────────────────────────────────────
    metrics_df = _build_metrics_table(X, kmeans_labels, hdbscan_labels)
    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(VALIDATION_CSV)
    logger.info("Validation metrics saved to %s", VALIDATION_CSV)

    # ── Bootstrap stability ────────────────────────────────────────────────
    ari_scores = _run_stability_analysis(X, kmeans_labels, hdbscan_labels, k)
    stability_df = _summarise_stability(ari_scores)
    _plot_stability_boxplot(ari_scores)

    # ── Algorithm selection ────────────────────────────────────────────────
    best = select_best_algorithm(metrics_df, stability_df)

    return ValidationResult(
        metrics_df=metrics_df,
        stability_df=stability_df,
        ari_scores=ari_scores,
        best_algorithm=best,
    )
