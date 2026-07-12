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
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.config import (
    AGGLOMERATIVE_LINKAGE,
    ARI_STABILITY_THRESHOLD,
    CLUSTER_PARQUET,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    MAX_HDBSCAN_NOISE_FRACTION,
    N_BOOTSTRAP_ITERATIONS,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    RANDOM_STATE,
    SCALED_FEATURES_PARQUET,
    SPECTRAL_AFFINITY,
    SPECTRAL_N_NEIGHBORS,
)
from src.preprocessing import FEATURE_COLS

logger = logging.getLogger(__name__)

VALIDATION_CSV = OUTPUTS_TABLES_DIR / "cluster_validation.csv"
CLUSTER_RANKING_CSV = OUTPUTS_TABLES_DIR / "cluster_ranking.csv"
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
    labels_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Assemble the internal validity comparison table for all algorithms.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    labels_dict:
        Mapping of algorithm name to integer label array.  Algorithms with
        noise points (label == -1) have those points excluded from metrics.

    Returns
    -------
    pd.DataFrame
        One row per algorithm with silhouette, davies_bouldin,
        calinski_harabasz, n_clusters, and noise_fraction.
    """
    rows = [_compute_internal_metrics(X, lbl, algo)
            for algo, lbl in labels_dict.items()]
    df = pd.DataFrame(rows).set_index("Algorithm")
    logger.info("Validation metrics:\n%s", df.to_string())
    return df


# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

def _one_bootstrap_round(
    X: np.ndarray,
    ref: np.ndarray,
    algo: str,
    params: dict,
    rng: np.random.Generator,
) -> float:
    """One bootstrap ARI round for any of the four supported algorithms.

    Algorithms that expose a ``predict`` method (K-Means, GMM) are fitted on
    the bootstrap subsample and then applied to the full dataset, so ARI is
    measured against all n_customers reference labels.

    Algorithms without ``predict`` (DBSCAN, HDBSCAN) are fitted on the unique
    bootstrap subsample and ARI is measured only on those in-sample indices.
    ARI treats -1 (noise) as a valid label, so noise agreement contributes to
    the score rather than inflating it.

    Parameters
    ----------
    X:
        Full scaled feature matrix.
    ref:
        Reference labels from the original fit.
    algo:
        One of ``"K-Means"``, ``"DBSCAN"``, ``"GMM"``, ``"HDBSCAN"``.
    params:
        Algorithm-specific hyper-parameters (e.g. k, n_components, eps).
    rng:
        Seeded numpy Generator for reproducibility.
    """
    idx = rng.choice(len(X), size=len(X), replace=True)
    uid = np.unique(idx)
    seed = int(rng.integers(1_000_000))

    if algo == "K-Means":
        k = params["k"]
        m = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
        m.fit(X[uid])
        return adjusted_rand_score(ref, m.predict(X))

    if algo == "GMM":
        n = params["n"]
        m = GaussianMixture(n_components=n, covariance_type="full",
                            random_state=seed, n_init=3, max_iter=200)
        m.fit(X[uid])
        return adjusted_rand_score(ref, m.predict(X))

    if algo == "DBSCAN":
        eps = params["eps"]
        m = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean")
        boot_labels = m.fit_predict(X[uid])
        return adjusted_rand_score(ref[uid], boot_labels)

    if algo == "HDBSCAN":
        m = hdbscan_lib.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        boot_labels = m.fit_predict(X[uid])
        return adjusted_rand_score(ref[uid], boot_labels)

    if algo == "Agglomerative":
        k = params["k"]
        m = AgglomerativeClustering(n_clusters=k, linkage=AGGLOMERATIVE_LINKAGE)
        boot_labels = m.fit_predict(X[uid])
        return adjusted_rand_score(ref[uid], boot_labels)

    if algo == "Spectral":
        k = params["k"]
        m = SpectralClustering(n_clusters=k, affinity=SPECTRAL_AFFINITY,
                               n_neighbors=SPECTRAL_N_NEIGHBORS,
                               assign_labels="kmeans", random_state=seed)
        boot_labels = m.fit_predict(X[uid])
        return adjusted_rand_score(ref[uid], boot_labels)

    raise ValueError(f"Unknown algorithm: {algo}")


def _run_stability_analysis(
    X: np.ndarray,
    labels_dict: dict[str, np.ndarray],
    algo_params: dict[str, dict],
) -> dict[str, list[float]]:
    """Run N_BOOTSTRAP_ITERATIONS ARI rounds for every algorithm.

    Parameters
    ----------
    X:
        Scaled feature matrix.
    labels_dict:
        Reference labels keyed by algorithm name.
    algo_params:
        Hyper-parameters needed for refitting each algorithm in bootstrap
        rounds (e.g. ``{"K-Means": {"k": 2}, "GMM": {"n": 3}, ...}``).

    Returns
    -------
    dict
        ``{algo: [ari_0, …, ari_N]}`` for each algorithm.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    ari_scores: dict[str, list[float]] = {a: [] for a in labels_dict}

    logger.info("Bootstrap stability: %d iterations, %d algorithms.",
                N_BOOTSTRAP_ITERATIONS, len(labels_dict))
    for i in range(N_BOOTSTRAP_ITERATIONS):
        for algo, ref in labels_dict.items():
            ari = _one_bootstrap_round(X, ref, algo,
                                       algo_params.get(algo, {}), rng)
            ari_scores[algo].append(ari)
        if (i + 1) % 10 == 0:
            summary = "  |  ".join(
                f"{a} ARI={np.mean(ari_scores[a]):.3f}"
                for a in labels_dict
            )
            logger.info("  Iter %2d/%d | %s", i + 1,
                        N_BOOTSTRAP_ITERATIONS, summary)

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


def _rank_algorithms(metrics_df: pd.DataFrame, stability_df: pd.DataFrame) -> pd.DataFrame:
    """Rank algorithms across all internal + stability metrics into a mean rank.

    Combines Silhouette (higher better), Davies-Bouldin (lower better),
    Calinski-Harabasz (higher better) and bootstrap ARI (higher better) into a
    single 'who wins overall' table.  Algorithms with undefined metrics (e.g.
    DBSCAN with one cluster) are ranked last on those metrics.
    """
    combined = metrics_df.join(stability_df, how="inner")
    specs = {
        "silhouette": False,         # higher better
        "davies_bouldin": True,      # lower better
        "calinski_harabasz": False,  # higher better
        "ARI_mean": False,           # higher better
    }
    ranks = pd.DataFrame(index=combined.index)
    for col, ascending in specs.items():
        ranks[f"{col}_rank"] = combined[col].rank(
            ascending=ascending, method="min", na_option="bottom")
    ranks["mean_rank"] = ranks.mean(axis=1).round(2)
    ranks = ranks.sort_values("mean_rank")
    ranks["overall_rank"] = range(1, len(ranks) + 1)
    return ranks


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
    _palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
    colours = [_palette[i % len(_palette)] for i in range(len(labels))]

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
    labels_dict: dict[str, np.ndarray] | None = None,
    algo_params: dict[str, dict] | None = None,
) -> ValidationResult:
    """Run the full validation and stability suite for all four algorithms.

    If ``labels_dict`` is ``None``, all four label columns are loaded from the
    cluster parquet produced by :func:`~src.clustering.run_all_clustering`.

    Parameters
    ----------
    X:
        Scaled feature matrix.  Pass ``None`` to load from
        ``data/processed/scaled_features.parquet``.
    labels_dict:
        Mapping ``{algorithm_name: label_array}``.  Pass ``None`` to load
        KMeans_Cluster, DBSCAN_Cluster, GMM_Cluster, and HDBSCAN_Cluster from
        ``data/processed/clustered_customers.parquet``.
    algo_params:
        Hyper-parameters for the bootstrap refitting step, e.g.::

            {
                "K-Means": {"k": 3},
                "GMM":     {"n": 4},
                "DBSCAN":  {"eps": 0.85},
            }

        HDBSCAN needs no extra params.  Pass ``None`` to infer k from the
        label array; ``eps`` falls back to ``DBSCAN_EPS`` if set.

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
    # ── Load scaled features ───────────────────────────────────────────────
    if X is None:
        logger.info("Loading scaled features from %s", SCALED_FEATURES_PARQUET)
        scaled_df = pd.read_parquet(SCALED_FEATURES_PARQUET)
        X = scaled_df[FEATURE_COLS].values

    # ── Load cluster labels ────────────────────────────────────────────────
    _col_map = {
        "K-Means": "KMeans_Cluster",
        "DBSCAN":  "DBSCAN_Cluster",
        "GMM":     "GMM_Cluster",
        "HDBSCAN": "HDBSCAN_Cluster",
        "Agglomerative": "Agglomerative_Cluster",
        "Spectral":      "Spectral_Cluster",
    }
    if labels_dict is None:
        logger.info("Loading cluster labels from %s", CLUSTER_PARQUET)
        cluster_df = pd.read_parquet(CLUSTER_PARQUET)
        labels_dict = {}
        for algo, col in _col_map.items():
            if col in cluster_df.columns:
                labels_dict[algo] = cluster_df[col].values
            else:
                logger.warning("Column %s not found in cluster parquet; skipping %s.", col, algo)

    # ── Build algo_params if not supplied ──────────────────────────────────
    if algo_params is None:
        algo_params = {}
        if "K-Means" in labels_dict:
            algo_params["K-Means"] = {"k": int(len(np.unique(labels_dict["K-Means"])))}
        if "GMM" in labels_dict:
            algo_params["GMM"] = {"n": int(len(np.unique(labels_dict["GMM"])))}
        if "DBSCAN" in labels_dict:
            _eps = DBSCAN_EPS if DBSCAN_EPS is not None else 0.5
            algo_params["DBSCAN"] = {"eps": _eps}
        if "Agglomerative" in labels_dict:
            algo_params["Agglomerative"] = {"k": int(len(np.unique(labels_dict["Agglomerative"])))}
        if "Spectral" in labels_dict:
            algo_params["Spectral"] = {"k": int(len(np.unique(labels_dict["Spectral"])))}

    logger.info(
        "Validating %d customers across %d algorithms: %s",
        len(X), len(labels_dict), list(labels_dict.keys()),
    )

    # ── Internal validity metrics ──────────────────────────────────────────
    metrics_df = _build_metrics_table(X, labels_dict)
    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(VALIDATION_CSV)
    logger.info("Validation metrics saved to %s", VALIDATION_CSV)

    # ── Bootstrap stability ────────────────────────────────────────────────
    ari_scores = _run_stability_analysis(X, labels_dict, algo_params)
    stability_df = _summarise_stability(ari_scores)
    _plot_stability_boxplot(ari_scores)

    # ── Overall ranking across all metrics ─────────────────────────────────
    ranking = _rank_algorithms(metrics_df, stability_df)
    ranking.to_csv(CLUSTER_RANKING_CSV)
    logger.info("Algorithm ranking saved to %s\n%s",
                CLUSTER_RANKING_CSV, ranking.to_string())

    # ── Algorithm selection ────────────────────────────────────────────────
    best = select_best_algorithm(metrics_df, stability_df)

    return ValidationResult(
        metrics_df=metrics_df,
        stability_df=stability_df,
        ari_scores=ari_scores,
        best_algorithm=best,
    )
