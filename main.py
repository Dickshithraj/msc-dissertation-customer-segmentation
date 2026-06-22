"""
Pipeline orchestrator for the customer segmentation dissertation project.

Run the full pipeline end-to-end::

    python main.py

Each stage is imported from ``src/`` and executed in order. Comment out
individual stages during development to skip completed steps.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from src.config import CLV_TIME_MONTHS
from src.validation import ValidationResult

# Ensure the project root is on the import path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_W = 72  # summary table width


def _banner(title: str) -> None:
    print(f"\n{'=' * _W}")
    print(f"  {title}")
    print(f"{'=' * _W}")


def _check(path: str) -> str:
    return f"  [ok] {path}"


def _row(label: str, value: str) -> None:
    print(f"  {label:<38} {value}")


def _divider() -> None:
    print(f"  {'-' * (_W - 2)}")


def _print_pipeline_summary(
    raw_rows: int,
    clean_rows: int,
    removal_steps: list[dict],
    n_customers: int,
    feature_stats: pd.DataFrame,
    log1p_cols: list[str],
    best_k: int,
    kmeans_metrics: pd.DataFrame,
    kmeans_sizes: dict[int, int],
    hdbscan_sizes: dict[int, int],
    validation: ValidationResult,
    profiles: "pd.DataFrame",
    clv_df: "pd.DataFrame",
    churn_df: "pd.DataFrame",
    churn_metrics: "pd.DataFrame",
    migration: dict,
    elapsed: float,
) -> None:
    """Print a structured end-of-run summary covering all stages."""

    _banner("PIPELINE SUMMARY - Customer Segmentation Project")

    # ── Stage 1: Loading & Cleaning ────────────────────────────────────────
    print(f"\n  STAGE 1 - Data Loading & Cleaning")
    _divider()
    _row("Raw rows loaded (both sheets):", f"{raw_rows:,}")
    for step in removal_steps:
        label = f"  Step {step['Step']} - {step['Description'][:30]}:"
        _row(label, f"-{step['Rows Removed']:,}  ({step['% Removed']}%)")
    _row("Final clean rows:", f"{clean_rows:,}")
    _row("Rows removed total:",
         f"{raw_rows - clean_rows:,}  ({(raw_rows - clean_rows)/raw_rows*100:.1f}%)")

    # ── Stage 2: Feature Engineering ───────────────────────────────────────
    print(f"\n  STAGE 2 - Feature Engineering")
    _divider()
    _row("Unique customers:", f"{n_customers:,}")
    _row("Features computed:", "Recency, Frequency, Monetary, Tenure,")
    _row("", "AvgOrderValue, AvgInterPurchaseDays, DistinctProducts")
    _divider()
    _row("Feature", "Median        Mean          Max")
    _divider()
    for feat in ["Recency", "Frequency", "Monetary", "Tenure",
                 "AvgOrderValue", "AvgInterPurchaseDays", "DistinctProducts"]:
        s = feature_stats[feat]
        _row(f"  {feat}",
             f"{s['50%']:<14.1f}{s['mean']:<14.1f}{s['max']:.1f}")

    # ── Stage 2b: Preprocessing ────────────────────────────────────────────
    print(f"\n  STAGE 2b - Preprocessing (log1p + StandardScaler)")
    _divider()
    _row("log1p applied to:", ", ".join(log1p_cols))
    _row("StandardScaler:", "all 7 features -> mean=0, std=1")

    # ── Stage 3: Clustering ────────────────────────────────────────────────
    print(f"\n  STAGE 3 - Clustering")
    _divider()

    print(f"\n    K-Means sweep (k = 2 … {kmeans_metrics['k'].max()}):")
    _row("    k", "Inertia       Silhouette    Davies-Bouldin")
    _divider()
    for _, r in kmeans_metrics.iterrows():
        marker = " <-- chosen" if r["k"] == best_k else ""
        _row(f"    k={int(r['k'])}",
             f"{r['inertia']:<14,.0f}{r['silhouette']:<14.4f}{r['davies_bouldin']:.4f}{marker}")

    print(f"\n    Final K-Means  (k = {best_k}):")
    for cid, n in sorted(kmeans_sizes.items()):
        pct = n / sum(kmeans_sizes.values()) * 100
        _row(f"      Cluster {cid}:", f"{n:,} customers  ({pct:.1f}%)")

    n_noise = hdbscan_sizes.get(-1, 0)
    print(f"\n    HDBSCAN:")
    for cid, n in sorted(hdbscan_sizes.items()):
        tag = "(noise)" if cid == -1 else ""
        pct = n / sum(hdbscan_sizes.values()) * 100
        _row(f"      Cluster {cid} {tag}:", f"{n:,} customers  ({pct:.1f}%)")
    _row("    Genuine clusters:", str(len(hdbscan_sizes) - (1 if -1 in hdbscan_sizes else 0)))
    _row("    Noise points:", f"{n_noise:,}  ({n_noise/sum(hdbscan_sizes.values())*100:.1f}%)")

    # ── Stage 3b: Validation & Stability ───────────────────────────────────
    print(f"\n  STAGE 3b - Cluster Validation & Stability")
    _divider()
    m = validation.metrics_df
    algos = list(m.index)
    _row("Algorithm", "Silhouette    DB-Index   CH-Index  Noise%  Clusters")
    _divider()
    for algo in algos:
        row = m.loc[algo]
        _row(
            f"  {algo}",
            f"{row['silhouette']:<14.4f}{row['davies_bouldin']:<11.4f}"
            f"{row['calinski_harabasz']:<10.1f}{row['noise_fraction']*100:<8.1f}"
            f"{int(row['n_clusters'])}",
        )
    _divider()
    _row("Stability (bootstrap ARI)", "Mean           Std            Stable?")
    _divider()
    s = validation.stability_df
    for algo in s.index:
        mean = s.loc[algo, "ARI_mean"]
        std = s.loc[algo, "ARI_std"]
        stable = "Yes" if s.loc[algo, "Stable"] else "No"
        _row(f"  {algo}", f"{mean:<15.4f}{std:<15.4f}{stable}")
    _divider()
    _row("Selected algorithm:", validation.best_algorithm)

    # ── Stage 6: Cluster profiling ──────────────────────────────────────────
    print(f"\n  STAGE 6 - Segment Profiles ({validation.best_algorithm})")
    _divider()
    _row("Segment", "n        %      Recency  Frequency  Monetary")
    _divider()
    for cid, row in profiles.iterrows():
        tag = " (noise)" if cid == -1 else ""
        _row(
            f"  C{cid}{tag}: {row['segment_name']}",
            f"{int(row['size']):<9,}{row['pct_of_total']:<7.1f}"
            f"{row['Recency']:<9.0f}{row['Frequency']:<11.1f}{row['Monetary']:.0f}",
        )

    # ── Stage 7: Customer Lifetime Value ────────────────────────────────────
    print(f"\n  STAGE 7 - Customer Lifetime Value (BG/NBD + Gamma-Gamma)")
    _divider()
    clv = clv_df["clv"]
    total_clv = clv.sum()
    top_decile = clv.quantile(0.90)
    top_share = clv[clv >= top_decile].sum() / total_clv * 100
    _row("Forecast horizon:", f"{CLV_TIME_MONTHS} months (discounted)")
    _row("Total portfolio CLV:", f"{total_clv:,.0f}")
    _row("Median customer CLV:", f"{clv.median():,.0f}")
    _row("Mean customer CLV:", f"{clv.mean():,.0f}")
    _row("Top-decile CLV share:", f"{top_share:.1f}%  (top 10% of customers)")
    _row("Mean P(alive):", f"{clv_df['prob_alive'].mean():.3f}")

    # ── Stage 8: Churn classification ───────────────────────────────────────
    print(f"\n  STAGE 8 - Churn Classification")
    _divider()
    _row("Churn rate (label):", f"{churn_df['churn_label'].mean()*100:.1f}%  "
                                 f"({int(churn_df['churn_label'].sum()):,} churned)")
    _divider()
    _row("Model", "ROC-AUC   PR-AUC    F1       CV ROC-AUC")
    _divider()
    for model, row in churn_metrics.iterrows():
        _row(f"  {model}",
             f"{row['ROC_AUC']:<10.4f}{row['PR_AUC']:<10.4f}"
             f"{row['F1']:<9.4f}{row['CV_ROC_AUC_mean']:.4f}")
    _divider()
    best_churn = churn_metrics["ROC_AUC"].idxmax()
    _row("Best churn model:", f"{best_churn}  (ROC-AUC={churn_metrics.loc[best_churn,'ROC_AUC']:.4f})")
    _row("Mean churn probability:", f"{churn_df['churn_probability'].mean():.3f}")

    # ── Stage 9: Segment migration (year-on-year) ───────────────────────────
    print(f"\n  STAGE 9 - Year-on-Year Segment Migration")
    _divider()
    ctx = migration["context"]
    _row("Retained (both years):", f"{ctx['retained_both_years']:,}")
    _row("Lapsed (year 1 only):", f"{ctx['year1_only_lapsed']:,}")
    _row("New (year 2 only):", f"{ctx['year2_only_new']:,}")
    _divider()
    _row("Segment retention", "Stayed in same segment (year 1 -> year 2)")
    _divider()
    rates = migration["rates"]
    counts = migration["counts"]
    for seg in rates.index:
        if counts.loc[seg].sum() > 0:
            _row(f"  {seg}", f"{rates.loc[seg, seg]*100:.1f}%")

    # ── Files written ───────────────────────────────────────────────────────
    print(f"\n  OUTPUTS")
    _divider()
    outputs = [
        "data/processed/cleaned_transactions.parquet",
        "data/processed/customer_features.parquet",
        "data/processed/scaled_features.parquet",
        "data/processed/fitted_scaler.joblib",
        "data/processed/clustered_customers.parquet",
        "outputs/tables/cleaning_summary.csv",
        "outputs/tables/feature_summary.csv",
        "outputs/tables/kmeans_metrics.csv",
        "outputs/figures/rfm_distributions.png",
        "outputs/figures/scaling_effect.png",
        "outputs/figures/kmeans_selection.png",
        "outputs/figures/cluster_pca_projection.png",
        "outputs/tables/cluster_validation.csv",
        "outputs/figures/stability_ari.png",
        "outputs/tables/segment_profiles.csv",
        "outputs/figures/segment_profiles.png",
        "outputs/figures/radar_profiles.png",
        "data/processed/customer_clv.parquet",
        "outputs/tables/clv_summary.csv",
        "outputs/figures/clv_distribution.png",
        "data/processed/customer_churn.parquet",
        "outputs/tables/churn_model_comparison.csv",
        "outputs/figures/churn_roc_curves.png",
        "outputs/figures/churn_feature_importance.png",
        "outputs/tables/segment_migration_counts.csv",
        "outputs/tables/segment_migration_rates.csv",
        "outputs/figures/segment_migration.png",
    ]
    for path in outputs:
        print(_check(path))

    print(f"\n  Total wall time: {elapsed:.1f}s")
    print(f"{'=' * _W}\n")


def run_pipeline() -> None:
    """Execute the full segmentation pipeline in sequence."""
    logger.info("Pipeline started.")
    t0 = time.time()

    # ── Stage 1 ─────────────────────────────────────────────────────────────
    from src.data_loading import load_raw_data
    from src.cleaning import clean_transactions
    raw = load_raw_data()
    raw_rows = len(raw)
    cleaned = clean_transactions(raw)
    clean_rows = len(cleaned)

    # Reload cleaning summary for the final report
    from src.config import OUTPUTS_TABLES_DIR
    removal_steps = pd.read_csv(OUTPUTS_TABLES_DIR / "cleaning_summary.csv").to_dict("records")

    # ── Stage 2 ─────────────────────────────────────────────────────────────
    from src.features import build_customer_features
    features_df = build_customer_features()
    n_customers = len(features_df)
    feature_stats = features_df.describe().to_dict()

    # ── Stage 2b ────────────────────────────────────────────────────────────
    from src.preprocessing import preprocess_features
    prep = preprocess_features()

    # ── Stage 3 ─────────────────────────────────────────────────────────────
    from src.clustering import run_all_clustering
    from src.config import OUTPUTS_TABLES_DIR as TABLES_DIR
    clustering_result = run_all_clustering()
    cluster_df = clustering_result["cluster_df"]
    kmeans_metrics = pd.read_csv(TABLES_DIR / "kmeans_metrics.csv")
    best_k = int(cluster_df["KMeans_Cluster"].nunique())
    kmeans_sizes = cluster_df["KMeans_Cluster"].value_counts().sort_index().to_dict()
    hdbscan_sizes = cluster_df["HDBSCAN_Cluster"].value_counts().sort_index().to_dict()

    # Build the labels dict for validation (all 4 algorithms)
    label_cols = {
        "K-Means": "KMeans_Cluster",
        "DBSCAN":  "DBSCAN_Cluster",
        "GMM":     "GMM_Cluster",
        "HDBSCAN": "HDBSCAN_Cluster",
    }
    labels_dict = {
        algo: cluster_df[col].values
        for algo, col in label_cols.items()
        if col in cluster_df.columns
    }

    # Stage 3b – Cluster validation & stability
    from src.validation import run_validation
    validation = run_validation(X=prep.X_scaled, labels_dict=labels_dict)

    # Stage 6 – Cluster profiling & segment naming
    from src.profiling import profile_clusters
    best_algo = validation.best_algorithm
    profiles = profile_clusters(algo=best_algo)  # used in summary below

    # Stage 7 – Customer Lifetime Value (BG/NBD + Gamma-Gamma)
    from src.clv import build_clv
    clv_df = build_clv(transactions=cleaned)

    # Stage 8 – Churn classification (LogReg / RandomForest / XGBoost)
    from src.churn import run_churn
    churn_df = run_churn(features=features_df)
    churn_metrics = pd.read_csv(TABLES_DIR / "churn_model_comparison.csv", index_col=0)

    # Stage 9 – Year-on-year segment migration
    from src.migration import run_migration
    migration = run_migration(transactions=cleaned)

    logger.info("Pipeline complete.")

    # ── Print consolidated summary ───────────────────────────────────────────
    _print_pipeline_summary(
        raw_rows=raw_rows,
        clean_rows=clean_rows,
        removal_steps=removal_steps,
        n_customers=n_customers,
        feature_stats=feature_stats,
        log1p_cols=prep.log1p_cols,
        best_k=best_k,
        kmeans_metrics=kmeans_metrics,
        kmeans_sizes=kmeans_sizes,
        hdbscan_sizes=hdbscan_sizes,
        validation=validation,
        profiles=profiles,
        clv_df=clv_df,
        churn_df=churn_df,
        churn_metrics=churn_metrics,
        migration=migration,
        elapsed=time.time() - t0,
    )


if __name__ == "__main__":
    run_pipeline()
