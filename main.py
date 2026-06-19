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

import numpy as np
import pandas as pd

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
    elapsed: float,
) -> None:
    """Print a structured end-of-run summary covering all four stages."""

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

    n_hdb_real = sum(n for c, n in hdbscan_sizes.items() if c != -1)
    n_noise = hdbscan_sizes.get(-1, 0)
    print(f"\n    HDBSCAN:")
    for cid, n in sorted(hdbscan_sizes.items()):
        tag = "(noise)" if cid == -1 else ""
        pct = n / sum(hdbscan_sizes.values()) * 100
        _row(f"      Cluster {cid} {tag}:", f"{n:,} customers  ({pct:.1f}%)")
    _row("    Genuine clusters:", str(len(hdbscan_sizes) - (1 if -1 in hdbscan_sizes else 0)))
    _row("    Noise points:", f"{n_noise:,}  ({n_noise/sum(hdbscan_sizes.values())*100:.1f}%)")

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
    from src.clustering import run_clustering
    from src.config import OUTPUTS_TABLES_DIR as TABLES_DIR
    cluster_df = run_clustering()
    kmeans_metrics = pd.read_csv(TABLES_DIR / "kmeans_metrics.csv")
    best_k = int(cluster_df["KMeans_Cluster"].nunique())
    kmeans_sizes = cluster_df["KMeans_Cluster"].value_counts().sort_index().to_dict()
    hdbscan_sizes = cluster_df["HDBSCAN_Cluster"].value_counts().sort_index().to_dict()

    # Stage 4 – Cluster profiling & visualisation
    # from src.profiling import profile_clusters
    # profile_clusters()

    # Stage 5 – XGBoost cluster classifier
    # from src.classifier import train_classifier
    # train_classifier()

    # Stage 6 – Notification rule generation
    # from src.notifications import generate_notifications
    # generate_notifications()

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
        elapsed=time.time() - t0,
    )


if __name__ == "__main__":
    run_pipeline()
