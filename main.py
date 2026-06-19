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
from pathlib import Path

# Ensure the project root is on the import path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Execute the full segmentation pipeline in sequence."""
    logger.info("Pipeline started.")

    # Stage 1 – Data ingestion & cleaning
    from src.data_loading import load_raw_data
    from src.cleaning import clean_transactions
    raw = load_raw_data()
    clean_transactions(raw)

    # Stage 2 – Feature engineering (RFM + CLV)
    # from src.features import build_rfm, build_clv
    # build_rfm()
    # build_clv()

    # Stage 3 – Clustering
    # from src.clustering import run_kmeans, run_hdbscan
    # run_kmeans()
    # run_hdbscan()

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


if __name__ == "__main__":
    run_pipeline()
