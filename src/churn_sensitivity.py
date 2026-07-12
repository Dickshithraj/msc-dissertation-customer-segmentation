"""
Sensitivity of the churn results to the label threshold.

The churn label is a *proxy*: a customer is "churned" when their Recency
exceeds the ``CHURN_RECENCY_PERCENTILE`` (90th) percentile.  That percentile
is a design choice, so this module checks whether the Phase 8 conclusions --
tree ensembles beat the linear baseline, and the achievable ROC-AUC is high
-- survive moving the threshold to the 85th and 95th percentiles.

For each threshold the label is rebuilt, the same six classifiers are
retrained on the same stratified 80/20 split (same features, same
``RANDOM_STATE``), and held-out ROC-AUC / PR-AUC / F1 are recorded.  The
output table lets the report state whether the model ranking is stable
across plausible labellings rather than tuned to one.

Outputs
-------
outputs/tables/churn_threshold_sensitivity.csv
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.churn import CHURN_FEATURES, _build_models
from src.config import (
    CUSTOMER_FEATURES_PARQUET,
    OUTPUTS_TABLES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

THRESHOLD_SENSITIVITY_CSV = OUTPUTS_TABLES_DIR / "churn_threshold_sensitivity.csv"

# Percentile thresholds to test; 0.90 is the value reported in Phase 8.
THRESHOLDS: list[float] = [0.85, 0.90, 0.95]


def _evaluate_threshold(features: pd.DataFrame, percentile: float) -> list[dict]:
    """Rebuild the label at ``percentile`` and score all six models."""
    threshold_days = features["Recency"].quantile(percentile)
    y = (features["Recency"] > threshold_days).astype(int)
    X = features[CHURN_FEATURES]
    logger.info(
        "P%.0f label: Recency > %.0f days -> %d churned (%.1f%%).",
        percentile * 100, threshold_days, int(y.sum()), y.mean() * 100,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    rows = []
    for name, model in _build_models(pos_weight).items():
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "threshold_percentile": int(percentile * 100),
            "threshold_days": round(float(threshold_days), 0),
            "churn_rate_pct": round(float(y.mean() * 100), 1),
            "Model": name,
            "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
            "PR_AUC": round(average_precision_score(y_test, proba), 4),
            "F1": round(f1_score(y_test, pred, zero_division=0), 4),
        })
    return rows


def run_churn_threshold_sensitivity(
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score all six churn models at each candidate label threshold.

    Returns
    -------
    pd.DataFrame
        Long-format table (threshold x model), also written to
        ``outputs/tables/churn_threshold_sensitivity.csv``, with a
        per-threshold ROC-AUC rank column so ranking stability is explicit.
    """
    if features is None:
        logger.info("Loading customer features from %s", CUSTOMER_FEATURES_PARQUET)
        features = pd.read_parquet(CUSTOMER_FEATURES_PARQUET)

    rows: list[dict] = []
    for pct in THRESHOLDS:
        rows.extend(_evaluate_threshold(features, pct))
    results = pd.DataFrame(rows)

    results["ROC_AUC_rank"] = results.groupby("threshold_percentile")["ROC_AUC"] \
                                     .rank(ascending=False).astype(int)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(THRESHOLD_SENSITIVITY_CSV, index=False)
    logger.info("Churn threshold sensitivity saved to %s\n%s",
                THRESHOLD_SENSITIVITY_CSV, results.to_string(index=False))
    return results
