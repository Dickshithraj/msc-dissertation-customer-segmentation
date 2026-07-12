"""Tests for src.churn, focusing on the target-leakage guard and labelling."""

from __future__ import annotations

from src import churn
from src.churn import CHURN_FEATURES, _build_label


def test_recency_excluded_from_features():
    """The single most important property: Recency must not be a feature.

    Churn is defined *from* Recency, so including it would let any model
    trivially recover the threshold and report a meaningless ~1.0 ROC-AUC.
    """
    assert "Recency" not in CHURN_FEATURES
    # The remaining six behavioural features should all be present.
    assert set(CHURN_FEATURES) == {
        "Frequency", "Monetary", "Tenure",
        "AvgOrderValue", "AvgInterPurchaseDays", "DistinctProducts",
    }


def test_build_label_threshold(customer_features, monkeypatch):
    # Force a clean 90th-percentile split regardless of config default.
    monkeypatch.setattr(churn, "CHURN_RECENCY_PERCENTILE", 0.90)
    y = _build_label(customer_features)

    assert set(y.unique()) <= {0, 1}
    # ~10% should be churned at the 90th percentile (allow tolerance for ties).
    assert 0.05 <= y.mean() <= 0.15
    # Every churned customer must have Recency above the threshold.
    threshold = customer_features["Recency"].quantile(0.90)
    assert (customer_features.loc[y == 1, "Recency"] > threshold).all()
