"""Tests for the pure RFM/extended feature computation in src.features."""

from __future__ import annotations

import pandas as pd

from src.features import _build_extended, _build_rfm


def test_build_rfm_values(clean_transactions_small):
    snapshot = pd.Timestamp("2011-01-12 10:00")
    rfm = _build_rfm(clean_transactions_small, snapshot).set_index("Customer ID")

    # Customer 1: last purchase 2011-01-11 -> recency 1 day; 2 invoices;
    # monetary = 2*10 + 1*30 = 50.
    assert rfm.loc[1, "Recency"] == 1
    assert rfm.loc[1, "Frequency"] == 2
    assert rfm.loc[1, "Monetary"] == 50.0

    # Customer 2: last purchase 2011-01-06 -> recency 6 days; 1 invoice;
    # monetary = 5*10 = 50.
    assert rfm.loc[2, "Recency"] == 6
    assert rfm.loc[2, "Frequency"] == 1
    assert rfm.loc[2, "Monetary"] == 50.0


def test_build_extended_derivations(clean_transactions_small):
    snapshot = pd.Timestamp("2011-01-12 10:00")
    rfm = _build_rfm(clean_transactions_small, snapshot)
    feats = _build_extended(clean_transactions_small, rfm).set_index("Customer ID")

    # Customer 1: tenure = 10 days (01-01 to 01-11), 2 distinct products,
    # AvgOrderValue = 50/2 = 25, AvgInterPurchaseDays = 10/2 = 5.
    assert feats.loc[1, "Tenure"] == 10
    assert feats.loc[1, "DistinctProducts"] == 2
    assert feats.loc[1, "AvgOrderValue"] == 25.0
    assert feats.loc[1, "AvgInterPurchaseDays"] == 5.0

    # Customer 2: one-time buyer -> tenure 0, AvgInterPurchaseDays 0.
    assert feats.loc[2, "Tenure"] == 0
    assert feats.loc[2, "AvgInterPurchaseDays"] == 0.0

    # Column order / completeness.
    assert list(feats.columns) == [
        "Recency", "Frequency", "Monetary", "Tenure",
        "AvgOrderValue", "AvgInterPurchaseDays", "DistinctProducts",
    ]
