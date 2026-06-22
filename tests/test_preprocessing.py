"""Tests for src.preprocessing skew detection and scaling behaviour."""

from __future__ import annotations

import numpy as np

from src.preprocessing import FEATURE_COLS, _identify_log1p_cols


def test_log1p_selects_skewed_columns(customer_features):
    selected = _identify_log1p_cols(customer_features, FEATURE_COLS)
    # Monetary and AvgOrderValue are exponential -> strongly right-skewed.
    assert "Monetary" in selected
    assert "AvgOrderValue" in selected


def test_feature_cols_exclude_customer_id():
    assert "Customer ID" not in FEATURE_COLS
    assert len(FEATURE_COLS) == 7


def test_standardisation_property(customer_features):
    from sklearn.preprocessing import StandardScaler

    X = customer_features[FEATURE_COLS].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    # Each column ~ zero mean, unit variance.
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-9)
