"""Tests for segment naming and the migration segment ordering."""

from __future__ import annotations

import pandas as pd

from src.migration import SEGMENT_ORDER
from src.profiling import _assign_name


def test_segment_order_is_complete_and_unique():
    assert len(SEGMENT_ORDER) == len(set(SEGMENT_ORDER))
    # Canonical best -> worst ordering starts with Champions, ends with Lost.
    assert SEGMENT_ORDER[0] == "Champions"
    assert SEGMENT_ORDER[-1] == "Lost Customers"


def test_assign_name_champions():
    overall = pd.Series({"Recency": 100.0, "Frequency": 5.0, "Monetary": 1000.0})
    # Very recent, very frequent, very high spend -> Champions.
    row = pd.Series({"Recency": 10.0, "Frequency": 20.0, "Monetary": 5000.0})
    assert _assign_name(row, overall) == "Champions"


def test_assign_name_lost():
    overall = pd.Series({"Recency": 100.0, "Frequency": 5.0, "Monetary": 1000.0})
    # Long dormant, low frequency -> Lost Customers.
    row = pd.Series({"Recency": 400.0, "Frequency": 1.0, "Monetary": 50.0})
    assert _assign_name(row, overall) == "Lost Customers"
