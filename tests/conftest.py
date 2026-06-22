"""Shared pytest fixtures: small synthetic datasets for fast, isolated tests.

These fixtures deliberately avoid the real ~1M-row Excel file so the suite runs
in seconds and does not depend on any pipeline artefact existing on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """A tiny raw transaction frame exercising every cleaning rule.

    Contains: a missing Customer ID, a cancellation (Invoice 'C...'), a
    non-positive quantity, a zero price, and an exact duplicate row, plus a
    handful of valid rows across two customers and two years.
    """
    rows = [
        # valid rows, customer 1, year 2009-2010
        ("489001", "A", "Widget", 2, "2010-01-05 10:00", 5.0, 1.0, "UK", "2009-2010"),
        ("489002", "B", "Gadget", 1, "2010-03-10 11:00", 20.0, 1.0, "UK", "2009-2010"),
        # valid rows, customer 2, year 2010-2011
        ("489003", "A", "Widget", 3, "2011-02-01 09:00", 5.0, 2.0, "UK", "2010-2011"),
        ("489004", "C", "Thing", 1, "2011-06-15 14:00", 50.0, 2.0, "UK", "2010-2011"),
        # missing Customer ID -> dropped by step 1
        ("489005", "A", "Widget", 1, "2011-07-01 10:00", 5.0, np.nan, "UK", "2010-2011"),
        # cancellation -> dropped by step 2
        ("C489006", "A", "Widget", -2, "2011-07-02 10:00", 5.0, 1.0, "UK", "2010-2011"),
        # non-positive quantity -> dropped by step 3
        ("489007", "A", "Widget", 0, "2011-07-03 10:00", 5.0, 1.0, "UK", "2010-2011"),
        # zero price -> dropped by step 3
        ("489008", "A", "Freebie", 1, "2011-07-04 10:00", 0.0, 1.0, "UK", "2010-2011"),
        # exact duplicate of first valid row -> dropped by step 4
        ("489001", "A", "Widget", 2, "2010-01-05 10:00", 5.0, 1.0, "UK", "2009-2010"),
    ]
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
            "Price", "Customer ID", "Country", "Year"]
    df = pd.DataFrame(rows, columns=cols)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@pytest.fixture
def clean_transactions_small() -> pd.DataFrame:
    """A clean, multi-purchase transaction frame for feature-engineering tests."""
    rows = [
        # Customer 1: two invoices, two products
        ("1001", "A", "Widget", 2, "2011-01-01 10:00", 10.0, 1, "UK", "2010-2011"),
        ("1002", "B", "Gadget", 1, "2011-01-11 10:00", 30.0, 1, "UK", "2010-2011"),
        # Customer 2: single invoice (one-time buyer)
        ("1003", "A", "Widget", 5, "2011-01-06 10:00", 10.0, 2, "UK", "2010-2011"),
    ]
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
            "Price", "Customer ID", "Country", "Year"]
    df = pd.DataFrame(rows, columns=cols)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df


@pytest.fixture
def customer_features() -> pd.DataFrame:
    """A synthetic customer feature table with a clear skew, for preprocessing."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "Customer ID": np.arange(1, n + 1),
        "Recency": rng.integers(1, 700, n),
        "Frequency": rng.integers(1, 50, n),
        "Monetary": rng.exponential(1000, n),           # heavily right-skewed
        "Tenure": rng.integers(0, 700, n),
        "AvgOrderValue": rng.exponential(200, n),
        "AvgInterPurchaseDays": rng.integers(0, 200, n),
        "DistinctProducts": rng.integers(1, 300, n),
    })
