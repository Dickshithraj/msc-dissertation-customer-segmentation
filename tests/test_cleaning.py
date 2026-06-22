"""Tests for src.cleaning.clean_transactions.

Output paths are redirected to a temporary directory via monkeypatch so the
test never overwrites real pipeline artefacts.
"""

from __future__ import annotations

import pandas as pd

from src import cleaning


def test_clean_transactions_applies_every_rule(raw_transactions, tmp_path, monkeypatch):
    monkeypatch.setattr(cleaning, "CLEANED_PARQUET", tmp_path / "cleaned.parquet")
    monkeypatch.setattr(cleaning, "CLEANING_SUMMARY_CSV", tmp_path / "summary.csv")
    monkeypatch.setattr(cleaning, "DATA_PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(cleaning, "OUTPUTS_TABLES_DIR", tmp_path)

    cleaned = cleaning.clean_transactions(raw_transactions)

    # 9 raw rows -> 4 valid (1 missing ID, 1 cancel, 1 zero-qty, 1 zero-price,
    # 1 duplicate removed).
    assert len(cleaned) == 4
    # No missing IDs, no cancellations, all positive qty/price.
    assert cleaned["Customer ID"].notna().all()
    assert not cleaned["Invoice"].str.startswith("C").any()
    assert (cleaned["Quantity"] >= 1).all()
    assert (cleaned["Price"] > 0).all()
    # TotalPrice derived and correct.
    assert "TotalPrice" in cleaned.columns
    assert (cleaned["TotalPrice"] == cleaned["Quantity"] * cleaned["Price"]).all()
    # Customer ID cast to integer dtype.
    assert pd.api.types.is_integer_dtype(cleaned["Customer ID"])


def test_cleaning_summary_written(raw_transactions, tmp_path, monkeypatch):
    monkeypatch.setattr(cleaning, "CLEANED_PARQUET", tmp_path / "cleaned.parquet")
    monkeypatch.setattr(cleaning, "CLEANING_SUMMARY_CSV", tmp_path / "summary.csv")
    monkeypatch.setattr(cleaning, "DATA_PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(cleaning, "OUTPUTS_TABLES_DIR", tmp_path)

    cleaning.clean_transactions(raw_transactions)

    summary = pd.read_csv(tmp_path / "summary.csv")
    assert list(summary["Step"]) == [1, 2, 3, 4]
    assert summary["Rows Removed"].sum() == 5
