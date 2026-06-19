"""
Cleaning module for the Online Retail II transaction data.

Each cleaning step is documented with the business or statistical reason for
the rule so the rationale is traceable back to this source file when writing
the methodology chapter.

Design choices
--------------
* Steps are applied sequentially so that each removal count is measured
  against the *already-filtered* dataset — this prevents double-counting rows
  that would have been removed by multiple rules.
* A summary DataFrame is built alongside the cleaning so the exact impact of
  every rule is recorded without requiring a second pass over the data.
* All thresholds come from ``src/config.py``; nothing is hard-coded here.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    CANCELLATION_PREFIX,
    CLEANED_PARQUET,
    DATA_PROCESSED_DIR,
    MIN_QUANTITY,
    MIN_UNIT_PRICE,
    OUTPUTS_TABLES_DIR,
)

logger = logging.getLogger(__name__)

# Output path for the row-removal audit table.
CLEANING_SUMMARY_CSV = OUTPUTS_TABLES_DIR / "cleaning_summary.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _record_step(
    summary_rows: list[dict],
    step: int,
    description: str,
    rows_before: int,
    rows_after: int,
) -> None:
    """Append one row to the running audit list."""
    removed = rows_before - rows_after
    pct = removed / rows_before * 100 if rows_before else 0.0
    summary_rows.append(
        {
            "Step": step,
            "Description": description,
            "Rows Before": rows_before,
            "Rows After": rows_after,
            "Rows Removed": removed,
            "% Removed": round(pct, 2),
        }
    )


def _print_summary(summary: pd.DataFrame) -> None:
    """Print a formatted audit table to stdout."""
    col_widths = {
        "Step": 4,
        "Description": 45,
        "Rows Before": 12,
        "Rows After": 11,
        "Rows Removed": 13,
        "% Removed": 10,
    }
    header = "  ".join(c.ljust(w) for c, w in col_widths.items())
    divider = "-" * len(header)
    print("\n" + divider)
    print("CLEANING SUMMARY")
    print(divider)
    print(header)
    print(divider)
    for _, row in summary.iterrows():
        line = "  ".join(
            str(row[c]).ljust(w) for c, w in col_widths.items()
        )
        print(line)
    print(divider)
    total_removed = summary["Rows Removed"].sum()
    initial = summary.iloc[0]["Rows Before"]
    final = summary.iloc[-1]["Rows After"]
    print(
        f"Total removed: {total_removed:,}  "
        f"({total_removed / initial * 100:.1f}% of raw data)  |  "
        f"Final dataset: {final:,} rows"
    )
    print(divider + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline to the raw combined transaction data.

    Each step below removes a distinct category of noise or invalid records.
    The docstring for each step explains *why* it is necessary so the
    rationale can be reproduced directly in the dissertation methodology.

    Parameters
    ----------
    df:
        Raw combined DataFrame produced by :func:`src.data_loading.load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame saved to ``data/processed/cleaned_transactions.parquet``
        and ready for feature engineering.

    Side effects
    ------------
    * Writes ``outputs/tables/cleaning_summary.csv``.
    * Prints a formatted summary table to stdout.
    * Saves a Parquet file to the path specified by ``config.CLEANED_PARQUET``.
    """
    summary_rows: list[dict] = []
    data = df.copy()

    # ------------------------------------------------------------------
    # Step 1 — Drop rows with missing Customer ID
    # ------------------------------------------------------------------
    # Customer ID is the primary key for all customer-level analyses (RFM,
    # CLV, segmentation). Rows without an ID cannot be attributed to any
    # customer and therefore contribute no signal to the model. Retaining
    # them would silently inflate basket-level aggregates while being
    # invisible in per-customer summaries, introducing an inconsistency
    # between transaction counts and customer counts.
    rows_before = len(data)
    data = data.dropna(subset=["Customer ID"])
    _record_step(summary_rows, 1, "Drop missing Customer ID", rows_before, len(data))

    # ------------------------------------------------------------------
    # Step 2 — Remove cancelled orders (Invoice starts with 'C')
    # ------------------------------------------------------------------
    # The UCI dataset encodes cancellations by prefixing the Invoice number
    # with the letter 'C'. Cancelled transactions record *negative* quantities
    # and represent reversed purchases that never generated revenue. Including
    # them would artificially deflate Monetary value in RFM and bias the
    # Frequency count, because a cancel + re-order pair would count as two
    # events rather than one. The matching original order is retained;
    # only the cancellation record is discarded.
    rows_before = len(data)
    is_cancellation = data["Invoice"].str.startswith(CANCELLATION_PREFIX)
    data = data[~is_cancellation]
    _record_step(
        summary_rows, 2, "Remove cancelled invoices (prefix 'C')", rows_before, len(data)
    )

    # ------------------------------------------------------------------
    # Step 3 — Remove non-positive Quantity or Price
    # ------------------------------------------------------------------
    # Quantity <= 0 that was not already removed by the cancellation filter
    # indicates data-entry errors, sample or test stock movements, or write-
    # offs — none of which represent genuine customer purchases. Price <= 0
    # (including free samples and manual adjustments with Price = 0) cannot
    # produce a valid Monetary value and would distort unit-price distributions
    # used in product-level profiling. Both conditions are filtered together
    # because they represent the same underlying problem: records that do not
    # correspond to a real sale at market price.
    rows_before = len(data)
    valid_qty = data["Quantity"] >= MIN_QUANTITY
    valid_price = data["Price"] >= MIN_UNIT_PRICE
    data = data[valid_qty & valid_price]
    _record_step(
        summary_rows,
        3,
        f"Remove Quantity < {MIN_QUANTITY} or Price < {MIN_UNIT_PRICE}",
        rows_before,
        len(data),
    )

    # ------------------------------------------------------------------
    # Step 4 — Drop exact duplicate rows
    # ------------------------------------------------------------------
    # Exact duplicates (all columns identical) are most likely the result of
    # double-loading one of the Excel sheets, ETL pipeline retries, or
    # upstream data-entry errors in the retailer's system. Keeping them would
    # inflate transaction counts and Monetary totals for affected customers,
    # biasing their RFM scores upward relative to customers whose records
    # contain no duplicates.
    rows_before = len(data)
    data = data.drop_duplicates()
    _record_step(summary_rows, 4, "Drop exact duplicate rows", rows_before, len(data))

    # ------------------------------------------------------------------
    # Step 5 — Derive TotalPrice = Quantity × Price
    # ------------------------------------------------------------------
    # TotalPrice is the line-level revenue figure used as the building block
    # for the Monetary dimension of RFM and for CLV estimation. Computing it
    # here — after all invalid rows have been removed — guarantees that no
    # negative or zero-revenue lines contaminate downstream aggregations.
    data["TotalPrice"] = data["Quantity"] * data["Price"]

    # ------------------------------------------------------------------
    # Step 6 — Enforce correct column dtypes
    # ------------------------------------------------------------------
    # InvoiceDate must be datetime64 for time-series grouping (Recency
    # calculation) and for BG/NBD model calibration which requires ordered
    # date arithmetic. Customer ID is cast to int after NaN removal; storing
    # it as float (the pandas default when NaNs are present) would silently
    # propagate floating-point representations into join keys, causing subtle
    # mismatches if the column is ever used in a merge with an integer-typed
    # key.
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["Customer ID"] = data["Customer ID"].astype(int)

    # ------------------------------------------------------------------
    # Build and persist the audit summary
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    _print_summary(summary_df)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(CLEANING_SUMMARY_CSV, index=False)
    logger.info("Cleaning summary saved to %s", CLEANING_SUMMARY_CSV)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(CLEANED_PARQUET, index=False)
    logger.info("Cleaned data (%d rows) saved to %s", len(data), CLEANED_PARQUET)

    return data
