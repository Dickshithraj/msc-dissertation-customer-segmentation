"""
Customer-level feature engineering for the segmentation pipeline.

Transforms the cleaned line-item transaction table into one row per customer
containing eight behavioural features. The first three are the canonical RFM
dimensions; the remaining four extend RFM with purchasing-pattern signals that
help distinguish, for example, a high-spend one-off buyer from a frequent
low-spend loyal customer — a distinction that raw Monetary alone cannot make.

All intermediate aggregations are kept as separate named steps so each can be
unit-tested and so the derivation chain is transparent to a dissertation reader.

Snapshot date convention
------------------------
The snapshot date is set to ``max(InvoiceDate) + RFM_SNAPSHOT_DATE_OFFSET_DAYS``
rather than to today's date. This makes the Recency values fully reproducible
regardless of when the code is run — an essential property for an academic
project where results must be exactly replicable from the raw data file.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    CLEANED_PARQUET,
    CUSTOMER_FEATURES_PARQUET,
    DATA_PROCESSED_DIR,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    RFM_SNAPSHOT_DATE_OFFSET_DAYS,
)

logger = logging.getLogger(__name__)

FEATURE_SUMMARY_CSV = OUTPUTS_TABLES_DIR / "feature_summary.csv"
RFM_DISTRIBUTIONS_PNG = OUTPUTS_FIGURES_DIR / "rfm_distributions.png"


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Return the analysis snapshot date: max InvoiceDate + configured offset.

    Using a fixed offset from the last observed transaction rather than the
    current calendar date ensures that Recency values are identical on every
    run, satisfying the reproducibility requirement of the dissertation.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame with a ``datetime64`` InvoiceDate column.
    """
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=RFM_SNAPSHOT_DATE_OFFSET_DAYS)
    logger.info("Snapshot date: %s", snapshot.date())
    return snapshot


def _build_rfm(df: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Compute the three canonical RFM dimensions per customer.

    Recency
        Days between the customer's most recent purchase and the snapshot date.
        Lower values indicate more recently active customers. Recency is used
        as an inverse engagement signal: a customer who bought yesterday is more
        likely to respond to a marketing message than one who bought two years
        ago. Using elapsed days (rather than the raw date) converts the temporal
        dimension into a numeric feature compatible with distance-based
        clustering algorithms.

    Frequency
        Count of *unique* invoices attributed to the customer. Each invoice
        represents a distinct shopping session, so this measures behavioural
        engagement rather than product volume. Counting invoices rather than
        line items prevents customers who purchase many SKUs per basket from
        being artificially inflated relative to those who buy fewer products
        per visit.

    Monetary
        Sum of TotalPrice across all line items for the customer. This is the
        total revenue attributed to that customer over the observation window
        and serves as the primary value signal in the RFM framework. Using the
        sum (rather than the mean) captures the full economic contribution,
        which is appropriate for CLV-oriented segmentation.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame.
    snapshot:
        Analysis snapshot date produced by :func:`_compute_snapshot_date`.

    Returns
    -------
    pd.DataFrame
        Customer-indexed DataFrame with columns Recency, Frequency, Monetary.
    """
    rfm = (
        df.groupby("Customer ID")
        .agg(
            _last_purchase=("InvoiceDate", "max"),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    rfm["Recency"] = (snapshot - rfm["_last_purchase"]).dt.days
    rfm = rfm.drop(columns=["_last_purchase"])
    return rfm


def _build_extended(df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """Append four extended behavioural features to the RFM base table.

    Tenure
        Number of days between a customer's first and last recorded purchase.
        Tenure distinguishes long-standing loyal customers from recent high-
        spenders who have the same Recency but fundamentally different
        relationship histories with the retailer. It also acts as the
        denominator for AvgInterPurchaseDays, making that feature meaningful.

    AvgOrderValue
        Monetary divided by Frequency (mean revenue per shopping session).
        Two customers with identical Monetary can have very different purchasing
        styles: one may make many small purchases, the other a few large ones.
        AvgOrderValue exposes this distinction and is used in segment profiling
        to identify premium vs. volume buyers.

    AvgInterPurchaseDays
        Tenure divided by Frequency (mean gap in days between purchases).
        This is a proxy for purchase cadence and is particularly useful for
        identifying customers whose buying rhythm could inform the timing of
        marketing notifications. For one-time buyers (Frequency = 1) the value
        is zero, reflecting the absence of a repeat-purchase interval; these
        customers form a distinct strategic group regardless.

    DistinctProducts
        Count of unique StockCodes purchased by the customer across all
        transactions. Breadth of assortment signals cross-category engagement
        and distinguishes niche enthusiasts (low DistinctProducts, high
        Monetary) from exploratory shoppers (high DistinctProducts, moderate
        Monetary). This is not captured by any standard RFM dimension.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame.
    rfm:
        Base RFM table produced by :func:`_build_rfm`.

    Returns
    -------
    pd.DataFrame
        Combined feature table with all eight columns.
    """
    extras = (
        df.groupby("Customer ID")
        .agg(
            _first_purchase=("InvoiceDate", "min"),
            _last_purchase=("InvoiceDate", "max"),
            DistinctProducts=("StockCode", "nunique"),
        )
        .reset_index()
    )
    extras["Tenure"] = (extras["_last_purchase"] - extras["_first_purchase"]).dt.days
    extras = extras.drop(columns=["_first_purchase", "_last_purchase"])

    features = rfm.merge(extras, on="Customer ID", how="left")
    features["AvgOrderValue"] = features["Monetary"] / features["Frequency"]
    # For customers with Frequency == 1, Tenure is 0 → AvgInterPurchaseDays = 0.
    # This is intentional: one-time buyers have no inter-purchase interval.
    features["AvgInterPurchaseDays"] = features["Tenure"] / features["Frequency"]

    col_order = [
        "Customer ID",
        "Recency",
        "Frequency",
        "Monetary",
        "Tenure",
        "AvgOrderValue",
        "AvgInterPurchaseDays",
        "DistinctProducts",
    ]
    return features[col_order]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_rfm_distributions(features: pd.DataFrame) -> None:
    """Save a three-panel histogram of the R, F, M distributions.

    The distributions are plotted on a log-scale x-axis for Frequency and
    Monetary because both are heavily right-skewed (a small number of
    customers account for a disproportionate share of transactions and
    revenue). Log-scaling keeps the bulk of the distribution visible rather
    than compressing it against the y-axis. Recency is plotted on a linear
    scale because it is naturally bounded and closer to uniform.

    The figure is saved to ``outputs/figures/rfm_distributions.png`` and is
    not displayed interactively (the ``Agg`` backend is used so the pipeline
    can run headlessly on a server).

    Parameters
    ----------
    features:
        Customer feature table containing at least Recency, Frequency,
        and Monetary columns.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("RFM Feature Distributions", fontsize=13, fontweight="bold", y=1.01)

    panels = [
        ("Recency", "Days since last purchase", "steelblue", False),
        ("Frequency", "Number of unique invoices", "darkorange", True),
        ("Monetary", "Total spend (£)", "forestgreen", True),
    ]

    for ax, (col, xlabel, colour, log_scale) in zip(axes, panels):
        data = features[col].dropna()
        ax.hist(data, bins=60, color=colour, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Number of customers", fontsize=10)
        ax.set_title(col, fontsize=11, fontweight="bold")
        if log_scale:
            ax.set_xscale("log")
            ax.set_xlabel(f"{xlabel} (log scale)", fontsize=10)
        median_val = data.median()
        ax.axvline(median_val, color="red", linestyle="--", linewidth=1.2,
                   label=f"Median: {median_val:,.1f}")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RFM_DISTRIBUTIONS_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("RFM distribution plot saved to %s", RFM_DISTRIBUTIONS_PNG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_customer_features() -> pd.DataFrame:
    """Build the full customer-level feature table from cleaned transactions.

    Reads the cleaned parquet artefact produced by :mod:`src.cleaning`,
    computes all eight features, and writes the result to
    ``data/processed/customer_features.parquet``.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        Customer ID, Recency, Frequency, Monetary, Tenure,
        AvgOrderValue, AvgInterPurchaseDays, DistinctProducts.

    Side effects
    ------------
    * Writes ``data/processed/customer_features.parquet``.
    * Writes ``outputs/tables/feature_summary.csv`` (describe() output).
    * Writes ``outputs/figures/rfm_distributions.png``.
    """
    logger.info("Loading cleaned transactions from %s", CLEANED_PARQUET)
    df = pd.read_parquet(CLEANED_PARQUET)
    logger.info("  → %d transaction rows, %d unique customers.",
                len(df), df["Customer ID"].nunique())

    snapshot = _compute_snapshot_date(df)
    rfm = _build_rfm(df, snapshot)
    features = _build_extended(df, rfm)

    logger.info("Feature table shape: %s", features.shape)

    # ── Persist outputs ────────────────────────────────────────────────────
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(CUSTOMER_FEATURES_PARQUET, index=False)
    logger.info("Customer features saved to %s", CUSTOMER_FEATURES_PARQUET)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = features.describe().T
    summary.to_csv(FEATURE_SUMMARY_CSV)
    logger.info("Feature summary saved to %s", FEATURE_SUMMARY_CSV)

    _plot_rfm_distributions(features)

    return features
