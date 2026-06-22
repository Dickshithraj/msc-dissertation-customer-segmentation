"""
Phase 9: Year-on-year segment migration analysis.

The dataset spans two retail years ("2009-2010" and "2010-2011").  This module
segments each customer *independently within each year* using the same
rule-based RFM naming applied in :mod:`src.profiling`, then builds a transition
matrix describing how customers move between segments from one year to the next.

Why per-year rule-based segments (not the global clusters)?
-----------------------------------------------------------
The global clustering (Stage 3) was fitted on the full two-year window, so it
assigns each customer a single label and cannot express *movement*.  To measure
migration we need a comparable segment definition that can be recomputed on each
year's transactions.  The rule-based RFM namer is deterministic and
year-agnostic — "Champions" means the same thing in both years — so its labels
are directly comparable across time, which cluster IDs are not.

Each year's Recency is measured relative to that year's own snapshot date
(max InvoiceDate within the year + offset), so a customer active late in
2009-2010 is not penalised by the gap to the 2010-2011 window.

Outputs
-------
outputs/tables/segment_migration_counts.csv  -- raw transition counts
outputs/tables/segment_migration_rates.csv   -- row-normalised probabilities
outputs/figures/segment_migration.png        -- annotated heatmap
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CLEANED_PARQUET,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    RFM_SNAPSHOT_DATE_OFFSET_DAYS,
)
from src.features import _build_extended, _build_rfm
from src.profiling import _assign_name

logger = logging.getLogger(__name__)

MIGRATION_COUNTS_CSV = OUTPUTS_TABLES_DIR / "segment_migration_counts.csv"
MIGRATION_RATES_CSV = OUTPUTS_TABLES_DIR / "segment_migration_rates.csv"
MIGRATION_PNG = OUTPUTS_FIGURES_DIR / "segment_migration.png"

# Canonical ordering from most to least valuable, so the matrix diagonal reads
# top-left (best) to bottom-right (worst) and up/down-grades are easy to see.
SEGMENT_ORDER: list[str] = [
    "Champions",
    "Loyal Customers",
    "Big Spenders",
    "Potential Loyalists",
    "General Customers",
    "At Risk",
    "Lost Customers",
]


# ---------------------------------------------------------------------------
# Per-year segmentation
# ---------------------------------------------------------------------------

def _segment_year(year_df: pd.DataFrame, year_label: str) -> pd.DataFrame:
    """Compute per-customer RFM features and segment names for one year.

    Parameters
    ----------
    year_df:
        Cleaned transactions filtered to a single ``Year`` value.
    year_label:
        The year string (for logging only).

    Returns
    -------
    pd.DataFrame
        Indexed by Customer ID with a single ``segment`` column.
    """
    snapshot = year_df["InvoiceDate"].max() + pd.Timedelta(days=RFM_SNAPSHOT_DATE_OFFSET_DAYS)
    rfm = _build_rfm(year_df, snapshot)
    feats = _build_extended(year_df, rfm)

    overall = feats[["Recency", "Frequency", "Monetary"]].mean()
    feats["segment"] = feats.apply(lambda r: _assign_name(r, overall), axis=1)

    logger.info(
        "%s: %d customers segmented. Distribution:\n%s",
        year_label, len(feats),
        feats["segment"].value_counts().to_string(),
    )
    return feats.set_index("Customer ID")[["segment"]]


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def _build_transition(
    seg_y1: pd.DataFrame,
    seg_y2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build count and row-normalised transition matrices for shared customers.

    Parameters
    ----------
    seg_y1, seg_y2:
        Per-year segment tables indexed by Customer ID.

    Returns
    -------
    (counts, rates, context)
        ``counts``: transition count matrix (year-1 rows -> year-2 cols).
        ``rates``: each row divided by its sum (transition probabilities).
        ``context``: customer counts for retained / new / lapsed cohorts.
    """
    joined = seg_y1.join(seg_y2, how="inner", lsuffix="_y1", rsuffix="_y2")
    context = {
        "retained_both_years": len(joined),
        "year1_only_lapsed": len(seg_y1) - len(joined),
        "year2_only_new": len(seg_y2) - len(joined),
    }
    logger.info(
        "Migration cohort: %d retained, %d lapsed, %d new.",
        context["retained_both_years"], context["year1_only_lapsed"],
        context["year2_only_new"],
    )

    counts = pd.crosstab(joined["segment_y1"], joined["segment_y2"])
    # Enforce the canonical ordering on both axes, filling absent segments.
    counts = counts.reindex(index=SEGMENT_ORDER, columns=SEGMENT_ORDER, fill_value=0)

    row_sums = counts.sum(axis=1).replace(0, np.nan)
    rates = counts.div(row_sums, axis=0).fillna(0.0)

    return counts, rates, context


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_migration(counts: pd.DataFrame, rates: pd.DataFrame) -> None:
    """Save an annotated heatmap of transition probabilities.

    Colour encodes the row-normalised transition probability; each cell is
    annotated with the probability and the underlying customer count, so the
    figure conveys both the likelihood and the volume of each move.
    """
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(rates.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability (row-normalised)", fontsize=9)

    n = len(SEGMENT_ORDER)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SEGMENT_ORDER, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(SEGMENT_ORDER, fontsize=9)
    ax.set_xlabel("Year 2  (2010-2011) segment", fontsize=10, fontweight="bold")
    ax.set_ylabel("Year 1  (2009-2010) segment", fontsize=10, fontweight="bold")

    for i in range(n):
        for j in range(n):
            rate = rates.values[i, j]
            cnt = int(counts.values[i, j])
            if cnt == 0:
                continue
            txt_colour = "white" if rate > 0.5 else "black"
            ax.text(j, i, f"{rate*100:.0f}%\n(n={cnt})",
                    ha="center", va="center", fontsize=8, color=txt_colour)

    ax.set_title(
        "Year-on-year segment migration\n"
        "(row = origin segment in Y1; cell = P(move to Y2 segment))",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MIGRATION_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Migration heatmap saved to %s", MIGRATION_PNG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_migration(transactions: pd.DataFrame | None = None) -> dict:
    """Compute the year-on-year segment transition matrix.

    Parameters
    ----------
    transactions:
        Cleaned transactions with a ``Year`` column.  Pass ``None`` to load
        from ``data/processed/cleaned_transactions.parquet``.

    Returns
    -------
    dict
        ``{"counts": DataFrame, "rates": DataFrame, "context": dict}``.

    Side effects
    ------------
    * Writes the count and rate CSVs and the heatmap PNG.
    """
    if transactions is None:
        logger.info("Loading cleaned transactions from %s", CLEANED_PARQUET)
        transactions = pd.read_parquet(CLEANED_PARQUET)

    years = sorted(transactions["Year"].unique())
    if len(years) < 2:
        raise ValueError(f"Need 2 years for migration; found {years}.")
    y1_label, y2_label = years[0], years[1]
    logger.info("Migration between '%s' and '%s'.", y1_label, y2_label)

    seg_y1 = _segment_year(transactions[transactions["Year"] == y1_label], y1_label)
    seg_y2 = _segment_year(transactions[transactions["Year"] == y2_label], y2_label)

    counts, rates, context = _build_transition(seg_y1, seg_y2)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(MIGRATION_COUNTS_CSV)
    rates.round(4).to_csv(MIGRATION_RATES_CSV)
    logger.info("Migration counts saved to %s", MIGRATION_COUNTS_CSV)
    logger.info("Migration rates saved to %s", MIGRATION_RATES_CSV)

    _plot_migration(counts, rates)

    # Retention diagonal: fraction of each segment that stayed put.
    diag = {seg: float(rates.loc[seg, seg]) for seg in SEGMENT_ORDER
            if counts.loc[seg].sum() > 0}
    logger.info("Segment retention (stayed in same segment):\n%s",
                pd.Series(diag).round(3).to_string())

    return {"counts": counts, "rates": rates, "context": context}
