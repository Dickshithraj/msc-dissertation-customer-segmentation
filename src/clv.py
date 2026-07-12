"""
Phase 7: Customer Lifetime Value (CLV) via BG/NBD + Gamma-Gamma.

Two complementary probabilistic models from the ``lifetimes`` package are
combined to estimate forward-looking customer value:

BG/NBD (Beta-Geometric / Negative-Binomial Distribution)
    Models the *purchasing process*: how many transactions a customer is
    expected to make in a future window, and the probability they are still
    "alive" (have not silently churned).  Fitted on three RFM-style summary
    statistics per customer:
      - frequency: number of *repeat* purchases (total distinct purchase
        occasions minus one),
      - recency:   age of the customer at their last purchase (in the chosen
        time unit, here days),
      - T:         age of the customer (time between first purchase and the
        observation period end).

Gamma-Gamma
    Models the *monetary process*: the expected average transaction value per
    customer, conditioned on their observed average.  It assumes monetary
    value is independent of purchase frequency (verified by a low correlation
    check, logged as a diagnostic).  Fitted only on returning customers
    (frequency > 0) with positive monetary value.

The two are combined by :meth:`GammaGammaFitter.customer_lifetime_value`,
which discounts expected future cash flows (DCF) over a horizon of
``CLV_TIME_MONTHS`` months at ``CLV_DISCOUNT_RATE_MONTHLY``.

Outputs
-------
data/processed/customer_clv.parquet  -- per-customer CLV + forecast columns
outputs/tables/clv_summary.csv       -- describe() of the CLV columns
outputs/figures/clv_distribution.png -- CLV histogram + predicted-purchases plot
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

from src.config import (
    BGMD_PENALIZER_COEF,
    CLEANED_PARQUET,
    CLV_DISCOUNT_RATE_MONTHLY,
    CLV_FORECAST_DAYS,
    CLV_TIME_MONTHS,
    CUSTOMER_CLV_PARQUET,
    CUSTOMER_FEATURES_PARQUET,
    GAMMA_GAMMA_PENALIZER_COEF,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
)

logger = logging.getLogger(__name__)

CLV_SUMMARY_CSV = OUTPUTS_TABLES_DIR / "clv_summary.csv"
CLV_DIST_PNG = OUTPUTS_FIGURES_DIR / "clv_distribution.png"


# ---------------------------------------------------------------------------
# RFM summary for the lifetimes models
# ---------------------------------------------------------------------------

def _build_clv_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into the frequency/recency/T/monetary summary.

    Uses ``InvoiceDate`` as the timestamp and ``Customer ID`` as the customer
    key.  Multiple line items sharing an invoice on the same day count as a
    single purchase occasion (``freq="D"`` deduplicates within-day rows).

    Parameters
    ----------
    transactions:
        Cleaned transaction rows with ``Customer ID``, ``InvoiceDate``, and
        ``TotalPrice`` columns.

    Returns
    -------
    pd.DataFrame
        Indexed by Customer ID with columns ``frequency``, ``recency``, ``T``,
        and ``monetary_value`` (mean transaction value of repeat purchases).
    """
    obs_end = transactions["InvoiceDate"].max()
    logger.info("CLV observation period end: %s", obs_end)

    summary = summary_data_from_transaction_data(
        transactions,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=obs_end,
        freq="D",
    )
    logger.info(
        "CLV summary built for %d customers (%.1f%% are repeat buyers).",
        len(summary), (summary["frequency"] > 0).mean() * 100,
    )
    return summary


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _fit_bgnbd(summary: pd.DataFrame) -> BetaGeoFitter:
    """Fit the BG/NBD model on frequency, recency, and T."""
    bgf = BetaGeoFitter(penalizer_coef=BGMD_PENALIZER_COEF)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    logger.info("BG/NBD fitted. Params:\n%s", bgf.summary.to_string())
    return bgf


def _fit_gamma_gamma(returning: pd.DataFrame) -> GammaGammaFitter:
    """Fit the Gamma-Gamma model on returning customers' monetary values.

    Logs the frequency/monetary correlation as an independence diagnostic:
    the Gamma-Gamma model assumes these are uncorrelated, so a value near 0
    supports the model assumption.
    """
    corr = returning["frequency"].corr(returning["monetary_value"])
    logger.info(
        "Frequency/monetary correlation = %.4f (Gamma-Gamma assumes ~0).", corr,
    )
    ggf = GammaGammaFitter(penalizer_coef=GAMMA_GAMMA_PENALIZER_COEF)
    ggf.fit(returning["frequency"], returning["monetary_value"])
    logger.info("Gamma-Gamma fitted. Params:\n%s", ggf.summary.to_string())
    return ggf


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _predict_clv(
    summary: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
) -> pd.DataFrame:
    """Assemble per-customer CLV, expected value, and purchase forecasts.

    Returns a copy of ``summary`` augmented with:
      - ``prob_alive``           : P(customer still active)
      - ``pred_purchases_{d}d``  : expected purchase count over each horizon in
                                   ``CLV_FORECAST_DAYS``
      - ``exp_avg_value``        : Gamma-Gamma expected average transaction value
      - ``clv``                  : discounted CLV over ``CLV_TIME_MONTHS`` months
    """
    out = summary.copy()

    out["prob_alive"] = bgf.conditional_probability_alive(
        out["frequency"], out["recency"], out["T"],
    )

    for days in CLV_FORECAST_DAYS:
        out[f"pred_purchases_{days}d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            days, out["frequency"], out["recency"], out["T"],
        )

    # Gamma-Gamma expected value is only valid for returning customers.
    # When the fitted q parameter is < 1 (heavy-tailed monetary distribution),
    # the population baseline p*v/(q-1) is negative and contaminates one-time
    # buyers (frequency == 0, monetary == 0).  Conditional estimates for active
    # repeat customers remain valid; we clip the degenerate baseline at 0 so a
    # customer never carries a negative expected spend.
    q_param = float(ggf.params_["q"])
    if q_param <= 1:
        logger.warning(
            "Gamma-Gamma q=%.3f <= 1: population-baseline expected value is "
            "unreliable for non-returning customers; clipping at 0.", q_param,
        )
    out["exp_avg_value"] = ggf.conditional_expected_average_profit(
        out["frequency"], out["monetary_value"],
    ).clip(lower=0.0)

    # Discounted CLV over the configured horizon. customer_lifetime_value
    # expects the BG/NBD model plus the same summary inputs; monetary_value
    # must be positive, so non-returning / zero-value rows are clipped.
    safe_monetary = out["monetary_value"].clip(lower=0.0)
    out["clv"] = ggf.customer_lifetime_value(
        bgf,
        out["frequency"],
        out["recency"],
        out["T"],
        safe_monetary,
        time=CLV_TIME_MONTHS,
        freq="D",
        discount_rate=CLV_DISCOUNT_RATE_MONTHLY,
    )

    # CLV can be NaN for degenerate rows (zero monetary, zero frequency).
    out["clv"] = out["clv"].fillna(0.0).clip(lower=0.0)
    return out


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_clv(clv_df: pd.DataFrame) -> None:
    """Save a 2-panel figure: CLV distribution and 365-day purchase forecast."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: CLV distribution (clip the long tail at the 99th percentile
    # for readability; annotate how many customers fall beyond it).
    clv = clv_df["clv"]
    cap = clv.quantile(0.99)
    beyond = int((clv > cap).sum())
    axes[0].hist(clv.clip(upper=cap), bins=60, color="#4C72B0",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(clv.median(), color="#C44E52", linestyle="--",
                    linewidth=1.6, label=f"Median = {clv.median():,.0f}")
    axes[0].set_xlabel(f"Predicted {CLV_TIME_MONTHS}-month CLV (currency)", fontsize=10)
    axes[0].set_ylabel("Number of customers", fontsize=10)
    axes[0].set_title(
        f"CLV distribution\n(x-axis capped at 99th pct; {beyond:,} customers beyond)",
        fontsize=11, fontweight="bold",
    )
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Panel 2: predicted 365-day purchases vs probability alive.
    longest = max(CLV_FORECAST_DAYS)
    col = f"pred_purchases_{longest}d"
    sc = axes[1].scatter(
        clv_df[col], clv_df["prob_alive"],
        c=clv_df["clv"].clip(upper=cap), cmap="viridis",
        s=14, alpha=0.6,
    )
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label("CLV (capped)", fontsize=9)
    axes[1].set_xlabel(f"Expected purchases in next {longest} days", fontsize=10)
    axes[1].set_ylabel("P(customer still alive)", fontsize=10)
    axes[1].set_title("Engagement vs retention\n(colour = CLV)",
                      fontsize=11, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CLV_DIST_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("CLV distribution plot saved to %s", CLV_DIST_PNG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_clv(
    transactions: pd.DataFrame | None = None,
    merge_features: bool = True,
) -> pd.DataFrame:
    """Fit BG/NBD + Gamma-Gamma and produce a per-customer CLV table.

    Parameters
    ----------
    transactions:
        Cleaned transaction rows.  Pass ``None`` to load from
        ``data/processed/cleaned_transactions.parquet``.
    merge_features:
        If ``True`` (default), left-merge the CLV columns onto the customer
        feature table so the result carries RFM + extended features + CLV.

    Returns
    -------
    pd.DataFrame
        Per-customer table including ``clv``, ``prob_alive``,
        ``exp_avg_value``, and ``pred_purchases_{d}d`` columns.  Saved to
        ``data/processed/customer_clv.parquet``.
    """
    if transactions is None:
        logger.info("Loading cleaned transactions from %s", CLEANED_PARQUET)
        transactions = pd.read_parquet(CLEANED_PARQUET)

    summary = _build_clv_summary(transactions)

    bgf = _fit_bgnbd(summary)
    returning = summary[summary["frequency"] > 0].copy()
    ggf = _fit_gamma_gamma(returning)

    clv_df = _predict_clv(summary, bgf, ggf)
    clv_df = clv_df.reset_index().rename(columns={"index": "Customer ID"})

    # Some lifetimes versions name the index column 'Customer ID' already.
    if "Customer ID" not in clv_df.columns and "CustomerID" in clv_df.columns:
        clv_df = clv_df.rename(columns={"CustomerID": "Customer ID"})

    if merge_features:
        logger.info("Loading customer features from %s", CUSTOMER_FEATURES_PARQUET)
        feats = pd.read_parquet(CUSTOMER_FEATURES_PARQUET)
        clv_df = feats.merge(clv_df, on="Customer ID", how="left", suffixes=("", "_clv"))

    # ── Persist ────────────────────────────────────────────────────────────
    CUSTOMER_CLV_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    clv_df.to_parquet(CUSTOMER_CLV_PARQUET, index=False)
    logger.info("Customer CLV table (%d rows) saved to %s",
                len(clv_df), CUSTOMER_CLV_PARQUET)

    clv_cols = ["clv", "prob_alive", "exp_avg_value"] + \
               [f"pred_purchases_{d}d" for d in CLV_FORECAST_DAYS]
    summary_stats = clv_df[clv_cols].describe()
    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_stats.to_csv(CLV_SUMMARY_CSV)
    logger.info("CLV summary saved to %s\n%s", CLV_SUMMARY_CSV,
                summary_stats.to_string())

    _plot_clv(clv_df)

    # Headline figures for the dissertation narrative.
    total_clv = clv_df["clv"].sum()
    top_decile = clv_df["clv"].quantile(0.90)
    top_share = clv_df.loc[clv_df["clv"] >= top_decile, "clv"].sum() / total_clv * 100
    logger.info(
        "Total portfolio CLV = %s | top-decile customers hold %.1f%% of it.",
        f"{total_clv:,.0f}", top_share,
    )

    return clv_df
