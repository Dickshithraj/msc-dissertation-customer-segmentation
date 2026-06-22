"""
Phase 11: Monte Carlo ROI simulation for the notification plan.

The Phase 10 plan assigns every customer a campaign, channel, and offer, but
the *financial return* of executing it is uncertain: response rates, the
incremental revenue per conversion, and (to a lesser extent) per-contact costs
are all unknown in advance.  This module propagates that uncertainty through a
Monte Carlo simulation (``ROI_N_SIMULATIONS`` iterations) to produce a
distribution of campaign ROI rather than a single fragile point estimate.

Model
-----
Customers are grouped by campaign ``action``.  For each group *g* with
``N_g`` customers, the simulation draws, on every iteration:

1. **Response rate** ``p_g ~ Beta(a, b)`` where the Beta is centred on an
   action-specific prior mean (see ``_RESPONSE_PRIORS``) with concentration
   ``ROI_RESPONSE_CONCENTRATION``.  Targeted, high-touch campaigns (e.g.
   priority retention) have higher prior response than mass campaigns
   (e.g. newsletters).
2. **Conversions** ``c_g ~ Binomial(N_g, p_g)`` — the count of customers who
   act on the campaign.
3. **Revenue** ``c_g * AOV_g * m`` where ``AOV_g`` is the group's mean average
   order value (the incremental revenue of one extra order) and
   ``m ~ Normal(1, ROI_REVENUE_NOISE_SD)`` (clipped > 0) captures basket-size
   uncertainty.

Costs are treated as effectively known: ``cost_g = N_g * channel_cost_g``,
where ``channel_cost_g`` sums the per-contact costs of the channels used.

Per iteration the totals are aggregated across all groups and:

    ROI = (total_revenue - total_cost) / total_cost

The output reports the mean/median ROI, a central ``ROI_CI_LEVEL`` credible
interval, the probability of a positive ROI, and expected net profit.

Outputs
-------
outputs/tables/roi_simulation_summary.csv -- headline statistics
outputs/figures/roi_distribution.png      -- ROI + net-profit histograms
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CUSTOMER_CLV_PARQUET,
    NOTIFICATION_PLAN_CSV,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    RANDOM_STATE,
    ROI_CHANNEL_COSTS,
    ROI_CI_LEVEL,
    ROI_GROSS_MARGIN,
    ROI_N_SIMULATIONS,
    ROI_OFFER_DISCOUNT,
    ROI_RESPONSE_CONCENTRATION,
    ROI_REVENUE_NOISE_SD,
)

logger = logging.getLogger(__name__)

ROI_SUMMARY_CSV = OUTPUTS_TABLES_DIR / "roi_simulation_summary.csv"
ROI_DIST_PNG = OUTPUTS_FIGURES_DIR / "roi_distribution.png"

# Prior mean response rate for each campaign action.  These are documented
# planning assumptions, not fitted values; they are deliberately conservative
# and ordered by how targeted / high-intent each campaign is.
_RESPONSE_PRIORS: dict[str, float] = {
    "Priority retention intervention": 0.30,
    "VIP loyalty reward + early access": 0.25,
    "Premium product recommendations": 0.20,
    "Cross-sell + loyalty tier upgrade": 0.15,
    "Win-back outreach": 0.12,
    "Frequency-building nudge": 0.10,
    "Engagement / category promotion": 0.08,
    "Reactivation campaign": 0.05,
    "Low-cost automated reactivation": 0.03,
    "Standard newsletter": 0.02,
}
_DEFAULT_RESPONSE = 0.05


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _channel_cost(channel_str: str) -> float:
    """Sum the per-contact costs of every channel named in ``channel_str``."""
    return sum(cost for name, cost in ROI_CHANNEL_COSTS.items()
               if name in channel_str)


def _build_campaign_table(
    plan: pd.DataFrame,
    clv: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the plan into per-campaign rows for the simulation.

    Returns a DataFrame indexed by action with columns: ``n_customers``,
    ``mean_aov`` (incremental revenue per conversion), ``cost_per_contact``,
    ``response_prior``, and ``total_cost``.
    """
    merged = plan.merge(
        clv[["Customer ID", "AvgOrderValue"]], on="Customer ID", how="left",
    )
    merged["cost_per_contact"] = merged["channel"].apply(_channel_cost)

    grp = merged.groupby("action").agg(
        n_customers=("Customer ID", "size"),
        mean_aov=("AvgOrderValue", "mean"),
        cost_per_contact=("cost_per_contact", "mean"),
    )
    grp["response_prior"] = [
        _RESPONSE_PRIORS.get(a, _DEFAULT_RESPONSE) for a in grp.index
    ]
    grp["total_cost"] = grp["n_customers"] * grp["cost_per_contact"]
    logger.info("Campaign table:\n%s", grp.round(3).to_string())
    return grp


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _simulate(campaigns: pd.DataFrame) -> dict[str, np.ndarray]:
    """Run the vectorised Monte Carlo simulation.

    Returns
    -------
    dict
        Arrays of length ``ROI_N_SIMULATIONS``: ``roi``, ``revenue``,
        ``cost`` (scalar-broadcast), ``profit``.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_iter = ROI_N_SIMULATIONS

    total_revenue = np.zeros(n_iter)
    total_cost = float(campaigns["total_cost"].sum())

    for action, row in campaigns.iterrows():
        n = int(row["n_customers"])
        aov = float(row["mean_aov"]) if np.isfinite(row["mean_aov"]) else 0.0
        prior = float(row["response_prior"])

        # Beta prior on response rate, centred on the planning assumption.
        a = prior * ROI_RESPONSE_CONCENTRATION
        b = (1.0 - prior) * ROI_RESPONSE_CONCENTRATION
        rates = rng.beta(a, b, size=n_iter)

        # Conversions and gross order revenue (with basket-size noise).
        conversions = rng.binomial(n, rates)
        noise = np.clip(rng.normal(1.0, ROI_REVENUE_NOISE_SD, size=n_iter), 0.0, None)
        total_revenue += conversions * aov * noise

    # Convert gross order value to profit contribution: apply retail gross
    # margin and net out the promotional discount embedded in the offers.
    # Counting full order value as profit (margin = 1, no discount) is what
    # produces the naive, implausibly high ROI; this adjustment keeps the
    # estimate in a defensible range comparable to published email-marketing ROI.
    contribution = total_revenue * (ROI_GROSS_MARGIN - ROI_OFFER_DISCOUNT)
    profit = contribution - total_cost
    roi = profit / total_cost if total_cost > 0 else np.zeros(n_iter)

    return {
        "roi": roi,
        "revenue": total_revenue,
        "contribution": contribution,
        "cost": np.full(n_iter, total_cost),
        "profit": profit,
    }


def _summarise(sim: dict[str, np.ndarray]) -> pd.DataFrame:
    """Reduce the simulation arrays to a one-column summary table."""
    roi = sim["roi"]
    profit = sim["profit"]
    revenue = sim["revenue"]
    contribution = sim["contribution"]
    cost = float(sim["cost"][0])

    lo_q = (1.0 - ROI_CI_LEVEL) / 2.0
    hi_q = 1.0 - lo_q

    stats = {
        "n_simulations": ROI_N_SIMULATIONS,
        "total_cost": round(cost, 2),
        "mean_gross_revenue": round(revenue.mean(), 2),
        "mean_contribution": round(contribution.mean(), 2),
        "mean_profit": round(profit.mean(), 2),
        "mean_roi": round(roi.mean(), 4),
        "median_roi": round(float(np.median(roi)), 4),
        f"roi_ci_low_{int(ROI_CI_LEVEL*100)}": round(float(np.quantile(roi, lo_q)), 4),
        f"roi_ci_high_{int(ROI_CI_LEVEL*100)}": round(float(np.quantile(roi, hi_q)), 4),
        "prob_positive_roi": round(float((roi > 0).mean()), 4),
        "prob_roi_gt_1": round(float((roi > 1.0).mean()), 4),
    }
    df = pd.DataFrame.from_dict(stats, orient="index", columns=["value"])
    logger.info("ROI simulation summary:\n%s", df.to_string())
    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_distribution(sim: dict[str, np.ndarray]) -> None:
    """Save a 2-panel histogram: ROI distribution and net-profit distribution."""
    roi = sim["roi"]
    profit = sim["profit"]
    lo_q = (1.0 - ROI_CI_LEVEL) / 2.0
    hi_q = 1.0 - lo_q
    roi_lo, roi_hi = np.quantile(roi, [lo_q, hi_q])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(roi, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(roi.mean(), color="#C44E52", linestyle="-", linewidth=1.8,
                    label=f"Mean ROI = {roi.mean():.2f}")
    axes[0].axvline(roi_lo, color="#333", linestyle="--", linewidth=1.3,
                    label=f"{int(ROI_CI_LEVEL*100)}% CI = [{roi_lo:.2f}, {roi_hi:.2f}]")
    axes[0].axvline(roi_hi, color="#333", linestyle="--", linewidth=1.3)
    axes[0].axvline(0, color="grey", linewidth=1.0)
    axes[0].set_xlabel("Campaign ROI  =  (revenue - cost) / cost", fontsize=10)
    axes[0].set_ylabel("Simulation count", fontsize=10)
    axes[0].set_title(f"ROI distribution\n({ROI_N_SIMULATIONS:,} Monte Carlo runs)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].hist(profit, bins=60, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].axvline(profit.mean(), color="#C44E52", linestyle="-", linewidth=1.8,
                    label=f"Mean profit = {profit.mean():,.0f}")
    axes[1].axvline(0, color="grey", linewidth=1.0)
    axes[1].set_xlabel("Net profit (revenue - cost)", fontsize=10)
    axes[1].set_ylabel("Simulation count", fontsize=10)
    axes[1].set_title("Net-profit distribution", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROI_DIST_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROI distribution plot saved to %s", ROI_DIST_PNG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_roi_simulation(
    plan: pd.DataFrame | None = None,
    clv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the Monte Carlo ROI simulation for the notification plan.

    Parameters
    ----------
    plan:
        Notification plan.  Pass ``None`` to load from
        ``outputs/tables/notification_plan.csv``.
    clv:
        CLV table (provides AvgOrderValue).  Pass ``None`` to load from
        ``data/processed/customer_clv.parquet``.

    Returns
    -------
    pd.DataFrame
        The summary statistics table (also written to CSV).

    Side effects
    ------------
    * Writes ``outputs/tables/roi_simulation_summary.csv``.
    * Writes ``outputs/figures/roi_distribution.png``.
    """
    if plan is None:
        logger.info("Loading notification plan from %s", NOTIFICATION_PLAN_CSV)
        plan = pd.read_csv(NOTIFICATION_PLAN_CSV)
    if clv is None:
        logger.info("Loading CLV table from %s", CUSTOMER_CLV_PARQUET)
        clv = pd.read_parquet(CUSTOMER_CLV_PARQUET)

    campaigns = _build_campaign_table(plan, clv)
    sim = _simulate(campaigns)
    summary = _summarise(sim)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ROI_SUMMARY_CSV)
    logger.info("ROI summary saved to %s", ROI_SUMMARY_CSV)

    _plot_distribution(sim)
    return summary
