"""
Sensitivity analysis for the Monte Carlo ROI simulation.

The Phase 11 headline (targeted plan beats the static blanket campaign) rests
on *assumed* response-rate priors, channel costs, and margin parameters.  This
module stress-tests that conclusion: each scenario perturbs one assumption
(or, in the worst-case scenario, several at once) and re-runs the full
10,000-iteration simulation for both the targeted plan and the static
baseline.  If the profit uplift of targeting stays positive across every
scenario -- including the deliberately pessimistic ones -- the headline claim
is robust to the planning assumptions rather than an artefact of them.

Scenario design
---------------
- ``response x0.5 / x0.75 / x1.5``: scale the *targeted* campaign priors while
  leaving the static baseline untouched (halving only the targeted rates is
  the conservative direction: it handicaps the plan, not the baseline).
- ``static response x2``: double the blanket campaign's response rate instead.
- ``costs x2 / x5``: scale every per-contact channel cost for both plans.
- ``margin 30% -> 20%`` and ``discount 10% -> 15%``: squeeze the profit
  conversion for both plans.
- ``worst case``: response x0.5, static x2, costs x2, margin 20%, discount 15%
  simultaneously.

Outputs
-------
outputs/tables/roi_sensitivity.csv     -- per-scenario summary table
outputs/figures/roi_sensitivity.png    -- tornado-style uplift chart
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
    ROI_GROSS_MARGIN,
    ROI_N_SIMULATIONS,
    ROI_OFFER_DISCOUNT,
    ROI_RESPONSE_CONCENTRATION,
    ROI_REVENUE_NOISE_SD,
    ROI_STATIC_RESPONSE,
)
from src.roi import _build_campaign_table, _build_static_baseline

logger = logging.getLogger(__name__)

SENSITIVITY_CSV = OUTPUTS_TABLES_DIR / "roi_sensitivity.csv"
SENSITIVITY_PNG = OUTPUTS_FIGURES_DIR / "roi_sensitivity.png"


@dataclass
class Scenario:
    """One perturbation of the ROI simulation's planning assumptions."""
    name: str
    response_scale: float = 1.0      # multiplier on targeted campaign priors
    static_response: float = ROI_STATIC_RESPONSE
    cost_scale: float = 1.0          # multiplier on per-contact costs
    gross_margin: float = ROI_GROSS_MARGIN
    offer_discount: float = ROI_OFFER_DISCOUNT
    notes: str = field(default="")


SCENARIOS: list[Scenario] = [
    Scenario("Baseline (reported)", notes="Assumptions as configured in Phase 11"),
    Scenario("Response priors x0.5", response_scale=0.5,
             notes="Targeted response rates halved; static unchanged"),
    Scenario("Response priors x0.75", response_scale=0.75),
    Scenario("Response priors x1.5", response_scale=1.5),
    Scenario("Static response x2 (4%)", static_response=0.04,
             notes="Blanket campaign responds twice as well as assumed"),
    Scenario("Channel costs x2", cost_scale=2.0),
    Scenario("Channel costs x5", cost_scale=5.0),
    Scenario("Gross margin 30% -> 20%", gross_margin=0.20),
    Scenario("Offer discount 10% -> 15%", offer_discount=0.15),
    Scenario("Worst case (all pessimistic)", response_scale=0.5,
             static_response=0.04, cost_scale=2.0,
             gross_margin=0.20, offer_discount=0.15,
             notes="Every assumption moved against the targeted plan at once"),
]


def _simulate_parameterised(
    campaigns: pd.DataFrame,
    response_scale: float,
    cost_scale: float,
    gross_margin: float,
    offer_discount: float,
    rng_seed: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Re-implementation of :func:`src.roi._simulate` with explicit knobs.

    Identical sampling scheme (Beta response -> Binomial conversions ->
    Normal basket noise -> margin/discount contribution), but response priors,
    costs, and profit-conversion parameters are arguments rather than module
    constants so scenarios can perturb them independently.
    """
    rng = np.random.default_rng(rng_seed)
    n_iter = ROI_N_SIMULATIONS

    total_revenue = np.zeros(n_iter)
    total_cost = float(campaigns["total_cost"].sum()) * cost_scale

    for _, row in campaigns.iterrows():
        n = int(row["n_customers"])
        aov = float(row["mean_aov"]) if np.isfinite(row["mean_aov"]) else 0.0
        prior = min(float(row["response_prior"]) * response_scale, 0.99)

        a = prior * ROI_RESPONSE_CONCENTRATION
        b = (1.0 - prior) * ROI_RESPONSE_CONCENTRATION
        rates = rng.beta(a, b, size=n_iter)

        conversions = rng.binomial(n, rates)
        noise = np.clip(rng.normal(1.0, ROI_REVENUE_NOISE_SD, size=n_iter), 0.0, None)
        total_revenue += conversions * aov * noise

    contribution = total_revenue * (gross_margin - offer_discount)
    profit = contribution - total_cost
    roi = profit / total_cost if total_cost > 0 else np.zeros(n_iter)
    return {"roi": roi, "profit": profit, "cost": np.full(n_iter, total_cost)}


def _run_scenario(
    sc: Scenario,
    campaigns: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, float | str]:
    """Simulate one scenario for both plans and summarise the uplift."""
    static = baseline.copy()
    static["response_prior"] = sc.static_response

    targeted_sim = _simulate_parameterised(
        campaigns, sc.response_scale, sc.cost_scale,
        sc.gross_margin, sc.offer_discount,
    )
    # Static baseline: response_scale=1 because its response rate is already
    # set explicitly above; cost/margin/discount perturbations apply to both.
    static_sim = _simulate_parameterised(
        static, 1.0, sc.cost_scale, sc.gross_margin, sc.offer_discount,
    )

    t_roi = targeted_sim["roi"]
    profit_uplift = targeted_sim["profit"] - static_sim["profit"]
    row = {
        "scenario": sc.name,
        "targeted_mean_roi": round(float(t_roi.mean()), 2),
        "static_mean_roi": round(float(static_sim["roi"].mean()), 2),
        "mean_profit_uplift": round(float(profit_uplift.mean()), 0),
        "prob_positive_uplift": round(float((profit_uplift > 0).mean()), 4),
        "prob_positive_roi": round(float((t_roi > 0).mean()), 4),
    }
    logger.info("Scenario %-32s targeted ROI=%8.2f  uplift=%12.0f  P(uplift>0)=%.4f",
                sc.name, row["targeted_mean_roi"], row["mean_profit_uplift"],
                row["prob_positive_uplift"])
    return row


def _plot_sensitivity(results: pd.DataFrame) -> None:
    """Save a tornado-style horizontal bar chart of mean profit uplift."""
    df = results.iloc[::-1]  # baseline at top after inversion
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52" if v <= 0 else "#4C72B0" for v in df["mean_profit_uplift"]]
    ax.barh(df["scenario"], df["mean_profit_uplift"], color=colors, alpha=0.85)
    ax.axvline(0, color="grey", linewidth=1.0)
    for y, v in enumerate(df["mean_profit_uplift"]):
        ax.text(v, y, f" {v:,.0f}", va="center", fontsize=8.5,
                ha="left" if v >= 0 else "right")
    ax.set_xlabel("Mean profit uplift of targeted plan vs static baseline (currency)",
                  fontsize=10)
    ax.set_title("ROI sensitivity: profit uplift under perturbed assumptions\n"
                 f"({ROI_N_SIMULATIONS:,} Monte Carlo runs per scenario)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SENSITIVITY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROI sensitivity plot saved to %s", SENSITIVITY_PNG)


def run_roi_sensitivity(
    plan: pd.DataFrame | None = None,
    clv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run every scenario and produce the sensitivity table + tornado chart.

    Returns
    -------
    pd.DataFrame
        One row per scenario (also written to
        ``outputs/tables/roi_sensitivity.csv``).
    """
    if plan is None:
        logger.info("Loading notification plan from %s", NOTIFICATION_PLAN_CSV)
        plan = pd.read_csv(NOTIFICATION_PLAN_CSV)
    if clv is None:
        logger.info("Loading CLV table from %s", CUSTOMER_CLV_PARQUET)
        clv = pd.read_parquet(CUSTOMER_CLV_PARQUET)

    campaigns = _build_campaign_table(plan, clv)
    baseline = _build_static_baseline(plan, clv)

    results = pd.DataFrame([_run_scenario(sc, campaigns, baseline)
                            for sc in SCENARIOS])

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(SENSITIVITY_CSV, index=False)
    logger.info("ROI sensitivity table saved to %s\n%s",
                SENSITIVITY_CSV, results.to_string(index=False))

    _plot_sensitivity(results)
    return results
