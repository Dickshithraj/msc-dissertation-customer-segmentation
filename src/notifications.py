"""
Phase 10: Rule-based marketing notification engine.

Turns the analytical outputs of earlier stages into a concrete, per-customer
marketing action plan.  Each customer is assigned:

- a **segment** — the marketing name of the **unsupervised cluster** they were
  assigned in Stage 3 (``NOTIF_CLUSTER_ALGO``, default HDBSCAN), so the campaign
  engine is driven by the clustering result, not a separate rule pass,
- a **value tier** (High / Medium / Low) from their predicted CLV,
- a **churn-risk band** (High / Medium / Low) from their churn probability,

and these three signals drive a transparent decision matrix that outputs a
campaign action, channel, offer, priority, and recommended contact timing.

Design rationale
----------------
The engine is deliberately *rule-based* rather than learned: marketing
stakeholders must be able to read, audit, and override every recommendation,
and the rules encode well-established CRM playbook logic (reward the loyal,
win back the at-risk, do not overspend on low-value churners).  The three
inputs are complementary:

- **Segment** sets the baseline strategy (what kind of customer this is).
- **Churn risk** decides urgency (should we intervene now?).
- **CLV tier** decides budget (how much is this customer worth spending on?).

The interaction matters: a high-churn / high-CLV customer warrants an expensive
personal retention offer, whereas a high-churn / low-CLV customer gets only a
low-cost automated nudge so marketing budget is not wasted.

Contact timing
--------------
The recommended re-contact window is a fraction (``NOTIF_CADENCE_FRACTION``) of
the customer's own average inter-purchase gap, so the message lands *before*
they would naturally lapse.  One-time buyers (no cadence) fall back to
``NOTIF_DEFAULT_CONTACT_DAYS``.

Delivery model
--------------
This is a **logic-based automation framework**, not a real-time streaming
system: :func:`generate_notifications` builds the full plan as a batch, and
:func:`recommend` exposes the same logic for **on-demand** single-customer
look-ups (e.g. behind an API endpoint).  Because the rules are deterministic,
the per-customer recommendation can be (re)computed the moment a customer's
CLV / churn / cluster inputs change, which is what makes the engine "dynamic"
without requiring a live event pipeline.

Outputs
-------
outputs/tables/notification_plan.csv -- one row per customer with the full plan
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    CLUSTER_PARQUET,
    CUSTOMER_CHURN_PARQUET,
    CUSTOMER_CLV_PARQUET,
    NOTIF_CADENCE_FRACTION,
    NOTIF_CHURN_HIGH,
    NOTIF_CHURN_MED,
    NOTIF_CLUSTER_ALGO,
    NOTIF_CLV_HIGH_Q,
    NOTIF_CLV_LOW_Q,
    NOTIF_DEFAULT_CONTACT_DAYS,
    NOTIFICATION_PLAN_CSV,
    OUTPUTS_TABLES_DIR,
)
from src.profiling import (
    FEATURE_COLS,
    _LABEL_COL,
    _build_profiles,
    _name_clusters,
)

logger = logging.getLogger(__name__)

# Baseline strategy per segment: (action, channel, base offer, base priority).
# Priority is a 1-5 integer (5 = most urgent) before churn/CLV modulation.
_SEGMENT_PLAYBOOK: dict[str, dict] = {
    "Champions": {
        "action": "VIP loyalty reward + early access",
        "channel": "Email + App push",
        "offer": "Exclusive previews, loyalty points bonus",
        "priority": 4,
    },
    "Loyal Customers": {
        "action": "Cross-sell + loyalty tier upgrade",
        "channel": "Email + App push",
        "offer": "Bundle discount on complementary categories",
        "priority": 3,
    },
    "Big Spenders": {
        "action": "Premium product recommendations",
        "channel": "Email + Personal outreach",
        "offer": "Concierge / personal-shopper invite",
        "priority": 4,
    },
    "Potential Loyalists": {
        "action": "Frequency-building nudge",
        "channel": "App push + Email",
        "offer": "Second-purchase incentive, membership signup",
        "priority": 3,
    },
    "General Customers": {
        "action": "Engagement / category promotion",
        "channel": "Email",
        "offer": "Seasonal category promotion",
        "priority": 2,
    },
    "At Risk": {
        "action": "Win-back outreach",
        "channel": "Email + SMS",
        "offer": "'We miss you' targeted discount",
        "priority": 5,
    },
    "Lost Customers": {
        "action": "Reactivation campaign",
        "channel": "Email",
        "offer": "Aggressive reactivation discount",
        "priority": 2,
    },
    "Noise / Uncategorised": {
        "action": "Standard newsletter",
        "channel": "Email",
        "offer": "General newsletter, no targeted offer",
        "priority": 1,
    },
}

# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _cluster_segment_names(clusters: pd.DataFrame, label_col: str) -> dict[int, str]:
    """Map each cluster id to a marketing segment name from its mean RFM profile.

    Reuses the same rule-based namer used to label clusters in profiling, so a
    cluster whose averaged behaviour looks like "Champions" drives the Champions
    playbook.  Cluster -1 (HDBSCAN/DBSCAN noise) becomes "Noise / Uncategorised".
    """
    profiles = _build_profiles(clusters, label_col)
    overall = clusters[FEATURE_COLS].mean()
    return _name_clusters(profiles, overall)


def _load_customer_table() -> pd.DataFrame:
    """Merge CLV, churn, and unsupervised cluster labels into one table.

    The CLV parquet already carries the full feature set (it was built by
    merging onto customer_features), so it is the base; churn scores and the
    cluster label from the configured clustering algorithm (``NOTIF_CLUSTER_ALGO``)
    are joined on Customer ID.  Each customer's marketing ``segment`` is the name
    of *their cluster* (derived from the cluster's mean RFM profile), so the
    campaign engine is driven by the clustering result rather than a separate
    per-customer rule pass.
    """
    logger.info("Loading CLV table from %s", CUSTOMER_CLV_PARQUET)
    clv = pd.read_parquet(CUSTOMER_CLV_PARQUET)
    logger.info("Loading churn table from %s", CUSTOMER_CHURN_PARQUET)
    churn = pd.read_parquet(CUSTOMER_CHURN_PARQUET)
    logger.info("Loading %s cluster labels from %s", NOTIF_CLUSTER_ALGO, CLUSTER_PARQUET)
    clusters = pd.read_parquet(CLUSTER_PARQUET)

    df = clv.merge(
        churn[["Customer ID", "churn_label", "churn_probability"]],
        on="Customer ID", how="left",
    )

    # Attach each customer's cluster label and its cluster-derived segment name.
    label_col = _LABEL_COL[NOTIF_CLUSTER_ALGO]
    cluster_names = _cluster_segment_names(clusters, label_col)
    df = df.merge(clusters[["Customer ID", label_col]], on="Customer ID", how="left")
    df["cluster"] = df[label_col]
    df["segment"] = df["cluster"].map(cluster_names).fillna("Noise / Uncategorised")
    df = df.drop(columns=[label_col])

    logger.info(
        "Merged customer table: %d rows. Segments (from %s clusters):\n%s",
        len(df), NOTIF_CLUSTER_ALGO, df["segment"].value_counts().to_string(),
    )
    return df


# ---------------------------------------------------------------------------
# Tier / band / segment assignment
# ---------------------------------------------------------------------------

def _assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Add clv_tier and churn_risk columns (returns copy).

    The ``segment`` column is already set in :func:`_load_customer_table` from
    the customer's cluster, so this step only adds the value and churn-risk
    bands that modulate each segment's baseline campaign.
    """
    out = df.copy()

    clv_high = out["clv"].quantile(NOTIF_CLV_HIGH_Q)
    clv_low = out["clv"].quantile(NOTIF_CLV_LOW_Q)

    def _clv_tier(v: float) -> str:
        if v >= clv_high:
            return "High"
        if v < clv_low:
            return "Low"
        return "Medium"

    out["clv_tier"] = out["clv"].apply(_clv_tier)

    def _churn_band(p: float) -> str:
        if p >= NOTIF_CHURN_HIGH:
            return "High"
        if p >= NOTIF_CHURN_MED:
            return "Medium"
        return "Low"

    out["churn_risk"] = out["churn_probability"].fillna(0.0).apply(_churn_band)

    logger.info("CLV tier thresholds: High>=%.0f, Low<%.0f", clv_high, clv_low)
    return out


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def _recommend_row(row: pd.Series) -> dict:
    """Produce a campaign recommendation for a single customer row.

    Combines the segment playbook baseline with churn/CLV modulation:

    - High churn risk escalates priority and, for High/Medium CLV, swaps in a
      retention-focused offer; for Low CLV it down-shifts to a low-cost
      automated nudge so spend is not wasted on unprofitable churners.
    - High CLV nudges priority up by one (capped at 5) regardless of segment.
    """
    seg = row["segment"]
    base = _SEGMENT_PLAYBOOK.get(seg, _SEGMENT_PLAYBOOK["General Customers"])

    action = base["action"]
    channel = base["channel"]
    offer = base["offer"]
    priority = base["priority"]

    churn_risk = row["churn_risk"]
    clv_tier = row["clv_tier"]

    # Churn-driven escalation.
    if churn_risk == "High":
        priority = min(priority + 2, 5)
        if clv_tier in ("High", "Medium"):
            action = "Priority retention intervention"
            offer = "High-value personalised retention offer"
            channel = "Email + SMS + Personal outreach"
        else:
            action = "Low-cost automated reactivation"
            offer = "Single automated discount email (budget-capped)"
            channel = "Email"
    elif churn_risk == "Medium":
        priority = min(priority + 1, 5)

    # Value-driven escalation (independent of churn).
    if clv_tier == "High":
        priority = min(priority + 1, 5)

    # Contact timing from the customer's own cadence.
    cadence = row.get("AvgInterPurchaseDays", 0) or 0
    if cadence and cadence > 0:
        contact_days = max(int(round(cadence * NOTIF_CADENCE_FRACTION)), 1)
    else:
        contact_days = NOTIF_DEFAULT_CONTACT_DAYS

    return {
        "action": action,
        "channel": channel,
        "offer": offer,
        "priority": int(priority),
        "recommended_contact_days": contact_days,
    }


def _build_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full notification plan table for all customers."""
    recs = df.apply(_recommend_row, axis=1, result_type="expand")
    plan = pd.concat([
        df[["Customer ID", "segment", "clv", "clv_tier",
            "churn_probability", "churn_risk"]].reset_index(drop=True),
        recs.reset_index(drop=True),
    ], axis=1)
    plan = plan.sort_values(
        ["priority", "clv"], ascending=[False, False],
    ).reset_index(drop=True)
    return plan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level cache so repeated recommend() calls don't reload parquet files.
_PLAN_CACHE: pd.DataFrame | None = None


def generate_notifications(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build and persist the per-customer notification plan.

    Parameters
    ----------
    df:
        Pre-merged customer table.  Pass ``None`` to assemble it from the CLV
        and churn parquet artefacts.

    Returns
    -------
    pd.DataFrame
        The notification plan, sorted by descending priority then CLV.  Saved
        to ``outputs/tables/notification_plan.csv``.
    """
    global _PLAN_CACHE

    if df is None:
        df = _load_customer_table()

    tiered = _assign_tiers(df)
    plan = _build_plan(tiered)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    plan.to_csv(NOTIFICATION_PLAN_CSV, index=False)
    logger.info("Notification plan (%d customers) saved to %s",
                len(plan), NOTIFICATION_PLAN_CSV)

    # Distribution diagnostics for the dissertation.
    logger.info("Action distribution:\n%s",
                plan["action"].value_counts().to_string())
    logger.info("Priority distribution:\n%s",
                plan["priority"].value_counts().sort_index(ascending=False).to_string())

    _PLAN_CACHE = plan
    return plan


def recommend(customer_id: int, refresh: bool = False) -> dict:
    """Return the marketing recommendation for a single customer.

    Parameters
    ----------
    customer_id:
        The ``Customer ID`` to look up.
    refresh:
        If ``True``, rebuild the plan from source parquet files before
        looking up; otherwise reuse the cached plan (built on first call).

    Returns
    -------
    dict
        The customer's full recommendation row as a dictionary.

    Raises
    ------
    KeyError
        If the customer ID is not present in the plan.
    """
    # Read-only access to the module cache; no `global` needed here.
    if _PLAN_CACHE is None or refresh:
        generate_notifications()

    assert _PLAN_CACHE is not None
    match = _PLAN_CACHE[_PLAN_CACHE["Customer ID"] == customer_id]
    if match.empty:
        raise KeyError(f"Customer ID {customer_id} not found in notification plan.")
    return match.iloc[0].to_dict()
