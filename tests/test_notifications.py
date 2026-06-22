"""Tests for the rule-based recommendation logic in src.notifications."""

from __future__ import annotations

import pandas as pd

from src.notifications import _recommend_row


def _row(**kwargs) -> pd.Series:
    base = {
        "segment": "General Customers",
        "churn_risk": "Low",
        "clv_tier": "Medium",
        "AvgInterPurchaseDays": 40.0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_high_churn_high_value_triggers_retention():
    rec = _recommend_row(_row(churn_risk="High", clv_tier="High"))
    assert rec["action"] == "Priority retention intervention"
    assert rec["priority"] == 5  # escalated and capped


def test_high_churn_low_value_is_budget_capped():
    rec = _recommend_row(_row(churn_risk="High", clv_tier="Low"))
    assert rec["action"] == "Low-cost automated reactivation"
    assert rec["channel"] == "Email"  # cheapest channel only


def test_champion_baseline_and_value_escalation():
    low = _recommend_row(_row(segment="Champions", clv_tier="Medium"))
    high = _recommend_row(_row(segment="Champions", clv_tier="High"))
    # High CLV nudges priority up by one.
    assert high["priority"] == min(low["priority"] + 1, 5)


def test_contact_window_uses_cadence():
    rec = _recommend_row(_row(AvgInterPurchaseDays=40.0))
    # 0.6 * 40 = 24 days (NOTIF_CADENCE_FRACTION default 0.6).
    assert rec["recommended_contact_days"] == 24


def test_one_time_buyer_falls_back_to_default():
    rec = _recommend_row(_row(AvgInterPurchaseDays=0.0))
    assert rec["recommended_contact_days"] == 30  # NOTIF_DEFAULT_CONTACT_DAYS
