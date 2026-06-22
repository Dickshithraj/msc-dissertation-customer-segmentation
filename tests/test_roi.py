"""Tests for the Monte Carlo ROI engine in src.roi."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ROI_N_SIMULATIONS
from src.roi import _channel_cost, _simulate


def test_channel_cost_sums_components():
    # Email (0.05) + SMS (0.15) + Personal outreach (5.00) = 5.20.
    assert _channel_cost("Email + SMS + Personal outreach") == 5.20
    assert _channel_cost("Email") == 0.05
    assert _channel_cost("App push") == 0.02


def test_simulate_shapes_and_finiteness():
    campaigns = pd.DataFrame(
        {
            "n_customers": [1000, 500],
            "mean_aov": [200.0, 150.0],
            "cost_per_contact": [0.05, 0.15],
            "response_prior": [0.10, 0.20],
            "total_cost": [50.0, 75.0],
        },
        index=["Campaign A", "Campaign B"],
    )
    sim = _simulate(campaigns)

    for key in ("roi", "revenue", "contribution", "profit", "cost"):
        assert sim[key].shape == (ROI_N_SIMULATIONS,)
        assert np.isfinite(sim[key]).all()

    # Total cost is the deterministic sum of per-campaign costs.
    assert np.allclose(sim["cost"], 125.0)
    # Contribution must be strictly less than gross revenue (margin < 1).
    assert (sim["contribution"] <= sim["revenue"]).all()


def test_simulate_zero_cost_safe():
    campaigns = pd.DataFrame(
        {
            "n_customers": [10],
            "mean_aov": [100.0],
            "cost_per_contact": [0.0],
            "response_prior": [0.1],
            "total_cost": [0.0],
        },
        index=["Free"],
    )
    sim = _simulate(campaigns)
    # No division-by-zero blow-up when total cost is zero.
    assert np.isfinite(sim["roi"]).all()
