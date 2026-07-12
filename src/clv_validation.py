"""
Temporal (calibration / holdout) validation of the BG/NBD purchase model.

Phase 7 fits BG/NBD + Gamma-Gamma on the *full* two-year window, which leaves
the CLV forecasts without an out-of-sample check.  This module closes that gap
with the standard Fader-Hardie calibration/holdout procedure:

1. Split the transaction history at ``max(InvoiceDate) - CLV_HOLDOUT_DAYS``:
   everything before is the **calibration** period, everything after the
   **holdout** period (unseen by the model).
2. Fit BG/NBD on calibration-period frequency/recency/T only.
3. Predict each customer's expected number of purchases during the holdout
   window and compare against the purchases they *actually* made.

Reported metrics
----------------
- MAE / RMSE of per-customer holdout purchase counts,
- Pearson correlation between predicted and actual counts,
- aggregate bias: total predicted vs total actual holdout purchases,
- the classic conditional-expectation plot (mean actual vs mean predicted
  holdout purchases, grouped by calibration-period frequency), which is the
  "reproduction of known results" check from Fader, Hardie & Lee (2005).

Outputs
-------
outputs/tables/clv_holdout_metrics.csv        -- headline validation metrics
outputs/figures/clv_holdout_validation.png    -- 2-panel diagnostic figure
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes.utils import calibration_and_holdout_data

from src.config import (
    BGMD_PENALIZER_COEF,
    CLEANED_PARQUET,
    CLV_HOLDOUT_DAYS,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
)

logger = logging.getLogger(__name__)

HOLDOUT_METRICS_CSV = OUTPUTS_TABLES_DIR / "clv_holdout_metrics.csv"
HOLDOUT_PNG = OUTPUTS_FIGURES_DIR / "clv_holdout_validation.png"

# Calibration-frequency bins beyond this are pooled into one "7+" group so the
# conditional-expectation plot is not dominated by a handful of extreme buyers.
_MAX_FREQ_BIN = 7


def _split_calibration_holdout(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Build the lifetimes calibration/holdout summary table.

    The calibration period ends ``CLV_HOLDOUT_DAYS`` days before the last
    observed transaction, so the holdout window is a full year of genuinely
    unseen behaviour.
    """
    obs_end = transactions["InvoiceDate"].max()
    calib_end = obs_end - pd.Timedelta(days=CLV_HOLDOUT_DAYS)
    logger.info(
        "Calibration period: %s -> %s | holdout: %s -> %s (%d days)",
        transactions["InvoiceDate"].min().date(), calib_end.date(),
        calib_end.date(), obs_end.date(), CLV_HOLDOUT_DAYS,
    )

    summary = calibration_and_holdout_data(
        transactions,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        calibration_period_end=calib_end,
        observation_period_end=obs_end,
        freq="D",
    )
    logger.info(
        "Calibration/holdout summary built for %d customers active in the "
        "calibration period.", len(summary),
    )
    return summary, calib_end, obs_end


def _fit_and_predict(summary: pd.DataFrame) -> pd.DataFrame:
    """Fit BG/NBD on the calibration columns and predict holdout purchases.

    On the one-year calibration window the *unpenalized* likelihood converges
    to sensible interior parameters, while small L2 penalties (0.001-0.01)
    paradoxically push the dropout parameters (a, b) to the boundary and the
    optimiser reports non-convergence.  The ladder therefore starts at 0 and
    only escalates if the clean fit fails; every prediction is checked for
    finiteness downstream, and the penalizer actually used is logged.
    """
    bgf = None
    for pen in (0.0, BGMD_PENALIZER_COEF, 0.01, 0.1):
        try:
            candidate = BetaGeoFitter(penalizer_coef=pen)
            candidate.fit(summary["frequency_cal"], summary["recency_cal"],
                          summary["T_cal"])
        except Exception as exc:  # lifetimes raises ConvergenceError
            logger.warning("BG/NBD calibration fit failed at penalizer=%g: %s",
                           pen, exc)
            continue
        bgf = candidate
        logger.info("BG/NBD (calibration only) fitted with penalizer=%g. "
                    "Params:\n%s", pen, bgf.summary.to_string())
        break
    if bgf is None:
        raise RuntimeError("BG/NBD calibration fit failed at every penalizer.")

    out = summary.copy()
    out["predicted_holdout"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        out["duration_holdout"],
        out["frequency_cal"], out["recency_cal"], out["T_cal"],
    )
    bad = ~np.isfinite(out["predicted_holdout"])
    if bad.any():
        logger.warning(
            "Dropping %d/%d customers with numerically undefined predictions "
            "(degenerate parameter/input combinations).", int(bad.sum()), len(out),
        )
        out = out[~bad]
    return out


def _compute_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-customer predictions to the headline validation metrics."""
    actual = pred_df["frequency_holdout"].to_numpy(dtype=float)
    predicted = pred_df["predicted_holdout"].to_numpy(dtype=float)

    err = predicted - actual
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    corr = float(np.corrcoef(predicted, actual)[0, 1])
    total_pred = float(predicted.sum())
    total_actual = float(actual.sum())
    bias_pct = (total_pred - total_actual) / total_actual * 100 if total_actual else float("nan")

    metrics = pd.DataFrame.from_dict(
        {
            "n_customers": len(pred_df),
            "holdout_days": CLV_HOLDOUT_DAYS,
            "mae_purchases": round(mae, 4),
            "rmse_purchases": round(rmse, 4),
            "pearson_r": round(corr, 4),
            "total_actual_purchases": int(total_actual),
            "total_predicted_purchases": round(total_pred, 1),
            "aggregate_bias_pct": round(bias_pct, 2),
        },
        orient="index", columns=["value"],
    )
    logger.info("CLV holdout validation metrics:\n%s", metrics.to_string())
    return metrics


def _plot_validation(pred_df: pd.DataFrame) -> None:
    """Save the 2-panel holdout diagnostic figure.

    Panel 1 is the Fader-Hardie conditional-expectation plot: customers are
    grouped by how many repeat purchases they made in the calibration period,
    and the group means of actual vs predicted holdout purchases are compared.
    Panel 2 aggregates customers into deciles of predicted holdout purchases
    and compares each decile's mean prediction with its mean actual count
    (a calibration/reliability curve).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: conditional expectation by calibration frequency.
    grouped = pred_df.copy()
    grouped["freq_bin"] = grouped["frequency_cal"].clip(upper=_MAX_FREQ_BIN)
    means = grouped.groupby("freq_bin")[["frequency_holdout", "predicted_holdout"]].mean()
    labels = [f"{int(i)}" if i < _MAX_FREQ_BIN else f"{_MAX_FREQ_BIN}+"
              for i in means.index]
    axes[0].plot(means.index, means["frequency_holdout"], "o-", color="#4C72B0",
                 linewidth=1.8, label="Actual (holdout)")
    axes[0].plot(means.index, means["predicted_holdout"], "s--", color="#C44E52",
                 linewidth=1.8, label="Predicted (BG/NBD)")
    axes[0].set_xticks(list(means.index))
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Repeat purchases in calibration period", fontsize=10)
    axes[0].set_ylabel("Mean purchases in holdout year", fontsize=10)
    axes[0].set_title("Conditional expectation of holdout purchases\n"
                      "(Fader-Hardie calibration/holdout check)",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Panel 2: decile reliability curve.
    dec = pred_df.copy()
    dec["decile"] = pd.qcut(dec["predicted_holdout"].rank(method="first"),
                            10, labels=False)
    dmeans = dec.groupby("decile")[["predicted_holdout", "frequency_holdout"]].mean()
    lim = float(dmeans.to_numpy().max()) * 1.1
    axes[1].plot([0, lim], [0, lim], color="grey", linewidth=1.0,
                 label="Perfect calibration")
    axes[1].plot(dmeans["predicted_holdout"], dmeans["frequency_holdout"],
                 "o-", color="#55A868", linewidth=1.8, label="Decile means")
    axes[1].set_xlabel("Mean predicted holdout purchases (decile)", fontsize=10)
    axes[1].set_ylabel("Mean actual holdout purchases (decile)", fontsize=10)
    axes[1].set_title("Reliability of per-customer forecasts\n"
                      "(customers binned by predicted activity)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(HOLDOUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("CLV holdout validation plot saved to %s", HOLDOUT_PNG)


def run_clv_holdout_validation(
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the calibration/holdout validation of the BG/NBD model.

    Parameters
    ----------
    transactions:
        Cleaned transaction rows.  Pass ``None`` to load from
        ``data/processed/cleaned_transactions.parquet``.

    Returns
    -------
    pd.DataFrame
        The validation metrics table (also written to CSV).
    """
    if transactions is None:
        logger.info("Loading cleaned transactions from %s", CLEANED_PARQUET)
        transactions = pd.read_parquet(CLEANED_PARQUET)

    summary, _, _ = _split_calibration_holdout(transactions)
    pred_df = _fit_and_predict(summary)
    metrics = _compute_metrics(pred_df)

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(HOLDOUT_METRICS_CSV)
    logger.info("CLV holdout metrics saved to %s", HOLDOUT_METRICS_CSV)

    _plot_validation(pred_df)
    return metrics
