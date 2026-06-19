"""
Feature preprocessing for distance-based clustering.

Why scaling is essential for K-Means and DBSCAN
------------------------------------------------
Both K-Means and DBSCAN rely on Euclidean distance (or, for HDBSCAN, a
density estimate derived from it) to measure similarity between customers.
Euclidean distance is dominated by whichever feature has the largest
numerical range: without scaling, Monetary (£3 – £608 K) would completely
overwhelm Recency (1 – 739 days), reducing the effective dimensionality to
one and making the chosen k meaningless. StandardScaler removes this bias by
transforming every feature to zero mean and unit variance, so each dimension
contributes equally to the distance calculation.

K-Means has an additional sensitivity: it minimises within-cluster *squared*
distances, so a single outlier that is 1 000 units away on a raw scale
exerts the same pull as 1 000 normal points that are 1 unit away. Log-
compressing right-skewed features before standardising brings extreme values
closer to the bulk of the distribution and produces more compact, spherical
clusters — the geometry that K-Means implicitly assumes. HDBSCAN does not
assume spherical clusters, but it still benefits from log-compression because
the density contrast between the core of a skewed distribution and its long
tail is so extreme that the algorithm either over-fragments the dense core or
merges genuinely distinct sparse groups.

Transformation pipeline
-----------------------
1. ``log1p`` is applied to every feature whose absolute skewness exceeds
   ``config.SKEW_THRESHOLD`` (default 0.5).  ``log1p(x) = log(1 + x)``
   is preferred over ``log(x)`` because it is defined at zero, handling the
   one-time buyers whose Tenure and AvgInterPurchaseDays are exactly 0.
2. ``StandardScaler`` is fitted on the (possibly log-compressed) features and
   applied to produce zero-mean, unit-variance columns.

The fitted scaler and the list of log-transformed columns are both returned
so that cluster centroids can be inverse-transformed back to the original
scale for interpretable segment profiling.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    CUSTOMER_FEATURES_PARQUET,
    DATA_PROCESSED_DIR,
    OUTPUTS_FIGURES_DIR,
    SCALED_FEATURES_PARQUET,
    SCALER_PATH,
    SKEW_THRESHOLD,
)

logger = logging.getLogger(__name__)

SCALING_EFFECT_PNG = OUTPUTS_FIGURES_DIR / "scaling_effect.png"

# Columns used as clustering features (Customer ID is the key, not a feature).
FEATURE_COLS: list[str] = [
    "Recency",
    "Frequency",
    "Monetary",
    "Tenure",
    "AvgOrderValue",
    "AvgInterPurchaseDays",
    "DistinctProducts",
]


class PreprocessingResult(NamedTuple):
    """Container returned by :func:`preprocess_features`.

    Attributes
    ----------
    X_scaled:
        2-D float64 array of shape ``(n_customers, n_features)`` with
        zero mean and unit variance, ready to pass directly to a clustering
        estimator.
    scaler:
        The fitted ``StandardScaler`` instance.  Call
        ``scaler.inverse_transform(X_scaled)`` to recover the log1p-
        compressed (but not yet expm1-inverted) values; then apply
        ``np.expm1`` to the columns listed in ``log1p_cols`` to get back
        approximate original-scale values.
    log1p_cols:
        Names of the features that received a ``log1p`` transform.  Needed
        to correctly invert the full pipeline when reporting cluster
        centroids in the original scale.
    feature_names:
        Ordered list of column names corresponding to columns of ``X_scaled``.
    """

    X_scaled: np.ndarray
    scaler: StandardScaler
    log1p_cols: list[str]
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _identify_log1p_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return the subset of ``cols`` whose absolute skewness exceeds the threshold.

    Skewness is computed on the raw (un-transformed) values so the decision
    is always grounded in the original distribution, regardless of how many
    times this function is called.

    Parameters
    ----------
    df:
        Customer feature DataFrame.
    cols:
        Candidate feature column names.
    """
    skewness = df[cols].skew().abs()
    selected = skewness[skewness > SKEW_THRESHOLD].index.tolist()
    logger.info(
        "log1p will be applied to %d / %d features (|skew| > %.2f): %s",
        len(selected), len(cols), SKEW_THRESHOLD, selected,
    )
    return selected


def _apply_log1p(df: pd.DataFrame, log1p_cols: list[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with ``log1p`` applied to the specified columns.

    ``log1p(x) = log(1 + x)`` is used instead of ``log(x)`` because several
    features (Tenure, AvgInterPurchaseDays) are zero for one-time buyers, and
    ``log(0)`` is undefined.  The +1 shift introduces a negligible bias at
    the scale of the values seen here (all positive, most well above 1).

    Parameters
    ----------
    df:
        Customer feature DataFrame containing at least the columns in
        ``log1p_cols``.
    log1p_cols:
        Column names to transform in-place on the copy.
    """
    out = df.copy()
    out[log1p_cols] = np.log1p(out[log1p_cols])
    return out


def _plot_scaling_effect(
    raw: pd.DataFrame,
    scaled_df: pd.DataFrame,
    log1p_cols: list[str],
    feature_names: list[str],
) -> None:
    """Save a before/after grid showing the effect of log1p + StandardScaler.

    Layout: two rows × ``n_features`` columns.
    - Top row: raw feature distributions (with skewness annotated).
    - Bottom row: scaled distributions after log1p + StandardScaler.

    Features that received log1p are labelled with a dagger (†) in the
    bottom row to make the transformation choice traceable in the figure.

    Parameters
    ----------
    raw:
        Customer feature DataFrame in original units.
    scaled_df:
        DataFrame of the same shape with scaled values (column names preserved).
    log1p_cols:
        Features that received log1p before scaling.
    feature_names:
        Ordered list of column names to plot.
    """
    n = len(feature_names)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6))
    fig.suptitle(
        "Feature distributions: raw (top) vs log1p + StandardScaler (bottom)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    row_labels = ["Raw", "Scaled"]
    row_colours = ["#4C72B0", "#DD8452"]

    for col_idx, feat in enumerate(feature_names):
        for row_idx, (data, colour, label) in enumerate(
            zip([raw[feat], scaled_df[feat]], row_colours, row_labels)
        ):
            ax = axes[row_idx, col_idx]
            ax.hist(data.dropna(), bins=50, color=colour,
                    edgecolor="white", linewidth=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)

            skew_val = data.skew()
            ax.set_title(
                f"{feat}{'†' if feat in log1p_cols and row_idx == 1 else ''}\n"
                f"skew={skew_val:.2f}",
                fontsize=8, pad=3,
            )
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")

    fig.text(
        0.01, 0.01,
        "† log1p applied before scaling",
        fontsize=8, color="grey", va="bottom",
    )
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCALING_EFFECT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scaling effect plot saved to %s", SCALING_EFFECT_PNG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_features(
    df: pd.DataFrame | None = None,
) -> PreprocessingResult:
    """Apply log1p compression and StandardScaler to customer features.

    If ``df`` is not supplied the function loads
    ``data/processed/customer_features.parquet`` automatically, making it
    convenient to call from :mod:`main` without manually threading DataFrames
    through the pipeline.

    Parameters
    ----------
    df:
        Customer feature table produced by :func:`src.features.build_customer_features`.
        Must contain at least the columns listed in ``FEATURE_COLS`` plus
        ``Customer ID``.  Pass ``None`` to load from the default parquet path.

    Returns
    -------
    PreprocessingResult
        Named tuple with fields:

        * ``X_scaled``     – ``(n_customers, 7)`` float64 array for clustering.
        * ``scaler``       – Fitted ``StandardScaler``.
        * ``log1p_cols``   – Features that had log1p applied.
        * ``feature_names``– Column order matching ``X_scaled`` columns.

    Side effects
    ------------
    * Writes ``data/processed/scaled_features.parquet``.
    * Writes ``data/processed/fitted_scaler.joblib``.
    * Writes ``outputs/figures/scaling_effect.png``.
    """
    if df is None:
        logger.info("Loading customer features from %s", CUSTOMER_FEATURES_PARQUET)
        df = pd.read_parquet(CUSTOMER_FEATURES_PARQUET)

    raw = df[FEATURE_COLS].copy()

    # ── Step 1: identify and apply log1p ───────────────────────────────────
    log1p_cols = _identify_log1p_cols(raw, FEATURE_COLS)
    transformed = _apply_log1p(raw, log1p_cols)

    # ── Step 2: fit and apply StandardScaler ───────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(transformed[FEATURE_COLS])
    scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=df.index)

    logger.info(
        "Scaling complete. Means ~= %s",
        np.round(X_scaled.mean(axis=0), 4),
    )
    logger.info(
        "Std devs ~= %s",
        np.round(X_scaled.std(axis=0), 4),
    )

    # ── Persist artefacts ──────────────────────────────────────────────────
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    scaled_out = scaled_df.copy()
    scaled_out.insert(0, "Customer ID", df["Customer ID"].values)
    scaled_out.to_parquet(SCALED_FEATURES_PARQUET, index=False)
    logger.info("Scaled features saved to %s", SCALED_FEATURES_PARQUET)

    joblib.dump(scaler, SCALER_PATH)
    logger.info("Fitted scaler saved to %s", SCALER_PATH)

    # ── Plot ───────────────────────────────────────────────────────────────
    _plot_scaling_effect(raw, scaled_df, log1p_cols, FEATURE_COLS)

    return PreprocessingResult(
        X_scaled=X_scaled,
        scaler=scaler,
        log1p_cols=log1p_cols,
        feature_names=FEATURE_COLS,
    )
