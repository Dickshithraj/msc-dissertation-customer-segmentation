"""
Phase 8: Churn classification (Logistic Regression vs Random Forest vs XGBoost).

Churn definition
----------------
A customer is labelled *churned* (y = 1) if their Recency exceeds the
``CHURN_RECENCY_PERCENTILE`` (default 90th) percentile of the customer base —
i.e. they are among the most dormant customers, having gone the longest
without a purchase.  This yields an intentionally imbalanced target
(~10% positive), which is handled with class weighting rather than resampling.

Target-leakage avoidance
-------------------------
Because the label is *derived from Recency*, Recency itself is excluded from
the feature set.  Including it would let any model trivially recover the
percentile threshold and report a meaningless ~1.0 ROC-AUC.  The models must
instead predict dormancy from behavioural signals that are *not* mechanically
tied to the label: Frequency, Monetary, Tenure, AvgOrderValue,
AvgInterPurchaseDays, and DistinctProducts.

Models compared
---------------
- Logistic Regression  (linear baseline, class_weight="balanced")
- Random Forest        (bagged trees, class_weight="balanced")
- XGBoost              (gradient-boosted trees, scale_pos_weight for imbalance)

All three are evaluated on a held-out stratified test split and via stratified
cross-validation, ranked primarily by ROC-AUC (threshold-independent and
robust to class imbalance).  The best model produces a churn probability for
every customer.

Outputs
-------
data/processed/customer_churn.parquet      -- Customer ID, churn_label, churn_probability
outputs/tables/churn_model_comparison.csv  -- per-model metrics
outputs/figures/churn_roc_curves.png       -- ROC curves for all three models
outputs/figures/churn_feature_importance.png -- importance from the best tree model
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CHURN_CV_FOLDS,
    CHURN_RECENCY_PERCENTILE,
    CUSTOMER_CHURN_PARQUET,
    CUSTOMER_FEATURES_PARQUET,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_TABLES_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    XGB_COLSAMPLE_BYTREE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)

logger = logging.getLogger(__name__)

# Behavioural features used to predict churn. Recency is deliberately omitted
# (see module docstring: it defines the label, so including it would leak).
CHURN_FEATURES: list[str] = [
    "Frequency", "Monetary", "Tenure",
    "AvgOrderValue", "AvgInterPurchaseDays", "DistinctProducts",
]

COMPARISON_CSV = OUTPUTS_TABLES_DIR / "churn_model_comparison.csv"
ROC_PNG = OUTPUTS_FIGURES_DIR / "churn_roc_curves.png"
IMPORTANCE_PNG = OUTPUTS_FIGURES_DIR / "churn_feature_importance.png"


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def _build_label(features: pd.DataFrame) -> pd.Series:
    """Return the binary churn label from the Recency percentile threshold."""
    threshold = features["Recency"].quantile(CHURN_RECENCY_PERCENTILE)
    y = (features["Recency"] > threshold).astype(int)
    logger.info(
        "Churn label: Recency > %.0f days (P%.0f) -> %d churned / %d total (%.1f%%).",
        threshold, CHURN_RECENCY_PERCENTILE * 100, int(y.sum()), len(y),
        y.mean() * 100,
    )
    return y


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_models(pos_weight: float) -> dict[str, object]:
    """Instantiate the three classifiers with imbalance handling.

    Parameters
    ----------
    pos_weight:
        Ratio of negative to positive samples, used as ``scale_pos_weight``
        for XGBoost.  Logistic Regression and Random Forest instead use
        ``class_weight="balanced"``.
    """
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    models: dict[str, object],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    """Fit each model and compute held-out + cross-validated metrics.

    Returns
    -------
    (metrics_df, roc_data, fitted)
        ``metrics_df``: one row per model (ROC-AUC, PR-AUC, precision, recall,
        F1, CV ROC-AUC mean/std).
        ``roc_data``: {model_name: (fpr, tpr)} for the ROC plot.
        ``fitted``: {model_name: fitted_estimator}.
    """
    cv = StratifiedKFold(n_splits=CHURN_CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    rows = []
    roc_data: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr)

        rows.append({
            "Model": name,
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "CV_ROC_AUC_mean": round(cv_scores.mean(), 4),
            "CV_ROC_AUC_std": round(cv_scores.std(), 4),
        })
        logger.info(
            "%s: ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f CV-AUC=%.4f+/-%.4f",
            name, roc_auc, pr_auc, f1, cv_scores.mean(), cv_scores.std(),
        )

    metrics_df = pd.DataFrame(rows).set_index("Model")
    return metrics_df, roc_data, fitted


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_roc(roc_data: dict[str, np.ndarray], metrics_df: pd.DataFrame) -> None:
    """Save overlaid ROC curves for all models with AUC in the legend."""
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {"LogisticRegression": "#4C72B0", "RandomForest": "#55A868",
               "XGBoost": "#C44E52"}
    for name, (fpr, tpr) in roc_data.items():
        auc = metrics_df.loc[name, "ROC_AUC"]
        ax.plot(fpr, tpr, linewidth=2, color=palette.get(name, "#333"),
                label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1.2, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("Churn model ROC curves\n(held-out test set)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROC_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved to %s", ROC_PNG)


def _plot_importance(model: object, model_name: str) -> None:
    """Save a feature-importance bar chart from a fitted tree model."""
    if not hasattr(model, "feature_importances_"):
        logger.info("%s has no feature_importances_; skipping importance plot.",
                    model_name)
        return
    importances = pd.Series(model.feature_importances_, index=CHURN_FEATURES)
    importances = importances.sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(importances.index, importances.values, color="#55A868", alpha=0.85)
    ax.set_xlabel("Feature importance", fontsize=10)
    ax.set_title(f"Churn drivers — {model_name} feature importance",
                 fontsize=11, fontweight="bold")
    for i, v in enumerate(importances.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(IMPORTANCE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", IMPORTANCE_PNG)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_churn(features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train and compare churn models; score every customer with the best one.

    Parameters
    ----------
    features:
        Customer feature table.  Pass ``None`` to load from
        ``data/processed/customer_features.parquet``.

    Returns
    -------
    pd.DataFrame
        Per-customer table: ``Customer ID``, ``churn_label``,
        ``churn_probability`` (from the best model).  Saved to
        ``data/processed/customer_churn.parquet``.
    """
    if features is None:
        logger.info("Loading customer features from %s", CUSTOMER_FEATURES_PARQUET)
        features = pd.read_parquet(CUSTOMER_FEATURES_PARQUET)

    y = _build_label(features)
    X = features[CHURN_FEATURES].values

    # Standardise features: required for Logistic Regression, harmless for trees.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    logger.info("Train/test split: %d train, %d test (stratified).",
                len(X_train), len(X_test))

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = _build_models(pos_weight)

    metrics_df, roc_data, fitted = _evaluate(
        models, X_train, X_test, y_train, y_test,
    )

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(COMPARISON_CSV)
    logger.info("Model comparison saved to %s\n%s",
                COMPARISON_CSV, metrics_df.to_string())

    # Best model by held-out ROC-AUC.
    best_name = metrics_df["ROC_AUC"].idxmax()
    best_model = fitted[best_name]
    logger.info("Best churn model: %s (ROC-AUC=%.4f).",
                best_name, metrics_df.loc[best_name, "ROC_AUC"])

    _plot_roc(roc_data, metrics_df)
    # Prefer a tree model for the importance plot if the best is linear.
    imp_name = best_name if hasattr(best_model, "feature_importances_") else "RandomForest"
    _plot_importance(fitted[imp_name], imp_name)

    # Score every customer with the best model (re-fit on all data so the
    # probabilities use the full sample, not just the training split).
    best_model.fit(X_scaled, y)
    churn_prob = best_model.predict_proba(X_scaled)[:, 1]

    out = pd.DataFrame({
        "Customer ID": features["Customer ID"].values,
        "churn_label": y.values,
        "churn_probability": churn_prob,
    })
    CUSTOMER_CHURN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CUSTOMER_CHURN_PARQUET, index=False)
    logger.info("Customer churn scores (%d rows) saved to %s",
                len(out), CUSTOMER_CHURN_PARQUET)

    return out
