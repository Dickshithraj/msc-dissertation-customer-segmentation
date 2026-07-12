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
- Gradient Boosting    (sklearn boosted trees)
- Decision Tree        (single interpretable tree, class_weight="balanced")
- K-Nearest Neighbours (distance-weighted instance-based baseline)

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
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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
MCNEMAR_CSV = OUTPUTS_TABLES_DIR / "churn_mcnemar_pvalues.csv"
RANKING_CSV = OUTPUTS_TABLES_DIR / "churn_model_ranking.csv"
ROC_PNG = OUTPUTS_FIGURES_DIR / "churn_roc_curves.png"
PR_PNG = OUTPUTS_FIGURES_DIR / "churn_pr_curves.png"
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
    """Instantiate the candidate classifiers with imbalance handling where supported.

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
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
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
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray],
           dict[str, np.ndarray], dict[str, object]]:
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
    pr_data: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        preds[name] = pred

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba)
        pr_data[name] = (rec_curve, prec_curve)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

        rows.append({
            "Model": name,
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "CV_ROC_AUC_mean": round(cv_scores.mean(), 4),
            "CV_ROC_AUC_std": round(cv_scores.std(), 4),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })
        logger.info(
            "%s: ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f CV-AUC=%.4f+/-%.4f",
            name, roc_auc, pr_auc, f1, cv_scores.mean(), cv_scores.std(),
        )

    metrics_df = pd.DataFrame(rows).set_index("Model")
    return metrics_df, roc_data, pr_data, preds, fitted


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def _mcnemar_pvalue(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Exact McNemar p-value for two models' correctness arrays on the same test set."""
    b = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, p=0.5).pvalue)


def _mcnemar_tests(preds: dict[str, np.ndarray], y_test: pd.Series) -> pd.DataFrame:
    """Pairwise McNemar p-value matrix across all models.

    McNemar's test compares two classifiers on the *same* test samples and asks
    whether their disagreements are statistically significant.  p < 0.05 means
    the two models make significantly different errors (one is genuinely better);
    p >= 0.05 means the score gap between them is not statistically meaningful.
    """
    names = list(preds.keys())
    y = np.asarray(y_test)
    correct = {nm: (np.asarray(preds[nm]) == y) for nm in names}
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            mat.loc[a, b] = 1.0 if a == b else _mcnemar_pvalue(correct[a], correct[b])
    return mat


def _rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models on each key metric and combine into a mean overall rank.

    Gives a single 'who wins overall' view across ROC-AUC, PR-AUC, F1 and
    cross-validated AUC (rank 1 = best on that metric; lowest mean rank = best
    overall).
    """
    higher_better = ["ROC_AUC", "PR_AUC", "F1", "CV_ROC_AUC_mean"]
    ranks = pd.DataFrame(index=metrics_df.index)
    for col in higher_better:
        ranks[f"{col}_rank"] = metrics_df[col].rank(ascending=False, method="min")
    ranks["mean_rank"] = ranks.mean(axis=1).round(2)
    ranks = ranks.sort_values("mean_rank")
    ranks["overall_rank"] = range(1, len(ranks) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_roc(roc_data: dict[str, np.ndarray], metrics_df: pd.DataFrame) -> None:
    """Save overlaid ROC curves for all models with AUC in the legend."""
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = ["#4C72B0", "#55A868", "#C44E52", "#DD8452",
               "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
    for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
        auc = metrics_df.loc[name, "ROC_AUC"]
        ax.plot(fpr, tpr, linewidth=2, color=palette[i % len(palette)],
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


def _plot_pr(pr_data: dict[str, np.ndarray], metrics_df: pd.DataFrame) -> None:
    """Save overlaid precision-recall curves with PR-AUC in the legend.

    Under the strong class imbalance of the churn label (~10% positive), the
    precision-recall curve is more informative than ROC: it focuses on the
    minority (churner) class that the campaign actually cares about.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = ["#4C72B0", "#55A868", "#C44E52", "#DD8452",
               "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
    for i, (name, (rec_curve, prec_curve)) in enumerate(pr_data.items()):
        pr_auc = metrics_df.loc[name, "PR_AUC"]
        ax.plot(rec_curve, prec_curve, linewidth=2, color=palette[i % len(palette)],
                label=f"{name} (PR-AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Churn model precision-recall curves\n"
                 "(held-out test set; robust to class imbalance)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    OUTPUTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PR_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curves saved to %s", PR_PNG)


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

    metrics_df, roc_data, pr_data, preds, fitted = _evaluate(
        models, X_train, X_test, y_train, y_test,
    )

    OUTPUTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(COMPARISON_CSV)
    logger.info("Model comparison saved to %s\n%s",
                COMPARISON_CSV, metrics_df.to_string())

    # Statistical comparison: pairwise significance + overall ranking.
    mcnemar = _mcnemar_tests(preds, y_test)
    mcnemar.to_csv(MCNEMAR_CSV)
    logger.info("McNemar pairwise p-values saved to %s\n%s",
                MCNEMAR_CSV, mcnemar.round(4).to_string())
    ranking = _rank_models(metrics_df)
    ranking.to_csv(RANKING_CSV)
    logger.info("Model ranking saved to %s\n%s", RANKING_CSV, ranking.to_string())

    # Best model by held-out ROC-AUC.
    best_name = metrics_df["ROC_AUC"].idxmax()
    best_model = fitted[best_name]
    logger.info("Best churn model: %s (ROC-AUC=%.4f).",
                best_name, metrics_df.loc[best_name, "ROC_AUC"])

    _plot_roc(roc_data, metrics_df)
    _plot_pr(pr_data, metrics_df)
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
