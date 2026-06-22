"""
Central configuration for the customer segmentation pipeline.

All file paths, hyper-parameter constants, and tunable flags live here so
that the rest of the codebase never hard-codes a path or magic number.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUTS_FIGURES_DIR: Path = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_TABLES_DIR: Path = PROJECT_ROOT / "outputs" / "tables"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
APP_DIR: Path = PROJECT_ROOT / "app"
SRC_DIR: Path = PROJECT_ROOT / "src"

# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
RAW_EXCEL_PATH: Path = DATA_RAW_DIR / "raj.xlsx"
SHEET_2009_2010: str = "Year 2009-2010"
SHEET_2010_2011: str = "Year 2010-2011"

# ---------------------------------------------------------------------------
# Processed data artefacts
# ---------------------------------------------------------------------------
CLEANED_PARQUET: Path = DATA_PROCESSED_DIR / "cleaned_transactions.parquet"
CUSTOMER_FEATURES_PARQUET: Path = DATA_PROCESSED_DIR / "customer_features.parquet"
SCALED_FEATURES_PARQUET: Path = DATA_PROCESSED_DIR / "scaled_features.parquet"
SCALER_PATH: Path = DATA_PROCESSED_DIR / "fitted_scaler.joblib"
RFM_PARQUET: Path = DATA_PROCESSED_DIR / "rfm_features.parquet"
CLV_PARQUET: Path = DATA_PROCESSED_DIR / "clv_features.parquet"
CLUSTER_PARQUET: Path = DATA_PROCESSED_DIR / "clustered_customers.parquet"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
SKEW_THRESHOLD: float = 0.5   # features with |skew| > threshold receive log1p

# ---------------------------------------------------------------------------
# Data-cleaning thresholds
# ---------------------------------------------------------------------------
MIN_QUANTITY: int = 1                   # drop returns / cancellations
MIN_UNIT_PRICE: float = 0.01            # drop zero-price anomalies
CANCELLATION_PREFIX: str = "C"          # invoices starting with C are returns

# ---------------------------------------------------------------------------
# RFM feature engineering
# ---------------------------------------------------------------------------
RFM_SNAPSHOT_DATE_OFFSET_DAYS: int = 1  # snapshot = max(InvoiceDate) + offset

# ---------------------------------------------------------------------------
# BG/NBD + Gamma-Gamma CLV model (lifetimes)
# ---------------------------------------------------------------------------
BGMD_PENALIZER_COEF: float = 0.001
GAMMA_GAMMA_PENALIZER_COEF: float = 0.001
CLV_TIME_MONTHS: int = 12               # forecast horizon in months
CLV_FORECAST_DAYS: list[int] = [90, 180, 365]  # purchase-count forecast horizons
CLV_DISCOUNT_RATE_MONTHLY: float = 0.01  # ~12.7% annual; for DCF in customer_lifetime_value
CUSTOMER_CLV_PARQUET: Path = DATA_PROCESSED_DIR / "customer_clv.parquet"

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
# K-Means sweep range
KMEANS_K_MIN: int = 2
KMEANS_K_MAX: int = 10
# Set to an int to override automatic silhouette-based selection; None = auto.
KMEANS_BEST_K: int | None = None

# ---------------------------------------------------------------------------
# Cluster validation
# ---------------------------------------------------------------------------
N_BOOTSTRAP_ITERATIONS: int = 50      # bootstrap resampling rounds for ARI stability
ARI_STABILITY_THRESHOLD: float = 0.70 # min mean ARI to declare a solution "stable"
MAX_HDBSCAN_NOISE_FRACTION: float = 0.30  # above this, HDBSCAN is penalised in selection

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE: int = 50
HDBSCAN_MIN_SAMPLES: int = 5

# DBSCAN
DBSCAN_MIN_SAMPLES: int = 5        # also used as k in the k-distance plot
DBSCAN_EPS: float | None = None    # None = auto-detected from k-distance knee

# Gaussian Mixture Model
GMM_N_MIN: int = 2
GMM_N_MAX: int = 10
GMM_BEST_N: int | None = None      # None = auto from lowest BIC

# ---------------------------------------------------------------------------
# XGBoost cluster classifier
# ---------------------------------------------------------------------------
XGB_N_ESTIMATORS: int = 300
XGB_MAX_DEPTH: int = 5
XGB_LEARNING_RATE: float = 0.05
XGB_SUBSAMPLE: float = 0.8
XGB_COLSAMPLE_BYTREE: float = 0.8
TEST_SIZE: float = 0.2

# ---------------------------------------------------------------------------
# Churn classification (Phase 8)
# ---------------------------------------------------------------------------
CHURN_RECENCY_PERCENTILE: float = 0.90   # Recency above this percentile = churned
CHURN_CV_FOLDS: int = 5                  # stratified CV folds for ROC-AUC
CUSTOMER_CHURN_PARQUET: Path = DATA_PROCESSED_DIR / "customer_churn.parquet"

# ---------------------------------------------------------------------------
# Notification / marketing rules (Phase 10)
# ---------------------------------------------------------------------------
# Churn-probability bands used to escalate / de-prioritise campaigns.
NOTIF_CHURN_HIGH: float = 0.50    # >= this churn prob = high risk
NOTIF_CHURN_MED: float = 0.25     # >= this (and < HIGH) = medium risk
# CLV quantiles used to assign value tiers (computed on the customer base).
NOTIF_CLV_HIGH_Q: float = 0.80    # >= 80th pct CLV = high value
NOTIF_CLV_LOW_Q: float = 0.40     # <  40th pct CLV = low value
# Fraction of a customer's typical inter-purchase gap at which to re-contact.
NOTIF_CADENCE_FRACTION: float = 0.6
NOTIF_DEFAULT_CONTACT_DAYS: int = 30   # fallback for one-time buyers
NOTIFICATION_PLAN_CSV: Path = OUTPUTS_TABLES_DIR / "notification_plan.csv"

# ---------------------------------------------------------------------------
# Monte Carlo ROI simulation (Phase 11)
# ---------------------------------------------------------------------------
ROI_N_SIMULATIONS: int = 10_000   # number of Monte Carlo iterations
ROI_CI_LEVEL: float = 0.95        # central credible interval width
# Beta concentration for the response-rate prior (higher = tighter).
ROI_RESPONSE_CONCENTRATION: float = 150.0
# Per-conversion revenue uncertainty (multiplier ~ Normal(1, this), clipped >0).
ROI_REVENUE_NOISE_SD: float = 0.20
# Financial conversion of gross order value -> profit contribution.
ROI_GROSS_MARGIN: float = 0.30    # retail gross margin on an order
ROI_OFFER_DISCOUNT: float = 0.10  # mean promotional discount given in offers
# Per-contact channel costs (currency), summed across a campaign's channels.
ROI_CHANNEL_COSTS: dict[str, float] = {
    "Email": 0.05,
    "App push": 0.02,
    "SMS": 0.15,
    "Personal outreach": 5.00,
}
