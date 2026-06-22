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
# Notification / marketing rules (cluster-based)
# ---------------------------------------------------------------------------
NOTIFICATION_CONFIG: dict[str, dict] = {
    # Keys are cluster label strings; values are campaign parameters.
    # Populated once cluster profiles are stable.
}
