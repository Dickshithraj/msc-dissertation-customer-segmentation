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

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
# K-Means sweep range
KMEANS_K_MIN: int = 2
KMEANS_K_MAX: int = 10

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE: int = 50
HDBSCAN_MIN_SAMPLES: int = 5

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
# Notification / marketing rules (cluster-based)
# ---------------------------------------------------------------------------
NOTIFICATION_CONFIG: dict[str, dict] = {
    # Keys are cluster label strings; values are campaign parameters.
    # Populated once cluster profiles are stable.
}
