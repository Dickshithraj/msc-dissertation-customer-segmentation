"""
build_notebook.py — generate an executed Jupyter notebook of the pipeline.

Builds ``dissertation_pipeline.ipynb`` where each stage is a code cell that
calls the real pipeline function and shows its output (result tables as
DataFrames + figures as inline images) directly beneath the code.  The
notebook is then executed so every output is embedded, and a copy is saved to
the user's Downloads folder.

Run from the project root::

    python build_notebook.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
LOCAL_IPYNB = PROJECT_ROOT / "dissertation_pipeline.ipynb"
DOWNLOAD_IPYNB = DOWNLOADS / "dissertation_pipeline.ipynb"

PROJECT_STR = str(PROJECT_ROOT)


def _md(text: str):
    return new_markdown_cell(text)


def _code(src: str):
    return new_code_cell(src)


def build_cells() -> list:
    cells = []

    # ── Title ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "# Customer Segmentation & Marketing Pipeline\n"
        "**MSc Data Science dissertation — UCI Online Retail II**\n\n"
        "Each cell below runs a real pipeline stage and shows its output "
        "(result tables and figures) directly underneath. The slow data-loading "
        "and cleaning step is run once by `main.py`; this notebook reuses its "
        "saved artefacts so it executes in a few minutes."
    ))

    # ── Setup ─────────────────────────────────────────────────────────────────
    cells.append(_md("## Setup"))
    cells.append(_code(
        "import sys, os\n"
        f"PROJECT = r'{PROJECT_STR}'\n"
        "os.chdir(PROJECT)\n"
        "if PROJECT not in sys.path:\n"
        "    sys.path.insert(0, PROJECT)\n"
        "import logging\n"
        "logging.disable(logging.INFO)  # keep cell output focused on results\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "pd.set_option('display.max_columns', 30)\n"
        "pd.set_option('display.width', 160)\n"
        "print('Setup complete. Working directory:', os.getcwd())"
    ))

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 1 — Data Loading & Cleaning\n"
        "Both Excel sheets are loaded, then cleaned (drop missing IDs, "
        "cancellations, invalid quantity/price, duplicates). The audit table "
        "below records every removal."
    ))
    cells.append(_code(
        "cleaned = pd.read_parquet('data/processed/cleaned_transactions.parquet')\n"
        "summary = pd.read_csv('outputs/tables/cleaning_summary.csv')\n"
        "print(f'Cleaned transactions: {len(cleaned):,} rows, "
        "{cleaned[\"Customer ID\"].nunique():,} customers')\n"
        "display(summary)"
    ))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 2 — Feature Engineering\n"
        "RFM + four extended behavioural features, one row per customer."
    ))
    cells.append(_code(
        "from src.features import build_customer_features\n"
        "features = build_customer_features()\n"
        "print(f'Feature table: {features.shape[0]:,} customers x "
        "{features.shape[1]} columns')\n"
        "display(features.head())\n"
        "display(features.describe().round(1))\n"
        "Image('outputs/figures/rfm_distributions.png')"
    ))

    # ── Stage 2b ──────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 2b — Preprocessing\n"
        "`log1p` on skewed features, then `StandardScaler`."
    ))
    cells.append(_code(
        "from src.preprocessing import preprocess_features\n"
        "prep = preprocess_features()\n"
        "print('log1p applied to:', ', '.join(prep.log1p_cols))\n"
        "print('Scaled matrix shape:', prep.X_scaled.shape)\n"
        "Image('outputs/figures/scaling_effect.png')"
    ))

    # ── Stage 3-4 ─────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 3–4 — Clustering (4 algorithms)\n"
        "K-Means, DBSCAN, GMM, and HDBSCAN are fitted; cluster labels are saved "
        "for every customer."
    ))
    cells.append(_code(
        "from src.clustering import run_all_clustering\n"
        "clustering = run_all_clustering()\n"
        "cluster_df = clustering['cluster_df']\n"
        "metrics = pd.read_csv('outputs/tables/kmeans_metrics.csv')\n"
        "print('K-Means sweep (silhouette by k):')\n"
        "display(metrics)\n"
        "Image('outputs/figures/cluster_pca_projection.png')"
    ))

    # ── Stage 3b ──────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 3b — Validation & Stability\n"
        "Internal metrics plus bootstrap ARI stability; the best algorithm is "
        "selected automatically."
    ))
    cells.append(_code(
        "from src.validation import run_validation\n"
        "validation = run_validation(X=prep.X_scaled)\n"
        "print('Internal validity metrics:')\n"
        "display(validation.metrics_df)\n"
        "print('Bootstrap stability (ARI):')\n"
        "display(validation.stability_df)\n"
        "print('Selected algorithm:', validation.best_algorithm)\n"
        "Image('outputs/figures/stability_ari.png')"
    ))

    # ── Stage 6 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 6 — Segment Profiling\n"
        "Un-scaled per-segment means with rule-based marketing names."
    ))
    cells.append(_code(
        "from src.profiling import profile_clusters\n"
        "profiles = profile_clusters(algo=validation.best_algorithm)\n"
        "display(profiles)\n"
        "display(Image('outputs/figures/segment_profiles.png'))\n"
        "Image('outputs/figures/radar_profiles.png')"
    ))

    # ── Stage 7 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 7 — Customer Lifetime Value\n"
        "BG/NBD + Gamma-Gamma forecast of discounted 12-month CLV."
    ))
    cells.append(_code(
        "from src.clv import build_clv\n"
        "clv_df = build_clv(transactions=cleaned)\n"
        "print(f'Total portfolio CLV: GBP {clv_df[\"clv\"].sum():,.0f}')\n"
        "print(f'Median CLV: GBP {clv_df[\"clv\"].median():,.0f}')\n"
        "display(clv_df.nlargest(10, 'clv')[['Customer ID','clv','prob_alive',"
        "'pred_purchases_365d']].reset_index(drop=True))\n"
        "Image('outputs/figures/clv_distribution.png')"
    ))

    # ── Stage 8 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 8 — Churn Classification\n"
        "Logistic Regression / Random Forest / XGBoost, ranked by ROC-AUC. "
        "Recency is excluded from the features to avoid target leakage."
    ))
    cells.append(_code(
        "from src.churn import run_churn\n"
        "churn_df = run_churn(features=features)\n"
        "comparison = pd.read_csv('outputs/tables/churn_model_comparison.csv', index_col=0)\n"
        "display(comparison)\n"
        "display(Image('outputs/figures/churn_roc_curves.png'))\n"
        "Image('outputs/figures/churn_feature_importance.png')"
    ))

    # ── Stage 9 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 9 — Segment Migration\n"
        "Year-on-year segment transition matrix."
    ))
    cells.append(_code(
        "from src.migration import run_migration\n"
        "migration = run_migration(transactions=cleaned)\n"
        "print('Cohort:', migration['context'])\n"
        "print('Transition probabilities (%):')\n"
        "display((migration['rates'] * 100).round(1))\n"
        "Image('outputs/figures/segment_migration.png')"
    ))

    # ── Stage 10 ──────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 10 — Notification Engine\n"
        "Rule-based campaign per customer; `recommend(customer_id)` for lookups."
    ))
    cells.append(_code(
        "from src.notifications import generate_notifications, recommend\n"
        "plan = generate_notifications()\n"
        "print('Campaign action distribution:')\n"
        "display(plan['action'].value_counts().rename('customers').to_frame())\n"
        "print('Example recommendation for one customer:')\n"
        "recommend(int(plan['Customer ID'].iloc[0]))"
    ))

    # ── Stage 11 ──────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Stage 11 — Monte Carlo ROI\n"
        "10,000-iteration ROI simulation with a 95% credible interval."
    ))
    cells.append(_code(
        "from src.roi import run_roi_simulation\n"
        "roi_summary = run_roi_simulation(plan=plan, clv=clv_df)\n"
        "display(roi_summary)\n"
        "Image('outputs/figures/roi_distribution.png')"
    ))

    # ── Closing ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Summary\n"
        "All 11 analytical stages executed end-to-end with their outputs shown "
        "above. An interactive version is available via "
        "`streamlit run app/streamlit_app.py`."
    ))

    return cells


def main() -> None:
    nb = new_notebook(cells=build_cells())
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }

    print("Executing notebook (this runs the real pipeline; ~3-5 min)...")
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": PROJECT_STR}})

    nbformat.write(nb, LOCAL_IPYNB)
    print(f"Saved executed notebook -> {LOCAL_IPYNB}")

    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LOCAL_IPYNB, DOWNLOAD_IPYNB)
    size_kb = DOWNLOAD_IPYNB.stat().st_size / 1024
    print(f"Copied to Downloads -> {DOWNLOAD_IPYNB}  ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
