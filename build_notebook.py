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
LOCAL_IPYNB = PROJECT_ROOT / "Dissertation_Pipeline.ipynb"
DOWNLOAD_IPYNB = DOWNLOADS / "Dissertation_Pipeline.ipynb"

PROJECT_STR = str(PROJECT_ROOT)


def _md(text: str):
    return new_markdown_cell(text)


def _code(src: str):
    return new_code_cell(src)


def build_cells() -> list:
    cells = []

    # ── Title page ────────────────────────────────────────────────────────────
    cells.append(_md(
        "# Customer Segmentation and Cluster-Based Marketing Notifications\n"
        "### An End-to-End Data Science Pipeline on the UCI Online Retail II Dataset\n\n"
        "**MSc Data Science Dissertation — University of Leeds (COMP5200M)**\n\n"
        "---\n\n"
        "## Abstract\n\n"
        "This project develops a reproducible, end-to-end analytical pipeline that "
        "transforms approximately one million raw e-commerce transactions into an "
        "actionable, customer-level marketing strategy. Customers are first described "
        "through Recency–Frequency–Monetary (RFM) features and four extended "
        "behavioural signals, then segmented using four clustering algorithms "
        "(K-Means, DBSCAN, Gaussian Mixture Models, and HDBSCAN) whose solutions are "
        "compared with internal validity metrics and bootstrap stability analysis. "
        "The selected segmentation is enriched with probabilistic Customer Lifetime "
        "Value (CLV) estimates (BG/NBD + Gamma-Gamma), supervised churn-risk models "
        "(Logistic Regression, Random Forest, XGBoost), and a year-on-year segment "
        "migration analysis. These signals feed a transparent, rule-based "
        "notification engine, whose financial value is quantified through a Monte "
        "Carlo Return-on-Investment (ROI) simulation.\n\n"
        "Every result presented below is produced by executing the project's own "
        "source modules (`src/`); the code and its output appear together in each "
        "cell, so the notebook is a faithful, runnable record of the methodology."
    ))

    cells.append(_md(
        "## Dataset\n\n"
        "The **UCI Online Retail II** dataset records all transactions for a "
        "UK-based online gift retailer between 01 December 2009 and 09 December 2011, "
        "split across two annual sheets. Each row is a single invoice line item with "
        "the fields: `Invoice`, `StockCode`, `Description`, `Quantity`, "
        "`InvoiceDate`, `Price`, `Customer ID`, and `Country`.\n\n"
        "> Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5CG6D\n\n"
        "## Pipeline overview\n\n"
        "| Stage | Task | Principal technique |\n"
        "|:-----:|------|---------------------|\n"
        "| 1 | Data loading & cleaning | Rule-based filtering with an audit trail |\n"
        "| 2 | Feature engineering | RFM + extended behavioural features |\n"
        "| 2b | Preprocessing | `log1p` + `StandardScaler` |\n"
        "| 3–4 | Clustering | K-Means, DBSCAN, GMM, HDBSCAN |\n"
        "| 3b | Validation | Silhouette / DB / CH + bootstrap ARI |\n"
        "| 6 | Profiling | Rule-based segment naming |\n"
        "| 7 | Lifetime value | BG/NBD + Gamma-Gamma |\n"
        "| 8 | Churn modelling | LogReg / Random Forest / XGBoost |\n"
        "| 9 | Migration | Year-on-year transition matrix |\n"
        "| 10 | Notifications | Rule-based campaign engine |\n"
        "| 11 | ROI | Monte Carlo simulation |\n"
    ))

    # ── Setup ─────────────────────────────────────────────────────────────────
    cells.append(_md(
        "## Environment setup\n\n"
        "The cell below puts the project on the import path and configures display "
        "options. Informational logging is suppressed so that each cell shows only "
        "its analytical result. All randomness in the pipeline is seeded "
        "(`config.RANDOM_STATE = 42`) so the notebook is fully reproducible."
    ))
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
        "---\n"
        "## Stage 1 — Data Loading and Cleaning\n\n"
        "**Objective.** Convert the two raw Excel sheets into a single, trustworthy "
        "transaction table suitable for customer-level analysis.\n\n"
        "**Methodology.** Four sequential cleaning rules are applied, each measured "
        "against the already-filtered data so removals are never double-counted: "
        "(1) drop rows with a missing `Customer ID` (they cannot be attributed to "
        "any customer); (2) remove cancelled invoices (prefixed `C`, which carry "
        "negative quantities); (3) remove non-positive quantities or prices "
        "(data-entry errors and free samples); and (4) drop exact duplicate rows "
        "(double-loaded records). A line-level `TotalPrice = Quantity × Price` is "
        "then derived as the basis for the Monetary dimension.\n\n"
        "The cleaning is executed once by `main.py`; here we load the cleaned "
        "artefact and display the audit table it produced."
    ))
    cells.append(_code(
        "cleaned = pd.read_parquet('data/processed/cleaned_transactions.parquet')\n"
        "summary = pd.read_csv('outputs/tables/cleaning_summary.csv')\n"
        "print(f'Cleaned transactions: {len(cleaned):,} rows, "
        "{cleaned[\"Customer ID\"].nunique():,} unique customers')\n"
        "display(summary)"
    ))
    cells.append(_md(
        "**Interpretation.** Cleaning reduces the raw ~1.07M rows to **793,591** "
        "valid transactions for **5,878** customers (about a quarter of rows "
        "removed). The single largest reduction is the removal of records with no "
        "`Customer ID`, which is expected for a guest-checkout retailer and is "
        "essential because every downstream model is customer-centric."
    ))

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 2 — Feature Engineering\n\n"
        "**Objective.** Summarise each customer's entire transaction history as one "
        "row of behavioural features.\n\n"
        "**Methodology.** The three canonical **RFM** dimensions are computed: "
        "*Recency* (days since last purchase, relative to a fixed snapshot date), "
        "*Frequency* (number of distinct invoices), and *Monetary* (total spend). "
        "These are augmented with four features that RFM alone cannot express: "
        "*Tenure*, *AvgOrderValue*, *AvgInterPurchaseDays*, and *DistinctProducts*. "
        "The snapshot date is fixed at `max(InvoiceDate) + 1 day` rather than the "
        "current date, guaranteeing identical Recency values on every run."
    ))
    cells.append(_code(
        "from src.features import build_customer_features\n"
        "features = build_customer_features()\n"
        "print(f'Feature table: {features.shape[0]:,} customers x "
        "{features.shape[1]} columns')\n"
        "display(features.head())\n"
        "display(features.describe().round(1))"
    ))
    cells.append(_code("Image('outputs/figures/rfm_distributions.png')"))
    cells.append(_md(
        "**Interpretation.** The summary statistics confirm the strong right-skew "
        "typical of retail data: median Frequency and Monetary are far below their "
        "means, indicating that a small number of high-value customers dominate "
        "revenue. This skew motivates the log-transformation applied in the next "
        "stage."
    ))

    # ── Stage 2b ──────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 2b — Preprocessing\n\n"
        "**Objective.** Place all features on a comparable scale so that "
        "distance-based clustering is not dominated by any single dimension.\n\n"
        "**Methodology.** Features whose absolute skewness exceeds 0.5 receive a "
        "`log1p` transform (`log(1 + x)`, which is defined at zero and therefore "
        "safe for one-time buyers). All features are then standardised with "
        "`StandardScaler` to zero mean and unit variance. The fitted scaler and the "
        "list of log-transformed columns are retained so cluster centroids can later "
        "be returned to their original, interpretable scale."
    ))
    cells.append(_code(
        "from src.preprocessing import preprocess_features\n"
        "prep = preprocess_features()\n"
        "print('log1p applied to:', ', '.join(prep.log1p_cols))\n"
        "print('Scaled matrix shape:', prep.X_scaled.shape)\n"
        "Image('outputs/figures/scaling_effect.png')"
    ))
    cells.append(_md(
        "**Interpretation.** Six of the seven features are log-compressed; the "
        "before/after panels show the heavily skewed raw distributions becoming "
        "approximately symmetric, which is the geometry K-Means implicitly assumes "
        "and which prevents extreme spenders from distorting the distance metric."
    ))

    # ── Stage 3-4 ─────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 3–4 — Clustering (Four Algorithms)\n\n"
        "**Objective.** Discover natural customer segments without imposing a "
        "pre-defined number of groups.\n\n"
        "**Methodology.** Four complementary algorithms are fitted on the scaled "
        "features: **K-Means** (centroid-based, with a silhouette sweep over "
        "k = 2…10), **DBSCAN** (density-based, with `eps` chosen automatically from "
        "the k-distance knee), a **Gaussian Mixture Model** (probabilistic, with the "
        "component count chosen by the Bayesian Information Criterion), and "
        "**HDBSCAN** (hierarchical density-based, which labels low-density points as "
        "noise). Principal Component Analysis is used only for 2-D visualisation, "
        "never as a clustering input."
    ))
    cells.append(_code(
        "from src.clustering import run_all_clustering\n"
        "clustering = run_all_clustering()\n"
        "cluster_df = clustering['cluster_df']\n"
        "metrics = pd.read_csv('outputs/tables/kmeans_metrics.csv')\n"
        "print('K-Means model-selection sweep (k = 2 to 10):')\n"
        "display(metrics)\n"
        "Image('outputs/figures/cluster_pca_projection.png')"
    ))
    cells.append(_md(
        "**Interpretation.** For K-Means the silhouette score is maximised at "
        "**k = 2**. The PCA panels reveal the qualitatively different behaviour of "
        "each algorithm: K-Means and GMM partition the whole space, whereas the "
        "density-based methods isolate a dense core and label dispersed "
        "high-spenders as noise. The next stage adjudicates between them objectively."
    ))

    # ── Stage 3b ──────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 3b — Validation and Stability\n\n"
        "**Objective.** Select the most trustworthy segmentation using objective, "
        "reproducible criteria rather than visual judgement.\n\n"
        "**Methodology.** Three internal validity indices are computed for each "
        "algorithm — **Silhouette** (higher is better), **Davies–Bouldin** (lower "
        "is better), and **Calinski–Harabasz** (higher is better) — with noise "
        "points excluded for the density-based methods. **Bootstrap stability** is "
        "then assessed by refitting each algorithm on 50 resamples and measuring the "
        "Adjusted Rand Index (ARI) against the reference partition. The selection "
        "rule disqualifies excessively noisy solutions, requires mean ARI ≥ 0.70, "
        "and finally chooses the highest silhouette."
    ))
    cells.append(_code(
        "from src.validation import run_validation\n"
        "validation = run_validation(X=prep.X_scaled)\n"
        "print('Internal validity metrics:')\n"
        "display(validation.metrics_df)\n"
        "print('\\nBootstrap stability (Adjusted Rand Index):')\n"
        "display(validation.stability_df)\n"
        "print('\\nSelected algorithm:', validation.best_algorithm)\n"
        "Image('outputs/figures/stability_ari.png')"
    ))
    cells.append(_md(
        "**Interpretation.** **HDBSCAN** is selected: it achieves the highest "
        "silhouette (~0.42) while remaining stable (mean ARI ~0.84). K-Means is the "
        "most stable algorithm overall (ARI ~0.98) but scores lower on silhouette; "
        "DBSCAN collapses into a single cluster and is correctly disqualified. This "
        "demonstrates why stability and validity must be assessed jointly rather "
        "than relying on a single run."
    ))

    # ── Stage 6 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 6 — Segment Profiling and Naming\n\n"
        "**Objective.** Translate abstract cluster labels into interpretable, "
        "marketing-ready customer segments.\n\n"
        "**Methodology.** For the selected algorithm, each cluster's mean feature "
        "values are computed on the **original (un-scaled)** scale, then mapped to a "
        "business name (Champions, Loyal Customers, At-Risk, Lost, etc.) by a "
        "transparent rule set expressed as multiples of the overall customer mean. "
        "Results are visualised as a z-score heatmap and a min-max radar chart."
    ))
    cells.append(_code(
        "from src.profiling import profile_clusters\n"
        "profiles = profile_clusters(algo=validation.best_algorithm)\n"
        "display(profiles)\n"
        "display(Image('outputs/figures/segment_profiles.png'))\n"
        "Image('outputs/figures/radar_profiles.png')"
    ))
    cells.append(_md(
        "**Interpretation.** The profiling reveals a notable property of "
        "density-based segmentation: HDBSCAN's *noise* cluster actually contains the "
        "highest-value customers (very high Frequency and Monetary). Because such "
        "premium buyers are heterogeneous outliers, the algorithm declines to force "
        "them into a dense group — a genuine limitation worth noting when "
        "density-based methods are used for marketing segmentation."
    ))

    # ── Stage 7 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 7 — Customer Lifetime Value\n\n"
        "**Objective.** Estimate the forward-looking monetary value of each "
        "customer to prioritise marketing investment.\n\n"
        "**Methodology.** Two complementary probabilistic models from the "
        "`lifetimes` library are combined. The **BG/NBD** model captures the "
        "*purchasing process* — how many future transactions a customer will make "
        "and the probability they are still active — while the **Gamma-Gamma** model "
        "captures the *monetary process* — their expected average order value. Their "
        "product, discounted over a 12-month horizon, yields each customer's CLV; "
        "90/180/365-day purchase forecasts are also produced."
    ))
    cells.append(_code(
        "from src.clv import build_clv\n"
        "clv_df = build_clv(transactions=cleaned)\n"
        "print(f'Total portfolio CLV: GBP {clv_df[\"clv\"].sum():,.0f}')\n"
        "print(f'Median customer CLV: GBP {clv_df[\"clv\"].median():,.0f}')\n"
        "top_decile = clv_df['clv'].quantile(0.90)\n"
        "share = clv_df.loc[clv_df['clv'] >= top_decile, 'clv'].sum() / clv_df['clv'].sum() * 100\n"
        "print(f'Top-decile customers hold {share:.1f}% of total CLV')\n"
        "display(clv_df.nlargest(10, 'clv')[['Customer ID','clv','prob_alive',"
        "'pred_purchases_365d']].reset_index(drop=True))\n"
        "Image('outputs/figures/clv_distribution.png')"
    ))
    cells.append(_md(
        "**Interpretation.** Total predicted 12-month portfolio value is "
        "approximately **£8.3M**, and the **top decile of customers holds ~65%** of "
        "it. This strong Pareto concentration is the central business justification "
        "for segment-targeted marketing: a small, identifiable group accounts for "
        "the majority of future value and warrants disproportionate retention "
        "investment."
    ))

    # ── Stage 8 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 8 — Churn Classification\n\n"
        "**Objective.** Predict which customers are at risk of lapsing so they can "
        "be targeted before they leave.\n\n"
        "**Methodology.** A customer is labelled *churned* if their Recency exceeds "
        "the 90th percentile. Three classifiers — **Logistic Regression**, **Random "
        "Forest**, and **XGBoost** — are compared on a stratified hold-out set and "
        "via 5-fold cross-validation, ranked by ROC-AUC. Crucially, **Recency is "
        "excluded from the predictors** because the label is derived from it; "
        "including it would constitute target leakage and produce a meaningless "
        "near-perfect score. The models therefore learn churn from purely "
        "behavioural signals."
    ))
    cells.append(_code(
        "from src.churn import run_churn\n"
        "churn_df = run_churn(features=features)\n"
        "comparison = pd.read_csv('outputs/tables/churn_model_comparison.csv', index_col=0)\n"
        "display(comparison)\n"
        "display(Image('outputs/figures/churn_roc_curves.png'))\n"
        "Image('outputs/figures/churn_feature_importance.png')"
    ))
    cells.append(_md(
        "**Interpretation.** All three models reach a ROC-AUC of about **0.85**, "
        "with Random Forest marginally best. That the score is ~0.85 rather than "
        "~1.0 is the desired outcome: it confirms the leakage guard worked and the "
        "models are learning genuine behavioural patterns. The low PR-AUC (~0.29) "
        "honestly reflects the 10% class imbalance, which is why ROC-AUC and class "
        "weighting are used throughout."
    ))

    # ── Stage 9 ───────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 9 — Year-on-Year Segment Migration\n\n"
        "**Objective.** Quantify how customers move between segments from one year "
        "to the next.\n\n"
        "**Methodology.** Each customer is segmented *independently within each "
        "year* using the same deterministic RFM naming rules, so labels are directly "
        "comparable across time (unlike cluster IDs). A transition matrix is then "
        "built over customers present in both years, reported both as raw counts and "
        "as row-normalised probabilities, alongside the sizes of the retained, "
        "lapsed, and newly-acquired cohorts."
    ))
    cells.append(_code(
        "from src.migration import run_migration\n"
        "migration = run_migration(transactions=cleaned)\n"
        "print('Cohort sizes:', migration['context'])\n"
        "print('\\nTransition probabilities (%, row = Year 1 origin):')\n"
        "display((migration['rates'] * 100).round(1))\n"
        "Image('outputs/figures/segment_migration.png')"
    ))
    cells.append(_md(
        "**Interpretation.** Around **2,770 customers are retained** across both "
        "years. **Champions are the most loyal segment** (~62% remain Champions), "
        "validating retention spend on this group, whereas the At-Risk segment never "
        "stays put — making it the prime target for timely intervention. The "
        "dominant leakage path (Potential Loyalists → General/Lost) directly informs "
        "the campaign rules in the next stage."
    ))

    # ── Stage 10 ──────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 10 — Rule-Based Notification Engine\n\n"
        "**Objective.** Convert the analytical signals into a concrete, auditable "
        "marketing action for every customer.\n\n"
        "**Methodology.** Each customer's **segment**, **CLV tier**, and **churn-risk "
        "band** drive a transparent decision matrix that outputs a campaign action, "
        "channel, offer, priority, and recommended contact window. The three signals "
        "are complementary: the segment sets the baseline strategy, churn risk sets "
        "urgency, and CLV sets budget. A deliberately budget-aware interaction "
        "ensures expensive personal outreach is reserved for high-value at-risk "
        "customers, while low-value churners receive only a low-cost automated "
        "nudge. The engine exposes `recommend(customer_id)` for single lookups."
    ))
    cells.append(_code(
        "from src.notifications import generate_notifications, recommend\n"
        "plan = generate_notifications()\n"
        "print('Campaign action distribution across the customer base:')\n"
        "display(plan['action'].value_counts().rename('customers').to_frame())\n"
        "print('\\nExample: full recommendation for a single customer')\n"
        "pd.Series(recommend(int(plan['Customer ID'].iloc[0]))).to_frame('value')"
    ))
    cells.append(_md(
        "**Interpretation.** The plan assigns every one of the 5,878 customers to a "
        "campaign. The budget logic is visible in the distribution: the large "
        "'low-cost automated reactivation' group are high-churn but low-value "
        "customers routed to a single cheap email, whereas only a small, high-value "
        "at-risk group triggers the expensive 'priority retention intervention'."
    ))

    # ── Stage 11 ──────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Stage 11 — Monte Carlo ROI Simulation\n\n"
        "**Objective.** Estimate the financial return of executing the notification "
        "plan, with an honest quantification of uncertainty.\n\n"
        "**Methodology.** Because response rates, per-conversion revenue, and costs "
        "are uncertain, a single point estimate would be fragile. The simulation "
        "runs **10,000 iterations**, drawing each campaign's response rate from a "
        "**Beta** prior, conversions from a **Binomial** distribution, and revenue "
        "from the mean order value with multiplicative noise. Gross order value is "
        "converted to profit contribution via a retail gross margin net of the "
        "promotional discount — avoiding the naive assumption that the full order "
        "value is profit. The output is a *distribution* of ROI with a 95% credible "
        "interval."
    ))
    cells.append(_code(
        "from src.roi import run_roi_simulation\n"
        "roi_summary = run_roi_simulation(plan=plan, clv=clv_df)\n"
        "display(roi_summary)\n"
        "Image('outputs/figures/roi_distribution.png')"
    ))
    cells.append(_md(
        "**Interpretation.** The campaign is profitable in **100% of simulations**, "
        "with a mean ROI of roughly **60×** and a 95% credible interval of about "
        "**[44×, 77×]**. While high, this is the correct order of magnitude for "
        "digital-first campaigns: the near-zero marginal cost of email and app "
        "channels mirrors the widely-cited ~36:1 email-marketing benchmark. The "
        "credible interval, rather than a single number, communicates the residual "
        "uncertainty honestly."
    ))

    # ── Conclusion ────────────────────────────────────────────────────────────
    cells.append(_md(
        "---\n"
        "## Conclusion\n\n"
        "This notebook has demonstrated a complete, reproducible journey from raw "
        "transactions to a costed marketing strategy. The key findings are:\n\n"
        "- **Segmentation.** HDBSCAN provides the most valid and stable segmentation "
        "(silhouette ~0.42, ARI ~0.84), selected by an objective rule rather than "
        "visual inspection.\n"
        "- **Value concentration.** Predicted 12-month portfolio CLV is ~£8.3M, with "
        "the top decile of customers holding ~65% of it — a strong mandate for "
        "targeted retention.\n"
        "- **Churn.** Behavioural models predict lapse risk at ROC-AUC ~0.85 with a "
        "deliberate leakage guard, identifying at-risk customers from behaviour "
        "alone.\n"
        "- **Dynamics.** Year-on-year migration shows Champions are the stickiest "
        "segment, while At-Risk customers require immediate intervention.\n"
        "- **Action and value.** A transparent, budget-aware notification engine "
        "assigns a campaign to every customer, and a Monte Carlo simulation projects "
        "a robustly positive ROI.\n\n"
        "**Limitations and future work.** CLV estimates inherit the Gamma-Gamma "
        "assumptions (the fitted shape parameter q < 1 makes population baselines "
        "unreliable for one-time buyers); the ROI response-rate priors are planning "
        "assumptions rather than measured uplift, and would ideally be calibrated "
        "against a control group via A/B testing. Future work could incorporate "
        "product-level recommendations and a uplift-modelling framework to estimate "
        "*incremental* rather than *gross* campaign response.\n\n"
        "**Reproducibility.** The entire pipeline runs end-to-end via `python "
        "main.py`; an interactive dashboard is available via "
        "`streamlit run app/streamlit_app.py`; and the test suite (`pytest tests/`) "
        "validates the core logic, including the churn leakage guard."
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
