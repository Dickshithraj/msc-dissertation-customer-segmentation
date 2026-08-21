# Customer Segmentation & Cluster-Based Marketing Notifications

**MSc Data Science dissertation** — University of Leeds (COMP5200M)
Dataset: **UCI Online Retail II** (`data/raw/raj.xlsx`)

This pipeline takes ~1M raw transactions and produces customer segments,
lifetime-value forecasts, churn risk scores, year-on-year segment migration, a
rule-based marketing notification plan, and a Monte Carlo ROI estimate — all
reproducible from a single command.

---

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.11)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the raw data file
#    data/raw/raj.xlsx   (two sheets: "Year 2009-2010", "Year 2010-2011")

# 4. Run the full pipeline end-to-end
python main.py

# 5. Run the three post-pipeline validation analyses
python scripts/run_new_validations.py

# 6. Explore the results interactively
streamlit run app/streamlit_app.py
```

The pipeline prints a consolidated stage-by-stage summary at the end and writes
every artefact to `data/processed/` and `outputs/`.

---

## Pipeline stages

| Stage | Module | What it does |
|-------|--------|--------------|
| 1     | `src/data_loading.py`, `src/cleaning.py` | Load both Excel sheets; clean (drop missing IDs, cancellations, invalid qty/price, duplicates) with an audit trail |
| 2     | `src/features.py` | Build customer-level RFM + extended features (Tenure, AvgOrderValue, AvgInterPurchaseDays, DistinctProducts) |
| 2b    | `src/preprocessing.py` | `log1p` skewed features, then `StandardScaler` |
| 3     | `src/clustering.py` | Six algorithms across five families: K-Means (Elbow + silhouette sweep), DBSCAN (k-distance knee), GMM (BIC), HDBSCAN, Agglomerative (Ward), Spectral |
| 3b    | `src/validation.py` | Silhouette / Davies-Bouldin / Calinski-Harabasz + 50-round bootstrap ARI stability across all six algorithms; selects the best (**HDBSCAN**) |
| 4     | `src/profiling.py` | Per-segment un-scaled means, rule-based marketing names, heatmap + radar |
| 5     | `src/clv.py` | BG/NBD + Gamma-Gamma CLV with 90/180/365-day purchase forecasts |
| 6     | `src/churn.py` | Six classifiers (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, Decision Tree, KNN), multi-metric ranking + pairwise McNemar tests |
| 7     | `src/migration.py` | Year-on-year segment transition matrix + heatmap |
| 8     | `src/notifications.py` | Cluster-driven campaign engine (segment = customer's cluster, modulated by CLV/churn); on-demand `recommend(customer_id)` |
| 9     | `src/roi.py` | Monte Carlo ROI simulation (10,000 runs) with credible interval **+ uplift vs a static blanket-marketing baseline** |
| 10    | `app/streamlit_app.py` | Eight-page interactive dashboard over all artefacts |

Three further validation analyses run separately via `python scripts/run_new_validations.py`:

| Module | What it does |
|--------|--------------|
| `src/clv_validation.py` | BG/NBD calibration/holdout temporal validation on a one-year holdout |
| `src/churn_sensitivity.py` | Re-derives the churn label at the 85th/90th/95th Recency percentile and re-scores every model |
| `src/roi_sensitivity.py` | Re-runs the ROI simulation under perturbed margin, response-rate, cost, and compound-worst-case assumptions |

---

## Headline results

Reproduced by `python main.py` on the full dataset (~160 s, CPU only):

| Metric | Result |
|---|---|
| Transactions cleaned | 1,067,371 → **793,591** rows over 4 audited rules |
| Customers profiled | **5,878**, 7 behavioural features |
| Segmentation chosen | **HDBSCAN** — silhouette **0.416**, best mean rank (1.75) of six algorithms |
| Churn model | Random Forest, ROC-AUC **0.849** (McNemar: statistically tied with XGBoost) |
| CLV validation | Pearson **r = 0.83** against unseen year-2 behaviour (absolute forecasts run ~53% high) |
| Simulated ROI | **38.4×** (95% CrI [21.8×, 60.9×]) vs **30.4×** blanket baseline → **+168%** profit uplift |

---

## Key methodological choices

- **Reproducibility** — the Recency snapshot date is fixed at
  `max(InvoiceDate) + 1 day`, not the wall-clock date, so results are identical
  on every run. All randomness is seeded via `config.RANDOM_STATE`.
- **Skew handling** — `log1p` (not `log`) is applied to features with
  `|skew| > 0.5`, keeping the transform defined at zero for one-time buyers.
- **Algorithm selection** — the best clustering algorithm is chosen by a
  transparent rule: disqualify excessively noisy HDBSCAN solutions, require mean
  bootstrap ARI ≥ 0.70, then pick the highest silhouette.
- **Churn leakage avoidance** — churn is labelled from Recency, so **Recency is
  excluded from the churn feature set**; the resulting ROC-AUC ≈ 0.85 (rather
  than a suspicious ~1.0) confirms the models learn from behaviour, not the
  label definition.
- **ROI realism** — gross order value is converted to profit contribution via
  retail gross margin net of promotional discount, avoiding the naive
  "full order value = profit" inflation.
- **Cluster-driven targeting** — every customer's campaign segment is the name
  of the unsupervised cluster (Stage 3) they belong to, so the notification
  engine is genuinely cluster-based rather than a parallel rule pass.
- **Uplift vs static marketing** — the ROI stage also simulates an untargeted
  "blanket" campaign and reports the ROI/profit uplift of the targeted plan over
  it, quantifying the value of segmentation rather than an absolute figure alone.

---

## Project layout

```
.
├── main.py                  # orchestrator (runs every stage, prints summary)
├── src/                     # pipeline modules (one per stage)
│   ├── config.py            # all paths + tunable constants (no magic numbers)
│   ├── data_loading.py
│   ├── cleaning.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── validation.py
│   ├── profiling.py
│   ├── clv.py
│   ├── churn.py
│   ├── migration.py
│   ├── notifications.py
│   ├── roi.py
│   ├── clv_validation.py    # CLV temporal holdout validation
│   ├── churn_sensitivity.py # churn label-threshold sensitivity
│   └── roi_sensitivity.py   # ROI assumption sensitivity
├── app/streamlit_app.py     # interactive dashboard
├── scripts/                 # report generators (run after the pipeline)
│   ├── run_new_validations.py   # -> the three validation analyses above
│   ├── build_notebook.py        # -> executed Jupyter notebook of the pipeline
│   ├── generate_report.py       # -> single self-contained HTML report
│   ├── build_github_report.py   # -> docs/RESULTS.md (GitHub-renderable)
│   ├── make_diagrams.py         # -> architecture / component diagrams
│   └── make_presentation.py     # -> docs/presentation.pptx
├── tests/                   # 20-test pytest suite (synthetic data, no Excel needed)
├── data/
│   ├── raw/raj.xlsx         # input (not committed)
│   └── processed/           # parquet artefacts (generated)
├── outputs/
│   ├── figures/             # PNG charts (generated)
│   └── tables/              # CSV results (generated)
├── docs/                    # results report + dissertation documents
│   ├── RESULTS.md           # GitHub-renderable results write-up
│   ├── figures/             # figures embedded in RESULTS.md
│   └── dissertation/        # specs + marking guidance (Word/PDF)
└── requirements.txt
```

---

## Running the tests

```bash
pytest tests/ -v
```

20 tests across 7 modules. The suite uses small synthetic datasets (no raw Excel
required) and covers cleaning rules, RFM derivations, skew detection, the churn
leakage guard, the segment-migration logic, the notification decision rules, and
the ROI simulator. It runs in a few seconds.

---

## Configuration

All tunable constants (file paths, model hyper-parameters, random seed) are
centralised in [src/config.py](src/config.py). Edit that file before running the
pipeline — values are never hard-coded inside individual modules.

---

## Data & citation

The raw dataset is the **UCI Online Retail II** dataset, split across two Excel
sheets covering 2009-2010 and 2010-2011. It is never altered in place; all
transformations write to `data/processed/`.

> Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository.
> <https://doi.org/10.24432/C5CG6D>
