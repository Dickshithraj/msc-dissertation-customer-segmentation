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

# 5. Explore the results interactively
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
| 3–4   | `src/clustering.py` | K-Means (silhouette sweep), DBSCAN (k-distance knee), GMM (BIC), HDBSCAN |
| 3b    | `src/validation.py` | Silhouette / Davies-Bouldin / Calinski-Harabasz + bootstrap ARI stability; selects the best algorithm |
| 6     | `src/profiling.py` | Per-segment un-scaled means, rule-based marketing names, heatmap + radar |
| 7     | `src/clv.py` | BG/NBD + Gamma-Gamma CLV with 90/180/365-day purchase forecasts |
| 8     | `src/churn.py` | Logistic Regression / Random Forest / XGBoost churn models, ranked by ROC-AUC |
| 9     | `src/migration.py` | Year-on-year segment transition matrix + heatmap |
| 10    | `src/notifications.py` | Rule-based campaign engine; `recommend(customer_id)` |
| 11    | `src/roi.py` | Monte Carlo ROI simulation (10,000 runs) with credible interval |
| 12    | `app/streamlit_app.py` | Interactive dashboard over all artefacts |

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
│   └── roi.py
├── app/streamlit_app.py     # interactive dashboard
├── tests/                   # pytest suite (synthetic data, no Excel needed)
├── data/
│   ├── raw/raj.xlsx         # input (not committed)
│   └── processed/           # parquet artefacts (generated)
├── outputs/
│   ├── figures/             # PNG charts (generated)
│   └── tables/              # CSV results (generated)
└── requirements.txt
```

---

## Running the tests

```bash
pytest tests/ -v
```

The suite uses small synthetic datasets (no raw Excel required) and covers
cleaning rules, RFM derivations, skew detection, the churn leakage guard, the
notification decision logic, and the ROI simulator. It runs in a few seconds.

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
