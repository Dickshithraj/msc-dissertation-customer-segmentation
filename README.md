# Optimizing Targeted Marketing

### A Framework for Customer Segmentation and Automated Cluster-Based Notification

MSc Advanced Computer Science dissertation — University of Leeds, 2025/26
Dickshith Raj Nagaraj (202006797)

This repository contains the complete software artefact for the dissertation: a
reproducible Python pipeline that takes raw retail transactions and produces a
prioritised marketing action for every customer, together with a quantified
estimate of the financial return of acting on it.

---

## 1. What this software does

Most marketing analytics work stops at a model score. This pipeline continues to
the decision:

```
raw transactions  →  cleaned data  →  behavioural features
                  →  customer segments        (6 algorithms compared)
                  →  customer value + churn risk
                  →  a notification plan      (what to send whom, and when)
                  →  a Monte Carlo ROI estimate of acting on that plan
```

Every stage writes its output to disk, so any intermediate result can be
inspected without re-running the whole thing.

---

## 2. Running it

**Requirements:** Python 3.11. No GPU. The full run takes about 160 seconds on a
normal laptop.

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run the whole pipeline  (~160 s — regenerates every table and figure)
python main.py

# 3. run the three validation analyses reported in Chapter 5
python scripts/run_new_validations.py

# 4. explore the results interactively
streamlit run app/streamlit_app.py
```

**The dataset is already in the repository** (`data/raw/raj.xlsx`, 45 MB), so
nothing needs to be downloaded first.

To check the code is behaving correctly:

```bash
pytest          # 20 unit tests, synthetic data, no Excel file needed
```

### Nothing needs to be run to inspect the results

Every table and figure the dissertation reports is already committed under
`outputs/`. Running `main.py` regenerates them identically — the pipeline is
seeded, so a fresh run reproduces the committed files byte for byte.

---

## 3. What the pipeline produces

| Metric | Result |
|---|---|
| Transactions cleaned | 1,067,371 → **793,591** rows, under 4 audited rules |
| Customers profiled | **5,878**, with 7 behavioural features |
| Segmentation selected | **HDBSCAN** — silhouette **0.416**, best mean rank (1.75) of six algorithms |
| Churn model | Random Forest, ROC-AUC **0.849** (McNemar: statistically tied with XGBoost) |
| CLV validation | Pearson **r = 0.83** against unseen year-2 behaviour |
| Simulated ROI | **38.4×** vs **30.4×** blanket baseline → **+168%** profit uplift |

---

## 4. Where to find things

| Looking for | Location |
|---|---|
| The command to run | `main.py` |
| The pipeline logic | `src/` — one module per stage |
| Every setting, threshold and random seed | `src/config.py` |
| The dashboard | `app/streamlit_app.py` |
| Result tables (18 CSV) | `outputs/tables/` |
| Figures (22 PNG) | `outputs/figures/` |
| Unit tests | `tests/` |
| Input data | `data/raw/raj.xlsx` |

**`src/config.py` is the single place where behaviour is controlled.** There are
no magic numbers elsewhere in the codebase — every threshold, percentile and seed
used in the dissertation is declared there and imported.

---

## 5. Which code produces which part of the dissertation

| Report section | Code |
|---|---|
| 4.2 Data loading and cleaning | `src/data_loading.py`, `src/cleaning.py` |
| 4.3 Feature engineering | `src/features.py` |
| 4.4 Preprocessing | `src/preprocessing.py` |
| 4.5 Clustering (six algorithms) | `src/clustering.py` |
| 4.6 Validation, stability, selection | `src/validation.py` |
| 4.7 Segment profiling | `src/profiling.py` |
| 4.8 Customer lifetime value | `src/clv.py` |
| 4.9 Churn classification | `src/churn.py` |
| 4.10 Segment migration | `src/migration.py` |
| 4.11 Notification engine | `src/notifications.py` |
| 4.12 Monte Carlo ROI | `src/roi.py` |
| 4.13 Dashboard | `app/streamlit_app.py` |
| 5.3.2 Churn threshold sensitivity | `src/churn_sensitivity.py` |
| 5.4 CLV temporal holdout | `src/clv_validation.py` |
| 5.7 ROI assumption sensitivity | `src/roi_sensitivity.py` |
| Figures 3.1 and 3.2 | `scripts/make_diagrams.py` |

---

## 6. Design decisions worth knowing

- **Six clustering algorithms across five families** are compared on the same
  data, under three internal validity indices plus 50-round bootstrap stability.
  The winner is chosen by a rule declared before the results were seen.
- **Churn labelling excludes Recency from the features.** The label is defined
  from Recency, so leaving it in would let any model appear near-perfect while
  learning nothing. This guard is enforced in `src/churn.py`.
- **Model differences are significance-tested** with McNemar's exact test, so no
  "best model" claim is made that the data does not support.
- **The decision layer is rule-based, not learned.** A marketer can read, question
  and override every recommendation — auditability was a stated requirement.
- **Everything is seeded.** `RANDOM_STATE = 42` throughout; repeated runs produce
  identical outputs.

---

## 7. Reproducibility

- Fixed random seed for every stochastic component
- Snapshot date pinned to `max(InvoiceDate) + 1 day`, never to today's date, so
  Recency values do not drift between runs
- Raw data treated as read-only; every derived artefact written elsewhere
- Dependencies pinned in `requirements.txt`
- 20 unit tests run on synthetic fixtures, independent of the real dataset

---

## 8. Data

`data/raw/raj.xlsx` is the **Online Retail II** dataset from the UCI Machine
Learning Repository — a public, secondary dataset containing no personally
identifiable information beyond an anonymised numeric Customer ID.

> D. Chen, "Online Retail II," UCI Machine Learning Repository, 2019.
> https://archive.ics.uci.edu/dataset/502/online+retail+ii

---

## 9. Note on scope

This repository holds the pipeline, its tests, and the results it generates. The
dissertation itself is submitted separately as a PDF.
