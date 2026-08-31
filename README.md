# Optimizing Targeted Marketing

### A Framework for Customer Segmentation and Automated Cluster-Based Notification

MSc Advanced Computer Science dissertation University of Leeds, 2025/26
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

## 2. Running it — step by step

**Requirements:** Python 3.11, about 500 MB of disk space, no GPU.
The full pipeline takes roughly **160 seconds** on an ordinary laptop.

### Step 1 — check your Python version

```bash
python --version          # must be 3.11.x
```

If it reports 3.12 or newer, some pinned packages will not install. Use a
3.11 interpreter (`py -3.11` on Windows).

### Step 2 — get the code

```bash
git clone https://github.com/Dickshithraj/msc-dissertation-customer-segmentation.git
cd msc-dissertation-customer-segmentation
```

The repository includes the dataset and all generated results, so the download
is around 250 MB and nothing further needs to be fetched.

### Step 3 — create an isolated environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Your prompt should now begin with `(.venv)`.

### Step 4 — install the dependencies

```bash
pip install -r requirements.txt
```

Every version is pinned to the one the reported results were produced with.
This takes two to three minutes on a first install.

### Step 5 — confirm the data is present

```bash
# Windows
dir data\raw\raj.xlsx
# macOS / Linux
ls -lh data/raw/raj.xlsx
```

You should see a file of about **45 MB**. It is already in the repository —
there is nothing to download.

### Step 6 — run the pipeline

```bash
python main.py
```

Run this **from the project root**, with the virtual environment active. The
console prints a banner for each of the twelve stages as it completes, then a
summary table of the headline numbers. Expect roughly 160 seconds.

When it finishes, `outputs/tables/` holds 18 CSV files and `outputs/figures/`
holds 22 PNG files — every result reported in the dissertation.

### Step 7 — run the Chapter 5 validation analyses

```bash
python scripts/run_new_validations.py
```

This produces the three analyses reported in Sections 5.3.2 (churn-threshold
sensitivity), 5.4 (CLV temporal holdout) and 5.7 (ROI assumption sensitivity).
It reads the artefacts written by Step 6, so run it afterwards.

### Step 8 — run the tests

```bash
pytest
```

20 tests across 7 modules. They use synthetic fixtures, so they pass without the
Excel file and complete in a few seconds.

### Step 9 — open the dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens an eight-page interactive dashboard in your browser, including a live
per-customer lookup. Press `Ctrl+C` in the terminal to stop it.

---

### You do not have to run anything to inspect the results

Every table and figure the dissertation reports is already committed under
`outputs/`. Steps 6 and 7 regenerate them identically — the pipeline is fully
seeded, so a fresh run reproduces the committed files exactly.

---

### If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | You are not in the project root. `cd` to the folder containing `main.py`. |
| `pip install` fails on `hdbscan` | Needs C++ build tools. On Windows install the Microsoft C++ Build Tools; on Linux `sudo apt install build-essential`. |
| A package refuses to install | Check `python --version` is 3.11. Newer versions have no wheels for some pinned releases. |
| `FileNotFoundError: data/raw/raj.xlsx` | The clone did not complete. Confirm the file is present and about 45 MB. |
| `streamlit: command not found` | The virtual environment is not active. Re-run the Step 3 activation command. |
| The run seems stuck | Spectral clustering and the 50-round bootstrap are the slow stages. Give it the full 160 seconds before assuming a problem. |

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
