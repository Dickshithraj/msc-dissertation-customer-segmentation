"""
build_github_report.py — generate a GitHub-renderable results report.

Produces ``docs/RESULTS.md``: for every pipeline stage it shows a short
description, the source code (in a collapsible block), the result tables
(as Markdown), and the figures (embedded as images).  Figures are copied into
``docs/figures/`` so they are committed alongside the report and render inline
on GitHub (the live ``outputs/`` artefacts stay git-ignored as regenerable).

Run from the project root::

    python scripts/build_github_report.py

Then commit ``docs/`` and view ``docs/RESULTS.md`` on GitHub.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (scripts/ is one level down)
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_FIGURES = DOCS_DIR / "figures"
OUT_MD = DOCS_DIR / "RESULTS.md"

MAX_TABLE_ROWS = 30  # cap long tables so the page stays readable

# ── Per-stage definition: (anchor title, description, code files, tables, figures)
STAGES: list[dict] = [
    {
        "title": "Stage 1 — Data Loading & Cleaning",
        "desc": "Load both Excel sheets (2009-2010, 2010-2011), then clean: drop "
                "missing Customer IDs, cancellations, invalid quantity/price, and "
                "exact duplicates, recording an audit trail of every removal.",
        "code": ["src/data_loading.py", "src/cleaning.py"],
        "tables": ["outputs/tables/cleaning_summary.csv"],
        "figures": [],
    },
    {
        "title": "Stage 2 — Feature Engineering",
        "desc": "Build a customer-level table of RFM features plus four extended "
                "behavioural features (Tenure, AvgOrderValue, AvgInterPurchaseDays, "
                "DistinctProducts).",
        "code": ["src/features.py"],
        "tables": ["outputs/tables/feature_summary.csv"],
        "figures": ["outputs/figures/rfm_distributions.png"],
    },
    {
        "title": "Stage 2b — Preprocessing",
        "desc": "Apply log1p to skewed features (|skew| > 0.5), then StandardScaler "
                "so every feature contributes equally to distance-based clustering.",
        "code": ["src/preprocessing.py"],
        "tables": [],
        "figures": ["outputs/figures/scaling_effect.png"],
    },
    {
        "title": "Stage 3-4 — Clustering (4 algorithms)",
        "desc": "K-Means (silhouette sweep), DBSCAN (k-distance knee for eps), "
                "Gaussian Mixture (BIC selection), and HDBSCAN.",
        "code": ["src/clustering.py"],
        "tables": ["outputs/tables/kmeans_metrics.csv"],
        "figures": [
            "outputs/figures/kmeans_selection.png",
            "outputs/figures/dbscan_kdistance.png",
            "outputs/figures/gmm_bic.png",
            "outputs/figures/cluster_pca_projection.png",
        ],
    },
    {
        "title": "Stage 3b — Validation & Stability",
        "desc": "Internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) "
                "and bootstrap ARI stability across 50 resamples; the best algorithm "
                "is selected by a transparent rule (noise filter, ARI >= 0.70, then "
                "highest silhouette).",
        "code": ["src/validation.py"],
        "tables": ["outputs/tables/cluster_validation.csv"],
        "figures": ["outputs/figures/stability_ari.png"],
    },
    {
        "title": "Stage 6 — Segment Profiling",
        "desc": "Per-segment un-scaled feature means with rule-based marketing names "
                "(Champions, Loyal, At-Risk, Lost, ...), shown as a heatmap and radar.",
        "code": ["src/profiling.py"],
        "tables": ["outputs/tables/segment_profiles.csv"],
        "figures": [
            "outputs/figures/segment_profiles.png",
            "outputs/figures/radar_profiles.png",
        ],
    },
    {
        "title": "Stage 7 — Customer Lifetime Value",
        "desc": "BG/NBD (purchase process) + Gamma-Gamma (monetary process) to "
                "forecast discounted 12-month CLV and 90/180/365-day purchase counts.",
        "code": ["src/clv.py"],
        "tables": ["outputs/tables/clv_summary.csv"],
        "figures": ["outputs/figures/clv_distribution.png"],
    },
    {
        "title": "Stage 8 — Churn Classification",
        "desc": "Logistic Regression / Random Forest / XGBoost compared by ROC-AUC. "
                "Recency is excluded from the features to avoid target leakage "
                "(churn is defined from Recency).",
        "code": ["src/churn.py"],
        "tables": ["outputs/tables/churn_model_comparison.csv"],
        "figures": [
            "outputs/figures/churn_roc_curves.png",
            "outputs/figures/churn_feature_importance.png",
        ],
    },
    {
        "title": "Stage 9 — Segment Migration",
        "desc": "Year-on-year segment transition matrix over customers present in "
                "both years, with retained / lapsed / new cohort sizes.",
        "code": ["src/migration.py"],
        "tables": [
            "outputs/tables/segment_migration_counts.csv",
            "outputs/tables/segment_migration_rates.csv",
        ],
        "figures": ["outputs/figures/segment_migration.png"],
    },
    {
        "title": "Stage 10 — Notification Engine",
        "desc": "Rule-based campaign engine combining segment, CLV tier, and churn "
                "risk into an action / channel / offer / priority per customer; "
                "exposes recommend(customer_id).",
        "code": ["src/notifications.py"],
        "tables": ["outputs/tables/notification_plan.csv"],
        "figures": [],
    },
    {
        "title": "Stage 11 — Monte Carlo ROI",
        "desc": "10,000-iteration simulation of campaign ROI with Beta response "
                "priors, Binomial conversions, and a margin/discount adjustment; "
                "reports mean ROI, a 95% credible interval, and P(ROI > 0).",
        "code": ["src/roi.py"],
        "tables": ["outputs/tables/roi_simulation_summary.csv"],
        "figures": ["outputs/figures/roi_distribution.png"],
    },
    {
        "title": "Stage 12 — Streamlit Dashboard",
        "desc": "Interactive 8-page dashboard over every artefact, including a live "
                "customer-lookup that calls recommend(customer_id).",
        "code": ["app/streamlit_app.py"],
        "tables": [],
        "figures": [],
    },
]


def _csv_to_markdown(rel: str) -> str:
    """Render a CSV file as a GitHub Markdown table (capped at MAX_TABLE_ROWS)."""
    path = PROJECT_ROOT / rel
    if not path.exists():
        return f"_Table not found: `{rel}` — run `python main.py` first._\n"

    with path.open(newline="", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
    if not reader:
        return f"_Empty table: `{rel}`._\n"

    header, *rows = reader
    total = len(rows)
    truncated = total > MAX_TABLE_ROWS
    rows = rows[:MAX_TABLE_ROWS]

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines = ["| " + " | ".join(esc(h) for h in header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    md = "\n".join(lines) + "\n"
    if truncated:
        md += f"\n_Showing first {MAX_TABLE_ROWS} of {total:,} rows._\n"
    return md


def _code_block(rel: str) -> str:
    """Render a source file inside a collapsible <details> code block."""
    path = PROJECT_ROOT / rel
    if not path.exists():
        return f"_Source not found: `{rel}`._\n"
    code = path.read_text(encoding="utf-8", errors="replace")
    lang = "python" if rel.endswith(".py") else "text"
    return (
        f"<details>\n<summary>📄 View code: <code>{rel}</code> "
        f"({code.count(chr(10)) + 1} lines)</summary>\n\n"
        f"```{lang}\n{code}\n```\n\n</details>\n"
    )


def _copy_figure(rel: str) -> str | None:
    """Copy a figure into docs/figures/ and return its docs-relative path."""
    src = PROJECT_ROOT / rel
    if not src.exists():
        return None
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    dest = DOCS_FIGURES / src.name
    shutil.copy2(src, dest)
    return f"figures/{src.name}"


def build() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    parts: list[str] = [
        "# Pipeline Results\n",
        "Customer Segmentation & Cluster-Based Marketing Notifications — "
        "UCI Online Retail II.\n",
        "Each stage below shows a short description, the source code "
        "(click to expand), the result tables, and the figures. "
        "Generated by `build_github_report.py`.\n",
        "## Contents\n",
    ]

    # Table of contents
    for s in STAGES:
        anchor = s["title"].lower().replace(" — ", "--").replace(" ", "-")
        anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
        parts.append(f"- [{s['title']}](#{anchor})")
    parts.append("\n---\n")

    # Stage sections
    for s in STAGES:
        parts.append(f"## {s['title']}\n")
        parts.append(s["desc"] + "\n")

        for code_rel in s["code"]:
            parts.append(_code_block(code_rel))

        if s["tables"]:
            parts.append("**Results:**\n")
            for tbl in s["tables"]:
                name = Path(tbl).stem.replace("_", " ").title()
                parts.append(f"*{name}*\n")
                parts.append(_csv_to_markdown(tbl))

        for fig_rel in s["figures"]:
            docs_path = _copy_figure(fig_rel)
            caption = Path(fig_rel).stem.replace("_", " ").title()
            if docs_path:
                parts.append(f"\n![{caption}]({docs_path})\n")
            else:
                parts.append(f"\n_Figure not found: `{fig_rel}`._\n")

        parts.append("\n---\n")

    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    n_figs = len(list(DOCS_FIGURES.glob("*.png"))) if DOCS_FIGURES.exists() else 0
    print(f"Wrote {OUT_MD}  ({OUT_MD.stat().st_size/1024:,.0f} KB)")
    print(f"Copied {n_figs} figures into {DOCS_FIGURES}")


if __name__ == "__main__":
    build()
