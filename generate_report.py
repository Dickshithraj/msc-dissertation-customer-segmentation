"""
generate_report.py — single-file HTML report for the dissertation pipeline.

Embeds every source file, output table, and figure into one self-contained
HTML document (no external dependencies, no internet required) and saves it
to the user's Downloads folder.

Run from the project root::

    python generate_report.py
"""

from __future__ import annotations

import base64
import html
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
OUT_HTML = DOWNLOADS / "dissertation_report.html"

# ── Files to include ─────────────────────────────────────────────────────────

SOURCE_FILES = [
    ("main.py",                    "Pipeline Orchestrator"),
    ("src/config.py",              "Configuration"),
    ("src/data_loading.py",        "Stage 1 – Data Loading"),
    ("src/cleaning.py",            "Stage 1 – Data Cleaning"),
    ("src/features.py",            "Stage 2 – Feature Engineering"),
    ("src/preprocessing.py",       "Stage 2b – Preprocessing"),
    ("src/clustering.py",          "Stage 3-4 – Clustering (4 algorithms)"),
    ("src/validation.py",          "Stage 3b – Validation & Stability"),
    ("src/profiling.py",           "Stage 6 – Segment Profiling"),
    ("src/clv.py",                 "Stage 7 – Customer Lifetime Value"),
    ("src/churn.py",               "Stage 8 – Churn Classification"),
    ("src/migration.py",           "Stage 9 – Segment Migration"),
    ("src/notifications.py",       "Stage 10 – Notification Engine"),
    ("src/roi.py",                 "Stage 11 – Monte Carlo ROI"),
    ("app/streamlit_app.py",       "Stage 12 – Streamlit Dashboard"),
    ("requirements.txt",           "Requirements"),
]

TABLE_FILES = [
    ("outputs/tables/cleaning_summary.csv",          "Cleaning Summary"),
    ("outputs/tables/feature_summary.csv",           "Feature Summary (describe)"),
    ("outputs/tables/kmeans_metrics.csv",            "K-Means Sweep Metrics"),
    ("outputs/tables/cluster_validation.csv",        "Cluster Validation Metrics"),
    ("outputs/tables/segment_profiles.csv",          "Segment Profiles"),
    ("outputs/tables/clv_summary.csv",               "CLV Summary"),
    ("outputs/tables/churn_model_comparison.csv",    "Churn Model Comparison"),
    ("outputs/tables/segment_migration_counts.csv",  "Migration Counts"),
    ("outputs/tables/segment_migration_rates.csv",   "Migration Rates"),
    ("outputs/tables/roi_simulation_summary.csv",    "ROI Simulation Summary"),
    ("outputs/tables/notification_plan.csv",         "Notification Plan (first 100 rows)"),
]

FIGURE_FILES = [
    ("outputs/figures/rfm_distributions.png",       "RFM Distributions"),
    ("outputs/figures/scaling_effect.png",          "Scaling Effect (before / after)"),
    ("outputs/figures/kmeans_selection.png",        "K-Means Model Selection"),
    ("outputs/figures/dbscan_kdistance.png",        "DBSCAN k-distance Knee"),
    ("outputs/figures/gmm_bic.png",                 "GMM BIC Selection"),
    ("outputs/figures/cluster_pca_projection.png",  "Cluster PCA Projection (4 algorithms)"),
    ("outputs/figures/stability_ari.png",           "Bootstrap Stability (ARI)"),
    ("outputs/figures/segment_profiles.png",        "Segment Profile Heatmap"),
    ("outputs/figures/radar_profiles.png",          "Segment Radar Chart"),
    ("outputs/figures/clv_distribution.png",        "CLV Distribution"),
    ("outputs/figures/churn_roc_curves.png",        "Churn ROC Curves"),
    ("outputs/figures/churn_feature_importance.png", "Churn Feature Importance"),
    ("outputs/figures/segment_migration.png",       "Segment Migration Heatmap"),
    ("outputs/figures/roi_distribution.png",        "Monte Carlo ROI Distribution"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_text(rel: str) -> str:
    path = PROJECT_ROOT / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else f"[File not found: {rel}]"


def _img_b64(rel: str) -> str | None:
    path = PROJECT_ROOT / rel
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _csv_to_html_table(rel: str, max_rows: int = 100) -> str:
    path = PROJECT_ROOT / rel
    if not path.exists():
        return f"<p class='missing'>[File not found: {rel}]</p>"
    df = pd.read_csv(path)
    note = ""
    if len(df) > max_rows:
        note = (f"<p class='missing'>Showing first {max_rows:,} of "
                f"{len(df):,} rows.</p>")
        df = df.head(max_rows)
    table = df.to_html(classes="data-table", index=True, border=0,
                       float_format="{:.4f}".format)
    return note + table


def _lang(filename: str) -> str:
    ext = Path(filename).suffix
    return {"py": "python", "md": "markdown", "txt": "text"}.get(ext.lstrip("."), "text")


# ── Pipeline summary via subprocess ──────────────────────────────────────────

def _capture_pipeline_summary() -> str:
    """Return the last pipeline summary block from a quick dry run of main.py.

    Instead of re-running the full pipeline (slow), we read the already-saved
    CSVs and reconstruct the key numbers without touching the raw Excel.
    """
    lines = []
    # Cleaning summary
    cs_path = PROJECT_ROOT / "outputs/tables/cleaning_summary.csv"
    if cs_path.exists():
        cs = pd.read_csv(cs_path)
        lines.append("CLEANING")
        for _, r in cs.iterrows():
            lines.append(f"  Step {int(r['Step'])} – {r['Description']}: "
                         f"removed {int(r['Rows Removed']):,} rows ({r['% Removed']}%)")
        lines.append(f"  Final: {int(cs.iloc[-1]['Rows After']):,} rows\n")

    # Feature summary
    fs_path = PROJECT_ROOT / "outputs/tables/feature_summary.csv"
    if fs_path.exists():
        fs = pd.read_csv(fs_path, index_col=0)
        lines.append("FEATURES  (median | mean | max)")
        for feat in fs.index:
            lines.append(f"  {feat:<28} {fs.loc[feat,'50%']:>10.1f} | "
                         f"{fs.loc[feat,'mean']:>10.1f} | {fs.loc[feat,'max']:>12.1f}")
        lines.append("")

    # K-Means metrics
    km_path = PROJECT_ROOT / "outputs/tables/kmeans_metrics.csv"
    if km_path.exists():
        km = pd.read_csv(km_path)
        best_k = int(km.loc[km["silhouette"].idxmax(), "k"])
        lines.append(f"K-MEANS SWEEP  (best k = {best_k} by silhouette)")
        lines.append(f"  {'k':>3}  {'Inertia':>12}  {'Silhouette':>10}  {'DB':>8}")
        for _, r in km.iterrows():
            mark = " <-- chosen" if int(r["k"]) == best_k else ""
            lines.append(f"  {int(r['k']):>3}  {r['inertia']:>12,.1f}  "
                         f"{r['silhouette']:>10.4f}  {r['davies_bouldin']:>8.4f}{mark}")
        lines.append("")

    # Validation
    val_path = PROJECT_ROOT / "outputs/tables/cluster_validation.csv"
    if val_path.exists():
        val = pd.read_csv(val_path, index_col=0)
        lines.append("CLUSTER VALIDATION")
        lines.append(f"  {'Algorithm':<12} {'Silhouette':>10} {'DB':>10} {'CH':>12} {'Noise%':>8}")
        for algo, row in val.iterrows():
            lines.append(f"  {algo:<12} {row['silhouette']:>10.4f} "
                         f"{row['davies_bouldin']:>10.4f} "
                         f"{row['calinski_harabasz']:>12.2f} "
                         f"{row['noise_fraction']*100:>7.1f}%")
        lines.append("")

    # Churn model comparison
    churn_path = PROJECT_ROOT / "outputs/tables/churn_model_comparison.csv"
    if churn_path.exists():
        ch = pd.read_csv(churn_path, index_col=0)
        best = ch["ROC_AUC"].idxmax()
        lines.append(f"CHURN MODELS  (best = {best})")
        lines.append(f"  {'Model':<20} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>9}")
        for model, row in ch.iterrows():
            mark = " <-- best" if model == best else ""
            lines.append(f"  {model:<20} {row['ROC_AUC']:>9.4f} "
                         f"{row['PR_AUC']:>9.4f} {row['F1']:>9.4f}{mark}")
        lines.append("")

    # CLV summary
    clv_path = PROJECT_ROOT / "outputs/tables/clv_summary.csv"
    if clv_path.exists():
        clv = pd.read_csv(clv_path, index_col=0)
        if "clv" in clv.columns:
            lines.append("CUSTOMER LIFETIME VALUE  (12-month horizon)")
            lines.append(f"  Mean CLV:   {clv.loc['mean', 'clv']:>12,.0f}")
            lines.append(f"  Median CLV: {clv.loc['50%', 'clv']:>12,.0f}")
            lines.append(f"  Max CLV:    {clv.loc['max', 'clv']:>12,.0f}")
            lines.append("")

    # Migration cohort
    mig_path = PROJECT_ROOT / "outputs/tables/segment_migration_counts.csv"
    if mig_path.exists():
        mig = pd.read_csv(mig_path, index_col=0)
        retained = int(mig.values.sum())
        lines.append("YEAR-ON-YEAR MIGRATION")
        lines.append(f"  Customers retained across both years: {retained:,}")
        lines.append("")

    # ROI
    roi_path = PROJECT_ROOT / "outputs/tables/roi_simulation_summary.csv"
    if roi_path.exists():
        roi = pd.read_csv(roi_path, index_col=0)["value"]
        lines.append("MONTE CARLO ROI  (10,000 simulations)")
        lines.append(f"  Mean ROI:   {roi.get('mean_roi', float('nan')):>8.1f}x")
        if "roi_ci_low_95" in roi.index:
            lines.append(f"  95% CI:     [{roi['roi_ci_low_95']:.1f}x, "
                         f"{roi['roi_ci_high_95']:.1f}x]")
        lines.append(f"  P(ROI > 0): {roi.get('prob_positive_roi', float('nan'))*100:>7.1f}%")

    return "\n".join(lines)


# ── HTML assembly ─────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9;
       color: #222; display: flex; min-height: 100vh; }

/* Sidebar */
nav { width: 260px; min-width: 260px; background: #1a2540; color: #cdd6f4;
      position: sticky; top: 0; height: 100vh; overflow-y: auto;
      padding: 0 0 2rem 0; }
nav h2 { padding: 1.4rem 1.2rem 0.8rem; font-size: 0.95rem;
          text-transform: uppercase; letter-spacing: 1px; color: #7b8ec8; }
nav a { display: block; padding: 0.45rem 1.2rem; font-size: 0.86rem;
        color: #cdd6f4; text-decoration: none; border-left: 3px solid transparent; }
nav a:hover, nav a.active { background: #24305e; border-left-color: #6c8ebf;
                              color: #fff; }
nav .section-header { padding: 0.9rem 1.2rem 0.25rem;
                       font-size: 0.75rem; text-transform: uppercase;
                       letter-spacing: 1px; color: #6c7faa; margin-top: 0.5rem; }

/* Main */
main { flex: 1; padding: 2rem 2.4rem; max-width: 1100px; }

.report-title { font-size: 1.7rem; font-weight: 700; color: #1a2540;
                margin-bottom: 0.3rem; }
.report-meta  { font-size: 0.85rem; color: #666; margin-bottom: 2rem; }

/* Section cards */
.section { background: #fff; border-radius: 8px; padding: 1.5rem 1.8rem;
           margin-bottom: 2rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.section h2 { font-size: 1.15rem; color: #1a2540; margin-bottom: 1rem;
              padding-bottom: 0.5rem; border-bottom: 2px solid #e8ecf3; }
.section h3 { font-size: 0.97rem; color: #2d4070; margin: 1.2rem 0 0.5rem; }

/* Code blocks */
pre { background: #1e1e2e; color: #cdd6f4; border-radius: 6px;
      padding: 1.1rem 1.2rem; overflow-x: auto; font-size: 0.8rem;
      line-height: 1.55; tab-size: 4; white-space: pre; }
code { font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace; }

/* Tables */
.data-table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
.data-table th { background: #1a2540; color: #fff; padding: 0.5rem 0.75rem;
                  text-align: left; }
.data-table td { padding: 0.4rem 0.75rem; border-bottom: 1px solid #e8ecf3; }
.data-table tr:nth-child(even) td { background: #f7f9fc; }
.table-wrap { overflow-x: auto; margin-top: 0.5rem; }

/* Figures */
.figure-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.4rem;
               margin-top: 0.5rem; }
.figure-card { background: #f7f9fc; border-radius: 6px; padding: 0.9rem;
               border: 1px solid #e0e5ee; }
.figure-card p { font-size: 0.82rem; color: #555; margin-top: 0.5rem;
                  text-align: center; }
.figure-card img { width: 100%; border-radius: 4px; }
.missing { color: #c0392b; font-style: italic; }

/* Summary block */
.summary-block { background: #1e1e2e; color: #a6e3a1; border-radius: 6px;
                 padding: 1.1rem 1.2rem; overflow-x: auto;
                 font-family: 'Cascadia Code','Consolas',monospace;
                 font-size: 0.8rem; line-height: 1.6; white-space: pre; }

@media (max-width: 900px) {
    .figure-grid { grid-template-columns: 1fr; }
    nav { display: none; }
}
"""

JS = """
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('.section[id]');
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    const observer = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                navLinks.forEach(a => a.classList.remove('active'));
                const active = document.querySelector('nav a[href="#' + e.target.id + '"]');
                if (active) active.classList.add('active');
            }
        });
    }, { threshold: 0.25 });
    sections.forEach(s => observer.observe(s));
});
"""


def _nav_link(anchor: str, label: str) -> str:
    return f'<a href="#{anchor}">{html.escape(label)}</a>'


def _section_open(anchor: str, title: str) -> str:
    return (f'<div class="section" id="{anchor}">\n'
            f'<h2>{html.escape(title)}</h2>\n')


def _section_close() -> str:
    return '</div>\n'


def build_html() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_text = _capture_pipeline_summary()

    # ── Navigation ────────────────────────────────────────────────────────
    nav_parts = [
        '<nav>',
        '<h2>Dissertation Report</h2>',
        '<div class="section-header">Overview</div>',
        _nav_link("overview", "Pipeline Summary"),
        '<div class="section-header">Source Files</div>',
    ]
    for rel, label in SOURCE_FILES:
        anchor = Path(rel).stem.replace("_", "-")
        nav_parts.append(_nav_link(anchor, label))

    nav_parts.append('<div class="section-header">Output Tables</div>')
    for rel, label in TABLE_FILES:
        anchor = "tbl-" + Path(rel).stem.replace("_", "-")
        nav_parts.append(_nav_link(anchor, label))

    nav_parts.append('<div class="section-header">Figures</div>')
    for rel, label in FIGURE_FILES:
        anchor = "fig-" + Path(rel).stem.replace("_", "-")
        nav_parts.append(_nav_link(anchor, label))

    nav_parts.append('</nav>')
    nav_html = "\n".join(nav_parts)

    # ── Main content ──────────────────────────────────────────────────────
    body_parts = [
        '<main>',
        f'<p class="report-title">Customer Segmentation — Dissertation Pipeline Report</p>',
        f'<p class="report-meta">Generated: {now} &nbsp;|&nbsp; '
        f'Dataset: UCI Online Retail II &nbsp;|&nbsp; Python 3.11</p>',
    ]

    # Overview / pipeline summary
    body_parts.append(_section_open("overview", "Pipeline Summary"))
    body_parts.append(f'<div class="summary-block">{html.escape(summary_text)}</div>')
    body_parts.append(_section_close())

    # Source files
    for rel, label in SOURCE_FILES:
        anchor = Path(rel).stem.replace("_", "-")
        body_parts.append(_section_open(anchor, label))
        body_parts.append(f'<h3>{html.escape(rel)}</h3>')
        lang = _lang(rel)
        code = html.escape(_read_text(rel))
        body_parts.append(f'<pre><code class="language-{lang}">{code}</code></pre>')
        body_parts.append(_section_close())

    # Output tables
    for rel, label in TABLE_FILES:
        anchor = "tbl-" + Path(rel).stem.replace("_", "-")
        body_parts.append(_section_open(anchor, label))
        body_parts.append(f'<h3>{html.escape(rel)}</h3>')
        body_parts.append('<div class="table-wrap">')
        body_parts.append(_csv_to_html_table(rel))
        body_parts.append('</div>')
        body_parts.append(_section_close())

    # Figures — two per row
    body_parts.append(_section_open("figures", "Output Figures"))
    body_parts.append('<div class="figure-grid">')
    for rel, label in FIGURE_FILES:
        anchor = "fig-" + Path(rel).stem.replace("_", "-")
        src = _img_b64(rel)
        body_parts.append(f'<div class="figure-card" id="{anchor}">')
        if src:
            body_parts.append(f'<img src="{src}" alt="{html.escape(label)}" loading="lazy">')
        else:
            body_parts.append(f'<p class="missing">[Image not found: {rel}]</p>')
        body_parts.append(f'<p>{html.escape(label)}</p>')
        body_parts.append('</div>')
    body_parts.append('</div>')  # figure-grid
    body_parts.append(_section_close())

    body_parts.append('</main>')
    body_html = "\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dissertation Pipeline Report</title>
<style>{CSS}</style>
</head>
<body>
{nav_html}
{body_html}
<script>{JS}</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building HTML report …")
    html_content = build_html()
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html_content, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"Saved to: {OUT_HTML}  ({size_kb:,.0f} KB)")
