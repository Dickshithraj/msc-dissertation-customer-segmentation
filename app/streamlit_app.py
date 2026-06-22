"""
Phase 12: Interactive Streamlit dashboard.

A single-page-application view over every artefact the pipeline produces:
customer segments, CLV, churn risk, year-on-year migration, the rule-based
notification plan, and the Monte Carlo ROI simulation.  It reads the saved
parquet/CSV/PNG outputs directly (it does *not* re-run the pipeline), so it
launches instantly and is safe to demo.

Run from the project root::

    streamlit run app/streamlit_app.py

If an artefact is missing, the relevant page shows a friendly notice telling
the user to run ``python main.py`` first, rather than crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable so we can reuse src.config + recommend().
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_parquet(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def _load_csv(path_str: str, index_col: int | None = None) -> pd.DataFrame | None:
    path = Path(path_str)
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


def _img_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.info(f"Figure not found: `{path.name}`. Run `python main.py` to generate it.")


def _missing(name: str) -> None:
    st.warning(f"**{name}** not found. Run `python main.py` to generate the pipeline outputs.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("📊 Customer Segmentation & Marketing Dashboard")
    st.caption("UCI Online Retail II — MSc dissertation pipeline")

    feats = _load_parquet(str(config.CUSTOMER_FEATURES_PARQUET))
    clv = _load_parquet(str(config.CUSTOMER_CLV_PARQUET))
    churn = _load_parquet(str(config.CUSTOMER_CHURN_PARQUET))
    plan = _load_csv(str(config.NOTIFICATION_PLAN_CSV))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(feats):,}" if feats is not None else "—")
    if clv is not None:
        c2.metric("Total portfolio CLV", f"£{clv['clv'].sum():,.0f}")
    else:
        c2.metric("Total portfolio CLV", "—")
    if churn is not None:
        c3.metric("Churn rate", f"{churn['churn_label'].mean()*100:.1f}%")
    else:
        c3.metric("Churn rate", "—")
    if plan is not None:
        c4.metric("Campaigns planned", f"{len(plan):,}")
    else:
        c4.metric("Campaigns planned", "—")

    st.markdown("---")
    st.subheader("Pipeline stages")
    st.markdown(
        """
        | Stage | Output |
        |-------|--------|
        | 1–2b  | Data cleaning, RFM + extended features, scaling |
        | 3–4   | Clustering: K-Means, DBSCAN, GMM, HDBSCAN |
        | 3b    | Internal validation + bootstrap stability |
        | 6     | Segment profiling & marketing names |
        | 7     | CLV (BG/NBD + Gamma-Gamma) |
        | 8     | Churn classification (LogReg / RF / XGBoost) |
        | 9     | Year-on-year segment migration |
        | 10    | Rule-based notification engine |
        | 11    | Monte Carlo ROI simulation |
        """
    )


def page_segments() -> None:
    st.header("Customer Segments")
    profiles = _load_csv(str(config.OUTPUTS_TABLES_DIR / "segment_profiles.csv"), index_col=0)
    if profiles is None:
        _missing("segment_profiles.csv")
        return

    if "segment_name" in profiles.columns:
        sizes = profiles[["segment_name", "size", "pct_of_total"]].copy()
        st.subheader("Segment sizes")
        st.bar_chart(sizes.set_index("segment_name")["size"])

    st.subheader("Profile table (un-scaled means)")
    st.dataframe(profiles, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        _img_if_exists(config.OUTPUTS_FIGURES_DIR / "segment_profiles.png", "Segment heatmap")
    with col2:
        _img_if_exists(config.OUTPUTS_FIGURES_DIR / "radar_profiles.png", "Segment radar chart")


def page_clv() -> None:
    st.header("Customer Lifetime Value")
    clv = _load_parquet(str(config.CUSTOMER_CLV_PARQUET))
    if clv is None:
        _missing("customer_clv.parquet")
        return

    total = clv["clv"].sum()
    top_decile = clv["clv"].quantile(0.90)
    top_share = clv.loc[clv["clv"] >= top_decile, "clv"].sum() / total * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total CLV", f"£{total:,.0f}")
    c2.metric("Median CLV", f"£{clv['clv'].median():,.0f}")
    c3.metric("Top-decile share", f"{top_share:.1f}%")

    _img_if_exists(config.OUTPUTS_FIGURES_DIR / "clv_distribution.png",
                   "CLV distribution & engagement")

    st.subheader("Top 20 customers by CLV")
    cols = [c for c in ["Customer ID", "clv", "prob_alive",
                        "pred_purchases_365d"] if c in clv.columns]
    st.dataframe(clv.nlargest(20, "clv")[cols].reset_index(drop=True),
                 use_container_width=True)


def page_churn() -> None:
    st.header("Churn Risk")
    metrics = _load_csv(str(config.OUTPUTS_TABLES_DIR / "churn_model_comparison.csv"), index_col=0)
    churn = _load_parquet(str(config.CUSTOMER_CHURN_PARQUET))
    if metrics is None or churn is None:
        _missing("churn outputs")
        return

    st.subheader("Model comparison")
    st.dataframe(metrics, use_container_width=True)
    best = metrics["ROC_AUC"].idxmax()
    st.success(f"Best model: **{best}**  (ROC-AUC = {metrics.loc[best, 'ROC_AUC']:.4f})")

    col1, col2 = st.columns(2)
    with col1:
        _img_if_exists(config.OUTPUTS_FIGURES_DIR / "churn_roc_curves.png", "ROC curves")
    with col2:
        _img_if_exists(config.OUTPUTS_FIGURES_DIR / "churn_feature_importance.png",
                       "Feature importance")

    st.subheader("Churn-probability distribution")
    st.bar_chart(
        pd.cut(churn["churn_probability"], bins=20).value_counts().sort_index()
        .rename_axis("probability_bin").rename("customers")
    )


def page_migration() -> None:
    st.header("Year-on-Year Segment Migration")
    counts = _load_csv(str(config.OUTPUTS_TABLES_DIR / "segment_migration_counts.csv"), index_col=0)
    rates = _load_csv(str(config.OUTPUTS_TABLES_DIR / "segment_migration_rates.csv"), index_col=0)
    if counts is None or rates is None:
        _missing("segment_migration_*.csv")
        return

    _img_if_exists(config.OUTPUTS_FIGURES_DIR / "segment_migration.png", "Migration heatmap")

    st.subheader("Transition probabilities (row-normalised)")
    st.dataframe((rates * 100).round(1).astype(str) + "%", use_container_width=True)

    st.subheader("Raw transition counts")
    st.dataframe(counts, use_container_width=True)


def page_notifications() -> None:
    st.header("Notification Plan")
    plan = _load_csv(str(config.NOTIFICATION_PLAN_CSV))
    if plan is None:
        _missing("notification_plan.csv")
        return

    st.subheader("Campaign action distribution")
    st.bar_chart(plan["action"].value_counts())

    st.subheader("Filter the plan")
    actions = ["(all)"] + sorted(plan["action"].unique())
    chosen = st.selectbox("Campaign action", actions)
    view = plan if chosen == "(all)" else plan[plan["action"] == chosen]
    st.caption(f"{len(view):,} customers")
    st.dataframe(view.head(500), use_container_width=True)


def page_roi() -> None:
    st.header("Monte Carlo ROI Simulation")
    summary = _load_csv(str(config.OUTPUTS_TABLES_DIR / "roi_simulation_summary.csv"), index_col=0)
    if summary is None:
        _missing("roi_simulation_summary.csv")
        return

    v = summary["value"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean ROI", f"{v.get('mean_roi', float('nan')):.1f}×")
    if "roi_ci_low_95" in v.index:
        c2.metric("95% CI", f"[{v['roi_ci_low_95']:.1f}×, {v['roi_ci_high_95']:.1f}×]")
    c3.metric("P(ROI > 0)", f"{v.get('prob_positive_roi', float('nan'))*100:.0f}%")

    _img_if_exists(config.OUTPUTS_FIGURES_DIR / "roi_distribution.png",
                   "ROI & net-profit distributions")

    st.subheader("Full summary")
    st.dataframe(summary, use_container_width=True)


def page_lookup() -> None:
    st.header("🔎 Customer Lookup")
    st.caption("Enter a Customer ID to see their profile and recommended campaign.")

    clv = _load_parquet(str(config.CUSTOMER_CLV_PARQUET))
    if clv is None:
        _missing("customer_clv.parquet")
        return

    ids = clv["Customer ID"].astype(int).tolist()
    default = ids[0] if ids else 0
    cid = st.number_input("Customer ID", value=int(default), step=1)

    if st.button("Look up"):
        try:
            from src.notifications import recommend
            rec = recommend(int(cid))
        except KeyError:
            st.error(f"Customer ID {cid} not found.")
            return
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"Could not generate recommendation: {exc}")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Segment", str(rec.get("segment", "—")))
        c2.metric("CLV", f"£{rec.get('clv', 0):,.0f}  ({rec.get('clv_tier','—')})")
        c3.metric("Churn risk", str(rec.get("churn_risk", "—")))

        st.markdown("### Recommended campaign")
        st.markdown(
            f"""
            - **Action:** {rec.get('action', '—')}
            - **Channel:** {rec.get('channel', '—')}
            - **Offer:** {rec.get('offer', '—')}
            - **Priority:** {rec.get('priority', '—')} / 5
            - **Recommended contact window:** {rec.get('recommended_contact_days', '—')} days
            """
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Segments": page_segments,
    "Lifetime Value": page_clv,
    "Churn Risk": page_churn,
    "Migration": page_migration,
    "Notifications": page_notifications,
    "ROI Simulation": page_roi,
    "Customer Lookup": page_lookup,
}


def main() -> None:
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Reads saved pipeline outputs. If a page is empty, run "
        "`python main.py` from the project root first."
    )
    PAGES[choice]()


if __name__ == "__main__":
    main()
