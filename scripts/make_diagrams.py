"""
make_diagrams.py — regenerate the two design-documentation figures.

Produces ``outputs/figures/system_architecture.png`` (Figure 3.1, layered
architecture and data flow) and ``outputs/figures/component_diagram.png``
(Figure 3.2, component view), with the sequential stage numbering
1, 2, 2b, 3, 3b, 4, 5, 6, 7, 8, 9, 10.

Run from the project root::

    python scripts/make_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "figures"

GREY = "#555555"


def _box(ax, x, y, w, h, fc, ec, lw=2.0, rounding=0.8):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        )
    )


def _arrow(ax, x0, y0, x1, y1, label=None, lw=2.2, label_dx=1.5, fontsize=10.5):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=18,
            color=GREY, linewidth=lw, zorder=3,
        )
    )
    if label:
        ax.text(x1 + label_dx, (y0 + y1) / 2, label, fontsize=fontsize,
                style="italic", color="#333333", ha="left", va="center")


def system_architecture() -> None:
    layers = [
        ("LAYER 1 - DATA FOUNDATION", "#3B6FC4", "#dbe7f7", [
            ("Stage 1\nLoad & Clean", "4 audited rules\n1,067,371 -> 793,591 rows"),
            ("Stage 2\nFeature Engineering", "RFM + 4 extended\nfeatures per customer"),
            ("Stage 2b\nPreprocessing", "log1p (|skew|>0.5)\n+ StandardScaler"),
        ], "5,878 customers x 7 scaled features"),
        ("LAYER 2 - SEGMENTATION", "#7A5FBF", "#ece7f8", [
            ("Stage 3\nClustering x 6", "K-Means, DBSCAN, GMM,\nHDBSCAN, Agglo., Spectral"),
            ("Stage 3b\nValidation & Stability", "Silhouette, DB, CH +\n50-round bootstrap ARI"),
            ("Stage 4\nProfiling & Naming", "per-segment means,\nmarketing names"),
        ], "named segment per customer"),
        ("LAYER 3 - CUSTOMER INTELLIGENCE", "#C8862A", "#faf3e3", [
            ("Stage 5\nCLV (BG/NBD + G-G)", "P(alive), forecasts,\n12-month discounted CLV"),
            ("Stage 6\nChurn x 6 models", "leakage-guarded,\nMcNemar significance"),
            ("Stage 7\nSegment Migration", "year-1 -> year-2\ntransition matrix"),
        ], "segment + CLV + churn risk per customer"),
        ("LAYER 4 - DECISION ENGINE", "#2FA45C", "#e9f7ee", [
            ("Stage 8\nNotification Engine",
             "segment + CLV tier +\nchurn band -> action,\nchannel, offer, priority"),
            ("Stage 9\nMonte Carlo ROI", "10,000 runs, uplift vs\nstatic blanket baseline"),
        ], "prioritised, costed campaign plan"),
        ("LAYER 5 - DELIVERY", "#D6537A", "#fdeef2", [
            ("Stage 10\nStreamlit Dashboard", "8 pages incl. live\ncustomer lookup"),
            ("Artefact Store", "Parquet + CSV + PNG\n(re-runnable stages)"),
        ], None),
    ]

    fig, ax = plt.subplots(figsize=(11.0, 15.4), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(140, 0)
    ax.axis("off")

    # input box
    _box(ax, 6, 2, 88, 6, "#eceef2", "#444444")
    ax.text(50, 5, "INPUT: Online Retail II - ~1.07M transactions (2009-2011, UK online retailer)",
            ha="center", va="center", fontsize=11.5, fontweight="bold")
    _arrow(ax, 50, 8, 50, 13, "raw transaction rows")

    y = 13.0
    layer_h = 18.0
    for title, ec, fc, boxes, flow in layers:
        _box(ax, 4, y, 92, layer_h, fc, ec, lw=2.4)
        ax.text(7, y + 2.4, title, fontsize=13.5, fontweight="bold", color=ec,
                ha="left", va="center")
        n = len(boxes)
        inner_left, inner_right, gap = 7, 93, 2.2
        bw = (inner_right - inner_left - gap * (n - 1)) / n
        by, bh = y + 4.2, layer_h - 5.6
        for i, (head, detail) in enumerate(boxes):
            bx = inner_left + i * (bw + gap)
            _box(ax, bx, by, bw, bh, "white", ec, lw=1.6, rounding=0.5)
            ax.text(bx + bw / 2, by + bh * 0.30, head, ha="center", va="center",
                    fontsize=11.5, fontweight="bold")
            ax.text(bx + bw / 2, by + bh * 0.72, detail, ha="center", va="center",
                    fontsize=9.5, color="#333333")
        y += layer_h
        if flow:
            _arrow(ax, 50, y, 50, y + 4.5, flow)
            y += 4.5

    # output box
    _arrow(ax, 50, y, 50, y + 4.5)
    y += 4.5
    _box(ax, 8, y, 84, 8, "#117A82", "#117A82")
    ax.text(50, y + 4, "OUTPUT: a prioritised, costed marketing action per customer\n"
                       "mean ROI 38.4x; +168% profit uplift vs blanket marketing",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="white")

    fig.savefig(OUT_DIR / "system_architecture.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", OUT_DIR / "system_architecture.png")


def component_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 10.0), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(76, 0)
    ax.axis("off")

    def block(x, y, w, h, head, sub, fc, ec):
        _box(ax, x, y, w, h, fc, ec, lw=2.0, rounding=0.6)
        ax.text(x + w / 2, y + h * 0.38, head, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.72, sub, ha="center", va="center",
                fontsize=10, color="#333333")

    # top: config -> orchestrator, tests to the side
    block(35, 2, 30, 8, "src/config.py", "single source of truth:\npaths + hyper-parameters",
          "#fdf3d7", "#C8862A")
    _arrow(ax, 50, 10, 50, 14.5)
    block(35, 14.5, 30, 8.5, "main.py (orchestrator)", "runs stages in dependency order",
          "#dbe7f7", "#3B6FC4")
    block(77, 14.5, 20, 8, "tests/ (pytest)", "20 unit tests on\nsynthetic data",
          "white", "#2FA45C")

    # stage-module row
    modules = [
        ("data_loading /\ncleaning.py", "Stage 1"),
        ("features /\npreprocessing.py", "Stage 2, 2b"),
        ("clustering /\nvalidation.py", "Stage 3, 3b"),
        ("profiling.py", "Stage 4"),
        ("clv / churn /\nmigration.py", "Stage 5-7"),
        ("notifications /\nroi.py", "Stage 8-9"),
    ]
    mw, gap, left, my, mh = 15.0, 1.4, 1.6, 29, 11
    for i, (head, stage) in enumerate(modules):
        mx = left + i * (mw + gap)
        _box(ax, mx, my, mw, mh, "white", "#3B6FC4", lw=1.6, rounding=0.5)
        ax.text(mx + mw / 2, my + mh * 0.36, head, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        ax.text(mx + mw / 2, my + mh * 0.78, stage, ha="center", va="center",
                fontsize=10, color="#333333")
        _arrow(ax, 50, 23, mx + mw / 2, my, lw=1.6)
        _arrow(ax, mx + mw / 2, my + mh, mx + mw / 2 + (2 if i == 0 else -2 if i == 5 else 0),
               47, lw=1.6)

    # artefact store
    _box(ax, 14, 47, 74, 8, "#eceef2", "#555555")
    ax.text(51, 50, "Artefact store  (data/processed + outputs/)",
            ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(51, 52.8, "Parquet data  |  CSV tables  |  PNG figures   ->   "
                      "every stage re-runnable in isolation",
            ha="center", va="center", fontsize=10.5, color="#333333")

    # consumers
    _arrow(ax, 36, 55, 28, 61.5, lw=1.8)
    ax.text(33, 58.5, "read", fontsize=9.5, color="#333333")
    _arrow(ax, 66, 55, 72, 61.5, lw=1.8)
    ax.text(70, 58.5, "read", fontsize=9.5, color="#333333")

    block(10, 61.5, 34, 12, "app/streamlit_app.py",
          "8-page interactive dashboard\n(reads artefacts only)", "#fdeef2", "#D6537A")
    block(56, 61.5, 34, 12, "notifications.recommend(id)",
          "on-demand single-customer\nrecommendation API", "#e9f7ee", "#2FA45C")
    _arrow(ax, 44, 67.5, 56, 67.5, lw=1.6)
    ax.text(50, 66.6, "calls", fontsize=9.5, color="#333333", ha="center")

    fig.savefig(OUT_DIR / "component_diagram.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", OUT_DIR / "component_diagram.png")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    system_architecture()
    component_diagram()
