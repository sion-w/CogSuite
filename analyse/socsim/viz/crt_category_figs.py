from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from socsim.scoring.crt_categories import CRT_CATEGORY_LEVELS
from socsim.viz.common import save_figure, setup_style

CATEGORY_COLORS = {
    "correct": "#2ca02c",
    "intuitive_incorrect": "#ff7f0e",
    "other_incorrect": "#d62728",
    "missing_or_unparseable": "#7f7f7f",
}


def plot_response_category_distribution(distribution: pd.DataFrame, out_path: Path) -> None:
    setup_style()
    if distribution.empty:
        _plot_empty(out_path, "CRT2/bCRT response-category distribution by source and run_id")
        return
    plot_df = _collapse_distribution(distribution, ["dataset", "run_id", "instrument", "response_category"])
    plot_df["label"] = plot_df["dataset"].astype(str) + " / " + plot_df["run_id"].astype(str) + " / " + plot_df["instrument"].astype(str)
    labels = plot_df[["label"]].drop_duplicates()["label"].tolist()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.42 * len(labels))))
    _stacked_barh(ax, plot_df, labels, "label")
    ax.set_title("CRT2/bCRT response-category distribution by source and run_id")
    save_figure(fig, out_path)


def plot_response_category_by_specified_level(distribution: pd.DataFrame, out_path: Path) -> None:
    setup_style()
    if distribution.empty:
        _plot_empty(out_path, "CRT2/bCRT response-category distribution by specified CRT level")
        return
    plot_df = _collapse_distribution(distribution, ["dataset", "run_id", "instrument", "specified_level", "response_category"])
    plot_df["label"] = (
        plot_df["dataset"].astype(str)
        + " / "
        + plot_df["run_id"].astype(str)
        + " / "
        + plot_df["instrument"].astype(str)
        + " specified="
        + plot_df["specified_level"].astype(str)
    )
    labels = plot_df[["label"]].drop_duplicates()["label"].tolist()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.36 * len(labels))))
    _stacked_barh(ax, plot_df, labels, "label")
    ax.set_title("CRT2/bCRT response-category distribution by specified CRT level")
    save_figure(fig, out_path)


def _collapse_distribution(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = df.groupby(group_cols, dropna=False)["count"].sum().reset_index()
    totals = counts.groupby(group_cols[:-1], dropna=False)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    return counts


def _stacked_barh(ax, plot_df: pd.DataFrame, labels: list[str], label_col: str) -> None:
    y_pos = np.arange(len(labels))
    left = np.zeros(len(labels), dtype=float)
    for category in CRT_CATEGORY_LEVELS:
        values = []
        for label in labels:
            subset = plot_df[(plot_df[label_col] == label) & (plot_df["response_category"] == category)]
            values.append(float(subset["proportion"].sum()) if not subset.empty else 0.0)
        ax.barh(y_pos, values, left=left, color=CATEGORY_COLORS[category], edgecolor="white", label=category)
        left += np.array(values)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of item responses")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)


def _plot_empty(out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, "No eligible rows for this output in the selected datasets.", ha="center", va="center")
    save_figure(fig, out_path)

