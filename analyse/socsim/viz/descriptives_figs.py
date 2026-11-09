from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from socsim.viz.common import save_figure, setup_style


def plot_descriptives_overview(table: pd.DataFrame, out_path: Path) -> None:
    setup_style()
    core = table[
        (table["variable_type"] == "numeric")
        & (table["variable"].isin(["CRT2_Total", "BCRT_Total", "NFC_Total"]))
    ].copy()
    if core.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Descriptives Overview (no data)")
        save_figure(fig, out_path)
        return
    pivot = core.pivot_table(index="variable", columns="dataset", values="mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Descriptives Overview by Dataset (Mean Scores)")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Mean")
    ax.legend(title="Dataset", loc="best")
    save_figure(fig, out_path)


def plot_demographic_overview(table: pd.DataFrame, out_path: Path) -> None:
    setup_style()
    numeric = table[(table["section"] == "demographic") & (table["variable_type"] == "numeric")].copy()
    categorical = table[(table["section"] == "demographic") & (table["variable_type"] == "categorical")].copy()

    if numeric.empty and categorical.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Demographic Overview (no data)")
        save_figure(fig, out_path)
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    age_ax, cat_ax = axes

    if numeric.empty:
        age_ax.axis("off")
        age_ax.set_title("Age summary unavailable")
    else:
        numeric_pivot = numeric.pivot_table(index="variable", columns="dataset", values="mean")
        numeric_pivot.plot(kind="bar", ax=age_ax)
        age_ax.set_title("Numeric Demographics")
        age_ax.set_xlabel("Variable")
        age_ax.set_ylabel("Mean")
        age_ax.legend(title="Dataset", loc="best")

    if categorical.empty:
        cat_ax.axis("off")
        cat_ax.set_title("Categorical demographic summary unavailable")
    else:
        top_levels: list[pd.DataFrame] = []
        for variable in categorical["variable"].dropna().unique():
            subset = categorical[categorical["variable"] == variable].copy()
            level_order = (
                subset.groupby("level", dropna=False)["count"]
                .sum()
                .sort_values(ascending=False)
                .head(4)
                .index
            )
            top_levels.append(subset[subset["level"].isin(level_order)])
        compact = pd.concat(top_levels, ignore_index=True) if top_levels else pd.DataFrame()
        if compact.empty:
            cat_ax.axis("off")
            cat_ax.set_title("Categorical demographic summary unavailable")
        else:
            compact["row_label"] = compact["variable"] + ": " + compact["level"].astype(str)
            heat = compact.pivot_table(index="row_label", columns="dataset", values="proportion")
            image = cat_ax.imshow(heat.fillna(0.0).values, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(float(np.nanmax(heat.values)), 0.01))
            cat_ax.set_title("Top categorical demographic proportions")
            cat_ax.set_xticks(range(len(heat.columns)))
            cat_ax.set_xticklabels(heat.columns, rotation=45, ha="right")
            cat_ax.set_yticks(range(len(heat.index)))
            cat_ax.set_yticklabels(heat.index, fontsize=8)
            for row_idx in range(heat.shape[0]):
                for col_idx in range(heat.shape[1]):
                    value = heat.iloc[row_idx, col_idx]
                    if pd.isna(value):
                        continue
                    cat_ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(image, ax=cat_ax, fraction=0.046, pad=0.04)

    fig.suptitle("Demographic Descriptives Overview", fontsize=14)
    fig.tight_layout()
    save_figure(fig, out_path)
