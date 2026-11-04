from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from socsim.viz.common import save_figure, setup_style

PALETTE = {
    "human": "#2c7fb8",
    "synthetic": "#f03b20",
    "agent": "#31a354",
}

LABELS = {
    "human": "Human",
    "synthetic": "Synthetic",
    "agent": "Agent",
}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _dataset_color(dataset: str) -> str:
    if dataset == "human":
        return PALETTE["human"]
    if dataset == "synthetic":
        return PALETTE["synthetic"]
    return PALETTE["agent"]


def _dataset_label(dataset: str) -> str:
    if dataset == "human":
        return LABELS["human"]
    if dataset == "synthetic":
        return LABELS["synthetic"]
    return LABELS["agent"]


def _estimate_se(row: pd.Series) -> float:
    ci_low = row.get("ci_low", np.nan)
    ci_high = row.get("ci_high", np.nan)
    if pd.isna(ci_low) or pd.isna(ci_high):
        return np.nan
    return float((ci_high - ci_low) / (2.0 * 1.96))


def _plot_single_relationship(
    rel_df: pd.DataFrame,
    datasets: list[str],
    out_path: Path,
) -> None:
    setup_style()
    sns.set_style("whitegrid")
    predictor = str(rel_df["predictor"].iloc[0]) if not rel_df.empty else "predictor"
    outcome = str(rel_df["outcome"].iloc[0]) if not rel_df.empty else "outcome"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )
    ax_top, ax_bottom = axes

    if rel_df.empty:
        ax_top.set_title(f"{predictor} vs {outcome} (no data)")
        ax_top.axis("off")
        ax_bottom.axis("off")
        save_figure(fig, out_path)
        return

    plotted_any = False
    for dataset in datasets:
        ds_df = rel_df[rel_df["dataset"] == dataset].sort_values("bin")
        if ds_df.empty:
            continue
        color = _dataset_color(dataset)
        label = _dataset_label(dataset)
        ax_top.plot(ds_df["bin_center"], ds_df["mean_outcome"], marker="o", linewidth=2, color=color, label=label)
        if ds_df["ci_low"].notna().any() and ds_df["ci_high"].notna().any():
            ax_top.fill_between(ds_df["bin_center"], ds_df["ci_low"], ds_df["ci_high"], color=color, alpha=0.18)
        plotted_any = True

    if not plotted_any:
        ax_top.set_title(f"{predictor} vs {outcome} (no data)")
        ax_top.axis("off")
        ax_bottom.axis("off")
        save_figure(fig, out_path)
        return

    human = rel_df[rel_df["dataset"] == "human"].sort_values("bin").copy()
    if not human.empty:
        human["se"] = human.apply(_estimate_se, axis=1)
    for dataset in [ds for ds in datasets if ds != "human"]:
        sim = rel_df[rel_df["dataset"] == dataset].sort_values("bin").copy()
        if sim.empty or human.empty:
            continue
        sim["se"] = sim.apply(_estimate_se, axis=1)
        merged = human.merge(sim, on="bin", how="inner", suffixes=("_human", "_sim"))
        if merged.empty:
            continue
        merged["diff"] = merged["mean_outcome_sim"] - merged["mean_outcome_human"]
        merged["diff_se"] = np.sqrt(np.square(merged["se_human"]) + np.square(merged["se_sim"]))
        merged["diff_ci_low"] = merged["diff"] - 1.96 * merged["diff_se"]
        merged["diff_ci_high"] = merged["diff"] + 1.96 * merged["diff_se"]
        color = _dataset_color(dataset)
        label = f"{_dataset_label(dataset)} - Human"
        ax_bottom.plot(merged["bin_center_human"], merged["diff"], marker="o", linewidth=2, color=color, label=label)
        if merged["diff_ci_low"].notna().any() and merged["diff_ci_high"].notna().any():
            ax_bottom.fill_between(
                merged["bin_center_human"],
                merged["diff_ci_low"],
                merged["diff_ci_high"],
                color=color,
                alpha=0.18,
            )

    ax_top.set_title(f"{outcome} by {predictor}")
    ax_top.set_ylabel("Mean outcome")
    ax_top.legend(loc="best", fontsize=9)
    ax_top.tick_params(labelsize=9)

    ax_bottom.axhline(0.0, linestyle="--", linewidth=1, color="#666666")
    ax_bottom.set_xlabel(predictor)
    ax_bottom.set_ylabel("Simulated - human")
    ax_bottom.legend(loc="best", fontsize=9)
    ax_bottom.tick_params(labelsize=9)
    sns.despine(fig=fig)
    save_figure(fig, out_path)


def plot_core_relationship_fidelity(
    bin_table: pd.DataFrame,
    comparison: str,
    out_root: Path,
) -> None:
    datasets = ["human", "synthetic", comparison]
    if bin_table.empty:
        return
    comparison_dir = out_root / comparison
    for (predictor, outcome), rel_df in bin_table.groupby(["predictor", "outcome"], observed=True):
        subset = rel_df[rel_df["dataset"].isin(datasets)].copy()
        if subset.empty or "human" not in set(subset["dataset"]) or comparison not in set(subset["dataset"]):
            continue
        out_path = comparison_dir / f"{_slug(predictor)}_{_slug(outcome)}_relationship_fidelity.png"
        _plot_single_relationship(subset, datasets, out_path)


def plot_brt_liwc_relationship_fidelity(
    bin_table: pd.DataFrame,
    comparison: str,
    out_root: Path,
) -> None:
    datasets = ["human", comparison]
    if bin_table.empty:
        return
    comparison_dir = out_root / comparison
    for (predictor, outcome), rel_df in bin_table.groupby(["predictor", "outcome"], observed=True):
        subset = rel_df[rel_df["dataset"].isin(datasets)].copy()
        if subset.empty or "human" not in set(subset["dataset"]) or comparison not in set(subset["dataset"]):
            continue
        out_path = comparison_dir / f"{_slug(predictor)}_{_slug(outcome)}_relationship_fidelity.png"
        _plot_single_relationship(subset, datasets, out_path)
