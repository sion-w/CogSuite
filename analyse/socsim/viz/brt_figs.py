from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from socsim.viz.common import save_figure, setup_style


def plot_brt_decision_distribution(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    setup_style()
    required = ["BRT1_decision", "BRT2_decision"]
    if any(c not in df.columns for c in required):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(f"BRT decision distribution: {dataset}\n(no data)")
        ax.axis("off")
        save_figure(fig, out_path)
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=True)
    specs = [
        ("BRT1_decision", "BRT1 decision"),
        ("BRT2_decision", "BRT2 decision"),
    ]
    for ax, (col, title) in zip(axes, specs):
        p = df[col].value_counts(normalize=True, dropna=True)
        cats = p.index.tolist()
        vals = p.values
        ax.bar(np.arange(len(cats)), vals, color="#56B4E9")
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("Proportion")
    fig.suptitle(f"BRT decision distributions: {dataset}")
    save_figure(fig, out_path)


def plot_brt_primary_forest(model_df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    setup_style()
    if model_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(f"BRT primary models: {dataset}\n(no data)")
        ax.axis("off")
        save_figure(fig, out_path)
        return
    work = model_df[
        (model_df["model_type"] == "primary_bcrt_only") & (model_df["predictor"] == "z_BCRT_Total")
    ].copy()
    if work.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(f"BRT primary models: {dataset}\n(no primary bCRT rows)")
        ax.axis("off")
        save_figure(fig, out_path)
        return
    work = work.sort_values("odds_ratio")
    y = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.55 * len(work))))
    xerr = np.vstack([work["odds_ratio"] - work["OR_CI95_low"], work["OR_CI95_high"] - work["odds_ratio"]])
    face = ["#0072B2" if p < 0.05 else "white" for p in work["p"]]
    for idx, (_, row) in enumerate(work.iterrows()):
        ax.errorbar(
            float(row["odds_ratio"]),
            y[idx],
            xerr=np.array([[float(row["odds_ratio"] - row["OR_CI95_low"])], [float(row["OR_CI95_high"] - row["odds_ratio"])]]),
            fmt="o",
            color="#0072B2",
            mfc=face[idx],
            mec="#0072B2",
            capsize=3,
        )
    ax.axvline(1, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(work["outcome"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_title(f"Primary BRT outcomes (bCRT): {dataset}")
    save_figure(fig, out_path)
