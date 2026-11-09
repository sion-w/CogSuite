from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from socsim.analysis.descriptives import resolve_var_column
from socsim.viz.common import save_figure, setup_style


def plot_primary_model_coefficients(table: pd.DataFrame, dataset: str, out_path: Path) -> None:
    setup_style()
    if table.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"{dataset}: primary model coefficients (no data)")
        ax.axis("off")
        save_figure(fig, out_path)
        return
    primary = table[table["model_type"] == "binomial_glm"].copy()
    if primary.empty:
        primary = table.copy()
    primary["label"] = primary["outcome"] + " :: " + primary["predictor"]
    primary = primary.sort_values(["outcome", "predictor"])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(primary))))
    y = np.arange(len(primary))
    xerr = np.vstack([primary["coef"] - primary["CI95_low"], primary["CI95_high"] - primary["coef"]])
    ax.errorbar(primary["coef"], y, xerr=xerr, fmt="o", color="#2a6f97", ecolor="#2a6f97", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1, color="#666666")
    ax.set_yticks(y)
    ax.set_yticklabels(primary["label"])
    ax.set_xlabel("Coefficient (log-odds scale, 95% CI)")
    ax.set_title(f"{dataset}: Binomial GLM primary coefficients")
    warn_count = int((primary["warning"].fillna("") != "").sum()) if "warning" in primary.columns else 0
    if warn_count > 0:
        ax.text(0.99, 0.01, f"Warnings in {warn_count} rows", ha="right", va="bottom", transform=ax.transAxes, fontsize=8)
    save_figure(fig, out_path)


def plot_focused_relationships(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    setup_style()
    xcol = resolve_var_column(df, "NFC_Total", dataset)
    yvars = ["CRT2_Total", "BCRT_Total"]
    if dataset == "human":
        yvars.append("CRT2_Total_raw")
    ycols = [resolve_var_column(df, v, dataset) for v in yvars]
    ylabels = yvars
    fig, axes = plt.subplots(1, len(yvars), figsize=(5 * len(yvars), 4), sharex=False)
    if len(yvars) == 1:
        axes = [axes]
    for ax, ycol, yl in zip(axes, ycols, ylabels):
        if not xcol or not ycol:
            ax.set_title(f"{dataset}: {yl}\n(no data)")
            ax.axis("off")
            continue
        work = pd.DataFrame({"x": pd.to_numeric(df[xcol], errors="coerce"), "y": pd.to_numeric(df[ycol], errors="coerce")}).dropna()
        n = len(work)
        if n < 3:
            ax.set_title(f"{dataset}: {yl}\nWARNING: N small or variance=0; interpret cautiously.")
            ax.axis("off")
            continue
        work["bin"] = pd.qcut(work["x"], q=min(10, max(3, n // 10)), duplicates="drop")
        g = work.groupby("bin", observed=True)
        xb = g["x"].mean()
        yb = g["y"].mean()
        se = g["y"].std(ddof=1) / np.sqrt(g["y"].count().clip(lower=1))
        ax.errorbar(xb, yb, yerr=1.96 * se, fmt="o-", label=f"Binned mean ±95% CI (N={n})")
        slope, intercept, r, p, _ = stats.linregress(work["x"], work["y"])
        xx = np.linspace(work["x"].min(), work["x"].max(), 100)
        ax.plot(xx, intercept + slope * xx, "-", label=f"OLS line (r={r:.2f}, p={p:.3g})")
        ax.set_title(f"{dataset}: {yl} vs NFC_Total")
        ax.set_xlabel("NFC_Total")
        ax.set_ylabel(yl)
        ax.legend(loc="best")
    fig.suptitle("Focused Relationships: Binned Means + OLS")
    save_figure(fig, out_path)


def plot_liwc_family_models(table: pd.DataFrame, dataset: str, out_path: Path, title: str) -> None:
    setup_style()
    if table.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title(f"{title}: {dataset}\n(no data)")
        ax.axis("off")
        save_figure(fig, out_path)
        return
    work = table.copy()
    coef_col = "coef" if "coef" in work.columns else "beta"
    p_col = "p" if "p" in work.columns else "p_raw" if "p_raw" in work.columns else None
    label_col = "predictor"
    if "liwc_var" in work.columns:
        label_col = "liwc_var"
        if "predictor" in work.columns:
            work["liwc_label"] = work["liwc_var"].astype(str) + " ~ " + work["predictor"].astype(str)
            label_col = "liwc_label"
    work = work.sort_values(coef_col)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(work))))
    y = np.arange(len(work))
    xerr = np.vstack([work[coef_col] - work["CI95_low"], work["CI95_high"] - work[coef_col]])
    ax.errorbar(work[coef_col], y, xerr=xerr, fmt="o", color="#0072B2", ecolor="#0072B2", capsize=3)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(work[label_col])
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(f"{title}: {dataset}")
    for i, (_, row) in enumerate(work.iterrows()):
        p = row.get(p_col, np.nan) if p_col is not None else np.nan
        if pd.isna(p):
            continue
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if stars:
            ax.text(float(row[coef_col]), i + 0.18, stars, fontsize=8, ha="center", va="bottom")
    save_figure(fig, out_path)
