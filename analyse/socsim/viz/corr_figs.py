from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from socsim.viz.common import save_figure, setup_style


def _heatmap(
    ax: plt.Axes,
    corr: pd.DataFrame,
    title: str,
    p_matrix: pd.DataFrame | None = None,
    n_matrix: pd.DataFrame | None = None,
) -> None:
    if corr.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return
    mask_upper = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
    corr_masked = np.ma.masked_where(mask_upper, corr.values)
    cmap = plt.colormaps["coolwarm"].copy()
    cmap.set_bad(color="white")
    im = ax.imshow(corr_masked, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    if p_matrix is not None and not p_matrix.empty:
        for i, row_name in enumerate(corr.index):
            for j, col_name in enumerate(corr.columns):
                if i >= len(corr.index) or j >= len(corr.columns):
                    continue
                if j >= i:
                    continue
                p = p_matrix.loc[row_name, col_name] if row_name in p_matrix.index and col_name in p_matrix.columns else np.nan
                if pd.isna(p) or row_name == col_name:
                    continue
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if stars:
                    r = corr.loc[row_name, col_name]
                    txt = f"r={r:.2f}{stars}"
                    ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=7, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_corr_matrix(
    corr: pd.DataFrame,
    dataset: str,
    out_path: Path,
    p_matrix: pd.DataFrame | None = None,
    n_matrix: pd.DataFrame | None = None,
    correction_label: str = "uncorrected",
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    _heatmap(ax, corr, f"Correlation Heatmap ({correction_label}): {dataset}", p_matrix=p_matrix, n_matrix=n_matrix)
    save_figure(fig, out_path)


def plot_corr_diff(
    diff: pd.DataFrame,
    dataset: str,
    out_path: Path,
    p_matrix: pd.DataFrame | None = None,
    n_matrix: pd.DataFrame | None = None,
    correction_label: str = "uncorrected",
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    _heatmap(ax, diff, f"Correlation Difference vs Human ({correction_label}): {dataset}", p_matrix=p_matrix, n_matrix=n_matrix)
    save_figure(fig, out_path)
