from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from socsim.analysis.descriptives import resolve_var_column
from socsim.viz.common import save_figure, setup_style


DATASET_STYLE = {
    "human": {"color": "#1f77b4", "ls": "-"},
    "synthetic": {"color": "#ff7f0e", "ls": "--"},
    "cogsim_agent": {"color": "#2ca02c", "ls": "-."},
    "socsim_agent": {"color": "#d62728", "ls": ":"},
}


def _levels_for_var(var: str, series_by_dataset: dict[str, pd.Series]) -> list[int]:
    if var in {"CRT2_Total", "BCRT_Total"}:
        return [0, 1, 2, 3, 4]
    if var.startswith("BFI_"):
        return [1, 2, 3, 4, 5]
    if var == "NFC_Total":
        return [1, 2, 3, 4, 5]
    return []


def _discretise_for_stacked(var: str, vals: pd.Series) -> pd.Series:
    if var in {"CRT2_Total", "BCRT_Total"}:
        return vals.round().clip(lower=0, upper=4)
    if var == "NFC_Total" or var.startswith("BFI_"):
        # NFC/BFI totals are often averaged item scores (fractional); bin to nearest Likert scale point.
        return vals.round().clip(lower=1, upper=5)
    return vals


def _mean_ci(series: pd.Series) -> tuple[float, float, float]:
    if len(series) == 0:
        return np.nan, np.nan, np.nan
    mean = float(series.mean())
    if len(series) < 2 or series.std(ddof=1) == 0:
        return mean, np.nan, np.nan
    se = float(series.std(ddof=1) / np.sqrt(len(series)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def plot_overlap_panel(by_dataset: dict[str, pd.DataFrame], variables: list[str], out_path: Path, title: str) -> None:
    setup_style()
    fig, axes = plt.subplots(len(variables), 2, figsize=(14, 3.2 * len(variables)), sharex=False)
    if len(variables) == 1:
        axes = np.array([axes])
    all_level_handles = []
    all_level_labels = []
    for row_idx, var in enumerate(variables):
        ax_left, ax_right = axes[row_idx, 0], axes[row_idx, 1]
        series_by_dataset: dict[str, pd.Series] = {}
        source_cols: dict[str, str] = {}
        for ds, frame in by_dataset.items():
            col = resolve_var_column(frame, var, ds)
            if not col:
                continue
            vals = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            series_by_dataset[ds] = vals
            source_cols[ds] = col
        if not series_by_dataset:
            ax_left.axis("off")
            ax_right.axis("off")
            continue
        dataset_order = list(series_by_dataset.keys())
        levels = _levels_for_var(var, series_by_dataset)
        cmap = plt.cm.Blues
        colors = [cmap(0.3 + 0.6 * i / max(1, (len(levels) - 1))) for i in range(len(levels))]
        y_pos = np.arange(len(dataset_order))
        left = np.zeros(len(dataset_order), dtype=float)
        small_n_warning = False
        series_disc_by_dataset = {ds: _discretise_for_stacked(var, s) for ds, s in series_by_dataset.items()}
        for i, level in enumerate(levels):
            props = []
            for ds in dataset_order:
                vals = series_disc_by_dataset[ds]
                p = float((vals == level).mean())
                props.append(p)
                if len(vals) < 5:
                    small_n_warning = True
            bars = ax_left.barh(y_pos, props, left=left, color=colors[i], edgecolor="white", label=str(level))
            left = left + np.array(props)
            if row_idx == 0:
                all_level_handles.append(bars[0])
                all_level_labels.append(str(level))
        for i, ds in enumerate(dataset_order):
            n = len(series_by_dataset[ds])
            if n < 5:
                ax_left.text(1.005, i, f"N={n}", va="center", ha="left", fontsize=8, color="#666666")
        ax_left.set_xlim(0, 1.0)
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(dataset_order)
        ax_left.invert_yaxis()
        ax_left.set_xlabel("Proportion")
        source_desc = " | ".join([f"{ds}:{source_cols[ds]}" for ds in dataset_order])
        bin_note = " (binned to nearest integer scale point)" if (var == "NFC_Total" or var.startswith("BFI_")) else ""
        ax_left.set_title(f"{var} distributions{bin_note} [{source_desc}]")
        human_mean = np.nan
        for i, ds in enumerate(dataset_order):
            vals = series_by_dataset[ds]
            mean, lo, hi = _mean_ci(vals)
            xerr = np.array([[mean - lo], [hi - mean]]) if np.isfinite(lo) and np.isfinite(hi) else None
            style = DATASET_STYLE.get(ds, {})
            ax_right.errorbar(mean, i, xerr=xerr, fmt="o", color=style.get("color", "#333333"), capsize=3)
            if ds == "human":
                human_mean = mean
            if len(vals) < 3 or (len(vals) > 1 and vals.std(ddof=1) == 0):
                small_n_warning = True
        if np.isfinite(human_mean):
            ax_right.axvline(human_mean, linestyle="--", linewidth=1, color="#999999", alpha=0.7)
        ax_right.set_yticks(y_pos)
        ax_right.set_yticklabels(dataset_order)
        ax_right.invert_yaxis()
        if var in {"CRT2_Total", "BCRT_Total"}:
            ax_right.set_xlim(0, 4)
        elif var == "NFC_Total" or var.startswith("BFI_"):
            ax_right.set_xlim(1, 5)
        ax_right.set_xlabel("Mean ± 95% CI")
        subtitle = "WARNING: N small or variance=0 in one or more datasets." if small_n_warning else ""
        ax_right.set_title(f"{var} means by dataset {subtitle}".strip())
    if all_level_handles:
        fig.legend(all_level_handles, all_level_labels, title="Score level", loc="upper center", ncol=min(8, len(all_level_labels)))
    fig.suptitle(title + "\nStacked proportions (left) and mean ±95% CI (right). Human=observed, synthetic=specified, agents=realised.")
    fig.subplots_adjust(top=0.9)
    save_figure(fig, out_path)
