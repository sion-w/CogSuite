from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from socsim.analysis.calibration import PAIR_MAP
from socsim.viz.common import save_figure, setup_style


def _warn_text(x: pd.Series, y: pd.Series) -> str:
    if len(x.dropna()) < 3 or len(y.dropna()) < 3 or x.dropna().std(ddof=1) == 0 or y.dropna().std(ddof=1) == 0:
        return "WARNING: N small or variance=0; interpret cautiously."
    return ""


def plot_calibration_scatter(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    setup_style()
    for spec in PAIR_MAP:
        construct, xcol, ycol = spec["construct"], spec["xcol"], spec["ycol"]
        fig, ax = plt.subplots(figsize=(5, 4))
        if xcol not in df.columns or ycol not in df.columns:
            ax.set_title(f"{construct}: missing columns")
            ax.axis("off")
            save_figure(fig, out_dir / f"calibration_scatter_{dataset}_{construct}.png")
            continue
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        work = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(work):
            ax.scatter(work["x"], work["y"], alpha=0.6, s=18)
            lo = min(work["x"].min(), work["y"].min())
            hi = max(work["x"].max(), work["y"].max())
            ax.plot([lo, hi], [lo, hi], "k--", label="Identity")
            ax.legend(loc="best")
        proxy_note = " (proxy: CRT2 adj specified -> BCRT realised)" if construct == "BCRT_proxy" else ""
        ax.set_title(
            f"Specified predictor vs realised behaviour{proxy_note}\n{dataset} - {construct}\nN={len(work)} {_warn_text(work['x'] if len(work) else x, work['y'] if len(work) else y)}"
        )
        ax.set_xlabel(spec["x_axis_label"])
        ax.set_ylabel("Realised behavioural expression")
        save_figure(fig, out_dir / f"calibration_scatter_{dataset}_{construct}.png")


def plot_calibration_panel(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, spec in zip(axes.ravel(), PAIR_MAP):
        construct, xcol, ycol = spec["construct"], spec["xcol"], spec["ycol"]
        if xcol not in df.columns or ycol not in df.columns:
            ax.set_title(f"{construct} (missing)")
            ax.axis("off")
            continue
        work = pd.DataFrame({"x": pd.to_numeric(df[xcol], errors="coerce"), "y": pd.to_numeric(df[ycol], errors="coerce")}).dropna()
        if len(work):
            ax.scatter(work["x"], work["y"], alpha=0.6, s=10)
        warn = _warn_text(work["x"], work["y"]) if len(work) else "WARNING: N small or variance=0; interpret cautiously."
        proxy_note = " (CRT2 adj specified proxy)" if construct == "BCRT_proxy" else ""
        ax.set_title(f"{construct}{proxy_note} | N={len(work)} {warn}".strip())
        ax.set_xlabel(spec["x_axis_label"])
        ax.set_ylabel("Realised behavioural expression")
    fig.suptitle(f"Specified predictor vs realised agent behaviour: {dataset}")
    save_figure(fig, out_path)


def plot_abs_error_hists(errors_df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    setup_style()
    groups = [("core", ["CRT2", "BCRT_proxy", "NFC"]), ("bfi", ["BFI_E", "BFI_A", "BFI_C", "BFI_N", "BFI_O"])]
    for suffix, constructs in groups:
        fig, axes = plt.subplots(len(constructs), 1, figsize=(8, 2.5 * len(constructs)), sharex=False)
        if len(constructs) == 1:
            axes = [axes]
        for ax, c in zip(axes, constructs):
            e = errors_df.loc[errors_df["construct"] == c, "abs_error"].dropna()
            if len(e):
                ax.hist(e, bins=15, histtype="stepfilled", alpha=0.4)
            warn = " WARNING: N small or variance=0; interpret cautiously." if len(e) < 3 or (len(e) > 1 and e.std(ddof=1) == 0) else ""
            ax.set_title(f"{c} absolute error | N={len(e)}{warn}")
            ax.set_xlabel("Absolute error")
            ax.set_ylabel("Count")
        fig.suptitle(f"Per-agent absolute error distributions ({dataset}, {suffix.upper()})")
        save_figure(fig, out_dir / f"calibration_abs_error_hist_{dataset}_{suffix}.png")
