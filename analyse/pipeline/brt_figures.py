"""BRT model figures: one forest per primary family and the outcome x run verdict grid.

Forest: per member outcome, the human OR on bCRT drawn as a reference line (dotted: its 95% interval) and
each run's OR on its own bCRT score (F3) beside it; open squares are unstable models (< 20 events or
non-events); a run whose bCRT score is constant is marked not estimable. Verdict grid: the F3 verdict of
every primary outcome in every run.

Reads ``reports/brt_models/tables/s11_part2_primary_outcomes_by_run.csv`` (brt_models.py). Writes
``reports/brt_models/figures/fig_s11_forest_H1..H4.{pdf,png}`` and ``fig_s11_verdict_grid.{pdf,png}``.

Usage: python analyse/pipeline/brt_figures.py [--root DIR]
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterSciNotation, NullFormatter  # noqa: E402

import spec  # noqa: E402
from common import BRT_MODELS, RUN_LABELS, RUN_SHORT, parser, require, resolve_root  # noqa: E402
from paper_palette import HUMAN_INK, RUN_COLORS  # noqa: E402

# embed TrueType fonts (no Type 3 fonts in the PDFs)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

VERDICT_ORDER = ["transfers", "indeterminate", "fails to transfer", "unstable",
                 "not estimable (no realised bCRT variance)", "unavailable"]
VERDICT_SHORT = {"transfers": "tran", "indeterminate": "indt", "fails to transfer": "fail",
                 "unstable": "unst", "not estimable (no realised bCRT variance)": "n/e",
                 "unavailable": "n/a"}
VERDICT_COLORS = ["#2c7fb8", "#c7c7c7", "#d95f02", "#7f7f7f", "#f0e442", "#ffffff"]

# Labelled ticks added beside the human reference lines (just above OR 1), per family; on the wide H1-H3
# axes two ticks bracket every human line and interval.
HUMAN_SIDE_TICKS = {"H1": (0.5, 2), "H2": (0.5, 2), "H3": (0.5, 1.5), "H4": (1.1, 1.2, 1.3, 1.4)}


class _DecimalLogFormatter(LogFormatterSciNotation):
    """Matplotlib's default log-axis formatter, printing the ticks it labels as plain decimals ("0.8").

    Which major and minor ticks carry a label is the default's choice; only the text changes.
    """

    def __call__(self, x, pos=None):
        if not super().__call__(x, pos):
            return ""
        return np.format_float_positional(float(f"{x:.6g}"), trim="-")


def _decimal_log_labels(axis) -> None:
    axis.set_major_formatter(_DecimalLogFormatter())
    axis.set_minor_formatter(_DecimalLogFormatter(labelOnlyBase=False))


def _add_labelled_ticks(ax, extra=()) -> None:
    """Keep every tick the default formatter labels and add `extra` (inside the axis range), same text.

    The labelled ticks become one fixed major set; unlabelled minor tick marks stay as they were.
    """
    if not extra:
        return
    ax.figure.canvas.draw()
    lo, hi = ax.get_xlim()
    shown = {round(t.get_loc(), 6) for t in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks()
             if t.label1.get_visible() and t.label1.get_text() and lo <= t.get_loc() <= hi}
    locs = sorted(shown | {v for v in extra if lo <= v <= hi})
    ax.xaxis.set_major_locator(FixedLocator(locs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: np.format_float_positional(float(f"{x:.6g}"), trim="-")))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo, hi)


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(base.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def forest(df: pd.DataFrame, family: str, title: str, fig_dir: Path) -> None:
    members = [c for _, c, _, _ in spec.PRIMARY_MEMBERS if spec.OUTCOME_FAMILY[c] == family]
    n_runs = len(RUN_LABELS)
    fig, ax = plt.subplots(figsize=(9.5, 1.4 + 0.30 * (n_runs + 2.6) * len(members)))
    yticks, ylabels = [], []
    y = 0.0
    for col in members:
        sub = df[df.outcome == col]
        h_or = float(sub["human_or"].iloc[0])
        h_lo, h_hi = float(sub["human_or_lo"].iloc[0]), float(sub["human_or_hi"].iloc[0])
        block_top = y
        for label in RUN_LABELS:
            r = sub[sub.run_label == label]
            if r.empty:
                y += 1
                continue
            r = r.iloc[0]
            yticks.append(y)
            ylabels.append(RUN_SHORT[label])
            if not bool(r.get("f3_estimable", True)):
                # the run's bCRT score is constant: nothing to plot
                ax.text(1.0, y, "  not estimable (realised bCRT constant at 4/4)",
                        fontsize=6.5, va="center", ha="left", color="#8a6d00", zorder=5,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.6})
                y += 1
                continue
            lo, hi, orr = r["f3_or_run_lo"], r["f3_or_run_hi"], r["f3_or_run"]
            marker = "s" if bool(r["c3_unstable"]) else "o"
            face = "none" if bool(r["c3_unstable"]) else RUN_COLORS[label]
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [y, y], color=RUN_COLORS[label], lw=1.5, alpha=.85, zorder=2)
            ax.plot([orr], [y], marker=marker, color=RUN_COLORS[label], markerfacecolor=face, markersize=6, zorder=3)
            y += 1
        # the human reference for this member's block
        ax.axhspan(block_top - .5, y - .5, color="#f5f5f5", zorder=0)
        ax.plot([h_or, h_or], [block_top - .5, y - .5], color=HUMAN_INK, lw=2.0, zorder=4)
        ax.plot([h_lo, h_lo], [block_top - .5, y - .5], color=HUMAN_INK, lw=.8, ls=":", zorder=4)
        ax.plot([h_hi, h_hi], [block_top - .5, y - .5], color=HUMAN_INK, lw=.8, ls=":", zorder=4)
        name = spec.PAPER_FACING.get(col, col) or col
        # x in axes fraction, y in data coordinates: the label stays inside the axes on any log range
        ax.text(0.01, block_top - .75,
                f"{name}  ·  human OR {h_or:.3f} [{h_lo:.3f}, {h_hi:.3f}]  ·  prior {spec.PRIOR.get(col, '')}",
                fontsize=8, va="bottom", style="italic", transform=ax.get_yaxis_transform())
        y += 1.6
    ax.axvline(1.0, color="#999999", lw=1, ls="--", zorder=1)
    ax.set_xscale("log")
    _decimal_log_labels(ax.xaxis)
    # headroom for the first block's label (a one-member family gets too small an automatic margin)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -1.3), hi)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("OR per realised bCRT item (F3); black line = human target OR, dotted = its 95% CI")
    ax.set_title(f"Part II - Family {family}: {title}\n"
                 "open square = unstable (C3: <20 events or <20 non-events)", fontsize=10)
    _add_labelled_ticks(ax, HUMAN_SIDE_TICKS.get(family, ()))
    _save(fig, fig_dir / f"fig_s11_forest_{family}")


def verdict_grid(df: pd.DataFrame, fig_dir: Path) -> None:
    idx = [(spec.OUTCOME_FAMILY[c], c) for _, c, _, _ in spec.PRIMARY_MEMBERS]
    mat = np.full((len(idx), len(RUN_LABELS)), len(VERDICT_ORDER) - 1, dtype=float)
    for i, (_, col) in enumerate(idx):
        for j, label in enumerate(RUN_LABELS):
            r = df[(df.outcome == col) & (df.run_label == label)]
            if not r.empty:
                v = str(r.iloc[0]["f3_verdict"])
                mat[i, j] = VERDICT_ORDER.index(v) if v in VERDICT_ORDER else len(VERDICT_ORDER) - 1
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(idx) + 2.4))
    ax.imshow(mat, cmap=ListedColormap(VERDICT_COLORS), vmin=0, vmax=len(VERDICT_ORDER) - 1, aspect="auto")
    ax.set_xticks(range(len(RUN_LABELS)))
    ax.set_xticklabels([RUN_SHORT[l] for l in RUN_LABELS], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([f"[{f}] {spec.PAPER_FACING.get(c, c) or c}" for f, c in idx], fontsize=7)
    for i in range(len(idx)):
        for j in range(len(RUN_LABELS)):
            ax.text(j, i, VERDICT_SHORT[VERDICT_ORDER[int(mat[i, j])]], ha="center", va="center", fontsize=6)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in VERDICT_COLORS[:-1]]
    ax.legend(handles, VERDICT_ORDER[:-1], loc="upper center", bbox_to_anchor=(.5, -.22),
              ncol=3, fontsize=7, frameon=False)
    ax.set_title("Part II - F3 association-transfer verdict grid (primary outcomes x run)", fontsize=10)
    _save(fig, fig_dir / "fig_s11_verdict_grid")


def main(argv: list[str] | None = None) -> int:
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    src = require(root / BRT_MODELS / "tables" / "s11_part2_primary_outcomes_by_run.csv", "run brt_models.py")
    df = pd.read_csv(src)
    fig_dir = root / BRT_MODELS / "figures"
    for fam in spec.FAMILIES:
        forest(df, fam["key"], fam["title"], fig_dir)
    verdict_grid(df, fig_dir)
    print(f"wrote {len(list(fig_dir.glob('fig_s11_*.png')))} figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
