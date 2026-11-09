from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def setup_style() -> None:
    plt.style.use("default")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 9


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
