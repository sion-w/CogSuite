"""LIWC family models of the seven runs in one table.

Each per-run analysis report fits the LIWC family regressions (LIWC outcome on CRT2 and bCRT, human and agent
side) and writes them to ``reports/cr13_<run>/tables/04_liwc_family_models_all.csv``. This step stacks the
seven tables with the run label and display name in front, and adds the DeepSeek V4-Pro BRT N note.

Writes ``reports/liwc_models/tables/liwc_family_models_all_runs.csv``.

Usage: python analyse/pipeline/liwc_models.py [--root DIR]
"""
from __future__ import annotations

import pandas as pd

from common import LIWC_TABLES, RUN_DISPLAY, RUN_LABELS, parser, require, resolve_root, run_dir, with_note

SOURCE = "tables/04_liwc_family_models_all.csv"


def gather(root) -> pd.DataFrame:
    out = []
    for label in RUN_LABELS:
        p = require(run_dir(root, label) / SOURCE, "per-run LIWC family models (run the per-run analysis)")
        df = pd.read_csv(p)
        df.insert(0, "run_display", RUN_DISPLAY[label])
        df.insert(0, "run_label", label)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    liwc = with_note(gather(root))
    tab = root / LIWC_TABLES
    tab.mkdir(parents=True, exist_ok=True)
    liwc.to_csv(tab / "liwc_family_models_all_runs.csv", index=False)
    agent = liwc[liwc["dataset"] == "socsim_agent"]
    n_sig = int((pd.to_numeric(agent["q_fdr"], errors="coerce") < 0.05).sum())
    print(f"wrote {tab / 'liwc_family_models_all_runs.csv'} ({len(liwc)} rows, {len(agent)} agent; "
          f"{n_sig} agent rows at q < .05)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
