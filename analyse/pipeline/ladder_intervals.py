"""Human bCRT lure rate per item with a 95% interval, for the lure figure's per-item panel.

Per bCRT item: the intuitive-lure rate among the scored answers of the 471 CRT2-complete people (blank or
unparseable answers dropped), with a Wilson 95% interval (the builder's wilson_ci). The human reference
frame is the builder's human lure-reference frame.

Reads ``reports/cr13_human/crt2_long_human_v2_frame473.csv`` and ``data/inputs/person_with_liwc_brt_coded.csv``
(human, controlled access). Writes ``reports/ladder_intervals/human_item_ci.csv``.

Usage: python analyse/pipeline/ladder_intervals.py [--root DIR]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import HUMAN_LONG, LADDER_INTERVALS, PERSON_FILE, builder, parser, require, resolve_root

HUMAN_CSV = "human_item_ci.csv"


def human_bcrt(root: Path) -> pd.DataFrame:
    B = builder()
    require(root / HUMAN_LONG, "human CRT item frame (run human_reference.py)")
    require(root / PERSON_FILE, "human person file (controlled-access human dataset)")
    long, _ = B.human_item_frame(root)
    scored = long[long["category"] != B.UNSCORED]
    rows = []
    for item in B.BCRT_ITEMS:
        s = scored[scored["item_id"] == item]
        k, n = int(s["category"].eq("intuitive_incorrect").sum()), int(len(s))
        lo, hi = B.wilson_ci(k, n)
        rows.append({"test": "bCRT", "item": item.split("_")[1], "reference": "raw", "rate": k / n,
                     "ci95_low": lo, "ci95_high": hi, "method": "Wilson",
                     "n_people": int(s["entity_id"].nunique())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    t = human_bcrt(root)
    out = root / LADDER_INTERVALS
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / HUMAN_CSV, index=False)
    print(f"wrote {out / HUMAN_CSV} ({len(t)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
