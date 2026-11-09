"""Human CRT item frame: one row per person and CRT2 / bCRT item, from the human dataset.

Reads ``data/inputs/person_with_liwc_brt_coded.csv`` (the companion study's person file with scoring_v2 item
labels) and writes ``reports/cr13_human/crt2_long_human_v2_frame473.csv`` (473 people x 8 items: raw answer,
normalised answer, v2 label). Every human CRT reference (the familiarity-adjusted CRT2 reference, the human
lure rates and their intervals) is computed from this file.

Usage: python analyse/pipeline/human_reference.py [--root DIR]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import HUMAN_LONG, PERSON_FILE, parser, require, resolve_root


def build_long(person_file) -> pd.DataFrame:
    from socsim.common import CRT_ITEM_TEXT_COLS, read_csv_keep_text
    from socsim.familiarity_reference import BCRT_ITEMS, CRT2_ITEMS
    p = read_csv_keep_text(person_file, CRT_ITEM_TEXT_COLS, low_memory=False)
    rows = []
    for item in CRT2_ITEMS + BCRT_ITEMS:
        rows.append(pd.DataFrame({
            "source": "human", "run_label": "human",
            "entity_id": p["PROLIFIC_PID"].astype(str),
            "item_id": item,
            "raw_response": p[item] if item in p.columns else np.nan,
            "normalised": (p[item].astype("string").str.strip().str.lower()
                           if item in p.columns else pd.Series([pd.NA] * len(p))),
            "label_v2": p[f"{item}_label"],
        }))
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    person = require(root / PERSON_FILE, "human person file (controlled-access human dataset)")
    long = build_long(person)
    out = root / HUMAN_LONG
    out.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out, index=False)
    print(f"wrote {out} ({long['entity_id'].nunique()} people, {len(long)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
