"""CRT tables: the familiarity-adjusted human CRT2 reference and each run's category-mass divergence from it.

* The human CRT2 reference adjusted for prior exposure: per CRT2 item a multinomial logit of the response
  category on the number of CRT2 items seen before, age, gender and education, averaged over people at
  seen = 0 (socsim.familiarity_reference.build_adjusted_crt2_reference); pooled = mean of the four items.
* Per run: Jensen-Shannon divergence of the run's CRT2 category mass from the adjusted and from the raw
  human reference, and of its bCRT mass from the raw human bCRT reference (bCRT is not adjusted); the
  per-item lure profile (run and human lure %) is written beside it.

Reads ``reports/cr13_human/crt2_long_human_v2_frame473.csv``, ``data/inputs/person_with_liwc_brt_coded.csv``
(human, controlled access) and the seven runs' agent person frames. Writes, under
``reports/crt_tables/tables/``: crt2ref_adjusted_pooled.csv, js_category_mass_by_run.csv (with the
DeepSeek V4-Pro BRT N note) and per_item_lure_profiles_beside_js.csv.

Usage: python analyse/pipeline/crt_tables.py [--root DIR]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CRT_TABLES, HUMAN_LONG, PERSON_FILE, RUN_DISPLAY, RUN_LABELS, parser, require,
                    resolve_root, stop, with_note)
from frames import agent_frame

CATEGORY_TO_V2 = {"correct": "correct", "intuitive_incorrect": "intuitive", "other_incorrect": "other"}


def run_long(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """A run's item frame in the shape the reference comparison reads (v2 label vocabulary)."""
    from socsim.familiarity_reference import BCRT_ITEMS, CRT2_ITEMS
    rows = []
    for item in CRT2_ITEMS + BCRT_ITEMS:
        col = f"{item}_final_response_category"
        if col not in df.columns:
            stop(f"{label} lacks named column `{col}`")
        cat = df[col].astype("string")
        rows.append(pd.DataFrame({
            "source": label, "run_label": label,
            "entity_id": df["entity_id"].astype(str), "item_id": item,
            "label_v2": cat.map(CATEGORY_TO_V2),
            "normalised": np.where(cat.isin(list(CATEGORY_TO_V2)), "x", ""),
        }))
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    from socsim.common import HUMAN_LONG_TEXT_COLS, read_csv_keep_text
    from socsim.familiarity_reference import (BCRT_ITEMS, OUTCOME_LEVELS, _props_from_long,
                                              adjusted_vectors_from_result, build_adjusted_crt2_reference,
                                              compare_cell_to_references, restrict_to_crt2_complete)
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    human_long = require(root / HUMAN_LONG, "human CRT item frame (run human_reference.py)")
    person = require(root / PERSON_FILE, "human person file (controlled-access human dataset)")
    frames = {label: agent_frame(root, label) for label in RUN_LABELS}
    tab = root / CRT_TABLES
    tab.mkdir(parents=True, exist_ok=True)

    res = build_adjusted_crt2_reference(human_long, person)
    adj_pooled, adj_item, raw_pooled, raw_item, lure_adj, lure_raw = adjusted_vectors_from_result(res)
    res.adjusted_pooled.drop(columns=["frame_caveat"], errors="ignore").to_csv(
        tab / "crt2ref_adjusted_pooled.csv", index=False)

    # human bCRT reference: the 471 CRT2-complete people, as answered
    hl = restrict_to_crt2_complete(read_csv_keep_text(human_long, HUMAN_LONG_TEXT_COLS))
    bcrt_raw_pooled = _props_from_long(hl, BCRT_ITEMS)
    bcrt_raw_item = {it: _props_from_long(hl, [it]) for it in BCRT_ITEMS}
    human_bcrt_lure = {it: 100.0 * bcrt_raw_item[it][1] for it in BCRT_ITEMS}

    js_rows, lure_rows = [], []
    for label in RUN_LABELS:
        df = frames[label]
        cmp = compare_cell_to_references(
            run_long(df, label), source=label,
            adj_crt2_pooled=adj_pooled, adj_crt2_by_item=adj_item,
            raw_crt2_pooled=raw_pooled, raw_crt2_by_item=raw_item,
            raw_bcrt_pooled=bcrt_raw_pooled, raw_bcrt_by_item=bcrt_raw_item,
            human_crt2_lure_adj=lure_adj, human_crt2_lure_raw=lure_raw,
            human_bcrt_lure=human_bcrt_lure)
        row = {k: v for k, v in cmp["js_row"].items() if k != "frame_caveat"}
        row.update({"run_label": label, "run_display": RUN_DISPLAY[label], "n_personas": int(len(df))})
        js_rows.append(row)
        for r in cmp["lure_rows"]:
            r = {k: v for k, v in r.items() if k != "frame_caveat"}
            r.update({"run_label": label, "run_display": RUN_DISPLAY[label]})
            lure_rows.append(r)

    js = with_note(pd.DataFrame(js_rows))
    js.to_csv(tab / "js_category_mass_by_run.csv", index=False)
    pd.DataFrame(lure_rows).to_csv(tab / "per_item_lure_profiles_beside_js.csv", index=False)

    print(f"human frame N = {res.n_persons_frame}; adjusted pooled CRT2 reference = "
          f"{dict(zip(OUTCOME_LEVELS, np.round(adj_pooled, 4)))}")
    print(js[["run_label", "js_crt2_vs_raw", "js_crt2_vs_adjusted", "js_bcrt_vs_raw"]].round(4).to_string(index=False))
    print(f"wrote {tab}/crt2ref_adjusted_pooled.csv, js_category_mass_by_run.csv, per_item_lure_profiles_beside_js.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
