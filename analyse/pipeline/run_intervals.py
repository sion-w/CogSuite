"""Persona-clustered lure-rate intervals for the seven runs and the human raw reference.

Per run and scope (CRT2, bCRT, pooled): the intuitive-lure rate over scored answers (blank or unparseable
answers dropped) with socsim.configuration_stats.cluster_bootstrap_proportion_ci (persona-clustered
percentile bootstrap, 5000 resamples, seed 42) on the three-column frame (entity_id, instrument,
response_category) that the builder's bootstrap_lure_ci builds; the same for the 471 CRT2-complete people
(the builder's human lure-reference frame). The item-level Wilson interval of each test is written beside it.
Method check: on a public configuration cell, this script's frame and the builder's bootstrap_lure_ci give
the same interval.

Reads ``reports/cr13_<run>/tables/comparisons/05_crt_response_categories_long.csv`` (agent rows; the file
also holds human rows, so it is controlled access), the human CRT item frame and person file. Writes
``reports/run_intervals/run_ci.csv`` and ``human_pooled_ci.csv``.

Usage: python analyse/pipeline/run_intervals.py [--root DIR]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import (HUMAN_LONG, PERSON_FILE, RUN_INTERVALS, RUN_LABELS, builder, parser, require, resolve_root,
                    run_dir, stop)

SCOPES = ("crt2", "bcrt", "pooled")
AGENT_ID_TYPE = str
HUMAN_ID_TYPE = str
N_BOOT = 5000
SEED = 42
UNSCORED = "missing_or_unparseable"
LURE = "intuitive_incorrect"
GUARD_CELL, GUARD_SCOPE, GUARD_ID_TYPE = "CRFSD_KIMI_K3_NUM", "crt2", int
RUN_METHOD = f"persona-clustered bootstrap, {N_BOOT} resamples; scored denominator"
HUMAN_METHOD = f"person-clustered bootstrap, {N_BOOT} resamples; scored denominator"


def three_col(entity_id: pd.Series, item_id: pd.Series, category: pd.Series, id_type: type) -> pd.DataFrame:
    """bootstrap_lure_ci's frame: entity_id cast to id_type, instrument from the item id, the category."""
    return pd.DataFrame({"entity_id": entity_id.astype(id_type).to_numpy(),
                         "instrument": np.where(item_id.str.startswith("crt2"), "crt2", "bcrt"),
                         "response_category": category.to_numpy()})


def agent_long(root: Path, run: str) -> pd.DataFrame:
    """A run's scored socsim_agent rows (CRT2 and bCRT items) as the three-column frame."""
    B = builder()
    path = require(run_dir(root, run) / "tables" / "comparisons" / "05_crt_response_categories_long.csv",
                   "per-run CRT response categories (controlled access: the file holds human rows)")
    d = pd.read_csv(path, dtype={"entity_id": str, "item_id": str})
    d = d[(d["dataset"] == "socsim_agent") & d["item_id"].isin(B.CRT2_ITEMS + B.BCRT_ITEMS)]
    if not (np.where(d["item_id"].str.startswith("crt2"), "crt2", "bcrt") == d["instrument"].to_numpy()).all():
        stop(f"{run}: the instrument column disagrees with the item ids.")
    d = d[d["response_category"] != UNSCORED]
    return three_col(d["entity_id"], d["item_id"], d["response_category"], AGENT_ID_TYPE)


def human_long(root: Path) -> pd.DataFrame:
    """The 471 CRT2-complete people's scored rows (v2 labels) as the three-column frame."""
    B = builder()
    require(root / HUMAN_LONG, "human CRT item frame (run human_reference.py)")
    require(root / PERSON_FILE, "human person file (controlled-access human dataset)")
    long, _ = B.human_item_frame(root)
    s = long[long["item_id"].isin(B.CRT2_ITEMS + B.BCRT_ITEMS) & (long["category"] != B.UNSCORED)]
    return three_col(s["entity_id"], s["item_id"], s["category"], HUMAN_ID_TYPE)


def interval(long: pd.DataFrame, scope: str) -> dict:
    from socsim.configuration_stats import cluster_bootstrap_proportion_ci
    rep = cluster_bootstrap_proportion_ci(long, None if scope == "pooled" else scope, category=LURE,
                                          n_boot=N_BOOT, seed=SEED)
    return {"n_events": rep["n_events"], "n_items": rep["n_items"], "n_persons": rep["n_personas"],
            "rate": rep["n_events"] / rep["n_items"], "ci95_low": rep["ci95_low"], "ci95_high": rep["ci95_high"]}


def method_guard(root: Path) -> str:
    """This script's frame and the builder's bootstrap_lure_ci give the same interval on a scored ladder cell."""
    B = builder()
    path = root / B.SCORED_TABLES / f"crt_long_{GUARD_CELL}.csv"
    if not path.exists():
        return f"method check skipped ({path} absent)"
    df = B.read_scored_long(root, GUARD_CELL)
    ref = B.bootstrap_lure_ci(df, GUARD_SCOPE, GUARD_ID_TYPE, scored_only=True)
    cats = B.long_categories(df)
    m = df["item_id"].isin(B.CRT2_ITEMS + B.BCRT_ITEMS) & (cats != B.UNSCORED)
    own = interval(three_col(df.loc[m, "entity_id"], df.loc[m, "item_id"], cats[m], GUARD_ID_TYPE), GUARD_SCOPE)
    if (ref["ci95_low"], ref["ci95_high"]) != (own["ci95_low"], own["ci95_high"]):
        stop(f"{GUARD_CELL} {GUARD_SCOPE}: interval {own['ci95_low'], own['ci95_high']} differs from the "
             f"builder's bootstrap_lure_ci {ref['ci95_low'], ref['ci95_high']}")
    return f"interval method agrees with the builder on {GUARD_CELL} ({GUARD_SCOPE})"


def wilson(r: dict, scope: str) -> tuple[float, float]:
    """The item-level Wilson interval of one test (none for the pooled scope)."""
    if scope == "pooled":
        return np.nan, np.nan
    return builder().wilson_ci(r["n_events"], r["n_items"])


def run_rows(root: Path) -> pd.DataFrame:
    rows = []
    for run in RUN_LABELS:
        long = agent_long(root, run)
        for scope in SCOPES:
            r = interval(long, scope)
            wl, wh = wilson(r, scope)
            rows.append({"run_label": run, "scope": scope, **r, "method": RUN_METHOD,
                         "id_type": AGENT_ID_TYPE.__name__, "seed": SEED, "wilson_low": wl, "wilson_high": wh})
    return pd.DataFrame(rows)


def human_rows(root: Path) -> pd.DataFrame:
    long = human_long(root)
    rows = []
    for scope in SCOPES:
        r = interval(long, scope)
        wl, wh = wilson(r, scope)
        rows.append({"test": scope, "reference": "raw", "n_events": r["n_events"], "n_items": r["n_items"],
                     "n_people": r["n_persons"], "rate": r["rate"], "ci95_low": r["ci95_low"],
                     "ci95_high": r["ci95_high"], "method": HUMAN_METHOD, "wilson_low": wl, "wilson_high": wh})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    print(method_guard(root))
    runs = run_rows(root)
    hum = human_rows(root)
    out = root / RUN_INTERVALS
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "run_ci.csv", index=False)
    hum.to_csv(out / "human_pooled_ci.csv", index=False)
    print(f"wrote {out / 'run_ci.csv'} ({len(runs)} rows), {out / 'human_pooled_ci.csv'} ({len(hum)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
