"""Prevalence of the BRT behaviours that people reliably link to reflection, at each reflection level.

The behaviours: the human primary-family rows (``reports/human_brt_families/tables/s5_primary_families.csv``)
with q_bh_within_family < .05 (five). The seven runs are split by the CRT2 level each persona was told
(crt2_level, 0-4; agent person frames, a missing outcome dropped per outcome); people by their own level
(familiarity-adjusted CRT2 at the frozen cutoffs; the builder's human lure-reference frame, 471
CRT2-complete people), joined on the person ID to the outcomes of the 473-person human frame. Each cell: n,
events, prevalence and a Wilson 95% interval.

Checks: the told-level and own-level mixes; per-level events and n sum to the run totals of the BRT model
family tables (``reports/brt_models/tables/s11_family_H1..H4_table.csv``), whose Wilson bounds wilson_ci
reproduces; the people's per-level events plus those of the people not CRT2-complete equal the human totals.

Writes ``reports/level_prevalence/level_prevalence_runs.csv`` and ``level_prevalence_human.csv``.
Participant IDs are read in memory, never written.

Usage: python analyse/pipeline/level_prevalence.py [--root DIR]
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import spec
from common import (BRT_MODELS, HUMAN_FAMILIES_TABLES, HUMAN_LONG, LEVEL_PREVALENCE, RUN_LABELS, builder,
                    parser, require, require_columns, resolve_root, stop)
from frames import agent_frame, brt_n, human_frame
from brt_stats import wilson_ci

LEVELS = (0, 1, 2, 3, 4)
SYNTH_TOLD_MIX = (109, 86, 103, 106, 67)
HUMDERIVED_TOLD_MIX = (94, 98, 106, 105, 68)
HUMAN_LEVEL_MIX = (107, 79, 106, 111, 68)
HUMAN_JOINED = 471
EXPECTED_BEHAVIOURS = ("BRT1_check_venue", "BRT2_check_institutional_sources", "BRT2_check_persons",
                       "BRT1_check_official", "BRT2_emo_worry")
Q_THRESHOLD = 0.05
WILSON_TOL = 1e-9

RUNS = tuple(RUN_LABELS)
HUMDERIVED = tuple(r for r in RUNS if r.startswith("CR_BASELINE_HUMAN_"))
LEVEL = spec.F2_PREDICTOR
RUN_LEVEL_SOURCE = "told (crt2_level)"
HUMAN_LEVEL_SOURCE = "own level: familiarity-adjusted CRT2 at the frozen cutoffs (471 CRT2-complete people)"
METHOD = "Wilson 95%"
COLUMNS = ["run_label", "outcome", "paper_facing_name", "level", "level_source", "n", "events", "prevalence",
           "wilson_lo", "wilson_hi", "method"]
FAMILY_TABLES = [f"s11_family_H{i}_table.csv" for i in range(1, 5)]


def told_mix(run: str) -> tuple[int, ...]:
    return HUMDERIVED_TOLD_MIX if run in HUMDERIVED else SYNTH_TOLD_MIX


# ---------------------------------------------------------------- inputs

def behaviours(root: Path) -> pd.DataFrame:
    """The human primary rows with q < .05 within family, in table order; STOP unless they are the five."""
    t = pd.read_csv(require(root / HUMAN_FAMILIES_TABLES / "s5_primary_families.csv", "run human_brt_families.py"))
    b = t.loc[t["q_bh_within_family"] < Q_THRESHOLD, ["outcome", "paper_facing_name"]].reset_index(drop=True)
    if sorted(b["outcome"]) != sorted(EXPECTED_BEHAVIOURS) or b["outcome"].duplicated().any():
        stop(f"the human table gives {list(b['outcome'])} at q < .05, not the five {list(EXPECTED_BEHAVIOURS)}.")
    return b


def family_tables(root: Path, outcomes: list[str]) -> pd.DataFrame:
    """The four BRT model family tables, one row for every run x behaviour."""
    t = pd.concat([pd.read_csv(require(root / BRT_MODELS / "tables" / f, "run brt_models.py"),
                               float_precision="round_trip") for f in FAMILY_TABLES], ignore_index=True)
    t = t[t["run_label"].isin(RUNS) & t["outcome"].isin(outcomes)]
    want = {(r, o) for r in RUNS for o in outcomes}
    have = list(zip(t["run_label"], t["outcome"]))
    if set(have) != want or len(have) != len(want):
        stop(f"family tables: expected one row per run x behaviour ({len(want)}), got {len(have)}.")
    return t.set_index(["run_label", "outcome"])


def human_joined(root: Path, outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """People's outcomes (473) joined to their own level (471). Returns (the joined 471, the whole 473 frame
    with a level column, NaN for the people not CRT2-complete, the join counts)."""
    h = human_frame(root)
    require_columns(h, ["entity_id", *outcomes], "human frame")
    require(root / HUMAN_LONG, "human CRT item frame (run human_reference.py)")
    _, people = builder().human_item_frame(root)
    h = h[["entity_id", *outcomes]].assign(entity_id=lambda d: d["entity_id"].astype(str))
    lv = people[[LEVEL]].copy()
    lv.index = lv.index.astype(str)
    full = h.merge(lv, left_on="entity_id", right_index=True, how="left", validate="one_to_one")
    joined = full[full[LEVEL].notna()].copy()
    joined[LEVEL] = joined[LEVEL].astype(int)
    counts = {"human_frame_n": len(h), "levels_n": len(people), "joined": len(joined),
              "not_crt2_complete": int(full[LEVEL].isna().sum()),
              "levels_not_in_frame": int((~lv.index.isin(h["entity_id"])).sum())}
    return joined.drop(columns="entity_id"), full.drop(columns="entity_id"), counts


# ---------------------------------------------------------------- the computation

def cell(y: pd.Series) -> dict:
    """n, events, prevalence and Wilson 95% bounds of a 0/1 series (missing values dropped)."""
    y = pd.to_numeric(y, errors="coerce").dropna()
    if not y.isin([0, 1]).all():
        stop(f"an outcome holds values other than 0/1: {sorted(set(y) - {0, 1})[:5]}")
    n, k = int(len(y)), int(y.sum())
    if n == 0:
        stop("an empty level cell; a prevalence is undefined.")
    lo, hi = wilson_ci(k, n)
    # Wilson's bounds are exactly 0 at k = 0 and exactly 1 at k = n; the closed form can miss by 1 ulp there
    # (67/67 -> 0.9999999999999998, below its prevalence), so the exact edge value is written.
    lo, hi = (0.0 if k == 0 else float(lo)), (1.0 if k == n else float(hi))
    return {"n": n, "events": k, "prevalence": k / n, "wilson_lo": lo, "wilson_hi": hi}


def level_rows(df: pd.DataFrame, label: str, beh: pd.DataFrame, level_source: str) -> list[dict]:
    rows = []
    for o, name in zip(beh["outcome"], beh["paper_facing_name"]):
        for lv in LEVELS:
            rows.append({"run_label": label, "outcome": o, "paper_facing_name": name, "level": lv,
                         "level_source": level_source, **cell(df.loc[df[LEVEL] == lv, o]), "method": METHOD})
    return rows


def run_table(frames: dict[str, pd.DataFrame], beh: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([r for run in RUNS for r in level_rows(frames[run], run, beh, RUN_LEVEL_SOURCE)],
                        columns=COLUMNS)


def human_table(joined: pd.DataFrame, beh: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(level_rows(joined, "human", beh, HUMAN_LEVEL_SOURCE), columns=COLUMNS)


# ---------------------------------------------------------------- checks

def check_frames(frames: dict[str, pd.DataFrame], outcomes: list[str]) -> list[str]:
    lines = []
    for run in RUNS:
        d = frames[run]
        require_columns(d, [LEVEL, *outcomes], run)
        mix = tuple(int((d[LEVEL] == lv).sum()) for lv in LEVELS)
        if mix != told_mix(run) or len(d) != sum(told_mix(run)):
            stop(f"{run}: told-level mix {mix} over {len(d)} personas, expected {told_mix(run)}.")
        lines.append(f"{run}: {len(d)} personas, told mix {mix}, BRT n {brt_n(d)}")
    return lines


def check_sums(runs: pd.DataFrame, fam: pd.DataFrame) -> int:
    """Per-level events and n sum to the family table's run_events and run_n, every run x behaviour."""
    s = runs.groupby(["run_label", "outcome"])[["events", "n"]].sum()
    for (run, o), r in s.iterrows():
        f = fam.loc[(run, o)]
        if (int(r["events"]), int(r["n"])) != (int(f["run_events"]), int(f["run_n"])):
            stop(f"{run} {o}: per-level events/n {int(r['events'])}/{int(r['n'])}, family table "
                 f"{int(f['run_events'])}/{int(f['run_n'])}.")
    return len(s)


def check_human(human: pd.DataFrame, full: pd.DataFrame, counts: dict, fam: pd.DataFrame) -> None:
    if counts["joined"] != HUMAN_JOINED or counts["levels_not_in_frame"]:
        stop(f"human join matched {counts['joined']} people ({counts['levels_not_in_frame']} levels with no frame "
             f"row), expected {HUMAN_JOINED}.")
    n_lv = tuple(int(human[(human["outcome"] == human["outcome"].iloc[0]) & (human["level"] == lv)]["n"].iloc[0])
                 for lv in LEVELS)
    if n_lv != HUMAN_LEVEL_MIX:
        stop(f"human level mix {n_lv}, expected {HUMAN_LEVEL_MIX}.")
    if counts["human_frame_n"] - counts["joined"] != counts["not_crt2_complete"]:
        stop("the human join does not account for every person.")
    for o, g in human.groupby("outcome", sort=False):
        f = fam.xs(o, level="outcome")
        he, hn = f["human_events"].unique(), f["human_n"].unique()
        left_out = full[full[LEVEL].isna()][o]
        full_events = int(pd.to_numeric(full[o], errors="coerce").sum())
        if len(he) != 1 or len(hn) != 1 or (full_events, len(full)) != (int(he[0]), int(hn[0])):
            stop(f"human {o}: frame events/n {full_events}/{len(full)}, family table {list(he)}/{list(hn)}.")
        if int(g["events"].sum()) + int(left_out.sum()) != full_events:
            stop(f"human {o}: per-level events {int(g['events'].sum())} + left-out {int(left_out.sum())} "
                 f"!= {full_events}.")


def check_wilson(fam: pd.DataFrame) -> float:
    """wilson_ci on each run x behaviour total reproduces the family table's run_prev_lo/hi_pct."""
    worst = 0.0
    for (run, o), f in fam.iterrows():
        lo, hi = wilson_ci(int(f["run_events"]), int(f["run_n"]))
        d = max(abs(lo - f["run_prev_lo_pct"] / 100), abs(hi - f["run_prev_hi_pct"] / 100))
        if not d <= WILSON_TOL:
            stop(f"{run} {o}: wilson_ci differs from the family table by {d:.2e}.")
        worst = max(worst, d)
    return worst


def main(argv: list[str] | None = None) -> int:
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    beh = behaviours(root)
    outcomes = list(beh["outcome"])
    fam = family_tables(root, outcomes)
    names = fam.reset_index().groupby("outcome")["paper_facing_name"].unique()
    if any(list(names[o]) != [n] for o, n in zip(beh["outcome"], beh["paper_facing_name"])):
        stop("the human table and the family tables name a behaviour differently.")

    frames = {run: agent_frame(root, run) for run in RUNS}
    for line in check_frames(frames, outcomes):
        print(line)
    runs = run_table(frames, beh)
    print(f"per-level sums equal the family tables for {check_sums(runs, fam)} run x behaviour totals")
    joined, full, counts = human_joined(root, outcomes)
    human = human_table(joined, beh)
    check_human(human, full, counts, fam)
    print(f"human join: {counts}")
    print(f"wilson_ci reproduces the family tables' run intervals; max |diff| {check_wilson(fam):.2e}")
    if len(runs) != len(RUNS) * len(outcomes) * len(LEVELS) or len(human) != len(outcomes) * len(LEVELS):
        stop(f"{len(runs)} run rows and {len(human)} human rows.")

    out = root / LEVEL_PREVALENCE
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "level_prevalence_runs.csv", index=False)
    human.to_csv(out / "level_prevalence_human.csv", index=False)
    print(f"wrote {out / 'level_prevalence_runs.csv'} ({len(runs)} rows), "
          f"{out / 'level_prevalence_human.csv'} ({len(human)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
