"""CR-30: prevalence of the five BRT behaviours people reliably link to reflection, at each reflection level
(D-253(d)): the seven runs by the level each persona was told (crt2_level, 0-4), people by their own level
(familiarity-adjusted CRT2 at the frozen cutoffs), each with a Wilson 95% interval.

The five behaviours: the CR-20 rows (reports/cr20_human/tables/s5_primary_families.csv) with
q_bh_within_family < .05. Agent frames: scripts/cr13b/frames.py agent_frame (the frame CR-13B fitted F2 on), a
missing outcome dropped per outcome exactly as CR-13B did. Human outcomes: the CR-20 frame frames.human_frame()
returns (473 people); human level: the builder's cr25_human_frame (471 CRT2-complete people), joined on the person
ID. Check values: the bundle of record's tables/s11_family_H1-H4_table.csv (per-level sums must equal run_events /
run_n; wilson_ci on the totals must reproduce run_prev_lo_pct / run_prev_hi_pct to 1e-9). Method:
scripts/cr13b/statlib.py wilson_ci, imported.

Code runs from this checkout; data are read from, and outputs written to, the git-ignored shared folders of the
main soc-sim checkout (DATA_ROOT), which a worktree does not have. frames.human_frame() locates its inputs from
the checkout it is loaded from, so human_frame() below calls the same CR-20 build_frame with only the two data
paths pointed at DATA_ROOT (the code, derived.py included, stays this checkout's).

Statistics only; no model call. Participant IDs and free text are read in memory, never written: outputs carry
only cell-level counts and intervals.

Usage:
    python scripts/cr30_level_prevalence.py [--data-root DIR] [--out DIR] [--bundle-dir DIR] [--inbound-only]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


SPEC = _load("cr13b_spec_cr30", REPO / "scripts" / "cr13b" / "spec.py")
FRAMES = _load("cr13b_frames_cr30", REPO / "scripts" / "cr13b" / "frames.py")
STATLIB = _load("cr13b_statlib_cr30", REPO / "scripts" / "cr13b" / "statlib.py")
B = _load("build_assets_cr30", REPO / "scripts" / "build_neurips_camera_ready_assets.py")
wilson_ci = STATLIB.wilson_ci

# Ruled values (CR-30 prompt).
CR30_LEVELS = (0, 1, 2, 3, 4)
CR30_SYNTH_TOLD_MIX = (109, 86, 103, 106, 67)
CR30_HUMDERIVED_TOLD_MIX = (94, 98, 106, 105, 68)
CR30_HUMAN_LEVEL_MIX = (107, 79, 106, 111, 68)
CR30_HUMAN_JOINED = 471

EXPECTED_BEHAVIOURS = ("BRT1_check_venue", "BRT2_check_institutional_sources", "BRT2_check_persons",
                       "BRT1_check_official", "BRT2_emo_worry")
HUMAN_FRAME_N = 473
Q_THRESHOLD = 0.05
C3_TOL = 1e-9
SPOT_N, SPOT_SEED = 3, 42

DATA_ROOT = Path.home() / "Desktop" / "phd" / "CR" / "soc-sim"     # git-ignored data shared by every checkout
OUT_DIR = DATA_ROOT / "reports" / "cr30_level_prevalence"
CR20_TABLE = Path("reports") / "cr20_human" / "tables" / "s5_primary_families.csv"
# The bundle of record (hub note, "Where things live", Figures/Tables (of record)).
BUNDLE_DIR = B.VAULT_CR / "cr-25d" / "neurips_paper_assets_v2" / "tables"
FAMILY_TABLES = [f"s11_family_H{i}_table.csv" for i in range(1, 5)]
# C4 only: the runs' stored prompts and the lexical CRT2 ladder they were written from.
SIM_OUTPUTS = DATA_ROOT.parent / "cognition-suite-sim" / "simulation_outputs"
LEXICAL_LADDER = (DATA_ROOT.parent / "cognition-suite-sim" / "data_fitting" / "persona_trait_experiment"
                  / "prompt_config" / "crt2_ladder.yaml")

RUNS = tuple(SPEC.RUN_LABELS)
HUMDERIVED = tuple(r for r in RUNS if r.startswith("CR_BASELINE_HUMAN_"))
LEVEL = SPEC.F2_PREDICTOR
RUN_LEVEL_SOURCE = "told (crt2_level)"
HUMAN_LEVEL_SOURCE = "own level: familiarity-adjusted CRT2 at the frozen cutoffs (471 CRT2-complete people)"
METHOD = "Wilson 95% (CR-30; statlib.wilson_ci)"
COLUMNS = ["run_label", "outcome", "paper_facing_name", "level", "level_source", "n", "events", "prevalence",
           "wilson_lo", "wilson_hi", "method"]

RUN_CSV = "level_prevalence_runs_cr30.csv"
HUMAN_CSV = "level_prevalence_human_cr30.csv"
HEADLINE_MD = "HEADLINE_RESULTS.md"


def stop(msg: str) -> None:
    raise SystemExit(f"MANUAL_ACTION_NEEDED: {msg}")


def told_mix(run: str) -> tuple[int, ...]:
    return CR30_HUMDERIVED_TOLD_MIX if run in HUMDERIVED else CR30_SYNTH_TOLD_MIX


# ---------------------------------------------------------------- inputs

def behaviours(data_root: Path) -> pd.DataFrame:
    """C1: the CR-20 rows with q < .05 within family, in table order; STOP unless they are the expected five."""
    t = pd.read_csv(data_root / CR20_TABLE)
    b = t.loc[t["q_bh_within_family"] < Q_THRESHOLD, ["outcome", "paper_facing_name"]].reset_index(drop=True)
    if sorted(b["outcome"]) != sorted(EXPECTED_BEHAVIOURS) or b["outcome"].duplicated().any():
        stop(f"CR-20 gives {list(b['outcome'])} at q < .05, not the expected five {list(EXPECTED_BEHAVIOURS)}.")
    return b


def family_tables(bundle_dir: Path, outcomes: list[str]) -> pd.DataFrame:
    """C1: the four family tables of record, one row for every run x behaviour."""
    t = pd.concat([pd.read_csv(bundle_dir / f, float_precision="round_trip") for f in FAMILY_TABLES],
                  ignore_index=True)
    t = t[t["run_label"].isin(RUNS) & t["outcome"].isin(outcomes)]
    want = {(r, o) for r in RUNS for o in outcomes}
    have = list(zip(t["run_label"], t["outcome"]))
    if set(have) != want or len(have) != len(want):
        stop(f"family tables: expected one row per run x behaviour ({len(want)}), got {len(have)} "
             f"(missing {sorted(want - set(have))[:3]}).")
    return t.set_index(["run_label", "outcome"])


def agent_frame(data_root: Path, run: str) -> pd.DataFrame:
    """frames.agent_frame(label), reading the shared reports folder."""
    FRAMES.REPORTS = data_root / "reports"
    return FRAMES.agent_frame(run)


def human_frame(data_root: Path) -> pd.DataFrame:
    """frames.human_frame() (CR-20 build_frame, unchanged) with its two data inputs read from data_root."""
    import matplotlib
    matplotlib.use("Agg")
    fl = _load("cr20_frame_lib_cr30", REPO / "scripts" / "cr20_human" / "frame_lib.py")
    sys.path.insert(0, str(REPO / "brt_v3"))
    import derived  # noqa: F401  (this checkout's derived.py, cached before build_frame imports it)
    fl.REPO = data_root
    fl.CANONICAL = data_root / "reports" / "brt_recode_v3" / "brt_labels_v3_human.csv"
    fl.CONTRACT = data_root / "data" / "inputs" / "person_with_liwc_brt_coded.csv"
    df, _ = fl.build_frame()
    return df


def human_joined(data_root: Path, outcomes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """People's outcomes (473) joined to their own level (cr25_human_frame, 471). Returns (joined 471, the
    whole 473 frame with a level column (NaN for the people not CRT2-complete), the join counts)."""
    h = human_frame(data_root)
    FRAMES.require_columns(h, ["entity_id", *outcomes], "human frame")
    _, people = B.cr25_human_frame(data_root)
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
    # Wilson's bounds are exactly 0 at k = 0 and exactly 1 at k = n; wilson_ci's closed form rounds there by up to
    # 1 ulp (67/67 -> 0.9999999999999998, below its prevalence), so the exact edge value is written.
    lo, hi = (0.0 if k == 0 else float(lo)), (1.0 if k == n else float(hi))
    return {"n": n, "events": k, "prevalence": k / n, "wilson_lo": lo, "wilson_hi": hi}


def edge_cells(table: pd.DataFrame) -> pd.DataFrame:
    """The cells at k = 0 or k = n, with wilson_ci's raw bound beside the written one."""
    e = table[(table["events"] == 0) | (table["events"] == table["n"])].copy()
    raw = [wilson_ci(int(k), int(n)) for k, n in zip(e["events"], e["n"])]
    e["raw_lo"], e["raw_hi"] = [float(r[0]) for r in raw], [float(r[1]) for r in raw]
    e["max_abs_diff"] = np.maximum((e["raw_lo"] - e["wilson_lo"]).abs(), (e["raw_hi"] - e["wilson_hi"]).abs())
    return e[["run_label", "outcome", "level", "n", "events", "raw_lo", "wilson_lo", "raw_hi", "wilson_hi",
              "max_abs_diff"]]


def level_rows(df: pd.DataFrame, label: str, beh: pd.DataFrame, level_source: str) -> list[dict]:
    rows = []
    for o, name in zip(beh["outcome"], beh["paper_facing_name"]):
        for lv in CR30_LEVELS:
            rows.append({"run_label": label, "outcome": o, "paper_facing_name": name, "level": lv,
                         "level_source": level_source, **cell(df.loc[df[LEVEL] == lv, o]), "method": METHOD})
    return rows


def run_table(frames: dict[str, pd.DataFrame], beh: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([r for run in RUNS for r in level_rows(frames[run], run, beh, RUN_LEVEL_SOURCE)],
                        columns=COLUMNS)


def human_table(joined: pd.DataFrame, beh: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(level_rows(joined, "human", beh, HUMAN_LEVEL_SOURCE), columns=COLUMNS)


# ---------------------------------------------------------------- inbound checks C1-C3

def check_frames(frames: dict[str, pd.DataFrame], outcomes: list[str]) -> list[str]:
    """C1 and the C2 told-level mix."""
    lines = []
    for run in RUNS:
        d = frames[run]
        FRAMES.require_columns(d, [LEVEL, *outcomes], run)
        mix = tuple(int((d[LEVEL] == lv).sum()) for lv in CR30_LEVELS)
        if mix != told_mix(run) or len(d) != sum(told_mix(run)):
            stop(f"{run}: told-level mix {mix} over {len(d)} personas, ruled {told_mix(run)}.")
        miss = {o: int(d[o].isna().sum()) for o in outcomes if d[o].isna().any()}
        lines.append(f"{run}: {len(d)} personas, told mix {mix}, BRT n {FRAMES.brt_n(d)}"
                     + (f", missing outcome rows {miss}" if miss else ""))
    return lines


def check_sums(runs: pd.DataFrame, fam: pd.DataFrame) -> int:
    """C2: per-level events and n sum to the family table's run_events and run_n, every run x behaviour."""
    s = runs.groupby(["run_label", "outcome"])[["events", "n"]].sum()
    for (run, o), r in s.iterrows():
        f = fam.loc[(run, o)]
        if (int(r["events"]), int(r["n"])) != (int(f["run_events"]), int(f["run_n"])):
            stop(f"{run} {o}: per-level events/n {int(r['events'])}/{int(r['n'])}, family table "
                 f"{int(f['run_events'])}/{int(f['run_n'])}.")
    return len(s)


def check_human(human: pd.DataFrame, full: pd.DataFrame, counts: dict, fam: pd.DataFrame) -> list[str]:
    """C2 (people): the join and level mix, and per-level sums beside the family table's 473-person events."""
    if counts["joined"] != CR30_HUMAN_JOINED or counts["levels_not_in_frame"]:
        stop(f"human join matched {counts['joined']} people ({counts['levels_not_in_frame']} levels with no frame "
             f"row), ruled {CR30_HUMAN_JOINED}.")
    n_lv = tuple(int(human[(human["outcome"] == human["outcome"].iloc[0]) & (human["level"] == lv)]["n"].iloc[0])
                 for lv in CR30_LEVELS)
    if n_lv != CR30_HUMAN_LEVEL_MIX:
        stop(f"human level mix {n_lv}, ruled {CR30_HUMAN_LEVEL_MIX}.")
    if counts["human_frame_n"] - counts["joined"] != counts["not_crt2_complete"]:
        stop(f"human frame {counts['human_frame_n']} - joined {counts['joined']} != "
             f"{counts['not_crt2_complete']} not CRT2-complete.")
    lines = []
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
        lines.append(f"{o}: per-level events {int(g['events'].sum())} of n {int(g['n'].sum())} (471 people) | "
                     f"family table human_events {int(he[0])} of human_n {int(hn[0])} (473) | the "
                     f"{len(left_out)} people not CRT2-complete carry {int(left_out.sum())}")
    return lines


def check_wilson(fam: pd.DataFrame) -> float:
    """C3: wilson_ci on each run x behaviour total reproduces the family table's run_prev_lo/hi_pct to 1e-9."""
    worst = 0.0
    for (run, o), f in fam.iterrows():
        lo, hi = wilson_ci(int(f["run_events"]), int(f["run_n"]))
        d = max(abs(lo - f["run_prev_lo_pct"] / 100), abs(hi - f["run_prev_hi_pct"] / 100))
        if not d <= C3_TOL:
            stop(f"{run} {o}: wilson_ci({int(f['run_events'])}, {int(f['run_n'])}) = ({lo}, {hi}); family table "
                 f"({f['run_prev_lo_pct']}, {f['run_prev_hi_pct']}) %; |diff| {d:.2e} > {C3_TOL}.")
        worst = max(worst, d)
    return worst


# ---------------------------------------------------------------- C4 spot-read

def _prompt_levels(run: str, ids: set[int]) -> dict[int, str]:
    """The CRT line of each wanted persona's stored system prompt (api_calls.jsonl)."""
    out: dict[int, str] = {}
    with open(SIM_OUTPUTS / run / "api_calls.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pid = int(r["persona_id"])
            if pid in ids and pid not in out:
                m = re.search(r"Cognitive Reflection tendency:\s*(.+)", r["system_prompt"])
                out[pid] = m.group(1).strip() if m else ""
            if len(out) == len(ids):
                break
    return out


def prompt_level(text: str, ladder: dict[int, str]) -> int | None:
    m = re.fullmatch(r"CRT level (\d) on a 0-4 scale\.", text)
    if m:
        return int(m.group(1))
    hits = [lv for lv, phrase in ladder.items() if text == phrase.strip()]
    return hits[0] if len(hits) == 1 else None


def spot_read(frames: dict[str, pd.DataFrame], data_root: Path) -> list[str]:
    """C4: 3 random personas per run (told level in the frame vs the stored prompt); 3 people (level by hand)."""
    import yaml
    ladder = {int(k): v for k, v in yaml.safe_load(LEXICAL_LADDER.read_text())["levels"].items()}
    rng = np.random.default_rng(SPOT_SEED)
    lines = []
    for run in RUNS:
        d = frames[run]
        pick = d.iloc[rng.choice(len(d), SPOT_N, replace=False)]
        texts = _prompt_levels(run, set(int(p) for p in pick["persona_id"]))
        for _, r in pick.iterrows():
            t = texts.get(int(r["persona_id"]), "")
            lv = prompt_level(t, ladder)
            ok = lv == int(r[LEVEL])
            lines.append(f"{run} persona {int(r['persona_id'])}: frame level {int(r[LEVEL])}, prompt "
                         f"\"{t[:60]}{'...' if len(t) > 60 else ''}\" -> {lv} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                stop(f"C4 {run} persona {int(r['persona_id'])}: frame level {int(r[LEVEL])}, prompt level {lv}.")
    cuts = yaml.safe_load((data_root / B.CUTOFFS_YAML).read_text())["cutoffs"]
    q = [cuts[k] for k in ("q1_20", "q2_40", "q3_60", "q4_80")]
    _, people = B.cr25_human_frame(data_root)
    for i, (_, p) in enumerate(people.iloc[rng.choice(len(people), SPOT_N, replace=False)].iterrows(), 1):
        adj = float(p["CRT2_Total_adj"])
        by_hand = sum(adj > c + 1e-9 for c in q)
        ok = by_hand == int(p[LEVEL])
        lines.append(f"person {i} (ID not written): CRT2_Total_adj {adj:.4f}, cutoffs "
                     f"{[round(c, 4) for c in q]} exceeded {by_hand} -> level {by_hand}; frame level "
                     f"{int(p[LEVEL])} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            stop(f"C4 person {i}: level by hand {by_hand}, frame {int(p[LEVEL])}.")
    return lines


# ---------------------------------------------------------------- outputs

def pct_cell(r: pd.Series) -> str:
    return f"{100 * r['prevalence']:.1f}% [{100 * r['wilson_lo']:.1f}, {100 * r['wilson_hi']:.1f}] ({int(r['n'])})"


def headline(run_csv: Path, human_csv: Path) -> str:
    """One plain table per behaviour, built from the written CSVs."""
    runs = pd.read_csv(run_csv, float_precision="round_trip")
    human = pd.read_csv(human_csv, float_precision="round_trip")
    both = pd.concat([human, runs], ignore_index=True)
    names = {"human": "People (own level)", **{r: f"{SPEC.RUN_DISPLAY[r]}" for r in RUNS}}
    out = ["# CR-30 — prevalence of the five reliable BRT behaviours at each reflection level", "",
           "Each cell: prevalence % [Wilson 95% CI] (n). Runs by the level each persona was told (crt2_level); "
           "people by their own level (familiarity-adjusted CRT2 at the frozen cutoffs; the 471 CRT2-complete "
           "people). Method: statlib.wilson_ci. Source: `level_prevalence_runs_cr30.csv`, "
           "`level_prevalence_human_cr30.csv`.", "",
           "Run key: " + "; ".join(f"{SPEC.RUN_DISPLAY[r]} = `{r}`" for r in RUNS) + ".", ""]
    for o, g in both.groupby("outcome", sort=False):
        out += [f"## {g['paper_facing_name'].iloc[0]} (`{o}`)", "",
                "| | " + " | ".join(f"Level {lv}" for lv in CR30_LEVELS) + " |",
                "|---|" + "---|" * len(CR30_LEVELS)]
        for label in ("human", *RUNS):
            cells = g[g["run_label"] == label].set_index("level")
            out.append(f"| {names[label]} | " + " | ".join(pct_cell(cells.loc[lv]) for lv in CR30_LEVELS) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR)
    ap.add_argument("--inbound-only", action="store_true", help="run C1-C4 and write nothing")
    a = ap.parse_args()

    beh = behaviours(a.data_root)
    outcomes = list(beh["outcome"])
    fam = family_tables(a.bundle_dir, outcomes)
    names = fam.reset_index().groupby("outcome")["paper_facing_name"].unique()
    if any(list(names[o]) != [n] for o, n in zip(beh["outcome"], beh["paper_facing_name"])):
        stop("the CR-20 table and the family tables name a behaviour differently.")
    print(f"C1 behaviours (q < .05 within family): {outcomes}")
    print(f"C1 family tables: {len(fam)} rows, one per run x behaviour")

    frames = {run: agent_frame(a.data_root, run) for run in RUNS}
    for line in check_frames(frames, outcomes):
        print(f"C1/C2 {line}")
    runs = run_table(frames, beh)
    print(f"C2 per-level sums equal the family table for {check_sums(runs, fam)} run x behaviour totals")

    joined, full, counts = human_joined(a.data_root, outcomes)
    human = human_table(joined, beh)
    print(f"C2 human join: {counts}")
    for line in check_human(human, full, counts, fam):
        print(f"C2 human {line}")
    print(f"C3 wilson_ci reproduces run_prev_lo/hi_pct for {len(fam)} totals; max |diff| {check_wilson(fam):.2e}")
    for line in spot_read(frames, a.data_root):
        print(f"C4 {line}")
    if a.inbound_only:
        return

    for _, e in edge_cells(pd.concat([runs, human])).iterrows():
        print(f"edge {e['run_label']} {e['outcome']} level {e['level']}: {e['events']}/{e['n']}; wilson_ci "
              f"({e['raw_lo']!r}, {e['raw_hi']!r}) -> written ({e['wilson_lo']!r}, {e['wilson_hi']!r})")
    if len(runs) != 175 or len(human) != 25:
        stop(f"{len(runs)} run rows and {len(human)} human rows, expected 175 and 25.")
    a.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(a.out / RUN_CSV, index=False)
    human.to_csv(a.out / HUMAN_CSV, index=False)
    (a.out / HEADLINE_MD).write_text(headline(a.out / RUN_CSV, a.out / HUMAN_CSV))
    print(f"wrote {a.out / RUN_CSV} ({len(runs)} rows), {a.out / HUMAN_CSV} ({len(human)} rows), "
          f"{a.out / HEADLINE_MD}")


if __name__ == "__main__":
    main()
