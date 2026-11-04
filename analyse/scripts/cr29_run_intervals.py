"""CR-29: persona-clustered lure intervals for the seven simulation runs and the human raw lines of main
Figure 2 panel A (D-237: one method throughout).

The 14 configuration rows of panel A already carry the persona-clustered bootstrap interval; the seven run rows
and the human lines carried Wilson intervals on item responses, which treat one persona's four answers as
independent. This script gives them the same interval as every other: socsim.rebuttal_analyses.
cluster_bootstrap_proportion_ci (persona-clustered percentile bootstrap, 5000 resamples, seed 42), called on the
three-column frame (entity_id, instrument, response_category) of scored rows, exactly as the builder's
bootstrap_lure_ci builds it (scope "pooled" -> instrument None).

Inputs: the seven CR-13 runs' 05_crt_response_categories_long.csv (socsim_agent rows only; the file's human rows
are the pipeline's v1-label rows, which the builder replaces) and the builder's cr25_human_frame (the 471
CRT2-complete people, v2 labels). Check values: the bundle of record's table_intuitive_lure_prevalence_by_run_task.csv
(counts must match exactly; its Wilson intervals are carried beside the new ones). Method guard: the builder's
bootstrap_lure_ci on CRFSD_KIMI_K3_NUM crt2 reproduces CR-28's interval to 1e-12.

Statistics only; no model call. Participant IDs and free text are read, never written: outputs carry only
cell-level counts and intervals.

Usage:
    python scripts/cr29_run_intervals.py [--data-root DIR] [--out DIR] [--bundle CSV] [--cr28 CSV]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from socsim.rebuttal_analyses import cluster_bootstrap_proportion_ci  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_assets_cr29", REPO / "scripts" / "build_neurips_camera_ready_assets.py")
B = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = B
_spec.loader.exec_module(B)

# Ruled values (CR-29 prompt).
CR29_SCOPES = ("crt2", "bcrt", "pooled")
CR29_AGENT_ID_TYPE = str
CR29_HUMAN_ID_TYPE = str                # id type changes only which persona each draw names, not the method
CR29_N_BOOT = 5000
CR29_SEED = 42

DATA_ROOT = Path.home() / "Desktop" / "phd" / "CR" / "soc-sim"     # git-ignored data shared by every checkout
OUT_DIR = DATA_ROOT / "reports" / "cr29_run_ci"
# The bundle of record (hub note, "Where things live", Figures/Tables (of record)).
BUNDLE_TABLE = B.VAULT_CR / "cr-25d" / "neurips_paper_assets_v2" / "tables" / "table_intuitive_lure_prevalence_by_run_task.csv"
CR28_CSV = DATA_ROOT / "reports" / "cr28_ladder_ci" / "ladder_ci_cr28.csv"

RUNS = B.TARGET_LABELS
GUARD_CELL, GUARD_SCOPE, GUARD_ID_TYPE = "CRFSD_KIMI_K3_NUM", "crt2", int    # CR-28's cell, its id type
UNSCORED = "missing_or_unparseable"
LURE = "intuitive_incorrect"
TEST_NAME = {"crt2": "CRT2", "bcrt": "bCRT"}                                # bundle table's test labels
RUN_METHOD = f"persona-clustered bootstrap, {CR29_N_BOOT} resamples (CR-29); scored denominator"
HUMAN_METHOD = f"person-clustered bootstrap, {CR29_N_BOOT} resamples (CR-29); scored denominator"

RUN_CSV = "run_ci_cr29.csv"
HUMAN_CSV = "human_pooled_ci_cr29.csv"
HEADLINE_MD = "HEADLINE_RESULTS.md"


def stop(msg: str) -> None:
    raise SystemExit(f"MANUAL_ACTION_NEEDED: {msg}")


def read_bundle(path: Path = BUNDLE_TABLE) -> pd.DataFrame:
    t = pd.read_csv(path, float_precision="round_trip")      # carry the Wilson bounds digit for digit
    want = {(r, T) for r in ("human", *RUNS) for T in TEST_NAME.values()}
    have = set(zip(t["run_label"], t["test"]))
    if not want <= have or len(t) != len(want):
        stop(f"{path.name}: expected the human row and the seven runs for CRT2 and bCRT ({len(want)} rows), got {len(t)}.")
    return t.set_index(["run_label", "test"])


# ---------------------------------------------------------------- three-column frames

def three_col(entity_id: pd.Series, item_id: pd.Series, category: pd.Series, id_type: type) -> pd.DataFrame:
    """bootstrap_lure_ci's frame: entity_id cast to id_type, instrument from the item id, the category."""
    return pd.DataFrame({"entity_id": entity_id.astype(id_type).to_numpy(),
                         "instrument": np.where(item_id.str.startswith("crt2"), "crt2", "bcrt"),
                         "response_category": category.to_numpy()})


def agent_long(data_root: Path, run: str) -> pd.DataFrame:
    """A run's scored socsim_agent rows (CRT2 and bCRT items) as the three-column frame."""
    path = data_root / "reports" / f"cr13_{run}" / "tables" / "comparisons" / "05_crt_response_categories_long.csv"
    d = pd.read_csv(path, dtype={"entity_id": str, "item_id": str})
    d = d[(d["dataset"] == "socsim_agent") & d["item_id"].isin(B.CRT2_ITEMS + B.BCRT_ITEMS)]
    if not (np.where(d["item_id"].str.startswith("crt2"), "crt2", "bcrt") == d["instrument"].to_numpy()).all():
        stop(f"{run}: the instrument column disagrees with the item ids.")
    d = d[d["response_category"] != UNSCORED]
    return three_col(d["entity_id"], d["item_id"], d["response_category"], CR29_AGENT_ID_TYPE)


def human_long(data_root: Path) -> pd.DataFrame:
    """The 471 CRT2-complete people's scored rows (cr25_human_frame, v2 labels) as the three-column frame."""
    long, _ = B.cr25_human_frame(data_root)
    s = long[long["item_id"].isin(B.CRT2_ITEMS + B.BCRT_ITEMS) & (long["category"] != B.UNSCORED)]
    return three_col(s["entity_id"], s["item_id"], s["category"], CR29_HUMAN_ID_TYPE)


# ---------------------------------------------------------------- the method

def interval(long: pd.DataFrame, scope: str) -> dict:
    rep = cluster_bootstrap_proportion_ci(long, None if scope == "pooled" else scope, category=LURE,
                                          n_boot=CR29_N_BOOT, seed=CR29_SEED)
    return {"n_events": rep["n_events"], "n_items": rep["n_items"], "n_persons": rep["n_personas"],
            "rate": rep["n_events"] / rep["n_items"], "ci95_low": rep["ci95_low"], "ci95_high": rep["ci95_high"]}


def method_guard(data_root: Path, cr28_csv: Path = CR28_CSV) -> dict:
    """C3: the builder's bootstrap_lure_ci and my own three-column call both reproduce CR-28's interval."""
    df = B.read_rescore_long(data_root, GUARD_CELL)
    builder = B.bootstrap_lure_ci(df, GUARD_SCOPE, GUARD_ID_TYPE, scored_only=True)
    cats = B.cr25_long_categories(df)
    m = df["item_id"].isin(B.CRT2_ITEMS + B.BCRT_ITEMS) & (cats != B.UNSCORED)
    own = interval(three_col(df.loc[m, "entity_id"], df.loc[m, "item_id"], cats[m], GUARD_ID_TYPE), GUARD_SCOPE)
    c = pd.read_csv(cr28_csv)
    c = c[(c["source"] == GUARD_CELL) & (c["scope"] == GUARD_SCOPE)]
    if len(c) != 1:
        stop(f"{cr28_csv.name}: expected one {GUARD_CELL} {GUARD_SCOPE} row, got {len(c)}.")
    stored = (float(c["ci95_low"].iloc[0]), float(c["ci95_high"].iloc[0]))
    diff_b = max(abs(builder["ci95_low"] - stored[0]), abs(builder["ci95_high"] - stored[1]))
    diff_o = max(abs(own["ci95_low"] - stored[0]), abs(own["ci95_high"] - stored[1]))
    return {"stored": stored, "builder": (builder["ci95_low"], builder["ci95_high"]),
            "own": (own["ci95_low"], own["ci95_high"]), "diff_builder": diff_b, "diff_own": diff_o,
            "ok": diff_b <= B.CI_REPRO_ATOL and diff_o <= B.CI_REPRO_ATOL
                  and (builder["ci95_low"], builder["ci95_high"]) == (own["ci95_low"], own["ci95_high"])}


# ---------------------------------------------------------------- rows

def check_counts(label: str, scope: str, r: dict, bundle: pd.DataFrame) -> tuple[float, float]:
    """C2: events / scored items equal the bundle's counts (pooled = the sum of the two tests). Returns the bundle's
    Wilson interval for crt2 / bcrt, (nan, nan) for pooled."""
    if scope == "pooled":
        k = sum(int(bundle.at[(label, T), "intuitive_count"]) for T in TEST_NAME.values())
        n = sum(int(bundle.at[(label, T), "n_responses"]) for T in TEST_NAME.values())
        wl = wh = np.nan
    else:
        T = TEST_NAME[scope]
        k, n = int(bundle.at[(label, T), "intuitive_count"]), int(bundle.at[(label, T), "n_responses"])
        wl, wh = float(bundle.at[(label, T), "wilson_ci_low"]), float(bundle.at[(label, T), "wilson_ci_high"])
    if (r["n_events"], r["n_items"]) != (k, n):
        stop(f"{label} {scope}: scored {r['n_events']}/{r['n_items']} != bundle {k}/{n}.")
    return wl, wh


def run_rows(data_root: Path, bundle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        long = agent_long(data_root, run)
        for scope in CR29_SCOPES:
            r = interval(long, scope)
            wl, wh = check_counts(run, scope, r, bundle)
            rows.append({"run_label": run, "scope": scope, **r, "method": RUN_METHOD,
                         "id_type": CR29_AGENT_ID_TYPE.__name__, "seed": CR29_SEED, "wilson_low": wl, "wilson_high": wh})
    return pd.DataFrame(rows)


def human_rows(data_root: Path, bundle: pd.DataFrame) -> pd.DataFrame:
    long = human_long(data_root)
    rows = []
    for scope in CR29_SCOPES:
        r = interval(long, scope)
        wl, wh = check_counts("human", scope, r, bundle)
        rows.append({"test": scope, "reference": "raw", "n_events": r["n_events"], "n_items": r["n_items"],
                     "n_people": r["n_persons"], "rate": r["rate"], "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"],
                     "method": HUMAN_METHOD, "wilson_low": wl, "wilson_high": wh})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- outputs

def pct_ci(rate: float, lo: float, hi: float) -> str:
    return f"{100 * rate:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def ratio(r: pd.Series) -> str:
    return f"{(r['ci95_high'] - r['ci95_low']) / (r['wilson_high'] - r['wilson_low']):.2f}"


def table(cells: dict[str, pd.DataFrame], first: str) -> list[str]:
    """One row per entry of `cells` (a frame indexed by scope)."""
    out = [f"| {first} | CRT2 % [95% CI] | CRT2 Wilson [95% CI] | CRT2 width ratio | bCRT % [95% CI] "
           "| bCRT Wilson [95% CI] | bCRT width ratio | pooled % [95% CI] |",
           "|---|---|---|---:|---|---|---:|---|"]
    for name, c in cells.items():
        cr, bc, po = c.loc["crt2"], c.loc["bcrt"], c.loc["pooled"]
        out.append(f"| {name} | {pct_ci(cr['rate'], cr['ci95_low'], cr['ci95_high'])} | {ci(cr['wilson_low'], cr['wilson_high'])} "
                   f"| {ratio(cr)} | {pct_ci(bc['rate'], bc['ci95_low'], bc['ci95_high'])} | {ci(bc['wilson_low'], bc['wilson_high'])} "
                   f"| {ratio(bc)} | {pct_ci(po['rate'], po['ci95_low'], po['ci95_high'])} |")
    return out


def headline(run_csv: Path, human_csv: Path) -> str:
    runs = pd.read_csv(run_csv)
    hum = pd.read_csv(human_csv)
    lines = ["# CR-29 headline results", "",
             f"Intuitive-lure rate, % [95% CI]: {RUN_METHOD} (seed {CR29_SEED}, persona ids as strings). Wilson = the "
             "interval of the bundle of record's table_intuitive_lure_prevalence_by_run_task.csv. Width ratio = "
             "clustered width / Wilson width.", ""]
    lines += table({run: runs[runs["run_label"] == run].set_index("scope") for run in RUNS}, "Run")
    lines += ["", f"Human reference, raw (the 471 CRT2-complete people): {HUMAN_METHOD} (seed {CR29_SEED}, ids as strings).", ""]
    lines += table({"Human, raw": hum.set_index("test")}, "Reference")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--bundle", type=Path, default=BUNDLE_TABLE)
    ap.add_argument("--cr28", type=Path, default=CR28_CSV)
    a = ap.parse_args()

    g = method_guard(a.data_root, a.cr28)
    print(f"C3 guard {GUARD_CELL} {GUARD_SCOPE}: CR-28 {g['stored']} builder {g['builder']} own {g['own']} "
          f"|diff| {max(g['diff_builder'], g['diff_own']):.2e} -> {'PASS' if g['ok'] else 'FAIL'}")
    if not g["ok"]:
        stop(f"C3 guard: CR-28 {g['stored']} vs builder {g['builder']} / own {g['own']}.")
    bundle = read_bundle(a.bundle)
    runs = run_rows(a.data_root, bundle)
    hum = human_rows(a.data_root, bundle)
    a.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(a.out / RUN_CSV, index=False)
    hum.to_csv(a.out / HUMAN_CSV, index=False)
    (a.out / HEADLINE_MD).write_text(headline(a.out / RUN_CSV, a.out / HUMAN_CSV), encoding="utf-8")
    print(f"wrote {a.out / RUN_CSV} ({len(runs)} rows), {a.out / HUMAN_CSV} ({len(hum)} rows), {a.out / HEADLINE_MD}")


if __name__ == "__main__":
    main()
