"""Lure-rate intervals (run_intervals.py, ladder_intervals.py) and level prevalence cells (level_prevalence.py):
known cases without data, and the written outputs when COGSUITE_INPUT_ROOT holds them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import brt_stats  # noqa: E402
import level_prevalence as LP  # noqa: E402
import run_intervals as RI  # noqa: E402
from common import LADDER_INTERVALS, LEVEL_PREVALENCE, RUN_INTERVALS, builder  # noqa: E402

ROOT = Path(os.environ.get("COGSUITE_INPUT_ROOT", "/nonexistent"))


# ---------------------------------------------------------------- run intervals

def test_constants():
    assert RI.SCOPES == ("crt2", "bcrt", "pooled")
    assert RI.AGENT_ID_TYPE is str and RI.HUMAN_ID_TYPE is str
    assert (RI.N_BOOT, RI.SEED) == (5000, 42)


def test_zero_events_give_zero_interval():
    long = pd.DataFrame({"entity_id": np.repeat([str(i) for i in range(50)], 8),
                         "instrument": np.tile(["crt2"] * 4 + ["bcrt"] * 4, 50), "response_category": "correct"})
    for scope in RI.SCOPES:
        r = RI.interval(long, scope)
        assert (r["n_events"], r["rate"], r["ci95_low"], r["ci95_high"]) == (0, 0.0, 0.0, 0.0)


def test_same_call_twice_is_identical():
    rng = np.random.default_rng(0)
    long = pd.DataFrame({"entity_id": np.repeat([str(i) for i in range(60)], 8),
                         "instrument": np.tile(["crt2"] * 4 + ["bcrt"] * 4, 60),
                         "response_category": rng.choice(["correct", "intuitive_incorrect"], 480)})
    assert RI.interval(long, "crt2") == RI.interval(long, "crt2")


def test_builder_wilson_known_cases():
    from statsmodels.stats.proportion import proportion_confint
    B = builder()
    lo, hi = B.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2365896, abs=1e-7) and hi == pytest.approx(0.7634104, abs=1e-7)
    for k, n in ((5, 10), (37, 600), (242, 600)):     # z = 1.959964 there, 1.96 here
        assert np.allclose(B.wilson_ci(k, n), proportion_confint(k, n, method="wilson"), atol=1e-4)
    lo, hi = B.wilson_ci(0, 600)
    assert lo == 0.0 and hi == pytest.approx(1.96 ** 2 / (600 + 1.96 ** 2), abs=1e-12)


# ---------------------------------------------------------------- level prevalence cells

def test_level_constants():
    assert LP.LEVELS == (0, 1, 2, 3, 4)
    assert sum(LP.SYNTH_TOLD_MIX) == sum(LP.HUMDERIVED_TOLD_MIX) == 471
    assert sum(LP.HUMAN_LEVEL_MIX) == LP.HUMAN_JOINED == 471
    assert LP.HUMDERIVED == ("CR_BASELINE_HUMAN_NUMERIC_CS46_S42", "CR_BASELINE_HUMAN_LEXICAL_CS46_S42")
    assert LP.wilson_ci is brt_stats.wilson_ci


@pytest.mark.parametrize("n", [1, 7, 67, 94, 103, 105, 471])
def test_zero_events_lower_bound_is_zero(n):
    c = LP.cell(pd.Series([0] * n))
    assert (c["n"], c["events"], c["prevalence"], c["wilson_lo"]) == (n, 0, 0.0, 0.0)
    assert 0.0 < c["wilson_hi"] < 1.0


@pytest.mark.parametrize("n", [1, 67, 103, 471])
def test_all_events_upper_bound_is_one(n):
    c = LP.cell(pd.Series([1] * n))
    assert (c["prevalence"], c["wilson_hi"]) == (1.0, 1.0) and 0.0 < c["wilson_lo"] < 1.0


@pytest.mark.parametrize("k,n", [(1, 7), (19, 109), (416, 467), (458, 467), (54, 471)])
def test_interior_cell_is_wilson_ci_unchanged(k, n):
    c = LP.cell(pd.Series([1] * k + [0] * (n - k) + [float("nan")] * 3))
    assert (c["n"], c["events"]) == (n, k)
    assert (c["wilson_lo"], c["wilson_hi"]) == tuple(float(v) for v in LP.wilson_ci(k, n))
    assert c["wilson_lo"] <= c["prevalence"] <= c["wilson_hi"]


def test_non_binary_outcome_stops():
    with pytest.raises(SystemExit, match="STOP"):
        LP.cell(pd.Series([0, 1, 2]))


def test_other_behaviour_set_stops(tmp_path):
    rows = [{"outcome": o, "paper_facing_name": o, "q_bh_within_family": 0.01} for o in LP.EXPECTED_BEHAVIOURS[:4]]
    p = tmp_path / "reports" / "human_brt_families" / "tables"
    p.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(p / "s5_primary_families.csv", index=False)
    with pytest.raises(SystemExit, match="not the five"):
        LP.behaviours(tmp_path)


# ---------------------------------------------------------------- written outputs

needs_runs = pytest.mark.skipif(not (ROOT / RUN_INTERVALS / "run_ci.csv").exists(), reason="run_intervals outputs absent")
needs_human_item = pytest.mark.skipif(not (ROOT / LADDER_INTERVALS / "human_item_ci.csv").exists(),
                                      reason="ladder_intervals output absent")
needs_levels = pytest.mark.skipif(not (ROOT / LEVEL_PREVALENCE / "level_prevalence_runs.csv").exists(),
                                  reason="level_prevalence outputs absent")


@needs_runs
def test_run_csv_shape_and_intervals():
    runs = pd.read_csv(ROOT / RUN_INTERVALS / "run_ci.csv")
    hum = pd.read_csv(ROOT / RUN_INTERVALS / "human_pooled_ci.csv")
    assert len(runs) == 21 and len(hum) == 3
    for t in (runs, hum):
        assert np.allclose(t["rate"], t["n_events"] / t["n_items"])
        assert ((t["ci95_low"] <= t["rate"]) & (t["rate"] <= t["ci95_high"])).all()
    pooled = runs[runs["scope"] == "pooled"].set_index("run_label")
    parts = runs[runs["scope"] != "pooled"].groupby("run_label")[["n_events", "n_items"]].sum()
    assert (pooled[["n_events", "n_items"]] == parts.loc[pooled.index]).all().all()


@needs_human_item
def test_human_item_csv():
    t = pd.read_csv(ROOT / LADDER_INTERVALS / "human_item_ci.csv")
    assert list(t["item"]) == ["q1", "q2", "q3", "q4"] and set(t["n_people"]) == {471}
    assert ((t["ci95_low"] <= t["rate"]) & (t["rate"] <= t["ci95_high"])).all()


@needs_levels
def test_level_tables():
    runs = pd.read_csv(ROOT / LEVEL_PREVALENCE / "level_prevalence_runs.csv")
    hum = pd.read_csv(ROOT / LEVEL_PREVALENCE / "level_prevalence_human.csv")
    assert list(runs.columns) == LP.COLUMNS and list(hum.columns) == LP.COLUMNS
    assert len(runs) == 175 and len(hum) == 25
    for t in (runs, hum):
        assert np.allclose(t["prevalence"], t["events"] / t["n"])
        assert ((t["wilson_lo"] <= t["prevalence"]) & (t["prevalence"] <= t["wilson_hi"])).all()
    for o, g in hum.groupby("outcome"):
        assert tuple(g.sort_values("level")["n"]) == LP.HUMAN_LEVEL_MIX
