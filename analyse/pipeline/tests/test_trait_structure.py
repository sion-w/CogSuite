"""Trait-structure transfer: the verdict rule on hand-made cases (every branch, and one case per adjacent pair of
clauses showing that the order decides), the interval code on cases with a known answer, the human resamples
shared across runs, and the written outputs (when COGSUITE_INPUT_ROOT holds them).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trait_structure as M  # noqa: E402

V = M.verdict


def test_not_estimable():
    assert V(True, 0.36, (0.28, 0.44), np.nan, (np.nan, np.nan), (np.nan, np.nan)) == "not estimable"


def test_introduced():
    assert V(False, 0.08, (-0.01, 0.17), 0.25, (0.16, 0.34), (0.04, 0.29)) == "introduced"


def test_no_human_target_run_ci_includes_zero():
    assert V(False, 0.08, (-0.01, 0.17), 0.04, (-0.05, 0.13), (-0.17, 0.09)) == "no human target"


def test_no_human_target_run_excludes_but_delta_includes_zero():
    assert V(False, 0.08, (-0.01, 0.17), 0.13, (0.04, 0.21), (-0.08, 0.17)) == "no human target"


def test_reversed_positive_human():
    assert V(False, 0.20, (0.11, 0.29), -0.15, (-0.24, -0.06), (-0.48, -0.22)) == "reversed"


def test_reversed_negative_human():
    assert V(False, -0.20, (-0.29, -0.11), 0.15, (0.06, 0.24), (0.22, 0.48)) == "reversed"


def test_exaggerated_positive():
    assert V(False, 0.36, (0.28, 0.44), 0.88, (0.85, 0.90), (0.43, 0.60)) == "exaggerated"


def test_exaggerated_negative():
    assert V(False, -0.32, (-0.41, -0.23), -0.44, (-0.51, -0.36), (-0.23, -0.01)) == "exaggerated"


def test_weakened_same_sign_smaller():
    assert V(False, 0.40, (0.32, 0.48), 0.20, (0.11, 0.29), (-0.32, -0.08)) == "weakened"


def test_weakened_run_ci_includes_zero():
    assert V(False, 0.18, (0.09, 0.26), 0.00, (-0.09, 0.09), (-0.30, -0.05)) == "weakened"


def test_weakened_opposite_sign_run_ci_includes_zero():
    assert V(False, 0.18, (0.09, 0.26), -0.03, (-0.12, 0.06), (-0.34, -0.08)) == "weakened"


def test_preserved_positive():
    assert V(False, 0.40, (0.32, 0.48), 0.46, (0.39, 0.53), (-0.05, 0.17)) == "preserved"


def test_preserved_negative():
    assert V(False, -0.22, (-0.31, -0.13), -0.26, (-0.35, -0.17), (-0.16, 0.08)) == "preserved"


def test_indeterminate():
    assert V(False, 0.09, (0.01, 0.18), 0.04, (-0.05, 0.14), (-0.18, 0.08)) == "indeterminate"


def test_order_1_before_2():
    assert V(True, 0.08, (-0.01, 0.17), np.nan, (np.nan, np.nan), (np.nan, np.nan)) == "not estimable"


def test_order_2_before_3():
    # human interval includes 0 while the run interval excludes 0 on the side opposite to rho_human
    assert V(False, 0.03, (-0.06, 0.12), -0.20, (-0.29, -0.11), (-0.36, -0.10)) == "introduced"


def test_order_3_before_4():
    # reversed and the delta interval excludes 0 with |rho_run| > |rho_human|
    assert V(False, 0.10, (0.02, 0.19), -0.30, (-0.38, -0.21), (-0.52, -0.28)) == "reversed"


def test_order_4_before_5():
    assert V(False, 0.40, (0.32, 0.48), 0.51, (0.44, 0.58), (0.004, 0.21)) == "exaggerated"
    assert V(False, 0.40, (0.32, 0.48), 0.20, (0.11, 0.29), (-0.32, -0.08)) == "weakened"


def test_order_5_before_6():
    assert V(False, 0.18, (0.09, 0.26), 0.20, (0.11, 0.29), (-0.10, 0.15)) == "preserved"


def test_ci_touching_zero_counts_as_including_zero():
    assert M.excludes_zero(0.0, 0.2) is False
    assert M.excludes_zero(-0.2, 0.0) is False
    assert M.excludes_zero(1e-9, 0.2) is True


def test_verdict_list_is_the_rule_order():
    assert M.VERDICTS == ["not estimable", "introduced", "no human target", "reversed",
                          "exaggerated", "weakened", "preserved", "indeterminate"]


def test_fisher_bonett_wright_known_value():
    lo, hi = M.fisher_bw_ci(0.3, 103)
    assert lo == pytest.approx(0.10873026771666391, abs=1e-9)
    assert hi == pytest.approx(0.46984966622570656, abs=1e-9)


def test_zou_symmetric_known_value():
    lo, hi = M.zou_diff_ci(0.0, (-0.1, 0.1), 0.0, (-0.1, 0.1))
    assert (lo, hi) == pytest.approx((-0.1 * math.sqrt(2), 0.1 * math.sqrt(2)))


def test_zou_reduces_to_the_two_interval_arms():
    lo, hi = M.zou_diff_ci(0.5, (0.4, 0.6), 0.2, (0.1, 0.3))
    assert (lo, hi) == pytest.approx((0.3 - math.sqrt(0.02), 0.3 + math.sqrt(0.02)))


def test_percentile_ci_known_value_and_dropped_count():
    draws = np.r_[np.arange(101.0), [np.nan] * 7].reshape(-1, 1)
    lo, hi, dropped = M.percentile_ci(draws)
    assert (lo[0], hi[0], dropped[0]) == (2.5, 97.5, 7)


def test_bootstrap_on_monotone_data_gives_one():
    x = np.tile(np.arange(30.0)[:, None], (1, 8))
    idx = M.draw_indices(np.random.default_rng(42), 30, 2000)
    lo, hi, dropped = M.percentile_ci(M.bootstrap_rho(x, idx))
    assert np.allclose(lo, 1) and np.allclose(hi, 1) and (dropped == 0).all()


def test_constant_resample_is_dropped_and_counted():
    x = np.tile(np.array([[0.0], [1.0]]), (1, 8))
    idx = M.draw_indices(np.random.default_rng(42), 2, 1000)
    lo, hi, dropped = M.percentile_ci(M.bootstrap_rho(x, idx))
    assert (dropped == (idx[:, 0] == idx[:, 1]).sum()).all()
    assert np.allclose(lo, 1) and np.allclose(hi, 1)


def test_spearman_matches_scipy_with_ties():
    x = np.random.default_rng(0).integers(0, 5, size=(60, 8)).astype(float)
    mine = M.spearman_pairs(x)
    for p, (i, j) in enumerate(M.PAIRS):
        assert mine[p] == pytest.approx(spearmanr(x[:, i], x[:, j]).statistic, abs=1e-12)


def test_spearman_constant_column_is_nan():
    x = np.random.default_rng(1).normal(size=(40, 8))
    x[:, 0] = 4.0
    rho = M.spearman_pairs(x)
    for p, (i, j) in enumerate(M.PAIRS):
        assert np.isnan(rho[p]) == (i == 0 or j == 0)


def _fake_reports(root: Path, n: int = 40) -> None:
    rng = np.random.default_rng(3)
    human = pd.DataFrame(rng.integers(0, 6, size=(n, 8)).astype(float), columns=M.HUMAN_COLS)
    human.insert(0, "PROLIFIC_PID", [f"p{k}" for k in range(n)])
    human.loc[n] = ["p_incomplete"] + [np.nan] * 8  # one incomplete row, dropped as in the real frames
    for run in M.RUNS:
        agent = pd.DataFrame(rng.integers(0, 6, size=(n, 8)).astype(float), columns=M.AGENT_COLS)
        if "CS46" in run:
            agent[M.AGENT_COLS[:2]] = 4.0
        for side, df, cols in [("human", human, M.HUMAN_COLS), ("socsim_agent", agent, M.AGENT_COLS)]:
            d = M.frame_dir(root, run, side)
            d.mkdir(parents=True)
            df.to_csv(d / "00_person_frame.csv", index=False)
            corr = df[cols].dropna().rank().corr()  # Spearman = Pearson on average ranks; constant -> NaN
            corr.index = corr.columns = M.STORED
            corr.to_csv(d / "01_correlations.csv")


def test_analyse_reuses_the_human_resamples_for_every_run(tmp_path):
    _fake_reports(tmp_path)
    pairs, summary, meta = M.analyse(tmp_path, names={r: r for r in M.RUNS})
    assert len(pairs) == 7 * 28 and meta["n_rho_checked"] > 0
    human_cols = ["rho_human", "rho_human_lo", "rho_human_hi", "human_target", "dropped_resamples_human"]
    first = pairs[pairs.run_id == M.RUNS[0]][human_cols].reset_index(drop=True)
    for run in M.RUNS[1:]:
        pd.testing.assert_frame_equal(pairs[pairs.run_id == run][human_cols].reset_index(drop=True), first)
    human = M.load_scores(tmp_path, M.RUNS[0], "human")
    lo, hi, _ = M.percentile_ci(M.bootstrap_rho(human, M.draw_indices(np.random.default_rng(42), len(human))))
    assert np.allclose(first["rho_human_lo"], lo) and np.allclose(first["rho_human_hi"], hi)
    assert summary.set_index("run_id")["estimable_pairs"].to_dict() == {r: (15 if "CS46" in r else 28) for r in M.RUNS}
    assert (pairs[pairs.verdict == "not estimable"].dropped_resamples_run == M.N_BOOT).all()


def test_names_and_focal_pairs():
    labels = [M.pair_label(i, j) for i, j in M.PAIRS]
    assert len(labels) == 28 and labels[0] == "CRT2–bCRT"
    assert sum((M.CONSTRUCTS[i], M.CONSTRUCTS[j]) in M.FOCAL for i, j in M.PAIRS) == 6


# ---------------------------------------------------------------- the written outputs

OUT = Path(os.environ.get("COGSUITE_INPUT_ROOT", "/nonexistent")) / M.TRAIT_STRUCTURE
needs_out = pytest.mark.skipif(not (OUT / "f4_pairs_by_run.csv").exists(), reason="trait_structure outputs not present")


@needs_out
def test_outputs_shape_and_estimable_counts():
    pairs = pd.read_csv(OUT / "f4_pairs_by_run.csv")
    summary = pd.read_csv(OUT / "f4_run_summary.csv")
    assert len(pairs) == 196 and pairs.groupby("run_id").size().eq(28).all()
    assert summary.set_index("run_id")["estimable_pairs"].to_dict() == {r: (15 if "CS46" in r else 28) for r in M.RUNS}
    assert set(pairs["verdict"]) <= set(M.VERDICTS) and set(pairs["verdict_analytic"]) <= set(M.VERDICTS)


@needs_out
def test_outputs_human_columns_identical_across_runs():
    pairs = pd.read_csv(OUT / "f4_pairs_by_run.csv")
    cols = ["pair", "rho_human", "rho_human_lo", "rho_human_hi", "human_target"]
    first = pairs[pairs.run_id == M.RUNS[0]][cols].reset_index(drop=True)
    for run in M.RUNS[1:]:
        pd.testing.assert_frame_equal(pairs[pairs.run_id == run][cols].reset_index(drop=True), first)


@needs_out
def test_outputs_verdicts_follow_the_rule_from_the_written_numbers():
    pairs = pd.read_csv(OUT / "f4_pairs_by_run.csv")
    for _, r in pairs.iterrows():
        const = bool(np.isnan(r["rho_run"]))
        v = V(const, r["rho_human"], (r["rho_human_lo"], r["rho_human_hi"]), r["rho_run"],
              (r["rho_run_lo"], r["rho_run_hi"]), (r["delta_rho_lo"], r["delta_rho_hi"]))
        assert v == r["verdict"], (r["run_id"], r["pair"])
