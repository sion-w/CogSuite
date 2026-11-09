"""BRT models per run: prevalence fidelity, steering fidelity and association transfer for every primary BRT
outcome, against the human frame.

F1  prevalence: human and run prevalence with Wilson 95% intervals; the gap in pp with a 2,000-resample
    bootstrap interval; "compressed" = |gap| >= 20 pp and non-overlapping intervals.
F2  steering: logistic regression of the outcome on the CRT2 level written into the persona (crt2_level).
F3  transfer: logistic regression on the run's own bCRT score (BCRT_Total_realised), compared with the human
    slope on BCRT_Total (independent-samples difference interval) and given a verdict. A run whose bCRT
    score is constant has no slope: its rows are "not estimable" and stay out of the transfer tally.
    Outcomes with < 20 events or < 20 non-events use Firth's logistic regression and are flagged unstable.
MDOR: the minimum detectable OR of each estimable F3 model (80% power, alpha .05 two-sided, 1,000
    simulated datasets at the observed N, prevalence and predictor distribution).

Reads the human frame (frames.py) and the seven agent person frames. Writes, under
``reports/brt_models/tables/``: s11_part2_primary_outcomes_by_run.csv (with the MDOR columns),
s11_family_H1..H4_table.csv and s11_verdict_grid.csv. Every table with a run column carries the DeepSeek
V4-Pro BRT N note. The figures are drawn by brt_figures.py.

Usage: python analyse/pipeline/brt_models.py [--root DIR]
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import spec
import brt_stats as S
from common import BRT_MODELS, RUN_DISPLAY, RUN_LABELS, RUN_SHORT, parser, require_columns, resolve_root, stop, with_note
from frames import agent_frame, brt_n, human_frame

NOT_ESTIMABLE = "not estimable (no realised bCRT variance)"


def _fit(df: pd.DataFrame, outcome: str, predictor: str) -> dict:
    y = pd.to_numeric(df[outcome], errors="coerce")
    X = df[[predictor]].apply(pd.to_numeric, errors="coerce")
    return S.fit_binary(y, X, predictor)


def run_rows(label: str, df: pd.DataFrame, hum: pd.DataFrame, hmods: dict[str, dict]) -> list[dict]:
    n_brt = brt_n(df)
    f3_sd = float(pd.to_numeric(df[spec.F3_PREDICTOR], errors="coerce").std())
    f3_estimable = bool(np.isfinite(f3_sd) and f3_sd > 0)
    rows = []
    for col in spec.PRIMARY_OUTCOMES:
        ry = pd.to_numeric(df[col], errors="coerce").dropna()
        hy = pd.to_numeric(hum[col], errors="coerce").dropna()
        rk, rn = int(ry.sum()), int(len(ry))
        hk, hn = int(hy.sum()), int(len(hy))
        r_prev, h_prev = (rk / rn if rn else np.nan), (hk / hn if hn else np.nan)
        r_lo, r_hi = S.wilson_ci(rk, rn)
        h_lo, h_hi = S.wilson_ci(hk, hn)
        gap, g_lo, g_hi = S.prevalence_gap_bootstrap(ry.to_numpy(float), hy.to_numpy(float),
                                                     n_boot=spec.BOOTSTRAP_N, seed=spec.BOOTSTRAP_SEED)
        unstable = S.unstable_flag(rk, rn, spec.UNSTABLE_EVENT_FLOOR)
        compressed = S.compressed_flag(gap, h_lo, h_hi, r_lo, r_hi, spec.COMPRESSED_GAP_PP)

        f2 = _fit(df, col, spec.F2_PREDICTOR)
        f3 = _fit(df, col, spec.F3_PREDICTOR)
        hm = hmods[col]
        if hm["n"] != hn:
            stop(f"human {col}: model N {hm['n']} differs from the prevalence N {hn}")
        tv = S.transfer_verdict(f3["beta"], f3["se"], hm["beta"], hm["se"], unstable)
        if not f3_estimable:
            # a constant bCRT score has no slope; the Firth output there is not an estimate
            tv = {"verdict": NOT_ESTIMABLE, "diff": float("nan"), "diff_lo": float("nan"), "diff_hi": float("nan")}

        rows.append({
            "run_label": label, "run_display": RUN_DISPLAY[label],
            "family": spec.OUTCOME_FAMILY[col], "outcome": col,
            "paper_facing_name": spec.PAPER_FACING.get(col, ""), "prior": spec.PRIOR.get(col, ""),
            "run_brt_n": n_brt,
            # F1
            "human_events": hk, "human_n": hn, "human_prev_pct": 100 * h_prev,
            "human_prev_lo_pct": 100 * h_lo, "human_prev_hi_pct": 100 * h_hi,
            "run_events": rk, "run_n": rn, "run_prev_pct": 100 * r_prev,
            "run_prev_lo_pct": 100 * r_lo, "run_prev_hi_pct": 100 * r_hi,
            "f1_gap_pp": gap, "f1_gap_lo_pp": g_lo, "f1_gap_hi_pp": g_hi, "f1_compressed": compressed,
            # F2: the CRT2 level written into the persona
            "f2_estimator": f2["estimator"], "f2_n": f2["n"], "f2_events": f2["events"],
            "f2_slope_beta": f2["beta"], "f2_or": f2["or"], "f2_or_lo": f2["or_lo"], "f2_or_hi": f2["or_hi"],
            "f2_p_raw": f2["p"], "f2_ci_type": f2["ci_type"], "f2_flags": f2["flags"],
            # F3: the run's own bCRT score against the human slope
            "f3_estimator": f3["estimator"], "f3_n": f3["n"], "f3_events": f3["events"],
            "f3_beta_run": f3["beta"], "f3_se_run": f3["se"], "f3_or_run": f3["or"],
            "f3_or_run_lo": f3["or_lo"], "f3_or_run_hi": f3["or_hi"],
            "f3_p_raw": f3["p"], "f3_ci_type": f3["ci_type"], "f3_flags": f3["flags"],
            "human_estimator": hm["estimator"],
            "human_beta": hm["beta"], "human_se": hm["se"], "human_or": hm["or"],
            "human_or_lo": hm["or_lo"], "human_or_hi": hm["or_hi"], "human_p_raw": hm["p"],
            "f3_beta_diff": tv["diff"], "f3_diff_lo": tv["diff_lo"], "f3_diff_hi": tv["diff_hi"],
            "f3_verdict": tv["verdict"],
            "f3_predictor_sd": f3_sd, "f3_estimable": f3_estimable,
            "c3_unstable": unstable,
        })
    return rows


def mdor_rows(label: str, df: pd.DataFrame) -> list[dict]:
    """MDOR above and below 1 for each outcome's F3 model; NaN where F3 is not estimable or too few events."""
    x_all = pd.to_numeric(df[spec.F3_PREDICTOR], errors="coerce")
    estimable = bool(np.isfinite(x_all.std()) and x_all.std() > 0)
    rows = []
    for col in spec.PRIMARY_OUTCOMES:
        sub = pd.concat([pd.to_numeric(df[col], errors="coerce").rename("y"), x_all.rename("x")], axis=1).dropna()
        n, k = len(sub), int(sub["y"].sum())
        rec = {"run_label": label, "outcome": col, "mdor_above_1": np.nan, "mdor_below_1": np.nan}
        if estimable and n and 2 <= k <= n - 2:
            x = sub["x"].to_numpy(float)
            base = k / n
            rec["mdor_above_1"] = S.mdor_binary(x, base, nsim=spec.MDOR_NSIM, seed=spec.MDOR_SEED_ABOVE,
                                                direction="above")
            rec["mdor_below_1"] = S.mdor_binary(x, base, nsim=spec.MDOR_NSIM, seed=spec.MDOR_SEED_BELOW,
                                                direction="below")
            print(f"  MDOR {RUN_SHORT[label]:14s} {col:36s} {rec['mdor_above_1']:.3f} / {rec['mdor_below_1']:.3f}",
                  flush=True)
        rows.append(rec)
    return rows


def main(argv: list[str] | None = None) -> int:
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    hum = human_frame(root)
    require_columns(hum, spec.PRIMARY_OUTCOMES + [spec.HUMAN_PREDICTOR], "human frame")
    frames = {}
    for label in RUN_LABELS:
        frames[label] = agent_frame(root, label)
        require_columns(frames[label], spec.PRIMARY_OUTCOMES + [spec.F2_PREDICTOR, spec.F3_PREDICTOR],
                        f"agent frame {label}")

    warnings.filterwarnings("ignore")
    hmods = {c: _fit(hum, c, spec.HUMAN_PREDICTOR) for c in spec.PRIMARY_OUTCOMES}
    part2 = pd.DataFrame([r for label in RUN_LABELS for r in run_rows(label, frames[label], hum, hmods)])
    mdor = pd.DataFrame([r for label in RUN_LABELS for r in mdor_rows(label, frames[label])])

    tab = root / BRT_MODELS / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    with_note(part2.merge(mdor, on=["run_label", "outcome"], how="left")).to_csv(
        tab / "s11_part2_primary_outcomes_by_run.csv", index=False)
    for fam in spec.FAMILIES:
        k = fam["key"]
        with_note(part2[part2["family"] == k]).to_csv(tab / f"s11_family_{k}_table.csv", index=False)
    grid = part2.pivot_table(index=["family", "outcome"], columns="run_label", values="f3_verdict", aggfunc="first")
    grid.reindex(columns=RUN_LABELS).to_csv(tab / "s11_verdict_grid.csv")

    tally = part2.groupby("run_label", sort=False)["f3_verdict"].value_counts().unstack(fill_value=0)
    print(tally.reindex(RUN_LABELS).to_string())
    print(f"wrote {tab}: s11_part2_primary_outcomes_by_run.csv ({len(part2)} rows), s11_family_H1..H4_table.csv, "
          "s11_verdict_grid.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
