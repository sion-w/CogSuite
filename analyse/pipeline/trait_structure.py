"""Trait-structure transfer: does each run reproduce the human rank correlations among the eight constructs?

Per run and per pair of the eight constructs (28 pairs): Spearman rho in the run and in the human reference,
recomputed from the per-run 00_person_frame.csv files the way 01_correlations.csv computes them, with
percentile bootstrap 95% intervals for rho_run, rho_human and delta = rho_run - rho_human (10,000 resamples,
numpy.random.default_rng(42), person = resampling unit, humans and agents resampled independently; the
human frame is resampled once and the i-th human resample is paired with the i-th agent resample of every
run). A verdict per pair by the ordered rule in ``verdict``, and a per-run tally. An analytic concordance
check (Fisher z with the Bonett-Wright variance, Zou's difference interval) recomputes every verdict.
Where a run folder carries 01_correlations.csv, every recomputed rho must equal the stored value.

Random stream: one Generator, default_rng(42). The human resample indices are drawn first
(10,000 x n_human); then each run's agent resample indices are drawn from the same Generator in RUNS order.
Participant IDs are read, never written: outputs carry only run- and pair-level numbers.

Reads ``reports/cr13_<run>/tables/by_dataset/{human,socsim_agent}/00_person_frame.csv`` (the human frames and
the human-derived runs' agent frames are controlled access). Writes ``reports/trait_structure/
f4_pairs_by_run.csv`` and ``f4_run_summary.csv``.

Usage: python analyse/pipeline/trait_structure.py [--root DIR] [--out DIR]
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from common import RUN_LABELS, TRAIT_STRUCTURE, builder, parser, require, resolve_root, run_dir, stop

RUNS = list(RUN_LABELS)
CONSTRUCTS = ["CRT2", "bCRT", "NFC", "Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
# Column labels of 01_correlations.csv (both sides) and the frame columns that reproduce them.
STORED = ["CRT2_Total", "BCRT_Total", "NFC_Total", "BFI_Extraversion", "BFI_Agreeableness",
          "BFI_Conscientiousness", "BFI_Neuroticism", "BFI_Openness"]
HUMAN_COLS = ["CRT2_Total_adj_observed"] + STORED[1:]
AGENT_COLS = [c + "_realised" for c in STORED]
FOCAL = {("CRT2", "bCRT"), ("CRT2", "NFC"), ("CRT2", "Conscientiousness"),
         ("bCRT", "NFC"), ("bCRT", "Conscientiousness"), ("NFC", "Conscientiousness")}
PAIRS = list(itertools.combinations(range(len(CONSTRUCTS)), 2))
DASH = "–"

N_BOOT = 10_000
SEED = 42
ALPHA = 0.05

VERDICTS = ["not estimable", "introduced", "no human target", "reversed",
            "exaggerated", "weakened", "preserved", "indeterminate"]


def pair_label(i: int, j: int) -> str:
    return f"{CONSTRUCTS[i]}{DASH}{CONSTRUCTS[j]}"


def run_names() -> dict[str, str]:
    """Readable run names, from the builder's CONDITIONS registry (key "name")."""
    return {r: builder().CONDITIONS[r]["name"] for r in RUNS}


# ---------------------------------------------------------------- frames

def frame_dir(root: Path, run: str, side: str) -> Path:
    return run_dir(root, run) / "tables" / "by_dataset" / side


def _frame_path(root: Path, run: str, side: str) -> Path:
    what = ("human person frame (controlled access)" if side == "human" or "HUMAN" in run
            else "agent person frame")
    return require(frame_dir(root, run, side) / "00_person_frame.csv", what)


def load_scores(root: Path, run: str, side: str) -> np.ndarray:
    """Complete-case score matrix (n x 8) in CONSTRUCTS order."""
    cols = HUMAN_COLS if side == "human" else AGENT_COLS
    df = pd.read_csv(_frame_path(root, run, side), usecols=cols, low_memory=False)
    return df[cols].dropna().to_numpy(dtype=float)


def human_cohort_hash(root: Path, run: str) -> str:
    """SHA-256 of the sorted PROLIFIC_PIDs of the complete human rows (the IDs themselves are never returned)."""
    df = pd.read_csv(_frame_path(root, run, "human"), usecols=["PROLIFIC_PID"] + HUMAN_COLS,
                     low_memory=False).dropna(subset=HUMAN_COLS)
    return hashlib.sha256("\n".join(sorted(df["PROLIFIC_PID"].astype(str))).encode()).hexdigest()


def load_stored(root: Path, run: str, side: str) -> pd.DataFrame | None:
    p = frame_dir(root, run, side) / "01_correlations.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


# ---------------------------------------------------------------- Spearman and the bootstrap

def spearman_pairs(x: np.ndarray) -> np.ndarray:
    """Spearman rho (average ranks for ties) for every pair in PAIRS.

    x has shape (..., n, k). Returns shape (..., len(PAIRS)); a pair with a constant score is NaN.
    """
    r = rankdata(x, method="average", axis=-2)
    r = r - r.mean(axis=-2, keepdims=True)
    ss = np.einsum("...ni,...ni->...i", r, r)
    const = np.ptp(x, axis=-2) == 0
    out = np.empty(x.shape[:-2] + (len(PAIRS),))
    for p, (i, j) in enumerate(PAIRS):
        cov = np.einsum("...n,...n->...", r[..., i], r[..., j])
        with np.errstate(invalid="ignore", divide="ignore"):
            v = cov / np.sqrt(ss[..., i] * ss[..., j])
        v = np.where(const[..., i] | const[..., j], np.nan, v)
        out[..., p] = v
    return out


def draw_indices(rng: np.random.Generator, n: int, n_boot: int = N_BOOT) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_rho(x: np.ndarray, idx: np.ndarray, chunk: int = 500) -> np.ndarray:
    """Spearman rho per resample (rows of idx) and pair: shape (n_boot, len(PAIRS)); constant resamples NaN."""
    out = np.empty((idx.shape[0], len(PAIRS)))
    for s in range(0, idx.shape[0], chunk):
        out[s:s + chunk] = spearman_pairs(x[idx[s:s + chunk]])
    return out


def percentile_ci(draws: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile interval per column over the non-NaN draws; returns (lo, hi, n_dropped)."""
    dropped = np.isnan(draws).sum(axis=0)
    lo = np.full(draws.shape[1], np.nan)
    hi = np.full(draws.shape[1], np.nan)
    for c in range(draws.shape[1]):
        v = draws[~np.isnan(draws[:, c]), c]
        if v.size:
            lo[c], hi[c] = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, dropped


# ---------------------------------------------------------------- analytic intervals (concordance check)

def fisher_bw_ci(r: float, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Fisher z interval for Spearman rho with the Bonett-Wright (2000) variance (1 + r^2/2)/(n - 3)."""
    if np.isnan(r):
        return np.nan, np.nan
    z = np.arctanh(r)
    se = np.sqrt((1 + r * r / 2) / (n - 3))
    q = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def zou_diff_ci(r1: float, ci1: tuple[float, float], r2: float, ci2: tuple[float, float]) -> tuple[float, float]:
    """Zou (2007) interval for r1 - r2, two independent correlations with intervals ci1, ci2."""
    (l1, u1), (l2, u2) = ci1, ci2
    d = r1 - r2
    return d - np.sqrt((r1 - l1) ** 2 + (u2 - r2) ** 2), d + np.sqrt((u1 - r1) ** 2 + (r2 - l2) ** 2)


# ---------------------------------------------------------------- the verdict rule

def excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def verdict(run_constant: bool, rho_h: float, h_ci: tuple[float, float],
            rho_r: float, r_ci: tuple[float, float], d_ci: tuple[float, float]) -> str:
    """The clauses are evaluated in order and the first that holds decides."""
    # (1) not estimable: a run score is constant
    if run_constant:
        return "not estimable"
    run_excl, d_excl = excludes_zero(*r_ci), excludes_zero(*d_ci)
    # (2) human interval includes 0: introduced when the run and delta intervals exclude 0, else no human target
    if not excludes_zero(*h_ci):
        return "introduced" if (run_excl and d_excl) else "no human target"
    h_sign = np.sign(rho_h)
    # (3) reversed: the run interval excludes 0 with the sign opposite to rho_human
    if run_excl and np.sign(r_ci[0]) == -h_sign:
        return "reversed"
    # (4) delta interval excludes 0: exaggerated when |rho_run| > |rho_human| with the same sign, else weakened
    if d_excl:
        if np.sign(rho_r) == h_sign and abs(rho_r) > abs(rho_h):
            return "exaggerated"
        return "weakened"
    # (5) preserved: the run interval excludes 0 in the human direction
    if run_excl and np.sign(r_ci[0]) == h_sign:
        return "preserved"
    # (6) indeterminate otherwise
    return "indeterminate"


# ---------------------------------------------------------------- the analysis

def check_against_stored(rho: np.ndarray, stored: pd.DataFrame | None, where: str) -> int:
    """Every recomputed rho equals the stored 01_correlations.csv value (blank <-> NaN), when it is present."""
    if stored is None:
        return 0
    n = 0
    for p, (i, j) in enumerate(PAIRS):
        s = stored.loc[STORED[i], STORED[j]]
        if np.isnan(rho[p]) and pd.isna(s):
            continue
        if np.isnan(rho[p]) or pd.isna(s) or abs(rho[p] - s) > 1e-6:
            stop(f"{where} {pair_label(i, j)}: recomputed rho {rho[p]} differs from 01_correlations.csv {s}")
        n += 1
    return n


def analyse(root: Path, names: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    names = names or run_names()
    if len({human_cohort_hash(root, r) for r in RUNS}) != 1:
        stop("the complete human cohort differs across run folders")
    human = load_scores(root, RUNS[0], "human")
    for r in RUNS[1:]:
        if not np.array_equal(load_scores(root, r, "human"), human):
            stop(f"human frame in {r} differs from {RUNS[0]}")
    n_h = human.shape[0]

    rng = np.random.default_rng(SEED)
    h_idx = draw_indices(rng, n_h)
    rho_h = spearman_pairs(human)
    n_checked = check_against_stored(rho_h, load_stored(root, RUNS[0], "human"), "human")
    h_boot = bootstrap_rho(human, h_idx)
    h_lo, h_hi, h_drop = percentile_ci(h_boot)
    for p in range(len(PAIRS)):  # clauses (3)-(5) read the sign of rho_human; it must sit on its interval's side
        if excludes_zero(h_lo[p], h_hi[p]) and np.sign(rho_h[p]) != np.sign(h_lo[p]):
            stop(f"rho_human {rho_h[p]} lies outside the side of its interval for {pair_label(*PAIRS[p])}")

    rows, meta = [], {"n_human": n_h, "n_agent": {}}
    for run in RUNS:
        n_checked += check_against_stored(rho_h, load_stored(root, run, "human"), f"{run} human")
        agent = load_scores(root, run, "socsim_agent")
        n_a = agent.shape[0]
        meta["n_agent"][run] = n_a
        a_idx = draw_indices(rng, n_a)
        rho_r = spearman_pairs(agent)
        n_checked += check_against_stored(rho_r, load_stored(root, run, "socsim_agent"), f"{run} agent")
        const = np.ptp(agent, axis=0) == 0
        r_boot = bootstrap_rho(agent, a_idx)
        r_lo, r_hi, r_drop = percentile_ci(r_boot)
        d_boot = r_boot - h_boot  # i-th agent resample paired with the i-th human resample
        d_lo, d_hi, d_drop = percentile_ci(d_boot)
        for p, (i, j) in enumerate(PAIRS):
            run_constant = bool(const[i] or const[j])
            v = verdict(run_constant, rho_h[p], (h_lo[p], h_hi[p]), rho_r[p], (r_lo[p], r_hi[p]), (d_lo[p], d_hi[p]))
            ah = fisher_bw_ci(rho_h[p], n_h)
            ar = fisher_bw_ci(rho_r[p], n_a)
            ad = zou_diff_ci(rho_r[p], ar, rho_h[p], ah) if not run_constant else (np.nan, np.nan)
            va = verdict(run_constant, rho_h[p], ah, rho_r[p], ar, ad)
            rows.append({
                "run_id": run, "run_name": names[run], "pair": pair_label(i, j),
                "focal": (CONSTRUCTS[i], CONSTRUCTS[j]) in FOCAL,
                "rho_human": rho_h[p], "rho_human_lo": h_lo[p], "rho_human_hi": h_hi[p],
                "human_target": excludes_zero(h_lo[p], h_hi[p]),
                "rho_run": rho_r[p], "rho_run_lo": r_lo[p], "rho_run_hi": r_hi[p],
                "delta_rho": rho_r[p] - rho_h[p], "delta_rho_lo": d_lo[p], "delta_rho_hi": d_hi[p],
                "verdict": v, "verdict_analytic": va,
                "n_human": n_h, "n_run": n_a,
                "dropped_resamples_human": int(h_drop[p]), "dropped_resamples_run": int(r_drop[p]),
                "dropped_resamples_delta": int(d_drop[p]),
            })
    meta["n_rho_checked"] = n_checked
    pairs = pd.DataFrame(rows)
    return pairs, summarise(pairs), meta


def summarise(pairs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for run, g in pairs.groupby("run_id", sort=False):
        rec = {"run_id": run, "run_name": g["run_name"].iloc[0]}
        for v in VERDICTS:
            rec[v.replace(" ", "_")] = int((g["verdict"] == v).sum())
        est = g["verdict"] != "not estimable"
        rec["pairs_with_human_target"] = int(g["human_target"].sum())
        rec["pairs_with_human_target_estimable"] = int((g["human_target"] & est).sum())
        rec["estimable_pairs"] = int(est.sum())
        rec["analytic_verdict_differs"] = int((g["verdict"] != g["verdict_analytic"]).sum())
        out.append(rec)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = parser(__doc__)
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <root>/reports/trait_structure)")
    a = ap.parse_args(argv)
    root = resolve_root(a.root)
    out = a.out or root / TRAIT_STRUCTURE
    pairs, summary, meta = analyse(root)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "f4_pairs_by_run.csv", index=False, float_format="%.6f")
    summary.to_csv(out / "f4_run_summary.csv", index=False)
    print(f"rho checked against 01_correlations.csv: {meta['n_rho_checked']}")
    print(summary.to_string(index=False))
    print(f"wrote {out / 'f4_pairs_by_run.csv'}, {out / 'f4_run_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
