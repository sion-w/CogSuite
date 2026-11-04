"""CR-27: F4 trait-structure transfer (ANALYSIS_PLAN.md §11, bullet "F4 Trait-structure transfer"; D-216).

Per run and per pair of the eight constructs (28 pairs): Spearman rho in the run and in the human
reference, recomputed from the per-run 00_person_frame.csv files exactly as 01_correlations.csv
computes them, with percentile bootstrap 95% CIs for rho_run, rho_human and delta = rho_run - rho_human
(10,000 resamples, numpy.random.default_rng(42), person = resampling unit, humans and agents resampled
independently; the human frame is resampled ONCE and the i-th human resample is paired with the i-th
agent resample of every run). Verdict per pair by the bullet's ordered rule; per-run tally; analytic
concordance check (Fisher z with Bonett-Wright variance, Zou 2007 difference interval) for the run
record only. No matrix-level similarity index.

Random stream: one Generator, default_rng(42). The human resample indices are drawn first
(10,000 x n_human); then each run's agent resample indices are drawn from the same Generator in the
fixed RUNS order below. Statistics only; no model call. Participant IDs and free text are read, never
written: outputs carry only run- and pair-level numbers.

Usage:
    python scripts/cr27_trait_structure_transfer.py [--reports DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = REPO / "reports"

RUNS = [
    "CR_MAIN_SYNTH_NUM_SONNET5_S42",
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42",
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42",
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42",
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42",
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42",
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42",
]
CONSTRUCTS = ["CRT2", "bCRT", "NFC", "Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
# Stored column labels in 01_correlations.csv (both sides) and the frame columns that reproduce them (D-216(d)).
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
    """Readable run names, read from the camera-ready builder's CONDITIONS registry (key "name")."""
    spec = importlib.util.spec_from_file_location("build_assets_cr27", REPO / "scripts" / "build_neurips_camera_ready_assets.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return {r: mod.CONDITIONS[r]["name"] for r in RUNS}


# ---------------------------------------------------------------- frames

def frame_dir(reports: Path, run: str, side: str) -> Path:
    return reports / f"cr13_{run}" / "tables" / "by_dataset" / side


def load_scores(reports: Path, run: str, side: str) -> np.ndarray:
    """Complete-case score matrix (n x 8) in CONSTRUCTS order. Only the eight score columns are kept."""
    cols = HUMAN_COLS if side == "human" else AGENT_COLS
    df = pd.read_csv(frame_dir(reports, run, side) / "00_person_frame.csv", usecols=cols, low_memory=False)
    return df[cols].dropna().to_numpy(dtype=float)


def human_cohort_hash(reports: Path, run: str) -> str:
    """SHA-256 of the sorted PROLIFIC_PIDs of the complete human rows (the IDs themselves are never returned)."""
    df = pd.read_csv(frame_dir(reports, run, "human") / "00_person_frame.csv",
                     usecols=["PROLIFIC_PID"] + HUMAN_COLS, low_memory=False).dropna(subset=HUMAN_COLS)
    return hashlib.sha256("\n".join(sorted(df["PROLIFIC_PID"].astype(str))).encode()).hexdigest()


def load_stored(reports: Path, run: str, side: str) -> pd.DataFrame:
    return pd.read_csv(frame_dir(reports, run, side) / "01_correlations.csv", index_col=0)


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
    """Percentile CI per column over the non-NaN draws; returns (lo, hi, n_dropped)."""
    dropped = np.isnan(draws).sum(axis=0)
    lo = np.full(draws.shape[1], np.nan)
    hi = np.full(draws.shape[1], np.nan)
    for c in range(draws.shape[1]):
        v = draws[~np.isnan(draws[:, c]), c]
        if v.size:
            lo[c], hi[c] = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, dropped


# ---------------------------------------------------------------- analytic intervals (concordance only)

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


# ---------------------------------------------------------------- the verdict rule (§11 F4, in its order)

def excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def verdict(run_constant: bool, rho_h: float, h_ci: tuple[float, float],
            rho_r: float, r_ci: tuple[float, float], d_ci: tuple[float, float]) -> str:
    """The §11 F4 verdict; the clauses are evaluated in the bullet's order and the first that holds decides."""
    # (1) not estimable: a run score is constant
    if run_constant:
        return "not estimable"
    run_excl, d_excl = excludes_zero(*r_ci), excludes_zero(*d_ci)
    # (2) human CI includes 0: introduced when the run CI and the delta CI exclude 0, otherwise no human target
    if not excludes_zero(*h_ci):
        return "introduced" if (run_excl and d_excl) else "no human target"
    h_sign = np.sign(rho_h)
    # (3) reversed: the run CI excludes 0 with the sign opposite to rho_human
    if run_excl and np.sign(r_ci[0]) == -h_sign:
        return "reversed"
    # (4) delta CI excludes 0: exaggerated when |rho_run| > |rho_human| with the same sign, otherwise weakened
    if d_excl:
        if np.sign(rho_r) == h_sign and abs(rho_r) > abs(rho_h):
            return "exaggerated"
        return "weakened"
    # (5) preserved: the run CI excludes 0 in the human direction
    if run_excl and np.sign(r_ci[0]) == h_sign:
        return "preserved"
    # (6) indeterminate otherwise
    return "indeterminate"


# ---------------------------------------------------------------- the analysis

def check_against_stored(rho: np.ndarray, stored: pd.DataFrame, where: str) -> int:
    """C3 contract: every recomputed rho equals the stored 01_correlations.csv value (blank <-> NaN)."""
    n = 0
    for p, (i, j) in enumerate(PAIRS):
        s = stored.loc[STORED[i], STORED[j]]
        if np.isnan(rho[p]) and pd.isna(s):
            continue
        if np.isnan(rho[p]) or pd.isna(s) or abs(rho[p] - s) > 1e-6:
            raise SystemExit(f"STOP C3 mismatch {where} {pair_label(i, j)}: recomputed {rho[p]} vs stored {s}")
        n += 1
    return n


def analyse(reports: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    names = run_names()
    hashes = {human_cohort_hash(reports, r) for r in RUNS}
    if len(hashes) != 1:
        raise SystemExit("STOP: the complete human cohort differs across run folders")
    human = load_scores(reports, RUNS[0], "human")
    for r in RUNS[1:]:
        if not np.array_equal(load_scores(reports, r, "human"), human):
            raise SystemExit(f"STOP: human frame in {r} differs from {RUNS[0]}")
    n_h = human.shape[0]

    rng = np.random.default_rng(SEED)
    h_idx = draw_indices(rng, n_h)
    rho_h = spearman_pairs(human)
    n_checked = check_against_stored(rho_h, load_stored(reports, RUNS[0], "human"), "human")
    h_boot = bootstrap_rho(human, h_idx)
    h_lo, h_hi, h_drop = percentile_ci(h_boot)
    for p in range(len(PAIRS)):  # clauses (3)-(5) read the sign of rho_human; it must sit on the CI's side
        if excludes_zero(h_lo[p], h_hi[p]) and np.sign(rho_h[p]) != np.sign(h_lo[p]):
            raise SystemExit(f"STOP: rho_human {rho_h[p]} lies outside the side of its CI for {pair_label(*PAIRS[p])}")

    rows, meta = [], {"n_human": n_h, "n_agent": {}, "seed": SEED, "n_boot": N_BOOT,
                      "human_idx_sha256": hashlib.sha256(h_idx.tobytes()).hexdigest()}
    for run in RUNS:
        n_checked += check_against_stored(rho_h, load_stored(reports, run, "human"), f"{run} human")
        agent = load_scores(reports, run, "socsim_agent")
        n_a = agent.shape[0]
        meta["n_agent"][run] = n_a
        a_idx = draw_indices(rng, n_a)
        rho_r = spearman_pairs(agent)
        n_checked += check_against_stored(rho_r, load_stored(reports, run, "socsim_agent"), f"{run} agent")
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


# ---------------------------------------------------------------- HEADLINE_RESULTS.md

def _ci(x: float, lo: float, hi: float, signed: bool = False) -> str:
    if np.isnan(x):
        return "—"
    f = "{:+.2f}" if signed else "{:.2f}"
    return f"{f.format(x)} [{f.format(lo)}, {f.format(hi)}]"


def headline(pairs: pd.DataFrame, summary: pd.DataFrame, meta: dict) -> str:
    L = ["# CR-27 — F4 trait-structure transfer: headline results", "",
         "Source: `f4_pairs_by_run.csv` and `f4_run_summary.csv` in this folder (ANALYSIS_PLAN.md §11, F4 bullet).",
         f"Spearman ρ, humans n = {meta['n_human']}, agents n = 471 per run; percentile bootstrap 95% CIs, "
         f"{meta['n_boot']:,} resamples, `default_rng({meta['seed']})`; Δρ = ρ_run − ρ_human. "
         "— = not estimable (a run score is constant).", "",
         "## 1. The six focal pairs, per run", ""]
    for run, g in pairs.groupby("run_id", sort=False):
        L += [f"### {g['run_name'].iloc[0]} (`{run}`)", "",
              "| Pair | ρ_human [95% CI] | ρ_run [95% CI] | Δρ [95% CI] | Verdict |", "|---|---|---|---|---|"]
        for _, r in g[g["focal"]].iterrows():
            L.append(f"| {r['pair']} | {_ci(r['rho_human'], r['rho_human_lo'], r['rho_human_hi'])} | "
                     f"{_ci(r['rho_run'], r['rho_run_lo'], r['rho_run_hi'])} | "
                     f"{_ci(r['delta_rho'], r['delta_rho_lo'], r['delta_rho_hi'], signed=True)} | {r['verdict']} |")
        L.append("")
    L += ["## 2. Verdict tally per run (all 28 pairs)", "",
          "| Run | " + " | ".join(VERDICTS) + " | Pairs with a human target | Estimable pairs |",
          "|---|" + "---:|" * (len(VERDICTS) + 2)]
    for _, s in summary.iterrows():
        L.append(f"| {s['run_name']} | " + " | ".join(str(s[v.replace(' ', '_')]) for v in VERDICTS)
                 + f" | {s['pairs_with_human_target']} | {s['estimable_pairs']} |")
    total = int(summary["analytic_verdict_differs"].sum())
    per = "; ".join(f"{s['run_name']} {s['analytic_verdict_differs']}" for _, s in summary.iterrows())
    L += ["", "Pairs with a human target: the human 95% CI excludes 0 (the same pairs in every run, since the human "
          "resamples are shared). Estimable pairs: no run score is constant.", "",
          "## 3. Concordance check (run record only)", "",
          f"Recomputing every verdict with analytic intervals (Fisher's z with the Bonett–Wright variance for each ρ, "
          f"Zou's interval for Δρ) changes {total} of {len(pairs)} verdicts ({per}).", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or a.reports / "cr27_trait_structure"
    pairs, summary, meta = analyse(a.reports)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "f4_pairs_by_run.csv", index=False, float_format="%.6f")
    summary.to_csv(out / "f4_run_summary.csv", index=False)
    # the headline is rendered from the CSVs just written, so every reader sees one source
    pairs_r = pd.read_csv(out / "f4_pairs_by_run.csv")
    summary_r = pd.read_csv(out / "f4_run_summary.csv")
    (out / "HEADLINE_RESULTS.md").write_text(headline(pairs_r, summary_r, meta), encoding="utf-8")
    print(f"rho checked against 01_correlations.csv: {meta['n_rho_checked']}; human resample sha256 {meta['human_idx_sha256'][:16]}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
