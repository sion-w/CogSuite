"""CR-28: lure-rate intervals for main Figure 2 and the configuration-ladder table (D-230, D-232(a)).

(1) The six ladder cells whose source jobs (CR-FSb, CR-FSd) stored only a pooled Wilson interval: the
    per-test (crt2, bcrt) and pooled intervals by the method that produced every other ladder interval,
    the builder's bootstrap_lure_ci (socsim.rebuttal_analyses.cluster_bootstrap_proportion_ci: persona-
    clustered percentile bootstrap, 5000 resamples, seed 42) on the scored denominator (blank answers
    dropped, D-187(b), CR-25b), persona ids clustered as integers (the CR-FS pipeline's id type).
(2) The human lure rate per item with a 95% interval: bCRT as answered (Wilson on the 471 CRT2-complete
    respondents' scored answers, the builder's wilson_ci), and CRT2 familiarity-adjusted (the CR-13B
    reference, socsim.familiarity_reference: per item a multinomial logit fitted by _fit_mnlogit_one_item,
    adjusted rate = mean over persons of _predict_mean_probs at seen = 0; pooled = mean of the four item
    vectors) with a percentile bootstrap over people that refits all four items on every resample.

Bootstrap of the human CRT2 reference. One Generator, default_rng(CR28_HUMAN_SEED); resample b draws
n persons with replacement (rng.choice over the frame's n persons, the draw style of
cluster_bootstrap_proportion_ci); every draw gets a unique id, so a person drawn twice counts twice in the
fit and in the prediction average. The original model specification is held fixed: the rare-level
collapse (_collapse_rare, < 5 people -> "Other_or_missing") is applied once, on the original frame's
persons, before resampling, so a resample cannot create or merge a level; a resample in which a level
present in an item's original fit rows (an outcome category, or a gender or education level) is absent
from that item's fit rows, or in which a level falls below the collapse threshold so that the fit function's
own collapse would change the design or its reference level, or in which any item's fit raises, does not
converge or predicts non-finite probabilities, is dropped and counted. More than CR28_MAX_FAIL_SHARE dropped -> MANUAL_ACTION_NEEDED.

D-235 (re-fire 2026-10-15): the first run dropped 2,217 of 5,000 of those resamples (recorded in
human_crt2_bootstrap_failures_cr28.csv), and the PI ruled that the CRT2 familiarity-adjusted line gets no
interval. With CR28_HUMAN_CRT2_ADJUSTED = False the bootstrap is not run (its code stays, for the record);
the C3 (ii) reproduction of the adjusted point estimates still runs, the human output is the four bCRT rows,
and the first run's ladder_ci_cr28.csv and failure counts are never rewritten: the ladder half is recomputed
and must equal the written file byte for byte.

Statistics only; no model call. Participant IDs and free text are read, never written: outputs carry only
cell-, item- and resample-count-level numbers.

Usage:
    python scripts/cr28_ladder_intervals.py [--data-root DIR] [--out DIR] [--ladder CSV] [--exposure CSV]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from socsim import familiarity_reference as FR  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_assets_cr28", REPO / "scripts" / "build_neurips_camera_ready_assets.py")
B = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = B
_spec.loader.exec_module(B)

# Ruled values (CR-28 prompt).
CR28_ID_TYPE = int                      # CR-FSb / CR-FSd are the CR-FS pipeline; builder CI_PERSONA_ID_TYPE["CR-FS"]
CR28_SCOPES = ("crt2", "bcrt", "pooled")
CR28_HUMAN_CRT2_ADJUSTED = False         # D-235: the familiarity-adjusted bootstrap is not run
CR28_HUMAN_BOOT = 5000                  # unused since D-235; kept for the record
CR28_HUMAN_SEED = 42                    # unused since D-235; kept for the record
CR28_MAX_FAIL_SHARE = 0.01              # unused since D-235; kept for the record

DATA_ROOT = Path.home() / "Desktop" / "phd" / "CR" / "soc-sim"     # git-ignored data shared by every checkout
OUT_DIR = DATA_ROOT / "reports" / "cr28_ladder_ci"
# The bundle of record (hub note, "Where things live", Figures/Tables (of record)).
BUNDLE_TABLES = B.VAULT_CR / "cr-25d" / "neurips_paper_assets_v2" / "tables"
LADDER_TABLE = BUNDLE_TABLES / "table_configuration_ladder.csv"
EXPOSURE_TABLE = BUNDLE_TABLES / "table_item_exposure_and_unseen_lure.csv"

CR28_CELLS = {
    "CRFSB_SONNET5_THINKOFF_NUM": "CR-FSb",
    "CRFSB_CLAUDE_TOP_NUM": "CR-FSb",
    "CRFSB_GPT56_REASONING_NUM": "CR-FSb",
    "CRFSD_DSV4PRO_TOG_NUM": "CR-FSd",
    "CRFSD_DSV4PRO_TOG_NOTHINK_NUM": "CR-FSd",
    "CRFSD_KIMI_K3_NUM": "CR-FSd",
}
REPRO_CELL = "CRFS_SONNET5_DEFAULT_NUM"     # a CR-FS cell with stored per-test intervals (C3 (i))
LADDER_SEED = 42                            # fixed inside bootstrap_lure_ci
LADDER_METHOD = "persona-clustered bootstrap, 5000 resamples (CR-28); scored denominator"
WILSON_METHOD = "Wilson"
BOOT_METHOD = f"person bootstrap, {CR28_HUMAN_BOOT} resamples, seed {CR28_HUMAN_SEED}, model refit"
SEEN_COL = "CRT2_SeenCount"                 # the primary reference's seen predictor (build_adjusted_crt2_reference)
MODEL_TAG = "primary_seen_count"
FIT_COLS = ["entity_id", "item_id", "response_category", SEEN_COL, FR.DEMO_AGE_COL, FR.DEMO_GENDER_COL, FR.DEMO_EDU_COL]
REASONS = ["outcome_level_absent", "covariate_level_absent", "level_collapsed", "fit_error", "not_converged", "non_finite"]

LADDER_CSV = "ladder_ci_cr28.csv"
HUMAN_CSV = "human_item_ci_cr28.csv"
FAILURES_CSV = "human_crt2_bootstrap_failures_cr28.csv"
HEADLINE_MD = "HEADLINE_RESULTS.md"


def stop(msg: str) -> None:
    raise SystemExit(f"MANUAL_ACTION_NEEDED: {msg}")


def read_ladder(path: Path = LADDER_TABLE) -> pd.DataFrame:
    ladder = pd.read_csv(path).set_index("source")
    if len(ladder) != 14:
        stop(f"{path.name} has {len(ladder)} rows, expected 14.")
    return ladder


def parse_count(s: str) -> tuple[int, int]:
    k, n = str(s).split("/")
    return int(k), int(n)


# ---------------------------------------------------------------- (1) ladder cells

def ladder_repro_guard(data_root: Path, ladder: pd.DataFrame) -> list[dict]:
    """C3 (i): the method on the source denominator reproduces REPRO_CELL's stored CR-FS intervals."""
    df = B.read_rescore_long(data_root, REPRO_CELL)
    rows = []
    for scope in ("crt2", "bcrt"):
        rep = B.bootstrap_lure_ci(df, scope, CR28_ID_TYPE, scored_only=False)
        lo, hi = float(ladder.at[REPRO_CELL, f"lure_{scope}_ci_low"]), float(ladder.at[REPRO_CELL, f"lure_{scope}_ci_high"])
        diff = max(abs(rep["ci95_low"] - lo), abs(rep["ci95_high"] - hi))
        rows.append({"scope": scope, "stored": (lo, hi), "reproduced": (rep["ci95_low"], rep["ci95_high"]),
                     "max_abs_diff": diff, "ok": diff <= B.CI_REPRO_ATOL})
    return rows


def cell_interval(df: pd.DataFrame, scope: str) -> dict:
    rep = B.bootstrap_lure_ci(df, scope, CR28_ID_TYPE, scored_only=True)
    return {"n_events": rep["n_events"], "n_items": rep["n_items"], "rate": rep["n_events"] / rep["n_items"],
            "ci95_low": rep["ci95_low"], "ci95_high": rep["ci95_high"]}


def ladder_intervals(data_root: Path, ladder: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stem, job in CR28_CELLS.items():
        df = B.read_rescore_long(data_root, stem)
        for scope in CR28_SCOPES:
            r = cell_interval(df, scope)
            stored = parse_count(ladder.at[stem, f"lure_{scope}_count"])
            if (r["n_events"], r["n_items"]) != stored:       # C4
                stop(f"{stem} {scope}: scored {r['n_events']}/{r['n_items']} != ladder {stored[0]}/{stored[1]}.")
            wl = wh = np.nan
            if scope == "pooled":
                if not str(ladder.at[stem, "lure_pooled_ci_method"]).startswith("Wilson"):
                    stop(f"{stem}: the stored pooled interval is not the Wilson one.")
                wl, wh = float(ladder.at[stem, "lure_pooled_ci_low"]), float(ladder.at[stem, "lure_pooled_ci_high"])
            rows.append({"source": stem, "source_job": job, "scope": scope, **r, "method": LADDER_METHOD,
                         "id_type": CR28_ID_TYPE.__name__, "seed": LADDER_SEED, "wilson_low": wl, "wilson_high": wh})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (2) human reference

def human_bcrt(data_root: Path) -> pd.DataFrame:
    """bCRT as answered: Wilson on the 471 CRT2-complete respondents' scored answers, per item."""
    long, _ = B.cr25_human_frame(data_root)
    scored = long[long["category"] != B.UNSCORED]
    rows = []
    for item in B.BCRT_ITEMS:
        s = scored[scored["item_id"] == item]
        k, n = int(s["category"].eq("intuitive_incorrect").sum()), int(len(s))
        lo, hi = B.wilson_ci(k, n)
        rows.append({"test": "bCRT", "item": item.split("_")[1], "reference": "raw", "rate": k / n,
                     "ci95_low": lo, "ci95_high": hi, "method": WILSON_METHOD,
                     "n_people": int(s["entity_id"].nunique())})
    return pd.DataFrame(rows)


def bcrt_lure_all_check(bcrt: pd.DataFrame, exposure_path: Path = EXPOSURE_TABLE) -> list[dict]:
    """C3 (ii), bCRT half: the rates equal lure_all of the bundle's exposure table."""
    ex = pd.read_csv(exposure_path)
    ex = ex[(ex["scope"] == "item") & (ex["test"] == "bCRT")].set_index("item")
    out = []
    for _, r in bcrt.iterrows():
        stored = float(ex.at[f"bCRT {r['item']}", "lure_all"])
        out.append({"item": r["item"], "stored": stored, "mine": r["rate"], "ok": abs(stored - r["rate"]) <= B.CI_REPRO_ATOL})
    return out


def human_crt2_frame(data_root: Path) -> pd.DataFrame:
    """The CR-13B reference's CRT2 frame (load_human_item_frame), with the original rare-level collapse fixed
    per person, reduced to the columns the fit and the prediction read."""
    merged, _ = FR.load_human_item_frame(data_root / B.HUMAN_V2_LONG, data_root / B.HUMAN_PERSON_FILE)
    crt2 = merged[merged["item_id"].isin(FR.CRT2_ITEMS)].copy()
    persons = crt2.drop_duplicates("entity_id")
    for col in (FR.DEMO_GENDER_COL, FR.DEMO_EDU_COL):
        label = dict(zip(persons["entity_id"], FR._collapse_rare(persons[col])))
        crt2[col] = crt2["entity_id"].map(label)
    return crt2[FIT_COLS].reset_index(drop=True)


def refit_adjusted(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """build_adjusted_crt2_reference's primary loop on `frame`: per item fit, predict over the frame's persons at
    seen = 0; pooled = the mean of the four item vectors (correct, intuitive_incorrect, other_incorrect)."""
    persons = frame.drop_duplicates("entity_id")
    vecs, metas = {}, {}
    for item in FR.CRT2_ITEMS:
        item_df = frame[frame["item_id"] == item]
        fit, _, mmeta, _ = FR._fit_mnlogit_one_item(item_df, seen_col=SEEN_COL, item_id=item, model_tag=MODEL_TAG)
        _, vec = FR._predict_mean_probs(fit, persons, seen_value=0.0, seen_col_name=SEEN_COL, meta=mmeta)
        vecs[item] = vec
        metas[item] = {"exog_names": list(mmeta["exog_names"]), "gender_ref": mmeta["gender_ref"],
                       "edu_ref": mmeta["edu_ref"], "converged": bool(fit.mle_retvals.get("converged", True))}
    vecs["pooled"] = np.mean(np.vstack([vecs[i] for i in FR.CRT2_ITEMS]), axis=0)
    return vecs, metas


def fit_levels(frame: pd.DataFrame) -> dict[str, dict[str, set]]:
    """Per item, the outcome / gender / education levels present in the fit rows (scored answer, complete covariates,
    the rows _fit_mnlogit_one_item fits on), and the gender / education levels its design matrix would take
    (_collapse_rare on the scored rows, as _design_matrix does)."""
    out = {}
    for item in FR.CRT2_ITEMS:
        u = frame[(frame["item_id"] == item) & frame["response_category"].isin(FR.OUTCOME_LEVELS)]
        d = u[pd.to_numeric(u[FR.DEMO_AGE_COL], errors="coerce").notna() & pd.to_numeric(u[SEEN_COL], errors="coerce").notna()]
        out[item] = {"outcome": set(d["response_category"]), "gender": set(d[FR.DEMO_GENDER_COL]),
                     "edu": set(d[FR.DEMO_EDU_COL]),
                     "design": (set(FR._collapse_rare(u[FR.DEMO_GENDER_COL])), set(FR._collapse_rare(u[FR.DEMO_EDU_COL])))}
    return out


def resample(frame: pd.DataFrame, rows_by_person: list[np.ndarray], draw: np.ndarray) -> pd.DataFrame:
    rows = np.concatenate([rows_by_person[i] for i in draw])
    ids = np.repeat([f"r{j}" for j in range(len(draw))], [len(rows_by_person[i]) for i in draw])
    out = frame.iloc[rows].copy()
    out["entity_id"] = ids
    return out.reset_index(drop=True)


def human_crt2_bootstrap(frame: pd.DataFrame, n_boot: int = CR28_HUMAN_BOOT, seed: int = CR28_HUMAN_SEED) -> dict:
    persons = frame["entity_id"].drop_duplicates().to_numpy()
    n = len(persons)
    pos = {p: i for i, p in enumerate(persons)}
    person_idx = frame["entity_id"].map(pos).to_numpy()
    rows_by_person = [np.flatnonzero(person_idx == i) for i in range(n)]
    ref_levels = fit_levels(frame)
    _, ref_meta = refit_adjusted(frame)

    rng = np.random.default_rng(seed)
    keys = FR.CRT2_ITEMS + ["pooled"]
    est = np.full((n_boot, len(keys)), np.nan)
    flags = {r: np.zeros(n_boot, dtype=bool) for r in REASONS}
    by_item = {(item, kind): 0 for item in FR.CRT2_ITEMS for kind in ("outcome", "gender", "edu", "collapsed")}
    for b in range(n_boot):
        draw = rng.choice(n, size=n, replace=True)
        rs = resample(frame, rows_by_person, draw)
        lv = fit_levels(rs)
        for item in FR.CRT2_ITEMS:
            for kind in ("outcome", "gender", "edu"):
                if not ref_levels[item][kind] <= lv[item][kind]:
                    by_item[(item, kind)] += 1
                    flags["outcome_level_absent" if kind == "outcome" else "covariate_level_absent"][b] = True
            if lv[item]["design"] != ref_levels[item]["design"]:
                by_item[(item, "collapsed")] += 1
                flags["level_collapsed"][b] = True
        if flags["outcome_level_absent"][b] or flags["covariate_level_absent"][b] or flags["level_collapsed"][b]:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vecs, meta = refit_adjusted(rs)
        except Exception:  # noqa: BLE001 - any failed fit is dropped and counted (the prompt's rule)
            flags["fit_error"][b] = True
            continue
        if any(meta[i]["exog_names"] != ref_meta[i]["exog_names"] for i in FR.CRT2_ITEMS):
            raise AssertionError("a resample's design differs from the original although every level is present")
        if not all(meta[i]["converged"] for i in FR.CRT2_ITEMS):
            flags["not_converged"][b] = True
            continue
        row = np.array([vecs[k][1] for k in keys])
        if not np.isfinite(row).all():
            flags["non_finite"][b] = True
            continue
        est[b] = row
    failed = np.zeros(n_boot, dtype=bool)
    for r in REASONS:
        failed |= flags[r]
    return {"keys": keys, "estimates": est, "failed": failed, "flags": flags, "by_item": by_item,
            "n_persons": n, "n_boot": n_boot, "seed": seed}


def failure_table(boot: dict) -> pd.DataFrame:
    rows = [{"what": "resamples", "count": boot["n_boot"]},
            {"what": "failed (dropped), any reason", "count": int(boot["failed"].sum())},
            {"what": "fail share", "count": float(boot["failed"].mean())},
            {"what": "gate CR28_MAX_FAIL_SHARE", "count": CR28_MAX_FAIL_SHARE}]
    rows += [{"what": f"reason: {r} (not exclusive)", "count": int(boot["flags"][r].sum())} for r in REASONS]
    rows += [{"what": f"level absent: {item} {kind}", "count": c} for (item, kind), c in boot["by_item"].items()]
    return pd.DataFrame(rows)


def human_crt2_rows(point: dict[str, np.ndarray], boot: dict) -> pd.DataFrame:
    ok = ~boot["failed"]
    rows = []
    for j, key in enumerate(boot["keys"]):
        lo, hi = np.percentile(boot["estimates"][ok, j], [2.5, 97.5])
        rows.append({"test": "CRT2", "item": "pooled" if key == "pooled" else key.split("_")[1],
                     "reference": "familiarity-adjusted", "rate": float(point[key][1]), "ci95_low": float(lo),
                     "ci95_high": float(hi), "method": BOOT_METHOD, "n_people": boot["n_persons"],
                     "failed_resamples": int(boot["failed"].sum())})
    return pd.DataFrame(rows)


def crt2_repro_check(point: dict[str, np.ndarray], data_root: Path) -> list[dict]:
    """C3 (ii), CRT2 half: the refit on the unresampled frame reproduces CR-13B's stored adjusted reference."""
    by_item = pd.read_csv(data_root / B.CR13B_CRT / "crt2ref_adjusted_by_item.csv")
    by_item = by_item[(by_item["reference"] == "adjusted_seen0") & (by_item["response_category"] == "intuitive_incorrect")]
    pooled = pd.read_csv(data_root / B.CR13B_CRT / "crt2ref_adjusted_pooled.csv").set_index("response_category")
    out = []
    for item in FR.CRT2_ITEMS:
        stored = float(by_item.set_index("item_id").at[item, "proportion"])
        out.append({"item": item, "stored": stored, "mine": float(point[item][1]),
                    "ok": abs(stored - point[item][1]) <= B.CI_REPRO_ATOL})
    stored = float(pooled.at["intuitive_incorrect", "proportion"])
    out.append({"item": "pooled", "stored": stored, "mine": float(point["pooled"][1]),
                "ok": abs(stored - point["pooled"][1]) <= B.CI_REPRO_ATOL})
    return out


# ---------------------------------------------------------------- outputs

def pct_ci(rate: float, lo: float, hi: float) -> str:
    return f"{100 * rate:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def no_interval_line(failures_csv: Path) -> str:
    """The D-235 line, its counts read from the first run's failure table."""
    f = pd.read_csv(failures_csv).set_index("what")["count"]
    return (f"CRT2 familiarity-adjusted: no interval (D-235); the person bootstrap dropped "
            f"{int(f['failed (dropped), any reason']):,} of {int(f['resamples']):,} resamples")


def headline(ladder_csv: Path, human_csv: Path, failures_csv: Path) -> str:
    lad = pd.read_csv(ladder_csv)
    hum = pd.read_csv(human_csv)
    lines = ["# CR-28 headline results", "",
             "Intuitive-lure rate, % [95% CI]. Ladder cells: " + LADDER_METHOD + ", persona ids as integers, seed 42; "
             "the old interval is the pooled Wilson interval of the ladder table of record.", "",
             "| Cell | CRT2 % [95% CI] | bCRT % [95% CI] | pooled % [95% CI] | pooled, old Wilson [95% CI] |",
             "|---|---|---|---|---|"]
    for src in CR28_CELLS:
        c = lad[lad["source"] == src].set_index("scope")
        p = c.loc["pooled"]
        lines.append(f"| {src} | " + " | ".join(pct_ci(c.at[s, "rate"], c.at[s, "ci95_low"], c.at[s, "ci95_high"])
                                                for s in CR28_SCOPES)
                     + f" | [{100 * p['wilson_low']:.1f}, {100 * p['wilson_high']:.1f}] |")
    lines += ["", "Human reference, intuitive-lure rate, % [95% CI].", "",
              "| Test | Item | Reference | % [95% CI] | Method | People |", "|---|---|---|---|---|---:|"]
    for _, r in hum.iterrows():
        lines.append(f"| {r['test']} | {r['item']} | {r['reference']} | {pct_ci(r['rate'], r['ci95_low'], r['ci95_high'])} "
                     f"| {r['method']} | {int(r['n_people'])} |")
    lines += ["", no_interval_line(failures_csv)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--ladder", type=Path, default=LADDER_TABLE)
    ap.add_argument("--exposure", type=Path, default=EXPOSURE_TABLE)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ladder = read_ladder(a.ladder)

    for g in ladder_repro_guard(a.data_root, ladder):
        print(f"C3 (i) {g['scope']}: stored {g['stored']} reproduced {g['reproduced']} |diff| {g['max_abs_diff']:.2e}")
        if not g["ok"]:
            stop(f"C3 (i) {REPRO_CELL} {g['scope']}: stored {g['stored']} vs reproduced {g['reproduced']}.")
    cells = ladder_intervals(a.data_root, ladder)
    new = cells.to_csv(index=False).encode("utf-8")
    ladder_path = a.out / LADDER_CSV
    if ladder_path.exists():
        # D-235: the first run's file is never changed; the recomputation must equal it byte for byte.
        if ladder_path.read_bytes() != new:
            stop(f"the recomputed ladder half differs from the written {ladder_path}; left unchanged.")
        print(f"ladder half recomputed: byte-identical to {ladder_path} ({len(cells)} rows); file left unchanged")
    else:
        ladder_path.write_bytes(new)
        print(f"wrote {ladder_path} ({len(cells)} rows)")

    bcrt = human_bcrt(a.data_root)
    for c in bcrt_lure_all_check(bcrt, a.exposure):
        print(f"C3 (ii) bCRT {c['item']}: lure_all {c['stored']!r} mine {c['mine']!r}")
        if not c["ok"]:
            stop(f"C3 (ii) bCRT {c['item']}: lure_all {c['stored']!r} vs {c['mine']!r}.")
    frame = human_crt2_frame(a.data_root)
    ident = resample(frame, [np.flatnonzero(frame["entity_id"].to_numpy() == p) for p in frame["entity_id"].drop_duplicates()],
                     np.arange(frame["entity_id"].nunique()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point, _ = refit_adjusted(ident)
    for c in crt2_repro_check(point, a.data_root):
        print(f"C3 (ii) CRT2 {c['item']}: stored {c['stored']!r} mine {c['mine']!r}")
        if not c["ok"]:
            stop(f"C3 (ii) CRT2 {c['item']}: stored {c['stored']!r} vs {c['mine']!r}.")

    if CR28_HUMAN_CRT2_ADJUSTED:
        stop("CR28_HUMAN_CRT2_ADJUSTED is True, but D-235 rules that the familiarity-adjusted bootstrap is not run.")
    if not (a.out / FAILURES_CSV).exists():
        stop(f"{a.out / FAILURES_CSV} (the first run's failure counts, D-235) is missing.")
    bcrt.to_csv(a.out / HUMAN_CSV, index=False)
    (a.out / HEADLINE_MD).write_text(headline(ladder_path, a.out / HUMAN_CSV, a.out / FAILURES_CSV), encoding="utf-8")
    print(f"wrote {a.out / HUMAN_CSV} ({len(bcrt)} rows), {a.out / HEADLINE_MD}")


if __name__ == "__main__":
    main()
