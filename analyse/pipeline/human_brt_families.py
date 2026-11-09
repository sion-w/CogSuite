"""Human BRT primary families: each primary BRT outcome regressed on bCRT in the 473-person human frame.

Per outcome: ML logistic regression of the 0/1 outcome on BCRT_Total (Firth-penalised logistic regression
with a profile-likelihood interval when the outcome has fewer than 20 events or non-events, or separates),
likelihood-ratio p, Benjamini-Hochberg q within the family, the OR per SD of bCRT, and the least reliable
coded ingredient of the outcome (dual-pass agreement and Fleiss' kappa on the 150-response coding panel,
``resources/brt_reliability_per_field.csv``).

Reads the human frame (see frames.py). Writes ``reports/human_brt_families/tables/s5_primary_families.csv``.

Usage: python analyse/pipeline/human_brt_families.py [--root DIR]
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import spec
from common import HUMAN_FAMILIES_TABLES, RESOURCES, parser, resolve_root, stop
from frames import dummies, human_frame
from brt_stats import bh, fit_binary

RELIABILITY = RESOURCES / "brt_reliability_per_field.csv"


def binary_row(df: pd.DataFrame, col: str, terms: list[str], focal: str, block: str, meta: dict) -> dict:
    res = fit_binary(df[col].astype(float), dummies(df, terms), focal)
    row = dict(meta)
    row.update({
        "outcome": col, "paper_facing_name": spec.PAPER_FACING.get(col, ""),
        "prior": spec.PRIOR.get(col, ""), "predictor_block": block, "focal": focal,
        "estimator": res["estimator"], "n": res["n"], "events": res["events"],
        "non_events": res["non_events"], "prevalence": res["prevalence"],
        "or": res["or"], "ci_lo": res["or_lo"], "ci_hi": res["or_hi"],
        "ci_type": res["ci_type"], "beta": res["beta"], "se": res["se"], "p_raw": res["p"],
        "ame_pp": res["ame_pp"], "flags": res["flags"], "terms": res["terms"],
    })
    return row


def reliability() -> pd.DataFrame:
    """Per primary outcome: the minimum % agreement and kappa over its coded ingredients."""
    lk = pd.read_csv(RELIABILITY).set_index("field")
    rows = []
    for c in spec.PRIMARY_OUTCOMES:
        ing = spec.ingredients(c)
        missing = [i for i in ing if i not in lk.index]
        if missing:
            stop(f"no reliability row for ingredient(s) {missing} of {c}")
        sub = lk.loc[ing]
        rows.append({"outcome": c, "min_agreement_pct": float(sub["dual_pass_agreement_pct"].min()),
                     "min_kappa": float(sub["fleiss_kappa"].min())})
    return pd.DataFrame(rows).set_index("outcome")


def primary_families(df: pd.DataFrame) -> pd.DataFrame:
    block, terms, focal, desc = spec.HUMAN_BLOCK
    rows = []
    for fam in spec.FAMILIES:
        part = [binary_row(df, c, terms, focal, block, {
            "section": "5", "placement": "primary", "family_key": fam["key"], "family": fam["title"],
            "tag": fam["key"], "m": len(fam["members"]), "member_order": i + 1})
            for i, (c, _, _) in enumerate(fam["members"])]
        for r, q in zip(part, bh([r["p_raw"] for r in part])):
            r["q_bh_within_family"] = q
        rows += part
    out = pd.DataFrame(rows)
    out["predictor_description"] = desc
    sd = float(pd.to_numeric(df[focal], errors="coerce").std(ddof=1))
    out["predictor_sd"] = sd
    for a, b in (("or", "or_per_sd"), ("ci_lo", "or_per_sd_lo"), ("ci_hi", "or_per_sd_hi")):
        out[b] = np.exp(np.log(out[a]) * sd)
    out["ci_excludes_null"] = (out["ci_lo"] > 1) | (out["ci_hi"] < 1)
    rel = reliability()
    out["ingredient_min_agreement_pct"] = out["outcome"].map(rel["min_agreement_pct"])
    out["ingredient_min_kappa"] = out["outcome"].map(rel["min_kappa"])
    return out


def main(argv: list[str] | None = None) -> int:
    a = parser(__doc__).parse_args(argv)
    root = resolve_root(a.root)
    df = human_frame(root)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = primary_families(df)
    out = root / HUMAN_FAMILIES_TABLES
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "s5_primary_families.csv", index=False)
    n_sig = int((t["q_bh_within_family"] < 0.05).sum())
    print(f"wrote {out / 's5_primary_families.csv'} ({len(t)} outcomes, {n_sig} at q < .05)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
