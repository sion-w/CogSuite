"""Disclosed BRT exclusions (PI ruling A, 2026-09-23; CR-13R).

Machine-readable source of record:
`../cognition-suite-sim/simulation_outputs/_cr13r_backfill/EXCLUSIONS.csv`.

Four personas of `CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42` returned no usable open-ended
response in the August generation, and the generating model snapshot has since been
retired by the provider, so the texts could not be regenerated under the original
conditions. They are dropped from that run's BRT frame before ANY BRT analysis,
table or LIWC model (CR-13B STEP 3b). **CRT analyses are unaffected** — the rows stay
in the frame and only the BRT-side columns are voided, so CRT N stays 471 while that
run's BRT N is 467.

This module reads the CSV; it never hard-codes the list.
"""
from __future__ import annotations

import csv
from pathlib import Path

EXCLUSION_NOTE = (
    "DSV4PRO BRT N = 467: four personas returned no usable open-ended response, and the "
    "generating model snapshot has since been retired by the provider."
)

_REPO = Path(__file__).resolve().parent.parent


def exclusions_path() -> Path:
    import os

    sim = os.environ.get("COGNITION_SUITE_SIM")
    base = Path(sim) if sim else _REPO.parent / "cognition-suite-sim"
    return base / "simulation_outputs" / "_cr13r_backfill" / "EXCLUSIONS.csv"


def load_brt_exclusions(path: Path | None = None) -> dict[str, set[str]]:
    """{run_label: {persona_id, ...}} — STOP if the ruling file is missing."""
    p = Path(path) if path is not None else exclusions_path()
    if not p.exists():
        raise SystemExit(
            f"STOP: PI exclusions file not found at {p}. CR-13B STEP 3b requires it "
            "(PI ruling A, 23 Sep 2026; CR-13R)."
        )
    out: dict[str, set[str]] = {}
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            pid = str(row.get("persona_id") or "").strip()
            if label and pid:
                out.setdefault(label, set()).add(pid)
    return out


def brt_columns(columns) -> list[str]:
    """Every BRT-side column: coded fields, evidence, agreement and §4.3 derived."""
    return [c for c in columns if str(c).startswith("BRT")]


def apply_brt_exclusions(df, run_label: str, exclusions: dict[str, set[str]] | None = None):
    """Void the BRT columns of the excluded personas of `run_label`.

    Returns (frame, dropped_ids). The rows are kept so CRT analyses are unaffected;
    every BRT column is set to NA, so every BRT analysis (which drops missing
    ingredients by the §4.3 rule) sees the reduced N.
    """
    ex = load_brt_exclusions() if exclusions is None else exclusions
    ids = ex.get(str(run_label), set())
    if not ids or df is None or len(df) == 0 or "entity_id" not in df.columns:
        return df, []
    out = df.copy()
    mask = out["entity_id"].astype(str).isin(ids)
    hit = sorted(out.loc[mask, "entity_id"].astype(str).tolist(), key=lambda s: int(s) if s.isdigit() else 0)
    if not hit:
        return out, []
    cols = brt_columns(out.columns)
    for col in cols:
        out.loc[mask, col] = None
    return out, hit
