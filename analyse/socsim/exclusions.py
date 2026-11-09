"""Disclosed BRT exclusions.

The list is ``resources/brt_exclusions.csv`` (columns label, persona_id, item_ids, reason).

Four personas of `CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42` returned no usable open-ended
response, and the generating model snapshot is not available from the provider, so the
texts cannot be regenerated under the run's conditions. They are dropped from that
run's BRT frame before ANY BRT analysis, table or LIWC model. **CRT analyses are
unaffected**: the rows stay in the frame and only the BRT-side columns are voided, so
that run's CRT N is 471 and its BRT N is 467.

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
    return Path(__file__).resolve().parent / "resources" / "brt_exclusions.csv"


def load_brt_exclusions(path: Path | None = None) -> dict[str, set[str]]:
    """{run_label: {persona_id, ...}} — STOP if the exclusions file is missing."""
    p = Path(path) if path is not None else exclusions_path()
    if not p.exists():
        raise SystemExit(
            f"STOP: BRT exclusions file not found at {p}. The BRT analyses "
            "require it."
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
    """Every BRT-side column: coded fields, evidence, agreement and derived variables."""
    return [c for c in columns if str(c).startswith("BRT")]


def apply_brt_exclusions(df, run_label: str, exclusions: dict[str, set[str]] | None = None):
    """Void the BRT columns of the excluded personas of `run_label`.

    Returns (frame, dropped_ids). The rows are kept so CRT analyses are unaffected;
    every BRT column is set to NA, so every BRT analysis (which drops rows with
    missing ingredients) sees the reduced N.
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
