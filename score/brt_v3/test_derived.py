"""Unit tests for the BRT v3 derived variables and the v2.4 mapping (no API calls).

The event-count tests at the end need the companion study's human BRT labels
(``<input root>/reports/brt_labels/brt_labels_v3_human.csv``, N=473; controlled access) and are
skipped when that file is absent.

Run:  python -m pytest score/brt_v3 -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derived import DERIVED_V3_COLUMNS, compute_all  # noqa: E402
from schema import BOOL_FIELDS, CODING_FIELDS, STRING_FIELDS  # noqa: E402


def _blank_row(**kwargs) -> dict:
    row = {k: list(v)[0] if k.endswith("decision") else "ANCHOR_NONE" for k, v in STRING_FIELDS.items()}
    row["BRT1_decision"] = "UNCLEAR_OR_CONDITIONAL"
    row["BRT2_decision"] = "UNCLEAR_OR_CONDITIONAL"
    row["BRT1_anchor"] = "ANCHOR_NONE"
    row["BRT2_anchor"] = "ANCHOR_NONE"
    for k in BOOL_FIELDS:
        row[k] = 0
    row.update(kwargs)
    return row


def _one(**kwargs) -> pd.DataFrame:
    return compute_all(pd.DataFrame([_blank_row(**kwargs)]))


def test_field_count():
    assert len(CODING_FIELDS) == 28


def test_quality_evidence_exclusive_professional_ok():
    r = _one(
        BRT2_check_institutional_sources=1,
        BRT2_check_professional=1,
        BRT2_check_lay_content=0,
    ).iloc[0]
    assert r["BRT2_research_quality_v3"] == "EVIDENCE_EXCLUSIVE"


def test_quality_consult_mixed():
    r = _one(BRT2_check_professional=1, BRT2_check_lay_content=1).iloc[0]
    assert r["BRT2_research_quality_v3"] == "CONSULT_MIXED"


def test_quality_bare_articles_unspecified():
    r = _one(BRT2_check_unspecified=1).iloc[0]
    assert r["BRT2_research_quality_v3"] == "UNSPECIFIED_ONLY"
    assert r["v24_BRT2_research_specificity"] == "VAGUE_RESEARCH"


def test_brt1_venue_neutral_exclusive():
    r = _one(BRT1_check_official=1, BRT1_check_venue=1).iloc[0]
    assert r["BRT1_research_quality_v3"] == "OFFICIAL_EXCLUSIVE"


def test_brt1_venue_only_noninstitutional():
    r = _one(BRT1_check_venue=1).iloc[0]
    assert r["BRT1_research_quality_v3"] == "NONINSTITUTIONAL_ONLY"


def test_practical_take_excluded_from_accept():
    """Heuristic take, no checks, need-now → PRACTICAL; cannot enter reflective_accept_v3."""
    r = _one(
        BRT2_decision="TAKE_MEDEX",
        BRT2_anchor="ANCHOR_PRACTICAL_GROUNDS",
    ).iloc[0]
    assert r["BRT2_research_quality_v3"] == "NONE"
    assert r["BRT2_reflective_accept_v3"] == 0.0


def test_anxious_delay_excluded():
    r = _one(
        BRT2_decision="DELAY_AND_INVESTIGATE",
        BRT2_check_institutional_sources=1,
        BRT2_contingency_stated=0,
    ).iloc[0]
    assert r["BRT2_research_quality_v3"] == "EVIDENCE_EXCLUSIVE"
    assert r["BRT2_reflective_delay_v3"] == 0.0
    assert r["BRT2_reflective_delay_v3_plan"] == 0.0


def test_delay_plan_mixed_tier_counts_strict_does_not():
    r = _one(
        BRT2_decision="DELAY_AND_INVESTIGATE",
        BRT2_check_institutional_sources=1,
        BRT2_check_persons=1,
        BRT2_contingency_stated=1,
    ).iloc[0]
    assert r["BRT2_research_quality_v3"] == "EVIDENCE_MIXED"
    assert r["BRT2_reflective_delay_v3_plan"] == 1.0
    assert r["BRT2_reflective_delay_v3"] == 0.0


def test_accept_plan_requires_a_plan():
    r = _one(BRT2_decision="TAKE_MEDEX", BRT2_anchor="ANCHOR_MED_SYSTEM").iloc[0]
    assert r["BRT2_research_quality_v3"] == "NONE"
    assert r["BRT2_reflective_accept_v3"] == 1.0
    assert r["BRT2_reflective_accept_v3_plan"] == 0.0
    r2 = _one(BRT2_decision="TAKE_MEDEX", BRT2_anchor="ANCHOR_MED_SYSTEM",
              BRT2_check_professional=1, BRT2_check_lay_content=1).iloc[0]
    assert r2["BRT2_research_quality_v3"] == "CONSULT_MIXED"
    assert r2["BRT2_reflective_accept_v3_plan"] == 1.0
    assert r2["BRT2_reflective_accept_v3"] == 0.0


def test_search_profile_no_plan_union():
    for kw in ({"BRT2_check_lay_content": 1}, {"BRT2_check_unspecified": 1}, {}):
        assert _one(**kw).iloc[0]["BRT2_search_profile_v3"] == "NO_PLAN"
    assert _one(BRT2_check_professional=1).iloc[0]["BRT2_search_profile_v3"] == "CONSULT_ONLY"


def test_takes_or_delays_matches_full_category_name():
    assert _one(BRT2_decision="DELAY_AND_INVESTIGATE").iloc[0]["BRT2_takes_or_delays"] == 1.0
    assert _one(BRT2_decision="NOT_TAKE_MEDEX").iloc[0]["BRT2_takes_or_delays"] == 0.0


def test_mapping_anchors():
    r = _one(
        BRT1_anchor="ANCHOR_PRACTICAL_GROUNDS",
        BRT2_anchor="ANCHOR_OWN_GP_PERSONAL",
    ).iloc[0]
    assert r["v24_BRT1_anchor"] == "ANCHOR_NONE_SPECIFIED"
    assert r["v24_BRT2_anchor"] == "ANCHOR_DOCTOR"
    r2 = _one(BRT1_anchor="ANCHOR_OWN_WEIGHING", BRT2_anchor="ANCHOR_OWN_WEIGHING").iloc[0]
    assert r2["v24_BRT1_anchor"] == "ANCHOR_BALANCED_CONFLICT"
    assert r2["v24_BRT2_anchor"] == "ANCHOR_BALANCED_CONFLICT"


def test_mapping_emo_union():
    r = _one(BRT2_emo_worry=0, BRT2_emo_prescriber_reactance=1).iloc[0]
    assert r["v24_BRT2_emotional_reaction"] == 1.0


def test_credulous_switch():
    r = _one(
        BRT1_decision="CHANGE_RESTAURANT",
        BRT1_hygiene_risk_reasoning=0,
    ).iloc[0]
    assert r["BRT1_credulous_switch_v3"] == 1.0
    r2 = _one(
        BRT1_decision="CHANGE_RESTAURANT",
        BRT1_check_crowd=1,
    ).iloc[0]
    assert r2["BRT1_credulous_switch_v3"] == 0.0


def test_quality_lay_only():
    r = _one(BRT2_check_lay_content=1).iloc[0]
    assert r["BRT2_research_quality_v3"] == "LAY_ONLY"


def test_reflective_keep_v3_epistemic_hygiene_without_moral():
    r = _one(
        BRT1_decision="KEEP_BOOKING",
        BRT1_hygiene_risk_reasoning=1,
        BRT1_moral_fairness_loyalty=0,
    ).iloc[0]
    assert r["BRT1_reflective_keep_v3_epistemic"] == 1.0


def test_reflective_keep_v3_epistemic_official_or_critics():
    r = _one(BRT1_decision="KEEP_BOOKING", BRT1_check_official=1).iloc[0]
    assert r["BRT1_reflective_keep_v3_epistemic"] == 1.0
    r2 = _one(
        BRT1_decision="KEEP_BOOKING",
        BRT1_anchor="ANCHOR_CRITICS_OFFICIAL",
    ).iloc[0]
    assert r2["BRT1_reflective_keep_v3_epistemic"] == 1.0


def test_reflective_keep_v3_epistemic_none_of_three():
    r = _one(BRT1_decision="KEEP_BOOKING").iloc[0]
    assert r["BRT1_reflective_keep_v3_epistemic"] == 0.0


def test_reflective_keep_v3_epistemic_not_keep():
    r = _one(
        BRT1_decision="CHANGE_RESTAURANT",
        BRT1_hygiene_risk_reasoning=1,
    ).iloc[0]
    assert r["BRT1_reflective_keep_v3_epistemic"] == 0.0


def test_reflective_keep_v3_epistemic_missing_ingredient():
    row = _blank_row(BRT1_decision="KEEP_BOOKING", BRT1_hygiene_risk_reasoning=1)
    row["BRT1_check_official"] = None
    r = compute_all(pd.DataFrame([row])).iloc[0]
    assert pd.isna(r["BRT1_reflective_keep_v3_epistemic"])


def test_composites_missing_ingredient_is_missing():
    row = _blank_row(BRT2_decision="TAKE_MEDEX", BRT2_anchor="ANCHOR_MED_SYSTEM",
                     BRT2_check_professional=1)
    row["BRT2_check_lay_content"] = None       # grid ingredient → grid missing
    row["BRT1_decision"] = None
    row["BRT2_contingency_stated"] = None
    r = compute_all(pd.DataFrame([row])).iloc[0]
    for c in ("BRT2_reflective_accept_v3_plan", "BRT2_reflective_delay_v3_plan",
              "BRT2_search_profile_v3", "BRT1_keeps_booking"):
        assert pd.isna(r[c]), c
    assert r["BRT2_takes_or_delays"] == 1.0


# --------------------------------------------------------------------------- #
# Event counts on the human BRT labels (N=473), recomputed from the 28 coded fields alone.
# --------------------------------------------------------------------------- #
HUMAN_LABELS = Path(os.environ.get("COGSUITE_INPUT_ROOT") or ".") / "reports" / "brt_labels" / "brt_labels_v3_human.csv"
_HUMAN = None


def _human() -> pd.DataFrame:
    global _HUMAN
    if not HUMAN_LABELS.exists():
        pytest.skip(f"human BRT labels not found at {HUMAN_LABELS} (controlled access)")
    if _HUMAN is None:
        raw = pd.read_csv(HUMAN_LABELS, low_memory=False)
        assert len(raw) == 473
        _HUMAN = compute_all(raw[["entity_id"] + CODING_FIELDS].copy())
    return _HUMAN


def _events(col: str) -> int:
    s = _human()[col]
    assert s.notna().all(), f"{col}: unexpected missing values"
    return int(s.sum())


def test_human_events_BRT1_reflective_keep_v3_epistemic():
    assert _events("BRT1_reflective_keep_v3_epistemic") == 272


def test_human_events_BRT2_reflective_accept_v3_plan():
    assert _events("BRT2_reflective_accept_v3_plan") == 70


def test_human_events_BRT2_reflective_delay_v3_plan():
    assert _events("BRT2_reflective_delay_v3_plan") == 158


def test_human_events_BRT2_credulous_adoption_v3():
    assert _events("BRT2_credulous_adoption_v3") == 75


def test_human_events_BRT2_fear_driven_rejection_v3():
    assert _events("BRT2_fear_driven_rejection_v3") == 29


def test_human_events_BRT1_keeps_booking():
    assert _events("BRT1_keeps_booking") == 335


def test_human_events_BRT2_takes_or_delays():
    assert _events("BRT2_takes_or_delays") == 395


def test_human_events_BRT1_anchor_official():
    assert _events("BRT1_anchor_official") == 85


def test_human_events_BRT2_anchor_med_system():
    assert _events("BRT2_anchor_med_system") == 170


def test_human_events_BRT2_take_and_investigate_v3():
    assert _events("BRT2_take_and_investigate_v3") == 134


def test_human_events_BRT2_reflective_accept_v3():
    assert _events("BRT2_reflective_accept_v3") == 58


def test_human_events_BRT2_reflective_delay_v3():
    assert _events("BRT2_reflective_delay_v3") == 84


def test_human_events_BRT2_search_profile_v3():
    vc = _human()["BRT2_search_profile_v3"].value_counts().to_dict()
    assert vc == {"EVIDENCE_EXCLUSIVE": 134, "EVIDENCE_MIXED": 124, "CONSULT_ONLY": 78,
                  "CONSULT_MIXED": 57, "NO_PLAN": 80}


def test_human_events_BRT1_credulous_switch_v3():
    assert _events("BRT1_credulous_switch_v3") == 9


def test_human_events_check_unspecified_sup():
    assert _events("BRT1_check_unspecified_sup") == 8
    assert _events("BRT2_check_unspecified_sup") == 21


def test_derived_contract_exact():
    """compute_all adds exactly DERIVED_V3_COLUMNS plus the v24_* mapping columns."""
    raw = pd.DataFrame([_blank_row()])
    added = [c for c in compute_all(raw).columns if c not in raw.columns]
    assert [c for c in added if not c.startswith("v24_")] == DERIVED_V3_COLUMNS


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all derived tests passed")
