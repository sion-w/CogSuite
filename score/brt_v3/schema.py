"""BRT v3 field inventory (SCHEMA_SPEC §1). 28 coded fields + evidence spans."""
from __future__ import annotations

from typing import Any

STRING_FIELDS: dict[str, set[str]] = {
    "BRT1_decision": {
        "KEEP_BOOKING",
        "CHANGE_RESTAURANT",
        "CANCEL_OR_POSTPONE",
        "UNCLEAR_OR_CONDITIONAL",
    },
    "BRT1_anchor": {
        "ANCHOR_CRITICS_OFFICIAL",
        "ANCHOR_CROWD",
        "ANCHOR_FRIEND_TESTIMONY",
        "ANCHOR_VENUE_DIRECT",
        "ANCHOR_PRACTICAL_GROUNDS",
        "ANCHOR_OWN_WEIGHING",
        "ANCHOR_UNRESOLVED_CONFLICT",
        "ANCHOR_NONE",
    },
    "BRT2_decision": {
        "TAKE_MEDEX",
        "DELAY_AND_INVESTIGATE",
        "NOT_TAKE_MEDEX",
        "UNCLEAR_OR_CONDITIONAL",
    },
    "BRT2_anchor": {
        "ANCHOR_MED_SYSTEM",
        "ANCHOR_OWN_GP_PERSONAL",
        "ANCHOR_LAY_ONLINE",
        "ANCHOR_PRACTICAL_GROUNDS",
        "ANCHOR_OWN_WEIGHING",
        "ANCHOR_UNRESOLVED_CONFLICT",
        "ANCHOR_NONE",
    },
}

BOOL_FIELDS = [
    "BRT1_check_official",
    "BRT1_check_crowd",
    "BRT1_check_venue",
    "BRT1_check_persons",
    "BRT1_check_unspecified",
    "BRT1_deposit_salience",
    "BRT1_hygiene_risk_reasoning",
    "BRT1_moral_fairness_loyalty",
    "BRT1_moral_extended",
    "BRT1_emotional_outburst",
    "BRT1_stem_misread",
    "BRT2_check_professional",
    "BRT2_check_institutional_sources",
    "BRT2_check_claim_verification",
    "BRT2_check_lay_content",
    "BRT2_check_persons",
    "BRT2_check_unspecified",
    "BRT2_contingency_stated",
    "BRT2_emo_worry",
    "BRT2_emo_prescriber_reactance",
    "BRT2_conspiracy_ideation",
    "BRT2_sceptic_grounded",
    "BRT2_sceptic_groundless",
    "BRT2_stem_misread",
]

CODING_FIELDS = list(STRING_FIELDS.keys()) + BOOL_FIELDS
EVIDENCE_FIELDS = [f"{k}_evidence" for k in CODING_FIELDS]
assert len(CODING_FIELDS) == 28, len(CODING_FIELDS)

SPLIT_DEFAULTS: dict[str, Any] = {
    "BRT1_decision": "UNCLEAR_OR_CONDITIONAL",
    "BRT1_anchor": "ANCHOR_NONE",
    "BRT2_decision": "UNCLEAR_OR_CONDITIONAL",
    "BRT2_anchor": "ANCHOR_NONE",
}
for _k in BOOL_FIELDS:
    SPLIT_DEFAULTS[_k] = 0

PRESENT_STRING: dict[str, set[str]] = {
    "BRT1_decision": {"KEEP_BOOKING", "CHANGE_RESTAURANT", "CANCEL_OR_POSTPONE"},
    "BRT1_anchor": STRING_FIELDS["BRT1_anchor"] - {"ANCHOR_NONE"},
    "BRT2_decision": {"TAKE_MEDEX", "DELAY_AND_INVESTIGATE", "NOT_TAKE_MEDEX"},
    "BRT2_anchor": STRING_FIELDS["BRT2_anchor"] - {"ANCHOR_NONE"},
}

# Dual-pass gates (RECODE_PLAN / D-79 restated for v3)
PRIMARY_FIELDS = ["BRT1_decision", "BRT2_decision", "BRT1_anchor", "BRT2_anchor"]
# Fields that enter a v3 composite (contributing ≥90%)
CONTRIBUTING_FIELDS = [
    "BRT1_decision",
    "BRT1_anchor",
    "BRT1_check_official",
    "BRT1_check_crowd",
    "BRT1_check_venue",
    "BRT1_check_persons",
    "BRT1_check_unspecified",
    "BRT1_hygiene_risk_reasoning",
    "BRT1_moral_fairness_loyalty",
    "BRT2_decision",
    "BRT2_anchor",
    "BRT2_check_professional",
    "BRT2_check_institutional_sources",
    "BRT2_check_claim_verification",
    "BRT2_check_lay_content",
    "BRT2_check_persons",
    "BRT2_check_unspecified",
    "BRT2_contingency_stated",
    "BRT2_emo_worry",
    "BRT2_conspiracy_ideation",
    "BRT2_sceptic_grounded",
    "BRT2_sceptic_groundless",
]

BRT1_CHECK = [
    "BRT1_check_official",
    "BRT1_check_crowd",
    "BRT1_check_venue",
    "BRT1_check_persons",
    "BRT1_check_unspecified",
]
BRT2_CHECK = [
    "BRT2_check_professional",
    "BRT2_check_institutional_sources",
    "BRT2_check_claim_verification",
    "BRT2_check_lay_content",
    "BRT2_check_persons",
    "BRT2_check_unspecified",
]

QUALITY_BRT2 = [
    "EVIDENCE_EXCLUSIVE",
    "EVIDENCE_MIXED",
    "CONSULT_ONLY",
    "CONSULT_MIXED",
    "LAY_ONLY",
    "UNSPECIFIED_ONLY",
    "NONE",
]
QUALITY_BRT1 = [
    "OFFICIAL_EXCLUSIVE",
    "OFFICIAL_MIXED",
    "NONINSTITUTIONAL_ONLY",
    "UNSPECIFIED_ONLY",
    "NONE",
]
INSTITUTIONAL_EXCLUSIVE = {"EVIDENCE_EXCLUSIVE", "CONSULT_ONLY"}
INSTITUTIONAL_MIXED = {"EVIDENCE_MIXED", "CONSULT_MIXED"}
HQR_V24_CELLS = INSTITUTIONAL_EXCLUSIVE | INSTITUTIONAL_MIXED
