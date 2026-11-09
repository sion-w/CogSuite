"""The BRT outcome families, their paper-facing names and priors, and the agent-side predictor names.

Primary families (BH-FDR within family): each member is (column, paper-facing name, prior direction).
"""
from __future__ import annotations

FAMILIES = [
    {"key": "H1", "title": "Epistemic search: which sources they plan to consult", "members": [
        ("BRT1_check_venue", "Phones the restaurant (primary source; H1a)", "up"),
        ("BRT2_check_institutional_sources", "Consults regulators / journals / official guidance", "up"),
        ("BRT2_check_persons", "Asks friends or family (medication)", "none"),
        ("BRT1_check_official", "Consults hygiene / official records", "up"),
        ("BRT2_check_professional", "Consults a qualified professional", "up"),
        ("BRT1_check_persons", "Asks friends or family (restaurant)", "none"),
        ("BRT2_check_claim_verification", "Verifies the video's cited sources", "up"),
    ]},
    {"key": "H2", "title": "Evidence-weighted decisions", "members": [
        ("BRT1_reflective_keep_v3_epistemic", "Keeps the booking on evidence grounds", "up"),
        ("BRT2_reflective_accept_v3_plan", "Accepts on institutional grounds with a verification plan", "up"),
        ("BRT2_reflective_delay_v3_plan", "Delays with a verification plan and a stated condition", "up"),
        ("BRT2_credulous_adoption_v3", "Acts on the claim with no doubt and no rule", "down"),
        ("BRT2_fear_driven_rejection_v3", "Refuses outright, worry-marked", "down"),
    ]},
    {"key": "H3", "title": "Affective engagement", "members": [
        ("BRT2_emo_worry", "Voices worry about the medication", "up"),
        ("BRT1_emotional_outburst", "Hostile rebuke of the unhelpful friend", "up"),
    ]},
    {"key": "H4", "title": "Fairness and norm-sensitive reasoning", "members": [
        ("BRT1_moral_fairness_loyalty", "Appeals to fairness or loyalty toward Friend B", "up"),
    ]},
]

PRIMARY_MEMBERS = [(fam["key"], c, name, prior) for fam in FAMILIES for c, name, prior in fam["members"]]
PRIMARY_OUTCOMES = [c for _, c, _, _ in PRIMARY_MEMBERS]
OUTCOME_FAMILY = {c: k for k, c, _, _ in PRIMARY_MEMBERS}
PAPER_FACING = {c: name for _, c, name, _ in PRIMARY_MEMBERS}
PRIOR = {c: prior for _, c, _, prior in PRIMARY_MEMBERS}

# Coded fields that each derived primary outcome is built from (formulas in score/brt_v3/derived.py);
# a coded field is its own single ingredient. A derived outcome reports its least reliable ingredient.
BRT2_GRID_INGREDIENTS = [
    "BRT2_check_professional", "BRT2_check_institutional_sources", "BRT2_check_claim_verification",
    "BRT2_check_lay_content", "BRT2_check_persons", "BRT2_check_unspecified",
]
INGREDIENTS: dict[str, list[str]] = {
    "BRT1_reflective_keep_v3_epistemic": [
        "BRT1_decision", "BRT1_hygiene_risk_reasoning", "BRT1_check_official", "BRT1_anchor"],
    "BRT2_reflective_accept_v3_plan": ["BRT2_decision", "BRT2_anchor", "BRT2_conspiracy_ideation"]
    + BRT2_GRID_INGREDIENTS,
    "BRT2_reflective_delay_v3_plan": ["BRT2_decision", "BRT2_contingency_stated"] + BRT2_GRID_INGREDIENTS,
    "BRT2_credulous_adoption_v3": [
        "BRT2_decision", "BRT2_sceptic_grounded", "BRT2_sceptic_groundless", "BRT2_contingency_stated"],
    "BRT2_fear_driven_rejection_v3": ["BRT2_decision", "BRT2_emo_worry", "BRT2_contingency_stated"],
}


def ingredients(column: str) -> list[str]:
    return INGREDIENTS.get(column, [column])


# ---- human-side predictor block ------------------------------------------------------------------------
HUMAN_PREDICTOR = "BCRT_Total"
HUMAN_BLOCK = ("bCRT", ["BCRT_Total"], "BCRT_Total", "raw score (OR per bCRT item)")

# ---- agent-run predictors ------------------------------------------------------------------------------
F2_PREDICTOR = "crt2_level"            # the CRT2 level written into the persona (0-4): steering fidelity
F3_PREDICTOR = "BCRT_Total_realised"   # the run's own bCRT score: association transfer

# ---- thresholds --------------------------------------------------------------------------------------
COMPRESSED_GAP_PP = 20.0   # F1 "compressed": |gap| >= 20 pp and non-overlapping 95% Wilson intervals
UNSTABLE_EVENT_FLOOR = 20  # < 20 events or < 20 non-events: Firth fit, flagged unstable
BOOTSTRAP_N = 2000         # F1 prevalence-gap interval
BOOTSTRAP_SEED = 42

# ---- minimum detectable OR -----------------------------------------------------------------------------
MDOR_NSIM = 1000
MDOR_SEED_ABOVE, MDOR_SEED_BELOW = 42, 43
