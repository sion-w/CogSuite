"""BRT v3 derived variables and the mapping of v3 codes onto the v2.4 category set (v24_* columns).

Every derived BRT variable is computed here. Missing-ingredient rule: any missing input gives a
missing derived value (NaN / <NA>), never 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from schema import (
    BRT1_CHECK,
    BRT2_CHECK,
    HQR_V24_CELLS,
)


def _flag01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _missing_any(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    miss = pd.Series(False, index=df.index)
    for c in cols:
        if c not in df.columns:
            return pd.Series(True, index=df.index)
        s = df[c]
        if s.dtype == object or str(s.dtype).startswith("string"):
            miss = miss | s.isna() | (s.astype("string").str.len() == 0)
        else:
            miss = miss | s.isna()
    return miss


def brt2_research_quality(df: pd.DataFrame) -> pd.Series:
    """Medication research grid (evidence facet x consultation facet, with or without lay sources).

    check_unspecified never qualifies or disqualifies a higher tier.
    """
    inst = _flag01(df["BRT2_check_institutional_sources"])
    claim = _flag01(df["BRT2_check_claim_verification"])
    prof = _flag01(df["BRT2_check_professional"])
    lay_c = _flag01(df["BRT2_check_lay_content"])
    lay_p = _flag01(df["BRT2_check_persons"])
    unspec = _flag01(df["BRT2_check_unspecified"])
    miss = (
        inst.isna() | claim.isna() | prof.isna() | lay_c.isna() | lay_p.isna() | unspec.isna()
    )
    evidence = (inst.fillna(0) >= 1) | (claim.fillna(0) >= 1)
    consult = prof.fillna(0) >= 1
    lay = (lay_c.fillna(0) >= 1) | (lay_p.fillna(0) >= 1)
    only_unspec = (unspec.fillna(0) >= 1) & ~evidence & ~consult & ~lay
    none = ~evidence & ~consult & ~lay & ~(unspec.fillna(0) >= 1)
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out = out.mask(~miss & evidence & ~lay, "EVIDENCE_EXCLUSIVE")
    out = out.mask(~miss & evidence & lay, "EVIDENCE_MIXED")
    out = out.mask(~miss & ~evidence & consult & ~lay, "CONSULT_ONLY")
    out = out.mask(~miss & ~evidence & consult & lay, "CONSULT_MIXED")
    out = out.mask(~miss & ~evidence & ~consult & lay, "LAY_ONLY")
    out = out.mask(~miss & only_unspec, "UNSPECIFIED_ONLY")
    out = out.mask(~miss & none, "NONE")
    return out


def brt1_research_quality(df: pd.DataFrame) -> pd.Series:
    """Restaurant research grid. Venue checks are ignored for the top tier; venue-only is NONINSTITUTIONAL_ONLY."""
    off = _flag01(df["BRT1_check_official"])
    crowd = _flag01(df["BRT1_check_crowd"])
    venue = _flag01(df["BRT1_check_venue"])
    persons = _flag01(df["BRT1_check_persons"])
    unspec = _flag01(df["BRT1_check_unspecified"])
    miss = off.isna() | crowd.isna() | venue.isna() | persons.isna() | unspec.isna()
    official = off.fillna(0) >= 1
    mixers = (crowd.fillna(0) >= 1) | (persons.fillna(0) >= 1)
    venue_on = venue.fillna(0) >= 1
    unspec_on = unspec.fillna(0) >= 1
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out = out.mask(~miss & official & ~mixers, "OFFICIAL_EXCLUSIVE")
    out = out.mask(~miss & official & mixers, "OFFICIAL_MIXED")
    out = out.mask(
        ~miss & ~official & (mixers | venue_on),
        "NONINSTITUTIONAL_ONLY",
    )
    out = out.mask(
        ~miss & ~official & ~mixers & ~venue_on & unspec_on,
        "UNSPECIFIED_ONLY",
    )
    out = out.mask(
        ~miss & ~official & ~mixers & ~venue_on & ~unspec_on,
        "NONE",
    )
    return out


def _bin_from_bool(mask: pd.Series, miss: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=mask.index, dtype=float)
    out = out.mask(~miss, mask.astype(float))
    return out


# Medication grid tiers: "upper four tiers" = an evidence-grade or
# professional plan, with or without lay sources alongside. No ordering is implied.
UPPER_FOUR_TIERS = ["EVIDENCE_EXCLUSIVE", "EVIDENCE_MIXED", "CONSULT_ONLY", "CONSULT_MIXED"]
NO_PLAN_TIERS = ["LAY_ONLY", "UNSPECIFIED_ONLY", "NONE"]

# The derived-column contract: compute_v3_derived() adds exactly these columns (compute_all()
# adds these plus the v24_* columns); test_derived.py asserts it.
DERIVED_V3_COLUMNS = [
    "BRT2_research_quality_v3",
    "BRT1_research_quality_v3",
    "BRT1_reflective_keep_v3_epistemic",
    "BRT2_reflective_accept_v3_plan",
    "BRT2_reflective_delay_v3_plan",
    "BRT2_credulous_adoption_v3",
    "BRT2_fear_driven_rejection_v3",
    "BRT1_keeps_booking",
    "BRT2_takes_or_delays",
    "BRT1_anchor_official",
    "BRT2_anchor_med_system",
    "BRT2_take_and_investigate_v3",
    "BRT2_reflective_accept_v3",
    "BRT2_reflective_delay_v3",
    "BRT2_search_profile_v3",
    "BRT1_credulous_switch_v3",
    "BRT1_check_unspecified_sup",
    "BRT2_check_unspecified_sup",
]


def compute_v3_derived(df: pd.DataFrame) -> pd.DataFrame:
    """The DERIVED_V3_COLUMNS variables. Missing any ingredient gives missing, never 0."""
    out = df.copy()
    out["BRT2_research_quality_v3"] = brt2_research_quality(out)
    out["BRT1_research_quality_v3"] = brt1_research_quality(out)
    q2 = out["BRT2_research_quality_v3"].astype("string")

    dec1 = out["BRT1_decision"].astype("string")
    dec2 = out["BRT2_decision"].astype("string")
    take = dec2 == "TAKE_MEDEX"
    delay = dec2 == "DELAY_AND_INVESTIGATE"
    not_take = dec2 == "NOT_TAKE_MEDEX"
    keep = dec1 == "KEEP_BOOKING"
    change_cancel = dec1.isin(["CHANGE_RESTAURANT", "CANCEL_OR_POSTPONE"])
    med_sys = out["BRT2_anchor"].astype("string") == "ANCHOR_MED_SYSTEM"
    critics = out["BRT1_anchor"].astype("string") == "ANCHOR_CRITICS_OFFICIAL"

    conspiracy = _flag01(out["BRT2_conspiracy_ideation"]) >= 1
    cont = _flag01(out["BRT2_contingency_stated"]) >= 1
    grounded = _flag01(out["BRT2_sceptic_grounded"]) >= 1
    groundless = _flag01(out["BRT2_sceptic_groundless"]) >= 1
    hygiene = _flag01(out["BRT1_hygiene_risk_reasoning"]) >= 1
    official = _flag01(out["BRT1_check_official"]) >= 1
    worry = _flag01(out["BRT2_emo_worry"]) >= 1

    upper4 = q2.isin(UPPER_FOUR_TIERS)
    accept_tiers = q2.isin(["NONE", "EVIDENCE_EXCLUSIVE", "CONSULT_ONLY"])
    delay_tiers = q2.isin(["EVIDENCE_EXCLUSIVE", "CONSULT_ONLY"])

    # KEEP_BOOKING ∧ (hygiene_risk_reasoning ∨ check_official ∨ ANCHOR_CRITICS_OFFICIAL)
    miss = _missing_any(out, ["BRT1_decision", "BRT1_hygiene_risk_reasoning",
                              "BRT1_check_official", "BRT1_anchor"])
    out["BRT1_reflective_keep_v3_epistemic"] = _bin_from_bool(
        keep & (hygiene | official | critics), miss)

    # TAKE_MEDEX ∧ ANCHOR_MED_SYSTEM ∧ grid ∈ upper four tiers ∧ conspiracy_ideation 0
    miss_accept = _missing_any(
        out, ["BRT2_decision", "BRT2_anchor", "BRT2_conspiracy_ideation"]) | q2.isna()
    out["BRT2_reflective_accept_v3_plan"] = _bin_from_bool(
        take & med_sys & upper4 & ~conspiracy, miss_accept)

    # DELAY_AND_INVESTIGATE ∧ grid ∈ upper four tiers ∧ contingency_stated 1
    miss_delay = _missing_any(out, ["BRT2_decision", "BRT2_contingency_stated"]) | q2.isna()
    out["BRT2_reflective_delay_v3_plan"] = _bin_from_bool(delay & upper4 & cont, miss_delay)

    # decision ∈ {DELAY, NOT_TAKE} ∧ sceptic_grounded 0 ∧ sceptic_groundless 0 ∧ contingency 0
    miss = _missing_any(out, ["BRT2_decision", "BRT2_sceptic_grounded",
                              "BRT2_sceptic_groundless", "BRT2_contingency_stated"])
    out["BRT2_credulous_adoption_v3"] = _bin_from_bool(
        (delay | not_take) & ~grounded & ~groundless & ~cont, miss)

    # NOT_TAKE_MEDEX ∧ emo_worry 1 ∧ contingency_stated 0
    miss = _missing_any(out, ["BRT2_decision", "BRT2_emo_worry", "BRT2_contingency_stated"])
    out["BRT2_fear_driven_rejection_v3"] = _bin_from_bool(not_take & worry & ~cont, miss)

    # single-field binaries
    out["BRT1_keeps_booking"] = _bin_from_bool(keep, _missing_any(out, ["BRT1_decision"]))
    out["BRT2_takes_or_delays"] = _bin_from_bool(
        take | delay, _missing_any(out, ["BRT2_decision"]))
    out["BRT1_anchor_official"] = _bin_from_bool(critics, _missing_any(out, ["BRT1_anchor"]))
    out["BRT2_anchor_med_system"] = _bin_from_bool(med_sys, _missing_any(out, ["BRT2_anchor"]))

    # TAKE_MEDEX ∧ any BRT2 check flag
    any_b2_check = False
    for c in BRT2_CHECK:
        any_b2_check = any_b2_check | (_flag01(out[c]).fillna(0) >= 1)
    miss = _missing_any(out, ["BRT2_decision"] + BRT2_CHECK)
    out["BRT2_take_and_investigate_v3"] = _bin_from_bool(take & any_b2_check, miss)

    # strict-tier sensitivity forms of the accept / delay composites
    out["BRT2_reflective_accept_v3"] = _bin_from_bool(
        take & med_sys & accept_tiers & ~conspiracy, miss_accept)
    out["BRT2_reflective_delay_v3"] = _bin_from_bool(delay & delay_tiers & cont, miss_delay)

    # categorical: medication grid collapsed to five unordered categories
    prof = pd.Series(pd.NA, index=out.index, dtype="string")
    for tier in UPPER_FOUR_TIERS:
        prof = prof.mask(q2.isin([tier]), tier)   # isin: a missing grid never matches
    prof = prof.mask(q2.isin(NO_PLAN_TIERS), "NO_PLAN")
    out["BRT2_search_profile_v3"] = prof

    # (CHANGE_RESTAURANT ∨ CANCEL_OR_POSTPONE) ∧ hygiene 0 ∧ no BRT1 check flag (descriptive)
    any_b1_check = False
    for c in BRT1_CHECK:
        any_b1_check = any_b1_check | (_flag01(out[c]).fillna(0) >= 1)
    miss = _missing_any(out, ["BRT1_decision", "BRT1_hygiene_risk_reasoning"] + BRT1_CHECK)
    out["BRT1_credulous_switch_v3"] = _bin_from_bool(
        change_cancel & ~hygiene & ~any_b1_check, miss)

    # Answer-level suppression of check_unspecified (a derived rule; the coded field is unchanged):
    # unspecified_sup = coded unspecified AND no specific channel coded anywhere in the answer.
    # Changes no grid tier or composite (those already ignore unspecified for higher tiers).
    import functools as _ft, operator as _op
    _b1_any = _ft.reduce(_op.or_, [(_flag01(df[c]).fillna(0) >= 1) for c in (
        "BRT1_check_official", "BRT1_check_crowd", "BRT1_check_venue", "BRT1_check_persons")])
    _b2_any = _ft.reduce(_op.or_, [(_flag01(df[c]).fillna(0) >= 1) for c in (
        "BRT2_check_professional", "BRT2_check_institutional_sources",
        "BRT2_check_claim_verification", "BRT2_check_lay_content", "BRT2_check_persons")])
    _u1 = _flag01(df["BRT1_check_unspecified"])
    _u2 = _flag01(df["BRT2_check_unspecified"])
    out["BRT1_check_unspecified_sup"] = ((_u1.fillna(0) >= 1) & ~_b1_any).astype(float)
    out["BRT2_check_unspecified_sup"] = ((_u2.fillna(0) >= 1) & ~_b2_any).astype(float)
    out.loc[_u1.isna(), "BRT1_check_unspecified_sup"] = np.nan
    out.loc[_u2.isna(), "BRT2_check_unspecified_sup"] = np.nan

    return out


def map_v3_to_v24(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic mapping of the v3 codes onto the v2.4 category set and its composites (v24_* columns)."""
    out = df.copy()
    out["v24_BRT1_decision"] = out["BRT1_decision"].astype("string")
    out["v24_BRT2_decision"] = out["BRT2_decision"].astype("string")

    a1 = out["BRT1_anchor"].astype("string")
    out["v24_BRT1_anchor"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["v24_BRT1_anchor"] = out["v24_BRT1_anchor"].mask(
        a1 == "ANCHOR_CRITICS_OFFICIAL", "ANCHOR_FOOD_CRITICS/PRO_REVIEWS"
    )
    out["v24_BRT1_anchor"] = out["v24_BRT1_anchor"].mask(a1 == "ANCHOR_CROWD", "ANCHOR_CROWD_REVIEWS")
    out["v24_BRT1_anchor"] = out["v24_BRT1_anchor"].mask(
        a1 == "ANCHOR_FRIEND_TESTIMONY", "ANCHOR_FRIEND_REPORT"
    )
    out["v24_BRT1_anchor"] = out["v24_BRT1_anchor"].mask(
        a1.isin(["ANCHOR_UNRESOLVED_CONFLICT", "ANCHOR_OWN_WEIGHING"]),
        "ANCHOR_BALANCED_CONFLICT",
    )
    out["v24_BRT1_anchor"] = out["v24_BRT1_anchor"].mask(
        a1.isin(["ANCHOR_NONE", "ANCHOR_PRACTICAL_GROUNDS", "ANCHOR_VENUE_DIRECT"]),
        "ANCHOR_NONE_SPECIFIED",
    )

    a2 = out["BRT2_anchor"].astype("string")
    out["v24_BRT2_anchor"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["v24_BRT2_anchor"] = out["v24_BRT2_anchor"].mask(
        a2.isin(["ANCHOR_MED_SYSTEM", "ANCHOR_OWN_GP_PERSONAL"]), "ANCHOR_DOCTOR"
    )
    out["v24_BRT2_anchor"] = out["v24_BRT2_anchor"].mask(a2 == "ANCHOR_LAY_ONLINE", "ANCHOR_SOCIAL_MEDIA")
    out["v24_BRT2_anchor"] = out["v24_BRT2_anchor"].mask(
        a2.isin(["ANCHOR_UNRESOLVED_CONFLICT", "ANCHOR_OWN_WEIGHING"]),
        "ANCHOR_BALANCED_CONFLICT",
    )
    out["v24_BRT2_anchor"] = out["v24_BRT2_anchor"].mask(
        a2.isin(["ANCHOR_NONE", "ANCHOR_PRACTICAL_GROUNDS"]),
        "ANCHOR_NONE_SPECIFIED",
    )

    q = out["BRT2_research_quality_v3"].astype("string")
    out["v24_BRT2_research_specificity"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["v24_BRT2_research_specificity"] = out["v24_BRT2_research_specificity"].mask(
        q.isin(sorted(HQR_V24_CELLS)), "HIGH_QUALITY_RESEARCH"
    )
    out["v24_BRT2_research_specificity"] = out["v24_BRT2_research_specificity"].mask(
        q == "LAY_ONLY", "SOCIAL_MEDIA_RESEARCH"
    )
    out["v24_BRT2_research_specificity"] = out["v24_BRT2_research_specificity"].mask(
        q == "UNSPECIFIED_ONLY", "VAGUE_RESEARCH"
    )
    out["v24_BRT2_research_specificity"] = out["v24_BRT2_research_specificity"].mask(
        q == "NONE", "NO_ADDITIONAL_RESEARCH"
    )

    worry = _flag01(out["BRT2_emo_worry"])
    react = _flag01(out["BRT2_emo_prescriber_reactance"])
    miss_emo = worry.isna() | react.isna()
    out["v24_BRT2_emotional_reaction"] = np.where(
        miss_emo, np.nan, ((worry.fillna(0) >= 1) | (react.fillna(0) >= 1)).astype(float)
    )
    g = _flag01(out["BRT2_sceptic_grounded"])
    gl = _flag01(out["BRT2_sceptic_groundless"])
    miss_sc = g.isna() | gl.isna()
    out["v24_BRT2_social_media_sceptic"] = np.where(
        miss_sc, np.nan, ((g.fillna(0) >= 1) | (gl.fillna(0) >= 1)).astype(float)
    )
    out["v24_BRT2_high_quality_research"] = np.where(
        q.isna(), np.nan, q.isin(sorted(HQR_V24_CELLS)).astype(float)
    )

    # v2.4 composites on the mapped fields
    dec2 = out["v24_BRT2_decision"].astype("string")
    anc2 = out["v24_BRT2_anchor"].astype("string")
    rs = out["v24_BRT2_research_specificity"].astype("string")
    miss_ra = dec2.isna() | anc2.isna() | rs.isna()
    out["v24_BRT2_reflective_accept"] = np.where(
        miss_ra,
        np.nan,
        (
            (dec2 == "TAKE_MEDEX")
            & anc2.isin(["ANCHOR_DOCTOR", "ANCHOR_BALANCED_CONFLICT"])
            & rs.isin(["NO_ADDITIONAL_RESEARCH", "HIGH_QUALITY_RESEARCH"])
        ).astype(float),
    )
    miss_rd = dec2.isna() | rs.isna()
    out["v24_BRT2_reflective_delay"] = np.where(
        miss_rd,
        np.nan,
        ((dec2 == "DELAY_AND_INVESTIGATE") & (rs == "HIGH_QUALITY_RESEARCH")).astype(float),
    )
    out["v24_BRT2_social_media_reject"] = np.where(
        miss_ra,
        np.nan,
        (
            dec2.isin(["NOT_TAKE_MEDEX", "DELAY_AND_INVESTIGATE"])
            & ((anc2 == "ANCHOR_SOCIAL_MEDIA") | (rs == "SOCIAL_MEDIA_RESEARCH"))
        ).astype(float),
    )
    dec1 = out["v24_BRT1_decision"].astype("string")
    anc1 = out["v24_BRT1_anchor"].astype("string")
    hyg = _flag01(out["BRT1_hygiene_risk_reasoning"])
    mor = _flag01(out["BRT1_moral_fairness_loyalty"])
    miss_rk = dec1.isna() | anc1.isna() | hyg.isna() | mor.isna()
    out["v24_BRT1_reflective_keep"] = np.where(
        miss_rk,
        np.nan,
        (
            (dec1 == "KEEP_BOOKING")
            & ((hyg.fillna(0) >= 1) | anc1.isin(["ANCHOR_FOOD_CRITICS/PRO_REVIEWS", "ANCHOR_BALANCED_CONFLICT"]))
            & (mor.fillna(0) >= 1)
        ).astype(float),
    )
    outburst = _flag01(out["BRT1_emotional_outburst"])
    miss_ecc = dec1.isna() | outburst.isna() | hyg.isna()
    out["v24_BRT1_emotional_change_cancel"] = np.where(
        miss_ecc,
        np.nan,
        (
            dec1.isin(["CHANGE_RESTAURANT", "CANCEL_OR_POSTPONE"])
            & (outburst.fillna(0) >= 1)
            & ~(hyg.fillna(0) >= 1)
        ).astype(float),
    )
    dep = _flag01(out["BRT1_deposit_salience"])
    out["v24_BRT1_deposit_driven"] = np.where(dep.isna(), np.nan, (dep >= 1).astype(float))
    miss_tai = dec2.isna() | rs.isna()
    out["v24_BRT2_take_and_investigate"] = np.where(
        miss_tai,
        np.nan,
        ((dec2 == "TAKE_MEDEX") & (rs != "NO_ADDITIONAL_RESEARCH")).astype(float),
    )
    # carried binaries (identity)
    for col in [
        "BRT1_deposit_salience",
        "BRT1_hygiene_risk_reasoning",
        "BRT1_moral_fairness_loyalty",
        "BRT1_moral_extended",
        "BRT1_emotional_outburst",
    ]:
        out[f"v24_{col}"] = _flag01(out[col])
    return out


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    return map_v3_to_v24(compute_v3_derived(df))
