from __future__ import annotations

import numpy as np
import pandas as pd

from socsim.scoring.crt_categories import add_person_response_categories_from_wide

OPTIONAL_HUMAN_DEMOGRAPHIC_COLUMNS = [
    "Age",
    "sex",
    "Highest education level completed",
    "Ethnicity Simplified",
    "Country",
    "Employment Status",
]
HUMAN_OCCUPATION_SOURCE_COLUMNS = ["Occupation_cat", "UK_occupation", "US_occupation"]
MISSING_TOKENS = {"", "nan", "none", "data_expired", "DATA_EXPIRED"}


def _clean_optional_category(series: pd.Series) -> pd.Series:
    text = series.astype("object")
    cleaned = text.where(text.notna(), np.nan).map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
    return cleaned.mask(cleaned.isin(MISSING_TOKENS), np.nan)


def _first_non_missing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype="object")
    out = _clean_optional_category(df[available[0]])
    for col in available[1:]:
        out = out.where(out.notna(), _clean_optional_category(df[col]))
    return out


CRT2_SEEN_COLS = [
    "Experience Check_1_seen",
    "Experience Check_2_seen",
    "Experience Check_3_seen",
    "Experience Check_4_seen",
]
CRT2_ADJ_COVARIATES = ["Age"]
# CR-23, PI ruling (c) 30 Sep 2026 (`completions/CR14REVIEW_CANDIDATES_2026-09-30.md` C-3):
# beta_fam is held FIXED at the value fitted once at persona construction, not refitted on
# whatever frame is loaded. Refitting on the corrected frame (CR-23 re-scored six "None"
# answers) is reported as a sensitivity only, via `fit_crt2_familiarity_beta`. Amends D-182's
# "beta from OLS" to "beta fitted once at persona construction, then held fixed".
# Full precision (PI, 1 Oct 2026): the ruling's "0.166998188329" is this value rounded to 12 dp.
# The human lattice scores sit exactly on the persona cutoffs, so the rounded number would move
# every score by ~2e-13 and put the 20th-percentile lattice point on the other side of q1_20.
CRT2_FAM_BETA_FIXED = 0.16699818832876753
CRT2_ADJ_NOTE = (
    "CRT2_Total_adj = CRT2_Total - beta_fam * seen_count, where seen_count sums the four CRT2 "
    "Experience Checks (1-4) only and beta_fam = 0.16699818832876753 is held fixed at the value fitted "
    "at persona construction (OLS of CRT2_Total on seen_count + Age, Age-complete rows). "
    "Counterfactual score at zero CRT2 items seen. No mean re-centring; bCRT is never adjusted."
)


def _crt2_seen_count(out: pd.DataFrame) -> pd.Series | None:
    if any(c not in out.columns for c in CRT2_SEEN_COLS):
        return None
    return out[CRT2_SEEN_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=1)


def fit_crt2_familiarity_beta(out: pd.DataFrame) -> float:
    """OLS familiarity coefficient of CRT2_Total on seen_count + Age (Age-complete rows).

    SENSITIVITY ONLY (CR-23 ruling (c)). This is the fit that ``brt_final``'s
    ``05_synth_personas.py::_apply_crt2_adjustment`` realised when the personas were built
    (only ``Age`` resolved from its requested covariates); on that frame it gives
    ``CRT2_FAM_BETA_FIXED``. The analysis never uses a refitted value.
    """
    raw = pd.to_numeric(out["CRT2_Total"], errors="coerce")
    seen = _crt2_seen_count(out)
    if seen is None:
        return float("nan")
    design = {"__y": raw, "__fam": seen}
    for col in CRT2_ADJ_COVARIATES:
        if col in out.columns:
            design[col] = pd.to_numeric(out[col], errors="coerce")
    dat = pd.DataFrame(design).dropna()
    if len(dat) < 3 or float(dat["__fam"].std(ddof=0)) == 0:
        return float("nan")
    blocks = [np.ones((len(dat), 1)), dat["__fam"].to_numpy(float).reshape(-1, 1)]
    for col in CRT2_ADJ_COVARIATES:
        if col in dat.columns:
            centred = dat[col].to_numpy(float)
            blocks.append((centred - centred.mean()).reshape(-1, 1))
    X = np.concatenate(blocks, axis=1)
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ dat["__y"].to_numpy(float))
    return float(beta[1])


def _compute_crt2_adjusted_from_familiarity(
    out: pd.DataFrame, beta_fam: float = CRT2_FAM_BETA_FIXED
) -> pd.Series:
    """Human CRT2 total at the counterfactual of zero CRT2-item familiarity.

    CR-21 (PI ruling 25 Sep 2026, `completions/CRT2ADJ_CANDIDATES_2026-09-25.md` C-1, amending
    D-181 clause 7) made this the ONE familiarity-adjusted CRT2: the persona-export adjustment
    as *realised* by ``brt_final/src/05_synth_personas.py::_apply_crt2_adjustment``, which the
    human persona file and ``crt2_level_cutoffs_v2.yaml`` were both built on.
    CR-23 (ruling (c), 30 Sep 2026) holds its coefficient FIXED at ``CRT2_FAM_BETA_FIXED``.

      - familiarity = the four CRT2 Experience Checks (1-4) ONLY. bCRT is never
        familiarity-adjusted. Including the four bCRT checks is what CR-21 retired.
      - beta_fam is the fixed constant, applied to EVERY row with a CRT2_Total (the Age-missing
        respondent included). It is not refitted: CR-23's re-scoring of six "None" answers
        would otherwise move every person's adjusted score off the persona scale.
      - NO mean re-centring. The result is a counterfactual level, not a residual.
    """
    raw = pd.to_numeric(out["CRT2_Total"], errors="coerce")
    if "CRT2_Total_adj" in out.columns:
        provided = pd.to_numeric(out["CRT2_Total_adj"], errors="coerce")
        if int(provided.notna().sum()) > 0:
            return provided

    seen = _crt2_seen_count(out)
    if seen is None:
        return raw
    return raw - float(beta_fam) * seen.astype(float).fillna(0.0)


def harmonise_human(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "PROLIFIC_PID" in out.columns:
        out["entity_id"] = out["PROLIFIC_PID"].astype(str)
    elif "persona_id" in out.columns:
        out["entity_id"] = out["persona_id"].astype(str)
    else:
        out["entity_id"] = out.index.astype(str)
    out["dataset"] = "human"

    out["BFI_Extraversion_observed"] = pd.to_numeric(out["BFI_Extraversion"], errors="coerce")
    out["BFI_Agreeableness_observed"] = pd.to_numeric(out["BFI_Agreeableness"], errors="coerce")
    out["BFI_Conscientiousness_observed"] = pd.to_numeric(out["BFI_Conscientiousness"], errors="coerce")
    out["BFI_Neuroticism_observed"] = pd.to_numeric(out["BFI_Neuroticism"], errors="coerce")
    out["BFI_Openness_observed"] = pd.to_numeric(out["BFI_Openness"], errors="coerce")
    out["NFC_Total_observed"] = pd.to_numeric(out["NFC_Total"], errors="coerce")
    out["CRT2_Total_raw_observed"] = pd.to_numeric(out["CRT2_Total"], errors="coerce")
    out["CRT2_Total_adj_observed"] = _compute_crt2_adjusted_from_familiarity(out)
    out["CRT2_Total_observed"] = out["CRT2_Total_raw_observed"]
    out["BCRT_Total_observed"] = pd.to_numeric(out["BCRT_Total"], errors="coerce")
    out["CRT_Combined_raw_observed"] = pd.to_numeric(out["CRT_Combined"], errors="coerce")
    out["CRT_Combined_observed"] = out["CRT2_Total_adj_observed"] + out["BCRT_Total_observed"]
    out = add_person_response_categories_from_wide(out)
    out["CRT2_adjustment_note"] = CRT2_ADJ_NOTE
    if "Age" in out.columns:
        out["Age_observed"] = pd.to_numeric(out["Age"], errors="coerce")
        out["Age"] = out["Age_observed"]
    for col in OPTIONAL_HUMAN_DEMOGRAPHIC_COLUMNS:
        if col not in out.columns or col == "Age":
            continue
        out[f"{col}_observed"] = _clean_optional_category(out[col])
        out[col] = out[f"{col}_observed"]
    out["Occupation_cat"] = _first_non_missing(out, HUMAN_OCCUPATION_SOURCE_COLUMNS)
    out["Occupation_cat_observed"] = out["Occupation_cat"]
    return out
