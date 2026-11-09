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
# beta_fam is held fixed at the value fitted once at persona construction (OLS of CRT2_Total on
# the CRT2 seen count + Age); it is not refitted on whatever frame is loaded. A refit is
# available as a sensitivity via `fit_crt2_familiarity_beta`. The value is kept at full
# precision: the human lattice scores sit exactly on the persona cutoffs, so a rounded
# coefficient would move scores across a cutoff.
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

    Sensitivity only. This is the fit that ``data/synthetic/05_synth_personas.py``
    (``_apply_crt2_adjustment``) makes when the personas are built; on the persona-construction
    frame it gives ``CRT2_FAM_BETA_FIXED``. The analysis itself always uses the fixed value.
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

    The familiarity adjustment of the persona export (``data/synthetic/05_synth_personas.py``,
    ``_apply_crt2_adjustment``), on which the human persona file and the CRT2 level cutoffs
    (``profiles/crt2_level_cutoffs.yaml``) are built:

      - familiarity = the four CRT2 Experience Checks (1-4) only; bCRT is never
        familiarity-adjusted.
      - beta_fam is the fixed constant ``CRT2_FAM_BETA_FIXED``, applied to every row with a
        CRT2_Total (the Age-missing respondent included), so adjusted scores stay on the
        persona scale.
      - No mean re-centring: the result is a counterfactual level, not a residual.
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
