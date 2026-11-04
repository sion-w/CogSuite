from __future__ import annotations

# src/05_synth_personas.py
"""
Synthetic persona generator for BCRT/CRT2/NFC/BFI.

Design goals
------------
1) **Naïve CRT2**: Estimate and remove familiarity inflation from CRT2 using the
   same per-item 'Experience Check_*_seen' fields you use elsewhere, producing
   `CRT2_Total_adj` as a counterfactual naïve score.
2) **Preserve numeric dependence**: Sample numeric scales jointly via a Gaussian
   copula so the synthetic personas reflect the observed cross-scale
   correlations.
3) **Deterministic + strict**: Fail loudly if required inputs are missing; no
   silent fallbacks. Controls exist, but default is strict.
4) **UK occupation harmonisation**: Collapse UK/US occupation inputs plus job
   titles into a single `Occupation_cat`.
5) **Targeted demographic dependence**: Keep the numeric Gaussian-copula block,
   but sample a few key categorical variables conditionally where justified
   (education and employment by age band; occupation by employment, education,
   and age band with hierarchical fallback).
6) **Phase 0 simplification**: Do not require `Country` for persona generation.
   Ethnicity is sampled marginally rather than conditional on country. This is
   intended to reduce unnecessary complexity given the current dataset and keep
   occupation modelling focused on the stronger structural predictors.

Notes
-----
- We intentionally use only NumPy/Pandas (no SciPy) to keep deps minimal.
- Personas use *totals* (CRT2 adjusted, BCRT raw, NFC, BFI) because totals are
  interpretable for LLM prompting. IRT theta can be added later as an optional
  column if you decide to integrate the IRT pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

AGE_BAND_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_BAND_BINS = [-np.inf, 24, 34, 44, 54, 64, np.inf]


def _age_to_band(age: object) -> Optional[str]:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    if np.isnan(a):
        return None
    for lo, hi, label in zip(AGE_BAND_BINS[:-1], AGE_BAND_BINS[1:], AGE_BAND_LABELS):
        if lo < a <= hi:
            return label
    return AGE_BAND_LABELS[-1]


def _series_with_missing(s: pd.Series, missing_label: str = "Missing") -> pd.Series:
    s = s.copy()
    return s.where(~s.isna(), missing_label)

# --------------------------------------------------------------------------
# Helper: value probabilities excluding missing values
# --------------------------------------------------------------------------
def _value_probs_excluding_missing(
    s: pd.Series,
    missing_label: str = "Missing",
) -> pd.Series:
    """
    Return normalised marginal probabilities excluding missing values.
    This is used for persona fields where we do not want to generate a literal
    'Missing' category for synthetic personas.
    """
    observed = s.dropna()
    if observed.empty:
        raise RuntimeError("Cannot build probabilities from an entirely missing series.")
    probs = observed.value_counts(normalize=True)
    if missing_label in probs.index:
        probs = probs.drop(index=missing_label)
    if probs.empty:
        raise RuntimeError("No non-missing categories available after excluding missing values.")
    return probs


def _draw_from_probs(
    probs: pd.Series,
    rng: np.random.Generator,
    normalize_index: bool = False,
) -> str:
    p = probs.astype(float)
    p = p / p.sum()
    cats = p.index.to_list()
    if normalize_index:
        cats = [None if x == "Missing" else x for x in cats]
    return rng.choice(cats, p=p.to_numpy())


def _conditional_probs(
    df: pd.DataFrame,
    target: str,
    givens: List[str],
    missing_label: str = "Missing",
) -> Dict[tuple, pd.Series]:
    work = df[givens + [target]].copy()
    for c in givens + [target]:
        work[c] = _series_with_missing(work[c], missing_label=missing_label)
    counts = work.groupby(givens + [target]).size().rename("n").reset_index()
    out: Dict[tuple, pd.Series] = {}
    for key, sub in counts.groupby(givens, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        probs = sub.set_index(target)["n"].astype(float)
        probs = probs / probs.sum()
        out[key] = probs
    return out

# --------------------------------------------------------------------------
# Helper: conditional probabilities excluding missing for the target
# --------------------------------------------------------------------------
def _conditional_probs_excluding_missing_target(
    df: pd.DataFrame,
    target: str,
    givens: List[str],
    missing_label: str = "Missing",
) -> Dict[tuple, pd.Series]:
    """
    Conditional probabilities where the target is sampled only from observed
    (non-missing) categories. Givens still get a missing sentinel so lookup keys
    remain well-defined.
    """
    work = df[givens + [target]].copy()
    for c in givens:
        work[c] = _series_with_missing(work[c], missing_label=missing_label)
    work = work[work[target].notna()].copy()
    work = work[work[target] != missing_label].copy()
    if work.empty:
        raise RuntimeError(
            f"Cannot build conditional probabilities for '{target}': all target values are missing."
        )
    counts = work.groupby(givens + [target]).size().rename("n").reset_index()
    out: Dict[tuple, pd.Series] = {}
    for key, sub in counts.groupby(givens, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        probs = sub.set_index(target)["n"].astype(float)
        probs = probs / probs.sum()
        out[key] = probs
    return out


def _lookup_conditional(
    key: tuple,
    table: Dict[tuple, pd.Series],
) -> Optional[pd.Series]:
    return table.get(key)

from utils import load_cfg


# -----------------------------------------------------------------------------
# Config & data
# -----------------------------------------------------------------------------
cfg = load_cfg()
processed = Path(cfg["paths"]["processed"])
reports_tabs = Path(cfg["paths"]["reports_tabs"])

DF_PATH = processed / "scored_person.csv"   # input tabular data


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# Column sets
# -----------------------------------------------------------------------------
DEMO_KEEP = [
    "Employment Status",
    "Occupation_cat",
    "sex",
    "Highest education level completed",
    "Ethnicity Simplified",
]

STUDENT_STATUSES = {"Student"}
EMPLOYED_STATUSES = {"Working full-time", "Working part-time"}
NON_WORKING_STATUSES = {
    "A homemaker or stay-at-home parent",
    "Retired",
    "Unemployed and looking for work",
    "Other",
}

US_TO_UK_OCC: Dict[str, str] = {
    "Architecture and Engineering": "Associate Professional and Technical Occupations",
    "Arts, Design, Entertainment, Sports, and Media": "Associate Professional and Technical Occupations",
    "Building and Grounds Cleaning and Maintenance": "Elementary Occupations",
    "Business and Financial Operations": "Associate Professional and Technical Occupations",
    "Community and Social Service": "Caring, Leisure and Other Service Occupations",
    "Computer and Mathematical": "Professional Occupations",
    "Construction and Extraction": "Skilled Trades Occupations",
    "Education, Training, and Library": "Professional Occupations",
    "Farming, Fishing, and Forestry": "Skilled Trades Occupations",
    "Food Preparation and Serving Related": "Elementary Occupations",
    "Healthcare Practitioners and Technical": "Professional Occupations",
    "Healthcare Support": "Caring, Leisure and Other Service Occupations",
    "Installation, Maintenance, and Repair": "Skilled Trades Occupations",
    "Legal Occupations": "Professional Occupations",
    "Life, Physical, and Social Science": "Professional Occupations",
    "Management Occupations": "Managers, Directors and Senior Officials",
    "Office and Administrative Support": "Administrative and Secretarial Occupations",
    "Other": "Other",
    "Personal Care and Service": "Caring, Leisure and Other Service Occupations",
    "Production Occupations": "Process, Plant and Machine Operatives",
    "Protective Service": "Associate Professional and Technical Occupations",
    "Sales and Related": "Sales and Customer Service Occupations",
    "Transportation and Material Moving": "Process, Plant and Machine Operatives",
}

OTHER_TITLE_TO_UK: Dict[str, str] = {
    "a senior it manager": "Managers, Directors and Senior Officials",
    "ai evaluator": "Associate Professional and Technical Occupations",
    "agricultural worker": "Skilled Trades Occupations",
    "artist": "Associate Professional and Technical Occupations",
    "assistant engineer": "Associate Professional and Technical Occupations",
    "assistant manager - studio": "Managers, Directors and Senior Officials",
    "assistant pastor": "Professional Occupations",
    "author": "Associate Professional and Technical Occupations",
    "business owner": "Managers, Directors and Senior Officials",
    "business services individual contributor": "Associate Professional and Technical Occupations",
    "care assistant": "Caring, Leisure and Other Service Occupations",
    "carpenter supervisor": "Skilled Trades Occupations",
    "cleaner": "Elementary Occupations",
    "consultant": "Professional Occupations",
    "currently unemployed for illness, but job title is registered veterinary nurse (rvn)": "Associate Professional and Technical Occupations",
    "custodial": "Elementary Occupations",
    "data analyst": "Associate Professional and Technical Occupations",
    "delivery driver": "Process, Plant and Machine Operatives",
    "dog groomer": "Caring, Leisure and Other Service Occupations",
    "domestic helper/babysitter": "Caring, Leisure and Other Service Occupations",
    "factory worker": "Process, Plant and Machine Operatives",
    "freelance 3d graphics": "Associate Professional and Technical Occupations",
    "freelance artist": "Associate Professional and Technical Occupations",
    "freelance software developer": "Professional Occupations",
    "freelancer - ux testing": "Associate Professional and Technical Occupations",
    "futures analyst": "Professional Occupations",
    "home based computer upgrade/repair business": "Associate Professional and Technical Occupations",
    "housekeeping assistant": "Elementary Occupations",
    "human resource manager": "Managers, Directors and Senior Officials",
    "kitchen staff": "Elementary Occupations",
    "lead brewer": "Process, Plant and Machine Operatives",
    "lecturer": "Professional Occupations",
    "medical doctor and an orthopaedics surgeon": "Professional Occupations",
    "minister of religion (retired)": "Professional Occupations",
    "project planner": "Associate Professional and Technical Occupations",
    "proof reader": "Associate Professional and Technical Occupations",
    "recovery worker - health and social care": "Caring, Leisure and Other Service Occupations",
    "researcher manager": "Managers, Directors and Senior Officials",
    "retail merchandiser and auditor (now retired)": "Sales and Customer Service Occupations",
    "self employed - blogger": "Associate Professional and Technical Occupations",
    "self-employed landscaper": "Skilled Trades Occupations",
    "senior management analyst": "Managers, Directors and Senior Officials",
    "sensory analyst (food industry)": "Associate Professional and Technical Occupations",
    "software developer": "Professional Occupations",
    "teacher": "Professional Occupations",
    "teaching assistant": "Associate Professional and Technical Occupations",
    "video editor": "Associate Professional and Technical Occupations",
    "warehouse worker, but currently self employed": "Process, Plant and Machine Operatives",
    "canvasser (but educator was usual profession)": "Sales and Customer Service Occupations",
    "restore associate": "Sales and Customer Service Occupations",
    "travel consultant": "Sales and Customer Service Occupations",
    "web designer": "Associate Professional and Technical Occupations",
}

# Primary scale totals used for persona traits (numeric)
SCALE_BASE = [
    "Age",             # demographic numeric
    "CRT2_Total",       # will be adjusted to CRT2_Total_adj if familiarity present
    "BCRT_Total",
    "NFC_Total",
]

def _bfi_cols(df: pd.DataFrame) -> List[str]:
    """All Big-Five short-form columns (treated as numeric scales)."""
    return [c for c in df.columns if c.startswith("BFI_") and c != "BFI_Attn_Pass"]


def _normalize_status(status: Optional[str]) -> Optional[str]:
    if isinstance(status, str):
        status = status.strip()
        return status or None
    return None


def _status_group(status: Optional[str]) -> str:
    norm = _normalize_status(status)
    if norm in STUDENT_STATUSES:
        return "student"
    if norm in EMPLOYED_STATUSES:
        return "employed"
    if norm in NON_WORKING_STATUSES:
        return "non_working"
    return "non_working"


def _normalize_title(title: Optional[str]) -> Optional[str]:
    if isinstance(title, str):
        title = title.strip().lower()
        return title or None
    return None


def _harmonize_occupations(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Build Occupation_harmonised / Occupation_cat based on UK, US, and title detail.
    Returns a copy plus summary statistics for reporting.
    """
    if "Employment Status" not in df.columns:
        raise RuntimeError("Missing required column: 'Employment Status'")
    if "Job Title" not in df.columns:
        raise RuntimeError("Missing required column: 'Job Title'")

    df = df.copy()
    status_norms: List[Optional[str]] = []
    base_occ: List[str] = []
    job_titles: List[Optional[str]] = []
    n_from_us_map = 0

    for _, row in df.iterrows():
        status_norm = _normalize_status(row.get("Employment Status"))
        status_norms.append(status_norm)
        job_titles.append(row.get("Job Title"))

        if status_norm in STUDENT_STATUSES:
            base_occ.append("Student")
            continue

        uk_occ = row.get("UK_occupation")
        if isinstance(uk_occ, str) and uk_occ and uk_occ != "Other":
            base_occ.append(uk_occ)
            continue

        us_occ = row.get("US_occupation")
        mapped_us = US_TO_UK_OCC.get(us_occ) if isinstance(us_occ, str) else None
        if mapped_us:
            base_occ.append(mapped_us)
            n_from_us_map += 1
            continue

        base_occ.append("Other")

    final_occ = base_occ.copy()
    n_from_title_map = 0
    for idx, occ in enumerate(base_occ):
        if occ != "Other":
            continue
        job_norm = _normalize_title(job_titles[idx])
        if not job_norm:
            continue
        mapped = OTHER_TITLE_TO_UK.get(job_norm)
        if mapped and mapped != "Other":
            final_occ[idx] = mapped
            n_from_title_map += 1

    df["Occupation_harmonised"] = final_occ
    df["Occupation_cat"] = final_occ

    n_real_other = sum(
        1
        for occ, status_norm in zip(final_occ, status_norms)
        if occ == "Other" and status_norm not in STUDENT_STATUSES
    )

    summary = {
        "n_total": len(df),
        "n_from_us_map": n_from_us_map,
        "n_from_title_map": n_from_title_map,
        "n_real_other": n_real_other,
    }
    return df, summary


def _print_occupation_summary(df: pd.DataFrame, summary: Dict[str, int]) -> None:
    print("[05_synth_personas] Occupation harmonisation summary:")
    print(f"  Total rows: N={summary['n_total']}")
    print(f"  From semantic US→UK mapping: {summary['n_from_us_map']}")
    print(f"  From semantic Job Title clean-up: {summary['n_from_title_map']}")
    print(f"  Remaining true 'Other': {summary['n_real_other']}")
    print("  Final Occupation_cat distribution:")
    dist = df["Occupation_cat"].value_counts()
    for cat, cnt in dist.items():
        print(f"    {cat}: {cnt}")


# -----------------------------------------------------------------------------
# Familiarity adjustment (CRT2 only)
# -----------------------------------------------------------------------------
CRT2_SEEN_COLS = [
    "Experience Check_1_seen",
    "Experience Check_2_seen",
    "Experience Check_3_seen",
    "Experience Check_4_seen",
]

def _ols_adjust_counterfactual(
    y: pd.Series, fam: pd.Series, covars: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Counterfactual adjustment via OLS: y ~ fam + covars
    Returns y_adj = y - beta_fam * fam, giving an estimate of y if fam=0 for each row.
    """
    dat = pd.concat({"y": y, "fam": fam}, axis=1)
    if covars is not None and not covars.empty:
        dat = pd.concat([dat, covars], axis=1)
    dat = dat.dropna()
    if dat.empty:
        raise RuntimeError("Counterfactual adjustment failed: no complete rows for OLS.")

    yv = dat["y"].astype(float).values
    famv = dat["fam"].astype(float).values.reshape(-1, 1)

    Xparts = [np.ones((len(dat), 1)), famv]
    if covars is not None and not covars.empty:
        cov = covars.loc[dat.index]
        blocks = []
        for c in cov.columns:
            s = cov[c]
            if pd.api.types.is_numeric_dtype(s):
                blocks.append((s.astype(float) - s.astype(float).mean()).values.reshape(-1, 1))
            else:
                dummies = pd.get_dummies(s.astype("category"), drop_first=True)
                if dummies.shape[1] == 0:
                    continue
                blocks.append(dummies.values)
        if blocks:
            Xparts.append(np.concatenate(blocks, axis=1))
    X = np.concatenate(Xparts, axis=1)

    beta = np.linalg.pinv(X.T @ X) @ (X.T @ yv)
    if len(beta) < 2:
        raise RuntimeError("Counterfactual adjustment failed: familiarity coefficient not identified.")
    beta_fam = float(beta[1])

    fam_full = fam.astype(float).reindex(y.index).fillna(0.0).values
    y_adj = y.astype(float) - beta_fam * fam_full
    return pd.Series(y_adj, index=y.index, name=(y.name + "_adj") if y.name else "y_adj")


def _apply_crt2_adjustment(df: pd.DataFrame, require_familiarity: bool = True) -> pd.DataFrame:
    """
    Create CRT2_Total_adj using per-item seen flags. If `require_familiarity` is True,
    raise when needed columns are absent.
    """
    missing = [c for c in CRT2_SEEN_COLS if c not in df.columns]
    if missing:
        msg = f"Missing CRT2 familiarity columns: {missing}"
        if require_familiarity:
            raise RuntimeError(msg)
        else:
            print("[05_synth_personas] Warning:", msg)
            return df

    if "CRT2_Total" not in df.columns:
        raise RuntimeError("Missing required column: 'CRT2_Total'")

    df = df.copy()
    df["CRT2_SeenCount"] = df[CRT2_SEEN_COLS].sum(axis=1)

    # BRT-J1 (B-24(a)): the covariate list is pinned to the one the personas were actually
    # built with. The old "if present" list ["Age", "Gender", "Education"] only ever resolved
    # Age (the frame names the others 'sex' and 'Highest education level completed'), so this
    # is no numeric change; it stops a column rename from silently changing the definition.
    # The resulting coefficient is held fixed for analysis in crt2_familiarity.py.
    if "Age" not in df.columns:
        raise RuntimeError("Missing required covariate for the CRT2 familiarity fit: 'Age'")
    covs = df[["Age"]]

    df["CRT2_Total_adj"] = _ols_adjust_counterfactual(df["CRT2_Total"], df["CRT2_SeenCount"], covs)
    return df


# =============================================================================
# CR-08 STEP 3 — D-80 synthetic-modelling inclusion rule (LOCKED, PI confirmed
# 2026-07-29, cognition-suite-sim/docs/DECISIONS.md). MARKED BLOCK: do not
# change this rule's criteria without a new decision superseding D-80.
#
# Copula-modelling set = respondents complete on ALL modelled numeric
# variables:
#   - all four CRT2 items scored (crt2_q1..q4 each individually non-null —
#     checked item-by-item, not via CRT2_Total/CRT2_Total_adj alone, because
#     CRT2_Total is a min_count=1 sum and can be non-null from a partial
#     3-of-4 response; D-80's text calls this out explicitly for CRT2)
#   - bCRT scored (BCRT_Total non-null)
#   - NFC scored (NFC_Total non-null)
#   - all five BFI traits scored (each non-null)
# Attention-check failures (BFI_Attn_Pass == 0) are RETAINED — never an
# exclusion criterion here, consistent with the preprint's covariate-not-
# exclusion treatment of attention checks; counted and logged, not filtered.
# Demographics: category-level handling as before — missing education/
# occupation are NOT a modelling-set exclusion and are NOT imputed; they flow
# through to the existing "Missing"-category marginal/conditional sampling in
# _sample_demos/_value_probs_excluding_missing untouched by this rule.
# The content-invalid respondent (67cf6dfb) stays excluded — it is absent
# from scored_person.csv upstream (00_load_qc.py content exclusion) and never
# reaches this function.
# =============================================================================
INCLUSION_LOG_PATH = Path("reports/synthetic/INCLUSION_LOG.md")


def _apply_d80_inclusion_rule(df: pd.DataFrame, log_path: Path = INCLUSION_LOG_PATH) -> pd.DataFrame:
    """D-80 copula-modelling-set filter. Returns the included rows and writes
    a per-rule exclusion-count log. Does not mutate `df` in place."""
    crt2_item_cols = [f"crt2_q{i}_final_correct" for i in range(1, 5)]
    missing_crt2_items = [c for c in crt2_item_cols if c not in df.columns]
    if missing_crt2_items:
        raise RuntimeError(f"D-80 inclusion rule: missing CRT2 item columns {missing_crt2_items}")
    for req in ("BCRT_Total", "NFC_Total"):
        if req not in df.columns:
            raise RuntimeError(f"D-80 inclusion rule: missing required column '{req}'")
    bfi_cols = _bfi_cols(df)
    if len(bfi_cols) != 5:
        raise RuntimeError(f"D-80 inclusion rule: expected 5 BFI trait columns, found {len(bfi_cols)}: {bfi_cols}")

    rule_crt2 = df[crt2_item_cols].notna().all(axis=1)
    rule_bcrt = df["BCRT_Total"].notna()
    rule_nfc = df["NFC_Total"].notna()
    rule_bfi = df[bfi_cols].notna().all(axis=1)
    included_mask = rule_crt2 & rule_bcrt & rule_nfc & rule_bfi

    n_total = len(df)
    n_included = int(included_mask.sum())
    fail_crt2 = int((~rule_crt2).sum())
    fail_bcrt = int((~rule_bcrt).sum())
    fail_nfc = int((~rule_nfc).sum())
    fail_bfi = int((~rule_bfi).sum())
    n_excluded = int((~included_mask).sum())

    attn_col = "BFI_Attn_Pass"
    n_attn_fail_included = None
    n_attn_fail_total = None
    if attn_col in df.columns:
        attn_numeric = pd.to_numeric(df[attn_col], errors="coerce")
        n_attn_fail_total = int((attn_numeric == 0).sum())
        n_attn_fail_included = int((attn_numeric[included_mask] == 0).sum())

    lines = [
        "# CR-08 STEP 3 — D-80 synthetic-modelling inclusion rule: realised N",
        "",
        "LOCKED per cognition-suite-sim/docs/DECISIONS.md D-80 (2026-07-29, PI confirmed).",
        "Copula-modelling set = complete on all four CRT2 items, bCRT, NFC, all five BFI traits.",
        "Attention-check failures retained (not an exclusion criterion). Demographics handled",
        "at category level (no drop, no imputation into the modelling set).",
        "",
        f"- Input frame (post-00_load_qc, post-01_score, content-excluded 67cf6dfb already absent): N={n_total}",
        f"- **Realised included N = {n_included}** (D-80 expected N=471 of the 473 analytic frame; ",
        "  actual is logged here per D-80's own instruction to report the realised figure).",
        f"- Excluded (fails >=1 rule): {n_excluded}",
        "",
        "## Per-rule exclusion counts (not mutually exclusive; a respondent can fail >1 rule)",
        "",
        f"- Fails CRT2 completeness (>=1 of 4 items not individually scored): {fail_crt2}",
        f"- Fails bCRT completeness (BCRT_Total null): {fail_bcrt}",
        f"- Fails NFC completeness (NFC_Total null): {fail_nfc}",
        f"- Fails BFI completeness (>=1 of 5 traits null): {fail_bfi}",
        "",
        "## Attention-check flag (retained, not filtered)",
        "",
        f"- Attention-check failures in the input frame (BFI_Attn_Pass==0): {n_attn_fail_total}",
        f"- Attention-check failures retained in the included modelling set: {n_attn_fail_included}",
        "",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[05_synth_personas][D-80] included={n_included}/{n_total}; "
          f"fail_crt2={fail_crt2} fail_bcrt={fail_bcrt} fail_nfc={fail_nfc} fail_bfi={fail_bfi}; "
          f"log -> {log_path}")

    return df.loc[included_mask].reset_index(drop=True)
# =============================================================================
# END CR-08 STEP 3 marked block
# =============================================================================


# -----------------------------------------------------------------------------
# Gaussian copula utilities
# -----------------------------------------------------------------------------
def _rank_gaussianize(x: np.ndarray) -> np.ndarray:
    """Map values to ~N(0,1) by rank → uniform → inverse-normal."""
    n = len(x)
    ranks = pd.Series(x).rank(method="average").to_numpy()
    u = (ranks - 0.5) / n
    u = np.clip(u, 1e-6, 1 - 1e-6)
    # inverse-normal via erfinv
    return np.sqrt(2.0) * _erfinv(2.0 * u - 1.0)

def _inv_rank_gaussianize(z: np.ndarray, ref: np.ndarray, *, discrete: bool = False) -> np.ndarray:
    """
    Map standard normal samples back to an empirical marginal.

    - For discrete/count-like variables, preserve observed support via stepwise lookup.
    - For continuous/adjusted variables, use linear interpolation over the empirical
      quantile function to avoid a heavily stepped inverse CDF.
    """
    q = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
    q = np.clip(q, 1e-6, 1 - 1e-6)
    ref_sorted = np.sort(np.asarray(ref, dtype=float))
    if discrete:
        idx = np.floor(q * (len(ref_sorted) - 1)).astype(int)
        return ref_sorted[idx]
    grid = np.linspace(0.0, 1.0, len(ref_sorted))
    return np.interp(q, grid, ref_sorted)

# Fast approximations to erf / erfinv (keeps dependencies light)
def _erf(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * x)
    tau = t * np.exp(
        -x * x
        - 1.26551223
        + 1.00002368 * t
        + 0.37409196 * t**2
        + 0.09678418 * t**3
        - 0.18628806 * t**4
        + 0.27886807 * t**5
        - 1.13520398 * t**6
        + 1.48851587 * t**7
        - 0.82215223 * t**8
        + 0.17087277 * t**9
    )
    return sign * (1.0 - tau)

def _erfinv(y: np.ndarray) -> np.ndarray:
    a = 0.147
    y = np.clip(y, -0.999999, 0.999999)
    ln = np.log(1 - y * y)
    s = (2 / (np.pi * a) + ln / 2)
    inside = s * s - ln / a
    x = np.sign(y) * np.sqrt(np.maximum(0.0, np.sqrt(np.maximum(0.0, inside)) - s))
    # one Newton step
    f = _erf(x) - y
    fp = 2.0 / np.sqrt(np.pi) * np.exp(-x * x)
    return x - f / np.maximum(fp, 1e-12)


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
@dataclass
class PersonaOptions:
    n: int = 1000
    seed: int = 42
    require_familiarity: bool = True  # raise if CRT2 familiarity fields missing
    save_path: Optional[str | Path] = None

def _build_column_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    missing_demo = [c for c in DEMO_KEEP if c not in df.columns]
    if missing_demo:
        raise RuntimeError(f"Missing required demographic columns for personas: {missing_demo}")
    if "Age" not in df.columns:
        raise RuntimeError("Missing required numeric column for personas: 'Age'")

    demo = [c for c in DEMO_KEEP if c in df.columns]
    scales = [c for c in SCALE_BASE if c in df.columns] + _bfi_cols(df)

    # Prefer adjusted CRT2 if present
    if "CRT2_Total_adj" in df.columns and "CRT2_Total" in scales:
        idx = scales.index("CRT2_Total")
        scales[idx] = "CRT2_Total_adj"
    return demo, scales

def _sample_demos(
    df: pd.DataFrame,
    demo_cols: List[str],
    sampled_age: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample demographics with a few targeted dependencies only.

    Structure:
    - Ethnicity Simplified: marginal
    - Highest education level completed | AgeBand
    - Employment Status | AgeBand
    - Occupation_cat | Employment Status + Education + AgeBand, with hierarchical fallback
    - sex: marginal
    """
    required = {
        "Employment Status",
        "Occupation_cat",
        "sex",
        "Highest education level completed",
        "Ethnicity Simplified",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing demographic columns for sampling: {sorted(missing)}")

    work = df.copy()
    work["AgeBand"] = work["Age"].apply(_age_to_band)
    out = pd.DataFrame(index=range(n))
    sampled_age_band = pd.Series(sampled_age).apply(_age_to_band).tolist()

    # 1) Ethnicity marginal
    eth_probs = _value_probs_excluding_missing(work["Ethnicity Simplified"])
    out["Ethnicity Simplified"] = [
        _draw_from_probs(eth_probs, rng, normalize_index=True) for _ in range(n)
    ]

    # 2) Education | AgeBand, fallback to marginal
    edu_marg = _value_probs_excluding_missing(work["Highest education level completed"])
    edu_by_ageband = _conditional_probs_excluding_missing_target(
        work,
        "Highest education level completed",
        ["AgeBand"],
    )
    education = []
    for band in sampled_age_band:
        key = (("Missing" if band is None else band),)
        probs = _lookup_conditional(key, edu_by_ageband)
        if probs is None:
            probs = edu_marg
        education.append(_draw_from_probs(probs, rng, normalize_index=True))
    out["Highest education level completed"] = education

    # 3) Employment | AgeBand, fallback to marginal
    emp_marg = _value_probs_excluding_missing(work["Employment Status"])
    emp_by_ageband = _conditional_probs_excluding_missing_target(
        work,
        "Employment Status",
        ["AgeBand"],
    )
    statuses = []
    for band in sampled_age_band:
        key = (("Missing" if band is None else band),)
        probs = _lookup_conditional(key, emp_by_ageband)
        if probs is None:
            probs = emp_marg
        statuses.append(_draw_from_probs(probs, rng, normalize_index=True))
    out["Employment Status"] = statuses

    # 4) sex marginal
    sex_probs = _value_probs_excluding_missing(work["sex"])
    out["sex"] = [
        _draw_from_probs(sex_probs, rng, normalize_index=True) for _ in range(n)
    ]

    # 5) Occupation | Employment Status + Education + AgeBand, with hierarchical fallback
    occ_marg = _value_probs_excluding_missing(work["Occupation_cat"])
    occ_by_status = _conditional_probs_excluding_missing_target(work, "Occupation_cat", ["Employment Status"])
    occ_by_status_edu = _conditional_probs_excluding_missing_target(
        work,
        "Occupation_cat",
        ["Employment Status", "Highest education level completed"],
    )
    occ_by_status_edu_age = _conditional_probs_excluding_missing_target(
        work,
        "Occupation_cat",
        ["Employment Status", "Highest education level completed", "AgeBand"],
    )

    occ_samples: List[Optional[str]] = []
    for status, edu, band in zip(out["Employment Status"], out["Highest education level completed"], sampled_age_band):
        norm_status = _normalize_status(status)
        if norm_status in STUDENT_STATUSES:
            occ_samples.append("Student")
            continue

        status_key = "Missing" if status is None else status
        edu_key = "Missing" if edu is None else edu
        band_key = "Missing" if band is None else band

        probs = _lookup_conditional((status_key, edu_key, band_key), occ_by_status_edu_age)
        if probs is None:
            probs = _lookup_conditional((status_key, edu_key), occ_by_status_edu)
        if probs is None:
            probs = _lookup_conditional((status_key,), occ_by_status)
        if probs is None:
            probs = occ_marg
        occ_samples.append(_draw_from_probs(probs, rng, normalize_index=True))

    out["Occupation_cat"] = occ_samples
    return out[demo_cols]

def _sample_scales_copula(df: pd.DataFrame, scale_cols: List[str], n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Gaussian copula sampler for numeric scales.
    - Ranks → Z
    - Estimate Z correlation
    - Sample MVN
    - Inverse map each column back to its empirical marginal
    """
    num_cols = []
    for c in scale_cols:
        if c not in df.columns:
            raise RuntimeError(f"Required scale column missing: '{c}'")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise RuntimeError(f"Scale column '{c}' must be numeric.")
        num_cols.append(c)

    M = df[num_cols].dropna()
    if len(M) < 20:
        raise RuntimeError("Not enough complete rows to estimate copula correlation (need >= 20).")

    # rank-Gaussianize
    Z_cols = []
    refs = {}
    for c in num_cols:
        vals = M[c].to_numpy()
        Z_cols.append(_rank_gaussianize(vals).reshape(-1, 1))
        refs[c] = vals
    Z = np.concatenate(Z_cols, axis=1)

    # correlation and PSD enforcement
    Sigma = np.corrcoef(Z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.clip(eigvals, 1e-6, None)
    Sigma_psd = (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(float)

    # Cholesky sample
    L = np.linalg.cholesky(Sigma_psd)
    z = rng.standard_normal(size=(n, len(num_cols))) @ L.T

    # inverse-map to original marginals
    discrete_support_cols = {"BCRT_Total", "CRT2_Total"}
    X_synth = {}
    for j, c in enumerate(num_cols):
        X_synth[c] = _inv_rank_gaussianize(
            z[:, j],
            refs[c],
            discrete=(c in discrete_support_cols),
        )
    return pd.DataFrame(X_synth)

def generate_personas(
    n: int,
    seed: int = 42,
    save_path: Optional[str | Path] = None,
    require_familiarity: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic personas.
    - Loads `scored_person.csv`
    - Builds CRT2_Total_adj (unless disabled and columns missing → raises if `require_familiarity=True`)
    - Samples demographics independently; samples scales via Gaussian copula
    - Writes CSV and YAML meta if `save_path` provided
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(DF_PATH)

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    df, occ_summary = _harmonize_occupations(df)
    _print_occupation_summary(df, occ_summary)

    # Adjust CRT2 if familiarity columns present (strict by default)
    df = _apply_crt2_adjustment(df, require_familiarity=require_familiarity)

    # CR-08 STEP 3 (D-80): restrict to the copula-modelling set before any
    # demographic or numeric sampling is fit. See the marked block above.
    df = _apply_d80_inclusion_rule(df)

    demo_cols, scale_cols = _build_column_sets(df)

    scales = _sample_scales_copula(df, scale_cols, n=n, rng=rng)
    demos = _sample_demos(df, demo_cols, sampled_age=scales["Age"].to_numpy(), n=n, rng=rng)
    samp = pd.concat([demos, scales], axis=1)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        samp.to_csv(save_path, index=False)
        meta = {
            "n": n,
            "seed": seed,
            "require_familiarity": require_familiarity,
            "used_columns": {
                "demo": demo_cols,
                "scales": scale_cols,
            },
            "source": str(DF_PATH),
            "n_source_frame": int(len(pd.read_csv(DF_PATH))),
            "n_d80_modelling_set": int(len(df)),
            "inclusion_rule": "D-80 (cognition-suite-sim/docs/DECISIONS.md, 2026-07-29)",
            "git_commit": _git_commit(),
            "scoring_version": "scoring_v2",
        }
        with open(save_path.with_suffix(".meta.yaml"), "w") as f:
            yaml.safe_dump(meta, f)

    return samp


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate synthetic personas with CRT2 de-familiarisation and Gaussian copula sampling.")
    ap.add_argument("--n", type=int, default=1000, help="Number of personas.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    ap.add_argument("--out", type=str, default=str(reports_tabs / "synthetic_personas.csv"), help="Output CSV path.")
    ap.add_argument("--allow-missing-familiarity", action="store_true",
                    help="If set, do NOT require CRT2 familiarity columns (skips adjustment).")
    args = ap.parse_args()

    samp = generate_personas(
        n=args.n,
        seed=args.seed,
        save_path=args.out,
        require_familiarity=not args.allow_missing_familiarity,
    )
    print("Saved:", args.out)
