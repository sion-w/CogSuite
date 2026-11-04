"""CR-03 / D-74.2 — familiarity-adjusted human CRT2 outcome reference.

Camera-ready / rebuttal-marked module. Does NOT modify manuscript pipeline
modules. CR-13 re-runs ``build_adjusted_crt2_reference`` on the corrected
473/471 frame; all numbers emitted under this path are
``PROVISIONAL_FRAME_469``.

Binding (D-73 / D-74 / D-75; Findings §12 later-timestamped wins over D-62):
  - CRT2 only is adjusted (bCRT stays raw v2).
  - Per-item multinomial logit of response_category on CRT2 seen-count (0–4)
    + demographics, predicted at seen=0, averaged over persons.
  - Unseen-only subgroup figures are UPPER BOUND, never estimates.
  - Aggregate category-mass JS is never reported without the per-item lure
    profile beside it (D-73c).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon

from socsim.common import HUMAN_LONG_TEXT_COLS, read_csv_keep_text

FRAME_CAVEAT = "PROVISIONAL_FRAME_469"
CRT2_ITEMS = [f"crt2_q{i}" for i in range(1, 5)]
BCRT_ITEMS = [f"bcrt_q{i}" for i in range(1, 5)]
ALL_ITEMS = CRT2_ITEMS + BCRT_ITEMS

CRT2_SEEN_COLS = [
    "Experience Check_1_seen",
    "Experience Check_2_seen",
    "Experience Check_3_seen",
    "Experience Check_4_seen",
]
ITEM_SEEN_COL = {f"crt2_q{i}": f"Experience Check_{i}_seen" for i in range(1, 5)}

V2_TO_CATEGORY = {
    "correct": "correct",
    "intuitive": "intuitive_incorrect",
    "other": "other_incorrect",
}
OUTCOME_LEVELS = ["correct", "intuitive_incorrect", "other_incorrect"]

DEMO_AGE_COL = "Age"
DEMO_GENDER_COL = "sex"
DEMO_EDU_COL = "Highest education level completed"
DEMO_COLS_RESOLVED = {
    "age": DEMO_AGE_COL,
    "gender": DEMO_GENDER_COL,
    "education": DEMO_EDU_COL,
    "note": (
        "Paper total-adjustment code seeks Age/Gender/Education; human file "
        "has Age, sex, Highest education level completed. CR-03 uses those "
        "three resolved columns."
    ),
}

UNSEEN_ONLY_UPPER_BOUND = {
    "source": "Rebuttal Findings.md §12.2 (2026-07-28 16:35)",
    "label": "UPPER_BOUND_NOT_ESTIMATE",
    "caveat": (
        "Seen/unseen is a self-selected subgroup, not a randomised exposure "
        "manipulation (D-74; Findings §12.5). These figures are UPPER BOUNDS "
        "on the exposure effect, never estimates."
    ),
    "rows": [
        {
            "scope": "CRT2",
            "all_pct": 23.0,
            "all_ci": [20.8, 25.2],
            "unseen_pct": 34.4,
            "unseen_ci": [30.7, 38.2],
            "seen_pct": 15.9,
            "seen_ci": [13.6, 18.4],
        },
        {
            "scope": "bCRT",
            "all_pct": 41.2,
            "all_ci": [38.5, 43.7],
            "unseen_pct": 45.0,
            "unseen_ci": [42.2, 47.8],
            "seen_pct": 16.0,
            "seen_ci": [10.9, 21.3],
        },
        {
            "scope": "pooled",
            "all_pct": 32.1,
            "all_ci": [30.3, 33.8],
            "unseen_pct": 41.8,
            "unseen_ci": [39.4, 44.1],
            "seen_pct": 16.0,
            "seen_ci": [13.7, 18.3],
        },
    ],
}

DEFAULT_HUMAN_LONG = Path("reports/rescore_v2/tables/rescore_long_human.csv")
DEFAULT_PERSON = Path("data/inputs/person_with_liwc_brt_coded.csv")
DEFAULT_TABLES_DIR = Path("reports/rescore_v2/tables")

# Filename stem (after rescore_long_) for every cell the task asks for.
EXPECTED_SOURCES = {
    "human": ["human"],
    "submitted": [
        "HumanNum467sim",
        "HumanLex467sim",
        "SynthNum467sim",
        "SynthLex467sim",
    ],
    "j2": [
        "J2_GPT56SOL_NONE",
        "J2_GEMINI35FLASH_MIN",
        "J2_DSV4FLASH_OFF",
        "J2_DSV4FLASH_ON",
        "J2_DSV4PRO_ON",
    ],
    "j3": [
        "J3_TEMP_DEFAULT",
        "J3_TEMP_000",
        "J3_TEMP_030",
        "J3_TEMP_070",
        "J3_TEMP_100",
    ],
    "j5": [
        "J5_PROBE_STANDARD",
        "J5_PROBE_SANCTION",
    ],
}


@dataclass
class AdjustedReferenceResult:
    """Tidy tables for the adjusted CRT2 reference (CR-13 re-entry point)."""

    demo_cols_used: dict[str, Any]
    n_persons_frame: int
    missing_dropped_by_item: pd.DataFrame
    coefficients: pd.DataFrame
    adjusted_by_item: pd.DataFrame
    adjusted_pooled: pd.DataFrame
    raw_by_item: pd.DataFrame
    raw_pooled: pd.DataFrame
    robustness_item_seen: pd.DataFrame
    person_level_predictions: pd.DataFrame


def map_label_v2(series: pd.Series) -> pd.Series:
    def _one(x: object) -> str:
        if pd.isna(x):
            return "missing_or_unparseable"
        return V2_TO_CATEGORY.get(str(x).strip().lower(), "missing_or_unparseable")

    return series.map(_one).astype(str)


def load_human_item_frame(
    human_long_path: Path | str = DEFAULT_HUMAN_LONG,
    person_path: Path | str = DEFAULT_PERSON,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join v2 human long to familiarity + demographics."""
    human = read_csv_keep_text(human_long_path, HUMAN_LONG_TEXT_COLS)
    person = pd.read_csv(person_path, low_memory=False)

    need = ["PROLIFIC_PID", *CRT2_SEEN_COLS, DEMO_AGE_COL, DEMO_GENDER_COL, DEMO_EDU_COL]
    missing_cols = [c for c in need if c not in person.columns]
    if missing_cols:
        raise RuntimeError(f"person file missing required columns: {missing_cols}")

    dem = person[need].copy()
    dem["PROLIFIC_PID"] = dem["PROLIFIC_PID"].astype(str)
    dem["CRT2_SeenCount"] = (
        dem[CRT2_SEEN_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    )
    dem[DEMO_AGE_COL] = pd.to_numeric(dem[DEMO_AGE_COL], errors="coerce")
    for c in CRT2_SEEN_COLS:
        dem[c] = pd.to_numeric(dem[c], errors="coerce").fillna(0).astype(int)

    human = human.copy()
    human["entity_id"] = human["entity_id"].astype(str)
    human["response_category"] = map_label_v2(human["label_v2"])
    if "normalised" in human.columns:
        blank = human["normalised"].fillna("").astype(str).eq("")
        human.loc[blank, "response_category"] = "missing_or_unparseable"

    merged = human.merge(dem, left_on="entity_id", right_on="PROLIFIC_PID", how="left", indicator=True)
    n_persons = int(human["entity_id"].nunique())
    n_matched = int(merged.loc[merged["_merge"] == "both", "entity_id"].nunique())
    if n_matched != n_persons:
        raise RuntimeError(f"PROLIFIC_PID join incomplete: matched {n_matched}/{n_persons}")

    meta = {
        "n_persons_frame": n_persons,
        "n_matched": n_matched,
        "demo_cols_used": DEMO_COLS_RESOLVED,
        "frame_caveat": FRAME_CAVEAT,
    }
    return merged.drop(columns=["_merge"]), meta


def _collapse_rare(series: pd.Series, min_count: int = 5) -> pd.Series:
    s = series.astype("object").where(series.notna(), "Other_or_missing").astype(str)
    counts = s.value_counts(dropna=False)
    rare = counts[counts < min_count].index
    return s.where(~s.isin(rare), "Other_or_missing")


def _design_matrix(
    df: pd.DataFrame,
    *,
    seen_col: str,
    age_mean: float | None = None,
    gender_levels: list[str] | None = None,
    edu_levels: list[str] | None = None,
) -> tuple[pd.DataFrame, float, list[str], list[str], pd.Series, str, str]:
    work = df.copy()
    work["_age"] = pd.to_numeric(work[DEMO_AGE_COL], errors="coerce")
    work["_gender"] = _collapse_rare(work[DEMO_GENDER_COL])
    work["_edu"] = _collapse_rare(work[DEMO_EDU_COL])
    work["_seen"] = pd.to_numeric(work[seen_col], errors="coerce")

    if age_mean is None:
        age_mean = float(work.loc[work["_age"].notna(), "_age"].mean())
    work["_age_c"] = work["_age"] - age_mean

    if gender_levels is None:
        gender_levels = sorted(work["_gender"].dropna().unique().tolist())
    if edu_levels is None:
        edu_levels = sorted(work["_edu"].dropna().unique().tolist())

    g_ref = gender_levels[0]
    e_ref = edu_levels[0]
    parts: dict[str, np.ndarray] = {
        "const": np.ones(len(work), dtype=float),
        "seen": work["_seen"].astype(float).to_numpy(),
        "age_c": work["_age_c"].astype(float).to_numpy(),
    }
    for g in gender_levels[1:]:
        parts[f"gender[{g}]"] = (work["_gender"] == g).astype(float).to_numpy()
    for e in edu_levels[1:]:
        parts[f"edu[{e}]"] = (work["_edu"] == e).astype(float).to_numpy()

    X = pd.DataFrame(parts, index=work.index)
    complete = (
        work["_age"].notna()
        & work["_seen"].notna()
        & work["_gender"].notna()
        & work["_edu"].notna()
    )
    return X, float(age_mean), gender_levels, edu_levels, complete, g_ref, e_ref


def _fit_mnlogit_one_item(
    item_df: pd.DataFrame,
    *,
    seen_col: str,
    item_id: str,
    model_tag: str,
) -> tuple[Any, pd.DataFrame, dict[str, Any], int]:
    usable = item_df[item_df["response_category"].isin(OUTCOME_LEVELS)].copy()
    n_missing = int((~item_df["response_category"].isin(OUTCOME_LEVELS)).sum())

    X_all, age_mean, g_levels, e_levels, complete, g_ref, e_ref = _design_matrix(
        usable, seen_col=seen_col
    )
    y = usable["response_category"].astype(
        pd.CategoricalDtype(categories=OUTCOME_LEVELS, ordered=False)
    )
    y_codes = y.cat.codes

    mask = complete & y_codes.ge(0)
    X = X_all.loc[mask]
    yv = y_codes.loc[mask]
    if len(X) < 20 or int(yv.nunique()) < 2:
        raise RuntimeError(f"MNLogit not identified for {item_id} ({model_tag}): n={len(X)}")

    fit = sm.MNLogit(yv, X).fit(method="bfgs", maxiter=500, disp=False)
    params = fit.params
    nonref = OUTCOME_LEVELS[1:]
    coeff_rows = []
    for j, cat in enumerate(nonref):
        col = params.columns[j] if hasattr(params, "columns") else j
        for pname in params.index:
            coeff_rows.append(
                {
                    "frame_caveat": FRAME_CAVEAT,
                    "model_tag": model_tag,
                    "item_id": item_id,
                    "seen_predictor": seen_col,
                    "outcome_category": cat,
                    "reference_category": OUTCOME_LEVELS[0],
                    "term": str(pname),
                    "coefficient": float(params.loc[pname, col]),
                    "n_fit": int(len(X)),
                    "n_missing_dropped": n_missing,
                    "gender_ref": g_ref,
                    "edu_ref": e_ref,
                    "age_mean_fit": age_mean,
                }
            )
    meta = {
        "age_mean": age_mean,
        "gender_levels": g_levels,
        "edu_levels": e_levels,
        "gender_ref": g_ref,
        "edu_ref": e_ref,
        "fit": fit,
        "exog_names": list(X.columns),
    }
    return fit, pd.DataFrame(coeff_rows), meta, n_missing


def _predict_mean_probs(
    fit: Any,
    persons: pd.DataFrame,
    *,
    seen_value: float,
    seen_col_name: str,
    meta: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    pred_df = persons.drop_duplicates(subset=["entity_id"]).copy()
    pred_df = pred_df[["entity_id", DEMO_AGE_COL, DEMO_GENDER_COL, DEMO_EDU_COL]].copy()
    pred_df[seen_col_name] = seen_value
    age = pd.to_numeric(pred_df[DEMO_AGE_COL], errors="coerce")
    pred_df[DEMO_AGE_COL] = age.fillna(meta["age_mean"])
    pred_df[DEMO_GENDER_COL] = pred_df[DEMO_GENDER_COL].where(
        pred_df[DEMO_GENDER_COL].notna(), "Other_or_missing"
    )
    pred_df[DEMO_EDU_COL] = pred_df[DEMO_EDU_COL].where(
        pred_df[DEMO_EDU_COL].notna(), "Other_or_missing"
    )

    X, _, _, _, _, _, _ = _design_matrix(
        pred_df,
        seen_col=seen_col_name,
        age_mean=meta["age_mean"],
        gender_levels=meta["gender_levels"],
        edu_levels=meta["edu_levels"],
    )
    for c in meta["exog_names"]:
        if c not in X.columns:
            X[c] = 0.0
    X = X[meta["exog_names"]]
    probs = np.asarray(fit.predict(X), dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise RuntimeError(f"Unexpected MNLogit predict shape: {probs.shape}")
    person_tbl = pd.DataFrame(
        {
            "entity_id": pred_df["entity_id"].to_numpy(),
            "p_correct": probs[:, 0],
            "p_intuitive_incorrect": probs[:, 1],
            "p_other_incorrect": probs[:, 2],
        }
    )
    return person_tbl, probs.mean(axis=0)


def _raw_props(item_df: pd.DataFrame) -> dict[str, float]:
    usable = item_df[item_df["response_category"].isin(OUTCOME_LEVELS)]
    counts = usable["response_category"].value_counts().reindex(OUTCOME_LEVELS, fill_value=0)
    total = float(counts.sum())
    if total <= 0:
        return {c: float("nan") for c in OUTCOME_LEVELS}
    return {c: float(counts[c] / total) for c in OUTCOME_LEVELS}


def build_adjusted_crt2_reference(
    human_long_path: Path | str = DEFAULT_HUMAN_LONG,
    person_path: Path | str = DEFAULT_PERSON,
) -> AdjustedReferenceResult:
    """Fit per-CRT2-item MN logits; return adjusted distributions at seen=0.

    Public CR-13 entry point. Returns tidy tables only — no side effects.
    """
    merged, meta = load_human_item_frame(human_long_path, person_path)
    crt2 = merged[merged["item_id"].isin(CRT2_ITEMS)].copy()
    persons = crt2.drop_duplicates("entity_id")

    coeff_parts: list[pd.DataFrame] = []
    miss_rows: list[dict] = []
    adj_rows: list[dict] = []
    raw_rows: list[dict] = []
    rob_rows: list[dict] = []
    person_pred_parts: list[pd.DataFrame] = []
    adj_vecs: list[np.ndarray] = []
    raw_vecs: list[np.ndarray] = []

    for item_id in CRT2_ITEMS:
        item_df = crt2[crt2["item_id"] == item_id].copy()
        fit, coeffs, mmeta, n_miss = _fit_mnlogit_one_item(
            item_df, seen_col="CRT2_SeenCount", item_id=item_id, model_tag="primary_seen_count"
        )
        coeff_parts.append(coeffs)
        miss_rows.append(
            {
                "frame_caveat": FRAME_CAVEAT,
                "item_id": item_id,
                "n_missing_dropped": n_miss,
                "n_rows": int(len(item_df)),
            }
        )

        person_tbl, mean_vec = _predict_mean_probs(
            fit, persons, seen_value=0.0, seen_col_name="CRT2_SeenCount", meta=mmeta
        )
        person_tbl.insert(0, "item_id", item_id)
        person_tbl.insert(0, "model_tag", "primary_seen_count")
        person_tbl.insert(0, "frame_caveat", FRAME_CAVEAT)
        person_pred_parts.append(person_tbl)
        adj_vecs.append(mean_vec)
        for cat, p in zip(OUTCOME_LEVELS, mean_vec, strict=True):
            adj_rows.append(
                {
                    "frame_caveat": FRAME_CAVEAT,
                    "scope": "item",
                    "item_id": item_id,
                    "instrument": "crt2",
                    "reference": "adjusted_seen0",
                    "response_category": cat,
                    "proportion": float(p),
                }
            )

        raw = _raw_props(item_df)
        raw_vecs.append(np.array([raw[c] for c in OUTCOME_LEVELS], dtype=float))
        for cat in OUTCOME_LEVELS:
            raw_rows.append(
                {
                    "frame_caveat": FRAME_CAVEAT,
                    "scope": "item",
                    "item_id": item_id,
                    "instrument": "crt2",
                    "reference": "raw_v2",
                    "response_category": cat,
                    "proportion": raw[cat],
                }
            )

        seen_flag_col = ITEM_SEEN_COL[item_id]
        fit_r, coeffs_r, mmeta_r, _ = _fit_mnlogit_one_item(
            item_df,
            seen_col=seen_flag_col,
            item_id=item_id,
            model_tag="robustness_item_seen_flag",
        )
        coeff_parts.append(coeffs_r)
        _, mean_r = _predict_mean_probs(
            fit_r, persons, seen_value=0.0, seen_col_name=seen_flag_col, meta=mmeta_r
        )
        for cat, p in zip(OUTCOME_LEVELS, mean_r, strict=True):
            rob_rows.append(
                {
                    "frame_caveat": FRAME_CAVEAT,
                    "scope": "item",
                    "item_id": item_id,
                    "instrument": "crt2",
                    "reference": "adjusted_seen0_item_flag",
                    "response_category": cat,
                    "proportion": float(p),
                }
            )

    adj_pooled = np.mean(np.vstack(adj_vecs), axis=0)
    raw_pooled = np.mean(np.vstack(raw_vecs), axis=0)
    for cat, p in zip(OUTCOME_LEVELS, adj_pooled, strict=True):
        adj_rows.append(
            {
                "frame_caveat": FRAME_CAVEAT,
                "scope": "pooled",
                "item_id": "crt2_pooled",
                "instrument": "crt2",
                "reference": "adjusted_seen0",
                "response_category": cat,
                "proportion": float(p),
            }
        )
    for cat, p in zip(OUTCOME_LEVELS, raw_pooled, strict=True):
        raw_rows.append(
            {
                "frame_caveat": FRAME_CAVEAT,
                "scope": "pooled",
                "item_id": "crt2_pooled",
                "instrument": "crt2",
                "reference": "raw_v2",
                "response_category": cat,
                "proportion": float(p),
            }
        )

    rob_df = pd.DataFrame(rob_rows)
    for cat in OUTCOME_LEVELS:
        p = float(rob_df.loc[rob_df["response_category"] == cat, "proportion"].mean())
        rob_rows.append(
            {
                "frame_caveat": FRAME_CAVEAT,
                "scope": "pooled",
                "item_id": "crt2_pooled",
                "instrument": "crt2",
                "reference": "adjusted_seen0_item_flag",
                "response_category": cat,
                "proportion": p,
            }
        )

    return AdjustedReferenceResult(
        demo_cols_used=meta["demo_cols_used"],
        n_persons_frame=meta["n_persons_frame"],
        missing_dropped_by_item=pd.DataFrame(miss_rows),
        coefficients=pd.concat(coeff_parts, ignore_index=True),
        adjusted_by_item=pd.DataFrame([r for r in adj_rows if r["scope"] == "item"]),
        adjusted_pooled=pd.DataFrame([r for r in adj_rows if r["scope"] == "pooled"]),
        raw_by_item=pd.DataFrame([r for r in raw_rows if r["scope"] == "item"]),
        raw_pooled=pd.DataFrame([r for r in raw_rows if r["scope"] == "pooled"]),
        robustness_item_seen=pd.DataFrame(rob_rows),
        person_level_predictions=pd.concat(person_pred_parts, ignore_index=True),
    )


HUMAN_REFERENCE_N = 471


def restrict_to_crt2_complete(long: pd.DataFrame, expected: int | None = HUMAN_REFERENCE_N) -> pd.DataFrame:
    """CR-23 ruling (d): the human reference is the CRT2-complete people (all four CRT2 items
    scored; 471 on the CR frame), for CRT2 and bCRT alike, not the 473-person frame."""
    crt2 = long[long["item_id"].isin(CRT2_ITEMS)]
    cats = map_label_v2(crt2["label_v2"])
    if "normalised" in crt2.columns:
        cats = cats.where(~crt2["normalised"].fillna("").astype(str).eq(""), "missing_or_unparseable")
    scored = crt2[cats.isin(OUTCOME_LEVELS).to_numpy()].groupby("entity_id")["item_id"].nunique()
    keep = set(scored[scored == len(CRT2_ITEMS)].index.astype(str))
    if expected is not None and len(keep) != expected:
        raise ValueError(f"{len(keep)} CRT2-complete humans in the long frame, expected {expected}")
    return long[long["entity_id"].astype(str).isin(keep)].copy()


def _props_from_long(df: pd.DataFrame, items: list[str]) -> np.ndarray:
    sub = df[df["item_id"].isin(items)].copy()
    cats = map_label_v2(sub["label_v2"])
    if "normalised" in sub.columns:
        blank = sub["normalised"].fillna("").astype(str).eq("")
        cats = cats.where(~blank, "missing_or_unparseable")
    usable = cats[cats.isin(OUTCOME_LEVELS)]
    counts = usable.value_counts().reindex(OUTCOME_LEVELS, fill_value=0)
    total = float(counts.sum())
    if total <= 0:
        return np.full(len(OUTCOME_LEVELS), np.nan)
    return (counts.astype(float) / total).to_numpy()


def _lure_by_item(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    rows = []
    for item in items:
        sub = df[df["item_id"] == item]
        cats = map_label_v2(sub["label_v2"])
        if "normalised" in sub.columns:
            blank = sub["normalised"].fillna("").astype(str).eq("")
            cats = cats.where(~blank, "missing_or_unparseable")
        ans = cats[cats.isin(OUTCOME_LEVELS)]
        n = int(len(ans))
        n_lure = int((ans == "intuitive_incorrect").sum())
        rows.append(
            {
                "item_id": item,
                "n_answered": n,
                "n_lure": n_lure,
                "lure_pct": (100.0 * n_lure / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _js(p: np.ndarray, q: np.ndarray) -> float:
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        return float("nan")
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    q = np.clip(np.asarray(q, dtype=float), 0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    return float(jensenshannon(p / p.sum(), q / q.sum()))


def adjusted_vectors_from_result(
    result: AdjustedReferenceResult,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    """Extract pooled/item probability vectors and lure % from an AdjustedReferenceResult."""

    def _vec(df: pd.DataFrame, item_id: str) -> np.ndarray:
        sub = df[df["item_id"] == item_id]
        return np.array(
            [float(sub.loc[sub["response_category"] == c, "proportion"].iloc[0]) for c in OUTCOME_LEVELS],
            dtype=float,
        )

    adj_by_item = {it: _vec(result.adjusted_by_item, it) for it in CRT2_ITEMS}
    raw_by_item = {it: _vec(result.raw_by_item, it) for it in CRT2_ITEMS}
    adj_pooled = _vec(result.adjusted_pooled, "crt2_pooled")
    raw_pooled = _vec(result.raw_pooled, "crt2_pooled")
    human_adj_lure = {it: 100.0 * adj_by_item[it][1] for it in CRT2_ITEMS}
    human_raw_lure = {it: 100.0 * raw_by_item[it][1] for it in CRT2_ITEMS}
    return adj_pooled, adj_by_item, raw_pooled, raw_by_item, human_adj_lure, human_raw_lure


def compare_cell_to_references(
    cell_long: pd.DataFrame,
    *,
    source: str,
    adj_crt2_pooled: np.ndarray,
    adj_crt2_by_item: dict[str, np.ndarray],
    raw_crt2_pooled: np.ndarray,
    raw_crt2_by_item: dict[str, np.ndarray],
    raw_bcrt_pooled: np.ndarray,
    raw_bcrt_by_item: dict[str, np.ndarray],
    human_crt2_lure_adj: dict[str, float],
    human_crt2_lure_raw: dict[str, float],
    human_bcrt_lure: dict[str, float],
) -> dict[str, Any]:
    """Category-mass JS + per-item lure profiles (always side by side; D-73c)."""
    cell_crt2 = _props_from_long(cell_long, CRT2_ITEMS)
    cell_bcrt = _props_from_long(cell_long, BCRT_ITEMS)
    cell_lure = _lure_by_item(cell_long, ALL_ITEMS)

    js_crt2_adj = _js(adj_crt2_pooled, cell_crt2)
    js_crt2_raw = _js(raw_crt2_pooled, cell_crt2)
    js_bcrt = _js(raw_bcrt_pooled, cell_bcrt)

    lure_rows = []
    for _, r in cell_lure.iterrows():
        item = str(r["item_id"])
        if item in CRT2_ITEMS:
            lure_rows.append(
                {
                    "frame_caveat": FRAME_CAVEAT,
                    "source": source,
                    "item_id": item,
                    "instrument": "crt2",
                    "cell_lure_pct": float(r["lure_pct"]),
                    "human_raw_lure_pct": human_crt2_lure_raw.get(item),
                    "human_adj_lure_pct": human_crt2_lure_adj.get(item),
                    "delta_vs_raw_pp": float(r["lure_pct"]) - human_crt2_lure_raw[item],
                    "delta_vs_adj_pp": float(r["lure_pct"]) - human_crt2_lure_adj[item],
                }
            )
        else:
            lure_rows.append(
                {
                    "frame_caveat": FRAME_CAVEAT,
                    "source": source,
                    "item_id": item,
                    "instrument": "bcrt",
                    "cell_lure_pct": float(r["lure_pct"]),
                    "human_raw_lure_pct": human_bcrt_lure.get(item),
                    "human_adj_lure_pct": np.nan,
                    "delta_vs_raw_pp": float(r["lure_pct"]) - human_bcrt_lure[item],
                    "delta_vs_adj_pp": np.nan,
                }
            )

    item_js_rows = []
    for item in CRT2_ITEMS:
        cell_item = _props_from_long(cell_long, [item])
        item_js_rows.append(
            {
                "frame_caveat": FRAME_CAVEAT,
                "source": source,
                "item_id": item,
                "instrument": "crt2",
                "js_vs_raw": _js(raw_crt2_by_item[item], cell_item),
                "js_vs_adj": _js(adj_crt2_by_item[item], cell_item),
            }
        )
    for item in BCRT_ITEMS:
        cell_item = _props_from_long(cell_long, [item])
        item_js_rows.append(
            {
                "frame_caveat": FRAME_CAVEAT,
                "source": source,
                "item_id": item,
                "instrument": "bcrt",
                "js_vs_raw": _js(raw_bcrt_by_item[item], cell_item),
                "js_vs_adj": np.nan,
            }
        )

    conclusion_changed = bool(
        np.isfinite(js_crt2_adj)
        and np.isfinite(js_crt2_raw)
        and abs(js_crt2_adj - js_crt2_raw) >= 0.02
    )
    return {
        "js_row": {
            "frame_caveat": FRAME_CAVEAT,
            "source": source,
            "js_crt2_vs_raw": js_crt2_raw,
            "js_crt2_vs_adjusted": js_crt2_adj,
            "js_crt2_delta": (
                js_crt2_adj - js_crt2_raw
                if np.isfinite(js_crt2_adj) and np.isfinite(js_crt2_raw)
                else np.nan
            ),
            "js_bcrt_vs_raw": js_bcrt,
            "cell_crt2_correct": float(cell_crt2[0]),
            "cell_crt2_intuitive": float(cell_crt2[1]),
            "cell_crt2_other": float(cell_crt2[2]),
            "ref_adj_crt2_correct": float(adj_crt2_pooled[0]),
            "ref_adj_crt2_intuitive": float(adj_crt2_pooled[1]),
            "ref_adj_crt2_other": float(adj_crt2_pooled[2]),
            "conclusion_flag_js_moved_ge_0p02": conclusion_changed,
        },
        "lure_rows": lure_rows,
        "item_js_rows": item_js_rows,
    }


def discover_rescore_long_csvs(tables_dir: Path | str = DEFAULT_TABLES_DIR) -> list[Path]:
    tables_dir = Path(tables_dir)
    return sorted(
        p for p in tables_dir.glob("rescore_long_*.csv") if p.name != "rescore_long_ALL_sources.csv"
    )


def source_stem_from_path(path: Path) -> str:
    name = path.name
    prefix = "rescore_long_"
    if not name.startswith(prefix):
        raise ValueError(path)
    return name[len(prefix) : -len(".csv")]
