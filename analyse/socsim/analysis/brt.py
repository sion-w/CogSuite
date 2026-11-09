from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

BRT_FREQ_COLS = ["BRT1_decision", "BRT2_decision", "BRT2_anchor", "BRT2_research_specificity"]

PRIMARY_BRT_OUTCOMES = [
    "BRT2_high_quality_research",
    "BRT1_emotional_outburst",
    "BRT1_moral_fairness_loyalty",
    "BRT2_reflective_accept",
    "BRT1_reflective_keep",
]


def _z(x: pd.Series) -> pd.Series:
    num = pd.to_numeric(x, errors="coerce")
    sd = num.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (num - num.mean()) / sd


def _binary_from_decision(series: pd.Series, positive: set[str], negative: set[str]) -> pd.Series:
    s = series.astype(str)
    return pd.Series(np.where(s.isin(positive), 1.0, np.where(s.isin(negative), 0.0, np.nan)), index=series.index)


def _ensure_brt_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "BRT2_research_specificity" in out.columns:
        out["BRT2_high_quality_research"] = (out["BRT2_research_specificity"] == "HIGH_QUALITY_RESEARCH").astype(float)
        out["BRT2_vague_research"] = (out["BRT2_research_specificity"] == "VAGUE_RESEARCH").astype(float)
        out["BRT2_social_media_research_bin"] = (out["BRT2_research_specificity"] == "SOCIAL_MEDIA_RESEARCH").astype(float)
        out["BRT2_any_research"] = (~out["BRT2_research_specificity"].isin(["NO_ADDITIONAL_RESEARCH"])).astype(float)
    if "BRT2_decision" in out.columns:
        out["BRT2_take_medex_bin"] = _binary_from_decision(
            out["BRT2_decision"],
            {"TAKE_MEDEX"},
            {"DELAY_AND_INVESTIGATE", "NOT_TAKE_MEDEX"},
        )
    if "BRT1_decision" in out.columns:
        out["BRT1_keep_booking_bin"] = _binary_from_decision(
            out["BRT1_decision"],
            {"KEEP_BOOKING"},
            {"CHANGE_RESTAURANT", "CANCEL_OR_POSTPONE"},
        )
    return out


def brt_descriptives(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    work = _ensure_brt_derived(df)
    rows: list[dict] = []
    for col in BRT_FREQ_COLS:
        if col not in work.columns:
            continue
        s = work[col].dropna()
        total = int(len(s))
        if total == 0:
            continue
        vc = s.value_counts(dropna=False)
        for category, count in vc.items():
            rows.append(
                {
                    "dataset": dataset,
                    "measure": col,
                    "category": str(category),
                    "count": int(count),
                    "proportion": float(count / total),
                    "N": total,
                }
            )
    return pd.DataFrame(rows)


def _fit_logit(work: pd.DataFrame, outcome: str, predictors: list[str], model_type: str) -> pd.DataFrame:
    cols = [outcome] + predictors
    sub = work[cols].dropna()
    if len(sub) < 20:
        return pd.DataFrame()
    y = pd.to_numeric(sub[outcome], errors="coerce")
    # Only binary outcomes are valid for Logit.
    if y.nunique(dropna=True) < 2:
        return pd.DataFrame()
    X = sm.add_constant(sub[predictors], has_constant="add")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.Logit(y, X).fit(disp=False, maxiter=200)
    except Exception:
        return pd.DataFrame()
    warning_parts = []
    if not bool(getattr(model, "mle_retvals", {}).get("converged", True)):
        warning_parts.append("logit_nonconverged")
    warning_parts.extend(sorted({w.category.__name__ for w in caught}))
    conf = model.conf_int()
    rows = []
    events = int((y == 1).sum())
    non_events = int((y == 0).sum())
    prevalence = float(y.mean())
    for p in predictors:
        if p not in model.params.index:
            continue
        beta = float(model.params[p])
        ci_low = float(conf.loc[p, 0])
        ci_high = float(conf.loc[p, 1])
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            odds_ratio = float(np.exp(beta))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))
        row_warning_parts = list(warning_parts)
        numeric_values = [beta, float(model.bse[p]), float(model.tvalues[p]), float(model.pvalues[p]), ci_low, ci_high, odds_ratio, or_ci_low, or_ci_high]
        if not np.isfinite(numeric_values).all():
            row_warning_parts.append("nonfinite_estimate")
        elif abs(float(model.bse[p])) > 100 or abs(ci_high) > 100 or abs(ci_low) > 100:
            row_warning_parts.append("unstable_estimate")
        rows.append(
            {
                "outcome": outcome,
                "predictor": p,
                "beta": beta,
                "SE": float(model.bse[p]),
                "z": float(model.tvalues[p]),
                "p": float(model.pvalues[p]),
                "CI95_low": ci_low,
                "CI95_high": ci_high,
                "odds_ratio": odds_ratio,
                "OR_CI95_low": or_ci_low,
                "OR_CI95_high": or_ci_high,
                "N": int(model.nobs),
                "event_count": events,
                "non_event_count": non_events,
                "outcome_prevalence": prevalence,
                "model_type": model_type,
                "model_warning": ";".join(sorted(set(row_warning_parts))),
            }
        )
    return pd.DataFrame(rows)


def run_brt_models(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    work = _ensure_brt_derived(df)
    if dataset == "human":
        bcrt_col = "BCRT_Total_observed" if "BCRT_Total_observed" in work.columns else "BCRT_Total"
        crt2_col = "CRT2_Total_adj_observed" if "CRT2_Total_adj_observed" in work.columns else "CRT2_Total"
        nfc_col = "NFC_Total_observed" if "NFC_Total_observed" in work.columns else "NFC_Total"
    elif dataset.endswith("_agent"):
        bcrt_col = "BCRT_Total_realised" if "BCRT_Total_realised" in work.columns else "BCRT_Total"
        crt2_col = "CRT2_Total_realised" if "CRT2_Total_realised" in work.columns else "CRT2_Total_adj"
        nfc_col = "NFC_Total_realised" if "NFC_Total_realised" in work.columns else "NFC_Total"
    else:
        bcrt_col = "BCRT_Total"
        crt2_col = "CRT2_Total"
        nfc_col = "NFC_Total"
    required = [bcrt_col, crt2_col, nfc_col]
    if any(c not in work.columns for c in required):
        return pd.DataFrame()
    work = work.copy()
    work["z_BCRT_Total"] = _z(work[bcrt_col])
    work["z_CRT2_Total"] = _z(work[crt2_col])
    work["z_NFC_Total"] = _z(work[nfc_col])
    predictor_source_map = {
        "z_BCRT_Total": bcrt_col,
        "z_CRT2_Total": crt2_col,
        "z_NFC_Total": nfc_col,
    }
    primary_predictors = ["z_BCRT_Total"]
    secondary_predictors = ["z_BCRT_Total", "z_CRT2_Total", "z_NFC_Total"]
    rows = []
    for outcome in PRIMARY_BRT_OUTCOMES:
        if outcome not in work.columns:
            continue
        rows.append(_fit_logit(work, outcome, primary_predictors, "primary_bcrt_only"))
        rows.append(_fit_logit(work, outcome, secondary_predictors, "secondary_full_cognitive"))
    if not rows:
        return pd.DataFrame()
    out = pd.concat([r for r in rows if not r.empty], ignore_index=True) if any(not r.empty for r in rows) else pd.DataFrame()
    if out.empty:
        return out
    out.insert(0, "dataset", dataset)
    out["predictor_source_col"] = out["predictor"].map(predictor_source_map)
    return out
