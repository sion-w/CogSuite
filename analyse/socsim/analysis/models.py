from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from socsim.analysis.descriptives import resolve_var_column

OUTCOMES = ["CRT2_Total", "BCRT_Total"]
PREDICTORS = ["nfc", "con"]
PREDICTOR_LABELS = {"nfc": "NFC_Total", "con": "BFI_Conscientiousness"}
MAX_DISCRETE_SCORE = 4


def _z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd


def _warning_rows(dataset: str, outcome: str, n: int, model_type: str, warning: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pred in PREDICTORS:
        rows.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "predictor": PREDICTOR_LABELS[pred],
                "model_type": model_type,
                "coef": np.nan,
                "SE": np.nan,
                "p": np.nan,
                "CI95_low": np.nan,
                "CI95_high": np.nan,
                "standardised_beta": np.nan,
                "N": int(n),
                "warning": warning,
            }
        )
    return rows


def _resolve_outcome_column(df: pd.DataFrame, dataset: str, outcome_var: str) -> str | None:
    if outcome_var != "CRT2_Total_adj":
        return resolve_var_column(df, outcome_var, dataset)
    if dataset == "human":
        if "CRT2_Total_adj_observed" in df.columns:
            return "CRT2_Total_adj_observed"
        if "CRT2_Total_observed" in df.columns:
            return "CRT2_Total_observed"
        return None
    if dataset == "synthetic":
        if "CRT2_Total_adj_specified" in df.columns:
            return "CRT2_Total_adj_specified"
        if "CRT2_Total_specified" in df.columns:
            return "CRT2_Total_specified"
        return None
    if "CRT2_Total_adj_realised" in df.columns:
        return "CRT2_Total_adj_realised"
    return resolve_var_column(df, "CRT2_Total", dataset)


def run_fixed_models(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    nfc = resolve_var_column(df, "NFC_Total", dataset)
    con = resolve_var_column(df, "BFI_Conscientiousness", dataset)
    if not nfc or not con:
        return pd.DataFrame()
    rows = []
    for outcome in OUTCOMES:
        # CR-01 ANALYSIS_INVENTORY §5: `_resolve_outcome_column` was correct-but-never-called.
        # It is wired in here by CR-13B. With the resolver itself now adjusted-first for the
        # human CRT2 score, the two agree; the call keeps the adjusted-outcome rule in one
        # named place instead of in dead code.
        y_col = _resolve_outcome_column(df, dataset, outcome)
        if not y_col:
            continue
        work = pd.DataFrame(
            {
                "y": pd.to_numeric(df[y_col], errors="coerce"),
                "nfc": pd.to_numeric(df[nfc], errors="coerce"),
                "con": pd.to_numeric(df[con], errors="coerce"),
            }
        ).dropna()
        n = len(work)
        if n < 3 or work["y"].std(ddof=1) == 0:
            rows.extend(_warning_rows(dataset, outcome, n, "binomial_glm", "N small or variance=0; interpret cautiously."))
            rows.extend(_warning_rows(dataset, outcome, n, "ols_sensitivity", "N small or variance=0; interpret cautiously."))
            continue
        y_glm = work["y"].copy()
        glm_warning = ""
        is_integer_like = bool(np.all(np.isclose(y_glm, np.round(y_glm), atol=1e-9)))
        in_bounds = bool(((y_glm >= 0) & (y_glm <= MAX_DISCRETE_SCORE)).all())
        if not is_integer_like or not in_bounds:
            y_glm = y_glm.round().clip(lower=0, upper=MAX_DISCRETE_SCORE)
            glm_warning = "Outcome was non-integer/out-of-range; rounded+clipped to [0,4] for binomial GLM."
        try:
            X = sm.add_constant(work[["nfc", "con"]], has_constant="add")
            successes = y_glm.values
            failures = MAX_DISCRETE_SCORE - successes
            glm = sm.GLM(
                np.column_stack([successes, failures]),
                X,
                family=sm.families.Binomial(),
            ).fit()
            glm_ci = glm.conf_int()
            for pred in PREDICTORS:
                ci = glm_ci.loc[pred]
                rows.append(
                    {
                        "dataset": dataset,
                        "outcome": outcome,
                        "predictor": PREDICTOR_LABELS[pred],
                        "model_type": "binomial_glm",
                        "coef": float(glm.params[pred]),
                        "SE": float(glm.bse[pred]),
                        "p": float(glm.pvalues[pred]),
                        "CI95_low": float(ci.iloc[0]),
                        "CI95_high": float(ci.iloc[1]),
                        "standardised_beta": np.nan,
                        "N": int(n),
                        "warning": glm_warning,
                    }
                )
        except Exception as exc:
            rows.extend(_warning_rows(dataset, outcome, n, "binomial_glm", f"GLM fit failed: {exc}"))
        try:
            X = sm.add_constant(work[["nfc", "con"]], has_constant="add")
            model = sm.OLS(work["y"], X).fit()
            zwork = pd.DataFrame({"y": _z(work["y"]), "nfc": _z(work["nfc"]), "con": _z(work["con"])}).dropna()
            zmodel = sm.OLS(zwork["y"], sm.add_constant(zwork[["nfc", "con"]], has_constant="add")).fit() if len(zwork) >= 3 else None
            ci_all = model.conf_int()
            for pred in PREDICTORS:
                ci = ci_all.loc[pred]
                rows.append(
                    {
                        "dataset": dataset,
                        "outcome": outcome,
                        "predictor": PREDICTOR_LABELS[pred],
                        "model_type": "ols_sensitivity",
                        "coef": float(model.params[pred]),
                        "SE": float(model.bse[pred]),
                        "p": float(model.pvalues[pred]),
                        "CI95_low": float(ci.iloc[0]),
                        "CI95_high": float(ci.iloc[1]),
                        "standardised_beta": float(zmodel.params[pred]) if zmodel is not None and pred in zmodel.params else np.nan,
                        "N": int(n),
                        "warning": "",
                    }
                )
        except Exception as exc:
            rows.extend(_warning_rows(dataset, outcome, n, "ols_sensitivity", f"OLS fit failed: {exc}"))
    return pd.DataFrame(rows)


def _liwc_predictors(df: pd.DataFrame, dataset: str) -> dict[str, str]:
    predictors: dict[str, str] = {}
    if dataset == "human":
        if "CRT2_Total_adj_observed" in df.columns:
            predictors["CRT2_Total_adj"] = "CRT2_Total_adj_observed"
        elif "CRT2_Total_observed" in df.columns:
            predictors["CRT2_Total_adj"] = "CRT2_Total_observed"
        if "BCRT_Total_observed" in df.columns:
            predictors["BCRT_Total"] = "BCRT_Total_observed"
    elif dataset.endswith("_agent"):
        if "CRT2_Total_realised" in df.columns:
            predictors["CRT2_Total_realised"] = "CRT2_Total_realised"
        if "BCRT_Total_realised" in df.columns:
            predictors["BCRT_Total_realised"] = "BCRT_Total_realised"
    else:
        crt = resolve_var_column(df, "CRT2_Total", dataset)
        bcrt = resolve_var_column(df, "BCRT_Total", dataset)
        if crt:
            predictors["CRT2_Total"] = crt
        if bcrt:
            predictors["BCRT_Total"] = bcrt
    return predictors


def run_liwc_family_models(df: pd.DataFrame, dataset: str, liwc_families: dict[str, list[str]]) -> pd.DataFrame:
    predictor_map = _liwc_predictors(df, dataset)
    if not predictor_map:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for family, liwc_vars in liwc_families.items():
        for liwc_var in liwc_vars:
            if liwc_var not in df.columns:
                continue
            for pred_name, pred_col in predictor_map.items():
                work = pd.DataFrame(
                    {
                        "y": pd.to_numeric(df[liwc_var], errors="coerce"),
                        "x": pd.to_numeric(df[pred_col], errors="coerce"),
                    }
                ).dropna()
                warning = ""
                if len(work) < 3 or work["x"].std(ddof=1) == 0 or work["y"].std(ddof=1) == 0:
                    warning = "N small or variance=0; interpret cautiously."
                    rows.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "liwc_var": liwc_var,
                            "outcome": liwc_var,
                            "predictor": pred_name,
                            "predictor_source_col": pred_col,
                            "beta": np.nan,
                            "SE": np.nan,
                            "p_raw": np.nan,
                            "q_fdr": np.nan,
                            "CI95_low": np.nan,
                            "CI95_high": np.nan,
                            "N": int(len(work)),
                            "r2": np.nan,
                            "warning": warning,
                        }
                    )
                    continue
                model = sm.OLS(work["y"], sm.add_constant(work["x"], has_constant="add")).fit()
                ci = model.conf_int().loc["x"]
                predictor_group = "CRT2" if "CRT2" in pred_name else "BCRT" if "BCRT" in pred_name else pred_name
                rows.append(
                    {
                        "dataset": dataset,
                        "family": family,
                        "liwc_var": liwc_var,
                        "outcome": liwc_var,
                        "predictor": pred_name,
                        "predictor_group": predictor_group,
                        "predictor_source_col": pred_col,
                        "beta": float(model.params["x"]),
                        "SE": float(model.bse["x"]),
                        "p_raw": float(model.pvalues["x"]),
                        "q_fdr": np.nan,
                        "CI95_low": float(ci.iloc[0]),
                        "CI95_high": float(ci.iloc[1]),
                        "N": int(len(work)),
                        "r2": float(model.rsquared),
                        "warning": warning,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for fam in out["family"].dropna().unique():
        fam_mask = (out["family"] == fam) & out["p_raw"].notna()
        pvals = out.loc[fam_mask, "p_raw"].values
        if len(pvals) == 0:
            continue
        _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        out.loc[fam_mask, "q_fdr"] = qvals
    return out
