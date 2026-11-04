from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from socsim.analysis.binned_relationships import CORE_RELATIONSHIPS, relationship_specs_for_brt_liwc, _stack_relationship


def _fit_interaction_model(work: pd.DataFrame, outcome_kind: str):
    if work.empty or "human" not in set(work["dataset"]):
        return None
    if work["x"].nunique(dropna=True) < 2 or work["dataset"].nunique(dropna=True) < 2:
        return None

    model_df = work.copy()
    ordered = ["human"] + sorted([ds for ds in model_df["dataset"].unique() if ds != "human"])
    model_df["dataset"] = pd.Categorical(model_df["dataset"], categories=ordered, ordered=True)
    formula = "y ~ x * C(dataset, Treatment(reference='human'))"
    try:
        if outcome_kind == "continuous":
            return smf.ols(formula, data=model_df).fit()
        if outcome_kind == "count":
            return smf.ols(formula, data=model_df).fit()
        if outcome_kind == "binary":
            unique_y = set(model_df["y"].dropna().unique())
            if not unique_y.issubset({0, 1, 0.0, 1.0}) or len(unique_y) < 2:
                return None
            return smf.glm(formula, data=model_df, family=sm.families.Binomial()).fit()
    except Exception:
        return None
    return None


def _interaction_term_name(model, comparison: str) -> str | None:
    for name in model.params.index:
        if f"[T.{comparison}]" in name and "dataset" in name and "x" in name:
            return str(name)
    return None


def _rows_from_model(
    model,
    predictor: str,
    outcome: str,
    comparisons: list[str],
) -> list[dict[str, object]]:
    if model is None or "x" not in model.params.index:
        return []

    conf_int = model.conf_int()
    slope_human = float(model.params["x"])
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        term_name = _interaction_term_name(model, comparison)
        if term_name is None or term_name not in model.params.index:
            continue
        interaction_beta = float(model.params[term_name])
        ci = conf_int.loc[term_name]
        rows.append(
            {
                "predictor": predictor,
                "outcome": outcome,
                "comparison": comparison,
                "slope_human": slope_human,
                "slope_sim": slope_human + interaction_beta,
                "interaction_beta": interaction_beta,
                "interaction_p": float(model.pvalues[term_name]),
                "ci_low": float(ci.iloc[0]),
                "ci_high": float(ci.iloc[1]),
            }
        )
    return rows


def run_slope_difference_tests(
    by_dataset: dict[str, pd.DataFrame],
    liwc_families: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_datasets = [ds for ds in by_dataset if ds == "human" or ds == "synthetic" or ds.endswith("_agent")]
    core_comparisons = [ds for ds in core_datasets if ds != "human"]
    core_rows: list[dict[str, object]] = []
    for spec in CORE_RELATIONSHIPS:
        stacked = _stack_relationship(by_dataset, spec["predictor"], spec["outcome"], core_datasets)
        model = _fit_interaction_model(stacked, spec["outcome_kind"])
        core_rows.extend(_rows_from_model(model, spec["predictor"], spec["outcome"], core_comparisons))

    brt_liwc_datasets = [ds for ds in by_dataset if ds == "human" or ds.endswith("_agent")]
    brt_liwc_comparisons = [ds for ds in brt_liwc_datasets if ds != "human"]
    brt_liwc_rows: list[dict[str, object]] = []
    for spec in relationship_specs_for_brt_liwc(liwc_families):
        stacked = _stack_relationship(by_dataset, spec["predictor"], spec["outcome"], brt_liwc_datasets)
        model = _fit_interaction_model(stacked, spec["outcome_kind"])
        brt_liwc_rows.extend(_rows_from_model(model, spec["predictor"], spec["outcome"], brt_liwc_comparisons))

    return pd.DataFrame(core_rows), pd.DataFrame(brt_liwc_rows)
