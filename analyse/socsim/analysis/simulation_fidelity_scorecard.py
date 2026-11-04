from __future__ import annotations

import numpy as np
import pandas as pd

from socsim.analysis.binned_relationships import _stack_relationship


def _sign_matches(a: float, b: float, tol: float = 1e-9) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    if abs(a) <= tol and abs(b) <= tol:
        return True
    return np.sign(a) == np.sign(b)


def _distribution_judgement(row: pd.Series) -> str:
    js = row.get("js_divergence", np.nan)
    d_value = abs(row.get("cohens_d", np.nan))
    if pd.notna(js) and js < 0.10 and pd.notna(d_value) and d_value < 0.20:
        return "GOOD"
    if pd.notna(js) and js < 0.20 and pd.notna(d_value) and d_value < 0.50:
        return "MODERATE"
    return "POOR"


def _relationship_judgement(row: pd.Series, human_sd: float) -> str:
    direction_matches = _sign_matches(row.get("human_value", np.nan), row.get("sim_value", np.nan))
    interaction_p = row.get("interaction_p", np.nan)
    rmse = row.get("rmse_bin", np.nan)
    if not direction_matches:
        return "POOR"

    good_rmse = pd.notna(human_sd) and human_sd > 0 and pd.notna(rmse) and rmse < (0.15 * human_sd)
    moderate_rmse = pd.notna(human_sd) and human_sd > 0 and pd.notna(rmse) and rmse < (0.30 * human_sd)
    if pd.notna(interaction_p) and interaction_p > 0.05 and good_rmse:
        return "GOOD"
    if (pd.notna(interaction_p) and interaction_p > 0.05) or moderate_rmse:
        return "MODERATE"
    return "POOR"


def _human_outcome_sd(
    by_dataset: dict[str, pd.DataFrame],
    predictor: str,
    outcome: str,
) -> float:
    stacked = _stack_relationship(by_dataset, predictor, outcome, ["human"])
    if stacked.empty or len(stacked["y"]) < 2:
        return np.nan
    return float(stacked["y"].std(ddof=1))


def build_simulation_fidelity_scorecard(
    by_dataset: dict[str, pd.DataFrame],
    distribution_df: pd.DataFrame,
    core_slopes_df: pd.DataFrame,
    brt_liwc_slopes_df: pd.DataFrame,
    core_rmse_df: pd.DataFrame,
    brt_liwc_rmse_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if distribution_df is not None and not distribution_df.empty:
        for _, row in distribution_df.iterrows():
            rows.append(
                {
                    "evaluation_type": "marginal_distribution",
                    "variable_or_relationship": f"{row['variable']} ({row['comparison']})",
                    "human_value": row["mean_human"],
                    "sim_value": row["mean_sim"],
                    "difference_metric": row["mean_diff"],
                    "js_divergence": row["js_divergence"],
                    "cohens_d": row["cohens_d"],
                    "interaction_p": np.nan,
                    "rmse_bin": np.nan,
                    "judgement": _distribution_judgement(row),
                }
            )

    rmse_lookup = {}
    if core_rmse_df is not None and not core_rmse_df.empty:
        for _, row in core_rmse_df.iterrows():
            rmse_lookup[(row["predictor"], row["outcome"], row["comparison"])] = row["rmse_bin"]
    if brt_liwc_rmse_df is not None and not brt_liwc_rmse_df.empty:
        for _, row in brt_liwc_rmse_df.iterrows():
            rmse_lookup[(row["predictor"], row["outcome"], row["comparison"])] = row["rmse_bin"]

    relationship_frames = [
        ("association_fidelity", core_slopes_df if core_slopes_df is not None else pd.DataFrame()),
        ("brt_liwc_relationship", brt_liwc_slopes_df if brt_liwc_slopes_df is not None else pd.DataFrame()),
    ]
    for evaluation_type, frame in relationship_frames:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            human_sd = _human_outcome_sd(by_dataset, row["predictor"], row["outcome"])
            scorecard_row = {
                "evaluation_type": evaluation_type,
                "variable_or_relationship": f"{row['predictor']} -> {row['outcome']} ({row['comparison']})",
                "human_value": row["slope_human"],
                "sim_value": row["slope_sim"],
                "difference_metric": row["interaction_beta"],
                "js_divergence": np.nan,
                "cohens_d": np.nan,
                "interaction_p": row["interaction_p"],
                "rmse_bin": rmse_lookup.get((row["predictor"], row["outcome"], row["comparison"]), np.nan),
            }
            scorecard_row["judgement"] = _relationship_judgement(pd.Series(scorecard_row), human_sd)
            rows.append(scorecard_row)

    return pd.DataFrame(rows)
