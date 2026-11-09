from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


PAIR_MAP = [
    {
        "construct": "CRT2",
        "xcol": "CRT2_Total_adj_specified",
        "ycol": "CRT2_Total_realised",
        "x_axis_label": "Specified CRT2 (adjusted)",
        "relation_note": "direct specified construct",
    },
    {
        "construct": "BCRT_proxy",
        "xcol": "CRT2_Total_adj_specified",
        "ycol": "BCRT_Total_realised",
        "x_axis_label": "Specified CRT2 adjusted (proxy specified cognitive-reflection parameter)",
        "relation_note": "BCRT has no direct specified parameter; CRT2_Total_adj_specified used as proxy specified predictor",
    },
    {
        "construct": "NFC",
        "xcol": "NFC_Total_specified",
        "ycol": "NFC_Total_realised",
        "x_axis_label": "Specified NFC",
        "relation_note": "direct specified construct",
    },
    {
        "construct": "BFI_E",
        "xcol": "BFI_Extraversion_specified",
        "ycol": "BFI_Extraversion_realised",
        "x_axis_label": "Specified BFI Extraversion",
        "relation_note": "direct specified construct",
    },
    {
        "construct": "BFI_A",
        "xcol": "BFI_Agreeableness_specified",
        "ycol": "BFI_Agreeableness_realised",
        "x_axis_label": "Specified BFI Agreeableness",
        "relation_note": "direct specified construct",
    },
    {
        "construct": "BFI_C",
        "xcol": "BFI_Conscientiousness_specified",
        "ycol": "BFI_Conscientiousness_realised",
        "x_axis_label": "Specified BFI Conscientiousness",
        "relation_note": "direct specified construct",
    },
    {
        "construct": "BFI_N",
        "xcol": "BFI_Neuroticism_specified",
        "ycol": "BFI_Neuroticism_realised",
        "x_axis_label": "Specified BFI Neuroticism",
        "relation_note": "direct specified construct",
    },
    {
        "construct": "BFI_O",
        "xcol": "BFI_Openness_specified",
        "ycol": "BFI_Openness_realised",
        "x_axis_label": "Specified BFI Openness",
        "relation_note": "direct specified construct",
    },
]


def _pearson_ci(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float]:
    xr, yr = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
    pair = pd.DataFrame({"x": xr, "y": yr}).dropna()
    if len(pair) < 3 or pair["x"].std(ddof=1) == 0 or pair["y"].std(ddof=1) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r, p = stats.pearsonr(pair["x"], pair["y"])
    z = np.arctanh(r)
    se = 1 / np.sqrt(len(pair) - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return float(r), float(lo), float(hi), float(p)


def calibration_tables(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    error_rows = []
    for spec in PAIR_MAP:
        label, xcol, ycol = spec["construct"], spec["xcol"], spec["ycol"]
        warning = ""
        if xcol not in df.columns or ycol not in df.columns:
            pair_rows.append(
                {
                    "dataset": dataset,
                    "construct": label,
                    "N": 0,
                    "Pearson_r": np.nan,
                    "r_CI95_low": np.nan,
                    "r_CI95_high": np.nan,
                    "r_p": np.nan,
                    "slope": np.nan,
                    "slope_CI95_low": np.nan,
                    "slope_CI95_high": np.nan,
                    "slope_p": np.nan,
                    "intercept": np.nan,
                    "R2": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "median_abs_error": np.nan,
                    "warning": f"Missing columns: {xcol}, {ycol}",
                    "specified_predictor_col": xcol,
                    "realised_outcome_col": ycol,
                    "specified_label": spec["x_axis_label"],
                    "realised_label": "Realised behavioural expression",
                    "relation_note": spec["relation_note"],
                }
            )
            continue
        work = pd.DataFrame({"entity_id": df["entity_id"], "x": pd.to_numeric(df[xcol], errors="coerce"), "y": pd.to_numeric(df[ycol], errors="coerce")}).dropna()
        n = len(work)
        if n < 3 or work["x"].std(ddof=1) == 0 or work["y"].std(ddof=1) == 0:
            warning = "N small or variance=0; interpret cautiously."
            slope = intercept = r2 = slope_p = s_lo = s_hi = np.nan
            r, r_lo, r_hi, r_p = _pearson_ci(work["x"], work["y"])
        else:
            X = sm.add_constant(work["x"], has_constant="add")
            model = sm.OLS(work["y"], X).fit()
            ci = model.conf_int().loc["x"]
            slope = float(model.params["x"])
            intercept = float(model.params["const"])
            r2 = float(model.rsquared)
            slope_p = float(model.pvalues["x"])
            s_lo, s_hi = float(ci.iloc[0]), float(ci.iloc[1])
            r, r_lo, r_hi, r_p = _pearson_ci(work["x"], work["y"])
        err = work["y"] - work["x"]
        pair_rows.append(
            {
                "dataset": dataset,
                "construct": label,
                "N": int(n),
                "Pearson_r": r,
                "r_CI95_low": r_lo,
                "r_CI95_high": r_hi,
                "r_p": r_p,
                "slope": slope,
                "slope_CI95_low": s_lo,
                "slope_CI95_high": s_hi,
                "slope_p": slope_p,
                "intercept": intercept,
                "R2": r2,
                "MAE": float(err.abs().mean()) if n else np.nan,
                "RMSE": float(np.sqrt((err**2).mean())) if n else np.nan,
                "median_abs_error": float(err.abs().median()) if n else np.nan,
                "warning": warning,
                "specified_predictor_col": xcol,
                "realised_outcome_col": ycol,
                "specified_label": spec["x_axis_label"],
                "realised_label": "Realised behavioural expression",
                "relation_note": spec["relation_note"],
            }
        )
        for _, row in work.iterrows():
            e = float(row["y"] - row["x"])
            error_rows.append(
                {
                    "entity_id": row["entity_id"],
                    "construct": label,
                    "specified_value": float(row["x"]),
                    "realised_value": float(row["y"]),
                    "error": e,
                    "abs_error": abs(e),
                    "specified_predictor_col": xcol,
                    "realised_outcome_col": ycol,
                    "specified_label": spec["x_axis_label"],
                    "realised_label": "Realised behavioural expression",
                    "relation_note": spec["relation_note"],
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    summary_df = pair_df[
        ["construct", "N", "Pearson_r", "r_CI95_low", "r_CI95_high", "r_p", "slope", "slope_CI95_low", "slope_CI95_high", "slope_p", "intercept", "R2", "MAE", "RMSE", "median_abs_error", "warning"]
    ].copy()
    construct_order = ["CRT2", "BCRT_proxy", "NFC", "BFI_E", "BFI_A", "BFI_C", "BFI_N", "BFI_O"]
    summary_df["construct"] = pd.Categorical(summary_df["construct"], categories=construct_order, ordered=True)
    summary_df = summary_df.sort_values("construct").reset_index(drop=True)
    summary_df["construct"] = summary_df["construct"].astype(str)
    return pair_df, summary_df, pd.DataFrame(error_rows)
