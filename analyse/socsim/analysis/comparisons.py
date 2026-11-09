from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from socsim.analysis.descriptives import KEY_VARS, provenance_for_dataset, resolve_var_column

DISCRETE_LEVELS: dict[str, list[int]] = {
    "CRT2_Total": [0, 1, 2, 3, 4],
    "BCRT_Total": [0, 1, 2, 3, 4],
    "BFI_Extraversion": [1, 2, 3, 4, 5],
    "BFI_Agreeableness": [1, 2, 3, 4, 5],
    "BFI_Conscientiousness": [1, 2, 3, 4, 5],
    "BFI_Neuroticism": [1, 2, 3, 4, 5],
    "BFI_Openness": [1, 2, 3, 4, 5],
}


def _discretise_for_counts(var: str, x: pd.Series) -> tuple[pd.Series, list[int]]:
    if var in {"CRT2_Total", "BCRT_Total"}:
        levels = [0, 1, 2, 3, 4]
        z = x.round().clip(lower=0, upper=4)
        return z, levels
    if var in {"NFC_Total", "BFI_Extraversion", "BFI_Agreeableness", "BFI_Conscientiousness", "BFI_Neuroticism", "BFI_Openness"}:
        levels = [1, 2, 3, 4, 5]
        z = x.round().clip(lower=1, upper=5)
        return z, levels
    levels = DISCRETE_LEVELS.get(var, [])
    return x, levels


def _mean_ci95(x: pd.Series) -> tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)


def overlap_summary(by_dataset: dict[str, pd.DataFrame], variables: list[str]) -> pd.DataFrame:
    rows = []
    for dataset, df in by_dataset.items():
        for var in variables:
            col = resolve_var_column(df, var, dataset)
            if not col:
                continue
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            ci_low, ci_high = _mean_ci95(x)
            x_disc, levels = _discretise_for_counts(var, x)
            row: dict[str, object] = {
                "dataset": dataset,
                "variable": var,
                "source_column": col,
                "provenance": provenance_for_dataset(dataset),
                "N": int(len(x)),
                "mean": float(x.mean()) if len(x) else np.nan,
                "SD": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                "median": float(x.median()) if len(x) else np.nan,
                "IQR": float(x.quantile(0.75) - x.quantile(0.25)) if len(x) else np.nan,
                "CI95_low": ci_low,
                "CI95_high": ci_high,
            }
            if len(x):
                counts = x_disc.value_counts(dropna=True)
                for level in levels:
                    c = int(counts.get(level, 0))
                    row[f"count_{level}"] = c
                    row[f"prop_{level}"] = float(c / len(x_disc))
            rows.append(
                row
            )
    return pd.DataFrame(rows)


def pairwise_univariate_comparisons(by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for d1, d2 in itertools.combinations(by_dataset.keys(), 2):
        for var in KEY_VARS:
            c1 = resolve_var_column(by_dataset[d1], var, d1)
            c2 = resolve_var_column(by_dataset[d2], var, d2)
            if not c1 or not c2:
                continue
            x = pd.to_numeric(by_dataset[d1][c1], errors="coerce").dropna()
            y = pd.to_numeric(by_dataset[d2][c2], errors="coerce").dropna()
            warning = ""
            if len(x) < 3 or len(y) < 3 or x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
                warning = "N small or variance=0; interpret cautiously."
            t = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit") if len(x) and len(y) else None
            pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / max(1, (len(x) + len(y) - 2))) if len(x) > 1 and len(y) > 1 else np.nan
            d = (x.mean() - y.mean()) / pooled if pooled and pooled > 0 else np.nan
            se_diff = np.sqrt((x.var(ddof=1) / len(x)) + (y.var(ddof=1) / len(y))) if len(x) > 1 and len(y) > 1 else np.nan
            diff = float(x.mean() - y.mean()) if len(x) and len(y) else np.nan
            rows.append(
                {
                    "dataset_a": d1,
                    "dataset_b": d2,
                    "variable": var,
                    "N_a": int(len(x)),
                    "N_b": int(len(y)),
                    "mean_a": float(x.mean()) if len(x) else np.nan,
                    "mean_b": float(y.mean()) if len(y) else np.nan,
                    "mean_diff": diff,
                    "mean_diff_CI95_low": diff - 1.96 * se_diff if pd.notna(se_diff) else np.nan,
                    "mean_diff_CI95_high": diff + 1.96 * se_diff if pd.notna(se_diff) else np.nan,
                    "cohens_d": float(d) if pd.notna(d) else np.nan,
                    "p_value": float(t.pvalue) if t is not None and pd.notna(t.pvalue) else np.nan,
                    "warning": warning,
                }
            )
    return pd.DataFrame(rows)
