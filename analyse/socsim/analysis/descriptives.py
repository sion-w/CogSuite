from __future__ import annotations

import numpy as np
import pandas as pd

KEY_VARS = [
    "Age",
    "CRT2_Total",
    "BCRT_Total",
    "NFC_Total",
    "BFI_Extraversion",
    "BFI_Agreeableness",
    "BFI_Conscientiousness",
    "BFI_Neuroticism",
    "BFI_Openness",
]
DESCRIPTIVE_CATEGORICAL_VARS = [
    "Country",
    "sex",
    "Highest education level completed",
    "Ethnicity Simplified",
    "Employment Status",
    "Occupation_cat",
]
DEMOGRAPHIC_VARS = ["Age"] + DESCRIPTIVE_CATEGORICAL_VARS

CORE_DISTRIBUTION_VARS = ["CRT2_Total", "BCRT_Total", "NFC_Total"]
BFI_DISTRIBUTION_VARS = [
    "BFI_Extraversion",
    "BFI_Agreeableness",
    "BFI_Conscientiousness",
    "BFI_Neuroticism",
    "BFI_Openness",
]


def resolve_var_column(df: pd.DataFrame, var: str, dataset: str) -> str | None:
    if dataset == "human":
        if var == "CRT2_Total":
            # D-74 / CR-01 ANALYSIS_INVENTORY §5 root cause, fixed by CR-13B: the human
            # CRT2 score used as a distribution, comparison target or model outcome is the
            # FAMILIARITY-ADJUSTED one. The preference order used to be the other way
            # round, which silently sent the raw (exposure-contaminated) score into
            # distribution_divergence, comparisons, the variance ratios, the descriptives
            # overview and run_fixed_models. The raw score is still reachable, by name,
            # as `CRT2_Total_raw` immediately below.
            if "CRT2_Total_adj_observed" in df.columns:
                return "CRT2_Total_adj_observed"
            c = "CRT2_Total_observed"
            return c if c in df.columns else None
        if var == "CRT2_Total_raw":
            c = "CRT2_Total_raw_observed"
            return c if c in df.columns else "CRT2_Total_observed" if "CRT2_Total_observed" in df.columns else None
        if var == "CRT_Combined":
            c = "CRT_Combined_observed"
            return c if c in df.columns else None
        if var == "CRT_Combined_raw":
            c = "CRT_Combined_raw_observed"
            return c if c in df.columns else "CRT_Combined_observed" if "CRT_Combined_observed" in df.columns else None
        c = f"{var}_observed"
        if c in df.columns:
            return c
        return var if var in df.columns else None
    if dataset == "synthetic":
        if var == "CRT2_Total":
            if "CRT2_Total_specified" in df.columns:
                return "CRT2_Total_specified"
            c = "CRT2_Total_adj_specified"
        else:
            c = f"{var}_specified"
        if c in df.columns:
            return c
        return var if var in df.columns else None
    c = f"{var}_realised"
    return c if c in df.columns else None


def provenance_for_dataset(dataset: str) -> str:
    if dataset == "human":
        return "observed"
    if dataset == "synthetic":
        return "specified"
    return "realised"


def _section_for_var(var: str) -> str:
    return "demographic" if var in DEMOGRAPHIC_VARS else "core"


def _numeric_summary_row(dataset: str, variable: str, source_column: str, x: pd.Series) -> dict[str, object]:
    return {
        "dataset": dataset,
        "section": _section_for_var(variable),
        "variable": variable,
        "variable_type": "numeric",
        "level": np.nan,
        "source_column": source_column,
        "provenance": provenance_for_dataset(dataset),
        "N_total": int(len(x)),
        "N_non_missing": int(len(x)),
        "count": np.nan,
        "proportion": np.nan,
        "mean": float(x.mean()) if len(x) else np.nan,
        "SD": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "median": float(x.median()) if len(x) else np.nan,
        "IQR": float(x.quantile(0.75) - x.quantile(0.25)) if len(x) else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
    }


def _categorical_summary_rows(dataset: str, variable: str, source_column: str, series: pd.Series) -> list[dict[str, object]]:
    clean = series.astype("object")
    clean = clean.where(clean.notna(), np.nan)
    clean = clean.map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
    clean = clean.mask(clean == "", np.nan)
    counts = clean.dropna().value_counts(dropna=True)
    n_non_missing = int(clean.notna().sum())
    n_total = int(len(clean))
    rows: list[dict[str, object]] = []
    for level, count in counts.items():
        rows.append(
            {
                "dataset": dataset,
                "section": _section_for_var(variable),
                "variable": variable,
                "variable_type": "categorical",
                "level": str(level),
                "source_column": source_column,
                "provenance": provenance_for_dataset(dataset),
                "N_total": n_total,
                "N_non_missing": n_non_missing,
                "count": int(count),
                "proportion": float(count / n_non_missing) if n_non_missing else np.nan,
                "mean": np.nan,
                "SD": np.nan,
                "median": np.nan,
                "IQR": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        )
    return rows


def make_descriptives_overview(by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, df in by_dataset.items():
        for var in KEY_VARS:
            col = resolve_var_column(df, var, dataset)
            if col is None:
                continue
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            rows.append(_numeric_summary_row(dataset, var, col, x))
        for var in DESCRIPTIVE_CATEGORICAL_VARS:
            col = resolve_var_column(df, var, dataset)
            if col is None:
                continue
            rows.extend(_categorical_summary_rows(dataset, var, col, df[col]))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["section", "variable_type", "variable", "dataset", "level"], na_position="last").reset_index(drop=True)
