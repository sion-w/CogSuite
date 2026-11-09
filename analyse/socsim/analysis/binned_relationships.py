from __future__ import annotations

import numpy as np
import pandas as pd

from socsim.analysis.brt import _ensure_brt_derived
from socsim.analysis.descriptives import resolve_var_column

CORE_RELATIONSHIPS = [
    {"predictor": "NFC_Total", "outcome": "CRT2_Total", "outcome_kind": "count"},
    {"predictor": "BFI_Conscientiousness", "outcome": "CRT2_Total", "outcome_kind": "count"},
    {"predictor": "NFC_Total", "outcome": "BCRT_Total", "outcome_kind": "count"},
    {"predictor": "BFI_Conscientiousness", "outcome": "BCRT_Total", "outcome_kind": "count"},
]

BRT_DECISION_OUTCOMES = ["BRT1_keep_booking_bin", "BRT2_take_medex_bin"]


def relationship_specs_for_brt_liwc(liwc_families: dict[str, list[str]]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    liwc_outcomes = sorted({liwc_var for family_vars in liwc_families.values() for liwc_var in family_vars})
    for predictor in ["CRT2_Total", "BCRT_Total"]:
        for outcome in liwc_outcomes:
            specs.append({"predictor": predictor, "outcome": outcome, "outcome_kind": "continuous"})
        for outcome in BRT_DECISION_OUTCOMES:
            specs.append({"predictor": predictor, "outcome": outcome, "outcome_kind": "binary"})
    return specs


def _resolve_cognitive_predictor_column(df: pd.DataFrame, dataset: str, predictor: str) -> str | None:
    if predictor == "CRT2_Total":
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
        return "CRT2_Total_realised" if "CRT2_Total_realised" in df.columns else None
    return resolve_var_column(df, predictor, dataset)


def _resolve_outcome_column(df: pd.DataFrame, dataset: str, outcome: str) -> tuple[pd.DataFrame, str | None]:
    if outcome in BRT_DECISION_OUTCOMES:
        work = _ensure_brt_derived(df)
        return work, outcome if outcome in work.columns else None
    # The resolver runs before the same-name shortcut, so the cognitive totals resolve to the
    # familiarity-adjusted human score and the realised simulated score, never to the human
    # frame's literal raw `CRT2_Total` column; raw and adjusted scores are therefore never
    # mixed in one binned-relationship figure or RMSE row. The same-name shortcut serves
    # every other outcome.
    resolved = resolve_var_column(df, outcome, dataset)
    if resolved:
        return df, resolved
    if outcome in df.columns:
        return df, outcome
    return df, None


def _stack_relationship(
    by_dataset: dict[str, pd.DataFrame],
    predictor: str,
    outcome: str,
    datasets: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for dataset in datasets:
        df = by_dataset.get(dataset)
        if df is None:
            continue
        predictor_col = _resolve_cognitive_predictor_column(df, dataset, predictor) if predictor in {"CRT2_Total", "BCRT_Total"} else resolve_var_column(df, predictor, dataset)
        outcome_df, outcome_col = _resolve_outcome_column(df, dataset, outcome)
        if not predictor_col or not outcome_col:
            continue
        work = pd.DataFrame(
            {
                "dataset": dataset,
                "x": pd.to_numeric(outcome_df[predictor_col], errors="coerce"),
                "y": pd.to_numeric(outcome_df[outcome_col], errors="coerce"),
            }
        ).dropna()
        if work.empty:
            continue
        rows.append(work)
    if not rows:
        return pd.DataFrame(columns=["dataset", "x", "y"])
    return pd.concat(rows, ignore_index=True)


def _bin_edges(values: pd.Series, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    return edges


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    if n < 2:
        return mean, np.nan, np.nan
    se = values.std(ddof=1) / np.sqrt(n)
    ci_half = 1.96 * se
    return mean, float(mean - ci_half), float(mean + ci_half)


def _summarise_bins(
    stacked: pd.DataFrame,
    predictor: str,
    outcome: str,
    n_bins: int,
) -> pd.DataFrame:
    if stacked.empty or stacked["x"].nunique(dropna=True) < 2:
        return pd.DataFrame(columns=["predictor", "outcome", "dataset", "bin", "bin_center", "mean_outcome", "ci_low", "ci_high"])
    edges = _bin_edges(stacked["x"], n_bins=n_bins)
    if len(edges) < 3:
        return pd.DataFrame(columns=["predictor", "outcome", "dataset", "bin", "bin_center", "mean_outcome", "ci_low", "ci_high"])

    work = stacked.copy()
    work["bin"] = pd.cut(work["x"], bins=edges, include_lowest=True, labels=False, duplicates="drop")
    work = work.dropna(subset=["bin"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["predictor", "outcome", "dataset", "bin", "bin_center", "mean_outcome", "ci_low", "ci_high"])
    work["bin"] = work["bin"].astype(int) + 1

    centers = {idx + 1: float((edges[idx] + edges[idx + 1]) / 2.0) for idx in range(len(edges) - 1)}
    rows: list[dict[str, object]] = []
    grouped = work.groupby(["dataset", "bin"], observed=True)
    for (dataset, bin_id), group in grouped:
        mean, ci_low, ci_high = _mean_ci(group["y"])
        rows.append(
            {
                "predictor": predictor,
                "outcome": outcome,
                "dataset": dataset,
                "bin": int(bin_id),
                "bin_center": centers.get(int(bin_id), np.nan),
                "mean_outcome": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def _rmse_against_human(bin_table: pd.DataFrame) -> pd.DataFrame:
    if bin_table.empty:
        return pd.DataFrame(columns=["predictor", "outcome", "comparison", "rmse_bin"])

    rows: list[dict[str, object]] = []
    for (predictor, outcome), rel_df in bin_table.groupby(["predictor", "outcome"], observed=True):
        human = rel_df[rel_df["dataset"] == "human"][["bin", "mean_outcome"]].rename(columns={"mean_outcome": "human_mean"})
        if human.empty:
            continue
        for comparison in [ds for ds in rel_df["dataset"].unique() if ds != "human"]:
            sim = rel_df[rel_df["dataset"] == comparison][["bin", "mean_outcome"]].rename(columns={"mean_outcome": "sim_mean"})
            merged = human.merge(sim, on="bin", how="inner").dropna()
            if merged.empty:
                continue
            rmse = float(np.sqrt(np.mean(np.square(merged["sim_mean"] - merged["human_mean"]))))
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "comparison": comparison,
                    "rmse_bin": rmse,
                }
            )
    return pd.DataFrame(rows)


def build_core_binned_relationships(
    by_dataset: dict[str, pd.DataFrame],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = [ds for ds in by_dataset if ds == "human" or ds == "synthetic" or ds.endswith("_agent")]
    tables = []
    for spec in CORE_RELATIONSHIPS:
        stacked = _stack_relationship(by_dataset, spec["predictor"], spec["outcome"], datasets)
        tables.append(_summarise_bins(stacked, spec["predictor"], spec["outcome"], n_bins=n_bins))
    bin_table = pd.concat([t for t in tables if not t.empty], ignore_index=True) if any(not t.empty for t in tables) else pd.DataFrame()
    rmse_table = _rmse_against_human(bin_table)
    return bin_table, rmse_table


def build_brt_liwc_binned_relationships(
    by_dataset: dict[str, pd.DataFrame],
    liwc_families: dict[str, list[str]],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = [ds for ds in by_dataset if ds == "human" or ds.endswith("_agent")]
    tables = []
    for spec in relationship_specs_for_brt_liwc(liwc_families):
        stacked = _stack_relationship(by_dataset, spec["predictor"], spec["outcome"], datasets)
        tables.append(_summarise_bins(stacked, spec["predictor"], spec["outcome"], n_bins=n_bins))
    bin_table = pd.concat([t for t in tables if not t.empty], ignore_index=True) if any(not t.empty for t in tables) else pd.DataFrame()
    rmse_table = _rmse_against_human(bin_table)
    return bin_table, rmse_table
