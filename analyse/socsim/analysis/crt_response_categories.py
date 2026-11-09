from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from socsim.scoring.crt_categories import CRT_CATEGORY_LEVELS, CRT_INSTRUMENT_ITEMS

# CRT2 specified parameter candidates. bCRT uses the same list: bCRT is not
# written into the persona prompt, so CRT2_Total_adj_specified is its proxy
# specified level (see the calibration note in the README). Do not key on
# BCRT_Total_specified: that column is a mechanical ingest suffix of the
# source human's BCRT_Total and is never shown to the model.
_CRT2_SPECIFIED_CANDIDATES = ["CRT2_Total_adj_specified", "CRT2_Total_specified"]


def _run_id_series(df: pd.DataFrame, dataset: str) -> pd.Series:
    if "run_id" in df.columns:
        return df["run_id"].fillna(dataset).astype(str)
    if "socsim_run_label" in df.columns:
        return df["socsim_run_label"].fillna(dataset).astype(str)
    return pd.Series(dataset, index=df.index, dtype="object")


def _specified_level_and_source(df: pd.DataFrame, instrument: str) -> tuple[pd.Series, pd.Series]:
    candidates = {
        "crt2": _CRT2_SPECIFIED_CANDIDATES,
        "bcrt": _CRT2_SPECIFIED_CANDIDATES,
    }.get(instrument, [])
    for col in candidates:
        if col in df.columns:
            level = pd.to_numeric(df[col], errors="coerce").round().clip(lower=0, upper=4)
            source = pd.Series(
                np.where(level.notna(), col, pd.NA),
                index=df.index,
                dtype=object,
            )
            return level, source
    return (
        pd.Series(np.nan, index=df.index, dtype=float),
        pd.Series(pd.NA, index=df.index, dtype=object),
    )


def _specified_level(df: pd.DataFrame, instrument: str) -> pd.Series:
    level, _source = _specified_level_and_source(df, instrument)
    return level


def crt_response_category_long(by_dataset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    empty_cols = [
        "dataset",
        "run_id",
        "entity_id",
        "instrument",
        "item_id",
        "specified_level",
        "specified_level_source",
        "response_category",
    ]
    for dataset, df in by_dataset.items():
        if dataset == "synthetic":
            continue
        run_ids = _run_id_series(df, dataset)
        for instrument, item_ids in CRT_INSTRUMENT_ITEMS.items():
            specified_level, specified_level_source = _specified_level_and_source(df, instrument)
            for item_id in item_ids:
                category_col = f"{item_id}_response_category"
                if category_col not in df.columns:
                    continue
                item_rows = pd.DataFrame(
                    {
                        "dataset": dataset,
                        "run_id": run_ids,
                        "entity_id": df["entity_id"].astype(str) if "entity_id" in df.columns else df.index.astype(str),
                        "instrument": instrument,
                        "item_id": item_id,
                        "specified_level": specified_level,
                        "specified_level_source": specified_level_source,
                        "response_category": df[category_col].fillna("missing_or_unparseable").astype(str),
                    }
                )
                rows.append(item_rows)
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    out = pd.concat(rows, ignore_index=True)
    out.loc[~out["response_category"].isin(CRT_CATEGORY_LEVELS), "response_category"] = "missing_or_unparseable"
    return out


def response_category_distribution(category_long: pd.DataFrame) -> pd.DataFrame:
    if category_long.empty:
        return pd.DataFrame(columns=["dataset", "run_id", "instrument", "item_id", "response_category", "count", "proportion"])
    group_cols = ["dataset", "run_id", "instrument", "item_id", "response_category"]
    counts = category_long.groupby(group_cols, dropna=False).size().rename("count").reset_index()
    totals = counts.groupby(["dataset", "run_id", "instrument", "item_id"], dropna=False)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    return counts.sort_values(["instrument", "item_id", "dataset", "run_id", "response_category"]).reset_index(drop=True)


def response_category_distribution_by_specified_level(category_long: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["dataset", "run_id", "instrument", "specified_level", "response_category", "count", "proportion"]
    if "specified_level_source" in category_long.columns:
        base_cols = [
            "dataset",
            "run_id",
            "instrument",
            "specified_level",
            "specified_level_source",
            "response_category",
            "count",
            "proportion",
        ]
    if category_long.empty:
        return pd.DataFrame(columns=base_cols)
    work = category_long[category_long["specified_level"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=base_cols)
    work["specified_level"] = work["specified_level"].astype(int)
    group_cols = ["dataset", "run_id", "instrument", "specified_level", "response_category"]
    if "specified_level_source" in work.columns:
        group_cols = [
            "dataset",
            "run_id",
            "instrument",
            "specified_level",
            "specified_level_source",
            "response_category",
        ]
    counts = work.groupby(group_cols, dropna=False).size().rename("count").reset_index()
    total_keys = ["dataset", "run_id", "instrument", "specified_level"]
    if "specified_level_source" in counts.columns:
        total_keys = ["dataset", "run_id", "instrument", "specified_level", "specified_level_source"]
    totals = counts.groupby(total_keys, dropna=False)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    sort_cols = ["instrument", "specified_level", "dataset", "run_id", "response_category"]
    return counts.sort_values(sort_cols).reset_index(drop=True)


def response_category_divergence(category_long: pd.DataFrame) -> pd.DataFrame:
    if category_long.empty or "human" not in set(category_long["dataset"]):
        return pd.DataFrame(columns=["comparison", "run_id", "instrument", "js_divergence", "l1_distance"])
    rows: list[dict[str, object]] = []
    human = category_long[category_long["dataset"] == "human"]
    comparisons = category_long[category_long["dataset"].ne("human") & category_long["dataset"].str.endswith("_agent")]
    for instrument in sorted(category_long["instrument"].dropna().unique()):
        human_inst = human[human["instrument"] == instrument]
        if human_inst.empty:
            continue
        human_probs = _category_probs(human_inst)
        for (dataset, run_id), sim_inst in comparisons[comparisons["instrument"] == instrument].groupby(["dataset", "run_id"], dropna=False):
            sim_probs = _category_probs(sim_inst)
            row: dict[str, object] = {
                "comparison": dataset,
                "run_id": run_id,
                "instrument": instrument,
                "js_divergence": float(jensenshannon(human_probs, sim_probs)),
                "l1_distance": float(np.abs(sim_probs - human_probs).sum()),
            }
            for category, human_p, sim_p in zip(CRT_CATEGORY_LEVELS, human_probs, sim_probs, strict=False):
                row[f"human_prop_{category}"] = float(human_p)
                row[f"sim_prop_{category}"] = float(sim_p)
                row[f"delta_prop_{category}"] = float(sim_p - human_p)
            rows.append(row)
    return pd.DataFrame(rows)


def _category_probs(df: pd.DataFrame) -> np.ndarray:
    counts = df["response_category"].value_counts(dropna=False)
    total = float(counts.sum())
    if total <= 0:
        return np.zeros(len(CRT_CATEGORY_LEVELS), dtype=float)
    return np.array([float(counts.get(category, 0.0)) / total for category in CRT_CATEGORY_LEVELS], dtype=float)
