from __future__ import annotations

import pandas as pd

from socsim.common import require_columns
from socsim.scoring.crt_categories import add_item_response_categories, add_person_category_totals


def compute_crt_totals(scored_items: pd.DataFrame, person_base: pd.DataFrame) -> pd.DataFrame:
    require_columns(scored_items, ["entity_id", "item_id", "final_correct", "final_ordinal012"], "scored_items")
    require_columns(person_base, ["entity_id"], "person_base")

    agg = add_item_response_categories(scored_items)
    agg["item_id"] = agg["item_id"].astype(str)
    agg["entity_id"] = agg["entity_id"].astype(str)
    crt2 = (
        agg.loc[agg["item_id"].str.startswith("crt2_"), ["entity_id", "final_correct"]]
        .groupby("entity_id", dropna=False)["final_correct"]
        .sum(min_count=1)
        .rename("CRT2_Total_realised")
    )
    bcrt = (
        agg.loc[agg["item_id"].str.startswith("bcrt_"), ["entity_id", "final_correct"]]
        .groupby("entity_id", dropna=False)["final_correct"]
        .sum(min_count=1)
        .rename("BCRT_Total_realised")
    )
    dcrt = (
        agg.loc[agg["item_id"].str.startswith("dcrt_"), ["entity_id", "final_correct"]]
        .groupby("entity_id", dropna=False)["final_correct"]
        .sum(min_count=1)
        .rename("DCRT_Total_realised")
    )
    totals = pd.concat([crt2, bcrt, dcrt], axis=1).reset_index()
    totals["CRT_Combined_realised"] = totals[["CRT2_Total_realised", "BCRT_Total_realised", "DCRT_Total_realised"]].sum(axis=1, min_count=1)
    category_totals = add_person_category_totals(
        agg.pivot_table(
            index="entity_id",
            columns="item_id",
            values="response_category",
            aggfunc="first",
        )
        .rename(columns=lambda item_id: f"{item_id}_response_category")
        .reset_index()
    )
    category_total_cols = [
        "entity_id",
        "crt2_correct_total",
        "crt2_intuitive_total",
        "crt2_other_total",
        "bcrt_correct_total",
        "bcrt_intuitive_total",
        "bcrt_other_total",
    ]
    totals = totals.merge(category_totals[[c for c in category_total_cols if c in category_totals.columns]], on="entity_id", how="left")

    wide = agg.pivot_table(
        index="entity_id",
        columns="item_id",
        values=[
            "llm_label",
            "llm_correct",
            "llm_ordinal012",
            "llm_reason",
            "final_correct",
            "final_ordinal012",
            "response_category",
            "final_response_category",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{item}_{metric}" for metric, item in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    rename_map = {}
    for c in wide.columns:
        c2 = c
        c2 = c2.replace("_llm_label", "_llm_label")
        c2 = c2.replace("_llm_correct", "_llm_correct")
        c2 = c2.replace("_llm_ordinal012", "_llm_ordinal012")
        c2 = c2.replace("_llm_reason", "_llm_reason")
        c2 = c2.replace("_final_correct", "_final_correct")
        c2 = c2.replace("_final_ordinal012", "_final_ordinal012")
        c2 = c2.replace("_response_category", "_response_category")
        c2 = c2.replace("_final_response_category", "_final_response_category")
        rename_map[c] = c2
    wide = wide.rename(columns=rename_map)

    base = person_base.copy()
    base["entity_id"] = base["entity_id"].astype(str)
    out = base.merge(totals, on="entity_id", how="left").merge(wide, on="entity_id", how="left")
    return out
