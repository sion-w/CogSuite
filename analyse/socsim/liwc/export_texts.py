from __future__ import annotations

from pathlib import Path

import pandas as pd

from socsim.common import ensure_dir, require_columns


def export_agent_brt_texts(
    items_df: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    run_label: str | None = None,
    write_legacy_alias: bool = True,
) -> Path:
    require_columns(items_df, ["entity_id", "item_id", "response_text"], f"{dataset}_items")
    ensure_dir(out_dir)
    texts = (
        items_df.loc[items_df["item_id"].astype(str).isin(["brt_1", "brt_2"]), ["entity_id", "item_id", "response_text"]]
        .pivot_table(index="entity_id", columns="item_id", values="response_text", aggfunc="first")
        .reset_index()
    )
    if "brt_1" not in texts.columns:
        texts["brt_1"] = ""
    if "brt_2" not in texts.columns:
        texts["brt_2"] = ""
    texts["brt_concat"] = texts["brt_1"].fillna("").astype(str) + "\n\n" + texts["brt_2"].fillna("").astype(str)
    stem = f"{dataset}_{run_label}" if run_label else dataset
    out_path = out_dir / f"{stem}_liwc_text_export.csv"
    texts.to_csv(out_path, index=False)
    legacy_path = out_dir / f"{dataset}_liwc_text_export.csv"
    if write_legacy_alias and run_label and legacy_path != out_path:
        texts.to_csv(legacy_path, index=False)
    return out_path
