from __future__ import annotations

import re

import numpy as np
import pandas as pd

from socsim.common import require_columns


def _item_index(item_id: str) -> float:
    m = re.search(r"(\d+)$", str(item_id))
    return float(m.group(1)) if m else np.nan


def _rev5(x: pd.Series) -> pd.Series:
    return 6 - x


def score_likert_bfi_nfc(items_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    require_columns(items_df, ["entity_id", "item_id", "item_type", "response_numeric"], "agent_items")
    bfi_cfg = cfg["bfi"]
    nfc_cfg = cfg["nfc"]
    out = items_df.copy()
    out["response_numeric"] = pd.to_numeric(out["response_numeric"], errors="coerce")
    out["item_idx"] = out["item_id"].astype(str).map(_item_index)

    bfi_mask = (out["item_type"] == "likert") & (out["item_idx"].between(1, 10))
    if "test_name" in out.columns:
        bfi_mask = bfi_mask & out["test_name"].fillna("").str.upper().isin(["BFI"])
    if "test_id" in out.columns:
        bfi_mask = bfi_mask | (
            (out["item_type"] == "likert")
            & out["item_idx"].between(1, 10)
            & out["test_id"].fillna("").str.lower().eq("bfi10")
        )
    bfi = out.loc[bfi_mask, ["entity_id", "item_idx", "response_numeric"]].copy()
    bfi["score"] = bfi["response_numeric"]
    bfi.loc[bfi["item_idx"].isin([float(v) for v in bfi_cfg["reverse_items"]]), "score"] = _rev5(
        bfi.loc[bfi["item_idx"].isin([float(v) for v in bfi_cfg["reverse_items"]]), "response_numeric"]
    )

    result = pd.DataFrame({"entity_id": out["entity_id"].drop_duplicates().sort_values()})
    for trait, ids in bfi_cfg["keys"].items():
        idx = [float(v) for v in ids]
        trait_vals = bfi.loc[bfi["item_idx"].isin(idx)].groupby("entity_id")["score"].mean()
        result[f"BFI_{trait}_realised"] = result["entity_id"].map(trait_vals)

    nfc_mask = (out["item_type"] == "likert") & (out["item_idx"].between(1, float(nfc_cfg["n_items"])))
    if "test_name" in out.columns:
        nfc_mask = nfc_mask & out["test_name"].fillna("").str.upper().isin(["NFC"])
    if "test_id" in out.columns:
        nfc_mask = nfc_mask | (
            (out["item_type"] == "likert")
            & out["item_idx"].between(1, float(nfc_cfg["n_items"]))
            & out["test_id"].fillna("").str.lower().eq("nfc18")
        )
    nfc = out.loc[nfc_mask, ["entity_id", "item_idx", "response_numeric"]].copy()
    nfc["score"] = nfc["response_numeric"]
    nfc.loc[nfc["item_idx"].isin([float(v) for v in nfc_cfg["reverse_items"]]), "score"] = _rev5(
        nfc.loc[nfc["item_idx"].isin([float(v) for v in nfc_cfg["reverse_items"]]), "response_numeric"]
    )
    nfc_total = nfc.groupby("entity_id")["score"].mean()
    result["NFC_Total_realised"] = result["entity_id"].map(nfc_total)
    return result
