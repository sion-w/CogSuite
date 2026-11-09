from __future__ import annotations

import numpy as np
import pandas as pd

CRT_CATEGORY_LEVELS = ["correct", "intuitive_incorrect", "other_incorrect", "missing_or_unparseable"]
CRT_ITEM_IDS = [*(f"crt2_q{i}" for i in range(1, 5)), *(f"bcrt_q{i}" for i in range(1, 5))]
CRT_INSTRUMENT_ITEMS = {
    "crt2": [f"crt2_q{i}" for i in range(1, 5)],
    "bcrt": [f"bcrt_q{i}" for i in range(1, 5)],
}


def response_category_from_label(label: object, correct: object = np.nan) -> str:
    """Map existing evaluator labels onto explicit response-category names."""
    text = str(label).strip().lower() if pd.notna(label) else ""
    if text == "correct":
        return "correct"
    if text == "intuitive":
        return "intuitive_incorrect"
    if text == "other":
        return "other_incorrect"
    numeric_correct = pd.to_numeric(pd.Series([correct]), errors="coerce").iloc[0]
    if pd.notna(numeric_correct) and float(numeric_correct) == 1.0:
        return "correct"
    return "missing_or_unparseable"


def add_item_response_categories(scored_items: pd.DataFrame) -> pd.DataFrame:
    out = scored_items.copy()
    label_col = "llm_label" if "llm_label" in out.columns else None
    correct_col = "final_correct" if "final_correct" in out.columns else "llm_correct" if "llm_correct" in out.columns else None
    if label_col is None:
        out["response_category"] = "missing_or_unparseable"
    else:
        correct_values = out[correct_col] if correct_col else pd.Series(np.nan, index=out.index)
        out["response_category"] = [
            response_category_from_label(label, correct)
            for label, correct in zip(out[label_col], correct_values, strict=False)
        ]
    out["final_response_category"] = out["response_category"]
    return out


def add_person_response_categories_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for item_id in CRT_ITEM_IDS:
        category_col = f"{item_id}_response_category"
        if category_col not in out.columns:
            label_col = f"{item_id}_llm_label"
            correct_col = f"{item_id}_final_correct" if f"{item_id}_final_correct" in out.columns else f"{item_id}_llm_correct"
            if label_col in out.columns:
                correct_values = out[correct_col] if correct_col in out.columns else pd.Series(np.nan, index=out.index)
                out[category_col] = [
                    response_category_from_label(label, correct)
                    for label, correct in zip(out[label_col], correct_values, strict=False)
                ]
            elif correct_col in out.columns:
                out[category_col] = [
                    response_category_from_label(np.nan, correct)
                    for correct in out[correct_col]
                ]
            else:
                continue
    return add_person_category_totals(out)


def add_person_category_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for instrument, item_ids in CRT_INSTRUMENT_ITEMS.items():
        category_cols = [f"{item_id}_response_category" for item_id in item_ids if f"{item_id}_response_category" in out.columns]
        if not category_cols:
            continue
        categories = out[category_cols].astype("object")
        out[f"{instrument}_correct_total"] = categories.eq("correct").sum(axis=1).astype(float)
        out[f"{instrument}_intuitive_total"] = categories.eq("intuitive_incorrect").sum(axis=1).astype(float)
        out[f"{instrument}_other_total"] = categories.eq("other_incorrect").sum(axis=1).astype(float)
    return out

