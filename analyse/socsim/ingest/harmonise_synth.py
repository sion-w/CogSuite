from __future__ import annotations

import pandas as pd

OPTIONAL_SYNTH_NUMERIC_COLUMNS = ["Age"]
OPTIONAL_SYNTH_CATEGORICAL_COLUMNS = [
    "Country",
    "sex",
    "Highest education level completed",
    "Ethnicity Simplified",
    "Employment Status",
    "Occupation_cat",
]


def harmonise_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "persona_id" in out.columns:
        out["entity_id"] = out["persona_id"].astype(str)
    else:
        out["entity_id"] = out.index.astype(str)
        out["persona_id"] = out["entity_id"]
    out["dataset"] = "synthetic"
    out["BFI_Extraversion_specified"] = pd.to_numeric(out["BFI_Extraversion"], errors="coerce")
    out["BFI_Agreeableness_specified"] = pd.to_numeric(out["BFI_Agreeableness"], errors="coerce")
    out["BFI_Conscientiousness_specified"] = pd.to_numeric(out["BFI_Conscientiousness"], errors="coerce")
    out["BFI_Neuroticism_specified"] = pd.to_numeric(out["BFI_Neuroticism"], errors="coerce")
    out["BFI_Openness_specified"] = pd.to_numeric(out["BFI_Openness"], errors="coerce")
    out["NFC_Total_specified"] = pd.to_numeric(out["NFC_Total"], errors="coerce")
    out["CRT2_Total_adj_specified"] = pd.to_numeric(out["CRT2_Total_adj"], errors="coerce")
    out["BCRT_Total_specified"] = pd.to_numeric(out["BCRT_Total"], errors="coerce")
    if "CRT_Combined" in out.columns:
        out["CRT_Combined_specified"] = pd.to_numeric(out["CRT_Combined"], errors="coerce")
    else:
        out["CRT_Combined_specified"] = out["CRT2_Total_adj_specified"] + out["BCRT_Total_specified"]
    for col in OPTIONAL_SYNTH_NUMERIC_COLUMNS:
        if col in out.columns:
            out[f"{col}_specified"] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[f"{col}_specified"]
    for col in OPTIONAL_SYNTH_CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[f"{col}_specified"] = out[col].astype("object")
    return out
