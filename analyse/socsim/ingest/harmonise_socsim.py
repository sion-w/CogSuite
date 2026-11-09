from __future__ import annotations

import pandas as pd


def harmonise_socsim_person(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = "socsim_agent"
    out["entity_id"] = out["persona_id"].astype(str)
    out["BFI_Extraversion_specified"] = pd.to_numeric(out.get("BFI_Extraversion"), errors="coerce")
    out["BFI_Agreeableness_specified"] = pd.to_numeric(out.get("BFI_Agreeableness"), errors="coerce")
    out["BFI_Conscientiousness_specified"] = pd.to_numeric(out.get("BFI_Conscientiousness"), errors="coerce")
    out["BFI_Neuroticism_specified"] = pd.to_numeric(out.get("BFI_Neuroticism"), errors="coerce")
    out["BFI_Openness_specified"] = pd.to_numeric(out.get("BFI_Openness"), errors="coerce")
    out["NFC_Total_specified"] = pd.to_numeric(out.get("NFC_Total"), errors="coerce")
    out["CRT2_Total_adj_specified"] = pd.to_numeric(out.get("CRT2_Total_adj"), errors="coerce")
    out["BCRT_Total_specified"] = pd.to_numeric(out.get("BCRT_Total"), errors="coerce")
    out["CRT_Combined_specified"] = pd.to_numeric(out.get("CRT_Combined"), errors="coerce")
    return out


def harmonise_socsim_items(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = "socsim_agent"
    out["entity_id"] = out["persona_id"].astype(str)
    return out
