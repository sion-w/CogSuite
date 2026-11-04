from __future__ import annotations

import pandas as pd


def harmonise_cogsim_person(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = "cogsim_agent"
    out["entity_id"] = out["run_id"].astype(str) + ":" + out["agent_id"].astype(str)
    return out


def harmonise_cogsim_items(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = "cogsim_agent"
    out["entity_id"] = out["run_id"].astype(str) + ":" + out["agent_id"].astype(str)
    out = out.rename(columns={"response_text_item": "response_text"})
    return out
