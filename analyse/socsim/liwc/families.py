from __future__ import annotations

import pandas as pd


def compute_liwc_families(df: pd.DataFrame, families_cfg: dict) -> pd.DataFrame:
    out = df.copy()
    fams = families_cfg.get("families", {})
    for fam_name, cols in fams.items():
        existing = [c for c in cols if c in out.columns]
        if not existing:
            out[f"LIWC_{fam_name}"] = pd.NA
        else:
            out[f"LIWC_{fam_name}"] = out[existing].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return out
