from __future__ import annotations

from pathlib import Path

import pandas as pd

from socsim.common import require_columns
from socsim.liwc.families import compute_liwc_families


def join_liwc_outputs(person_df: pd.DataFrame, liwc_output_path: Path, liwc_cfg: dict) -> pd.DataFrame:
    if not liwc_output_path.exists():
        out = person_df.copy()
        out["entity_id"] = out["entity_id"].astype(str)
        out["liwc_warning"] = f"LIWC output not found: {liwc_output_path.name}"
        for fam_name in liwc_cfg.get("families", {}).keys():
            out[f"LIWC_{fam_name}"] = pd.NA
        return out
    liwc = pd.read_csv(liwc_output_path)
    require_columns(liwc, ["entity_id"], liwc_output_path.name)
    person = person_df.copy()
    person["entity_id"] = person["entity_id"].astype(str)
    liwc["entity_id"] = liwc["entity_id"].astype(str)
    merged = person.merge(liwc, on="entity_id", how="left")
    merged = compute_liwc_families(merged, liwc_cfg)
    return merged
