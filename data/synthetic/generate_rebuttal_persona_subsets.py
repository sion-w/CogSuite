#!/usr/bin/env python3
"""Generate REBUTTAL_PANEL_150.csv and REBUTTAL_PROBE_100.csv (D-20 / J1R).

Source: simulation_outputs/run_20260506_113915_MAIN_SYNTH470_NUMERIC_RANDOM_S42/per_persona_results.csv
Rules (locked):
  PANEL — for crt2_level 0..4 in order: sort persona_id ascending; rng=default_rng(42)
           created ONCE before level 0; draw 30 without replacement via rng.choice.
  PROBE — fresh rng=default_rng(4242); 50 from level 0 sorted IDs, then 50 from level 1.
  Shortfall: take all available; never top up from outside this run.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = (
    REPO_ROOT
    / "simulation_outputs"
    / "run_20260506_113915_MAIN_SYNTH470_NUMERIC_RANDOM_S42"
    / "per_persona_results.csv"
)
OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    df = pd.read_csv(SRC)
    avail = {
        level: np.array(sorted(df.loc[df["crt2_level"] == level, "persona_id"].tolist()))
        for level in range(5)
    }

    rng = np.random.default_rng(42)
    panel_rows = []
    panel_shortfalls = {}
    for level in range(5):
        sorted_ids = avail[level]
        need = 30
        if len(sorted_ids) < need:
            chosen = sorted_ids.copy()
            panel_shortfalls[level] = need - len(sorted_ids)
        else:
            chosen = rng.choice(sorted_ids, size=need, replace=False)
            panel_shortfalls[level] = 0
        for pid in chosen:
            panel_rows.append({"persona_id": pid, "crt2_level": level})

    rng2 = np.random.default_rng(4242)
    probe_rows = []
    probe_shortfalls = {}
    for level, need in [(0, 50), (1, 50)]:
        sorted_ids = avail[level]
        if len(sorted_ids) < need:
            chosen = sorted_ids.copy()
            probe_shortfalls[level] = need - len(sorted_ids)
        else:
            chosen = rng2.choice(sorted_ids, size=need, replace=False)
            probe_shortfalls[level] = 0
        for pid in chosen:
            probe_rows.append({"persona_id": pid, "crt2_level": level})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(panel_rows).to_csv(OUT_DIR / "REBUTTAL_PANEL_150.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(OUT_DIR / "REBUTTAL_PROBE_100.csv", index=False)
    print("panel shortfalls:", panel_shortfalls)
    print("probe shortfalls:", probe_shortfalls)
    print("panel counts:", pd.DataFrame(panel_rows)["crt2_level"].value_counts().sort_index().to_dict())
    print("probe counts:", pd.DataFrame(probe_rows)["crt2_level"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
