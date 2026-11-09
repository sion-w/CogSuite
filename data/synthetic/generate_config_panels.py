#!/usr/bin/env python3
"""Draw the configuration-run profile panels CONFIG_PANEL_150.csv and CONFIG_PROBE_100.csv.

Reads the with-history reference run's per_persona_results.csv (columns persona_id, crt2_level; default
<root>/simulation_outputs/HISTORY_REFERENCE_SYNTH_NUM_N467_S42/per_persona_results.csv, where <root> is --root or
COGSUITE_INPUT_ROOT) and writes both panels (columns persona_id, crt2_level) to --out (default: this folder).

Sampling rules:
  PANEL: for crt2_level 0..4 in order, sort the persona_ids of that level ascending and draw 30 without
         replacement with rng.choice; one rng = default_rng(42) is created before level 0 and shared by all levels.
  PROBE: a separate rng = default_rng(4242); 50 from the sorted level-0 ids, then 50 from the sorted level-1 ids.
  Shortfall: a level with fewer ids than needed contributes all of them; the panel is never topped up from
  outside the source run.

Example:
    python data/synthetic/generate_config_panels.py --root <analysis folder> --out <output folder>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
SOURCE_RELPATH = Path("simulation_outputs") / "HISTORY_REFERENCE_SYNTH_NUM_N467_S42" / "per_persona_results.csv"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=os.environ.get("COGSUITE_INPUT_ROOT"),
                    help="input root (default: env COGSUITE_INPUT_ROOT)")
    ap.add_argument("--source", type=Path, default=None,
                    help=f"per_persona_results.csv of the reference run (default: <root>/{SOURCE_RELPATH})")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="folder for the two panel CSVs")
    args = ap.parse_args(argv)
    if args.source is None:
        if args.root is None:
            raise SystemExit("STOP: give --source, or --root / COGSUITE_INPUT_ROOT.")
        args.source = Path(args.root) / SOURCE_RELPATH
    src = args.source
    out_dir = args.out

    df = pd.read_csv(src)
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

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(panel_rows).to_csv(out_dir / "CONFIG_PANEL_150.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(out_dir / "CONFIG_PROBE_100.csv", index=False)
    print("panel shortfalls:", panel_shortfalls)
    print("probe shortfalls:", probe_shortfalls)
    print("panel counts:", pd.DataFrame(panel_rows)["crt2_level"].value_counts().sort_index().to_dict())
    print("probe counts:", pd.DataFrame(probe_rows)["crt2_level"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
