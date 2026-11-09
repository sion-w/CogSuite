"""Run the analysis chain: per-run analysis reports -> the intermediate tables the figure builder reads.

Steps, in order (each is also a script of its own; all write under the input root):

  human_reference     reports/cr13_human/crt2_long_human_v2_frame473.csv            (human dataset)
  human_brt_families  reports/human_brt_families/tables/s5_primary_families.csv     (human dataset)
  crt_tables          reports/crt_tables/tables/*.csv                               (human dataset)
  liwc_models         reports/liwc_models/tables/liwc_family_models_all_runs.csv
  brt_models          reports/brt_models/tables/s11_*.csv                           (human dataset)
  brt_figures         reports/brt_models/figures/fig_s11_*.{pdf,png}
  trait_structure     reports/trait_structure/f4_*.csv                              (human dataset)
  ladder_intervals    reports/ladder_intervals/human_item_ci.csv                    (human dataset)
  run_intervals       reports/run_intervals/{run_ci,human_pooled_ci}.csv            (human dataset)
  level_prevalence    reports/level_prevalence/level_prevalence_{runs,human}.csv    (human dataset)

A step whose input is missing (for most steps, the controlled-access human dataset or the human-derived
runs' agent frames) is skipped with a message naming the file; the later steps still run, and a step that
needs a skipped step's output is skipped in turn. No step calls a model API.

Usage: python analyse/pipeline/run_chain.py [--root DIR] [--only STEP ...] [--from STEP]
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MissingInput, parser, resolve_root  # noqa: E402

STEPS = ["human_reference", "human_brt_families", "crt_tables", "liwc_models", "brt_models", "brt_figures",
         "trait_structure", "ladder_intervals", "run_intervals", "level_prevalence"]


def main(argv: list[str] | None = None) -> int:
    ap = parser(__doc__)
    ap.add_argument("--only", nargs="+", choices=STEPS, help="run only these steps")
    ap.add_argument("--from", dest="start", choices=STEPS, help="start at this step")
    a = ap.parse_args(argv)
    root = resolve_root(a.root)
    steps = a.only or (STEPS[STEPS.index(a.start):] if a.start else STEPS)
    status = {}
    for name in steps:
        print(f"\n== {name}", flush=True)
        t0 = time.time()
        try:
            rc = importlib.import_module(name).main(["--root", str(root)])
            status[name] = "ok" if not rc else f"exit {rc}"
        except MissingInput as e:
            print(f"SKIPPED {name}: {e}")
            status[name] = "skipped (missing input)"
        print(f"   {name}: {status[name]} ({time.time() - t0:.0f} s)", flush=True)
    print("\nchain summary:")
    for name, s in status.items():
        print(f"  {name:20s} {s}")
    return 0 if all(s == "ok" or s.startswith("skipped") for s in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
