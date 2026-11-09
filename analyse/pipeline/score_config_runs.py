"""Score the CRT2/bCRT answers of every configuration run and of the with-history reference run.

Reads <root>/simulation_outputs/<RUN>/per_item_results.csv for the 19 configuration runs on the 150-profile panel
(the CRFS, CRFSB and CRFSD screen cells, the five CONFIG_MULTIMODEL cells, the five CONFIG_TEMP cells) and for the
with-history reference run HISTORY_REFERENCE_SYNTH_NUM_N467_S42 (all 467 personas; the builder keeps the 150 panel ids).
Each answer is scored by score/scoring_v2 (deterministic answer keys first, residuals from the adjudication cache)
and the scored long table is written to <root>/reports/crt_scored/crt_long_<STEM>.csv with columns source, run_label,
entity_id, item_id, raw_response, normalised, label_v2, method, agreement. No model is called unless --allow-api is
given; a residual absent from the cache then stops the run, since every released table is fully adjudicated.

Example:
    python analyse/pipeline/score_config_runs.py --root "$COGSUITE_INPUT_ROOT"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "score" / "scoring_v2"))
from score_cli import DEFAULT_CACHE, score_file, unadjudicated  # noqa: E402

SIM_OUTPUTS = Path("simulation_outputs")
SCORED_TABLES = Path("reports") / "crt_scored"

# (table stem, raw-run folder under simulation_outputs/, expected personas)
RUNS = (
    [(s, f"{s}_N150_S42", 150) for s in (
        "CRFS_SONNET5_DEFAULT_NUM", "CRFS_GEMINI35FLASH_MIN_LEX", "CRFS_DSV4PRO_ON_LEX",
        "CRFSB_SONNET5_THINKOFF_NUM", "CRFSB_CLAUDE_TOP_NUM", "CRFSB_GPT56_REASONING_NUM",
        "CRFSD_DSV4PRO_TOG_NUM", "CRFSD_DSV4PRO_TOG_NOTHINK_NUM", "CRFSD_KIMI_K3_NUM")]
    + [(f"CONFIG_MULTIMODEL_{m}", f"CONFIG_MULTIMODEL_{m}_N150_S42", 150) for m in (
        "GPT56SOL_NONE", "GEMINI35FLASH_MIN", "DSV4FLASH_OFF", "DSV4FLASH_ON", "DSV4PRO_ON")]
    + [(f"CONFIG_TEMP_{t}", f"CONFIG_TEMP_{t}_N150_S42", 150) for t in ("DEFAULT", "000", "030", "070", "100")]
    + [("HISTORY_REFERENCE_SYNTH_NUM", "HISTORY_REFERENCE_SYNTH_NUM_N467_S42", 467)]
)
N_CRT_ITEMS = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=os.environ.get("COGSUITE_INPUT_ROOT"),
                    help="input root (default: $COGSUITE_INPUT_ROOT)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <root>/reports/crt_scored)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="adjudication cache (JSONL)")
    ap.add_argument("--only", nargs="*", default=None, help="score only these table stems")
    ap.add_argument("--allow-api", action="store_true", help="adjudicate uncached residuals live (needs API keys)")
    args = ap.parse_args()
    if args.root is None:
        ap.error("give --root or set COGSUITE_INPUT_ROOT")
    out_dir = args.out or args.root / SCORED_TABLES
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [r for r in RUNS if args.only is None or r[0] in args.only]
    for stem, folder, n_personas in runs:
        items = args.root / SIM_OUTPUTS / folder / "per_item_results.csv"
        if not items.exists():
            raise SystemExit(f"STOP: raw run missing for {stem}: {items}")
        scored = score_file(items, args.cache, stem, folder, args.allow_api)
        pending = unadjudicated(scored)
        if pending:
            raise SystemExit(f"STOP: {stem} has {len(pending)} unadjudicated residual answer(s); "
                             "rerun with --allow-api and the annotator API keys, or supply a cache that holds them.")
        if scored["entity_id"].nunique() != n_personas or len(scored) != n_personas * N_CRT_ITEMS:
            raise SystemExit(f"STOP: {stem} has {scored['entity_id'].nunique()} personas / {len(scored)} answers, "
                             f"expected {n_personas} / {n_personas * N_CRT_ITEMS}.")
        path = out_dir / f"crt_long_{stem}.csv"
        scored.to_csv(path, index=False)
        print(f"{stem}: {len(scored)} answers, methods {scored['method'].value_counts().to_dict()} -> {path}")


if __name__ == "__main__":
    main()
