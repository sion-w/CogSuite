#!/usr/bin/env python3
"""Per-run analysis driver: stage one agent run into the input root and run the per-run pipeline.

For each run label it reads, under the input root (``--root`` / ``$COGSUITE_INPUT_ROOT``):
``simulation_outputs/<RUN>/{per_persona_results.csv, per_item_results.csv, metrics_and_config.json}``
(the simulator's outputs), ``liwc/socsim_agent_<RUN>_liwc_output.csv`` (LIWC-22 output for the
run's BRT texts), ``data/inputs/person_with_liwc_brt_coded.csv`` (the human person file, or
``--human``) and the agent BRT panel cache ``brt_cache/agent/``. It stops if the run's API
hard-failure rate exceeds 1%, copies the run's two CSVs and its LIWC output into
``data/inputs/`` (removing any other staged run, since the pipeline analyses exactly one), copies
the synthetic bank from the repository, and runs ``analyse/socsim`` with
``analyse/socsim/config/analysis_cr13_<RUN>.yaml``. The report is written to
``reports/cr13_<RUN>/``; scored intermediate frames go to ``data/intermediate/``.

    python analyse/pipeline/run_per_run_analysis.py --root /path/to/analysis --run CR_MAIN_SYNTH_NUM_SONNET5_S42
    python analyse/pipeline/run_per_run_analysis.py --root /path/to/analysis --all
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSE = REPO / "analyse"
CONFIG_DIR = ANALYSE / "socsim" / "config"
SYNTHETIC_BANK = REPO / "simulate" / "data_fitting" / "persona_trait_experiment" / "synthetic_personas.csv"

RUNS = [
    "CR_MAIN_SYNTH_NUM_SONNET5_S42",
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42",
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42",
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42",
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42",
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42",
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42",
]
HARD_FAILURE_LIMIT = 0.01
HUMAN_FILE = "person_with_liwc_brt_coded.csv"


def _copy(src: Path, dst: Path) -> None:
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def check_hard_failures(run_dir: Path, label: str) -> None:
    """STOP if the simulator recorded more than 1% hard-failed API calls for this run."""
    path = run_dir / "metrics_and_config.json"
    if not path.exists():
        print(f"{label}: metrics_and_config.json absent; hard-failure check skipped.")
        return
    rel = json.loads(path.read_text(encoding="utf-8")).get("api_reliability") or {}
    rate = rel.get("hard_failure_rate")
    print(f"{label}: hard_failures={rel.get('hard_failures')} rate={rate}")
    if rel.get("hard_failure_rate_exceeds_1pct") or (rate is not None and float(rate) > HARD_FAILURE_LIMIT):
        raise SystemExit(f"STOP: {label} hard-failure rate {rate} exceeds {HARD_FAILURE_LIMIT:.0%}")


def stage_run(root: Path, label: str, human: Path | None, liwc_dir: Path) -> None:
    run_dir = root / "simulation_outputs" / label
    person_src = run_dir / "per_persona_results.csv"
    items_src = run_dir / "per_item_results.csv"
    for p in (person_src, items_src):
        if not p.exists():
            raise SystemExit(f"STOP: missing {p}")
    check_hard_failures(run_dir, label)

    inputs = root / "data" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if human is not None:
        _copy(human, inputs / HUMAN_FILE)
    if not (inputs / HUMAN_FILE).exists():
        raise SystemExit(
            f"STOP: human person file not found at {inputs / HUMAN_FILE}. The per-run analysis compares agents with people "
            "and needs the companion study's person_with_liwc_brt_coded.csv (de-identified records: available from the "
            "authors; https://osf.io/xuv7e/). Place it there or pass --human."
        )
    keep = {f"socsim_per_persona_results_{label}.csv", f"socsim_per_item_results_{label}.csv"}
    for staged in list(inputs.glob("socsim_per_persona_results_*.csv")) + list(inputs.glob("socsim_per_item_results_*.csv")):
        if staged.name not in keep:
            staged.unlink()
    _copy(person_src, inputs / f"socsim_per_persona_results_{label}.csv")
    _copy(items_src, inputs / f"socsim_per_item_results_{label}.csv")

    liwc_name = f"socsim_agent_{label}_liwc_output.csv"
    liwc_src = liwc_dir / liwc_name
    if liwc_src.exists():
        _copy(liwc_src, inputs / liwc_name)
    elif not (inputs / liwc_name).exists():
        print(f"{label}: no LIWC-22 output at {liwc_src}; the pipeline prints how to make one.")

    _copy(SYNTHETIC_BANK, inputs / "synthetic_personas.csv")


def config_for(root: Path, label: str) -> Path:
    cfg = CONFIG_DIR / f"analysis_cr13_{label}.yaml"
    if cfg.exists():
        return cfg
    out = root / "data" / "intermediate" / f"analysis_cr13_{label}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text((CONFIG_DIR / "analysis.yaml").read_text(encoding="utf-8").replace(
        "analysis_run_id: auto", f"analysis_run_id: cr13_{label}"), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.environ.get("COGSUITE_INPUT_ROOT"), help="input root (default $COGSUITE_INPUT_ROOT)")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", action="append", help="run label (repeatable)")
    which.add_argument("--all", action="store_true", help="the seven full-battery runs")
    ap.add_argument("--human", type=Path, help=f"copy this file to <root>/data/inputs/{HUMAN_FILE}")
    ap.add_argument("--liwc-dir", type=Path, help="folder with socsim_agent_<RUN>_liwc_output.csv (default <root>/liwc)")
    args = ap.parse_args(argv)
    if not args.root:
        ap.error("--root or $COGSUITE_INPUT_ROOT is required")
    root = Path(args.root).resolve()
    liwc_dir = args.liwc_dir or root / "liwc"
    labels = RUNS if args.all else args.run

    sys.path.insert(0, str(ANALYSE))
    from socsim.pipeline import run_pipeline

    for label in labels:
        print(f"===== per-run analysis {label} =====", flush=True)
        stage_run(root, label, args.human, liwc_dir)
        report = run_pipeline(str(config_for(root, label)), root=str(root))
        print(f"===== done {label}: {report} =====", flush=True)

    from socsim.report_index import write_run_index
    csv_path, _ = write_run_index(root / "reports")
    print(f"Run index: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
