"""Command-line entry point for the per-run analysis.

    python -m socsim.cli --config analyse/socsim/config/analysis_cr13_<RUN>.yaml --root <input root>

Reads the staged inputs under ``<root>/data/inputs`` (see ``socsim/pipeline.py``) and writes
``<root>/data/intermediate``, ``<root>/data/exports`` and the report ``<root>/reports/<analysis_run_id>/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

# API keys are read from a git-ignored `.env` at the repository root.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from socsim.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the per-run analysis pipeline")
    p.add_argument("--config", required=True, help="Path to analysis config YAML")
    p.add_argument("--root", default=None,
                   help="Analysis root holding data/inputs and reports (default: $COGSUITE_INPUT_ROOT, else the current directory)")
    p.add_argument(
        "--include-datasets",
        nargs="+",
        default=None,
        help="Optional dataset whitelist. If set, only listed datasets are analysed.",
    )
    p.add_argument(
        "--exclude-datasets",
        nargs="+",
        default=None,
        help="Optional dataset blacklist to remove from all outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = run_pipeline(
        args.config,
        include_datasets=args.include_datasets,
        exclude_datasets=args.exclude_datasets,
        root=args.root,
    )
    print(f"Pipeline completed. Report written to: {report_dir}")


if __name__ == "__main__":
    main()
