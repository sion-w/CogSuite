from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

# D-27: keys live in a gitignored `.env` at THIS repo's root. Resolve relative
# to this file's location (never an absolute home-directory path, per D-18)
# so `python -m socsim.cli` works regardless of the caller's cwd/shell.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from socsim.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run soc-sim analysis pipeline")
    p.add_argument("--config", required=True, help="Path to analysis config YAML")
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
    )
    print(f"Pipeline completed. Report written to: {report_dir}")


if __name__ == "__main__":
    main()
