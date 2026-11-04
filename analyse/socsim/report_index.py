from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from socsim.report_readers import collect_run_assets, discover_report_runs, load_run_metadata


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_reports_root() -> Path:
    return _repo_root() / "reports"


def _created_timestamp(report_dir: Path) -> float:
    stat = report_dir.stat()
    return float(getattr(stat, "st_birthtime", stat.st_mtime))


def _format_created_at(report_dir: Path) -> str:
    ts = _created_timestamp(report_dir)
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise_dataset_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _join_warnings(messages: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for message in messages:
        if not message or message in seen:
            continue
        seen.add(message)
        ordered.append(message)
    return "; ".join(ordered)


def build_run_record(report_dir: Path, reports_root: Path) -> dict:
    metadata = load_run_metadata(report_dir)
    datasets_present = _normalise_dataset_list(metadata.get("datasets_present"))
    datasets_selected = _normalise_dataset_list(metadata.get("datasets_selected"))
    dataset_flags = set(datasets_present or datasets_selected)
    assets = collect_run_assets(report_dir)

    calibration_files = [path for path in assets["calibration_dir"].glob("*") if path.is_file()]
    simulation_fidelity_csvs = [path for path in assets["simulation_fidelity_csvs"] if path.is_file()]

    has_descriptives_table = assets["descriptives_table"].exists()
    has_core_distribution_panel = assets["core_distribution_panel"].exists()
    has_bfi_distribution_panel = assets["bfi_distribution_panel"].exists()
    has_models_table = assets["models_table"].exists()
    has_liwc_models_table = assets["liwc_models_table"].exists()
    has_brt_models_table = assets["brt_models_table"].exists()
    has_calibration = bool(calibration_files)
    has_simulation_fidelity = assets["simulation_fidelity_dir"].exists() and bool(simulation_fidelity_csvs)
    has_scorecard = assets["scorecard_table"].exists()

    warnings: list[str] = []
    if "analysis_run_id" not in metadata:
        warnings.append("Missing analysis_run_id in run_metadata.json")
    if "datasets_present" not in metadata:
        warnings.append("Missing datasets_present in run_metadata.json")
    if "datasets_selected" not in metadata:
        warnings.append("Missing datasets_selected in run_metadata.json")
    if not has_descriptives_table:
        warnings.append("Missing descriptives overview table")
    if not has_core_distribution_panel:
        warnings.append("Missing core distribution panel")
    if not has_bfi_distribution_panel:
        warnings.append("Missing BFI distribution panel")
    if not has_models_table:
        warnings.append("Missing fixed models table")
    if not has_liwc_models_table:
        warnings.append("Missing LIWC models table")
    if not has_brt_models_table:
        warnings.append("Missing BRT models table")
    if not has_calibration:
        warnings.append("Missing calibration files")
    if not has_simulation_fidelity:
        warnings.append("Missing simulation fidelity CSVs")
    if not has_scorecard:
        warnings.append("Missing simulation fidelity scorecard")

    analysis_run_id = str(metadata.get("analysis_run_id", report_dir.name))
    socsim_run_label = metadata.get("socsim_run_label")
    record = {
        "analysis_run_id": analysis_run_id,
        "socsim_run_label": str(socsim_run_label) if socsim_run_label else "",
        "report_path": str(report_dir.relative_to(reports_root)),
        "created_at": _format_created_at(report_dir),
        "datasets_present": datasets_present,
        "datasets_selected": datasets_selected,
        "has_human": "human" in dataset_flags,
        "has_synthetic": "synthetic" in dataset_flags,
        "has_cogsim_agent": "cogsim_agent" in dataset_flags,
        "has_socsim_agent": "socsim_agent" in dataset_flags,
        "has_descriptives_table": has_descriptives_table,
        "has_core_distribution_panel": has_core_distribution_panel,
        "has_bfi_distribution_panel": has_bfi_distribution_panel,
        "has_models_table": has_models_table,
        "has_liwc_models_table": has_liwc_models_table,
        "has_brt_models_table": has_brt_models_table,
        "has_calibration": has_calibration,
        "has_simulation_fidelity": has_simulation_fidelity,
        "has_scorecard": has_scorecard,
        "warnings": _join_warnings(warnings),
    }
    return record


def build_run_index(reports_root: Path | None = None) -> list[dict]:
    reports_root = Path(reports_root) if reports_root is not None else default_reports_root()
    run_dirs = discover_report_runs(reports_root)
    return [build_run_record(report_dir, reports_root) for report_dir in run_dirs]


def write_run_index(reports_root: Path | None = None) -> tuple[Path, Path]:
    reports_root = Path(reports_root) if reports_root is not None else default_reports_root()
    records = build_run_index(reports_root)

    csv_records: list[dict] = []
    for record in records:
        row = dict(record)
        row["datasets_present"] = json.dumps(row["datasets_present"])
        row["datasets_selected"] = json.dumps(row["datasets_selected"])
        csv_records.append(row)

    columns = [
        "analysis_run_id",
        "socsim_run_label",
        "report_path",
        "created_at",
        "datasets_present",
        "datasets_selected",
        "has_human",
        "has_synthetic",
        "has_cogsim_agent",
        "has_socsim_agent",
        "has_descriptives_table",
        "has_core_distribution_panel",
        "has_bfi_distribution_panel",
        "has_models_table",
        "has_liwc_models_table",
        "has_brt_models_table",
        "has_calibration",
        "has_simulation_fidelity",
        "has_scorecard",
        "warnings",
    ]

    df = pd.DataFrame(csv_records, columns=columns)
    csv_path = reports_root / "run_index.csv"
    json_path = reports_root / "run_index.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return csv_path, json_path


def main() -> None:
    csv_path, json_path = write_run_index()
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
