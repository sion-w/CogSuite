from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _created_timestamp(report_dir: Path) -> float:
    stat = report_dir.stat()
    return float(getattr(stat, "st_birthtime", stat.st_mtime))


def discover_report_runs(reports_root: Path) -> list[Path]:
    reports_root = Path(reports_root)
    run_dirs = [
        path
        for path in reports_root.iterdir()
        if path.is_dir() and (path / "run_metadata.json").exists()
    ]
    return sorted(run_dirs, key=lambda path: (-_created_timestamp(path), path.name))


def load_run_metadata(report_dir: Path) -> dict:
    path = Path(report_dir) / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def collect_run_assets(report_dir: Path) -> dict:
    report_dir = Path(report_dir)
    calibration_dir = report_dir / "tables" / "calibration"
    simulation_fidelity_dir = report_dir / "tables" / "simulation_fidelity"
    return {
        "report_dir": report_dir,
        "metadata_path": report_dir / "run_metadata.json",
        "descriptives_table": report_dir / "tables" / "00_descriptives_overview.csv",
        "core_distribution_panel": report_dir / "figures" / "comparisons" / "05_core_distribution_panel.png",
        "bfi_distribution_panel": report_dir / "figures" / "comparisons" / "05_bfi_distribution_panel.png",
        "models_table": report_dir / "tables" / "04_models_all_datasets.csv",
        "liwc_models_table": report_dir / "tables" / "04_liwc_family_models_all.csv",
        "brt_models_table": report_dir / "tables" / "06_brt_primary_models_all_datasets.csv",
        "scorecard_table": simulation_fidelity_dir / "simulation_fidelity_scorecard.csv",
        "calibration_dir": calibration_dir,
        "calibration_summary_files": sorted(calibration_dir.glob("calibration_summary_*.csv")),
        "simulation_fidelity_dir": simulation_fidelity_dir,
        "simulation_fidelity_csvs": sorted(simulation_fidelity_dir.glob("*.csv")),
    }


def load_run_index(reports_root: Path) -> pd.DataFrame:
    reports_root = Path(reports_root)
    index_path = reports_root / "run_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing run index: {index_path}. Regenerate it with `python -m socsim.report_index`."
        )
    return pd.read_csv(index_path, dtype={"analysis_run_id": str, "report_path": str, "socsim_run_label": str})


def resolve_selected_runs(
    reports_root: Path,
    selected_run_ids: list[str],
    default_n: int = 5,
    selected_run_labels: list[str] | None = None,
) -> list[Path]:
    reports_root = Path(reports_root)
    run_index = load_run_index(reports_root)
    required_columns = {"analysis_run_id", "report_path"}
    missing_columns = required_columns.difference(run_index.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"run_index.csv is missing required columns: {missing}")

    if default_n < 0:
        raise ValueError("default_n must be non-negative.")

    grouped_paths = run_index.groupby("analysis_run_id")["report_path"].agg(list).to_dict()
    ambiguous = {run_id: paths for run_id, paths in grouped_paths.items() if len(paths) > 1}
    if ambiguous:
        details = ", ".join(f"{run_id} -> {paths}" for run_id, paths in sorted(ambiguous.items()))
        raise ValueError(f"Run ids must be unique in run_index.csv. Ambiguous ids: {details}")

    path_lookup = {run_id: paths[0] for run_id, paths in grouped_paths.items()}

    if selected_run_ids:
        missing_run_ids = [run_id for run_id in selected_run_ids if run_id not in path_lookup]
        if missing_run_ids:
            missing = ", ".join(missing_run_ids)
            raise ValueError(f"Selected run ids were not found in run_index.csv: {missing}")
        return [reports_root / path_lookup[run_id] for run_id in selected_run_ids]

    selected_run_labels = [str(label) for label in (selected_run_labels or []) if str(label).strip()]
    if selected_run_labels:
        if "socsim_run_label" not in run_index.columns:
            raise ValueError("run_index.csv is missing required column for label selection: socsim_run_label")
        paths: list[Path] = []
        missing_labels: list[str] = []
        for label in selected_run_labels:
            matches = run_index[run_index["socsim_run_label"].fillna("").astype(str) == label]
            if matches.empty:
                missing_labels.append(label)
                continue
            paths.extend(reports_root / rel_path for rel_path in matches["report_path"].tolist())
        if missing_labels:
            missing = ", ".join(missing_labels)
            raise ValueError(f"Selected run labels were not found in run_index.csv: {missing}")
        return paths

    recent_paths = run_index["report_path"].head(default_n).tolist()
    return [reports_root / rel_path for rel_path in recent_paths]
