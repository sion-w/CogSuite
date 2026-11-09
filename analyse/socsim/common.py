from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")


CRT_ITEM_TEXT_COLS = [f"{p}_q{i}" for p in ("crt2", "bcrt", "dcrt") for i in range(1, 5)]
AGENT_ITEM_TEXT_COLS = ["response_text"]
HUMAN_LONG_TEXT_COLS = ["raw_response", "normalised"]


def read_csv_keep_text(path: Path, text_cols: Iterable[str], **kwargs) -> pd.DataFrame:
    """``pd.read_csv``, except that the named free-text columns keep literal answers.

    pandas' default NA parsing turns the answer text "None", "NA", "N/A", "null" and
    "nan" into missing. "None" is the correct answer to CRT2 item 4 and "n/a" is keyed correct
    on bCRT item 3, so in free-text answer columns only an empty cell is missing. Every other
    column is parsed exactly as ``pd.read_csv(path, **kwargs)`` would parse it.
    """
    df = pd.read_csv(path, **kwargs)
    present = [c for c in text_cols if c in df.columns]
    if present:
        text = pd.read_csv(path, usecols=present, dtype=str, keep_default_na=False, na_values=[""])
        for col in present:
            df[col] = text[col].to_numpy()
    return df


def read_csv_strict(
    path: Path,
    required: Iterable[str] | None = None,
    table_name: str | None = None,
    text_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Missing required file: {str(path)}")
    df = read_csv_keep_text(path, text_cols) if text_cols else pd.read_csv(path)
    if required is not None:
        require_columns(df, required, table_name or path.name)
    return df


def analysis_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["_".join([str(v) for v in tup if str(v) != ""]).strip("_") for tup in out.columns.to_flat_index()]
    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def json_write(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ci95_normal(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().astype(float)
    n = len(vals)
    if n < 2:
        return (np.nan, np.nan)
    m = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(n))
    return (m - 1.96 * se, m + 1.96 * se)
