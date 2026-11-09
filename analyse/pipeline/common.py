"""Shared paths, run roster and helpers for the analysis chain.

Every input and output path is resolved from one input root: the ``--root`` argument of a step, or the
environment variable COGSUITE_INPUT_ROOT. Code dependencies inside the repository (the per-run analysis
package ``analyse/socsim``, the BRT derived-field module ``score/brt_v3``, the figure builder in
``analyse/scripts``) are located relative to this file.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent
ANALYSE_DIR = PIPELINE_DIR.parent
REPO = ANALYSE_DIR.parent
SCRIPTS_DIR = ANALYSE_DIR / "scripts"
BRT_V3_DIR = REPO / "score" / "brt_v3"
RESOURCES = PIPELINE_DIR / "resources"

for _p in (ANALYSE_DIR, SCRIPTS_DIR, BRT_V3_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---- the seven agent runs, in the order every table lists them -------------------------------------
RUNS = [
    ("CR_MAIN_SYNTH_NUM_SONNET5_S42", "Synthetic / Numeric · Sonnet 5", "S-Num-Sonnet5"),
    ("CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42", "Synthetic / Numeric · Gemini 3.5 Flash (min)", "S-Num-Gem35F"),
    ("CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42", "Synthetic / Numeric · DeepSeek V4-Pro", "S-Num-DSV4Pro"),
    ("CR_BASELINE_HUMAN_NUMERIC_CS46_S42", "Human personas / Numeric", "H-Num"),
    ("CR_BASELINE_HUMAN_LEXICAL_CS46_S42", "Human personas / Lexical", "H-Lex"),
    ("CR_MAIN_SYNTH_NUMERIC_CS46_S42", "Synthetic / Numeric", "S-Num"),
    ("CR_MAIN_SYNTH_LEXICAL_CS46_S42", "Synthetic / Lexical", "S-Lex"),
]
RUN_LABELS = [r[0] for r in RUNS]
RUN_DISPLAY = {r[0]: r[1] for r in RUNS}
RUN_SHORT = {r[0]: r[2] for r in RUNS}

DSV4PRO_LABEL = "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42"
# Written beside every DeepSeek V4-Pro row of a table that reports an N.
BRT_N_NOTE = ("DSV4PRO BRT N = 467: four personas returned no usable open-ended response, and the "
              "generating model snapshot has since been retired by the provider.")

# ---- input-root layout ------------------------------------------------------------------------------
PERSON_FILE = Path("data") / "inputs" / "person_with_liwc_brt_coded.csv"
BRT_LABELS_HUMAN = Path("reports") / "brt_labels" / "brt_labels_v3_human.csv"
HUMAN_DIR = Path("reports") / "cr13_human"
HUMAN_LONG = HUMAN_DIR / "crt2_long_human_v2_frame473.csv"
CRT_TABLES = Path("reports") / "crt_tables" / "tables"
LIWC_TABLES = Path("reports") / "liwc_models" / "tables"
BRT_MODELS = Path("reports") / "brt_models"
HUMAN_FAMILIES_TABLES = Path("reports") / "human_brt_families" / "tables"
TRAIT_STRUCTURE = Path("reports") / "trait_structure"
LADDER_INTERVALS = Path("reports") / "ladder_intervals"
RUN_INTERVALS = Path("reports") / "run_intervals"
LEVEL_PREVALENCE = Path("reports") / "level_prevalence"
SCREEN_TABLE = Path("reports") / "config_screen" / "screen_lure_prevalence_with_ci.csv"


class MissingInput(SystemExit):
    """An input a step needs is absent (for the human-side steps, the controlled-access dataset)."""


def run_dir(root: Path, label: str) -> Path:
    return root / "reports" / f"cr13_{label}"


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise MissingInput(f"missing input: {path} ({what})")
    return path


def resolve_root(value: str | Path | None) -> Path:
    v = value or os.environ.get("COGSUITE_INPUT_ROOT")
    if not v:
        raise SystemExit("no input root: pass --root DIR or set COGSUITE_INPUT_ROOT")
    return Path(v).resolve()


def parser(doc: str | None) -> argparse.ArgumentParser:
    """A step's command line: --root (input root; outputs are written under it)."""
    ap = argparse.ArgumentParser(description=(doc or "").strip().split("\n\n")[0])
    ap.add_argument("--root", default=None, help="input root (default: $COGSUITE_INPUT_ROOT)")
    return ap


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def require_columns(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        stop(f"{where} lacks named column(s): {', '.join(missing)}")


def with_note(df: pd.DataFrame, label_col: str = "run_label") -> pd.DataFrame:
    """The table with a brt_n_note column: BRT_N_NOTE on the DeepSeek V4-Pro rows, blank elsewhere.

    The note is attached to the values as written to CSV, so the table is passed through one CSV write and
    read before the column is added.
    """
    from io import StringIO
    out = pd.read_csv(StringIO(df.to_csv(index=False)), low_memory=False)
    out["brt_n_note"] = ""
    out.loc[out[label_col].astype(str).str.contains("DSV4PRO", na=False), "brt_n_note"] = BRT_N_NOTE
    return out


_BUILDER = None


def builder():
    """The figure and table builder, imported as a module (its helpers define the lure conventions)."""
    global _BUILDER
    if _BUILDER is None:
        path = SCRIPTS_DIR / "build_paper_assets.py"
        if not path.exists():
            raise SystemExit(f"STOP: no figure builder found at {path}")
        spec = importlib.util.spec_from_file_location("paper_assets_builder", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _BUILDER = mod
    return _BUILDER
