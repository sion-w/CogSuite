#!/usr/bin/env python3
"""Build camera-ready NeurIPS E&D paper assets from existing soc-sim reports.

This script is intentionally read-only with respect to existing report outputs:
it selects the latest report for each target run label, reads already generated
CSV/JSON artifacts, and writes final figures/tables under
reports/neurips_paper_assets_v2/ (CR-14). The CR-13B bundle at
reports/neurips_paper_assets/ is read for its staged source_existing copies and is never written.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from socsim.common import HUMAN_LONG_TEXT_COLS, read_csv_keep_text  # noqa: E402
from socsim.analysis.binned_relationships import _resolve_cognitive_predictor_column  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
# D-208(b), D-210: the paper's one colour system (one hue per model family, one shape per run).
from paper_palette import (  # noqa: E402
    HUMAN_INK,
    HUMAN_MARKER_ALONG_X,
    HUMAN_MARKER_ALONG_Y,
    HUMAN_MARKER_EDGE,
    HUMAN_MARKER_SIZE,
    LADDER_FAMILY_COLORS,
    RUN_COLORS,
    RUN_MARKERS,
)


# CR-13B (D-164/D-174(a)): the camera-ready is built over the SEVEN CR-13 agent runs on
# v2 CRT scoring and v3.0 BRT labels, not the four submitted 467sim cells. The superseded
# four-run build is kept at archive/pre_cr13b/build_neurips_camera_ready_assets.pre_cr13b.py.
TARGET_LABELS = (
    "CR_MAIN_SYNTH_NUM_SONNET5_S42",
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42",
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42",
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42",
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42",
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42",
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42",
)

RUN_DISPLAY = {
    "human": "Human reference",
    "CR_MAIN_SYNTH_NUM_SONNET5_S42": "Synthetic / Numeric - Sonnet 5",
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": "Synthetic / Numeric - Gemini 3.5 Flash (min)",
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": "Synthetic / Numeric - DeepSeek V4-Pro",
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42": "Human personas / Numeric",
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": "Human personas / Lexical",
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42": "Synthetic / Numeric",
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42": "Synthetic / Lexical",
}

RUN_SHORT = {
    "human": "Human",
    "CR_MAIN_SYNTH_NUM_SONNET5_S42": "S-Num-Sonnet5",
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": "S-Num-Gem35F",
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": "S-Num-DSV4Pro",
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42": "H-Num",
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": "H-Lex",
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42": "S-Num",
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42": "S-Lex",
}

# PI ruling A (23 Sep 2026, CR-13R). Every table that prints the DSV4PRO BRT N carries it.
DSV4PRO_LABEL = "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42"
# S-8 (C-17): the same note for printed tables, in readable names (the CSV column keeps DSV4PRO_BRT_N_NOTE)
DSV4PRO_BRT_N_TEX = ("DeepSeek V4-Pro BRT N = 467: four personas returned no usable open-ended response, "
                     "and the provider has since retired the model snapshot.")
DSV4PRO_BRT_N_NOTE = (
    "DSV4PRO BRT N = 467: four personas returned no usable open-ended response, and the "
    "generating model snapshot has since been retired by the provider."
)

DISTRIBUTION_CONSTRUCTS = [
    "CRT2_Total",
    "BCRT_Total",
    "NFC_Total",
    "BFI_Extraversion",
    "BFI_Agreeableness",
    "BFI_Conscientiousness",
    "BFI_Neuroticism",
    "BFI_Openness",
]

CRT_CATEGORY_LEVELS = [
    "correct",
    "intuitive_incorrect",
    "other_incorrect",
    "missing_or_unparseable",
]

CRT_CATEGORY_COLORS = {
    "correct": "#2ca02c",
    "intuitive_incorrect": "#ff7f0e",
    "other_incorrect": "#d62728",
    "missing_or_unparseable": "#7f7f7f",
}

# Part II targets = the ANALYSIS_PLAN §5 primary set (15 members, four families), ratified
# 2026-09-17 (D-171(i)); every BRT prevalence / compression / transfer path reads this list.
# Columns are produced by brt_v3/derived.py (D-149) and must be present in every BRT frame
# this script loads (guard: require_brt_primary_columns).
BRT_PRIMARY_FAMILIES = {
    "H1": [
        "BRT1_check_venue",
        "BRT2_check_institutional_sources",
        "BRT2_check_persons",
        "BRT1_check_official",
        "BRT2_check_professional",
        "BRT1_check_persons",
        "BRT2_check_claim_verification",
    ],
    "H2": [
        "BRT1_reflective_keep_v3_epistemic",
        "BRT2_reflective_accept_v3_plan",
        "BRT2_reflective_delay_v3_plan",
        "BRT2_credulous_adoption_v3",
        "BRT2_fear_driven_rejection_v3",
    ],
    "H3": [
        "BRT2_emo_worry",
        "BRT1_emotional_outburst",
    ],
    "H4": [
        "BRT1_moral_fairness_loyalty",
    ],
}
BRT_PRIMARY_OUTCOMES = [c for members in BRT_PRIMARY_FAMILIES.values() for c in members]
BRT_OUTCOME_FAMILY = {c: fam for fam, members in BRT_PRIMARY_FAMILIES.items() for c in members}

# ANALYSIS_PLAN §11 / §8 C3 definitions (replace every earlier ad-hoc threshold)
COMPRESSED_GAP = 0.20          # F1: |gap| >= 20 pp with non-overlapping 95% Wilson CIs
UNSTABLE_EVENT_FLOOR = 20      # C3: < 20 events or < 20 non-events in the run
F3_PREDICTORS = {"BCRT_Total", "BCRT_Total_realised"}   # realised bCRT (F3)


def require_brt_primary_columns(df: pd.DataFrame, frame: str) -> None:
    """STOP with a named-column error if a BRT frame lacks any §5 primary column."""
    missing = [c for c in BRT_PRIMARY_OUTCOMES if c not in df.columns]
    if missing:
        raise SystemExit(
            f"STOP: BRT frame `{frame}` lacks ANALYSIS_PLAN §5 primary column(s): "
            + ", ".join(missing)
            + " — recode/derive with brt_v3/derived.py before building Part II assets."
        )


def compressed_flag(h_prev: float, h_lo: float, h_hi: float, r_prev: float, r_lo: float, r_hi: float) -> bool:
    """§11 F1: |gap| >= 20 pp AND the two 95% CIs do not overlap."""
    if not np.isfinite([h_prev, h_lo, h_hi, r_prev, r_lo, r_hi]).all():
        return False
    no_overlap = (r_hi < h_lo) or (r_lo > h_hi)
    return bool(abs(r_prev - h_prev) >= COMPRESSED_GAP and no_overlap)


def unstable_flag(events: float, n: float) -> bool:
    """§8 C3: < 20 events or < 20 non-events in the run."""
    if not np.isfinite([events, n]).all():
        return False
    return bool(min(events, n - events) < UNSTABLE_EVENT_FLOOR)


def transfer_verdict(b_run: float, se_run: float, b_hum: float, se_hum: float, unstable: bool) -> tuple[str, float, float, float]:
    """§11 F3 on realised bCRT: returns (verdict, diff, diff_lo, diff_hi).

    transfers = CI on beta_run - beta_human includes 0 AND the run's own CI excludes the null in
    the human direction; fails to transfer = difference CI excludes 0; indeterminate otherwise.
    Unstable (C3) models are reported but excluded from the verdict tally ("unstable").
    """
    if not np.isfinite([b_run, se_run, b_hum, se_hum]).all():
        return "unavailable", np.nan, np.nan, np.nan
    diff = b_run - b_hum
    se = math.sqrt(se_run**2 + se_hum**2)
    lo, hi = diff - 1.959964 * se, diff + 1.959964 * se
    if unstable:
        return "unstable", diff, lo, hi
    run_lo, run_hi = b_run - 1.959964 * se_run, b_run + 1.959964 * se_run
    same_direction = (run_lo > 0) if b_hum > 0 else (run_hi < 0) if b_hum < 0 else False
    if lo > 0 or hi < 0:
        return "fails to transfer", diff, lo, hi
    if same_direction:
        return "transfers", diff, lo, hi
    return "indeterminate", diff, lo, hi

CLEAN_LABELS = {
    "CRT2_Total": "CRT2",
    "BCRT_Total": "bCRT",
    "NFC_Total": "NFC",
    "BFI_Extraversion": "Extraversion",
    "BFI_Agreeableness": "Agreeableness",
    "BFI_Conscientiousness": "Conscientiousness",
    "BFI_Neuroticism": "Neuroticism",
    "BFI_Openness": "Openness",
    "BRT1_check_venue": "Phones the restaurant (primary source; H1a)",
    "BRT2_check_institutional_sources": "Consults regulators / journals / official guidance",
    "BRT2_check_persons": "Asks friends or family (medication)",
    "BRT1_check_official": "Consults hygiene / official records",
    "BRT2_check_professional": "Consults a qualified professional",
    "BRT1_check_persons": "Asks friends or family (restaurant)",
    "BRT2_check_claim_verification": "Verifies the video's cited sources",
    "BRT1_reflective_keep_v3_epistemic": "Keeps the booking on evidence grounds",
    "BRT2_reflective_accept_v3_plan": "Accepts on institutional grounds with a verification plan",
    "BRT2_reflective_delay_v3_plan": "Delays with a verification plan and a stated condition",
    "BRT2_credulous_adoption_v3": "Acts on the claim with no doubt and no rule",
    "BRT2_fear_driven_rejection_v3": "Refuses outright, worry-marked",
    "BRT2_emo_worry": "Voices worry about the medication",
    "BRT1_emotional_outburst": "Hostile rebuke of the unhelpful friend",
    "BRT1_moral_fairness_loyalty": "Appeals to fairness or loyalty toward Friend B",
    "BRT1_decision": "BRT1 decision",
    "BRT2_decision": "BRT2 decision",
    "BRT1_anchor": "BRT1 anchor",
    "BRT2_anchor": "BRT2 anchor",
    "BRT2_research_specificity": "BRT2 research specificity",
    "correct": "correct",
    "intuitive_incorrect": "intuitive incorrect",
    "other_incorrect": "other incorrect",
    "missing_or_unparseable": "missing/unparseable",
    "z_BCRT_Total": "z bCRT",
    "z_CRT2_Total": "z CRT2",
    "z_NFC_Total": "z NFC",
    "primary_bcrt_only": "primary bCRT only",
    "secondary_full_cognitive": "secondary full cognitive",
    "emo_neg": "Negative emotion",
    "differ": "Differentiation",
    "auxverb": "Auxiliary verbs",
    "risk": "Risk",
    "focuspresent": "Present focus",
    "WC": "Word count",
    "moral": "Moral language",
}

LIWC_THEORY_OUTCOMES = ["emo_neg", "differ", "auxverb", "risk", "focuspresent", "WC", "moral"]
LIWC_PREDICTORS = ["BCRT_Total", "CRT2_Total"]

# CR-14 (I10): the submitted Fig 3 dropped Extraversion and Agreeableness although its caption
# named the full set. All eight cognition constructs are plotted now; the six-variable subset
# the submitted association MAE was computed on is kept as ASSOCIATION_MAE_LEGACY_VARIABLES so
# the scorecard's association_MAE keeps its CR-13B definition (a metric-scope change is not a
# presentation fix).
CORE_ASSOCIATION_VARIABLES = [
    "CRT2_Total",
    "BCRT_Total",
    "NFC_Total",
    "BFI_Extraversion",
    "BFI_Agreeableness",
    "BFI_Conscientiousness",
    "BFI_Openness",
    "BFI_Neuroticism",
]
ASSOCIATION_MAE_LEGACY_VARIABLES = [
    "CRT2_Total", "BCRT_Total", "NFC_Total",
    "BFI_Conscientiousness", "BFI_Openness", "BFI_Neuroticism",
]

BRT_CATEGORICAL_MEASURES = [
    "BRT1_decision",
    "BRT2_decision",
    "BRT1_anchor",
    "BRT2_anchor",
    "BRT2_research_specificity",
]

ASSOCIATION_VARIABLES = CORE_ASSOCIATION_VARIABLES + BRT_PRIMARY_OUTCOMES

ASSOCIATION_ALIASES = {
    "CRT2_Total": ["CRT2_Total", "CRT2_Total_adj", "CRT2_Total_realised", "CRT2_Total_observed"],
    "BCRT_Total": ["BCRT_Total", "BCRT_Total_realised", "BCRT_Total_observed"],
    "NFC_Total": ["NFC_Total", "NFC_Total_realised", "NFC_Total_observed"],
    "BFI_Extraversion": ["BFI_Extraversion", "BFI_E", "BFI_E_realised", "BFI_E_observed"],
    "BFI_Agreeableness": ["BFI_Agreeableness", "BFI_A", "BFI_A_realised", "BFI_A_observed"],
    "BFI_Conscientiousness": ["BFI_Conscientiousness", "BFI_C", "BFI_C_realised", "BFI_C_observed"],
    "BFI_Openness": ["BFI_Openness", "BFI_O", "BFI_O_realised", "BFI_O_observed"],
    "BFI_Neuroticism": ["BFI_Neuroticism", "BFI_N", "BFI_N_realised", "BFI_N_observed"],
    **{c: [c] for c in BRT_PRIMARY_OUTCOMES},
}

CALIBRATION_CONSTRUCTS = {
    "CRT2": ["CRT2", "CRT2_Total"],
    "bCRT-derived score": ["BCRT_proxy", "BCRT", "BCRT_Total"],
    "NFC": ["NFC", "NFC_Total"],
    "Extraversion": ["BFI_E", "BFI_Extraversion"],
    "Agreeableness": ["BFI_A", "BFI_Agreeableness"],
    "Conscientiousness": ["BFI_C", "BFI_Conscientiousness"],
    "Neuroticism": ["BFI_N", "BFI_Neuroticism"],
    "Openness": ["BFI_O", "BFI_Openness"],
}

TIMESTAMPED_REPORT_RE = re.compile(r"^(?P<timestamp>\d{8}_\d{6})_(?P<label>.+)$")
STAGED_REPORT_RE = re.compile(r"^(?P<label>.+)__(?P<report_id>.+)$")


@dataclass(frozen=True)
class ReportCandidate:
    label: str
    report_dir: Path
    report_id: str
    timestamp: str
    source: str


class Diagnostics:
    def __init__(self) -> None:
        self.missing_labels: dict[str, str] = {}
        self.missing_files: list[str] = []
        self.missing_columns: list[str] = []
        self.skipped_panels: list[str] = []
        self.notes: list[str] = []
        self.alias_rows: list[dict[str, str]] = []
        self.regenerated_figures: list[str] = []
        self.recalculated_tables: list[str] = []
        self.brt_prevalence_complete = False
        self.association_includes_brt = False
        self.prompt_screening_available = False
        self.known_limitations: list[str] = []
        self.brt_recovery_rows: list[dict[str, str]] = []

    def file(self, label: str, path: Path, reason: str) -> None:
        self.missing_files.append(f"- `{label}`: `{path.as_posix()}` - {reason}")

    def column(self, context: str, column: str, reason: str) -> None:
        clean_context = context
        for raw, display in RUN_DISPLAY.items():
            clean_context = clean_context.replace(raw, display)
        self.missing_columns.append(f"- `{clean_context}` missing `{clean_label(column)}` - {reason}")

    def panel(self, panel: str, reason: str) -> None:
        self.skipped_panels.append(f"- `{panel}` - {reason}")

    def note(self, message: str) -> None:
        self.notes.append(f"- {message}")

    def alias(self, run_label: str, requested: str, actual: str, source_file: str) -> None:
        self.alias_rows.append(
            {
                "run_label": run_label,
                "requested_variable": requested,
                "actual_variable": actual,
                "source_file": source_file,
            }
        )

    def figure(self, filename: str) -> None:
        if filename not in self.regenerated_figures:
            self.regenerated_figures.append(filename)

    def table(self, filename: str) -> None:
        if filename not in self.recalculated_tables:
            self.recalculated_tables.append(filename)

    def limitation(self, message: str) -> None:
        if message not in self.known_limitations:
            self.known_limitations.append(message)

    def brt_recovery(self, run_label: str, source_file: Path, outcome: str, status: str, reason: str) -> None:
        self.brt_recovery_rows.append(
            {
                "run_label": display_run_label(run_label),
                "source_file": source_file.name,
                "outcome": clean_label(outcome),
                "status": status,
                "reason": reason,
            }
        )


def repo_root() -> Path:
    # CR-26 (D-309, PI ruling 3 Nov): COGSUITE_INPUT_ROOT points the builder at an input folder laid out like this
    # repository (the public release's data folder); unset, the builder reads this checkout as before.
    env = os.environ.get("COGSUITE_INPUT_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[1]


def output_root(root: Path) -> Path:
    # CR-14: the presentation-cluster rebuild ships under neurips_paper_assets_v2 so the
    # CR-13B bundle stays intact as the pre-CR-14 record.
    return root / "reports" / "neurips_paper_assets_v2"


def legacy_bundle_root(root: Path) -> Path:
    return root / "reports" / "neurips_paper_assets"


def staged_source_dir(root: Path, candidate: ReportCandidate) -> Path:
    return legacy_bundle_root(root) / "source_existing" / f"{candidate.label}__{candidate.report_id}"


def source_roots_for_candidate(root: Path, candidate: ReportCandidate) -> list[Path]:
    staged = staged_source_dir(root, candidate)
    roots = [staged] if staged.exists() else []
    if candidate.report_dir.exists() and candidate.report_dir not in roots:
        roots.append(candidate.report_dir)
    return roots


def mkdirs(out: Path) -> dict[str, Path]:
    dirs = {
        "figures": out / "figures",
        "tables": out / "tables",
        "captions": out / "captions",
        "diagnostics": out / "diagnostics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def timestamp_from_report_id(report_id: str) -> str:
    match = re.match(r"^(?P<timestamp>\d{8}_\d{6})(?:_|$)", report_id)
    return match.group("timestamp") if match else ""


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def sample_size_from_row(row: pd.Series) -> float:
    for col in ("N", "n", "N_non_missing", "n_non_missing", "N_total", "n_total"):
        if col in row.index:
            value = safe_float(row.get(col))
            if np.isfinite(value):
                return value
    return np.nan


def read_csv_optional(path: Path, diag: Diagnostics | None = None, label: str = "") -> pd.DataFrame:
    if not path.exists():
        if diag is not None:
            diag.file(label or "cross_run", path, "file not found")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - diagnostics path
        if diag is not None:
            diag.file(label or "cross_run", path, f"could not read CSV: {exc}")
        return pd.DataFrame()


def read_json_optional(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_index_rows(reports_root: Path) -> list[dict[str, str]]:
    path = reports_root / "run_index.csv"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [{k: v or "" for k, v in row.items()} for row in csv.DictReader(f)]
    except OSError:
        return []


def normalise_report_dir(reports_root: Path, report_path_value: str, analysis_run_id: str, label: str) -> Path:
    if report_path_value:
        report_path = Path(report_path_value)
        return report_path if report_path.is_absolute() else reports_root / report_path
    if analysis_run_id:
        direct = reports_root / analysis_run_id
        labelled = reports_root / f"{analysis_run_id}_{label}"
        return direct if direct.exists() else labelled
    return reports_root / f"*_{label}"


def index_candidates_for_label(reports_root: Path, label: str) -> list[ReportCandidate]:
    candidates: list[ReportCandidate] = []
    for row in read_index_rows(reports_root):
        row_label = row.get("socsim_run_label") or row.get("run_label") or row.get("target_label") or ""
        if row_label != label:
            continue
        analysis_run_id = row.get("analysis_run_id", "")
        report_dir = normalise_report_dir(reports_root, row.get("report_path", ""), analysis_run_id, label)
        if not report_dir.exists() or not report_dir.is_dir():
            continue
        candidates.append(
            ReportCandidate(
                label=label,
                report_dir=report_dir,
                report_id=report_dir.name,
                timestamp=timestamp_from_report_id(analysis_run_id) or timestamp_from_report_id(report_dir.name),
                source="run_index.csv",
            )
        )
    return candidates


def folder_candidates_for_label(reports_root: Path, label: str) -> list[ReportCandidate]:
    if not reports_root.exists():
        return []
    candidates: list[ReportCandidate] = []
    for path in reports_root.iterdir():
        if not path.is_dir() or path.name in {"neurips_paper_assets", "neurips_paper_assets_v2"}:
            continue
        match = TIMESTAMPED_REPORT_RE.match(path.name)
        if match and match.group("label") == label:
            candidates.append(
                ReportCandidate(
                    label=label,
                    report_dir=path,
                    report_id=path.name,
                    timestamp=match.group("timestamp"),
                    source="folder_scan",
                )
            )
    return candidates


def staged_candidates_for_label(source_existing: Path, label: str) -> list[ReportCandidate]:
    if not source_existing.exists():
        return []
    candidates: list[ReportCandidate] = []
    for path in source_existing.iterdir():
        if not path.is_dir():
            continue
        match = STAGED_REPORT_RE.match(path.name)
        if not match or match.group("label") != label:
            continue
        report_id = match.group("report_id")
        candidates.append(
            ReportCandidate(
                label=label,
                report_dir=path,
                report_id=report_id,
                timestamp=timestamp_from_report_id(report_id),
                source="source_existing",
            )
        )
    return candidates


def latest(candidates: Iterable[ReportCandidate]) -> ReportCandidate | None:
    ordered = sorted(candidates, key=lambda item: (item.timestamp, item.report_id), reverse=True)
    return ordered[0] if ordered else None


def select_reports(root: Path, diag: Diagnostics) -> dict[str, ReportCandidate]:
    reports_root = root / "reports"
    source_existing = reports_root / "neurips_paper_assets" / "source_existing"
    selected: dict[str, ReportCandidate] = {}
    for label in TARGET_LABELS:
        candidates = index_candidates_for_label(reports_root, label) + folder_candidates_for_label(reports_root, label)
        if not candidates:
            candidates = staged_candidates_for_label(source_existing, label)
            if candidates:
                diag.note(f"`{label}` selected from staged `source_existing` assets because no live report folder was found.")
        chosen = latest(candidates)
        if chosen is None:
            diag.missing_labels[label] = "No matching run_index.csv row, timestamped report folder, or staged source_existing folder was found."
            continue
        selected[label] = chosen
    return selected


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def empty_figure(base_path: Path, title: str, message: str, diag: Diagnostics | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.text(0.5, 0.48, message, ha="center", va="center", wrap=True, fontsize=10)
    save_figure(fig, base_path)
    if diag is not None:
        diag.figure(base_path.with_suffix(".pdf").name)
        diag.figure(base_path.with_suffix(".png").name)


# CR-14 (tasks 2 / P22): readable condition names. Archival run IDs live in table footnotes,
# never in a column. Model / thinking / host strings are quoted from the CR-10/10b/11/12
# CONFIG_DISCLOSURE files (vault Camera-Ready/assets/cr-10*, cr-11, cr-12); nothing is inferred.
CONDITIONS = {
    "CR_MAIN_SYNTH_NUM_SONNET5_S42": {
        "name": "Claude Sonnet 5",
        "persona_source": "Synthetic profiles", "spec": "Numeric",
        "model": "Claude Sonnet 5 (claude-sonnet-5)", "config": "provider defaults; adaptive thinking on",
    },
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": {
        "name": "Gemini 3.5 Flash",
        "persona_source": "Synthetic profiles", "spec": "Numeric",
        "model": "Gemini 3.5 Flash (gemini-3.5-flash)", "config": "thinking level minimal",
    },
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": {
        "name": "DeepSeek V4-Pro",
        "persona_source": "Synthetic profiles", "spec": "Numeric",
        "model": "DeepSeek V4-Pro (via Together AI)", "config": "reasoning effort high",
    },
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42": {
        "name": "Claude Sonnet 4.6 · human · numeric",
        "persona_source": "Human-derived profiles", "spec": "Numeric",
        "model": "Claude Sonnet 4.6 (claude-sonnet-4-6)", "config": "provider defaults",
    },
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": {
        "name": "Claude Sonnet 4.6 · human · lexical",
        "persona_source": "Human-derived profiles", "spec": "Lexical",
        "model": "Claude Sonnet 4.6 (claude-sonnet-4-6)", "config": "provider defaults",
    },
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42": {
        "name": "Claude Sonnet 4.6 · synthetic · numeric",
        "persona_source": "Synthetic profiles", "spec": "Numeric",
        "model": "Claude Sonnet 4.6 (claude-sonnet-4-6)", "config": "provider defaults",
    },
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42": {
        "name": "Claude Sonnet 4.6 · synthetic · lexical",
        "persona_source": "Synthetic profiles", "spec": "Lexical",
        "model": "Claude Sonnet 4.6 (claude-sonnet-4-6)", "config": "provider defaults",
    },
}
FOOTNOTE_MARKS = "abcdefghij"
# S-8b (presentation only): the persona-source wording printed in Table 2 (table_experimental_runs.tex); the CSV keeps
# the stored string.
PERSONA_SOURCE_DISPLAY = {"Synthetic bank v2": "Synthetic profiles"}


def condition_name(label: object) -> str:
    return CONDITIONS.get(str(label), {}).get("name", str(label))


_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "<": r"$<$", ">": r"$>$"}


def tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in text)


def tex_id(value: object) -> str:
    """An archival identifier in typewriter type that may break after any underscore (CR-23 STEP 6:
    unbreakable \\texttt IDs ran the footnotes past the right margin)."""
    return "\\texttt{" + tex_escape(value).replace("\\_", "\\_\\allowbreak ") + "}"


def tex_head(*lines: str) -> str:
    """A header cell set on one or more lines, so narrow numeric columns keep narrow headers."""
    if len(lines) == 1:
        return lines[0]
    return "\\begin{tabular}[b]{@{}c@{}}" + " \\\\ ".join(lines) + "\\end{tabular}"


def tex_x(weight: float, align: str = "raggedright") -> str:
    """A tabularx X column carrying `weight` of the average X width (weights in one table sum to the
    number of X columns)."""
    return f">{{\\hsize={weight:.2f}\\hsize\\{align}\\arraybackslash}}X"


def tex_booktabs(
    header: list[str],
    rows: list[list[str]],
    colspec: str,
    caption: str,
    label: str,
    notes: list[str] | None = None,
    panels: list[tuple[str, list[list[str]]]] | None = None,
    font: str = r"\footnotesize",
) -> str:
    """A booktabs table that fits the text block: tabularx at \\linewidth (the X columns absorb the
    width and wrap), no \\resizebox scaling, a real caption and label, and footnotes set beneath the
    rule. CR-23 STEP 6: every table fits \\linewidth at the NeurIPS text width (5.5 in). Requires
    \\usepackage{booktabs,tabularx}. `panels` splits one table into labelled blocks sharing the header."""
    env = "tabularx" if "X" in colspec else "tabular"
    width = "{\\linewidth}" if env == "tabularx" else ""
    lines = [
        "% Generated by scripts/build_neurips_camera_ready_assets.py (CR-14, layout CR-23). Do not hand-edit.",
        "% Requires \\usepackage{booktabs,tabularx}.",
        "\\begin{table}[t]",
        "\\centering",
        font,
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{{env}}}{width}{{{colspec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    blocks = panels if panels is not None else [("", rows)]
    for bi, (title, block) in enumerate(blocks):
        if title:
            if bi:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{len(header)}}}{{@{{}}p{{\\linewidth}}@{{}}}}{{\\textit{{{title}}}}} \\\\")
        for r in block:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", f"\\end{{{env}}}"]
    if notes:
        lines.append("\\par\\smallskip")
        lines.append("\\begin{minipage}{\\linewidth}\\footnotesize\\raggedright")
        for n in notes:
            lines.append(n + "\\par")
        lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def fmt_num(v: object, nd: int = 3) -> str:
    x = safe_float(v)
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}".replace("-", "$-$")


def fmt_p(v: object) -> str:
    x = safe_float(v)
    if not np.isfinite(x):
        return "--"
    # CR-23 STEP 6: one p-value style in every table -- no leading zero, "<.001" below .001.
    if x < 0.001:
        return "$<$.001"
    text = f"{x:.3f}"
    return "1.000" if text == "1.000" else text.lstrip("0")


def archival_footnotes() -> list[str]:
    out = []
    for i, label in enumerate(TARGET_LABELS):
        out.append(f"\\textsuperscript{{{FOOTNOTE_MARKS[i]}}}\\,{tex_escape(condition_name(label))}: "
                   f"simulator run {tex_id(label)}; analysis report {tex_id('cr13_' + str(label))}.")
    return out


def display_run_label(value: object, short: bool = False) -> str:
    text = str(value)
    mapping = RUN_SHORT if short else RUN_DISPLAY
    return mapping.get(text, text)


def clean_label(value: object) -> str:
    text = str(value)
    return CLEAN_LABELS.get(text, text.replace("_", " "))


def clean_category_label(value: object) -> str:
    return CLEAN_LABELS.get(str(value), str(value).replace("_", " "))


def clean_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_label" in out.columns:
        out["run_label"] = out["run_label"].map(display_run_label)
    for col in ("construct", "outcome", "liwc_outcome", "variable", "measure", "predictor", "model_type"):
        if col in out.columns:
            out[col] = out[col].map(clean_label)
    if "category" in out.columns:
        out["category"] = out["category"].map(clean_category_label)
    return out


def write_table(df: pd.DataFrame, path: Path, tex: bool = False, note: str | None = None, diag: Diagnostics | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = clean_for_output(df)
    out.to_csv(path, index=False)
    if diag is not None:
        diag.table(path.name)
    if tex:
        tex_path = path.with_suffix(".tex")
        latex_body = out.to_latex(index=False, escape=True)
        tex_text = "\\resizebox{\\linewidth}{!}{%\n" + latex_body + "}\n"
        if note:
            tex_text += "\n% Note: " + note.replace("\n", " ") + "\n"
        tex_path.write_text(tex_text, encoding="utf-8")
        if diag is not None:
            diag.table(tex_path.name)


def ci95_mean(mean: float, sd: float, n: float) -> tuple[float, float]:
    if not np.isfinite([mean, sd, n]).all() or n <= 1:
        return np.nan, np.nan
    half = 1.96 * sd / math.sqrt(n)
    return mean - half, mean + half


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if not np.isfinite([k, n]).all() or n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def choose_run_dataset(df: pd.DataFrame, label: str) -> str | None:
    for col in ("socsim_run_label", "run_label", "run_id"):
        if col in df.columns and label in set(df[col].astype(str)):
            rows = df[df[col].astype(str) == label]
            if "dataset" in rows.columns and not rows["dataset"].dropna().empty:
                return str(rows["dataset"].dropna().iloc[0])
    if "dataset" not in df.columns:
        return None
    datasets = [str(x) for x in df["dataset"].dropna().unique()]
    for preferred in ("socsim_agent", "cogsim_agent", "agent", "sim_agent", "synthetic"):
        if preferred in datasets:
            return preferred
    non_human = [x for x in datasets if x != "human"]
    return non_human[0] if non_human else None


def prop_vector_from_summary(row: pd.Series) -> tuple[np.ndarray, list[float]]:
    prop_cols = []
    for col in row.index:
        match = re.match(r"^prop_(-?\d+(?:\.\d+)?)$", str(col))
        if match:
            prop_cols.append((float(match.group(1)), col))
    prop_cols.sort(key=lambda item: item[0])
    levels = [level for level, _ in prop_cols]
    probs = np.array([safe_float(row[col]) for _, col in prop_cols], dtype=float)
    valid = np.isfinite(probs)
    if valid.any() and np.nansum(probs) > 0:
        probs = np.where(valid, probs, 0.0)
        probs = probs / probs.sum()
    else:
        probs = np.array([], dtype=float)
        levels = []
    return probs, levels


def js_from_rows(human_row: pd.Series, run_row: pd.Series) -> float:
    human_probs, human_levels = prop_vector_from_summary(human_row)
    run_probs, run_levels = prop_vector_from_summary(run_row)
    if len(human_probs) and human_levels == run_levels and np.isfinite(human_probs).all() and np.isfinite(run_probs).all():
        return float(jensenshannon(human_probs, run_probs))
    return np.nan


def load_descriptives(candidate: ReportCandidate, diag: Diagnostics) -> pd.DataFrame:
    return read_csv_optional(candidate.report_dir / "tables" / "00_descriptives_overview.csv", diag, candidate.label)


def load_overlap(candidate: ReportCandidate, diag: Diagnostics) -> pd.DataFrame:
    return read_csv_optional(candidate.report_dir / "tables" / "comparisons" / "05_overlap_summary.csv", diag, candidate.label)


def load_univariate(candidate: ReportCandidate, diag: Diagnostics) -> pd.DataFrame:
    return read_csv_optional(candidate.report_dir / "tables" / "comparisons" / "05_univariate_comparisons.csv", diag, candidate.label)


def row_for(df: pd.DataFrame, dataset: str, variable: str) -> pd.Series | None:
    if df.empty or "dataset" not in df.columns or "variable" not in df.columns:
        return None
    rows = df[(df["dataset"].astype(str) == dataset) & (df["variable"].astype(str) == variable)]
    return rows.iloc[0] if not rows.empty else None


def univariate_row(df: pd.DataFrame, run_dataset: str, variable: str) -> pd.Series | None:
    if df.empty or "variable" not in df.columns:
        return None
    rows = df[df["variable"].astype(str) == variable]
    if {"dataset_a", "dataset_b"}.issubset(rows.columns):
        rows = rows[(rows["dataset_a"].astype(str) == "human") & (rows["dataset_b"].astype(str) == run_dataset)]
    return rows.iloc[0] if not rows.empty else None


def build_distribution_assets(selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    human_summary: dict[str, pd.Series] = {}

    for label, candidate in selected.items():
        overview = load_descriptives(candidate, diag)
        overlap = load_overlap(candidate, diag)
        univariate = load_univariate(candidate, diag)
        summary = overlap if not overlap.empty else overview
        run_dataset = choose_run_dataset(summary, label)
        if run_dataset is None:
            diag.panel("fig2_distribution_fidelity_dashboard", f"No non-human dataset rows were found for `{label}`.")
            continue
        for construct in DISTRIBUTION_CONSTRUCTS:
            human_row = row_for(summary, "human", construct)
            run_row = row_for(summary, run_dataset, construct)
            if human_row is None:
                human_row = human_summary.get(construct)
            else:
                human_summary.setdefault(construct, human_row)
            if human_row is None or run_row is None:
                diag.column(f"{label} distribution", construct, "human or run summary row missing")
                continue
            human_mean = safe_float(human_row.get("mean"))
            run_mean = safe_float(run_row.get("mean"))
            human_sd = safe_float(human_row.get("SD"))
            run_sd = safe_float(run_row.get("SD"))
            n_human = sample_size_from_row(human_row)
            n_run = sample_size_from_row(run_row)
            uni = univariate_row(univariate, run_dataset, construct)
            cohens_d = safe_float(uni.get("cohens_d")) if uni is not None else np.nan
            variance_ratio = (run_sd * run_sd) / (human_sd * human_sd) if human_sd and np.isfinite([human_sd, run_sd]).all() else np.nan
            js_div = js_from_rows(human_row, run_row)
            rows.append(
                {
                    "run_label": label,
                    "construct": construct,
                    "human_mean": human_mean,
                    "run_mean": run_mean,
                    "mean_difference": run_mean - human_mean if np.isfinite([human_mean, run_mean]).all() else np.nan,
                    "cohens_d": cohens_d,
                    "js_divergence": js_div,
                    "variance_ratio": variance_ratio,
                    "n_human": n_human,
                    "n_run": n_run,
                }
            )
            for source, source_label, source_row, color_key in (
                ("human", "Human reference", human_row, "human"),
                (label, RUN_SHORT.get(label, label), run_row, label),
            ):
                probs, levels = prop_vector_from_summary(source_row)
                ci_low, ci_high = ci95_mean(
                    safe_float(source_row.get("mean")),
                    safe_float(source_row.get("SD")),
                    sample_size_from_row(source_row),
                )
                plot_rows.append(
                    {
                        "source": source,
                        "label": source_label,
                        "construct": construct,
                        "mean": safe_float(source_row.get("mean")),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "levels": levels,
                        "probs": probs,
                        "color": RUN_COLORS[color_key],
                        "marker": RUN_MARKERS.get(color_key),   # None for the human reference (a plain black line)
                    }
                )

    table = pd.DataFrame(rows)
    write_table(table, dirs["tables"] / "table_distribution_fidelity_metrics.csv", diag=diag)
    if not plot_rows:
        empty_figure(
            dirs["figures"] / "fig2_distribution_fidelity_dashboard",
            "Distribution Fidelity Dashboard",
            "No eligible distribution summary rows were found for the selected runs.",
            diag,
        )
        return table

    plot_df = pd.DataFrame(plot_rows).drop_duplicates(subset=["source", "construct"])
    _DIST_PLOT_DF["df"] = plot_df   # S-8: read by fig_main2_focal_grid after the calibration table is written
    plot_distribution_grid(plot_df, dirs, diag)
    plot_distribution_ecdf_overlay(plot_df, dirs["figures"] / "fig2b_distribution_fidelity_ecdf_overlay", diag)
    return table


def plot_distribution_ecdf_overlay(plot_df: pd.DataFrame, base_path: Path, diag: Diagnostics) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.2), constrained_layout=True)
    for ax, construct in zip(axes.flat, DISTRIBUTION_CONSTRUCTS, strict=False):
        subset = plot_df[plot_df["construct"] == construct]
        for _, row in subset.iterrows():
            levels = np.array(row["levels"], dtype=float)
            probs = np.array(row["probs"], dtype=float)
            if len(levels) == 0 or len(probs) == 0:
                continue
            order = np.argsort(levels)
            x = levels[order]
            y = np.cumsum(probs[order])
            is_human = row["source"] == "human"
            ax.step(
                x,
                y,
                where="post",
                color=HUMAN_INK if is_human else row["color"],
                lw=2.6 if is_human else 1.6,
                alpha=0.95 if is_human else 0.82,
                label="Human reference" if is_human else row["label"],
                marker=None if is_human else row["marker"],
                markersize=3.2,
            )
        ax.set_title(short_construct(construct), fontsize=10, weight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Cumulative proportion")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color="#eeeeee", lw=0.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=5, frameon=False)
    fig.suptitle("Distribution fidelity ECDF overlays", fontsize=13, weight="bold")
    save_figure(fig, base_path)
    diag.figure(base_path.with_suffix(".pdf").name)
    diag.figure(base_path.with_suffix(".png").name)


def short_construct(name: str) -> str:
    return clean_label(name)


# S-8 (PI ruling 2026-10-12): the appendix distribution figure is a small-multiples grid, one row per construct and
# one column per run, columns grouped under their model family. It replaces the overlaid dashboard and its
# mean +/- 95% CI strips. Same data as before; the corner numbers are read from the same table main_fig2 prints.
GRID_FAMILIES = (
    ("Claude Sonnet 5", ("CR_MAIN_SYNTH_NUM_SONNET5_S42",)),
    ("Gemini 3.5 Flash", ("CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42",)),
    ("DeepSeek V4-Pro", ("CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42",)),
    ("Claude Sonnet 4.6", ("CR_BASELINE_HUMAN_NUMERIC_CS46_S42", "CR_BASELINE_HUMAN_LEXICAL_CS46_S42",
                           "CR_MAIN_SYNTH_NUMERIC_CS46_S42", "CR_MAIN_SYNTH_LEXICAL_CS46_S42")),
)
GRID_SUBLABEL = {"CR_BASELINE_HUMAN_NUMERIC_CS46_S42": "Human\nnumeric", "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": "Human\nlexical",
                 "CR_MAIN_SYNTH_NUMERIC_CS46_S42": "Synthetic\nnumeric", "CR_MAIN_SYNTH_LEXICAL_CS46_S42": "Synthetic\nlexical"}
_DIST_PLOT_DF: dict[str, pd.DataFrame] = {}


FOCAL_CONSTRUCTS = ["CRT2_Total", "BCRT_Total", "NFC_Total", "BFI_Conscientiousness"]
FOCAL_COLUMNS = (   # (header, sub-header, numeric run, lexical run or None) - PI ruling D2 option A, 2026-10-13
    ("Claude\nSonnet 5", None, "CR_MAIN_SYNTH_NUM_SONNET5_S42", None),
    ("Gemini\n3.5 Flash", None, "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42", None),
    ("DeepSeek\nV4-Pro", None, "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42", None),
    ("Claude Sonnet 4.6", "Human-derived\npersonas", "CR_BASELINE_HUMAN_NUMERIC_CS46_S42", "CR_BASELINE_HUMAN_LEXICAL_CS46_S42"),
    ("Claude Sonnet 4.6", "Synthetic\npersonas", "CR_MAIN_SYNTH_NUMERIC_CS46_S42", "CR_MAIN_SYNTH_LEXICAL_CS46_S42"),
)


# S-8e (D-291(c)): the two CRTs have four items, so their score axes run 0-4; the summaries carry an always-empty
# score-5 bin, which is cropped (guarded empty) rather than drawn as a tick under a four-item test.
FOUR_ITEM_TESTS = ("CRT2_Total", "BCRT_Total")
# S-8e (D-293): constructs never specified to a persona carry no calibration display.
UNSPECIFIED_CONSTRUCTS = ("BCRT_Total",)
UNSPECIFIED_CALIBRATION_ROWS = ("bCRT-derived score",)


def score_ticks(construct: str, levels) -> list[float]:
    ticks = [t for t in levels if float(t).is_integer()]
    return [t for t in ticks if t <= 4] if construct in FOUR_ITEM_TESTS else ticks


def score_xlim(construct: str, edges, plot_df: pd.DataFrame) -> tuple[float, float]:
    if construct not in FOUR_ITEM_TESTS:
        return float(edges[0]), float(edges[-1])
    for _, r in plot_df[plot_df["construct"] == construct].iterrows():
        lv, pv = np.asarray(r["levels"], dtype=float), np.asarray(r["probs"], dtype=float)
        if len(pv) and float(pv[lv > 4].sum()) != 0.0:
            raise SystemExit(f"STOP: {construct} has mass above 4 for {r['source']}; the 0-4 axis would hide it.")
    return float(edges[0]), 4.5


def fig_main2_focal_grid(dirs: dict[str, Path], diag: Diagnostics) -> None:
    """S-8 main-text Figure 2 (PI rulings 2026-10-13, C-16): the four focal constructs as rows, five columns (the four
    Claude Sonnet 4.6 conditions folded to two by persona source: numeric = filled bars, lexical = dashed outline, same
    hue - lightness never separates runs, D-210). Corner: JS (distribution table) and calibration R^2 (calibration table),
    numeric then lexical where a cell holds two runs. No prose inside the figure: the caption carries it (C-20 b)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plot_df = _DIST_PLOT_DF["df"]
    inv = {v: k for k, v in RUN_DISPLAY.items()}
    dist = pd.read_csv(dirs["tables"] / "table_distribution_fidelity_metrics.csv")
    cal = pd.read_csv(dirs["tables"] / "table_calibration_fidelity.csv")
    dist["label"] = dist["run_label"].map(inv)
    cal["label"] = cal["run_label"].map(inv)
    cal_name = {"bCRT": "bCRT-derived score"}

    def js_of(label, name):
        d = dist[(dist["label"] == label) & (dist["construct"] == name)]
        return f"{float(d['js_divergence'].iloc[0]):.2f}"

    def r2_of(label, name):
        k = cal[(cal["label"] == label) & (cal["construct"] == cal_name.get(name, name))]
        v = float(k["r2"].iloc[0]) if len(k) else np.nan
        return f"{v:.2f}" if np.isfinite(v) else "n/e"

    def probs_of(label, construct):
        r = plot_df[(plot_df["source"] == label) & (plot_df["construct"] == construct)]
        return np.asarray(r["levels"].iloc[0], dtype=float), np.asarray(r["probs"].iloc[0], dtype=float)

    fig, axes = plt.subplots(len(FOCAL_CONSTRUCTS), len(FOCAL_COLUMNS), figsize=(5.1, 4.05), squeeze=False)
    fig.subplots_adjust(left=0.205, right=0.995, top=0.80, bottom=0.115, wspace=0.10, hspace=0.42)
    for i, construct in enumerate(FOCAL_CONSTRUCTS):
        name = short_construct(construct)
        hl, hp = probs_of("human", construct)
        step = np.min(np.diff(hl)) if len(hl) > 1 else 1.0
        edges = np.concatenate([[hl[0] - step / 2], hl + step / 2])
        row_max = float(hp.max())
        for _, _, num, lex in FOCAL_COLUMNS:
            for lab in (num, lex):
                if lab:
                    row_max = max(row_max, float(probs_of(lab, construct)[1].max()))
        for j, (_, _, num, lex) in enumerate(FOCAL_COLUMNS):
            ax = axes[i, j]
            hue = RUN_COLORS[num]
            lv, pv = probs_of(num, construct)
            ax.bar(lv, pv, width=0.8 * step, color=hue, alpha=0.45, lw=0)
            if lex:
                _, pl = probs_of(lex, construct)
                ax.stairs(pl, edges, color=hue, lw=1.1, ls=(0, (2.2, 1.2)), zorder=4)
            ax.stairs(hp, edges, color=HUMAN_INK, lw=1.0, zorder=3)
            js_s = js_of(num, name) + (f" · {js_of(lex, name)}" if lex else "")
            corner = f"JS {js_s}"
            if construct not in UNSPECIFIED_CONSTRUCTS:   # S-8e (D-293): the bCRT was never specified, so no R^2
                r2_s = r2_of(num, name) + (f" · {r2_of(lex, name)}" if lex else "")
                if r2_s == "n/e · n/e":
                    r2_s = "n/e"
                corner += f"\nR² {r2_s}"
            ax.text(0.04, 0.97, corner, transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    linespacing=1.05, color="#111111")
            ax.set_ylim(0, row_max * 1.75)
            ax.set_xlim(*score_xlim(construct, edges, plot_df))
            ticks = score_ticks(construct, hl)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks] if j == 0 else [], fontsize=7)
            ax.tick_params(axis="both", length=2, pad=1, labelsize=7)
            if j == 0:
                ax.set_ylabel(name, fontsize=7.5, weight="bold", rotation=0, ha="right", va="center", labelpad=4)
                y_tick = 1.0 if row_max >= 0.95 else 0.5 if row_max >= 0.45 else 0.2
                ax.set_yticks([0, y_tick])
                ax.set_yticklabels(["0", f"{y_tick:g}"])
            else:
                ax.set_yticks([])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.canvas.draw()
    top = axes[0, 0].get_position().y1
    b3, b4 = axes[0, 3].get_position(), axes[0, 4].get_position()
    hue46 = RUN_COLORS[FOCAL_COLUMNS[3][2]]
    for j, (head, sub, num, _) in enumerate(FOCAL_COLUMNS):
        b = axes[0, j].get_position()
        if sub is None:
            fig.text((b.x0 + b.x1) / 2, top + 0.012, head, ha="center", va="bottom", fontsize=7.5, weight="bold",
                     color=RUN_COLORS[num], linespacing=1.0)
        else:
            fig.text((b.x0 + b.x1) / 2, top + 0.012, sub, ha="center", va="bottom", fontsize=7, color=hue46, linespacing=1.0)
    fig.text((b3.x0 + b4.x1) / 2, top + 0.072, "Claude Sonnet 4.6", ha="center", va="bottom", fontsize=7.5, weight="bold",
             color=hue46)
    fig.add_artist(Line2D([b3.x0 + 0.005, b4.x1 - 0.005], [top + 0.066, top + 0.066], color=hue46, lw=0.8))
    handles = [Patch(facecolor="#888888", alpha=0.45, lw=0, label="run (numeric format)"),
               Line2D([], [], color="#888888", lw=1.1, ls=(0, (2.2, 1.2)), label="lexical format (Sonnet 4.6)"),
               Line2D([], [], color=HUMAN_INK, lw=1.0, label="human")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, handlelength=1.8,
               bbox_to_anchor=(0.6, 0.0))
    save_figure(fig, dirs["figures"] / "main_fig2_focal_distribution_grid")
    diag.figure("main_fig2_focal_distribution_grid.pdf")
    diag.figure("main_fig2_focal_distribution_grid.png")


def plot_distribution_grid(plot_df: pd.DataFrame, dirs: dict[str, Path], diag: Diagnostics) -> None:
    """Appendix figure: every run's score distribution against the human one, construct by construct.

    Drawn at print size (5.3 in + 0.1 in tight-bbox pad each side = the 5.5 in NeurIPS text width) so no text is
    scaled down; nothing is smaller than 7 pt.
    """
    metrics = pd.read_csv(dirs["tables"] / "table_distribution_fidelity_metrics.csv")
    inv = {v: k for k, v in RUN_DISPLAY.items()}
    metrics["label"] = metrics["run_label"].map(inv)
    runs = [label for _, members in GRID_FAMILIES for label in members]
    n_rows, n_cols = len(DISTRIBUTION_CONSTRUCTS), len(runs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.3, 7.9), squeeze=False)   # + 0.1 in tight-bbox pad each side = 5.5 in
    fig.subplots_adjust(left=0.235, right=0.995, top=0.925, bottom=0.03, wspace=0.12, hspace=0.42)
    for i, construct in enumerate(DISTRIBUTION_CONSTRUCTS):
        name = short_construct(construct)
        human = plot_df[(plot_df["source"] == "human") & (plot_df["construct"] == construct)]
        h_levels = list(human["levels"].iloc[0]) if len(human) else []
        h_probs = np.asarray(human["probs"].iloc[0]) if len(human) else np.array([])
        row_max = float(h_probs.max()) if len(h_probs) else 0.0
        for label in runs:
            r = plot_df[(plot_df["source"] == label) & (plot_df["construct"] == construct)]
            if len(r) and len(r["probs"].iloc[0]):
                row_max = max(row_max, float(np.max(r["probs"].iloc[0])))
        y_top = row_max * 1.55 if row_max > 0 else 1.0   # headroom: the corner numbers sit above the tallest bar
        for j, label in enumerate(runs):
            ax = axes[i, j]
            r = plot_df[(plot_df["source"] == label) & (plot_df["construct"] == construct)]
            if len(r) and len(r["probs"].iloc[0]):
                levels = np.asarray(r["levels"].iloc[0], dtype=float)
                width = 0.8 * (np.min(np.diff(levels)) if len(levels) > 1 else 1.0)
                ax.bar(levels, r["probs"].iloc[0], width=width, color=RUN_COLORS[label], alpha=0.45, lw=0)
            if len(h_probs):
                hl = np.asarray(h_levels, dtype=float)
                step = np.min(np.diff(hl)) if len(hl) > 1 else 1.0
                edges = np.concatenate([[hl[0] - step / 2], hl + step / 2])
                ax.stairs(h_probs, edges, color=HUMAN_INK, lw=1.0, zorder=3)
            m = metrics[(metrics["label"] == label) & (metrics["construct"] == name)]
            if len(m):
                js_v, shift = float(m["js_divergence"].iloc[0]), float(m["mean_difference"].iloc[0])
                shift_s = f"{shift:+.2f}"
                shift_s = "0.00" if shift_s in ("+0.00", "-0.00") else shift_s.replace("-", "\u2212")
                ax.text(0.04, 0.96, f"JS {js_v:.2f}\nΔ {shift_s}", transform=ax.transAxes, ha="left", va="top",
                        fontsize=7, linespacing=1.05, color="#111111")
            ax.set_ylim(0, y_top)
            if len(h_levels):
                ax.set_xlim(*score_xlim(construct, edges, plot_df))
                ticks = score_ticks(construct, h_levels)
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}" for t in ticks] if j == 0 else [], fontsize=7)
            ax.tick_params(axis="both", length=2, pad=1, labelsize=7)
            if j == 0:
                # construct names on one line, never broken (PI 2026-10-12); the label column is widened for them
                ax.set_ylabel(name, fontsize=7.5, weight="bold", rotation=0, ha="right",
                              va="center", labelpad=4, linespacing=1.0)
                y_tick = 1.0 if row_max >= 0.95 else 0.5 if row_max >= 0.45 else 0.2   # same rule as the focal grid
                ax.set_yticks([0, y_tick])
                ax.set_yticklabels(["0", f"{y_tick:g}"])
            else:
                ax.set_yticks([])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    # column headers: run names under their family, in the family hue (VOICE.md rule B)
    fig.canvas.draw()
    col = 0
    for family, members in GRID_FAMILIES:
        boxes = [axes[0, col + k].get_position() for k in range(len(members))]
        x0, x1 = boxes[0].x0, boxes[-1].x1
        hue = RUN_COLORS[members[0]]
        top = boxes[0].y1
        if len(members) > 1:
            fig.text((x0 + x1) / 2, top + 0.044, family, ha="center", va="bottom", fontsize=8, weight="bold", color=hue)
            fig.add_artist(plt.Line2D([x0 + 0.005, x1 - 0.005], [top + 0.041, top + 0.041], color=hue, lw=0.8))
            for k, label in enumerate(members):
                b = boxes[k]
                fig.text((b.x0 + b.x1) / 2, top + 0.008, GRID_SUBLABEL[label], ha="center", va="bottom", fontsize=7,
                         color=hue, linespacing=1.0)
        else:
            fig.text((x0 + x1) / 2, top + 0.008, family.replace(" ", "\n", 1) if len(family) > 12 else family,
                     ha="center", va="bottom", fontsize=7.5, weight="bold", color=hue, linespacing=1.0)
        col += len(members)
    save_figure(fig, dirs["figures"] / "fig2_distribution_fidelity_dashboard")
    diag.figure("fig2_distribution_fidelity_dashboard.pdf")
    diag.figure("fig2_distribution_fidelity_dashboard.png")


# S-8e (PI 2 Nov, round-2 Q5 A): two-lane workflow at print width (5.5 in), no internal title, nothing below 7 pt.
# People answer the cognition suite; their profiles (as given, or via the synthetic bank) condition LLM agents,
# which answer the same suite (decoys marked unscored); both are compared on the five fidelity targets of Section 3.
SCHEM_INK, SCHEM_AGENT, SCHEM_MUTED, SCHEM_FILL_P, SCHEM_FILL_A, SCHEM_FILL_F = "#111111", "#4b3fa0", "#6b6b6b", "#f2f2f2", "#eeecf8", "#fbf6e9"
SCHEM_ACCENT = "#b07d12"


def _schem_box(ax, x, y, w, h, title, body, ec, fc, title_c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.6, weight="bold", color=title_c)
    if body:
        ax.text(x + w / 2, y + h - 0.105, body, ha="center", va="top", fontsize=7, color="#222222", linespacing=1.15)


def _schem_arrow(ax, a, b, color, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def build_schematic(dirs: dict[str, Path]) -> None:
    fig = plt.figure(figsize=(5.5, 2.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # lane labels, rotated in the left margin
    ax.text(0.012, 0.785, "People", fontsize=7.4, weight="bold", color=SCHEM_INK, va="center", ha="center", rotation=90)
    ax.text(0.012, 0.345, "Agents", fontsize=7.4, weight="bold", color=SCHEM_AGENT, va="center", ha="center", rotation=90)
    ax.plot([0.03, 0.775], [0.665, 0.665], color="#d6d6d6", lw=0.6, ls=(0, (2, 2)))
    # people lane
    _schem_box(ax, 0.04, 0.70, 0.21, 0.17, "Human sample", "473 adults", SCHEM_INK, SCHEM_FILL_P, SCHEM_INK)
    _schem_box(ax, 0.42, 0.70, 0.27, 0.17, "Cognition suite", "the same measures", SCHEM_INK, SCHEM_FILL_P, SCHEM_INK)
    _schem_arrow(ax, (0.255, 0.785), (0.415, 0.785), SCHEM_INK)
    # agent lane
    _schem_box(ax, 0.04, 0.37, 0.21, 0.23, "Human-derived", "each person's\nown profile", SCHEM_AGENT, SCHEM_FILL_A, SCHEM_AGENT)
    _schem_box(ax, 0.04, 0.07, 0.21, 0.23, "Synthetic", "drawn from a bank\nfitted to people", SCHEM_AGENT, SCHEM_FILL_A, SCHEM_AGENT)
    _schem_box(ax, 0.30, 0.20, 0.19, 0.29, "LLM agents", "persona prompt:\ntraits and a\nreflection level", SCHEM_AGENT, SCHEM_FILL_A, SCHEM_AGENT)
    _schem_box(ax, 0.535, 0.07, 0.22, 0.53, "Cognition suite", "BRT\n\nreasoning: CRT2, bCRT\n(+ unscored decoys)\n\nNeed for Cognition\n\nBFI-10", SCHEM_AGENT, SCHEM_FILL_A, SCHEM_AGENT)
    _schem_arrow(ax, (0.255, 0.485), (0.295, 0.40), SCHEM_AGENT)
    _schem_arrow(ax, (0.255, 0.185), (0.295, 0.29), SCHEM_AGENT)
    _schem_arrow(ax, (0.495, 0.345), (0.53, 0.345), SCHEM_AGENT)
    # every profile comes from people
    _schem_arrow(ax, (0.145, 0.695), (0.145, 0.605), SCHEM_MUTED, ls=(0, (2, 1.5)))
    ax.text(0.155, 0.65, "profiles", fontsize=7, color=SCHEM_MUTED, va="center")
    # comparison
    _schem_box(ax, 0.79, 0.07, 0.20, 0.80, "Compared on", "", SCHEM_ACCENT, SCHEM_FILL_F, SCHEM_ACCENT)
    targets = ["marginal\ndistributions", "calibration", "error type", "association\nstructure", "behavioural\ntransfer"]
    for k, t in enumerate(targets):
        y = 0.72 - k * 0.14
        ax.add_patch(FancyBboxPatch((0.805, y - 0.05), 0.17, 0.10, boxstyle="round,pad=0.003,rounding_size=0.012",
                                    fc="white", ec=SCHEM_ACCENT, lw=0.6))
        ax.text(0.89, y, t, ha="center", va="center", fontsize=7, color="#222222", linespacing=0.95)
    _schem_arrow(ax, (0.695, 0.785), (0.785, 0.785), SCHEM_INK)
    _schem_arrow(ax, (0.76, 0.345), (0.785, 0.345), SCHEM_AGENT)
    save_figure(fig, dirs["figures"] / "fig1_cognition_suite_schematic")

def candidate_csvs(candidate: ReportCandidate, keywords: Iterable[str]) -> list[Path]:
    words = tuple(k.lower() for k in keywords)
    if not candidate.report_dir.exists():
        return []
    return sorted(
        p
        for p in candidate.report_dir.rglob("*.csv")
        if all(word in p.name.lower() or word in p.as_posix().lower() for word in words)
    )


def any_keyword_csvs(candidate: ReportCandidate, keywords: Iterable[str]) -> list[Path]:
    words = tuple(k.lower() for k in keywords)
    if not candidate.report_dir.exists():
        return []
    return sorted(p for p in candidate.report_dir.rglob("*.csv") if any(word in p.name.lower() for word in words))


def normalise_crt_distribution(df: pd.DataFrame, label: str, source_path: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    rows: list[dict[str, object]] = []
    if any(f"human_prop_{cat}" in df.columns for cat in CRT_CATEGORY_LEVELS):
        for _, row in df.iterrows():
            instrument = str(row.get("instrument") or row.get("test") or "").lower()
            if instrument not in {"crt2", "bcrt"}:
                continue
            for category in CRT_CATEGORY_LEVELS:
                rows.append(
                    {
                        "run_label": "human",
                        "dataset": "human",
                        "test": "CRT2" if instrument == "crt2" else "bCRT",
                        "item_id": row.get("item_id", ""),
                        "specified_level": row.get("specified_level", np.nan),
                        "category": category,
                        "count": np.nan,
                        "proportion": safe_float(row.get(f"human_prop_{category}")),
                        "source_file": source_path.name,
                    }
                )
                rows.append(
                    {
                        "run_label": label,
                        "dataset": str(row.get("comparison") or row.get("dataset") or "agent"),
                        "test": "CRT2" if instrument == "crt2" else "bCRT",
                        "item_id": row.get("item_id", ""),
                        "specified_level": row.get("specified_level", np.nan),
                        "category": category,
                        "count": np.nan,
                        "proportion": safe_float(row.get(f"sim_prop_{category}")),
                        "source_file": source_path.name,
                    }
                )
        return pd.DataFrame(rows)

    category_col = cols.get("response_category") or cols.get("category")
    if category_col is None:
        return pd.DataFrame()
    instrument_col = cols.get("instrument") or cols.get("test")
    item_col = cols.get("item_id")
    level_col = cols.get("specified_level") or cols.get("crt2_level")
    prop_col = cols.get("proportion") or cols.get("prop")
    count_col = cols.get("count") or cols.get("n")
    dataset_col = cols.get("dataset")
    run_col = cols.get("run_id") or cols.get("run_label") or cols.get("socsim_run_label")
    if instrument_col is None or (prop_col is None and count_col is None):
        return pd.DataFrame()

    work = df.copy()
    work["_instrument"] = work[instrument_col].astype(str).str.lower().str.replace(" ", "")
    work = work[work["_instrument"].isin(["crt2", "crt", "bcrt"])]
    if work.empty:
        return pd.DataFrame()
    if prop_col is None:
        group_cols = [dataset_col, run_col, instrument_col, category_col]
        group_cols = [c for c in group_cols if c is not None]
        counts = work.groupby(group_cols, dropna=False)[count_col].sum().reset_index()
        totals = counts.groupby([c for c in group_cols if c != category_col], dropna=False)[count_col].transform("sum")
        counts["_proportion"] = counts[count_col] / totals
        work = counts
        prop_col = "_proportion"
    for _, row in work.iterrows():
        instrument = str(row[instrument_col]).lower()
        run_value = str(row[run_col]) if run_col and pd.notna(row.get(run_col)) else ""
        dataset = str(row[dataset_col]) if dataset_col and pd.notna(row.get(dataset_col)) else ""
        is_human = dataset == "human" or run_value == "human"
        rows.append(
            {
                "run_label": "human" if is_human else label,
                "dataset": dataset,
                "test": "CRT2" if "crt2" in instrument or instrument == "crt" else "bCRT",
                "item_id": row.get(item_col, "") if item_col is not None else "",
                "specified_level": row.get(level_col, np.nan) if level_col is not None else np.nan,
                "category": str(row[category_col]),
                "count": safe_float(row[count_col]) if count_col is not None else np.nan,
                "proportion": safe_float(row[prop_col]),
                "source_file": source_path.name,
            }
        )
    return pd.DataFrame(rows)


HUMAN_V2_LONG = Path("reports") / "cr13_human" / "crt2_long_human_v2_frame473.csv"
HUMAN_REFERENCE_N = 471
UNSCORED = "missing_or_unparseable"
# CR-23 ruling (d), one convention everywhere: a rate is its count / SCORED responses (missing or
# unparseable excluded from the denominator); the unscored count is printed beside it.
LURE_CONVENTION_NOTE = (
    "Rates are counts over scored responses: missing or unparseable responses are excluded from the "
    "denominator and reported as the unscored count. Human reference: the 471 CRT2-complete respondents."
)


def scored_proportions(counts: pd.DataFrame, keys: list[str], count_col: str = "count") -> pd.Series:
    """count / scored responses within `keys` (CR-23 ruling (d)); NaN for the unscored category."""
    cat_col = "category" if "category" in counts.columns else "response_category"
    scored = counts[count_col].where(counts[cat_col] != UNSCORED, 0.0)
    denom = scored.groupby([counts[k] for k in keys], dropna=False).transform("sum")
    prop = counts[count_col] / denom.replace(0, np.nan)
    return prop.where(counts[cat_col] != UNSCORED)
_V2_TO_CATEGORY = {"correct": "correct", "intuitive": "intuitive_incorrect",
                   "other": "other_incorrect"}


def human_crt_categories(root: Path, diag: Diagnostics) -> pd.DataFrame:
    """The human CRT2/bCRT response-category mix from the v2 scoring labels.

    CR-13B defect fix: the per-run pipeline's `05_crt_response_category_distribution.csv`
    carries human rows with only `correct` and `missing_or_unparseable` — the human frame
    has no per-item response_category columns, so every human lure was being recorded as
    zero. That made `intuitive_gap` (Table 4) equal to the agent's own lure rate rather
    than a gap. The human mix is taken here from CR-13H's v2 long frame, the same source
    CR-03's adjusted-reference machinery reads.
    """
    path = root / HUMAN_V2_LONG
    if not path.exists():
        raise SystemExit(
            f"STOP: human v2 long frame missing at {path}; it is CR-13H's "
            "`crt2_long_human_v2_frame473.csv` and the human CRT category mix needs it."
        )
    long = read_csv_keep_text(path, HUMAN_LONG_TEXT_COLS)
    long = long.copy()
    long["category"] = (long["label_v2"].astype(str).str.strip().str.lower()
                        .map(_V2_TO_CATEGORY).fillna("missing_or_unparseable"))
    if "normalised" in long.columns:
        blank = long["normalised"].fillna("").astype(str).eq("")
        long.loc[blank, "category"] = "missing_or_unparseable"
    # CR-23 ruling (d): the human reference is the 471 CRT2-complete people (the persona frame),
    # not the 473-person frame, for CRT2 and bCRT alike.
    crt2 = long[long["item_id"].astype(str).str.startswith("crt2")]
    scored_items = crt2[crt2["category"] != "missing_or_unparseable"].groupby("entity_id")["item_id"].nunique()
    complete = set(scored_items[scored_items == 4].index.astype(str))
    if len(complete) != HUMAN_REFERENCE_N:
        raise SystemExit(f"STOP: {len(complete)} CRT2-complete humans in {path}, expected {HUMAN_REFERENCE_N}")
    long = long[long["entity_id"].astype(str).isin(complete)]
    long["test"] = np.where(long["item_id"].astype(str).str.startswith("crt2"), "CRT2", "bCRT")
    rows = (long.groupby(["test", "item_id", "category"], dropna=False)
            .size().reset_index(name="count"))
    rows["run_label"] = "human"
    rows["specified_level"] = np.nan
    totals = rows.groupby(["test", "item_id"], dropna=False)["count"].transform("sum")
    rows["proportion"] = rows["count"] / totals
    diag.note("Human CRT2/bCRT response-category mix taken from the v2 scoring labels "
              "(`reports/cr13_human/crt2_long_human_v2_frame473.csv`), restricted to the "
              f"{HUMAN_REFERENCE_N} CRT2-complete respondents (CR-23 ruling (d)); the pipeline's own "
              "human rows carry no lure category and were recording every human lure as 0.")
    return rows[["run_label", "test", "item_id", "specified_level", "category",
                 "count", "proportion"]]


def load_crt_data(selected: dict[str, ReportCandidate], diag: Diagnostics) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, candidate in selected.items():
        paths = [
            candidate.report_dir / "tables" / "comparisons" / "05_crt_response_category_distribution.csv",
            candidate.report_dir / "tables" / "comparisons" / "05_crt_response_category_by_specified_level.csv",
        ]
        for path in [p for p in paths if p.exists()]:
            df = read_csv_optional(path)
            norm = normalise_crt_distribution(df, label, path)
            if not norm.empty:
                frames.append(norm)
    if not frames:
        diag.panel("fig3_crt_bcrt_intuitive_lure_fidelity", "No CSV with usable CRT/bCRT response-category totals was found.")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["category"] = out["category"].replace({"missing": "missing_or_unparseable"})
    out = out[out["category"].isin(CRT_CATEGORY_LEVELS)]
    out = out.drop_duplicates(
        subset=["run_label", "test", "item_id", "specified_level", "category", "count", "proportion"],
        keep="first",
    )
    # Replace the pipeline's human rows with the v2-label mix (see human_crt_categories).
    out = out[out["run_label"].astype(str) != "human"]
    out = pd.concat([out, human_crt_categories(repo_root(), diag)], ignore_index=True)
    return out


def collapse_crt_props(crt: pd.DataFrame) -> pd.DataFrame:
    if crt.empty:
        return pd.DataFrame(columns=["run_label", "test", "category", "count", "proportion"])
    work = crt[crt["specified_level"].isna()].copy() if "specified_level" in crt.columns else crt.copy()
    if work["count"].notna().any():
        grouped = work.groupby(["run_label", "test", "category"], dropna=False)["count"].sum().reset_index()
        grouped["proportion"] = scored_proportions(grouped, ["run_label", "test"])
        return grouped
    grouped = work.groupby(["run_label", "test", "category"], dropna=False)["proportion"].mean().reset_index()
    grouped["count"] = np.nan
    return grouped


CR13B_CRT_TABLES = Path("reports") / "cr13b_crt" / "tables"


def attach_adjusted_reference_js(table: pd.DataFrame, root: Path, diag: Diagnostics) -> pd.DataFrame:
    """Merge the adjusted-reference JS and the per-item lure profile onto the CRT table."""
    js_path = root / CR13B_CRT_TABLES / "js_category_mass_by_run.csv"
    lure_path = root / CR13B_CRT_TABLES / "per_item_lure_profiles_beside_js.csv"
    if not js_path.exists() or not lure_path.exists():
        raise SystemExit(
            f"STOP: adjusted-reference JS tables missing under {root / CR13B_CRT_TABLES}. "
            "Run `scripts/cr13b/run_crt_assets.py` first (CR-13B STEP 5)."
        )
    js = pd.read_csv(js_path)
    lure = pd.read_csv(lure_path)
    # `table` may carry either the raw run label or its display name depending on where
    # in the build it is merged, so both are normalised to the raw label before joining.
    display_to_raw = {v: k for k, v in RUN_DISPLAY.items()}

    def to_raw(series: pd.Series) -> pd.Series:
        return series.astype(str).map(lambda x: display_to_raw.get(x, x))

    js["__key"] = to_raw(js["run_label"])
    lure["__key"] = to_raw(lure["run_label"])
    table = table.copy()
    table["__key"] = to_raw(table["run_label"])

    agg = js.rename(columns={
        "js_crt2_vs_raw": "js_crt2_pooled_vs_raw_v2_human",
        "js_crt2_vs_adjusted": "js_crt2_pooled_vs_adjusted_human",
        "js_crt2_delta": "js_crt2_adjusted_minus_raw",
        "js_bcrt_vs_raw": "js_bcrt_pooled_vs_raw_v2_human",
        "conclusion_flag_js_moved_ge_0p02": "js_conclusion_changed_flag",
    })[["__key", "js_crt2_pooled_vs_raw_v2_human",
        "js_crt2_pooled_vs_adjusted_human", "js_crt2_adjusted_minus_raw",
        "js_bcrt_pooled_vs_raw_v2_human", "js_conclusion_changed_flag"]]

    wide = lure.pivot_table(index="__key", columns="item_id", values="cell_lure_pct")
    wide.columns = [f"lure_pct_{c}" for c in wide.columns]
    wide = wide.reset_index()
    hum = lure.drop_duplicates("item_id").set_index("item_id")
    for item in sorted(hum.index):
        wide[f"human_raw_lure_pct_{item}"] = float(hum.loc[item, "human_raw_lure_pct"])
        adj = hum.loc[item, "human_adj_lure_pct"]
        wide[f"human_adj_lure_pct_{item}"] = float(adj) if pd.notna(adj) else np.nan

    out = table.merge(agg, on="__key", how="left").merge(wide, on="__key", how="left")
    missing = int(out["js_crt2_pooled_vs_adjusted_human"].isna().sum())
    if missing:
        raise SystemExit(
            f"STOP: adjusted-reference JS did not join for {missing} of {len(out)} CRT rows "
            f"(run labels: {sorted(out.loc[out['js_crt2_pooled_vs_adjusted_human'].isna(), 'run_label'].unique())})"
        )
    out = out.drop(columns=["__key"])
    out["crt2_reference"] = ("familiarity-adjusted @ seen=0 (CR-03, recomputed on the "
                             "corrected 473/471 frame); raw v2 shown beside it")
    out["bcrt_reference"] = "raw v2 human (bCRT is unadjusted)"
    diag.note("CRT2 JS is reported against CR-03's familiarity-adjusted reference "
              "(D-73/D-74), with the raw v2 figure beside it and the per-item lure "
              "profile in the same row (D-73c).")
    return out


def format_pct(p: float) -> str:
    """Exact percentage; a non-zero value below 0.05% keeps two decimals so it never rounds to 0.0%."""
    if p > 0 and p < 0.0005:
        return f"{100 * p:.2f}%"
    return f"{100 * p:.1f}%"


def build_lure_prevalence_export(props: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    """CR-14 task 8: exact intuitive-lure prevalence per run x task, written beside Fig 2C."""
    rows: list[dict[str, object]] = []
    for label in ["human", *TARGET_LABELS]:
        for test in ("CRT2", "bCRT"):
            sub = props[(props["run_label"] == label) & (props["test"] == test)]
            if sub.empty or not sub["count"].notna().any():
                continue
            unscored = float(sub.loc[sub["category"] == UNSCORED, "count"].sum())
            total = float(sub["count"].sum()) - unscored  # CR-23 ruling (d): scored responses only
            lure = float(sub.loc[sub["category"] == "intuitive_incorrect", "count"].sum())
            lo, hi = wilson_ci(lure, total)
            rows.append(
                {
                    "run_key": label,
                    "run_label": label,
                    "test": test,
                    "intuitive_count": int(round(lure)),
                    "n_responses": int(round(total)),
                    "prevalence": lure / total if total else np.nan,
                    "prevalence_pct": round(100 * lure / total, 4) if total else np.nan,
                    "wilson_ci_low": lo,
                    "wilson_ci_high": hi,
                    "correct_count": int(round(sub.loc[sub["category"] == "correct", "count"].sum())),
                    "other_incorrect_count": int(round(sub.loc[sub["category"] == "other_incorrect", "count"].sum())),
                    "unscored_count": int(round(unscored)),
                    "note": "n_responses = scored responses. " + LURE_CONVENTION_NOTE,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = df.copy()
    out["condition"] = out["run_label"].map(lambda v: "Human reference (Prolific, v2 scoring labels)" if v == "human" else condition_name(v))
    out = out.drop(columns=["run_key"])
    cols = ["run_label", "condition", "test"] + [c for c in out.columns if c not in {"run_label", "condition", "test"}]
    out[cols].to_csv(dirs["tables"] / "table_intuitive_lure_prevalence_by_run_task.csv", index=False)
    return df


def build_crt_assets(selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    crt = load_crt_data(selected, diag)
    props = collapse_crt_props(crt)
    rows: list[dict[str, object]] = []
    for label in TARGET_LABELS:
        for test in ("CRT2", "bCRT"):
            human = props[(props["run_label"] == "human") & (props["test"] == test)]
            run = props[(props["run_label"] == label) & (props["test"] == test)]
            human_vec = np.array([human.loc[human["category"] == c, "proportion"].sum() for c in CRT_CATEGORY_LEVELS], dtype=float)
            run_vec = np.array([run.loc[run["category"] == c, "proportion"].sum() for c in CRT_CATEGORY_LEVELS], dtype=float)
            js = float(jensenshannon(human_vec, run_vec)) if human_vec.sum() > 0 and run_vec.sum() > 0 else np.nan
            for category in CRT_CATEGORY_LEVELS:
                h = float(human.loc[human["category"] == category, "proportion"].sum()) if not human.empty else np.nan
                r = float(run.loc[run["category"] == category, "proportion"].sum()) if not run.empty else np.nan
                rows.append(
                    {
                        "run_label": label,
                        "test": test,
                        "category": category,
                        "human_prop": h,
                        "run_prop": r,
                        "absolute_gap": abs(r - h) if np.isfinite([h, r]).all() else np.nan,
                        "js_divergence_test_level": js,
                        "n_human_items": float(human["count"].sum()) if human["count"].notna().any() else np.nan,
                        "n_run_items": float(run["count"].sum()) if run["count"].notna().any() else np.nan,
                        "n_human_scored": float(human.loc[human["category"] != UNSCORED, "count"].sum()),
                        "n_run_scored": float(run.loc[run["category"] != UNSCORED, "count"].sum()),
                        "n_human_unscored": float(human.loc[human["category"] == UNSCORED, "count"].sum()),
                        "n_run_unscored": float(run.loc[run["category"] == UNSCORED, "count"].sum()),
                        "rate_convention": LURE_CONVENTION_NOTE,
                    }
                )
    table = pd.DataFrame(rows)
    # D-73/D-74: the raw v2 human reference is exposure-contaminated, so the camera-ready
    # CRT2 JS is the one against CR-03's familiarity-adjusted reference, recomputed on the
    # corrected 473/471 frame by `scripts/cr13b/run_crt_assets.py`. bCRT stays raw v2
    # (bCRT is unadjusted). D-73c: the per-item lure profile travels with every aggregate
    # JS, so the adjusted figures are merged in here rather than published on their own.
    table = attach_adjusted_reference_js(table, repo_root(), diag)
    write_table(table, dirs["tables"] / "table_crt_bcrt_response_category_fidelity.csv", diag=diag)
    if props.empty:
        empty_figure(
            dirs["figures"] / "fig3_crt_bcrt_intuitive_lure_fidelity",
            "CRT/bCRT intuitive-lure fidelity",
            "No existing response-category totals were available.",
            diag,
        )
        return table

    fig = plt.figure(figsize=(13.2, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.62, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    plot_crt_stacked(ax_a, props, "CRT2")
    plot_crt_stacked(ax_b, props, "bCRT")
    # CR-14 (P21 / CR-14 tasks 1 and 8): panel C uses the SAME colour semantics as A and B --
    # orange = intuitive incorrect everywhere. The two tasks are told apart by position (adjacent
    # bars) and hatching (CRT2 solid, bCRT hatched), never by reusing a colour for a task. Every bar,
    # including the zero bars that read as "missing data" in the submitted figure, carries its exact
    # value and its count. The same numbers are exported beside the figure.
    prevalence_df = build_lure_prevalence_export(props, dirs)
    if not prevalence_df.empty:
        x_labels = ["human", *TARGET_LABELS]
        x = np.arange(len(x_labels)) * 1.75
        bar_width = 0.42
        lure_color = CRT_CATEGORY_COLORS["intuitive_incorrect"]
        max_prevalence = 0.0
        for test, offset, hatch in (("CRT2", -(bar_width / 2 + 0.06), ""), ("bCRT", bar_width / 2 + 0.06, "///")):
            sub = prevalence_df[prevalence_df["test"] == test].set_index("run_key")
            vals = [float(sub.loc[k, "prevalence"]) if k in sub.index else np.nan for k in x_labels]
            finite_vals = [v for v in vals if np.isfinite(v)]
            max_prevalence = max(max_prevalence, max(finite_vals, default=0.0))
            bars = ax_c.bar(x + offset, vals, width=bar_width, color=lure_color, edgecolor="#5a2d00" if hatch else "white",
                            hatch=hatch, linewidth=0.8, zorder=3)
            for xi, k, v in zip(x + offset, x_labels, vals, strict=False):
                if not np.isfinite(v):
                    continue
                cnt = int(sub.loc[k, "intuitive_count"])
                tot = int(sub.loc[k, "n_responses"])
                ax_c.text(xi, v + 0.008, format_pct(v), ha="center", va="bottom", fontsize=9.5, weight="bold", zorder=4)
        ax_c.set_xticks(x)
        tick_text = []
        for k in x_labels:
            parts = [RUN_SHORT.get(k, k) if k != "human" else "Human ref."]
            for test in ("CRT2", "bCRT"):
                sub_t = prevalence_df[(prevalence_df["test"] == test) & (prevalence_df["run_key"] == k)]
                if not sub_t.empty:
                    parts.append(f"{test}: {int(sub_t.iloc[0]['intuitive_count'])}/{int(sub_t.iloc[0]['n_responses'])}")
            tick_text.append("\n".join(parts))
        ax_c.set_xticklabels(tick_text, fontsize=10)
        ax_c.set_xlim(x[0] - 1.0, x[-1] + 1.0)
        ax_c.set_ylim(0, min(1.0, max(0.12, max_prevalence * 1.22)))
        ax_c.set_ylabel("Intuitive-incorrect prevalence", fontsize=12)
        ax_c.set_title("C. Intuitive-lure prevalence by run", loc="left", weight="bold", fontsize=13)
        ax_c.grid(axis="y", color="#eeeeee", zorder=0)
        ax_c.tick_params(axis="y", labelsize=11)
        from matplotlib.patches import Patch
        ax_c.legend(
            handles=[
                Patch(facecolor=lure_color, edgecolor="white", label="CRT2 (solid, left bar)"),
                Patch(facecolor=lure_color, edgecolor="#5a2d00", hatch="///", label="bCRT (hatched, right bar)"),
            ],
            frameon=False, loc="upper right", ncol=2, fontsize=11, handletextpad=0.5, columnspacing=1.4,
        )
    else:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "No intuitive-incorrect prevalence rows available.", ha="center", va="center")
    handles, labels = ax_a.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4, frameon=False, fontsize=12, handlelength=1.7, columnspacing=1.5)
    fig.suptitle("CRT2 and bCRT response-category fidelity", fontsize=15, weight="bold", y=0.98)
    fig.subplots_adjust(bottom=0.17, top=0.90)
    save_figure(fig, dirs["figures"] / "fig3_crt_bcrt_intuitive_lure_fidelity")
    diag.figure("fig3_crt_bcrt_intuitive_lure_fidelity.pdf")
    diag.figure("fig3_crt_bcrt_intuitive_lure_fidelity.png")
    return table


def plot_crt_stacked(ax: plt.Axes, props: pd.DataFrame, test: str) -> None:
    labels = ["human", *TARGET_LABELS]
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    subset = props[props["test"] == test]
    for category in CRT_CATEGORY_LEVELS:
        vals = [float(subset[(subset["run_label"] == label) & (subset["category"] == category)]["proportion"].sum()) for label in labels]
        ax.barh(y, vals, left=left, color=CRT_CATEGORY_COLORS[category], edgecolor="white", label=clean_category_label(category))
        left += np.array(vals)
    ax.set_yticks(y)
    ax.set_yticklabels(["Human reference" if label == "human" else RUN_SHORT.get(label, label) for label in labels], fontsize=11)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Response-category proportion", fontsize=12)
    ax.set_title(("A. " if test == "CRT2" else "B. ") + f"{test} category proportions", loc="left", weight="bold", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)


BY_LEVEL_TASK_TITLES = {
    "crt2": "CRT2 items, by told CRT2 level",
    "bcrt": "bCRT items, by told CRT2 level",
}
BY_LEVEL_YLABELS = {"crt2": "Told CRT2 level", "bcrt": "Told CRT2 level"}
BY_LEVEL_GROUPING = {
    "crt2": "CRT2 level the agent was told (crt2_level)",
    "bcrt": "bCRT grouped by the CRT2 level the agent was told (bCRT is not directly specified; D-41)",
}


def build_by_level_assets(root: Path, selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    """CR-25 STEP 2 (PI-4; D-190(c), D-200(c)): the response categories are grouped by `crt2_level`, the level
    each agent was told, one group per level 0-4. The CR-14 version grouped by the rounded specified score
    (`CRT2_Total_adj_specified`) and folded a rounded -1 into 0; only about half of the agents land in the
    same group under both. bCRT was never specified, so its panels group the same agents by their told CRT2
    level. Every bar carries its lure prevalence and n (agents and item responses)."""
    raw = load_told_level_counts(root, selected)
    if raw.empty:
        diag.panel("appendix_crt_response_categories_by_crt_level", "No per-run CRT response-category tables were found.")
        return pd.DataFrame()
    grp = told_level_rates(raw)
    grp["rate_convention"] = LURE_CONVENTION_NOTE
    export = grp.copy()
    export.insert(1, "condition", export["run_label"].map(condition_name))
    export["task"] = export["instrument"].map({"crt2": "CRT2", "bcrt": "bCRT"})
    export["grouping"] = export["instrument"].map(BY_LEVEL_GROUPING)
    export["told_level_source"] = "crt2_level (person frame: the level written into the persona)"
    export = export.drop(columns=["instrument"])
    export.to_csv(dirs["tables"] / "table_crt_response_categories_by_specified_level.csv", index=False)
    diag.table("table_crt_response_categories_by_specified_level.csv")

    runs = [l for l in TARGET_LABELS if l in set(grp["run_label"])]
    levels = list(range(5))
    fig, axes = plt.subplots(len(runs), 2, figsize=(13.0, 2.25 * len(runs) + 1.9), squeeze=False, sharex=True)
    for ri, label in enumerate(runs):
        for ci, inst in enumerate(("crt2", "bcrt")):
            ax = axes[ri, ci]
            sub = grp[(grp["run_label"] == label) & (grp["instrument"] == inst)]
            y = np.arange(len(levels))
            left = np.zeros(len(levels))
            for category in CRT_CATEGORY_LEVELS:
                vals = np.array([float(sub[(sub["told_level"] == lv) & (sub["response_category"] == category)]["proportion"].sum())
                                 for lv in levels])
                ax.barh(y, vals, left=left, color=CRT_CATEGORY_COLORS[category], edgecolor="white", height=0.78,
                        label=clean_category_label(category))
                left += vals
            for yi, lv in enumerate(levels):
                at = sub[sub["told_level"] == lv]
                n_agents = int(at["n_agents"].iloc[0])
                n = int(at["n_responses"].iloc[0])
                lure = float(at[at["response_category"] == "intuitive_incorrect"]["proportion"].sum())
                ax.text(1.015, yi, f"lure {format_pct(lure)}  ({n_agents} / {n})", va="center", fontsize=8.5)
            ax.set_yticks(y)
            ax.set_yticklabels([str(lv) for lv in levels], fontsize=10)
            ax.set_ylim(len(levels) - 0.4, -0.6)
            ax.set_xlim(0, 1)
            ax.tick_params(axis="x", labelsize=10)
            if ci == 0:
                ax.set_ylabel(condition_name(label).replace("Synthetic, ", "Synthetic,\n").replace("Human-derived, ", "Human-derived,\n")
                              + "\n(N = 471)", rotation=0, ha="right", va="center", fontsize=10, weight="bold", labelpad=8)
            if ri == 0:
                ax.set_title(BY_LEVEL_TASK_TITLES[inst], loc="left", fontsize=11, weight="bold")
            if ri == len(runs) - 1:
                ax.set_xlabel("Response-category proportion", fontsize=11)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Which response categories does each told CRT2 level produce?", fontsize=13, weight="bold", y=0.995)
    fig.text(0.5, 0.045, "Rows within each panel are the CRT2 level the agent was told, 0-4 (crt2_level). bCRT itself was never "
             "specified: its panels group the same agents by their told CRT2 level. (a / r) = agents / item responses in the group.",
             ha="center", va="top", fontsize=9, color="#333333", wrap=True)
    fig.subplots_adjust(left=0.2, right=0.8, top=0.94, bottom=0.10, hspace=0.6, wspace=0.75)
    save_figure(fig, dirs["figures"] / "appendix_crt_response_categories_by_crt_level")
    diag.figure("appendix_crt_response_categories_by_crt_level.pdf")
    diag.figure("appendix_crt_response_categories_by_crt_level.png")
    return grp


def load_brt_descriptives(root: Path, candidate: ReportCandidate, diag: Diagnostics) -> pd.DataFrame:
    paths = []
    for base in source_roots_for_candidate(root, candidate):
        paths.extend(
            [
                base / "tables" / "06_brt_descriptives_all_datasets.csv",
                base / "tables" / "by_dataset" / "socsim_agent" / "06_brt_descriptives.csv",
            ]
        )
    frames = [read_csv_optional(path) for path in paths if path.exists()]
    if not frames:
        diag.file(candidate.label, paths[0], "BRT descriptives not found")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def load_brt_models(root: Path, candidate: ReportCandidate) -> pd.DataFrame:
    paths = []
    for base in source_roots_for_candidate(root, candidate):
        paths.extend(
            [
                base / "tables" / "06_brt_primary_models_all_datasets.csv",
                base / "tables" / "by_dataset" / "socsim_agent" / "06_brt_primary_models.csv",
                base / "tables" / "by_dataset" / "human" / "06_brt_primary_models.csv",
            ]
        )
    frames = [read_csv_optional(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def load_brt_person_sources(root: Path, label: str) -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    human_paths = [
        root / "reports" / "brt_recode_v3" / "brt_labels_v3_human.csv",   # canonical (D-149)
    ]
    for path in human_paths:
        df = read_csv_optional(path)
        if not df.empty:
            require_brt_primary_columns(df, rel(path, root))
            sources["human"] = df
            break

    agent_paths = [
        # CR-13B: the per-run analysis-ready frame the pipeline persists for each run.
        # The PI's disclosed exclusions are already applied in it. The two fallbacks
        # below are the single ROTATING copies, correct only for the run last ingested.
        root / "reports" / f"cr13_{label}" / "tables" / "by_dataset" / "socsim_agent"
        / "00_person_frame.csv",
        root / "data" / "intermediate" / "socsim_agent_person_brt_coded.csv",
        root / "data" / "inputs" / f"socsim_per_persona_results_{label}.csv",
    ]
    for path in agent_paths:
        df = read_csv_optional(path)
        if df.empty:
            continue
        if "socsim_run_label" in df.columns:
            df = df[df["socsim_run_label"].astype(str) == label].copy()
        if not df.empty:
            require_brt_primary_columns(df, f"{rel(path, root)} [{label}]")
            sources["socsim_agent"] = df
            break
    return sources


def brt_prevalence_from_person(df: pd.DataFrame, outcome: str) -> tuple[float, float, float]:
    if df.empty or outcome not in df.columns:
        return np.nan, np.nan, np.nan
    values = pd.to_numeric(df[outcome], errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, np.nan
    count = float((values == 1).sum())
    n = float(values.isin([0, 1]).sum())
    return (count / n if n > 0 else np.nan), count, n


def brt_prevalence_from_descriptives(df: pd.DataFrame, dataset: str, outcome: str) -> tuple[float, float, float]:
    if df.empty or "measure" not in df.columns:
        return np.nan, np.nan, np.nan
    mapping: dict[str, tuple[str, set[str]]] = {}   # no §5 primary outcome is a categorical-descriptive recode
    if outcome not in mapping:
        return np.nan, np.nan, np.nan
    measure, positives = mapping[outcome]
    rows = df[(df["dataset"].astype(str) == dataset) & (df["measure"].astype(str) == measure)]
    if rows.empty:
        return np.nan, np.nan, np.nan
    total = safe_float(rows["N"].dropna().iloc[0]) if "N" in rows and not rows["N"].dropna().empty else safe_float(rows["count"].sum())
    count = float(rows[rows["category"].astype(str).isin(positives)]["count"].sum()) if "count" in rows else np.nan
    prop = count / total if total and np.isfinite([count, total]).all() else float(rows[rows["category"].astype(str).isin(positives)]["proportion"].sum())
    return prop, count, total


def brt_prevalence_from_models(df: pd.DataFrame, dataset: str, outcome: str) -> tuple[float, float, float]:
    if df.empty or "outcome" not in df.columns:
        return np.nan, np.nan, np.nan
    rows = df[df["outcome"].astype(str) == outcome]
    if "dataset" in rows.columns:
        rows = rows[rows["dataset"].astype(str) == dataset]
    if rows.empty:
        return np.nan, np.nan, np.nan
    fallback_n = np.nan
    for _, row in rows.iterrows():
        prev = safe_float(row.get("outcome_prevalence"))
        n = safe_float(row.get("N"))
        fallback_n = n if np.isfinite(n) else fallback_n
        if np.isfinite(prev):
            return prev, prev * n if np.isfinite(n) else np.nan, n
        events = safe_float(row.get("event_count"))
        non_events = safe_float(row.get("non_event_count"))
        if np.isfinite([events, non_events]).all() and (events + non_events) > 0:
            return events / (events + non_events), events, events + non_events
    return np.nan, np.nan, fallback_n


def write_brt_prevalence_recovery_report(
    root: Path,
    selected: dict[str, ReportCandidate],
    dirs: dict[str, Path],
    diag: Diagnostics,
    brt: pd.DataFrame,
) -> None:
    label = "HumanNum467sim"
    candidate = selected.get(label)
    lines = [
        "# BRT Prevalence Recovery Report",
        "",
        "Recovery target: `Human / Numeric` primary BRT binary outcomes.",
        "",
    ]
    if candidate is None:
        lines.append("- Human / Numeric report was not selected.")
        (dirs["diagnostics"] / "brt_prevalence_recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    aliases = {c: [c] for c in BRT_PRIMARY_OUTCOMES}
    roots = source_roots_for_candidate(root, candidate)
    searched: list[str] = []
    findings: list[str] = []
    for base in roots:
        for path in sorted(base.rglob("*.csv")):
            if "brt" not in path.name.lower() and "simulator_inputs" not in path.as_posix().lower():
                continue
            searched.append(path.name)
            try:
                cols = pd.read_csv(path, nrows=0).columns.tolist()
            except Exception as exc:
                findings.append(f"- `{path.name}`: unreadable CSV header ({exc}).")
                continue
            lower_cols = {c.lower(): c for c in cols}
            hits = []
            prevalence_fields = [
                actual
                for col_lower, actual in lower_cols.items()
                if col_lower in {"event_count", "non_event_count", "outcome_prevalence", "n"}
            ]
            for outcome, pats in aliases.items():
                for pat in pats:
                    for col_lower, actual in lower_cols.items():
                        if pat.lower() in col_lower:
                            hits.append(f"{clean_label(outcome)} -> `{actual}`")
            if hits:
                detail = "; ".join(sorted(set(hits)))
                if prevalence_fields:
                    detail += "; prevalence fields: " + ", ".join(f"`{field}`" for field in sorted(set(prevalence_fields)))
                findings.append(f"- `{path.name}`: found {detail}.")
            elif prevalence_fields:
                findings.append(
                    f"- `{path.name}`: found prevalence fields "
                    + ", ".join(f"`{field}`" for field in sorted(set(prevalence_fields)))
                    + "."
                )
            else:
                findings.append(f"- `{path.name}`: no primary binary BRT prevalence columns or event-count fields found.")
    lines.extend(["## Files Searched", ""])
    lines.extend(f"- `{name}`" for name in sorted(set(searched))) if searched else lines.append("- None.")
    lines.extend(["", "## Findings", ""])
    lines.extend(findings or ["- No staged BRT CSVs were available to search."])
    run_labels = brt.get("run_label", pd.Series(dtype=str)).astype(str) if not brt.empty else pd.Series(dtype=str)
    hn = brt[run_labels.isin({label, "Human / Numeric"})] if not brt.empty else pd.DataFrame()
    complete = (
        not hn.empty
        and hn["human_prevalence"].map(safe_float).apply(np.isfinite).all()
        and hn["run_prevalence"].map(safe_float).apply(np.isfinite).all()
    )
    lines.extend(["", "## Outcome", ""])
    if complete:
        sources = sorted(
            set(hn["human_prevalence_source"].dropna().astype(str))
            | set(hn["run_prevalence_source"].dropna().astype(str))
        )
        lines.append(
            f"The selected Human / Numeric report `{candidate.report_id}` has complete prevalence for all "
            f"{len(hn)} primary BRT binary outcomes. Recovered prevalence sources: "
            + ", ".join(f"`{source}`" for source in sources)
            + "."
        )
    else:
        lines.append(
            "Human / Numeric BRT prevalence remains incomplete for one or more primary binary outcomes; "
            "inspect `table_brt_distributional_fidelity.csv` for the affected rows and source fields."
        )
    (dirs["diagnostics"] / "brt_prevalence_recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


PART2_TABLE = Path("reports") / "cr13b_partii" / "tables" / "s11_part2_primary_outcomes_by_run.csv"


def load_part2(root: Path) -> pd.DataFrame:
    """CR-13B's ANALYSIS_PLAN §11 Part II table — the ONE producer of F1/F2/F3.

    The per-run pipeline's `06_brt_primary_models_*.csv` predates the ratified plan and
    carries the older outcome names, so it cannot supply the §5 primary F3 slopes. Those
    come from `scripts/cr13b/run_partii.py`, which fits them under the plan's estimators
    and its C2/C3 rules. Reading them here keeps one set of Part II numbers in the bundle.
    """
    path = root / PART2_TABLE
    if not path.exists():
        raise SystemExit(
            f"STOP: Part II table missing at {path}. Run `scripts/cr13b/run_partii.py` "
            "before building the camera-ready assets (CR-13B STEP 5)."
        )
    return pd.read_csv(path)


def part2_row(part2: pd.DataFrame, label: str, outcome: str) -> pd.Series | None:
    sub = part2[(part2["run_label"].astype(str) == label)
                & (part2["outcome"].astype(str) == outcome)]
    return sub.iloc[0] if not sub.empty else None


def f3_slope(models: pd.DataFrame, dataset: str, outcome: str) -> tuple[float, float]:
    """Logit slope on realised bCRT (§11 F3) and its SE, from the run's model table."""
    if models.empty or not {"outcome", "predictor", "beta"}.issubset(models.columns):
        return np.nan, np.nan
    rows = models[(models["outcome"].astype(str) == outcome)
                  & models["predictor"].astype(str).isin(F3_PREDICTORS)]
    if "dataset" in rows.columns:
        rows = rows[rows["dataset"].astype(str) == dataset]
    if rows.empty:
        return np.nan, np.nan
    row = rows.iloc[0]
    beta = safe_float(row.get("beta"))
    se = safe_float(row.get("se", row.get("SE")))
    if not np.isfinite(se):
        lo, hi = safe_float(row.get("CI95_low")), safe_float(row.get("CI95_high"))
        se = (hi - lo) / (2 * 1.959964) if np.isfinite([lo, hi]).all() else np.nan
    return beta, se


def build_brt_assets(root: Path, selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    categorical_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []
    part2 = load_part2(root)
    for label, candidate in selected.items():
        desc = load_brt_descriptives(root, candidate, diag)
        models = load_brt_models(root, candidate)
        person_sources = load_brt_person_sources(root, label)
        if not desc.empty:
            run_dataset = choose_run_dataset(desc, label) or "socsim_agent"
            categorical = desc[desc.get("measure", pd.Series(dtype=str)).astype(str).isin(BRT_CATEGORICAL_MEASURES)].copy()
            categorical["run_label"] = categorical["dataset"].astype(str).map({"human": "human"}).fillna(label)
            categorical_frames.append(categorical)
        else:
            run_dataset = "socsim_agent"
        if not models.empty:
            model_frames.append(models.assign(run_label=label))
        for outcome in BRT_PRIMARY_OUTCOMES:
            h_source = ""
            r_source = ""
            h_prev, h_count, h_n = brt_prevalence_from_descriptives(desc, "human", outcome)
            r_prev, r_count, r_n = brt_prevalence_from_descriptives(desc, run_dataset, outcome)
            if np.isfinite(h_prev):
                h_source = "brt_descriptives"
            if np.isfinite(r_prev):
                r_source = "brt_descriptives"
            if not np.isfinite(h_prev):
                h_prev, h_count, h_n = brt_prevalence_from_person(person_sources.get("human", pd.DataFrame()), outcome)
                if np.isfinite(h_prev):
                    h_source = "person_level_binary"
            if not np.isfinite(r_prev):
                r_prev, r_count, r_n = brt_prevalence_from_person(person_sources.get("socsim_agent", pd.DataFrame()), outcome)
                if np.isfinite(r_prev):
                    r_source = "person_level_binary"
            if not np.isfinite(h_prev):
                h_prev, h_count, h_n = brt_prevalence_from_models(models, "human", outcome)
                if np.isfinite(h_prev):
                    h_source = "model_event_count"
            if not np.isfinite(r_prev):
                r_prev, r_count, r_n = brt_prevalence_from_models(models, run_dataset, outcome)
                if np.isfinite(r_prev):
                    r_source = "model_event_count"
            if not np.isfinite([h_prev, r_prev]).all():
                diag.column(
                    f"{label} BRT prevalence",
                    outcome,
                    "prevalence unavailable in BRT descriptives, BRT-coded person-level sources, and model diagnostics",
                )
            ci_low, ci_high = wilson_ci(r_count, r_n)
            h_ci_low, h_ci_high = wilson_ci(h_count, h_n)
            unstable = unstable_flag(r_count, r_n)
            pr = part2_row(part2, label, outcome)
            if pr is not None:
                b_hum, se_hum = safe_float(pr.get("human_beta")), safe_float(pr.get("human_se"))
                b_run, se_run = safe_float(pr.get("f3_beta_run")), safe_float(pr.get("f3_se_run"))
                diff = safe_float(pr.get("f3_beta_diff"))
                diff_lo, diff_hi = safe_float(pr.get("f3_diff_lo")), safe_float(pr.get("f3_diff_hi"))
                verdict = str(pr.get("f3_verdict"))
                unstable = bool(pr.get("c3_unstable"))
            else:
                b_hum, se_hum = f3_slope(models, "human", outcome)
                b_run, se_run = f3_slope(models, run_dataset, outcome)
                verdict, diff, diff_lo, diff_hi = transfer_verdict(
                    b_run, se_run, b_hum, se_hum, unstable)
                diag.column(f"{label} Part II", outcome,
                            "absent from the CR-13B §11 table; fell back to the run's model table")
            rows.append(
                {
                    "run_label": label,
                    "family": BRT_OUTCOME_FAMILY[outcome],
                    "outcome": outcome,
                    "human_prevalence": h_prev,
                    "run_prevalence": r_prev,
                    "absolute_gap": abs(r_prev - h_prev) if np.isfinite([h_prev, r_prev]).all() else np.nan,
                    "relative_gap": (r_prev - h_prev) / h_prev if np.isfinite([h_prev, r_prev]).all() and h_prev != 0 else np.nan,
                    "human_ci_low": h_ci_low,
                    "human_ci_high": h_ci_high,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "run_events": r_count,
                    "run_n": r_n,
                    "compressed_flag": compressed_flag(h_prev, h_ci_low, h_ci_high, r_prev, ci_low, ci_high),
                    "unstable_flag": unstable,
                    "beta_human_bcrt": b_hum,
                    "beta_run_bcrt_realised": b_run,
                    "beta_diff": diff,
                    "beta_diff_ci_low": diff_lo,
                    "beta_diff_ci_high": diff_hi,
                    "transfer_verdict": verdict,
                    "human_prevalence_source": h_source or "unavailable",
                    "run_prevalence_source": r_source or "unavailable",
                    "brt_n_note": DSV4PRO_BRT_N_NOTE if label == DSV4PRO_LABEL else "",
                    "f3_estimator": (str(pr.get("f3_estimator")) if pr is not None else ""),
                    "f3_ci_type": (str(pr.get("f3_ci_type")) if pr is not None else ""),
                    "f3_flags": (str(pr.get("f3_flags") or "") if pr is not None else ""),
                    "f3_estimable": (bool(pr.get("f3_estimable")) if pr is not None else False),
                    "f3_or_run": (safe_float(pr.get("f3_or_run")) if pr is not None else np.nan),
                    "f3_or_run_lo": (safe_float(pr.get("f3_or_run_lo")) if pr is not None else np.nan),
                    "f3_or_run_hi": (safe_float(pr.get("f3_or_run_hi")) if pr is not None else np.nan),
                    "f2_or": (safe_float(pr.get("f2_or")) if pr is not None else np.nan),
                    "f2_or_lo": (safe_float(pr.get("f2_or_lo")) if pr is not None else np.nan),
                    "f2_or_hi": (safe_float(pr.get("f2_or_hi")) if pr is not None else np.nan),
                    "f1_gap_lo_pp": (safe_float(pr.get("f1_gap_lo_pp")) if pr is not None else np.nan),
                    "f1_gap_hi_pp": (safe_float(pr.get("f1_gap_hi_pp")) if pr is not None else np.nan),
                }
            )
    prevalence_table = pd.DataFrame(rows)
    diag.brt_prevalence_complete = not prevalence_table[["human_prevalence", "run_prevalence"]].isna().any().any() if not prevalence_table.empty else False
    write_table(prevalence_table, dirs["tables"] / "table_brt_distributional_fidelity.csv", diag=diag)
    categorical_df = pd.concat(categorical_frames, ignore_index=True).drop_duplicates() if categorical_frames else pd.DataFrame()
    if prevalence_table.empty:
        empty_figure(
            dirs["figures"] / "fig4_brt_behavioural_compression",
            "BRT behavioural compression",
            "No BRT-coded outputs were available for the selected runs.",
            diag,
        )
    else:
        plot_brt(prevalence_table, dirs["figures"] / "fig4_brt_behavioural_compression", diag)
    if not categorical_df.empty:
        plot_brt_categorical_appendix(categorical_df, dirs["figures"] / "appendix_brt_categorical_distributions", diag)
    else:
        diag.panel("appendix_brt_categorical_distributions", "No categorical BRT distribution rows were available.")
    model_table = build_brt_flagged_table(pd.concat(model_frames, ignore_index=True).drop_duplicates() if model_frames else pd.DataFrame(), prevalence_table)
    write_table(model_table, dirs["tables"] / "table_brt_transfer_models_flagged.csv", diag=diag)
    plot_brt_flagged(model_table, dirs["figures"] / "appendix_brt_transfer_models_flagged", diag)
    return prevalence_table, model_table


def plot_brt(prev: pd.DataFrame, base_path: Path, diag: Diagnostics) -> None:
    y_labels = BRT_PRIMARY_OUTCOMES
    fig, ax_a = plt.subplots(figsize=(11.6, max(6.6, 0.42 * len(y_labels) + 1.6)))
    y = np.arange(len(y_labels))
    human = prev.groupby("outcome")["human_prevalence"].first()
    for yi, outcome in enumerate(y_labels):
        h = safe_float(human.get(outcome))
        if np.isfinite(h):
            ax_a.scatter(h, yi, marker=HUMAN_MARKER_ALONG_X, color=HUMAN_INK, s=HUMAN_MARKER_SIZE ** 2,
                         linewidths=HUMAN_MARKER_EDGE, label="Human reference" if yi == 0 else None, zorder=1.8)
    for i, label in enumerate(TARGET_LABELS):
        sub = prev[prev["run_label"] == label].set_index("outcome")
        offset = (i - 1.5) * 0.10
        xs = [safe_float(sub.loc[o, "run_prevalence"]) if o in sub.index else np.nan for o in y_labels]
        lows = [safe_float(sub.loc[o, "ci_low"]) if o in sub.index else np.nan for o in y_labels]
        highs = [safe_float(sub.loc[o, "ci_high"]) if o in sub.index else np.nan for o in y_labels]
        xerr = np.array(
            [
                [x - lo if np.isfinite([x, lo]).all() else np.nan for x, lo in zip(xs, lows, strict=False)],
                [hi - x if np.isfinite([x, hi]).all() else np.nan for x, hi in zip(xs, highs, strict=False)],
            ],
            dtype=float,
        )
        ax_a.errorbar(xs, y + offset, xerr=xerr, fmt=RUN_MARKERS[label], color=RUN_COLORS[label], label=RUN_SHORT[label], ms=5.2, lw=1)
        for x, yy, outcome in zip(xs, y + offset, y_labels, strict=False):
            compressed = bool(sub.loc[outcome, "compressed_flag"]) if outcome in sub.index else False
            if compressed and np.isfinite(x):
                ax_a.text(min(0.98, x + 0.025), yy, "C", fontsize=8, weight="bold", va="center", color=RUN_COLORS[label])
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([clean_label(o) for o in y_labels], fontsize=10)
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Prevalence", fontsize=11)
    ax_a.set_title("A. Primary BRT outcome prevalence and compression", loc="left", weight="bold", fontsize=12)
    ax_a.legend(frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=5, handletextpad=0.5, columnspacing=1.2)
    ax_a.grid(axis="x", color="#eeeeee")
    ax_a.tick_params(axis="x", labelsize=10)
    fig.suptitle("BRT behavioural compression in primary binary outcomes", fontsize=14, weight="bold", y=0.98)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.87, bottom=0.22)
    save_figure(fig, base_path)
    diag.figure(base_path.with_suffix(".pdf").name)
    diag.figure(base_path.with_suffix(".png").name)


def plot_brt_categorical_appendix(categorical: pd.DataFrame, base_path: Path, diag: Diagnostics) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 7.2), constrained_layout=True)
    work = categorical[categorical["measure"].astype(str).isin(BRT_CATEGORICAL_MEASURES)].copy()
    work["plot_label"] = work["measure"].map(clean_label) + " / " + work["category"].map(clean_category_label)
    labels = work["plot_label"].drop_duplicates().tolist()
    y = np.arange(len(labels))
    for i, run_label in enumerate(["human", *TARGET_LABELS]):
        sub = work[work["run_label"] == run_label]
        vals = [float(sub[sub["plot_label"] == lab]["proportion"].sum()) for lab in labels]
        if run_label == "human":   # D-212(a): a bar beneath the run marks, never a dot
            ax.scatter(vals, y + (i - 2) * 0.08, s=HUMAN_MARKER_SIZE ** 2, color=HUMAN_INK, marker=HUMAN_MARKER_ALONG_X,
                       linewidths=HUMAN_MARKER_EDGE, label="Human", alpha=0.8, zorder=0.9)
            continue
        ax.scatter(vals, y + (i - 2) * 0.08, s=18, color=RUN_COLORS[run_label], marker=RUN_MARKERS[run_label],
                   label=RUN_SHORT[run_label], alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Category proportion")
    ax.set_title("Appendix: BRT categorical distributions", loc="left", weight="bold")
    ax.grid(axis="x", color="#eeeeee")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=5)
    save_figure(fig, base_path)
    diag.figure(base_path.with_suffix(".pdf").name)
    diag.figure(base_path.with_suffix(".png").name)


def build_brt_flagged_table(models: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """§8 C3 / §11: every run x primary outcome whose model is flagged.

    Built from the CR-13B §11 Part II rows (`prev`), because the per-run pipeline's
    `06_brt_primary_models_*.csv` predates the ratified plan and does not carry the §5
    primary outcomes. `models` is accepted for signature compatibility and unused.
    """
    del models
    cols = ["run_label", "outcome", "family", "predictor", "estimator", "ci_type",
            "odds_ratio", "CI95_low", "CI95_high", "outcome_prevalence", "run_events",
            "run_n", "flag_reason", "brt_n_note"]
    if prev.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, object]] = []
    for _, row in prev.iterrows():
        reasons = []
        if bool(row.get("unstable_flag", False)):
            reasons.append("unstable_C3_lt20_events_or_non_events")
        if not bool(row.get("f3_estimable", True)):
            reasons.append("F3_not_estimable_realised_bCRT_constant")
        flags = str(row.get("f3_flags") or "")
        if "separation" in flags:
            reasons.append("C2_separation_or_non_convergence")
        if "Firth" in str(row.get("f3_estimator", "")):
            reasons.append("C2_Firth_penalised_fit")
        if bool(row.get("compressed_flag", False)):
            reasons.append("F1_compressed_ge20pp_non_overlapping_CIs")
        if not reasons:
            continue
        rows.append({
            "run_label": row.get("run_label", ""),
            "outcome": row.get("outcome", ""),
            "family": row.get("family", ""),
            "predictor": "BCRT_Total_realised",
            "estimator": row.get("f3_estimator", ""),
            "ci_type": row.get("f3_ci_type", ""),
            "odds_ratio": safe_float(row.get("f3_or_run")),
            "CI95_low": safe_float(row.get("f3_or_run_lo")),
            "CI95_high": safe_float(row.get("f3_or_run_hi")),
            "outcome_prevalence": safe_float(row.get("run_prevalence")),
            "run_events": safe_float(row.get("run_events")),
            "run_n": safe_float(row.get("run_n")),
            "flag_reason": ";".join(sorted(set(reasons))),
            "brt_n_note": row.get("brt_n_note", ""),
        })
    return pd.DataFrame(rows, columns=cols)


def plot_brt_flagged(table: pd.DataFrame, base_path: Path, diag: Diagnostics) -> None:
    if table.empty:
        empty_figure(
            base_path,
            "Appendix: flagged BRT transfer models",
            "No BRT transfer model rows met the instability flag criteria in available outputs.",
            diag,
        )
        return
    work = table.copy()
    work["plot_label"] = work["run_label"].map(lambda x: display_run_label(x, short=True)) + " / " + work["outcome"].map(clean_label)
    counts = work.groupby("plot_label", dropna=False).size().sort_values()
    fig, ax = plt.subplots(figsize=(9.0, max(3.2, 0.28 * len(counts))), constrained_layout=True)
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color="#d95f02", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=7)
    ax.set_xlabel("Flagged model rows")
    ax.set_title("Appendix: BRT transfer models flagged for compression or instability", weight="bold")
    ax.text(
        0.99,
        0.02,
        "Coefficients are not plotted because flagged rows include separation,\nnear-deterministic prevalence, non-finite estimates, or huge intervals.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    save_figure(fig, base_path)
    diag.figure(base_path.with_suffix(".pdf").name)
    diag.figure(base_path.with_suffix(".png").name)


def resolve_matrix_name(columns: list[str], requested: str, run_label: str, source_file: Path, diag: Diagnostics) -> str | None:
    def key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower().replace("observed", "").replace("realised", "").replace("socsimagent", "").replace("human", ""))

    expanded_aliases = list(ASSOCIATION_ALIASES.get(requested, [requested]))
    expanded_aliases.extend(
        [
            f"{requested}_observed",
            f"{requested}_realised",
            f"{requested}_socsim_agent",
            f"socsim_agent_{requested}",
            requested.lower(),
        ]
    )
    for alias in expanded_aliases:
        if alias in columns:
            if alias != requested:
                diag.alias(run_label, requested, alias, source_file.name)
            return alias
    lower = {c.lower(): c for c in columns}
    for alias in expanded_aliases:
        if alias.lower() in lower:
            actual = lower[alias.lower()]
            if actual != requested:
                diag.alias(run_label, requested, actual, source_file.name)
            return actual
    keyed = {key(str(c)): c for c in columns}
    for alias in expanded_aliases:
        actual = keyed.get(key(str(alias)))
        if actual is not None:
            if actual != requested:
                diag.alias(run_label, requested, str(actual), source_file.name)
            return str(actual)
    return None


def load_corr_diff(candidate: ReportCandidate) -> tuple[pd.DataFrame, Path | None]:
    for dataset in ("socsim_agent", "cogsim_agent", "synthetic"):
        path = candidate.report_dir / "tables" / "by_dataset" / dataset / "01_correlations_diff_vs_human.csv"
        if path.exists():
            return read_csv_optional(path), path
    matches = list((candidate.report_dir / "tables" / "by_dataset").glob("*/01_correlations_diff_vs_human.csv")) if (candidate.report_dir / "tables" / "by_dataset").exists() else []
    if matches:
        return read_csv_optional(matches[0]), matches[0]
    return pd.DataFrame(), None


def build_association_assets(selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    matrices: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    variables_for_plot: list[str] | None = None
    for label, candidate in selected.items():
        df, path = load_corr_diff(candidate)
        if df.empty or path is None:
            diag.file(label, candidate.report_dir / "tables" / "by_dataset" / "<dataset>" / "01_correlations_diff_vs_human.csv", "correlation difference table not found")
            continue
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.rename(columns={df.columns[0]: "variable"}).set_index("variable")
        elif df.columns[0] not in df.index:
            df = df.set_index(df.columns[0])
        all_resolved = {var: resolve_matrix_name(list(df.columns), var, label, path, diag) for var in ASSOCIATION_VARIABLES}
        all_index_resolved = {var: resolve_matrix_name(list(df.index.astype(str)), var, label, path, diag) for var in ASSOCIATION_VARIABLES}
        brt_available = any(all_resolved.get(v) and all_index_resolved.get(v) for v in BRT_PRIMARY_OUTCOMES)
        if variables_for_plot is None:
            variables_for_plot = ASSOCIATION_VARIABLES if brt_available else CORE_ASSOCIATION_VARIABLES
            diag.association_includes_brt = brt_available
            if not brt_available:
                diag.panel(
                    "fig5_association_fidelity_heatmaps",
                    "BRT association cells were unavailable in correlation-difference assets; Figure 5 reports core associations only.",
                )
                diag.limitation("BRT association fidelity is reported via prevalence/compression because BRT variables are absent from correlation-difference tables.")
        active_vars = variables_for_plot
        resolved = {var: all_resolved.get(var) for var in active_vars}
        index_resolved = {var: all_index_resolved.get(var) for var in active_vars}
        mat = pd.DataFrame(np.nan, index=active_vars, columns=active_vars, dtype=float)
        n_available = 0
        n_missing = 0
        vals = []
        for i, row_var in enumerate(active_vars):
            for j, col_var in enumerate(active_vars):
                if i <= j:
                    continue
                r_name = index_resolved.get(row_var)
                c_name = resolved.get(col_var)
                if r_name is None or c_name is None:
                    n_missing += 1
                    continue
                value = safe_float(df.loc[r_name, c_name]) if r_name in df.index and c_name in df.columns else np.nan
                if np.isfinite(value):
                    mat.loc[row_var, col_var] = value
                    vals.append(value)
                    n_available += 1
                else:
                    n_missing += 1
        matrices[label] = mat
        arr = np.array(vals, dtype=float)
        legacy = [mat.loc[a, b] for i, a in enumerate(active_vars) for j, b in enumerate(active_vars)
                  if i > j and a in ASSOCIATION_MAE_LEGACY_VARIABLES and b in ASSOCIATION_MAE_LEGACY_VARIABLES]
        legacy_arr = np.array([v for v in legacy if np.isfinite(v)], dtype=float)
        rows.append(
            {
                "run_label": label,
                    "metric_scope": "BRT+core associations" if variables_for_plot == ASSOCIATION_VARIABLES else "core associations only",
                "mean_absolute_correlation_error_6var_legacy": float(np.mean(np.abs(legacy_arr))) if len(legacy_arr) else np.nan,
                "mean_absolute_correlation_error": float(np.nanmean(np.abs(arr))) if len(arr) else np.nan,
                "frobenius_norm": float(np.sqrt(np.nansum(arr * arr))) if len(arr) else np.nan,
                "n_pairs_available": n_available,
                "n_pairs_missing": n_missing,
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, dirs["tables"] / "table_association_fidelity_metrics.csv", diag=diag)
    if not matrices:
        empty_figure(
            dirs["figures"] / "fig5_association_fidelity_heatmaps",
            "Association fidelity heatmaps",
            "No correlation difference tables were available for the selected runs.",
            diag,
        )
        return table
    active_vars = variables_for_plot or CORE_ASSOCIATION_VARIABLES
    ncols = 4
    nrows = int(math.ceil((len(TARGET_LABELS) + 1) / ncols))   # +1 slot for the colourbar
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.9 * nrows), squeeze=False, constrained_layout=True)
    vmax = max([np.nanmax(np.abs(m.values)) for m in matrices.values() if np.isfinite(m.values).any()] + [0.1])
    vmax = min(max(vmax, 0.1), 1.0)
    im = None
    flat = list(axes.flat)
    for ax, label in zip(flat, TARGET_LABELS, strict=False):
        mat = matrices.get(label, pd.DataFrame(np.nan, index=active_vars, columns=active_vars))
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(RUN_SHORT[label], fontsize=14, weight="bold")
        ax.set_xticks(range(len(active_vars)))
        ax.set_yticks(range(len(active_vars)))
        ax.set_xticklabels([assoc_short(v) for v in active_vars], rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels([assoc_short(v) for v in active_vars], fontsize=11)
        for i in range(len(active_vars)):
            for j in range(len(active_vars)):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if abs(v) > 0.6 * vmax else "#111111")
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in flat[len(TARGET_LABELS):]:
        ax.axis("off")
    if im is not None:
        cax = flat[len(TARGET_LABELS)]
        cb = fig.colorbar(im, ax=cax, fraction=0.5, shrink=0.9, aspect=18)
        cb.set_label("Correlation difference (r_sim - r_human)", fontsize=12)
        cb.ax.tick_params(labelsize=11)
    scope = "core associations" if active_vars == CORE_ASSOCIATION_VARIABLES else "core and BRT associations"
    title = ("Association fidelity: CRT2, bCRT, NFC and all five BFI traits" if active_vars == CORE_ASSOCIATION_VARIABLES
             else f"Association fidelity heatmaps ({scope})")
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.text(0.01, 0.005, "Lower triangle shown; each cell is r_sim - r_human for that pair. Blank cells in a bCRT row/column mean the "
             "correlation is undefined because realised bCRT is constant (4/4) in that run.", fontsize=9, color="#333333", va="bottom")
    save_figure(fig, dirs["figures"] / "fig5_association_fidelity_heatmaps")
    diag.figure("fig5_association_fidelity_heatmaps.pdf")
    diag.figure("fig5_association_fidelity_heatmaps.png")
    return table


def assoc_short(name: str) -> str:
    return clean_label(name).replace("Conscientiousness", "Consc.").replace("Neuroticism", "Neuro.")


def build_calibration_assets(selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    heat = pd.DataFrame(index=list(CALIBRATION_CONSTRUCTS), columns=TARGET_LABELS, dtype=float)
    r2_text = pd.DataFrame("", index=list(CALIBRATION_CONSTRUCTS), columns=TARGET_LABELS)
    r2_heat = pd.DataFrame(index=list(CALIBRATION_CONSTRUCTS), columns=TARGET_LABELS, dtype=float)
    for label, candidate in selected.items():
        paths = sorted((candidate.report_dir / "tables" / "calibration").glob("calibration_summary_*.csv")) if (candidate.report_dir / "tables" / "calibration").exists() else []
        chosen = None
        for path in paths:
            if "socsim_agent" in path.name:
                chosen = path
                break
        chosen = chosen or (paths[0] if paths else None)
        if chosen is None:
            diag.file(label, candidate.report_dir / "tables" / "calibration" / "calibration_summary_<dataset>.csv", "calibration summary not found")
            continue
        df = read_csv_optional(chosen)
        if df.empty or "construct" not in df.columns:
            diag.column(f"{label} calibration", "construct", "calibration summary schema not recognised")
            continue
        for display, aliases in CALIBRATION_CONSTRUCTS.items():
            sub = df[df["construct"].astype(str).isin(aliases)]
            if sub.empty:
                diag.column(f"{label} calibration", display, "construct row missing")
                continue
            row = sub.iloc[0]
            rmse = safe_float(row.get("RMSE") if "RMSE" in row else row.get("rmse"))
            r2 = safe_float(row.get("R2") if "R2" in row else row.get("r2"))
            heat.loc[display, label] = rmse
            r2_heat.loc[display, label] = r2
            r2_text.loc[display, label] = f"{r2:.2f}" if np.isfinite(r2) else ""
            rows.append(
                {
                    "run_label": label,
                    "construct": display,
                    "rmse": rmse,
                    "mae": safe_float(row.get("MAE") if "MAE" in row else row.get("mae")),
                    "r2": r2,
                    "slope": safe_float(row.get("slope")),
                    "intercept": safe_float(row.get("intercept")),
                    "n": safe_float(row.get("N") if "N" in row else row.get("n")),
                }
            )
    table = pd.DataFrame(rows)
    write_table(table, dirs["tables"] / "table_calibration_fidelity.csv", diag=diag)
    if table.empty:
        empty_figure(
            dirs["figures"] / "fig6_calibration_fingerprint",
            "Calibration fingerprint",
            "No calibration summary tables were available.",
            diag,
        )
        return table
    shown = [c for c in heat.index if c not in UNSPECIFIED_CALIBRATION_ROWS]   # S-8e (D-293): display only
    plot_calibration_fingerprint(r2_heat.loc[shown], heat.loc[shown], dirs["figures"] / "fig6_calibration_fingerprint")
    diag.figure("fig6_calibration_fingerprint.pdf")
    diag.figure("fig6_calibration_fingerprint.png")
    plot_relative_rmse_appendix(heat, dirs["figures"] / "appendix_calibration_relative_rmse", diag)
    return table


def plot_calibration_fingerprint(r2_heat: pd.DataFrame, heat: pd.DataFrame, base_path: Path) -> None:
    """S-8d (PI 29 Oct): R-squared and RMSE per construct and run at print width (5.5 in); run names in words, not codes;
    cells that cannot be estimated hatched and labelled n/e (VOICE, colour and labels); no internal title."""
    from matplotlib.patches import Rectangle
    names = [CONDITIONS[l]["name"].replace("Claude Sonnet 4.6 · ", "Claude Sonnet 4.6\n") for l in TARGET_LABELS]
    with plt.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.9), sharey=True)
        for ax, frame, title, cmap, vmin, vmax, ticks in (
            (axes[0], r2_heat, "A  Calibration $R^2$ (higher is closer)", "YlGnBu", 0, 1, [0, 0.5, 1]),
            (axes[1], heat, "B  Calibration RMSE (lower is closer)", "YlOrRd", 0, None, [0, 0.5, 1, 1.5]),
        ):
            arr = frame.to_numpy(dtype=float)
            im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if np.isfinite(arr[i, j]):
                        rgba = im.cmap(im.norm(arr[i, j]))
                        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                        ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if lum < 0.45 else "#111111")
                    else:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="#9a9a9a",
                                               hatch="////", lw=0))
                        ax.text(j, i, "n/e", ha="center", va="center", fontsize=6, color="#333333",
                                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=6.3, linespacing=1.0)
            ax.set_yticks(range(len(frame.index)))
            ax.set_yticklabels(frame.index, fontsize=7)
            ax.tick_params(length=0, pad=2)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.3, aspect=30, ticks=ticks)
            cb.ax.tick_params(labelsize=6, length=2, pad=1)
            cb.outline.set_visible(False)
            ax.set_title(title, loc="left", fontsize=7.5, weight="bold")
        fig.subplots_adjust(left=0.2, right=0.98, top=0.94, bottom=0.06, wspace=0.08)
        save_figure(fig, base_path)


def plot_relative_rmse_appendix(heat: pd.DataFrame, base_path: Path, diag: Diagnostics) -> None:
    arr = heat.to_numpy(dtype=float)
    scaled = arr.copy()
    for i in range(scaled.shape[0]):
        row = scaled[i, :]
        finite = np.isfinite(row)
        if finite.sum() > 1:
            mn, mx = np.nanmin(row), np.nanmax(row)
            scaled[i, finite] = (row[finite] - mn) / (mx - mn) if mx > mn else 0.5
        elif finite.sum() == 1:
            scaled[i, finite] = 0.5
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    im = ax.imshow(scaled, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(TARGET_LABELS)))
    ax.set_xticklabels([RUN_SHORT[l] for l in TARGET_LABELS], fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=10)
    # CR-14 (I11): ONE scale. Colour and printed value are both the row-normalised RMSE
    # (0 = lowest RMSE in that construct's row, 1 = highest); the submitted figure coloured by this
    # scale but printed raw RMSE (1.85-1.99 etc.), which cannot be read against a 0-1 colourbar.
    # Raw RMSE is in table_calibration_fidelity.csv and appendix_calibration_relative_rmse_data.csv.
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            v = scaled[i, j]
            text = "" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="white" if np.isfinite(v) and v > 0.6 else "#111111")
    fig.colorbar(im, ax=ax, label="Row-normalised RMSE: (RMSE - row min) / (row max - row min)")
    ax.set_title("Appendix: row-normalised calibration RMSE (cell values on the same 0-1 scale)", weight="bold", fontsize=11)
    long = []
    for i, construct in enumerate(heat.index):
        for j, label in enumerate(TARGET_LABELS):
            long.append({"construct": construct, "run_label": label, "condition": condition_name(label),
                         "rmse_raw": arr[i, j], "rmse_row_normalised": scaled[i, j]})
    pd.DataFrame(long).to_csv(base_path.parent.parent / "tables" / "appendix_calibration_relative_rmse_data.csv", index=False)
    save_figure(fig, base_path)
    diag.figure(base_path.with_suffix(".pdf").name)
    diag.figure(base_path.with_suffix(".png").name)


def build_experimental_runs_table(selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    """Table 3. The submitted table printed dataframe field names, `selected_report_id` strings and a
    model column that defaulted to Sonnet 4.6 for every run; persona source, format and model now come
    from the CONDITIONS registry (quoted from the run config disclosures)."""
    rows = []
    for label in TARGET_LABELS:
        candidate = selected.get(label)
        meta_c = CONDITIONS[label]
        if candidate is None:
            rows.append({"run_label": label, "warnings": diag.missing_labels.get(label, "missing selected report")})
            continue
        meta = read_json_optional(candidate.report_dir / "run_metadata.json")
        warning_parts = []
        if not meta:
            warning_parts.append("run_metadata.json missing or unreadable")
        files = {p.name.lower() for p in candidate.report_dir.rglob("*") if p.is_file()}
        n_crt = infer_n_from_descriptives(candidate)
        n_brt = 467 if label == DSV4PRO_LABEL else n_crt
        rows.append(
            {
                "run_label": label,
                "condition": meta_c["name"],
                "persona_source": meta_c["persona_source"],
                "spec_format": meta_c["spec"],
                "model": meta_c["model"],
                "model_config": meta_c["config"],
                "n_crt": n_crt,
                "n_brt": n_brt,
                "conversation_mode": meta.get("conversation_mode") or "continuous per-persona multi-turn, one API call per block",
                "brt_present": any("brt" in name for name in files),
                "liwc_present": any("liwc" in name for name in files),
                "simulator_run_id": label,
                "analysis_report_id": candidate.report_id,
                "warnings": "; ".join(warning_parts),
            }
        )
    table = pd.DataFrame(rows)
    out_csv = clean_for_output(table).assign(condition=table["condition"])
    out_csv.to_csv(dirs["tables"] / "table_experimental_runs.csv", index=False)
    diag.table("table_experimental_runs.csv")
    body = []
    for i, r in enumerate(rows):
        mark = ""   # S-8 (C-13): identifiers live in the run-identifier table, not in footnotes
        n_brt = int(r["n_brt"]) if np.isfinite(safe_float(r.get("n_brt"))) else None
        n_crt = int(r["n_crt"]) if np.isfinite(safe_float(r.get("n_crt"))) else None
        n_cell = f"{n_crt}" if n_brt == n_crt else f"{n_crt} / {n_brt}\\textsuperscript{{*}}"
        body.append([tex_escape(r["condition"]) + mark, tex_escape(PERSONA_SOURCE_DISPLAY.get(r["persona_source"], r["persona_source"])),
                     tex_escape(r["spec_format"]),
                     tex_escape(r["model"]), tex_escape(r["model_config"]), n_cell])
    notes = [   # S-8 (C-14): one paragraph
        "One model call per block (BRT, reasoning, Need for Cognition, BFI), with conversation history retained within each "
        "persona's session; CRT2, bCRT and decoy items in randomised order; numeric = explicit scale values, lexical = graded "
        "verbal descriptors; seed 42. N = simulated personas (CRT / BRT where they differ). \\textsuperscript{*}"
        + tex_escape(DSV4PRO_BRT_N_TEX)
        + " Run identifiers: Table~\\ref{tab:run-identifiers}.",
    ]
    tex = tex_booktabs(
        ["Condition", "Persona source", "Format", "Model", "Configuration", "N"],
        body, "@{}" + tex_x(1.15) + tex_x(0.90) + "l" + tex_x(1.00) + tex_x(0.95) + "r@{}",
        "Experimental conditions: one row per full-battery simulation run.",
        "tab:experimental-runs", notes, font=r"\footnotesize",
    )
    (dirs["tables"] / "table_experimental_runs.tex").write_text(tex, encoding="utf-8")
    diag.table("table_experimental_runs.tex")
    write_run_identifier_table(dirs, diag)
    return table


def write_run_identifier_table(dirs: dict[str, Path], diag: Diagnostics) -> None:
    """S-8 (C-13, PI 2026-10-13): one appendix table maps each readable condition name to its identifier in the
    released code and data; it replaces the per-table archival footnotes promised to reviewers (P22)."""
    rows = [{"condition": condition_name(l), "run_identifier": l, "analysis_report": f"cr13_{l}"} for l in TARGET_LABELS]
    write_table(pd.DataFrame(rows), dirs["tables"] / "table_run_identifiers.csv", diag=diag)
    body = [[tex_escape(r["condition"]), tex_id(r["run_identifier"])] for r in rows]
    tex = tex_booktabs(["Condition", "Run identifier"], body, "@{}" + tex_x(1.0) + "l@{}",
                       "Run identifiers in the released code and data.", "tab:run-identifiers",
                       ["Each analysis report is named \\texttt{cr13\\_} followed by the run identifier."],
                       font=r"\footnotesize")
    (dirs["tables"] / "table_run_identifiers.tex").write_text(tex, encoding="utf-8")
    diag.table("table_run_identifiers.tex")


def infer_persona_source(label: str) -> str:
    return "human_reference_profiles" if label.startswith("Human") else "statistically_sampled_synthetic_profiles"


def infer_prompt_mode(label: str) -> str:
    return "numeric" if "Num" in label else "lexical" if "Lex" in label else ""


def infer_n_from_descriptives(candidate: ReportCandidate) -> float:
    df = read_csv_optional(candidate.report_dir / "tables" / "00_descriptives_overview.csv")
    if df.empty:
        return np.nan
    run_dataset = choose_run_dataset(df, candidate.label)
    rows = df[df["dataset"].astype(str) == str(run_dataset)] if run_dataset else df[df["dataset"].astype(str) != "human"]
    if rows.empty:
        return np.nan
    return sample_size_from_row(rows.iloc[0])


def build_scorecard_table(
    distribution: pd.DataFrame,
    crt: pd.DataFrame,
    calibration: pd.DataFrame,
    association: pd.DataFrame,
    brt: pd.DataFrame,
    brt_flagged: pd.DataFrame,
    dirs: dict[str, Path],
    diag: Diagnostics,
) -> pd.DataFrame:
    adj = pd.read_csv(repo_root() / CR13B_CRT / "crt2ref_adjusted_pooled.csv")
    adj_lure = float(adj.loc[adj["response_category"] == "intuitive_incorrect", "proportion"].iloc[0])
    rows = []
    for label in TARGET_LABELS:
        dist = distribution[distribution["run_label"] == label] if not distribution.empty else pd.DataFrame()
        crt_rows = crt[crt["run_label"] == label] if not crt.empty else pd.DataFrame()
        cal = calibration[calibration["run_label"] == label] if not calibration.empty else pd.DataFrame()
        assoc = association[association["run_label"] == label] if not association.empty else pd.DataFrame()
        brt_rows = brt[brt["run_label"] == label] if not brt.empty else pd.DataFrame()
        flagged = brt_flagged[brt_flagged["run_label"] == label] if not brt_flagged.empty else pd.DataFrame()
        bfi_js = dist[dist["construct"].astype(str).str.startswith("BFI_")]["js_divergence"].mean() if not dist.empty else np.nan
        brt_unstable = int(brt_rows["unstable_flag"].sum()) if not brt_rows.empty and "unstable_flag" in brt_rows else np.nan
        verdicts = brt_rows["transfer_verdict"].astype(str) if not brt_rows.empty and "transfer_verdict" in brt_rows else pd.Series(dtype=str)
        brt_incomplete = not brt_rows.empty and brt_rows["run_prevalence"].isna().any()
        rows.append(
            {
                "run_label": label,
                "CRT2_JS": value_for(dist, "construct", "CRT2_Total", "js_divergence"),
                "BCRT_JS": value_for(dist, "construct", "BCRT_Total", "js_divergence"),
                "NFC_JS": value_for(dist, "construct", "NFC_Total", "js_divergence"),
                "BFI_avg_JS": bfi_js,
                "CRT2_intuitive_gap": crt2_gap_vs_adjusted(crt_rows, adj_lure),   # D-242: familiarity-adjusted ref.
                "BCRT_intuitive_gap": intuitive_gap(crt_rows, "bCRT"),
                "CRT2_unscored_run_n": unscored_n(crt_rows, "CRT2", "n_run_unscored"),
                "BCRT_unscored_run_n": unscored_n(crt_rows, "bCRT", "n_run_unscored"),
                "CRT2_unscored_human_n": unscored_n(crt_rows, "CRT2", "n_human_unscored"),
                "BCRT_unscored_human_n": unscored_n(crt_rows, "bCRT", "n_human_unscored"),
                "calibration_RMSE_core": (cal.loc[~cal["construct"].isin(UNSPECIFIED_CALIBRATION_ROWS), "rmse"].mean()
                                          if not cal.empty else np.nan),   # S-8e (D-293): seven specified constructs
                "association_MAE": safe_float(assoc.iloc[0]["mean_absolute_correlation_error_6var_legacy"]) if not assoc.empty else np.nan,
                "BRT_prevalence_MAE": "incomplete" if brt_incomplete else brt_rows["absolute_gap"].mean() if not brt_rows.empty else np.nan,
                "BRT_compressed_outcomes_n": int(brt_rows["compressed_flag"].sum()) if not brt_rows.empty and "compressed_flag" in brt_rows else 0,
                "BRT_unstable_models_n": int(brt_unstable) if np.isfinite(brt_unstable) else np.nan,
                "BRT_transfers_n": int((verdicts == "transfers").sum()),
                "BRT_fails_to_transfer_n": int((verdicts == "fails to transfer").sum()),
                "BRT_indeterminate_n": int((verdicts == "indeterminate").sum()),
                "BRT_transfer_not_estimable": bool(len(verdicts)) and bool(verdicts.str.startswith("not estimable").all()),
            }
        )
    table = pd.DataFrame(rows)
    for label, want in SCORECARD_CRT2_GAP_D242.items():   # guard: the values the sweep minted into R-T1
        got = float(table.loc[table["run_label"] == label, "CRT2_intuitive_gap"].iloc[0])
        if abs(got - want) > 5e-7:
            raise SystemExit(f"STOP: scorecard CRT2 gap {label} {got:.6f} != D-242(b) {want:.6f}.")
    # the n/e flag is for printing only; the CSV keeps its columns (S-8e change audit)
    write_table(table.drop(columns=["BRT_transfer_not_estimable"]), dirs["tables"] / "table_fidelity_scorecard.csv", diag=diag)
    write_scorecard_booktabs(table, dirs, diag)
    return table


def transfer_tally(r: pd.Series) -> str:
    """S-8e (D-291(c)): "n/e" where no transfer verdict is estimable (realised bCRT constant), not "0 / 0 / 0"."""
    t, f, i = int(r["BRT_transfers_n"]), int(r["BRT_fails_to_transfer_n"]), int(r["BRT_indeterminate_n"])
    return "n/e" if r.get("BRT_transfer_not_estimable", False) else f"{t} / {f} / {i}"


def write_scorecard_booktabs(table: pd.DataFrame, dirs: dict[str, Path], diag: Diagnostics) -> None:
    """S-8 (C-15, PI 2026-10-13): ONE scorecard of the run-level summaries no figure prints as numbers. The JS columns
    left (they are the cells of the main-text distribution figure); the lure gaps stay (promised to reviewers, P5).
    One note paragraph (C-14); the compressed / unstable / transfer rules are defined in Methods."""
    body = []
    for _, r in table.iterrows():
        brt_mae = r["BRT_prevalence_MAE"]
        body.append([tex_escape(condition_name(r["run_label"])),
                     fmt_num(r["CRT2_intuitive_gap"]), fmt_num(r["BCRT_intuitive_gap"]),
                     fmt_num(r["calibration_RMSE_core"]), fmt_num(r["association_MAE"]),
                     tex_escape(brt_mae) if isinstance(brt_mae, str) else fmt_num(brt_mae),
                     str(int(r["BRT_compressed_outcomes_n"])),
                     str(int(r["BRT_unstable_models_n"])) if np.isfinite(safe_float(r["BRT_unstable_models_n"])) else "--",
                     transfer_tally(r)])
    run_unscored = sum(int(np.nansum(table[f"{t}_unscored_run_n"].astype(float))) for t in ("CRT2", "BCRT"))
    hum_unscored = sum(int(np.nanmax(table[f"{t}_unscored_human_n"].astype(float))) for t in ("CRT2", "BCRT"))
    # PI 16 Oct: say nothing when every response was scored (an unattacked point); the human frame belongs to Methods.
    unscored = ("" if run_unscored == 0 and hum_unscored == 0
                else f"; {run_unscored} agent and {hum_unscored} human responses were unscored and excluded")
    note = (
        "Gap = run minus human intuitive-lure prevalence (negative: fewer lures than people; "
        f"human reference: the familiarity-adjusted rate for CRT2 and the rate as answered for bCRT{unscored}). "
        "Calib.\\ RMSE = mean over the seven specified constructs. Assoc.\\ MAE = mean absolute difference in pairwise correlations over CRT2, "
        "bCRT, NFC, Conscientiousness, Openness and Neuroticism (15 pairs; 6 for the Claude Sonnet 4.6 conditions, whose "
        "CRT2 and bCRT pairs are undefined). BRT MAE = mean absolute prevalence gap over the 15 coded BRT "
        "behaviours; compressed, unstable and the transfer verdicts (transfers / fails / indeterminate) count those behaviours "
        "as defined in Section~\\ref{subsec:statistical_analyses}; n/e: not estimable. "
        + tex_escape(DSV4PRO_BRT_N_TEX)
    )
    hdr = ["Condition", tex_head("CRT2", "gap"), tex_head("bCRT", "gap"), tex_head("Calib.", "RMSE"),
           tex_head("Assoc.", "MAE"), tex_head("BRT", "MAE"), tex_head("Com-", "pressed"), tex_head("Un-", "stable"),
           tex_head("Transfer /", "fail / indet.")]
    tex = tex_booktabs(hdr, body, "@{}" + tex_x(1.0) + "rrrrrrrc@{}",
                       "Fidelity scorecard: run-level summaries by condition.", "tab:scorecard", [note],
                       font=r"\footnotesize")
    (dirs["tables"] / "table_fidelity_scorecard.tex").write_text(tex, encoding="utf-8")
    diag.table("table_fidelity_scorecard.tex")


def value_for(df: pd.DataFrame, key_col: str, key: str, value_col: str) -> float:
    if df.empty or key_col not in df.columns or value_col not in df.columns:
        return np.nan
    sub = df[df[key_col].astype(str) == key]
    return safe_float(sub.iloc[0][value_col]) if not sub.empty else np.nan


def unscored_n(df: pd.DataFrame, test: str, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    sub = df[df["test"] == test]
    return safe_float(sub.iloc[0][col]) if not sub.empty else np.nan


def unscored_note(table: pd.DataFrame) -> str:
    """CR-23 ruling (d): every table that prints a rate says how many responses went unscored."""
    run = {t: int(np.nansum(table[f"{t}_unscored_run_n"].astype(float))) for t in ("CRT2", "BCRT")}
    hum = {t: int(np.nanmax(table[f"{t}_unscored_human_n"].astype(float))) for t in ("CRT2", "BCRT")}
    return (f"Unscored responses excluded from the gap denominators: CRT2 {run['CRT2']} and bCRT {run['BCRT']} "
            f"across all seven conditions; human reference CRT2 {hum['CRT2']}, bCRT {hum['BCRT']}.")


# D-242(b): the seven CRT2 lure gaps against the familiarity-adjusted reference (0.33338), as minted into R-T1.
SCORECARD_CRT2_GAP_D242 = {
    "CR_MAIN_SYNTH_NUM_SONNET5_S42": -0.000582, "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": -0.150263,
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": -0.229881, "CR_BASELINE_HUMAN_NUMERIC_CS46_S42": -0.333385,
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": -0.333385, "CR_MAIN_SYNTH_NUMERIC_CS46_S42": -0.333385,
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42": -0.333385}


def crt2_gap_vs_adjusted(df: pd.DataFrame, adj_lure: float) -> float:
    """D-242 / D-74: CRT2 lure gap = the run's lure share over scored responses minus the familiarity-adjusted human
    pooled lure share (crt2ref_adjusted_pooled.csv). bCRT keeps intuitive_gap (people as answered)."""
    if df.empty:
        return np.nan
    sub = df[(df["test"] == "CRT2") & (df["category"] == "intuitive_incorrect")]
    return safe_float(sub.iloc[0]["run_prop"]) - adj_lure if not sub.empty else np.nan


def intuitive_gap(df: pd.DataFrame, test: str) -> float:
    if df.empty:
        return np.nan
    sub = df[(df["test"] == test) & (df["category"] == "intuitive_incorrect")]
    return safe_float(sub.iloc[0]["run_prop"]) - safe_float(sub.iloc[0]["human_prop"]) if not sub.empty else np.nan


def build_prompt_screening_table(root: Path, dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    search_roots = [root / "reports", legacy_bundle_root(root) / "source_existing"]
    csvs: list[Path] = []
    for base in search_roots:
        if base.exists():
            csvs.extend(p for p in base.rglob("*.csv") if "prompt" in p.name.lower() and ("screen" in p.name.lower() or "score" in p.name.lower() or "comparison" in p.name.lower()))
    columns = [
        "prompt_mode",
        "marginal_good",
        "marginal_moderate",
        "marginal_poor",
        "association_good",
        "association_moderate",
        "association_poor",
        "overall_good",
        "overall_moderate",
        "overall_poor",
    ]
    if not csvs:
        diag.panel("table_prompt_screening_summary", "No previous prompt-screening comparison outputs were found.")
        table = pd.DataFrame(columns=columns)
    else:
        counts: dict[str, dict[str, int]] = {}
        for path in csvs:
            df = read_csv_optional(path)
            if df.empty:
                continue
            prompt_col = first_existing(df, ["prompt_mode", "mode", "format"])
            judgement_col = first_existing(df, ["judgement", "label", "rating"])
            eval_col = first_existing(df, ["evaluation_type", "type", "metric_group"])
            if prompt_col is None or judgement_col is None:
                continue
            for _, row in df.iterrows():
                mode = normalise_prompt_mode(str(row[prompt_col]))
                judgement = str(row[judgement_col]).upper()
                if judgement not in {"GOOD", "MODERATE", "POOR"}:
                    continue
                evaluation = str(row[eval_col]).lower() if eval_col else "overall"
                bucket = "association" if "assoc" in evaluation or "relation" in evaluation else "marginal" if "marginal" in evaluation or "distribution" in evaluation else "overall"
                counts.setdefault(mode, {c: 0 for c in columns if c != "prompt_mode"})
                counts[mode][f"{bucket}_{judgement.lower()}"] += 1
                counts[mode][f"overall_{judgement.lower()}"] += 1
        table = pd.DataFrame([{"prompt_mode": mode, **vals} for mode, vals in sorted(counts.items())], columns=columns)
        if table.empty:
            diag.panel("table_prompt_screening_summary", "Prompt-screening CSVs were found, but none had usable prompt mode and GOOD/MODERATE/POOR judgement columns.")
    if table.empty:
        for path in (
            dirs["tables"] / "table_prompt_screening_summary.csv",
            dirs["tables"] / "table_prompt_screening_summary.tex",
        ):
            if path.exists():
                path.unlink()
        diag.prompt_screening_available = False
        (dirs["diagnostics"] / "prompt_screening_unavailable.md").write_text(
            "# Prompt Screening Summary Unavailable\n\n"
            "No usable prompt-screening comparison output with prompt mode and GOOD/MODERATE/POOR judgement columns was found. "
            "The final asset bundle therefore omits `table_prompt_screening_summary.csv` and `.tex` rather than emitting an empty table.\n",
            encoding="utf-8",
        )
        return table
    note = (
        "These labels were useful for pilot prompt selection but are unweighted across constructs and evaluation types, "
        "so they should not be treated as the final criterion for full-run behavioural fidelity."
    )
    diag.prompt_screening_available = True
    write_table(table, dirs["tables"] / "table_prompt_screening_summary.csv", tex=True, note=note, diag=diag)
    return table


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def normalise_prompt_mode(value: str) -> str:
    v = value.lower()
    if "both" in v or ("num" in v and "lex" in v):
        return "both"
    if "num" in v:
        return "numeric"
    if "lex" in v:
        return "lexical"
    return value.strip() or "unknown"


# CR-24 (ruling: CR23REVIEW_CANDIDATES C-2): when an agent's predictor (CRT2 or bCRT score) takes one
# value for every persona in a run, the agent-vs-human slope on it is not identified -- the interaction
# model is rank-deficient and still prints numbers (e.g. slope 153.8, p ~ 1e-223). Those rows print this
# phrase instead. The rule is generic: any run, any predictor, decided from that run's agent person frame.
NOT_ESTIMABLE_CONSTANT = "not estimable (agent score constant)"
AGENT_SLOPE_COLUMNS = ("slope_sim", "slope_difference", "interaction_beta", "ci_low", "ci_high", "interaction_p")
AGENT_DATASET = "socsim_agent"


def agent_predictor_is_constant(agent_frame: pd.DataFrame, predictor: str) -> bool:
    """True when the agent's realised `predictor` score (the column the slope test uses for agents) has
    zero variance across the run's personas, i.e. fewer than two distinct non-missing values."""
    col = _resolve_cognitive_predictor_column(agent_frame, AGENT_DATASET, predictor)
    if not col:
        return False  # no predictor column -> the slope test produced no row for it either
    return pd.to_numeric(agent_frame[col], errors="coerce").dropna().nunique() < 2


def mark_agent_constant_rows(work: pd.DataFrame, constant_predictors: Iterable[str]) -> pd.DataFrame:
    """Replace the agent-dependent columns of every row whose predictor is constant for the agent with
    NOT_ESTIMABLE_CONSTANT. The human slope (and anything not listed) is left as computed; a frame with
    no such row is returned unchanged, dtypes included."""
    out = work.copy()
    mask = out["predictor"].isin(set(constant_predictors))
    if mask.any():
        for col in AGENT_SLOPE_COLUMNS:
            out[col] = out[col].astype(object)
            out.loc[mask, col] = NOT_ESTIMABLE_CONSTANT
    return out


def load_agent_person_frame(root: Path, candidate: ReportCandidate) -> pd.DataFrame:
    for base in source_roots_for_candidate(root, candidate):
        path = base / "tables" / "by_dataset" / AGENT_DATASET / "00_person_frame.csv"
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    raise SystemExit(
        f"STOP: agent person frame missing for {candidate.label} "
        f"({candidate.report_dir / 'tables' / 'by_dataset' / AGENT_DATASET / '00_person_frame.csv'}); "
        "cannot decide which Table 5 rows have a constant agent score (CR-24).")


def load_liwc_fidelity_rows(root: Path, selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for label, candidate in selected.items():
        source = None
        rmse = pd.DataFrame()
        for base in source_roots_for_candidate(root, candidate):
            path = base / "tables" / "simulation_fidelity" / "brt_liwc_slope_difference_tests.csv"
            rmse_path = base / "tables" / "simulation_fidelity" / "brt_liwc_binned_rmse.csv"
            if path.exists():
                source = read_csv_optional(path)
                rmse = read_csv_optional(rmse_path) if rmse_path.exists() else pd.DataFrame()
                break
        if source is None or source.empty:
            diag.file(label, candidate.report_dir / "tables" / "simulation_fidelity" / "brt_liwc_slope_difference_tests.csv", "LIWC/BRT slope fidelity table not found")
            continue
        work = source[source["predictor"].isin(LIWC_PREDICTORS) & source["outcome"].isin(LIWC_THEORY_OUTCOMES)].copy()
        available = set(work["outcome"].astype(str))
        for outcome in LIWC_THEORY_OUTCOMES:
            if outcome not in available:
                diag.column(f"{label} LIWC theory set", outcome, "theory-relevant LIWC outcome unavailable in slope fidelity table")
        if not rmse.empty:
            rmse = rmse.rename(columns={"outcome": "liwc_outcome", "rmse_bin": "binned_rmse_if_available"})
            work = work.rename(columns={"outcome": "liwc_outcome"}).merge(
                rmse[["predictor", "liwc_outcome", "binned_rmse_if_available"]],
                on=["predictor", "liwc_outcome"],
                how="left",
            )
        else:
            work = work.rename(columns={"outcome": "liwc_outcome"})
            work["binned_rmse_if_available"] = np.nan
        work["run_label"] = label
        work["slope_difference"] = work["slope_sim"] - work["slope_human"]
        agent_frame = load_agent_person_frame(root, candidate)
        constant = [p for p in work["predictor"].unique() if agent_predictor_is_constant(agent_frame, p)]
        if constant:
            diag.note(f"CR-24: `{label}`: agent {', '.join(constant)} constant across personas; "
                      f"{int(work['predictor'].isin(constant).sum())} LIWC slope rows printed as '{NOT_ESTIMABLE_CONSTANT}'.")
        work = mark_agent_constant_rows(work, constant)
        rows.append(work)
    columns = [
        "run_label",
        "predictor",
        "liwc_outcome",
        "slope_human",
        "slope_sim",
        "slope_difference",
        "interaction_beta",
        "interaction_p",
        "ci_low",
        "ci_high",
        "binned_rmse_if_available",
    ]
    table = pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)
    write_table(table, dirs["tables"] / "table_liwc_theory_set_slope_fidelity.csv", diag=diag)
    write_liwc_booktabs_by_run(table, dirs, diag)
    return table


def write_liwc_booktabs_by_run(table: pd.DataFrame, dirs: dict[str, Path], diag: Diagnostics) -> None:
    """Table 5 split by run: one booktabs table per condition plus an \\input wrapper. `interaction_beta`
    is identical to the slope difference in every row, so it is not printed as a second column."""
    hdr = [tex_head("Pre-", "dictor"), "LIWC outcome", tex_head("Human", "slope"), tex_head("Agent", "slope"),
           tex_head("Differ-", "ence"), "95\\% CI", "$p$", tex_head("Binned", "RMSE")]
    inputs = []
    for i, label in enumerate(TARGET_LABELS):
        sub = table[table["run_label"] == label]
        if sub.empty:
            continue
        def liwc_row(r: pd.Series) -> list[str]:
            head = [tex_escape(clean_label(r["predictor"])), tex_escape(clean_label(r["liwc_outcome"])), fmt_num(r["slope_human"])]
            rmse = fmt_num(r["binned_rmse_if_available"], 2)
            if r["slope_sim"] == NOT_ESTIMABLE_CONSTANT:  # agent slope, difference, CI and p: one spanning cell
                return head + [f"\\multicolumn{{4}}{{c}}{{{NOT_ESTIMABLE_CONSTANT}}}", rmse]
            return head + [fmt_num(r["slope_sim"]), fmt_num(r["slope_difference"]),
                           f"[{fmt_num(r['ci_low'])}, {fmt_num(r['ci_high'])}]", fmt_p(r["interaction_p"]), rmse]

        body = [liwc_row(r) for _, r in sub.iterrows()]
        constant_preds = sorted({clean_label(r["predictor"]) for _, r in sub.iterrows() if r["slope_sim"] == NOT_ESTIMABLE_CONSTANT})
        notes = ["Slopes are from regressing the LIWC outcome on the predictor (CRT2 score or bCRT score); Difference = agent minus human, "
                 "tested by the predictor$\\times$source interaction. Binned RMSE compares binned agent and human means.",
                 f"Archival identifiers: simulator run {tex_id(label)}; analysis report {tex_id('cr13_' + str(label))}."]
        if label == DSV4PRO_LABEL:
            notes.insert(1, DSV4PRO_BRT_N_TEX)   # S-8d (D-276(c)): readable run name, as in the other printed tables
        if constant_preds:
            notes.insert(1, f"Rows marked ``{NOT_ESTIMABLE_CONSTANT}'': the agent's {' and '.join(constant_preds)} score is the same for "
                            "every persona in this run, so no agent slope, difference, confidence interval or $p$ can be estimated; "
                            "the human slope is shown as computed.")
        tex = tex_booktabs(hdr, body, "@{}l" + tex_x(1.0) + "rrrrrr@{}",
                           f"LIWC relationship fidelity, {tex_escape(condition_name(label))}.",
                           f"tab:liwc-slopes-{i + 1}", notes)
        name = f"table_liwc_theory_set_slope_fidelity_{i + 1}_{label}.tex"
        (dirs["tables"] / name).write_text(tex, encoding="utf-8")
        diag.table(name)
        inputs.append(f"\\input{{{name[:-4]}}}")
    (dirs["tables"] / "table_liwc_theory_set_slope_fidelity.tex").write_text(
        "% Table 5 split by run: one table per condition (CR-14). Requires \\usepackage{booktabs,tabularx}.\n" + "\n\\clearpage\n".join(inputs) + "\n",
        encoding="utf-8")
    diag.table("table_liwc_theory_set_slope_fidelity.tex")


def build_liwc_appendix_assets(root: Path, selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    table = load_liwc_fidelity_rows(root, selected, dirs, diag)
    if table.empty:
        empty_figure(dirs["figures"] / "appendix_liwc_slope_fidelity_theory_set", "LIWC slope fidelity", "No theory-set LIWC fidelity rows were available.", diag)
        empty_figure(dirs["figures"] / "appendix_liwc_slope_difference_heatmap", "LIWC slope differences", "No theory-set LIWC fidelity rows were available.", diag)
        write_liwc_captions(dirs)
        return table
    plot_liwc_slope_fidelity(table, dirs["figures"] / "appendix_liwc_slope_fidelity_theory_set", diag)
    plot_liwc_slope_heatmap(table, dirs["figures"] / "appendix_liwc_slope_difference_heatmap", diag)
    write_liwc_captions(dirs)
    fig_appendix_liwc_dots_slopes(root, selected, table, dirs, diag)   # S-8d: design D (D-275), bCRT and CRT2
    return table


def plot_liwc_slope_fidelity(table: pd.DataFrame, base_path: Path, diag: Diagnostics) -> None:
    outcomes = [o for o in LIWC_THEORY_OUTCOMES if o in set(table["liwc_outcome"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True, constrained_layout=True)
    for ax, predictor in zip(axes, LIWC_PREDICTORS, strict=False):
        sub = table[table["predictor"] == predictor]
        y = np.arange(len(outcomes))
        human = sub.groupby("liwc_outcome")["slope_human"].first()
        ax.scatter([safe_float(human.get(o)) for o in outcomes], y, marker=HUMAN_MARKER_ALONG_X, s=HUMAN_MARKER_SIZE ** 2,
                   linewidths=HUMAN_MARKER_EDGE, color=HUMAN_INK, label="Human", zorder=0.9)
        for i, label in enumerate(TARGET_LABELS):
            run = sub[sub["run_label"] == label].set_index("liwc_outcome")
            vals = [safe_float(run.loc[o, "slope_sim"]) if o in run.index else np.nan for o in outcomes]
            ax.scatter(vals, y + (i - 1.5) * 0.08, s=30, color=RUN_COLORS[label], marker=RUN_MARKERS[label], label=RUN_SHORT[label])
        ax.axvline(0, color="#777777", lw=0.9)
        ax.set_title(clean_label(predictor), loc="left", weight="bold")
        ax.set_xlabel("Slope")
        ax.grid(axis="x", color="#eeeeee")
        ax.set_yticks(y)
        ax.set_yticklabels([clean_label(o) for o in outcomes])
        ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    fig.suptitle("Appendix: LIWC relationship-fidelity slopes for theory-relevant BRT language markers", weight="bold")
    save_figure(fig, base_path)
    diag.figure(base_path.with_suffix(".pdf").name)
    diag.figure(base_path.with_suffix(".png").name)


def plot_liwc_slope_heatmap(table: pd.DataFrame, base_path: Path, diag: Diagnostics) -> None:
    outcomes = [o for o in LIWC_THEORY_OUTCOMES if o in set(table["liwc_outcome"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True, constrained_layout=True)
    vals_all = pd.to_numeric(table["interaction_beta"], errors="coerce").dropna().abs()
    vmax = max(0.1, float(vals_all.max()) if not vals_all.empty else 0.1)
    im = None
    for ax, predictor in zip(axes, LIWC_PREDICTORS, strict=False):
        mat = pd.DataFrame(index=outcomes, columns=TARGET_LABELS, dtype=float)
        sub = table[table["predictor"] == predictor]
        for _, row in sub.iterrows():
            mat.loc[row["liwc_outcome"], row["run_label"]] = safe_float(row.get("interaction_beta"))
        im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(TARGET_LABELS)))
        ax.set_xticklabels([RUN_SHORT[l] for l in TARGET_LABELS])
        ax.set_yticks(range(len(outcomes)))
        ax.set_yticklabels([clean_label(o) for o in outcomes])
        ax.set_title(clean_label(predictor), loc="left", weight="bold")
        for i in range(len(outcomes)):
            for j in range(len(TARGET_LABELS)):
                value = mat.iloc[i, j]
                ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=7)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="slope_sim - slope_human")
    fig.suptitle("Appendix: LIWC slope-difference heatmap", weight="bold")
    save_figure(fig, base_path)
    diag.figure(base_path.with_suffix(".pdf").name)
    diag.figure(base_path.with_suffix(".png").name)


def write_liwc_captions(dirs: dict[str, Path]) -> None:
    text = """# LIWC Appendix Captions

**Appendix LIWC slope fidelity.** These LIWC analyses test whether simulations reproduce human relationships between cognitive-reflection scores and theory-relevant language markers in BRT responses. They are supplementary because the main paper focuses on distributional and behavioural fidelity, not full BRT construct validation.

**Appendix LIWC slope-difference heatmap.** Heatmap cells show simulated minus human slopes for bCRT and CRT2 relationships with selected BRT-LIWC outcomes; large deviations indicate that language-marker associations are not faithfully transferred even when marginal language outputs are available.

**Appendix LIWC theory-set slope table.** The table reports human slopes, simulated slopes, slope differences, interaction tests, confidence intervals where available, and binned RMSE for the theory-relevant LIWC outcome set.
"""
    (dirs["captions"] / "liwc_captions.md").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# S-8d (D-275; presentation only): the two appendix LIWC figures, design D. One column per run whose CRT scores vary,
# one row per theory-set LIWC category; every person and agent a dot at its score, with people's and the run's
# fitted lines (the published slopes, recomputed from the person frames and guarded against the slope table and the
# LIWC family table). Lines solid where the slope's 95% interval excludes 0, dashed where not. Each panel carries the
# verdict of Appendix A.1's transfer rules, from intervals that already exist: people's and the run's own slope
# intervals (liwc_family_models_all_runs.csv) and the difference interval (the slope table's ci_low/ci_high).
# ---------------------------------------------------------------------------------------------------
LIWC_FAMILY_TABLE = Path("reports") / "cr13b_liwc_calibration" / "tables" / "liwc_family_models_all_runs.csv"
LIWC_FIG_OUTCOMES = [("WC", "Word count"), ("auxverb", "Auxiliary verbs"), ("differ", "Differentiation"),
                     ("emo_neg", "Negative emotion"), ("focuspresent", "Present focus"), ("moral", "Moral language"),
                     ("risk", "Risk")]
LIWC_FIG_PREDICTORS = {"BCRT_Total": ("bCRT", "BCRT", "appendix_liwc_dots_slopes_bcrt"),
                       "CRT2_Total": ("CRT2", "CRT2", "appendix_liwc_dots_slopes_crt2")}
LIWC_VERDICT_LABEL = {"transfers": "T  transfers", "fails: weakened": "F  weakened", "fails: reversed": "F  reversed",
                      "fails: exaggerated": "F  exaggerated", "introduced": "I  introduced",
                      "indeterminate": "?  indeterminate", "no human target": "0  no human target"}
LIWC_PEOPLE_DOT = "#9a9a9a"


def _excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def liwc_transfer_verdict(h_b: float, h_lo: float, h_hi: float, r_b: float, r_lo: float, r_hi: float,
                          d_lo: float, d_hi: float) -> str:
    """Appendix A.1's transfer rules, with the trait-structure sub-types for a failure (D-275(b)). People's link is
    judged by its uncorrected 95% interval, as for the BRT (D-275(d))."""
    if not _excludes_zero(h_lo, h_hi):
        return "introduced" if _excludes_zero(r_lo, r_hi) and _excludes_zero(d_lo, d_hi) else "no human target"
    if _excludes_zero(d_lo, d_hi):
        if _excludes_zero(r_lo, r_hi) and np.sign(r_b) != np.sign(h_b):
            return "fails: reversed"
        if np.sign(r_b) == np.sign(h_b) and abs(r_b) > abs(h_b):
            return "fails: exaggerated"
        return "fails: weakened"
    if _excludes_zero(r_lo, r_hi) and np.sign(r_b) == np.sign(h_b):
        return "transfers"
    return "indeterminate"


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    m = ~(np.isnan(x) | np.isnan(y))
    b, a = np.polyfit(x[m], y[m], 1)
    return float(b), float(a), int(m.sum())


def fig_appendix_liwc_dots_slopes(root: Path, selected: dict[str, ReportCandidate], table: pd.DataFrame,
                                  dirs: dict[str, Path], diag: Diagnostics) -> None:
    fam = pd.read_csv(root / LIWC_FAMILY_TABLE)
    tab = table.copy()
    runs: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    human: pd.DataFrame | None = None
    for label in TARGET_LABELS:
        if label not in selected:
            continue
        agent = load_agent_person_frame(root, selected[label])
        if all(agent_predictor_is_constant(agent, p) for p in LIWC_FIG_PREDICTORS):
            continue   # every agent scores the same on both tests: no slope, so no column (the caption says so)
        runs.append(label)
        frames[label] = agent
        if human is None:
            for base in source_roots_for_candidate(root, selected[label]):
                path = base / "tables" / "by_dataset" / "human" / "00_person_frame.csv"
                if path.exists():
                    human = pd.read_csv(path, low_memory=False)
                    break
    if human is None or not runs:
        raise SystemExit("STOP: S-8d LIWC figures: no human person frame or no run with varying CRT scores.")
    diag.note(f"S-8d LIWC figures: {len(runs)} of {len(TARGET_LABELS)} runs kept (the rest score the same on both tests).")
    rows = []
    for pred, (pname, pgroup, stem) in LIWC_FIG_PREDICTORS.items():
        hcol = _resolve_cognitive_predictor_column(human, "human", pred)
        hx = pd.to_numeric(human[hcol], errors="coerce").to_numpy(dtype=float)
        rng = np.random.default_rng(42)
        nrow = len(LIWC_FIG_OUTCOMES)
        with plt.rc_context({"font.size": 7, "axes.linewidth": 0.6}):
            fig, axes = plt.subplots(nrow, len(runs), figsize=(5.5, 7.3), squeeze=False, sharex=True)
            fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.115, hspace=0.36, wspace=0.12)
            for i, (var, vname) in enumerate(LIWC_FIG_OUTCOMES):
                hy = pd.to_numeric(human[var], errors="coerce").to_numpy(dtype=float)
                hfam = fam[(fam.dataset == "human") & (fam.liwc_var == var) & (fam.predictor_group == pgroup)
                           & (fam.run_label == runs[0])].iloc[0]
                assert hfam.predictor_source_col == hcol, (pred, hfam.predictor_source_col, hcol)
                hb, ha, hn = _ols_slope(hx, hy)
                assert abs(hb - hfam.beta) < 1e-6 and hn == int(hfam.N), (pred, var, hb, hfam.beta, hn, hfam.N)
                every = [hy] + [pd.to_numeric(frames[r][var], errors="coerce").to_numpy(dtype=float) for r in runs]
                allv = np.concatenate(every)
                allv = allv[~np.isnan(allv)]
                lo, hi = float(allv.min()), float(allv.max())
                pad = 0.04 * (hi - lo)
                for j, label in enumerate(runs):
                    ax = axes[i, j]
                    agent = frames[label]
                    acol = _resolve_cognitive_predictor_column(agent, AGENT_DATASET, pred)
                    axv = pd.to_numeric(agent[acol], errors="coerce").to_numpy(dtype=float)
                    ay = pd.to_numeric(agent[var], errors="coerce").to_numpy(dtype=float)
                    afam = fam[(fam.dataset == AGENT_DATASET) & (fam.liwc_var == var) & (fam.predictor_group == pgroup)
                               & (fam.run_label == label)].iloc[0]
                    assert afam.predictor_source_col == acol, (label, pred, afam.predictor_source_col, acol)
                    rb, ra, rn = _ols_slope(axv, ay)
                    assert abs(rb - afam.beta) < 1e-6 and rn == int(afam.N), (label, pred, var, rb, afam.beta, rn, afam.N)
                    trow = tab[(tab.run_label == label) & (tab.predictor == pred) & (tab.liwc_outcome == var)].iloc[0]
                    assert abs(float(trow.slope_human) - hb) < 1e-6 and abs(float(trow.slope_sim) - rb) < 1e-6, (label, pred, var)
                    d_lo, d_hi = float(trow.ci_low), float(trow.ci_high)
                    assert _excludes_zero(d_lo, d_hi) == (float(trow.interaction_p) < 0.05), (label, pred, var)
                    verdict = liwc_transfer_verdict(hb, hfam.CI95_low, hfam.CI95_high, rb, afam.CI95_low,
                                                    afam.CI95_high, d_lo, d_hi)
                    rows.append(dict(test=pname, category=vname, run_label=label, run=CONDITIONS[label]["name"],
                                     human_slope=hb, human_ci_low=hfam.CI95_low, human_ci_high=hfam.CI95_high,
                                     human_N=hn, run_slope=rb, run_ci_low=afam.CI95_low, run_ci_high=afam.CI95_high,
                                     run_N=rn, diff_ci_low=d_lo, diff_ci_high=d_hi,
                                     human_excludes_0=_excludes_zero(hfam.CI95_low, hfam.CI95_high),
                                     run_excludes_0=_excludes_zero(afam.CI95_low, afam.CI95_high),
                                     diff_excludes_0=_excludes_zero(d_lo, d_hi), verdict=verdict,
                                     letter=LIWC_VERDICT_LABEL[verdict].split()[0]))
                    col = RUN_COLORS[label]
                    faded = verdict == "no human target"
                    alpha = 0.15 if faded else 0.45
                    if pred == "BCRT_Total":
                        px = hx - 0.17 + rng.uniform(-0.11, 0.11, len(hx))
                    else:   # people at their familiarity-adjusted scores, which are not whole numbers
                        px = hx + rng.uniform(-0.04, 0.04, len(hx))
                    ax.scatter(px, hy, s=2.2, color=LIWC_PEOPLE_DOT, alpha=alpha, lw=0, zorder=2, rasterized=True)
                    ax.scatter(axv + 0.17 + rng.uniform(-0.11, 0.11, len(axv)), ay, s=2.2, color=col, alpha=alpha, lw=0,
                               zorder=3, rasterized=True)
                    dash = (0, (3, 1.6))
                    for x, b, a, c, ex0 in ((hx, hb, ha, LIWC_PEOPLE_DOT if faded else HUMAN_INK,
                                             _excludes_zero(hfam.CI95_low, hfam.CI95_high)),
                                            (axv, rb, ra, col, _excludes_zero(afam.CI95_low, afam.CI95_high))):
                        xs = np.array([np.nanmin(x), np.nanmax(x)])
                        ax.plot(xs, a + b * xs, color=c, lw=1.6, zorder=5, solid_capstyle="round", ls="-" if ex0 else dash)
                    if faded:
                        for sp in ax.spines.values():
                            sp.set_color("#bbbbbb")
                    ax.text(0.03, 0.97, LIWC_VERDICT_LABEL[verdict], transform=ax.transAxes, ha="left", va="top",
                            fontsize=6.3, fontweight="normal" if faded else "bold", color="#888888" if faded else "#111111",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=10)
                    ax.set_ylim(lo - pad, hi + pad)
                    ax.set_xlim(-0.5 if pred == "BCRT_Total" else -0.9, 4.5)
                    ax.set_xticks(range(5))
                    ax.tick_params(length=2, pad=1, labelsize=6)
                    for sp in ("top", "right"):
                        ax.spines[sp].set_visible(False)
                    if j > 0:
                        ax.set_yticklabels([])
                    else:
                        ax.set_ylabel(vname, fontsize=7)
                    if i == 0:
                        ax.set_title(f"{CONDITIONS[label]['name']}\nN = {rn}", fontsize=7.5, fontweight="bold", color=col,
                                     linespacing=1.15)
                    if i == nrow - 1:
                        ax.set_xlabel(f"{pname} score", fontsize=6.5)
            from matplotlib.lines import Line2D
            where = ", left of each score" if pred == "BCRT_Total" else " at the adjusted score"
            handles = [
                Line2D([], [], marker="o", color=LIWC_PEOPLE_DOT, ls="none", ms=3, label=f"people, N = {hn} (one dot each{where})"),
                Line2D([], [], color=HUMAN_INK, lw=1.6, label="people's fitted slope"),
                Line2D([], [], marker="o", color="#666666", ls="none", ms=3, label="agents (run hue, right of each score)"),
                Line2D([], [], color="#666666", lw=1.6, label="run's fitted slope (run hue)"),
                Line2D([], [], color="#666666", lw=1.6, ls=(0, (3, 1.6)), label="dashed: slope's 95% interval includes 0"),
                Line2D([], [], color="#bbbbbb", lw=1.6, label="faded panel: people show no link (nothing to reproduce)"),
            ]
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.2)
            save_figure(fig, dirs["figures"] / stem)
        diag.figure(f"{stem}.pdf")
        diag.figure(f"{stem}.png")
    out = pd.DataFrame(rows)
    out.to_csv(dirs["tables"] / "appendix_liwc_transfer_verdicts.csv", index=False)
    diag.table("appendix_liwc_transfer_verdicts.csv")
    for (test, verdict), n in out.groupby(["test", "verdict"]).size().items():
        print(f"S-8d LIWC verdicts, {test}: {verdict} {n}")


# ---------------------------------------------------------------------------------------------------
# S-8d (D-268(b); presentation only): the appendix BRT categories figure, design C. People's share of responses in each
# category of the four coded categorical fields, and each run's share minus people's. Counts only. Guards: every group
# on the same rubric (people from the canonical rubric-3.0 recode, runs coded brt_v3), the category set equal to
# brt_v3/schema.py STRING_FIELDS, counts summing to N and shares to 1 per group and field.
# ---------------------------------------------------------------------------------------------------
BRT_CAT_FIELD_TITLES = {
    "BRT1_decision": "The restaurant decision",
    "BRT1_anchor": "What the restaurant decision rested on",
    "BRT2_decision": "The medication decision",
    "BRT2_anchor": "What the medication decision rested on",
}
BRT_CAT_WORDS = {   # reading order and plain words (S-8b last packet Q6; PI made no correction)
    "BRT1_decision": [("KEEP_BOOKING", "keeps the booking"), ("CHANGE_RESTAURANT", "changes restaurant"),
                      ("CANCEL_OR_POSTPONE", "cancels or postpones"), ("UNCLEAR_OR_CONDITIONAL", "unclear or conditional")],
    "BRT1_anchor": [("ANCHOR_CRITICS_OFFICIAL", "critics or official ratings"), ("ANCHOR_CROWD", "customer reviews"),
                    ("ANCHOR_FRIEND_TESTIMONY", "a friend's account"), ("ANCHOR_VENUE_DIRECT", "the restaurant's own word"),
                    ("ANCHOR_PRACTICAL_GROUNDS", "practical or social reasons"),
                    ("ANCHOR_OWN_WEIGHING", "own checking across sources"),
                    ("ANCHOR_UNRESOLVED_CONFLICT", "sources left in conflict"), ("ANCHOR_NONE", "nothing stated")],
    "BRT2_decision": [("TAKE_MEDEX", "takes the medication"), ("DELAY_AND_INVESTIGATE", "postpones and checks first"),
                      ("NOT_TAKE_MEDEX", "refuses the medication"), ("UNCLEAR_OR_CONDITIONAL", "no clear direction")],
    "BRT2_anchor": [("ANCHOR_MED_SYSTEM", "the medical system"), ("ANCHOR_OWN_GP_PERSONAL", "own doctor, on personal trust"),
                    ("ANCHOR_LAY_ONLINE", "online or lay sources"), ("ANCHOR_PRACTICAL_GROUNDS", "practical reasons"),
                    ("ANCHOR_OWN_WEIGHING", "own checking across sources"),
                    ("ANCHOR_UNRESOLVED_CONFLICT", "sources left in conflict"), ("ANCHOR_NONE", "nothing stated")],
}


def brt_category_shares(root: Path) -> pd.DataFrame:
    from brt_v3.schema import STRING_FIELDS
    assert list(BRT_CAT_WORDS) == list(STRING_FIELDS), "field set differs from brt_v3/schema.py"
    for f, cats in BRT_CAT_WORDS.items():
        assert {c for c, _ in cats} == STRING_FIELDS[f], f"{f}: category set differs from brt_v3/schema.py"
    frames = {"human": pd.read_csv(root / "reports" / "brt_recode_v3" / "brt_labels_v3_human.csv", low_memory=False)}
    for label in TARGET_LABELS:
        df = pd.read_csv(root / "reports" / f"cr13_{label}" / "tables" / "by_dataset" / AGENT_DATASET / "00_person_frame.csv",
                         low_memory=False)
        df = df[df["socsim_run_label"].astype(str) == label]
        coded = df["BRT_scoring_scorer_version"].dropna().unique().tolist()
        assert coded == ["brt_v3"], (label, coded)
        frames[label] = df
    expected_n = {"human": BRT_TABLE_HUMAN_N, **BRT_TABLE_N}
    rows = []
    for g, df in frames.items():
        for f, cats in BRT_CAT_WORDS.items():
            vals = df[f].dropna().astype(str)
            extra = set(vals) - STRING_FIELDS[f]
            assert not extra, f"{g} {f}: categories outside the rubric: {extra}"
            n = len(vals)
            assert n == expected_n[g], (g, f, n)
            for code, words in cats:
                k = int((vals == code).sum())
                rows.append(dict(group=g, name="People" if g == "human" else CONDITIONS[g]["name"], field=f, code=code,
                                 words=words, count=k, N=n, share=k / n))
    out = pd.DataFrame(rows)
    for (g, f), sub in out.groupby(["group", "field"]):
        assert abs(sub["share"].sum() - 1) <= 1e-9 and sub["count"].sum() == sub["N"].iloc[0], (g, f)
    hum = out[out.group == "human"].set_index(["field", "code"])["share"]
    out["diff_pp"] = [100 * (s - hum[(f, c)]) for s, f, c in zip(out["share"], out["field"], out["code"])]
    return out


def fig_appendix_brt_categories(root: Path, dirs: dict[str, Path], diag: Diagnostics) -> None:
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    d = brt_category_shares(root)
    d.to_csv(dirs["tables"] / "appendix_brt_category_shares.csv", index=False)
    diag.table("appendix_brt_category_shares.csv")
    pct = d.assign(pct=100 * d["share"]).set_index(["group", "field", "code"])["pct"]
    nval = d.groupby("group")["N"].first()
    W = 5.5
    rows = []
    for f, cats in BRT_CAT_WORDS.items():
        rows.append(("head", f))
        rows += [("cat", f, code, words) for code, words in cats]
    row_in, head_in, hdr_in, bot_in = 0.165, 0.21, 1.12, 0.50
    H = hdr_in + bot_in + sum(head_in if r[0] == "head" else row_in for r in rows)
    left_in, people_in, col_in, gap_in = 1.58, 0.56, 0.44, 0.08
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    assert d["diff_pp"].abs().max() <= 60, "a gap beyond the colour scale"
    with plt.rc_context({"font.size": 7}):
        fig = plt.figure(figsize=(W, H))
        xs_runs = left_in + people_in + gap_in
        y = H - hdr_in
        fig.text((left_in + people_in / 2) / W, (y + 0.05) / H, f"People\nN = {nval['human']}", ha="center", va="bottom",
                 fontsize=7, weight="bold", linespacing=1.1)
        for j, g in enumerate(TARGET_LABELS):
            cx = xs_runs + (j + 0.5) * col_in
            nm = CONDITIONS[g]["name"].replace("Claude Sonnet 4.6 · ", "Claude Sonnet 4.6\n")
            fig.text(cx / W, (y + 0.17) / H, f"{nm}\nN = {nval[g]}", ha="center", va="bottom", rotation=90, fontsize=7,
                     linespacing=1.05, color="#0b0b0b", multialignment="left")
            fig.add_artist(Line2D([cx / W], [(y + 0.08) / H], marker=RUN_MARKERS[g], color=RUN_COLORS[g], ms=4.2,
                                  mec="white", mew=0.3, ls="", transform=fig.transFigure))
        for r in rows:
            if r[0] == "head":
                y -= head_in
                fig.text(0.04 / W, (y + 0.05) / H, BRT_CAT_FIELD_TITLES[r[1]], fontsize=7.5, weight="bold", va="bottom")
                continue
            _, f, code, words = r
            y -= row_in
            fig.text((left_in - 0.06) / W, (y + row_in / 2) / H, words, ha="right", va="center", fontsize=7)
            hv = float(pct[("human", f, code)])
            bar_max = 0.24
            fig.add_artist(Rectangle((left_in / W, (y + 0.03) / H), bar_max / W, (row_in - 0.06) / H,
                                     transform=fig.transFigure, facecolor="#f1f0ec", edgecolor="none"))
            fig.add_artist(Rectangle((left_in / W, (y + 0.03) / H), bar_max * hv / 100 / W, (row_in - 0.06) / H,
                                     transform=fig.transFigure, facecolor=HUMAN_INK, edgecolor="none"))
            fig.text((left_in + people_in - 0.04) / W, (y + row_in / 2) / H, f"{hv:.1f}%" if 0 < hv < 1 else f"{hv:.0f}%",
                     ha="right", va="center", fontsize=7, color="#0b0b0b")
            for j, g in enumerate(TARGET_LABELS):
                dv = float(pct[(g, f, code)]) - hv
                c = cmap(norm(dv))
                x0 = xs_runs + j * col_in
                fig.add_artist(Rectangle((x0 / W, (y + 0.012) / H), (col_in - 0.025) / W, (row_in - 0.024) / H,
                                         transform=fig.transFigure, facecolor=c, edgecolor="none"))
                if abs(dv) >= 10:
                    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
                    fig.text((x0 + (col_in - 0.025) / 2) / W, (y + row_in / 2) / H, f"{dv:+.0f}".replace("-", "−"),
                             ha="center", va="center", fontsize=7, color="white" if lum < 0.5 else "#0b0b0b")
        cax = fig.add_axes([(xs_runs + 0.15) / W, 0.30 / H, (len(TARGET_LABELS) * col_in - 0.3) / W, 0.07 / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_ticks([-60, -30, 0, 30, 60])
        cb.set_ticklabels(["−60", "−30", "0", "+30", "+60"])
        cb.outline.set_visible(False)
        cax.tick_params(length=2, pad=1, labelsize=7)
        cax.set_xlabel("Run's share minus people's share (percentage points)", labelpad=2, fontsize=7)
        save_figure(fig, dirs["figures"] / "appendix_brt_categories_difference")
    diag.figure("appendix_brt_categories_difference.pdf")
    diag.figure("appendix_brt_categories_difference.png")


def write_captions(dirs: dict[str, Path], diag: Diagnostics) -> None:
    fig5_scope = "core cognitive/personality associations only" if not diag.association_includes_brt else "core and BRT associations"
    text = f"""# Draft Captions

## Figures

**Figure 1. Cognition-suite workflow.** The paper compares human-reference and statistically sampled synthetic profiles after they are expressed through trait-conditioned LLM agents, then evaluates whether the resulting cognition-suite outputs preserve distributional, calibration, intuitive-lure, association, and BRT behavioural patterns.

**Figure 2. Distribution fidelity dashboard.** Human-reference distributions provide the target shape for each construct; deviations in the simulated runs highlight where mean alignment can still mask differences in dispersion or category mass. The proportion axis starts at 0; the mean and 95% CI of each source are drawn in the strip beneath each panel, on the same score axis.

**Figure 2b. Distribution fidelity ECDF overlays.** ECDF overlays make the main distributional result easier to see: the agent runs diverge most strongly on CRT2 and bCRT while NFC and BFI traits are closer to the human reference.

**Figure 3. CRT2 and bCRT response-category fidelity.** Panels A and B show the response categories as shares of scored responses (green correct, orange intuitive incorrect, red other incorrect); missing or unparseable responses are excluded from every denominator and reported as the unscored count in table_intuitive_lure_prevalence_by_run_task.csv. The human reference is the 471 CRT2-complete respondents. Panel C shows the intuitive-incorrect prevalence (intuitive responses / scored responses) in the same orange: solid bars are CRT2, hatched bars are bCRT. Every bar carries its exact value, including the runs at exactly 0.0%, which are true zeros rather than missing data; counts (intuitive / scored) are printed under each run and exported in the same table.

**Figure 4. BRT behavioural compression.** Prevalence of the fifteen primary BRT outcomes (ANALYSIS_PLAN §5, four families) per run beside the human reference, with 95% Wilson intervals; "C" marks a compressed outcome (|gap| ≥ 20 pp with non-overlapping intervals).

**Figure 5. Association fidelity heatmaps.** Cells show Δr = r_sim - r_human for CRT2, bCRT, NFC and all five BFI traits (Extraversion and Agreeableness included; lower triangle shown): blue indicates a weaker or more negative simulated association than the human reference, while red indicates a stronger or more positive simulated association. Significance stars are omitted to preserve readability because validated FDR-corrected difference-test cells were not available in the selected assets; significance tests are reported in the appendix/table where available. BRT associations were unavailable in the selected difference tables. Blank bCRT rows/columns mean the correlation is undefined because realised bCRT is constant (4/4) in that run.

**Figure 6. Calibration fingerprint.** Low RMSE is better and high R² is better. CRT2 and bCRT calibration is poor across runs, whereas NFC and BFI calibration is much stronger; the two-panel display separates R² from raw RMSE so both rank-order tracking and error magnitude are visible.

**Appendix Figure. Relative calibration RMSE.** Cell values and colour share one scale: RMSE normalised within each construct row, (RMSE - row min) / (row max - row min), so 0 is the best run for that construct and 1 the worst. Raw RMSE is in table_calibration_fidelity.csv and appendix_calibration_relative_rmse_data.csv.

**Appendix Figure. Which response categories does each told CRT2 level produce?** One panel per run and task. Rows are the CRT2 level each agent was told, 0-4 (`crt2_level`, the level written into the persona). bCRT was never specified as a parameter, so the bCRT panels group the same agents by their told CRT2 level. Each bar carries its intuitive-lure prevalence and n (agents and item responses); groups sum to 471 agents per run.

**Appendix Figure. BRT categorical distributions.** Categorical BRT outputs show which response categories underlie the primary prevalence compression in the main text.

**Appendix Figure. Flagged BRT transfer models.** Models flagged unstable (fewer than 20 events or 20 non-events in the run) are reported as diagnostics and excluded from the transfer-verdict tally.

**Appendix Figure. LIWC/BRT relationship fidelity.** LIWC slope plots test whether simulations reproduce human relationships between cognitive-reflection scores and theory-relevant BRT language markers; these are supplementary to the main distributional and behavioural fidelity results.

## Tables

**Table 1. Distribution fidelity metrics.** Mean differences, effect sizes, JS divergence, and variance ratios quantify how closely each run matches human-reference marginal construct distributions.

**Table 2. CRT/bCRT response-category fidelity.** Category-level proportions and JS divergence isolate whether correctness, intuitive errors, other errors, and unparseable outputs match the human response mix.

**Table 3. BRT distributional fidelity.** Per run and primary outcome: human and run prevalence with 95% CIs, the gap, the compressed flag (|gap| ≥ 20 pp, non-overlapping CIs), the unstable flag (< 20 events or non-events) and the F3 transfer verdict on realised bCRT.

**Table 4. Association fidelity metrics.** Mean absolute correlation error and Frobenius norm compress the heatmap differences into run-level association-fidelity summaries.

**Table 5. Calibration fidelity.** Construct-level RMSE, MAE, R2, slope, and intercept report how well each run realises its specified psychological profile.

**Table 6. Experimental conditions.** Readable condition names with persona source, format, model and configuration for each run; archival run identifiers are in the footnotes.

**Table 7. Fidelity scorecard.** The scorecard combines marginal, intuitive-lure, calibration, association, and BRT compression summaries so the main behavioural-fidelity trade-offs are visible by run.

**Table 8. BRT transfer models flagged.** Primary-outcome model rows flagged unstable (< 20 events or < 20 non-events in the run).

**Table 9. Prompt-screening summary.** When available, pilot GOOD/MODERATE/POOR labels are useful for prompt selection but are unweighted across constructs and evaluation types, so they should not be treated as the final criterion for full-run behavioural fidelity.
"""
    (dirs["captions"] / "captions.md").write_text(text, encoding="utf-8")


def write_diagnostics(dirs: dict[str, Path], selected: dict[str, ReportCandidate], diag: Diagnostics) -> None:
    lines = [
        "# Missing Assets Report",
        "",
        "This report records missing labels, missing files, missing columns, skipped panels, and fallback decisions encountered while building the camera-ready assets.",
        "",
        "## Selected Reports",
        "",
    ]
    for label in TARGET_LABELS:
        candidate = selected.get(label)
        if candidate is None:
            lines.append(f"- `{RUN_DISPLAY[label]}`: missing; {diag.missing_labels.get(label, 'no selected report')}")
        else:
            lines.append(f"- `{RUN_DISPLAY[label]}`: `{candidate.report_id}` selected via `{candidate.source}`.")
    lines.extend(["", "## Missing Labels", ""])
    lines.extend([f"- `{label}`: {reason}" for label, reason in diag.missing_labels.items()] or ["- None."])
    lines.extend(["", "## Missing Files", ""])
    lines.extend(diag.missing_files or ["- None."])
    lines.extend(["", "## Missing Columns Or Rows", ""])
    lines.extend(diag.missing_columns or ["- None."])
    lines.extend(["", "## Skipped Panels Or Outputs", ""])
    lines.extend(diag.skipped_panels or ["- None."])
    lines.extend(["", "## Notes", ""])
    lines.extend(diag.notes or ["- None."])
    (dirs["diagnostics"] / "missing_assets_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    alias_path = dirs["diagnostics"] / "variable_aliases_used.md"
    if diag.alias_rows:
        alias_lines = [
            "# Variable Aliases Used",
            "",
            "| run_label | requested_variable | actual_variable | source_file |",
            "| --- | --- | --- | --- |",
        ]
        for row in diag.alias_rows:
            alias_lines.append(
                f"| `{row['run_label']}` | `{row['requested_variable']}` | `{row['actual_variable']}` | `{row['source_file']}` |"
            )
    else:
        alias_lines = ["# Variable Aliases Used", "", "No variable aliases were needed for available association tables."]
    alias_path.write_text("\n".join(alias_lines) + "\n", encoding="utf-8")

    quality = [
        "# Asset Quality Report",
        "",
        "## Regenerated Figures",
        "",
        *(f"- `{name}`" for name in sorted(diag.regenerated_figures)),
        "",
        "## Recalculated Tables",
        "",
        *(f"- `{name}`" for name in sorted(diag.recalculated_tables)),
        "",
        "## BRT Prevalence Extraction",
        "",
        "- Status: " + ("succeeded for all selected run/outcome cells." if diag.brt_prevalence_complete else "incomplete; see `missing_assets_report.md` for truly absent cells."),
        "",
        "## Association Heatmap Variables",
        "",
        "- BRT variables available: " + ("yes." if diag.association_includes_brt else "no; Figure 5 uses core associations only."),
        "- Figure 5 polish: updated the title, replaced the colourbar label with `Correlation difference (r_sim - r_human)`, kept clean construct/run labels, and omitted stars because the available FDR table is not a validated simulated-minus-human difference-test table.",
        "",
        "## Calibration Figure Polish",
        "",
        "- Figure 6 polish: relabelled `bCRT proxy` as `bCRT-derived score`, kept H-Num/H-Lex/S-Num/S-Lex run labels, retained two-decimal cell annotations, and clarified panel/colorbar labels for high R² and low RMSE.",
        "",
        "## Prompt-Screening Summary",
        "",
        "- Available: " + ("yes." if diag.prompt_screening_available else "no; see `prompt_screening_unavailable.md`."),
        "",
        "## Remaining Known Limitations",
        "",
    ]
    quality.extend(f"- {item}" for item in diag.known_limitations)
    if not diag.known_limitations:
        quality.append("- None beyond missing/skipped items listed in `missing_assets_report.md`.")
    (dirs["diagnostics"] / "asset_quality_report.md").write_text("\n".join(quality) + "\n", encoding="utf-8")


def write_compile_test_wrapper(dirs: dict[str, Path], diag: Diagnostics) -> None:
    """CR-23 STEP 6: a stand-alone document that sets every generated table at the NeurIPS text width
    (5.5 in). Compile it in the tables folder; the log must show no "Overfull \\hbox" over 1 pt."""
    tables = sorted(p.stem for p in dirs["tables"].glob("table_*.tex")
                    if "\\begin{table}" in p.read_text(encoding="utf-8"))
    body = "\n\\clearpage\n".join(f"\\input{{{t}}}" for t in tables)
    wrapper = (
        "% CR-23 STEP 6 compile test, generated by scripts/build_neurips_camera_ready_assets.py.\n"
        "% pdflatex -interaction=nonstopmode _cr23_compile_test.tex && grep -n 'Overfull' _cr23_compile_test.log\n"
        "\\documentclass{article}\n\\usepackage[textwidth=5.5in]{geometry}\n"
        "\\usepackage{booktabs,tabularx,graphicx,xurl}\n\\hfuzz=1pt\n"
        "\\begin{document}\n" + body + "\n\\end{document}\n"
    )
    (dirs["tables"] / "_cr23_compile_test.tex").write_text(wrapper, encoding="utf-8")
    diag.table("_cr23_compile_test.tex")


# ---------------------------------------------------------------------------------------------------
# CR-25 (D-205(a), UPDATE_MAP §2b, PI-4/PI-11): item exposure, the configuration ladder and the three
# design probes re-referenced to the corrected familiarity-adjusted human reference, the human CRT2
# reference raw vs adjusted per item, the five §2b main-text figures, and the Part II appendix copy.
# Colour and labels follow vault _tools/manuscript/VOICE.md "Colour and labels carry meaning": one hue
# per model family, shape within Claude Sonnet 4.6 (scripts/paper_palette.py, D-208(b) / CR-25b), human black,
# RdBu_r for every agent-minus-human
# quantity, n/e cells hatched and labelled, N on every panel, batteries filled and 150-profile cells
# hollow, every caption opening with the question the figure answers.
# ---------------------------------------------------------------------------------------------------

CRT2_ITEMS = ["crt2_q1", "crt2_q2", "crt2_q3", "crt2_q4"]
BCRT_ITEMS = ["bcrt_q1", "bcrt_q2", "bcrt_q3", "bcrt_q4"]
ITEM_LABELS = {**{f"crt2_q{i}": f"CRT2 q{i}" for i in range(1, 5)}, **{f"bcrt_q{i}": f"bCRT q{i}" for i in range(1, 5)}}
# B-15 (DECISIONS_BRT, instrument text): Experience Check 1-4 -> crt2_q1-q4, 5-8 -> bcrt_q1-q4.
ITEM_SEEN_COLS = {**{f"crt2_q{i}": f"Experience Check_{i}_seen" for i in range(1, 5)},
                  **{f"bcrt_q{i}": f"Experience Check_{i + 4}_seen" for i in range(1, 5)}}
HUMAN_PERSON_FILE = Path("data") / "inputs" / "person_with_liwc_brt_coded.csv"
HUMAN_REFERENCE_N = 471
V2_LABEL_TO_CATEGORY = {"correct": "correct", "intuitive": "intuitive_incorrect", "other": "other_incorrect"}
EXPOSURE_BOOT, EXPOSURE_SEED = 5000, 42   # S-8e (PI 2 Nov, Q8 A): 5,000 as every other lure interval and Methods 4.5 (was 4000)
EXPOSURE_BOOT_TEX = f"{EXPOSURE_BOOT:,}".replace(",", "{,}")   # 5{,}000, as Appendix Table 5's note prints it
NE_HATCH = "////"
NE_FACE = "#f2f2f2"
NE_EDGE = "#9a9a9a"

RESCORE_TABLES = Path("reports") / "rescore_v2" / "tables"
COG_SIM_OUTPUTS = Path("..") / "cognition-suite-sim" / "simulation_outputs"
PANEL_150 = Path("..") / "cognition-suite-sim" / "assets" / "persona_subsets" / "REBUTTAL_PANEL_150.csv"
# CR-26 (D-309, PI ruling 3 Nov): the three configuration-screen tables were read from the vault by absolute path; they
# are now copied byte-identical, under names without job codes, to reports/external_inputs/ (relative to the root).
EXTERNAL_INPUTS = Path("reports") / "external_inputs"
CRFS_CI_CSV = EXTERNAL_INPUTS / "screen_lure_prevalence_with_ci.csv"         # CR-FS stored CIs (5000 boot)
SCREEN_ROW_TABLES = {"cr-fsb": EXTERNAL_INPUTS / "screen_rows_followup_b.csv",   # CR-FSb screen rows
                     "cr-fsd": EXTERNAL_INPUTS / "screen_rows_followup_d.csv"}   # CR-FSd screen rows
# S-8 (D-235, D-237, D-238): the CR-28 and CR-29 interval files the builder reads. Required: Figure 2 and the ladder
# table use one interval method, the persona-clustered bootstrap, for every row.
CR28_LADDER_CSV = Path("reports") / "cr28_ladder_ci" / "ladder_ci_cr28.csv"
CR28_HUMAN_ITEM_CSV = Path("reports") / "cr28_ladder_ci" / "human_item_ci_cr28.csv"
CR29_RUN_CSV = Path("reports") / "cr29_run_ci" / "run_ci_cr29.csv"
CR29_HUMAN_CSV = Path("reports") / "cr29_run_ci" / "human_pooled_ci_cr29.csv"
NO_LURES = "no lures observed"   # D-232(b): a cell with 0 lures shows its point and this label, never a [0, 0] whisker
GUARD_ATOL = 1e-12


def read_required_csv(root: Path, rel: Path) -> pd.DataFrame:
    path = root / rel
    if not path.exists():
        raise SystemExit(f"STOP: required interval file missing: {path}")
    return pd.read_csv(path)
J_CI_CSV = RESCORE_TABLES / "lure_prevalence_v1_v2.csv"                       # J2 / J3 stored CIs (Jz-B)
J3_SUMMARY = Path("reports") / "20260728_110341_REBUTTAL_TEMP_J3_SUMMARY"
J5_SUMMARY = Path("reports") / "20260728_122307_REBUTTAL_PROBE_J5_SUMMARY"
J3_WITH_HISTORY_RESCORE = "SynthNum467sim"   # the original with-history run on the same 150 ids (J3 script)
CR13B_CRT = Path("reports") / "cr13b_crt" / "tables"
PARTII_SRC = Path("reports") / "cr13b_partii"
CUTOFFS_YAML = Path("..") / "brt_final" / "reports" / "synthetic" / "crt2_level_cutoffs_v2.yaml"

# The configuration ladder = every 150-profile cell (D-199: "the 150-profile cells are the configuration
# ladder"; PI-2: one table, every configuration, Fable 5 included). Configurations quoted from the J2,
# CR-FS, CR-FSb and CR-FSd config disclosures; nothing inferred.
LADDER_CELLS = [
    # stem, provider, model, regime, format, source job, family key for colour
    ("CRFS_SONNET5_DEFAULT_NUM", "Anthropic", "Claude Sonnet 5", "adaptive thinking (default)", "numeric", "CR-FS", "sonnet5"),
    ("CRFSB_SONNET5_THINKOFF_NUM", "Anthropic", "Claude Sonnet 5", "thinking disabled", "numeric", "CR-FSb", "sonnet5"),
    ("CRFSB_CLAUDE_TOP_NUM", "Anthropic", "Claude Fable 5", "provider defaults", "numeric", "CR-FSb", "other"),
    ("J2_GEMINI35FLASH_MIN", "Google", "Gemini 3.5 Flash", "thinking level minimal", "numeric", "J2", "gemini"),
    ("CRFS_GEMINI35FLASH_MIN_LEX", "Google", "Gemini 3.5 Flash", "thinking level minimal", "lexical", "CR-FS", "gemini"),
    ("J2_DSV4PRO_ON", "DeepSeek", "DeepSeek V4-Pro, first-party API", "thinking enabled", "numeric", "J2", "dsv4pro"),
    ("CRFS_DSV4PRO_ON_LEX", "DeepSeek", "DeepSeek V4-Pro, first-party API", "thinking enabled", "lexical", "CR-FS", "dsv4pro"),
    ("CRFSD_DSV4PRO_TOG_NUM", "DeepSeek", "DeepSeek V4-Pro, Together AI", "reasoning effort high", "numeric", "CR-FSd", "dsv4pro"),
    ("CRFSD_DSV4PRO_TOG_NOTHINK_NUM", "DeepSeek", "DeepSeek V4-Pro, Together AI", "reasoning disabled", "numeric", "CR-FSd", "dsv4pro"),
    ("J2_DSV4FLASH_ON", "DeepSeek", "DeepSeek V4-Flash", "thinking enabled", "numeric", "J2", "other"),
    ("J2_DSV4FLASH_OFF", "DeepSeek", "DeepSeek V4-Flash", "thinking disabled", "numeric", "J2", "other"),
    ("J2_GPT56SOL_NONE", "OpenAI", "GPT-5.6 Sol", "reasoning effort none", "numeric", "J2", "other"),
    ("CRFSB_GPT56_REASONING_NUM", "OpenAI", "GPT-5.6 Sol", "reasoning effort high", "numeric", "CR-FSb", "other"),
    ("CRFSD_KIMI_K3_NUM", "Moonshot AI", "Kimi K3, Together AI", "provider-default effort", "numeric", "CR-FSd", "other"),
]
LADDER_PROVIDER_ORDER = ["Anthropic", "Google", "DeepSeek", "OpenAI", "Moonshot AI"]
# Family hue = the model family's hue (paper_palette); models with no full battery share one neutral grey,
# drawn hollow and named on the axis (D-208(b)).
FAMILY_COLORS = LADDER_FAMILY_COLORS
BATTERY_FAMILY = {
    "CR_MAIN_SYNTH_NUM_SONNET5_S42": ("Anthropic", "Claude Sonnet 5", "adaptive thinking (default)"),
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": ("Google", "Gemini 3.5 Flash", "thinking level minimal"),
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": ("DeepSeek", "DeepSeek V4-Pro, Together AI", "reasoning effort high"),
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42": ("Anthropic", "Claude Sonnet 4.6", "provider defaults"),
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": ("Anthropic", "Claude Sonnet 4.6", "provider defaults"),
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42": ("Anthropic", "Claude Sonnet 4.6", "provider defaults"),
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42": ("Anthropic", "Claude Sonnet 4.6", "provider defaults"),
}
SONNET46_RUNS = [l for l in TARGET_LABELS if l.endswith("_CS46_S42")]
LURE_PRODUCING_RUNS = ["CR_MAIN_SYNTH_NUM_SONNET5_S42", "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42", "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42"]


def cr25_pct(x: float, nd: int = 1) -> str:
    return "n/e" if not np.isfinite(x) else f"{100 * x:.{nd}f}"


def cr25_ci(lo: float, hi: float, nd: int = 1) -> str:
    return "n/c" if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{100 * lo:.{nd}f}, {100 * hi:.{nd}f}]"


def cr25_long_categories(df: pd.DataFrame) -> pd.Series:
    """v2 label -> manuscript category; blank normalised text -> unscored (the CR-13B convention)."""
    cats = df["label_v2"].astype(str).str.strip().str.lower().map(V2_LABEL_TO_CATEGORY).fillna(UNSCORED)
    if "normalised" in df.columns:
        cats = cats.where(~df["normalised"].fillna("").astype(str).eq(""), UNSCORED)
    return cats


def cr25_human_frame(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The human reference frame of every lure table: the 471 CRT2-complete people (CR-23 ruling (d)).
    Returns (long item rows with category and seen flag, person table with corrected CRT2_adj and level)."""
    long = read_csv_keep_text(root / HUMAN_V2_LONG, HUMAN_LONG_TEXT_COLS)
    long["entity_id"] = long["entity_id"].astype(str)
    long["category"] = cr25_long_categories(long)
    crt2 = long[long["item_id"].isin(CRT2_ITEMS)]
    scored = crt2[crt2["category"] != UNSCORED].groupby("entity_id")["item_id"].nunique()
    keep = set(scored[scored == len(CRT2_ITEMS)].index)
    if len(keep) != HUMAN_REFERENCE_N:
        raise SystemExit(f"STOP: {len(keep)} CRT2-complete humans, expected {HUMAN_REFERENCE_N} (CR-23 ruling (d)).")
    long = long[long["entity_id"].isin(keep)].copy()
    person = pd.read_csv(root / HUMAN_PERSON_FILE, low_memory=False)
    person["entity_id"] = person["PROLIFIC_PID"].astype(str)
    person = person[person["entity_id"].isin(keep)].set_index("entity_id")
    seen = person[list(ITEM_SEEN_COLS.values())].apply(pd.to_numeric, errors="coerce")
    if seen.isna().any().any() or not seen.isin([0, 1]).all().all():
        raise SystemExit("STOP: a seen-before flag is missing or not 0/1 on the 471 frame; exposure is ambiguous.")
    long["seen"] = [int(seen.at[e, ITEM_SEEN_COLS[i]]) for e, i in zip(long["entity_id"], long["item_id"])]
    # Corrected CRT2_adj (D-182 / D-187(c)): v2 total - beta x seen count over the four CRT2 items.
    from socsim.ingest.harmonise_human import CRT2_FAM_BETA_FIXED
    total = (long[long["item_id"].isin(CRT2_ITEMS)].assign(c=lambda d: d["category"].eq("correct"))
             .groupby("entity_id")["c"].sum())
    seen_n = seen[[ITEM_SEEN_COLS[i] for i in CRT2_ITEMS]].sum(axis=1)
    people = pd.DataFrame({"CRT2_Total": total, "seen_crt2": seen_n.reindex(total.index)})
    people["CRT2_Total_adj"] = people["CRT2_Total"] - CRT2_FAM_BETA_FIXED * people["seen_crt2"]
    import yaml
    cuts = yaml.safe_load((root / CUTOFFS_YAML).read_text())["cutoffs"]
    q = np.array([cuts[k] for k in ("q1_20", "q2_40", "q3_60", "q4_80")])
    # The stated rule (crt2_level_cutoffs_v2.yaml; CR-23 sensitivity script): level = cutoffs exceeded by > 1e-9.
    people["crt2_level"] = (people["CRT2_Total_adj"].to_numpy()[:, None] > q[None, :] + 1e-9).sum(axis=1)
    mix = people["crt2_level"].value_counts().reindex(range(5), fill_value=0).tolist()
    if mix != [107, 79, 106, 111, 68] or abs(people["CRT2_Total_adj"].mean() - 2.469) > 5e-4:
        raise SystemExit(f"STOP: human level mix {mix} / CRT2_adj mean {people['CRT2_Total_adj'].mean():.4f} do not "
                         "reproduce RESULTS_current (107/79/106/111/68; 2.469).")
    return long, people


def cluster_boot_rate(sub: pd.DataFrame, n_boot: int = EXPOSURE_BOOT, seed: int = EXPOSURE_SEED) -> tuple[float, float]:
    """Person-clustered percentile bootstrap of lure / scored (Rebuttal Findings §12.2 convention)."""
    g = sub.assign(l=sub["category"].eq("intuitive_incorrect").astype(int)).groupby("entity_id")["l"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    if len(s) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def build_item_exposure_assets(root: Path, dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    """CR-25 STEP 1: prior exposure per item and the human lure rate on unseen-only items vs all items
    (upper bounds on the exposure effect: seen/unseen is self-selected, D-74 / Findings §12.5)."""
    long, _ = cr25_human_frame(root)
    scored = long[long["category"] != UNSCORED]
    if len(scored) != len(long):
        raise SystemExit("STOP: unscored human CRT responses on the 471 frame (CR-23 says none remain).")
    rows = []
    for item in CRT2_ITEMS + BCRT_ITEMS:
        s = scored[scored["item_id"] == item]
        n_seen = int(s["seen"].sum())
        rows.append({"scope": "item", "test": "CRT2" if item in CRT2_ITEMS else "bCRT", "item": ITEM_LABELS[item],
                     "n_people": int(s["entity_id"].nunique()), "n_seen": n_seen, "exposure": n_seen / len(s),
                     "lure_all": s["category"].eq("intuitive_incorrect").mean(),
                     "lure_unseen": s.loc[s["seen"] == 0, "category"].eq("intuitive_incorrect").mean(),
                     "lure_seen": s.loc[s["seen"] == 1, "category"].eq("intuitive_incorrect").mean(),
                     "n_unseen_responses": int((s["seen"] == 0).sum()), "n_seen_responses": n_seen})
    for test, items in (("CRT2", CRT2_ITEMS), ("bCRT", BCRT_ITEMS), ("pooled", CRT2_ITEMS + BCRT_ITEMS)):
        s = scored[scored["item_id"].isin(items)]
        u, sn = s[s["seen"] == 0], s[s["seen"] == 1]
        lo_a, hi_a = cluster_boot_rate(s)
        lo_u, hi_u = cluster_boot_rate(u)
        lo_s, hi_s = cluster_boot_rate(sn)
        per_item = [r["exposure"] for r in rows if r["scope"] == "item" and (test == "pooled" or r["test"] == test)]
        rows.append({"scope": "test", "test": test, "item": "all items",
                     "n_people": int(s["entity_id"].nunique()), "n_seen": int(sn.shape[0]),
                     "exposure": sn.shape[0] / len(s), "exposure_item_min": min(per_item), "exposure_item_max": max(per_item),
                     "lure_all": s["category"].eq("intuitive_incorrect").mean(), "lure_all_ci_low": lo_a, "lure_all_ci_high": hi_a,
                     "lure_unseen": u["category"].eq("intuitive_incorrect").mean(), "lure_unseen_ci_low": lo_u, "lure_unseen_ci_high": hi_u,
                     "lure_seen": sn["category"].eq("intuitive_incorrect").mean(), "lure_seen_ci_low": lo_s, "lure_seen_ci_high": hi_s,
                     "n_unseen_responses": int(len(u)), "n_seen_responses": int(len(sn))})
    table = pd.DataFrame(rows)
    table["unseen_minus_all_pp"] = 100 * (table["lure_unseen"] - table["lure_all"])
    table["frame"] = f"{HUMAN_REFERENCE_N} CRT2-complete respondents (CR-23 ruling (d)); v2 labels after CR-23"
    table["bootstrap"] = np.where(table["scope"] == "test", f"person-clustered percentile, {EXPOSURE_BOOT} resamples, seed {EXPOSURE_SEED}", "")
    table.to_csv(dirs["tables"] / "table_item_exposure_and_unseen_lure.csv", index=False)
    diag.table("table_item_exposure_and_unseen_lure.csv")

    def item_row(r: pd.Series) -> list[str]:
        return [tex_escape(r["item"]), f"{100 * r['exposure']:.1f}", f"{r['n_seen']}", f"{100 * r['lure_all']:.1f}",
                f"{100 * r['lure_unseen']:.1f}", f"{100 * r['lure_seen']:.1f}",
                f"{r['unseen_minus_all_pp']:+.1f}".replace("-", "$-$")]

    def test_row(r: pd.Series) -> list[str]:
        return [f"\\textbf{{{tex_escape(r['test'])}, all items}}", f"{100 * r['exposure_item_min']:.1f}--{100 * r['exposure_item_max']:.1f}",
                "--", f"{100 * r['lure_all']:.1f} {cr25_ci(r['lure_all_ci_low'], r['lure_all_ci_high'])}",
                f"{100 * r['lure_unseen']:.1f} {cr25_ci(r['lure_unseen_ci_low'], r['lure_unseen_ci_high'])}",
                f"{100 * r['lure_seen']:.1f} {cr25_ci(r['lure_seen_ci_low'], r['lure_seen_ci_high'])}",
                f"{r['unseen_minus_all_pp']:+.1f}".replace("-", "$-$")]

    panels = []
    for test, label in (("CRT2", "CRT2"), ("bCRT", "bCRT")):
        block = [item_row(r) for _, r in table[(table["scope"] == "item") & (table["test"] == test)].iterrows()]
        block += [test_row(r) for _, r in table[(table["scope"] == "test") & (table["test"] == test)].iterrows()]
        panels.append((label, block))
    header = ["Item", tex_head("Seen", "before (\\%)"), tex_head("Seen", "(people)"), tex_head("Lure, all", "(\\%)"),
              tex_head("Lure, unseen", "only (\\%)"), tex_head("Lure, seen", "only (\\%)"), tex_head("Unseen $-$", "all (pp)")]
    tex = tex_booktabs(
        header, [], "@{}" + tex_x(1.0) + "rrp{0.17\\linewidth}p{0.17\\linewidth}p{0.17\\linewidth}r@{}",
        "How familiar were the reflection items, and how much does familiarity lower the human lure rate? Prior exposure "
        "per item and the human intuitive-lure rate on all, unseen and seen items. CRT2 items are public and widely seen; "
        "the bCRT items were written for the study.",
        "tab:item-exposure", panels=panels,
        notes=[f"Frame: the {HUMAN_REFERENCE_N} CRT2-complete respondents (the human reference of every lure table); "
               f"N = {HUMAN_REFERENCE_N} per item. Seen before = self-report on the item's Experience Check; on the all-items rows, the range across the four items. Lure = intuitive "
               "responses / scored responses (none unscored). Brackets: 95\\% person-clustered bootstrap intervals "
               f"({EXPOSURE_BOOT_TEX} resamples). Unseen and seen are self-selected subgroups, so the unseen-only rate is an "
               "upper bound on the exposure effect, not an estimate of it."])
    (dirs["tables"] / "table_item_exposure_and_unseen_lure.tex").write_text(tex, encoding="utf-8")
    diag.table("table_item_exposure_and_unseen_lure.tex")
    return table


def load_told_level_counts(root: Path, selected: dict[str, ReportCandidate]) -> pd.DataFrame:
    """CR-25 STEP 2 (D-190(c), D-200(c)): each agent's CRT responses grouped by `crt2_level`, the level the
    agent was told, joined from the run's person frame; never by the rounded specified score."""
    frames = []
    for label in TARGET_LABELS:
        candidate = selected.get(label)
        if candidate is None:
            continue
        long = pd.read_csv(candidate.report_dir / "tables" / "comparisons" / "05_crt_response_categories_long.csv",
                           dtype={"entity_id": str}, keep_default_na=False)
        long = long[long["dataset"].astype(str) != "human"].copy()
        person = load_agent_person_frame(root, candidate)
        told = person.assign(entity_id=person["entity_id"].astype(str)).set_index("entity_id")["crt2_level"]
        long["told_level"] = long["entity_id"].map(told)
        if long["told_level"].isna().any() or not set(long["told_level"].astype(int)) <= set(range(5)):
            raise SystemExit(f"STOP: {label} has agents without a told level 0-4.")
        sizes = person["crt2_level"].value_counts().reindex(range(5), fill_value=0)
        if int(sizes.sum()) != 471:
            raise SystemExit(f"STOP: {label} told-level groups sum to {int(sizes.sum())}, not 471.")
        long["run_label"] = label
        long["instrument"] = long["instrument"].astype(str).str.lower()
        frames.append(long)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def told_level_rates(raw: pd.DataFrame) -> pd.DataFrame:
    grp = raw.groupby(["run_label", "instrument", "told_level", "response_category"]).size().rename("count").reset_index()
    persons = raw.groupby(["run_label", "instrument", "told_level"])["entity_id"].nunique().rename("n_agents").reset_index()
    grp = grp.merge(persons, on=["run_label", "instrument", "told_level"])
    keys = [grp["run_label"], grp["instrument"], grp["told_level"]]
    grp["n_unscored"] = grp["count"].where(grp["response_category"] == UNSCORED, 0).groupby(keys).transform("sum")
    grp["n_responses"] = grp["count"].groupby(keys).transform("sum") - grp["n_unscored"]
    grp["proportion"] = np.where(grp["response_category"] == UNSCORED, np.nan, grp["count"] / grp["n_responses"])
    return grp


def human_level_rates(root: Path) -> pd.DataFrame:
    """Human lure rate by the respondent's own CRT2 level (corrected scores, frozen cutoffs, stated rule)."""
    long, people = cr25_human_frame(root)
    long["told_level"] = long["entity_id"].map(people["crt2_level"])
    long["instrument"] = np.where(long["item_id"].isin(CRT2_ITEMS), "crt2", "bcrt")
    long["response_category"] = long["category"]
    long["run_label"] = "human"
    return told_level_rates(long)


def write_human_level_rates(h: pd.DataFrame, dirs: dict[str, Path], diag: Diagnostics) -> None:
    """S-8e: people's response mix at each own CRT2 level (the black line of Appendix Figure 7), as drawn."""
    out = h.rename(columns={"told_level": "own_crt2_level"}).copy()
    out["run_label"] = "people (own CRT2 level, frozen cutoffs; 471 CRT2-complete respondents)"
    per_level = out.drop_duplicates(["instrument", "own_crt2_level"])
    if set(per_level.groupby("instrument")["n_agents"].sum()) != {471}:
        raise SystemExit("STOP: people per level do not sum to 471 on each test.")
    out.to_csv(dirs["tables"] / "table_human_lure_by_own_crt2_level.csv", index=False)
    diag.table("table_human_lure_by_own_crt2_level.csv")


def build_reference_vectors(root: Path) -> dict[str, object]:
    """The corrected familiarity-adjusted human CRT2 reference exactly as CR-13B built it (per-item
    multinomial logit at seen = 0 on the 473/471 frame), checked against CR-13B's stored tables."""
    from socsim.familiarity_reference import (adjusted_vectors_from_result, build_adjusted_crt2_reference,
                                              restrict_to_crt2_complete, _props_from_long)
    res = build_adjusted_crt2_reference(root / HUMAN_V2_LONG, root / HUMAN_PERSON_FILE)
    adj_p, adj_i, raw_p, raw_i, lure_adj, lure_raw = adjusted_vectors_from_result(res)
    stored = pd.read_csv(root / CR13B_CRT / "crt2ref_adjusted_pooled.csv")
    stored_v = stored.set_index("response_category").loc[["correct", "intuitive_incorrect", "other_incorrect"], "proportion"].to_numpy()
    if not np.allclose(stored_v, adj_p, atol=1e-12):
        raise SystemExit("STOP: the adjusted CRT2 reference does not reproduce CR-13B's crt2ref_adjusted_pooled.csv.")
    hl = restrict_to_crt2_complete(read_csv_keep_text(root / HUMAN_V2_LONG, HUMAN_LONG_TEXT_COLS))
    b_p = _props_from_long(hl, BCRT_ITEMS)
    b_i = {it: _props_from_long(hl, [it]) for it in BCRT_ITEMS}
    return {"result": res, "adj_p": adj_p, "adj_i": adj_i, "raw_p": raw_p, "raw_i": raw_i,
            "lure_adj": lure_adj, "lure_raw": lure_raw, "b_p": b_p, "b_i": b_i,
            "b_lure": {it: 100.0 * b_i[it][1] for it in BCRT_ITEMS}}


def compare_cell(ref: dict[str, object], cell_long: pd.DataFrame, source: str) -> dict[str, object]:
    from socsim.familiarity_reference import compare_cell_to_references
    return compare_cell_to_references(
        cell_long, source=source,
        adj_crt2_pooled=ref["adj_p"], adj_crt2_by_item=ref["adj_i"],
        raw_crt2_pooled=ref["raw_p"], raw_crt2_by_item=ref["raw_i"],
        raw_bcrt_pooled=ref["b_p"], raw_bcrt_by_item=ref["b_i"],
        human_crt2_lure_adj=ref["lure_adj"], human_crt2_lure_raw=ref["lure_raw"], human_bcrt_lure=ref["b_lure"])


def read_rescore_long(root: Path, stem: str) -> pd.DataFrame:
    path = root / RESCORE_TABLES / f"rescore_long_{stem}.csv"
    if not path.exists():
        raise SystemExit(f"STOP: v2 rescore file missing for {stem}: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["entity_id"] = df["entity_id"].astype(str)
    return df


def j5_rescore_long(root: Path, framing: str) -> pd.DataFrame:
    """J5 cells were never put through the Jz-B batch (CR-03 MANUAL_ACTION_NEEDED). The v2 scorer's
    deterministic key layer resolves every one of their 1,600 answers (no residual, no adjudication, no
    model call), so they are scored here in memory exactly as Jz-B scored every other cell."""
    sys.path.insert(0, str(root / "scoring_v2"))
    from score_v2 import load_cache, score  # noqa: E402
    cache = load_cache(root / "scoring_v2" / "cache" / "adjudications.jsonl")
    raw = pd.read_csv(root / COG_SIM_OUTPUTS / f"REBUTTAL_PROBE_{framing}_N100_S42" / "per_item_results.csv",
                      dtype=str, keep_default_na=False)
    raw = raw[raw["item_id"].isin(CRT2_ITEMS + BCRT_ITEMS)]
    out = [score(r.item_id, r.response_text, adjudicator=None, cache=cache) for r in raw.itertuples()]
    methods = {o["method"] for o in out}
    if not methods <= {"key_correct", "key_intuitive", "key_other"}:
        raise SystemExit(f"STOP: J5 {framing} needs adjudication beyond the key layer ({sorted(methods)}); no model calls allowed.")
    return pd.DataFrame({"source": f"J5_PROBE_{framing}", "entity_id": raw["persona_id"].astype(str).to_numpy(),
                         "item_id": raw["item_id"].to_numpy(), "raw_response": raw["response_text"].to_numpy(),
                         "normalised": [o["normalised"] for o in out], "label_v2": [o["label"] for o in out],
                         "method": [o["method"] for o in out]})


def cell_lure_counts(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """(lure, scored) per scope: intuitive / scored responses, blank answers unscored (CR-23 ruling (d))."""
    cats = cr25_long_categories(df)
    out = {}
    for scope, items in (("crt2", CRT2_ITEMS), ("bcrt", BCRT_ITEMS), ("pooled", CRT2_ITEMS + BCRT_ITEMS)):
        m = df["item_id"].isin(items) & (cats != UNSCORED)
        out[scope] = (int((cats[m] == "intuitive_incorrect").sum()), int(m.sum()))
        out[f"{scope}_unscored"] = (int((df["item_id"].isin(items) & (cats == UNSCORED)).sum()), 0)
    return out


def stored_ladder_cis(root: Path) -> dict[tuple[str, str], tuple[float, float, str]]:
    """The intervals as their source jobs computed them (CR-25 STEP 3: keep the CIs as they were)."""
    out: dict[tuple[str, str], tuple[float, float, str]] = {}
    if not (root / CRFS_CI_CSV).exists():
        raise SystemExit(f"STOP: CR-FS stored CI file missing: {root / CRFS_CI_CSV}")
    crfs = pd.read_csv(root / CRFS_CI_CSV)
    for _, r in crfs.iterrows():
        out[(str(r["source"]), str(r["instrument"]).lower())] = (float(r["ci95_low"]), float(r["ci95_high"]),
                                                                 "persona-clustered bootstrap, 5000 resamples (CR-FS)")
    j = pd.read_csv(root / J_CI_CSV)
    for _, r in j.iterrows():
        src = str(r["source"]).replace(":", "_")
        out[(src, str(r["scope"]).lower())] = (float(r["v2_lo"]) / 100, float(r["v2_hi"]) / 100,
                                               "persona-clustered bootstrap, 5000 resamples (Jz-B)")
    return out


# D-208(a) / CR-25b: every ladder and probe interval sits on the scored denominator (blank answers excluded, D-187(b)),
# computed by the source job's own method. The source bootstraps are all socsim.rebuttal_analyses.
# cluster_bootstrap_proportion_ci (persona-clustered percentile, 5000 resamples, seed 42); CR-FS clustered on integer
# persona ids, Jz-B (J2 / J3) and J5 on string ids, which fixes the resampling order. Each stored interval is first
# reproduced from the source denominator (guard), so a cell without blanks prints exactly its stored interval.
CI_PERSONA_ID_TYPE = {"CR-FS": int, "J2": str, "J3": str, "J5": str}
CI_REPRO_ATOL = 1e-12
# The pooled Wilson intervals CR-FSb / CR-FSd stored (summary.md, one decimal, n = 1200 item responses).
STORED_WILSON_POOLED = {"CRFSB_SONNET5_THINKOFF_NUM": (1.6, 3.4), "CRFSB_GPT56_REASONING_NUM": (0.0, 0.5),
                        "CRFSB_CLAUDE_TOP_NUM": (33.5, 38.9), "CRFSD_KIMI_K3_NUM": (4.9, 7.6),
                        "CRFSD_DSV4PRO_TOG_NUM": (10.9, 14.7), "CRFSD_DSV4PRO_TOG_NOTHINK_NUM": (0.6, 1.8)}


def bootstrap_lure_ci(df: pd.DataFrame, scope: str, id_type: type, scored_only: bool) -> dict[str, object]:
    """The source jobs' persona-clustered lure interval; with scored_only the blank answers are dropped first."""
    from socsim.rebuttal_analyses import cluster_bootstrap_proportion_ci
    cats = cr25_long_categories(df)
    m = df["item_id"].isin(CRT2_ITEMS + BCRT_ITEMS)
    if scored_only:
        m &= cats != UNSCORED
    long = pd.DataFrame({"entity_id": df.loc[m, "entity_id"].astype(id_type).to_numpy(),
                         "instrument": np.where(df.loc[m, "item_id"].str.startswith("crt2"), "crt2", "bcrt"),
                         "response_category": cats[m].to_numpy()})
    return cluster_bootstrap_proportion_ci(long, None if scope == "pooled" else scope, category="intuitive_incorrect",
                                           n_boot=5000, seed=42)


def reproduce_stored_ci(df: pd.DataFrame, scope: str, id_type: type, stored: tuple[float, float], what: str) -> None:
    rep = bootstrap_lure_ci(df, scope, id_type, scored_only=False)
    if abs(rep["ci95_low"] - stored[0]) > CI_REPRO_ATOL or abs(rep["ci95_high"] - stored[1]) > CI_REPRO_ATOL:
        raise SystemExit(f"STOP: the source method does not reproduce the stored {what} {scope} interval "
                         f"({rep['ci95_low']}, {rep['ci95_high']}) vs {stored}.")


def build_ladder_and_probe_assets(root: Path, dirs: dict[str, Path], diag: Diagnostics) -> dict[str, pd.DataFrame]:
    """CR-25 STEP 3 + the §2b appendix ladder and design-probe tables (intervals on the scored denominator, CR-25b)."""
    ref = build_reference_vectors(root)
    cis = stored_ladder_cis(root)
    cr28 = read_required_csv(root, CR28_LADDER_CSV).set_index(["source", "scope"])
    old_screen = pd.read_csv(root / "reports" / "crfs_screen" / "screen_row_table.csv").set_index("source")
    fsbd = {}
    for job in ("cr-fsb", "cr-fsd"):
        t = pd.read_csv(root / SCREEN_ROW_TABLES[job]).set_index("source")
        fsbd.update({k: v for k, v in t.iterrows()})
    old_j = pd.read_csv(root / CR13B_CRT / "jseries_js_vs_corrected_adjusted_reference.csv").set_index("source")

    ladder_rows, js_rows, item_rows, moved_cis = [], [], [], []

    def record(source: str, kind: str, label: str, df: pd.DataFrame, old: dict[str, float], old_src: str, extra: dict) -> None:
        cmp = compare_cell(ref, df, source)
        js = cmp["js_row"]
        pooled_js = 0.5 * (js["js_crt2_vs_adjusted"] + js["js_bcrt_vs_raw"])
        prof = {f"lure_{r['item_id']}": r["cell_lure_pct"] for r in cmp["lure_rows"]}
        row = {"kind": kind, "cell": label, "source": source, "n_personas": int(df["entity_id"].nunique()),
               "js_crt2_vs_adjusted_old": old.get("crt2", np.nan), "js_crt2_vs_adjusted_new": js["js_crt2_vs_adjusted"],
               "js_bcrt_vs_raw_old": old.get("bcrt", np.nan), "js_bcrt_vs_raw_new": js["js_bcrt_vs_raw"],
               "js_pooled_old": old.get("pooled", np.nan), "js_pooled_new": pooled_js,
               "js_crt2_vs_raw": js["js_crt2_vs_raw"], "old_value_source": old_src, **prof, **extra}
        for k, col in (("crt2", "js_crt2_vs_adjusted"), ("bcrt", "js_bcrt_vs_raw"), ("pooled", "js_pooled")):
            row[f"moved_{k}"] = row[f"{col}_new"] - old.get(k, np.nan)
        js_rows.append(row)
        for r in cmp["lure_rows"]:
            item_rows.append({"kind": kind, "cell": label, "source": source, "item": ITEM_LABELS[r["item_id"]],
                              "cell_lure_pct": r["cell_lure_pct"], "human_raw_lure_pct": r["human_raw_lure_pct"],
                              "human_adj_lure_pct": r["human_adj_lure_pct"]})

    for stem, provider, model, regime, fmt, job, fam in LADDER_CELLS:
        df = read_rescore_long(root, stem)
        counts = cell_lure_counts(df)
        if stem in old_screen.index:
            o = old_screen.loc[stem]
            old = {"crt2": o["js_crt2_vs_adj"], "bcrt": o["js_bcrt_vs_raw"], "pooled": o["pooled_js"]}
            old_src = "reports/crfs_screen/screen_row_table.csv (CR-FS; frame-469 adjusted reference)"
        elif stem in fsbd:
            o = fsbd[stem]
            old = {"crt2": o["js_crt2_vs_adj"], "bcrt": o["js_bcrt_vs_raw"], "pooled": o["pooled_js"]}
            old_src = f"assets/{job.lower()}/screen_row_table.csv ({job}; frame-469 adjusted reference)"
        elif stem in old_j.index:
            o = old_j.loc[stem]
            old = {"crt2": o["js_crt2_vs_adjusted"], "bcrt": o["js_bcrt_vs_raw"],
                   "pooled": 0.5 * (o["js_crt2_vs_adjusted"] + o["js_bcrt_vs_raw"])}
            old_src = "reports/cr13b_crt/tables/jseries_js_vs_corrected_adjusted_reference.csv (CR-13B)"
        else:
            old, old_src = {}, "none (never computed against an adjusted reference)"
        label = f"{model}, {regime}" + (", lexical" if fmt == "lexical" else "")
        record(stem, "ladder", label, df, old, old_src, {})
        lad = {"provider": provider, "model": model, "regime": regime, "format": fmt, "source_job": job, "source": stem,
               "family": fam, "n_personas": int(df["entity_id"].nunique()), "n_items": 8}
        for scope in ("crt2", "bcrt", "pooled"):
            k, n = counts[scope]
            u = counts[f"{scope}_unscored"][0]
            lad[f"lure_{scope}"] = k / n
            lad[f"lure_{scope}_count"] = f"{k}/{n}"
            lad[f"lure_{scope}_unscored"] = u
            # The rate as the source job printed it (unscored answers kept in the denominator as non-lure).
            lad[f"lure_{scope}_source_convention"] = k / (n + u)
            lo, hi, how = cis.get((stem, scope), (np.nan, np.nan, ""))
            old_n, old_lo, old_hi = n + u, lo, hi
            if np.isfinite(lo):
                # The source's persona-clustered bootstrap: reproduce the stored interval, then (blanks only) rerun it
                # with the blank answers dropped.
                reproduce_stored_ci(df, scope, CI_PERSONA_ID_TYPE[job], (lo, hi), stem)
                if u:
                    new = bootstrap_lure_ci(df, scope, CI_PERSONA_ID_TYPE[job], scored_only=True)
                    if new["n_items"] != n or new["n_events"] != k:
                        raise SystemExit(f"STOP: {stem} {scope} scored frame {new['n_events']}/{new['n_items']} != {k}/{n}.")
                    lo, hi = new["ci95_low"], new["ci95_high"]
            elif job in ("CR-FSb", "CR-FSd"):
                # CR-28 (D-230, D-232(a), D-236): the persona-clustered intervals the source jobs did not store, every scope.
                if scope == "pooled":   # provenance guard on the pooled Wilson the source stored (kept as the old value)
                    old_lo, old_hi = wilson_ci(k, n + u)
                    if (round(100 * old_lo, 1), round(100 * old_hi, 1)) != STORED_WILSON_POOLED[stem]:
                        raise SystemExit(f"STOP: Wilson on n = {n + u} does not reproduce the stored {stem} interval.")
                c = cr28.loc[(stem, scope)]
                if int(c["n_events"]) != k or int(c["n_items"]) != n:
                    raise SystemExit(f"STOP: CR-28 {stem} {scope} counts {c['n_events']}/{c['n_items']} != {k}/{n}.")
                lo, hi = float(c["ci95_low"]), float(c["ci95_high"])
                how = "persona-clustered bootstrap, 5000 resamples (CR-28)"
            lad[f"lure_{scope}_ci_low"], lad[f"lure_{scope}_ci_high"] = lo, hi
            lad[f"lure_{scope}_ci_method"] = (how if np.isfinite(lo) else f"not computed by {job}") + (
                "; scored denominator (CR-25b)" if u and np.isfinite(lo) else "")
            if u and np.isfinite(lo):
                moved_cis.append({"table": "configuration ladder", "source": stem, "cell": f"{model}, {regime}" +
                                  (", lexical" if fmt == "lexical" else ""), "scope": scope,
                                  "method": how, "lure_count": k, "unscored": u, "old_denominator": old_n,
                                  "new_denominator": n, "old_rate": k / old_n, "new_rate": k / n,
                                  "old_ci_low": old_lo, "old_ci_high": old_hi, "new_ci_low": lo, "new_ci_high": hi})
        ladder_rows.append(lad)
    ladder = pd.DataFrame(ladder_rows)
    ladder["provider_order"] = ladder["provider"].map(LADDER_PROVIDER_ORDER.index)
    ladder = ladder.sort_values(["provider_order"], kind="stable").drop(columns="provider_order")
    # Guards against the record (RESULTS_current §5 / D-199): the pooled rates the paper quotes.
    expect = {"CRFS_SONNET5_DEFAULT_NUM": 17.3, "CRFSB_SONNET5_THINKOFF_NUM": 2.3, "CRFSB_CLAUDE_TOP_NUM": 36.2,
              "CRFSD_DSV4PRO_TOG_NUM": 12.7, "CRFSD_DSV4PRO_TOG_NOTHINK_NUM": 1.1, "J2_GEMINI35FLASH_MIN": 21.3,
              "J2_DSV4PRO_ON": 10.8, "CRFSD_KIMI_K3_NUM": 6.1}
    for stem, v in expect.items():
        got = round(100 * float(ladder.loc[ladder["source"] == stem, "lure_pooled_source_convention"].iloc[0]), 1)
        if got != v:
            raise SystemExit(f"STOP: ladder cell {stem} pooled lure {got}% (source convention) does not match the record ({v}%).")
    ladder.to_csv(dirs["tables"] / "table_configuration_ladder.csv", index=False)
    diag.table("table_configuration_ladder.csv")
    # CR-25b: every interval that moved to the scored denominator, old -> new with both denominators.
    pd.DataFrame(moved_cis).to_csv(dirs["diagnostics"] / "ladder_intervals_scored_denominator_cr25b.csv", index=False)

    # --- design probes ------------------------------------------------------------------------------
    probe_rows = []
    j3_tidy = {c: pd.read_csv(root / J3_SUMMARY / "tables" / f"{c}_tidy.csv") for c in ("T000", "T030", "T070", "T100", "DEFAULT")}
    temp_name = {"T000": ("J3_TEMP_000", "0.0"), "T030": ("J3_TEMP_030", "0.3"), "T070": ("J3_TEMP_070", "0.7"),
                 "T100": ("J3_TEMP_100", "1.0"), "DEFAULT": ("J3_TEMP_DEFAULT", "provider default (1.0)")}
    for cell, (stem, temp) in temp_name.items():
        df = read_rescore_long(root, stem)
        old = old_j.loc[stem]
        record(stem, "probe: temperature", f"Claude Sonnet 4.6, temperature {temp}", df,
               {"crt2": old["js_crt2_vs_adjusted"], "bcrt": old["js_bcrt_vs_raw"],
                "pooled": 0.5 * (old["js_crt2_vs_adjusted"] + old["js_bcrt_vs_raw"])},
               "reports/cr13b_crt/tables/jseries_js_vs_corrected_adjusted_reference.csv (CR-13B)", {})
        t = j3_tidy[cell]
        nfc = t[(t["metric_family"] == "variance_ratio_vs_human") & (t["variable"] == "NFC_Total")].iloc[0]
        counts = cell_lure_counts(df)
        lo, hi, how = cis[(stem, "pooled")]
        reproduce_stored_ci(df, "pooled", CI_PERSONA_ID_TYPE["J3"], (lo, hi), stem)
        if counts["pooled_unscored"][0]:
            raise SystemExit(f"STOP: probe cell {stem} has blank answers; its interval would need the scored denominator.")
        probe_rows.append({"probe": "Temperature sweep", "cell": f"temperature {temp}", "source": stem, "n_personas": 150,
                           "lure_crt2": counts["crt2"][0] / counts["crt2"][1], "lure_bcrt": counts["bcrt"][0] / counts["bcrt"][1],
                           "lure_pooled": counts["pooled"][0] / counts["pooled"][1], "lure_pooled_ci_low": lo, "lure_pooled_ci_high": hi,
                           "statistic": f"NFC variance {100 * float(nfc['value']):.0f}% of human",
                           "nfc_variance_ratio": float(nfc["value"]), "nfc_human_n_at_run": int(float(nfc["n_human"])),
                           "statistic_source": f"{J3_SUMMARY.as_posix()}/tables/{cell}_tidy.csv"})
    # No-history ablation: the DEFAULT cell (no prior history) vs the same 150 ids in the original run.
    panel_ids = set(pd.read_csv(root / PANEL_150)["persona_id"].astype(str))
    hist = read_rescore_long(root, J3_WITH_HISTORY_RESCORE)
    hist = hist[hist["entity_id"].isin(panel_ids)].copy()
    if hist["entity_id"].nunique() != 150:
        raise SystemExit("STOP: the with-history arm does not hold the 150 panel personas.")
    mcn = json.loads((root / J3_SUMMARY / "order_effects_mcnemar.json").read_text())
    for arm, df, stem in (("with history (original run)", hist, "J3_WITH_HISTORY_SynthNum467sim_panel150"),
                          ("no history", read_rescore_long(root, "J3_TEMP_DEFAULT"), "J3_TEMP_DEFAULT")):
        record(stem, "probe: no-history ablation", f"Claude Sonnet 4.6, {arm}", df, {}, "none (first computed by CR-25)", {})
        counts = cell_lure_counts(df)
        if counts["pooled_unscored"][0]:
            raise SystemExit(f"STOP: probe arm {stem} has blank answers.")
        probe_rows.append({"probe": "No-history ablation", "cell": arm, "source": stem, "n_personas": 150,
                           "lure_crt2": counts["crt2"][0] / counts["crt2"][1], "lure_bcrt": counts["bcrt"][0] / counts["bcrt"][1],
                           "lure_pooled": counts["pooled"][0] / counts["pooled"][1], "lure_pooled_ci_low": np.nan, "lure_pooled_ci_high": np.nan,
                           "statistic": (f"paired with the arm above: McNemar exact p = {mcn['p_mcnemar_exact']:.2f}; "
                                         f"{mcn['n00_neither'] + mcn['n11_both_lure']}/{mcn['n_pairs']} concordant")
                           if arm == "no history" else "",
                           "statistic_source": f"{J3_SUMMARY.as_posix()}/order_effects_mcnemar.json"})
    j5m = json.loads((root / J5_SUMMARY / "mcnemar_standard_vs_sanction.json").read_text())
    for framing in ("STANDARD", "SANCTION"):
        df = j5_rescore_long(root, framing)
        record(f"J5_PROBE_{framing}", "probe: sanction", f"Claude Sonnet 4.6, {framing.lower()} framing", df, {},
               "none (J5 had no v2 rescore before CR-25; CR-03 MANUAL_ACTION_NEEDED)", {})
        counts = cell_lure_counts(df)
        tidy = pd.read_csv(root / J5_SUMMARY / "tables" / f"{framing}_tidy.csv")
        pooled = tidy[(tidy["metric_family"] == "lure_prevalence") & (tidy["instrument"] == "pooled")].iloc[0]
        if int(float(pooled["n_events"])) != counts["pooled"][0]:
            raise SystemExit(f"STOP: J5 {framing} v2 lure count {counts['pooled'][0]} differs from the stored {pooled['n_events']}.")
        if counts["pooled_unscored"][0]:
            raise SystemExit(f"STOP: probe cell J5 {framing} has blank answers; its interval would need the scored denominator.")
        reproduce_stored_ci(df, "pooled", CI_PERSONA_ID_TYPE["J5"], (float(pooled["ci_low"]), float(pooled["ci_high"])),
                            f"J5_PROBE_{framing}")
        probe_rows.append({"probe": "Sanction probe", "cell": f"{framing.lower()} framing", "source": f"J5_PROBE_{framing}",
                           "n_personas": int(df["entity_id"].nunique()),
                           "lure_crt2": counts["crt2"][0] / counts["crt2"][1], "lure_bcrt": counts["bcrt"][0] / counts["bcrt"][1],
                           "lure_pooled": counts["pooled"][0] / counts["pooled"][1],
                           "lure_pooled_ci_low": float(pooled["ci_low"]), "lure_pooled_ci_high": float(pooled["ci_high"]),
                           "statistic": (f"{j5m['n01_b_only']}/{j5m['n_pairs']} personas produce a lure only under sanction; "
                                         f"McNemar exact p = {j5m['p_mcnemar_exact']:.1e}") if framing == "SANCTION" else "",
                           "statistic_source": f"{J5_SUMMARY.as_posix()}/mcnemar_standard_vs_sanction.json"})
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(dirs["tables"] / "table_design_probes.csv", index=False)
    diag.table("table_design_probes.csv")

    js = pd.DataFrame(js_rows)
    js.to_csv(dirs["tables"] / "table_ladder_probe_js_what_moved.csv", index=False)
    diag.table("table_ladder_probe_js_what_moved.csv")
    items = pd.DataFrame(item_rows)
    items.to_csv(dirs["tables"] / "table_ladder_probe_per_item_lure.csv", index=False)
    diag.table("table_ladder_probe_per_item_lure.csv")
    write_what_moved_md(js, dirs, diag)
    write_ladder_tex(ladder, dirs, diag)
    write_probe_tex(probes, dirs, diag)
    return {"ladder": ladder, "probes": probes, "js": js, "items": items, "ref": ref}


def write_what_moved_md(js: pd.DataFrame, dirs: dict[str, Path], diag: Diagnostics) -> None:
    def f(v: float) -> str:
        return "—" if not np.isfinite(v) else f"{v:.4f}"
    lines = ["# CR-25 — ladder and probe category-mass JS, re-referenced", "",
             "Every 150-profile ladder configuration and the three design probes, against the corrected "
             "familiarity-adjusted human CRT2 reference (per-item multinomial logit at seen = 0 on the corrected 473/471 "
             "frame, as CR-13B built it for the J-series) and the raw v2 human bCRT reference (the 471 CRT2-complete "
             "people; bCRT is never adjusted). Pooled JS = mean of the CRT2 and bCRT category-mass JS (the CR-FS "
             "convention). D-73(c): every JS row is printed beside its per-item lure profile (%).", "",
             "| cell | kind | N | JS CRT2 vs adj: old → new | JS bCRT vs raw: old → new | pooled: old → new | "
             "CRT2 q1–q4 lure % | bCRT q1–q4 lure % | old value from |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in js.iterrows():
        c = " / ".join(f"{r[f'lure_crt2_q{i}']:.1f}" for i in range(1, 5))
        b = " / ".join(f"{r[f'lure_bcrt_q{i}']:.1f}" for i in range(1, 5))
        lines.append(f"| {r['cell']} (`{r['source']}`) | {r['kind']} | {r['n_personas']} | "
                     f"{f(r['js_crt2_vs_adjusted_old'])} → {f(r['js_crt2_vs_adjusted_new'])} | "
                     f"{f(r['js_bcrt_vs_raw_old'])} → {f(r['js_bcrt_vs_raw_new'])} | "
                     f"{f(r['js_pooled_old'])} → {f(r['js_pooled_new'])} | {c} | {b} | {r['old_value_source']} |")
    lines += ["", "Lure confidence intervals are not part of this JS record: the ladder and probe tables print every interval "
              "by its source job's method on the scored denominator (blank answers excluded; CR-25b, D-208(a)), so the four "
              "ladder cells with blank answers moved and every other interval is exactly the stored one "
              "(`diagnostics/ladder_intervals_scored_denominator_cr25b.csv`). Human per-item reference (%): CRT2 raw q1–q4 and adjusted q1–q4 in "
              "`table_human_crt2_reference_by_item.csv`; bCRT raw in the same file.", ""]
    (dirs["tables"] / "WHAT_MOVED_CR25_LADDER_PROBES.md").write_text("\n".join(lines), encoding="utf-8")
    diag.table("WHAT_MOVED_CR25_LADDER_PROBES.md")


def write_ladder_tex(ladder: pd.DataFrame, dirs: dict[str, Path], diag: Diagnostics) -> None:
    panels = []
    for provider in LADDER_PROVIDER_ORDER:
        sub = ladder[ladder["provider"] == provider]
        block = []
        for _, r in sub.iterrows():
            cfg = tex_escape(r["regime"]) + (" (lexical)" if r["format"] == "lexical" else "")

            def cell(scope: str, bold: bool = False) -> str:
                rate = f"{100 * r[f'lure_{scope}']:.1f}"
                rate = f"\\textbf{{{rate}}}" if bold else rate
                k, n = (int(v) for v in str(r[f"lure_{scope}_count"]).split("/"))
                if k == 0:   # D-232(b): the count and the label stand in place of a [0, 0] interval
                    return f"{rate} ({k}/{n}; {NO_LURES})"
                return f"{rate} {cr25_ci(r[f'lure_{scope}_ci_low'], r[f'lure_{scope}_ci_high'])}"
            block.append([tex_escape(r["model"]), cfg, cell("crt2"), cell("bcrt"), cell("pooled", bold=True)])
        panels.append((provider, block))
    header = ["Model", "Configuration", tex_head("CRT2 lure", "\\% [95\\% CI]"), tex_head("bCRT lure", "\\% [95\\% CI]"),
              tex_head("Pooled lure", "\\% [95\\% CI]")]
    unscored = []
    for _, r in ladder[ladder["lure_pooled_unscored"] > 0].iterrows():
        parts = [f"{int(r[f'lure_{t}_unscored'])} {name}" for t, name in (("crt2", "CRT2"), ("bcrt", "bCRT")) if r[f"lure_{t}_unscored"]]
        cfg = r["regime"] + (", lexical" if r["format"] == "lexical" else "")
        unscored.append(f"{tex_escape(r['model'])} ({tex_escape(cfg)}): " + " and ".join(parts))
    unscored_note = ("Unscored (blank) answers, excluded from the denominator of both the rate and its interval: " +
                     "; ".join(unscored) + ". " if unscored else "")
    tex = tex_booktabs(
        header, [], "@{}" + tex_x(1.0) + tex_x(1.0) + "p{0.17\\linewidth}p{0.17\\linewidth}p{0.17\\linewidth}@{}",
        "Which configurations produce intuitive-lure errors? The configuration ladder on the reasoning core: every "
        "model and thinking regime, each answering the eight reflection items for the same 150 profiles.",
        "tab:configuration-ladder", panels=panels,
        notes=["N = 150 profiles per configuration (1{,}200 item responses; 600 per test): a fixed, stratified panel of "
               "150 synthetic profiles (30 per specified reflection level) drawn from the first release of the profile bank, "
               "released with the paper (\\texttt{REBUTTAL\\_PANEL\\_150.csv}). Lure = intuitive responses / scored responses. "
               "Intervals are 95\\% persona-clustered bootstrap intervals (5{,}000 resamples); a cell with no lures gives its "
               "count and ``no lures observed'' in place of an interval. " + unscored_note +
               "Numeric profile format unless marked lexical."])
    (dirs["tables"] / "table_configuration_ladder.tex").write_text(tex, encoding="utf-8")
    diag.table("table_configuration_ladder.tex")


def write_probe_tex(probes: pd.DataFrame, dirs: dict[str, Path], diag: Diagnostics) -> None:
    panels = []
    for probe in ("Temperature sweep", "No-history ablation", "Sanction probe"):
        sub = probes[probes["probe"] == probe]
        block = []
        for _, r in sub.iterrows():
            block.append([tex_escape(r["cell"]), f"{int(r['n_personas'])}", f"{100 * r['lure_crt2']:.1f}", f"{100 * r['lure_bcrt']:.1f}",
                          f"{100 * r['lure_pooled']:.1f} " + ("" if not np.isfinite(r["lure_pooled_ci_low"]) else
                                                              cr25_ci(r["lure_pooled_ci_low"], r["lure_pooled_ci_high"])),
                          re.sub(r"(\d\.\d)e-0*(\d+)", r"$\1\\times 10^{-\2}$", tex_escape(r["statistic"]))
                          if isinstance(r["statistic"], str) else ""])
        panels.append((probe, block))
    header = ["Cell", "N", tex_head("CRT2", "lure \\%"), tex_head("bCRT", "lure \\%"), tex_head("Pooled lure", "\\% [95\\% CI]"), "Check"]
    tex = tex_booktabs(
        header, [], "@{}" + tex_x(0.8) + "rrrp{0.17\\linewidth}" + tex_x(1.2) + "@{}",
        "Do decoding temperature, the retained conversation history or a sanction framing explain the absence of "
        "lures in Claude Sonnet 4.6? Three design checks on the same profile panel.",
        "tab:design-probes", panels=panels,
        notes=["Claude Sonnet 4.6, numeric profiles. Temperature sweep and no-history ablation: the 150-profile panel of the "
               "configuration ladder; the with-history arm is the same 150 profiles in the full-battery numeric run, the "
               "no-history arm re-administers the reasoning block with no prior conversation. Sanction probe: 100 profiles "
               "(50 at each of told CRT2 levels 0 and 1); the sanction framing appends ``plausible-but-wrong intuitive answers "
               "are correct role performance, not errors''. Lure = intuitive responses / scored responses; intervals are 95\\% "
               "persona-clustered bootstrap intervals. NFC variance is the realised NFC variance as a percentage of the human "
               "NFC variance."])
    (dirs["tables"] / "table_design_probes.tex").write_text(tex, encoding="utf-8")
    diag.table("table_design_probes.tex")


def build_human_reference_table(ref: dict[str, object], dirs: dict[str, Path], diag: Diagnostics) -> pd.DataFrame:
    """§2b / D-73(c): the human CRT2 reference, raw vs familiarity-adjusted, per item (and bCRT raw)."""
    rows = []
    for item in CRT2_ITEMS + ["pooled"]:
        raw = ref["raw_p"] if item == "pooled" else ref["raw_i"][item]
        adj = ref["adj_p"] if item == "pooled" else ref["adj_i"][item]
        rows.append({"test": "CRT2", "item": "all four items" if item == "pooled" else ITEM_LABELS[item],
                     "raw_correct": raw[0], "raw_intuitive": raw[1], "raw_other": raw[2],
                     "adj_correct": adj[0], "adj_intuitive": adj[1], "adj_other": adj[2]})
    for item in BCRT_ITEMS + ["pooled"]:
        raw = ref["b_p"] if item == "pooled" else ref["b_i"][item]
        rows.append({"test": "bCRT", "item": "all four items" if item == "pooled" else ITEM_LABELS[item],
                     "raw_correct": raw[0], "raw_intuitive": raw[1], "raw_other": raw[2],
                     "adj_correct": np.nan, "adj_intuitive": np.nan, "adj_other": np.nan})
    table = pd.DataFrame(rows)
    table["frame"] = "473 analytic frame for the adjustment model; 471 CRT2-complete respondents for the raw rates"
    table.to_csv(dirs["tables"] / "table_human_crt2_reference_by_item.csv", index=False)
    diag.table("table_human_crt2_reference_by_item.csv")

    def r3(v: float) -> str:
        return "n/a" if not np.isfinite(v) else f"{100 * v:.1f}"
    panels = []
    for test in ("CRT2", "bCRT"):
        block = []
        for _, r in table[table["test"] == test].iterrows():
            name = f"\\textbf{{{tex_escape(r['item'])}}}" if r["item"] == "all four items" else tex_escape(r["item"])
            block.append([name, r3(r["raw_correct"]), r3(r["raw_intuitive"]), r3(r["raw_other"]),
                          r3(r["adj_correct"]), r3(r["adj_intuitive"]), r3(r["adj_other"])])
        panels.append(("CRT2 (familiarity-adjusted reference used for every CRT2 comparison)" if test == "CRT2"
                       else "bCRT (never adjusted: 5--32\\% exposure per item)", block))
    header = ["Item", tex_head("Raw", "correct"), tex_head("Raw", "lure"), tex_head("Raw", "other"),
              tex_head("Adjusted", "correct"), tex_head("Adjusted", "lure"), tex_head("Adjusted", "other")]
    tex = tex_booktabs(
        header, [], "@{}" + tex_x(1.0) + "rrrrrr@{}",
        "What would the human response mix be if no one had seen the items before? The human reference, raw and "
        "familiarity-adjusted, per item (\\% of scored responses).",
        "tab:human-crt2-reference", panels=panels,
        notes=[f"Raw: the {HUMAN_REFERENCE_N} CRT2-complete respondents. Adjusted: per-item multinomial logit of the response "
               "category on the number of CRT2 items seen before, age, sex and education, predicted at zero items seen and "
               "averaged over the 473 respondents of the analytic frame (fitted on the CRT2-complete respondents). "
               "n/a: the bCRT is not adjusted."])
    (dirs["tables"] / "table_human_crt2_reference_by_item.tex").write_text(tex, encoding="utf-8")
    diag.table("table_human_crt2_reference_by_item.tex")
    return table


# ----------------------------------------- figures ------------------------------------------------

def _ne_cell(ax: plt.Axes, x: float, y: float, w: float = 1.0, h: float = 1.0, fontsize: float = 7.0) -> None:
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=NE_FACE, edgecolor=NE_EDGE, hatch=NE_HATCH,
                           linewidth=0.0, zorder=2))
    ax.text(x, y, "n/e", ha="center", va="center", fontsize=fontsize, color="#333333", zorder=3,
            bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})


# S-8 (PI rulings 2026-10-14): the lure ladder at print width. One row per run or configuration and ONE marker per
# test per row; every row the same height, model families separated by a hairline rather than a gap (PI 14 Oct,
# second round); an N column (bold 471 = simulation run, grey 150 = configuration ladder on the reasoning core, PI-2b)
# with its header on the panel titles' baseline; labels model-first in the family colour; human references named in
# each panel header. Third round (PI 14-15 Oct): no N column (fill carries N), circles in A, no y spine, one key under
# both panels naming the panel-B series; B draws people's CRT2 line familiarity-adjusted only, light horizontal
# gridlines, Claude Fable 5 added. Three files: A+B combined, A alone, B alone (placement fallback).
LADDER_ROWS = [   # (family group, label, kind, key): kind "battery" -> run label; "config" -> ladder `source`
    ("Anthropic", "Claude Sonnet 5, thinking adaptive", "battery", "CR_MAIN_SYNTH_NUM_SONNET5_S42"),
    ("Anthropic", "Claude Sonnet 5, thinking adaptive", "config", "CRFS_SONNET5_DEFAULT_NUM"),
    ("Anthropic", "Claude Sonnet 5, thinking off", "config", "CRFSB_SONNET5_THINKOFF_NUM"),
    ("Anthropic", "Claude Sonnet 4.6, all four runs", "battery", "CR_BASELINE_HUMAN_NUMERIC_CS46_S42"),
    ("Anthropic", "Claude Fable 5, defaults", "config", "CRFSB_CLAUDE_TOP_NUM"),
    ("Google", "Gemini 3.5 Flash, thinking minimal", "battery", "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42"),
    ("Google", "Gemini 3.5 Flash, thinking minimal", "config", "J2_GEMINI35FLASH_MIN"),
    ("Google", "Gemini 3.5 Flash, minimal, lexical", "config", "CRFS_GEMINI35FLASH_MIN_LEX"),
    ("DeepSeek", "DeepSeek V4-Pro, reasoning high", "battery", "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42"),
    ("DeepSeek", "DeepSeek V4-Pro, reasoning high", "config", "CRFSD_DSV4PRO_TOG_NUM"),
    ("DeepSeek", "DeepSeek V4-Pro, reasoning off", "config", "CRFSD_DSV4PRO_TOG_NOTHINK_NUM"),
    ("DeepSeek", "DeepSeek V4-Pro (API), thinking on", "config", "J2_DSV4PRO_ON"),
    ("DeepSeek", "DeepSeek V4-Pro (API), on, lexical", "config", "CRFS_DSV4PRO_ON_LEX"),
    ("DeepSeek", "DeepSeek V4-Flash, thinking on", "config", "J2_DSV4FLASH_ON"),
    ("DeepSeek", "DeepSeek V4-Flash, thinking off", "config", "J2_DSV4FLASH_OFF"),
    ("OpenAI", "GPT-5.6 Sol, reasoning none", "config", "J2_GPT56SOL_NONE"),
    ("OpenAI", "GPT-5.6 Sol, reasoning high", "config", "CRFSB_GPT56_REASONING_NUM"),
    ("Moonshot AI", "Kimi K3, defaults", "config", "CRFSD_KIMI_K3_NUM"),
]
LADDER_ROW_IN = 0.13   # inches per row; every row the same height
HUMAN_ADJ_DASH = (0, (3, 2))   # the familiarity-adjusted human line, and nothing else, is dashed
HUMAN_BAND = "#d6d6d6"   # grey = the 95% interval of a raw human rate, and nothing else (D-235)
KEY_HUMAN_RAW = "Human, raw (95% band)"   # the key entry draws the raw line over its band
# Human-line style in Figure 2. "reference_solid" (PI ruling 16 Oct, option a; amends D-235 and D-237's bands): solid =
# the reference each test is compared against (CRT2 familiarity-adjusted, bCRT as answered), dashed = CRT2 raw, no
# human bands. "raw_solid_bands" is the superseded D-235/D-237 design, kept selectable for the record.
FIG2_HUMAN_STYLE = os.environ.get("BHL_FIG2_HUMAN_STYLE", "reference_solid")


def _ladder_records(lure: pd.DataFrame, ladder: pd.DataFrame, runci: pd.DataFrame) -> list[dict]:
    """Panel A rows. Simulation runs: CR-29's persona-clustered intervals (D-237), each rate guarded against the
    bundle lure table; configuration rows: the ladder table (CR-FS / Jz-B / CR-28 intervals, one method)."""
    recs = []
    rc = runci.set_index(["run_label", "scope"])
    for l in SONNET46_RUNS:   # the "all four runs" row stands for four identical zero-lure runs
        if any(int(rc.loc[(l, sc), "n_events"]) != 0 for sc in ("crt2", "bcrt")):
            raise SystemExit(f"STOP: {l} has lures; the shared Claude Sonnet 4.6 row no longer holds.")
    for fam, name, kind, key in LADDER_ROWS:
        if kind == "battery":
            s = lure[lure["run_label"] == key].set_index("test")
            vals, zero = {}, {}
            for t in ("CRT2", "bCRT"):
                c = rc.loc[(key, t.lower())]
                if abs(float(c["rate"]) - float(s.loc[t, "prevalence"])) > GUARD_ATOL:
                    raise SystemExit(f"STOP: CR-29 {key} {t} rate {c['rate']} != bundle {s.loc[t, 'prevalence']}.")
                vals[t] = (100 * float(c["rate"]), 100 * float(c["ci95_low"]), 100 * float(c["ci95_high"]))
                zero[t] = int(c["n_events"]) == 0
            recs.append({"fam": fam, "name": name, "n": 471, "color": RUN_COLORS[key], "marker": RUN_MARKERS[key],
                         "filled": True, "vals": vals, "zero": zero})
        else:
            r = ladder[ladder["source"] == key].iloc[0]
            vals = {t: tuple(100 * float(r[f"lure_{t.lower()}{k}"]) for k in ("", "_ci_low", "_ci_high"))
                    for t in ("CRT2", "bCRT")}
            zero = {t: str(r[f"lure_{t.lower()}_count"]).split("/")[0] == "0" for t in ("CRT2", "bCRT")}
            recs.append({"fam": fam, "name": name, "n": int(r["n_personas"]), "color": FAMILY_COLORS[r["family"]],
                         "marker": "o", "filled": False, "vals": vals, "zero": zero})
    for k, rec in enumerate(recs):
        rec["y"] = float(k)
        rec["new_family"] = k > 0 and rec["fam"] != recs[k - 1]["fam"]
    return recs


def _draw_ladder_panel(fig, recs, rect, h_raw, h_adj, h_b, w_in, h_band):
    """Panel A in figure fraction `rect` = (left, bottom, width, height): two axes (CRT2, bCRT) sharing the rows.
    Every row is labelled, so every marker is a circle (filled = simulation run, hollow = configuration run); this is
    D-210(b)'s own exemption for row-labelled plots (the Part II forests), applied at the PI's request (14 Oct)."""
    left, bottom, width, height = rect
    gapx = 0.25 / w_in
    axw = (width - gapx) / 2
    axes = [fig.add_axes([left, bottom, axw, height]), fig.add_axes([left + axw + gapx, bottom, axw, height])]
    y_end = recs[-1]["y"]
    for ax, t in zip(axes, ("CRT2", "bCRT")):
        for k, rec in enumerate(recs):
            if k % 2 == 0:
                ax.axhspan(rec["y"] - 0.5, rec["y"] + 0.5, color="#f3f3f3", lw=0, zorder=0)
            if rec["new_family"]:
                ax.axhline(rec["y"] - 0.5, color="#a6a6a6", lw=0.5, zorder=0.7)
        # One line code across the figure (PI 15 Oct): solid black = people as answered (raw), on both tests;
        # dashed black = familiarity-adjusted, which only CRT2 has. Grey is kept for the human 95% bands.
        ref_style = FIG2_HUMAN_STYLE == "reference_solid"
        if not ref_style:
            ax.axvspan(*h_band[t], color=HUMAN_BAND, lw=0, zorder=0.5)   # 95% interval of the raw human rate (D-235)
        if t == "CRT2":
            ax.axvline(h_raw, color=HUMAN_INK, lw=1.1 if ref_style else 1.3, ls=HUMAN_ADJ_DASH if ref_style else "-",
                       zorder=1)
            ax.axvline(h_adj, color=HUMAN_INK, lw=1.3 if ref_style else 1.1, ls="-" if ref_style else HUMAN_ADJ_DASH,
                       zorder=1)
            ax.set_title(f"Human {h_adj:.1f}% (familiarity-adjusted)" if ref_style else
                         f"CRT2 · Human {h_adj:.1f}% familiarity-adjusted,\n{h_raw:.1f}% raw",
                         fontsize=7, loc="left", pad=3, color=HUMAN_INK, linespacing=1.1)
        else:
            ax.axvline(h_b, color=HUMAN_INK, lw=1.3, zorder=1)
            ax.set_title(f"Human {h_b:.1f}%" if FIG2_HUMAN_STYLE == "reference_solid" else f"bCRT · Human {h_b:.1f}%",
                         fontsize=7, loc="right", pad=3, color=HUMAN_INK)
        for rec in recs:
            p, lo, hi = rec["vals"][t]
            if rec["zero"][t]:   # D-232(b): no [0, 0] whisker; the label says what the data show
                row_bg = "#f3f3f3" if int(rec["y"]) % 2 == 0 else "white"   # the text sits over the human lines
                ax.text(p + 2.0, rec["y"], NO_LURES, fontsize=7, va="center", color="#666666", zorder=4,
                        bbox=dict(facecolor=row_bg, edgecolor="none", pad=0.4))
            elif np.isfinite(lo):
                ax.plot([lo, hi], [rec["y"]] * 2, color=rec["color"], lw=1.1, zorder=2, solid_capstyle="butt")
            ax.scatter([p], [rec["y"]], s=18, marker="o", facecolor=rec["color"] if rec["filled"] else "white",
                       edgecolor=rec["color"], linewidths=1.0, zorder=3)
        # no y spine: the rows are bands, and the 0 gridline is the baseline, with room left of it for the markers
        ax.set_xlim(-3, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.tick_params(axis="x", labelsize=7, length=2, pad=1)
        ax.set_xlabel(f"{t} lure rate (%)", fontsize=7, labelpad=1, weight="bold")
        ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0.6)
        ax.axvline(0, color="#b3b3b3", lw=0.6, zorder=0.65)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_ylim(y_end + 0.5, -0.5)
        ax.set_yticks([r["y"] for r in recs])
        ax.tick_params(axis="y", length=0, pad=2)
    axes[0].set_yticklabels([r["name"] for r in recs], fontsize=7)
    for tl, r in zip(axes[0].get_yticklabels(), recs):
        tl.set_color(r["color"])
    axes[1].set_yticklabels([])
    return axes


def _ladder_key_handles(which: str) -> list:
    """The one key under the figure. 'a': marker fill and n/c; 'b': the panel-B series; 'both': A's then B's."""
    from matplotlib.lines import Line2D
    a = [Line2D([], [], marker="o", ls="", markerfacecolor="#555555", markeredgecolor="#555555", ms=4.5,
                label="Simulation run (N = 471)"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#555555", ms=4.5,
                label="Configuration run (N = 150)")]
    from matplotlib.patches import Patch
    if FIG2_HUMAN_STYLE == "reference_solid":
        h = [Line2D([], [], color=HUMAN_INK, lw=1.3, label="Human reference"),
             Line2D([], [], color=HUMAN_INK, lw=1.1, ls=HUMAN_ADJ_DASH, label="Human, CRT2 raw")]
    else:
        h = [(Patch(facecolor=HUMAN_BAND, edgecolor="none"), Line2D([], [], color=HUMAN_INK, lw=1.3)),
             Line2D([], [], color=HUMAN_INK, lw=1.1, ls=HUMAN_ADJ_DASH, label="Human, familiarity-adjusted")]
    b = [Line2D([], [], color=RUN_COLORS[l], lw=1.1, marker=RUN_MARKERS[l], ms=3.5, label=CONDITIONS[l]["name"])
         for l in LURE_PRODUCING_RUNS]
    b += [Line2D([], [], color=FAMILY_COLORS["other"], lw=1.1, marker="o", ms=3.5, markerfacecolor="white",
                 label="Claude Fable 5 (N = 150)")]
    # filled column-major in four columns: pair the long labels so no column is wider than it must be
    both = [a[0], h[0], a[1], h[1], b[0], b[1], b[2], b[3]]   # row 1: markers, runs; row 2: human lines, runs
    return {"a": a + h, "b": h + b, "both": both}[which]


def _draw_item_panel(ax, ref, per_run, pi, b_item_band):
    """Panel B: people's per-item lure rate (CRT2 familiarity-adjusted; bCRT as answered) and the lure-producing
    runs, with Claude Fable 5; series are named in the key under the figure (PI 14 Oct), shapes as D-210(b)."""
    x = np.arange(8)
    adj_i = [ref["lure_adj"][i] for i in CRT2_ITEMS]
    b_i = [ref["b_lure"][i] for i in BCRT_ITEMS]
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    ref_style = FIG2_HUMAN_STYLE == "reference_solid"
    ax.plot(x[:4], adj_i, color=HUMAN_INK, lw=1.3 if ref_style else 1.2, ls="-" if ref_style else HUMAN_ADJ_DASH,
            zorder=2)                                                              # CRT2: familiarity-adjusted
    if not ref_style:
        ax.fill_between(x[4:], [lo for lo, _ in b_item_band], [hi for _, hi in b_item_band], color=HUMAN_BAND, lw=0,
                        zorder=1)                                                  # Wilson 95%, one answer per person
    ax.plot(x[4:], b_i, color=HUMAN_INK, lw=1.3, zorder=2)                        # bCRT: as answered
    order = ["CRT2 q1", "CRT2 q2", "CRT2 q3", "CRT2 q4", "bCRT q1", "bCRT q2", "bCRT q3", "bCRT q4"]
    series = [([per_run.get((l, i), np.nan) for i in CRT2_ITEMS + BCRT_ITEMS], RUN_COLORS[l], RUN_MARKERS[l], "-", l)
              for l in LURE_PRODUCING_RUNS]
    fab = pi[pi["source"] == "CRFSB_CLAUDE_TOP_NUM"].set_index("item")["cell_lure_pct"]
    series.append(([float(fab[i]) for i in order], FAMILY_COLORS["other"], "o", "-", None))   # dashed = adjusted only
    for v, c, mk, ls, _ in series:
        for seg in (slice(0, 4), slice(4, 8)):
            ax.plot(x[seg], v[seg], color=c, lw=1.1, ls=ls, marker=mk, ms=3.5, mfc="white" if mk == "o" else c,
                    zorder=3)
    ax.axvline(3.5, color="#bbbbbb", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["q1", "q2", "q3", "q4"] * 2, fontsize=7)
    ax.set_xlim(-0.4, 7.4)
    ax.set_ylim(0, 65)
    ax.set_yticks([0, 20, 40, 60])
    ax.tick_params(labelsize=7, length=2, pad=1)
    ax.set_ylabel("Lure rate (%)", fontsize=7, labelpad=1, weight="bold")
    ax.text(1.5, -0.2, "CRT2 items", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, weight="bold")
    ax.text(5.5, -0.2, "bCRT items", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, weight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_main1_lure_ladder(selected: dict[str, ReportCandidate], ladder: pd.DataFrame, ref: dict[str, object],
                          dirs: dict[str, Path], diag: Diagnostics) -> None:
    lure = pd.read_csv(dirs["tables"] / "table_intuitive_lure_prevalence_by_run_task.csv")
    pi = pd.read_csv(dirs["tables"] / "table_ladder_probe_per_item_lure.csv")
    hum = lure[lure["run_label"] == "human"].set_index("test")
    h_raw, h_b = 100 * float(hum.loc["CRT2", "prevalence"]), 100 * float(hum.loc["bCRT", "prevalence"])
    h_adj = 100 * float(ref["adj_p"][1])
    root = repo_root()
    runci = read_required_csv(root, CR29_RUN_CSV)
    recs = _ladder_records(lure, ladder, runci)
    hc = read_required_csv(root, CR29_HUMAN_CSV).set_index("test")
    h_band = {}
    for t, h in (("CRT2", h_raw), ("bCRT", h_b)):
        if abs(100 * float(hc.loc[t.lower(), "rate"]) - h) > 1e-9:
            raise SystemExit(f"STOP: CR-29 human {t} rate {hc.loc[t.lower(), 'rate']} != bundle {h / 100}.")
        h_band[t] = (100 * float(hc.loc[t.lower(), "ci95_low"]), 100 * float(hc.loc[t.lower(), "ci95_high"]))
    hi_ = read_required_csv(root, CR28_HUMAN_ITEM_CSV).set_index("item")
    b_item_band = []
    for q, it in zip(("q1", "q2", "q3", "q4"), BCRT_ITEMS):
        if abs(100 * float(hi_.loc[q, "rate"]) - ref["b_lure"][it]) > 1e-9:
            raise SystemExit(f"STOP: CR-28 human bCRT {q} rate {hi_.loc[q, 'rate']} != reference {ref['b_lure'][it] / 100}.")
        b_item_band.append((100 * float(hi_.loc[q, "ci95_low"]), 100 * float(hi_.loc[q, "ci95_high"])))
    per_run = lure_by_item_from_reports(selected)
    W, label_in = 5.3, 1.80
    a_body = LADDER_ROW_IN * (recs[-1]["y"] + 1)
    a_top = 0.30 if FIG2_HUMAN_STYLE == "reference_solid" else 0.42   # header: panel letter + one-line titles
    a_bot = 0.30                       # x labels
    b_body, b_top, b_bot = 0.85, 0.22, 0.30
    for variant in ("combined", "a", "b"):
        key = {"combined": "both", "a": "a", "b": "b"}[variant]
        key_h = 0.36
        H = key_h + (a_top + a_body + a_bot if variant != "b" else 0) + (b_top + b_body + b_bot if variant != "a" else 0)
        fig = plt.figure(figsize=(W, H))
        if variant != "b":
            y0 = H - a_top - a_body
            _draw_ladder_panel(fig, recs, (label_in / W, y0 / H, 1 - label_in / W - 0.01, a_body / H), h_raw, h_adj, h_b, W,
                               h_band)
            if variant == "combined":
                fig.text(0.0, 1 - 0.06 / H, "A   Lure rate by model and configuration", fontsize=7.5, weight="bold",
                         va="top")
        if variant != "a":
            yb = key_h + b_bot
            axB = fig.add_axes([0.45 / W, yb / H, 1 - 0.45 / W - 0.02, b_body / H])
            _draw_item_panel(axB, ref, per_run, pi, b_item_band)
            if variant == "combined":
                fig.text(0.0, (yb + b_body + b_top - 0.04) / H, "B   Lure rate by item", fontsize=7.5,
                         weight="bold", va="top")
        handles = _ladder_key_handles(key)
        labels = [KEY_HUMAN_RAW if isinstance(h, tuple) else h.get_label() for h in handles]
        fig.legend(handles, labels, loc="center", ncol={"both": 4, "a": 2, "b": 3}[key], frameon=False, fontsize=7, handletextpad=0.25,
                   columnspacing=0.45, handlelength=1.15, borderaxespad=0, borderpad=0, bbox_to_anchor=(0.5, (key_h / 2) / H))
        name = {"combined": "main_fig1_lure_ladder", "a": "main_fig1a_lure_ladder_rates",
                "b": "main_fig1b_lure_ladder_items"}[variant]
        save_figure(fig, dirs["figures"] / name)
        diag.figure(f"{name}.pdf")
        diag.figure(f"{name}.png")


def lure_by_item_from_reports(selected: dict[str, ReportCandidate]) -> dict[tuple[str, str], float]:
    """Per-item agent lure (%) from each run's category long file (the source of the bundle lure table)."""
    out: dict[tuple[str, str], float] = {}
    for label in LURE_PRODUCING_RUNS:
        long = pd.read_csv(selected[label].report_dir / "tables" / "comparisons" / "05_crt_response_categories_long.csv",
                           dtype={"entity_id": str}, keep_default_na=False)
        long = long[(long["dataset"] != "human") & (long["response_category"] != UNSCORED)]
        for item, sub in long.groupby("item_id"):
            out[(label, str(item))] = 100 * float(sub["response_category"].eq("intuitive_incorrect").mean())
    return out


def fig_main2_distribution_calibration(dirs: dict[str, Path], diag: Diagnostics) -> None:
    dist = pd.read_csv(dirs["tables"] / "table_distribution_fidelity_metrics.csv")
    cal = pd.read_csv(dirs["tables"] / "table_calibration_fidelity.csv")
    inv = {v: k for k, v in RUN_DISPLAY.items()}
    dist["label"] = dist["run_label"].map(inv)
    cal["label"] = cal["run_label"].map(inv)
    constructs = ["CRT2", "bCRT", "NFC", "Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    cal_name = {"bCRT": "bCRT-derived score"}
    js = np.full((8, 7), np.nan)
    r2 = np.full((8, 7), np.nan)
    for j, label in enumerate(TARGET_LABELS):
        for i, c in enumerate(constructs):
            d = dist[(dist["label"] == label) & (dist["construct"] == c)]
            k = cal[(cal["label"] == label) & (cal["construct"] == cal_name.get(c, c))]
            if len(d):
                js[i, j] = float(d["js_divergence"].iloc[0])
            if len(k) and c != "bCRT":   # S-8e (D-293): the bCRT was never specified, so no calibration R^2
                r2[i, j] = float(k["r2"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    panels = ((axes[0], js, "A. Score-distribution JS divergence, agent vs human", "YlGnBu_r", "JS divergence (0 = identical; lower is better)", 0, max(0.8, np.nanmax(js))),
              (axes[1], r2, "B. Specified-to-realised calibration R²", "YlGnBu", "Calibration R² (higher is better)", 0, 1))
    for ax, arr, title, cmap, cbl, vmin, vmax in panels:
        masked = np.ma.masked_invalid(arr)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cm = plt.get_cmap(cmap)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if not np.isfinite(v):
                    _ne_cell(ax, j, i)
                    continue
                rgba = cm((v - vmin) / (vmax - vmin))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if lum < 0.5 else "#111111")
        ax.set_xticks(range(7))
        ax.set_xticklabels([RUN_SHORT[l] for l in TARGET_LABELS], rotation=30, ha="right", fontsize=8.5)
        for t, l in zip(ax.get_xticklabels(), TARGET_LABELS):
            t.set_color("#111111")
        ax.set_yticks(range(8))
        ax.set_yticklabels(constructs, fontsize=9)
        for j, l in enumerate(TARGET_LABELS):
            ax.add_patch(plt.Rectangle((j - 0.5, -0.95), 1.0, 0.35, color=RUN_COLORS[l], clip_on=False))
        ax.set_ylim(7.5, -0.95)
        ax.set_title(title + "\nN = 471 per run; human N = 471 (bCRT 473)", loc="left", fontsize=10.5, weight="bold", pad=14)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(cbl, fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Are specified traits realised, and do the realised distributions match humans?", fontsize=12.5, weight="bold")
    fig.text(0.5, -0.02, "n/e = not estimable: the agent's realised score is constant (every Claude Sonnet 4.6 agent scores 4/4), so "
             "specified-to-realised R² is undefined.", ha="center", fontsize=8, color="#333333")
    save_figure(fig, dirs["figures"] / "main_fig2_distribution_calibration_heatmaps")
    diag.figure("main_fig2_distribution_calibration_heatmaps.pdf")
    diag.figure("main_fig2_distribution_calibration_heatmaps.png")


# D-222(e): F4 (trait-structure transfer) is produced by CR-27 alone; the builder reads its CSV and never recomputes
# a verdict or a human correlation. Absent file -> the Δρ figures build without marks or human column, no F4 table.
F4_PAIRS_CSV = Path("reports") / "cr27_trait_structure" / "f4_pairs_by_run.csv"
F4_LETTER = {"preserved": "P", "exaggerated": "E", "weakened": "W", "reversed": "R", "introduced": "I",
             "no human target": "0", "indeterminate": "?", "not estimable": ""}   # ASCII only (fonts, D-222(b))
# PI ruling 2026-10-14: "0" = no human target (the human interval includes 0; a dash read as a minus or a reversal);
# "?" = indeterminate (a blank asked the reader to notice an absence). Same symbols in every F4 figure and table.
F4_FOCAL_PAIRS = ["CRT2–bCRT", "CRT2–NFC", "CRT2–Conscientiousness", "bCRT–NFC", "bCRT–Conscientiousness",
                  "NFC–Conscientiousness"]   # the plan's order (ANALYSIS_PLAN §11 F4)
F4_TEXT_NOTE = ("Spearman $\\rho$; $\\Delta\\rho = \\rho_{\\mathrm{run}} - \\rho_{\\mathrm{human}}$; 95\\% percentile "
                "bootstrap intervals, 10,000 resamples (seed 42), humans and agents resampled independently and the human "
                "frame resampled once for all runs; $N = 471$ per side. Verdict: P preserved, E exaggerated, W weakened, "
                "R reversed, I introduced, 0 no human target (the human interval includes 0), ? indeterminate; no "
                "correction across pairs (Section~\\ref{subsec:statistical_analyses}).")


def load_f4(root: Path, diag: Diagnostics | None = None) -> pd.DataFrame | None:
    path = root / F4_PAIRS_CSV
    if not path.exists():
        if diag is not None:
            diag.panel("main_fig3_focal_delta_rho_grid", f"`{F4_PAIRS_CSV}` absent: Δρ figures built without verdict marks or "
                       "human column; F4 tables skipped (D-222(e)).")
        return None
    return pd.read_csv(path)


def _f4_pair_name(a: str, b: str, names: set[str]) -> str:
    """CR-27 names a pair 'X–Y' in its own construct order; match on the two names, never on position."""
    sa, sb = clean_label(a), clean_label(b)
    for cand in (f"{sa}–{sb}", f"{sb}–{sa}"):
        if cand in names:
            return cand
    raise KeyError(f"F4 pair not found for {sa} / {sb}")


def _fmt_rho(v: float) -> str:
    t = f"{v:.2f}"
    t = t.replace("0.", ".", 1) if t.startswith("0.") else t.replace("-0.", "−.", 1)
    return t


def _f4_matrices(f4: pd.DataFrame, pair_names: list[str]):
    """Δρ, verdict letters and human-ρ strings in figure layout (rows = pair_names, columns = TARGET_LABELS)."""
    vals = np.full((len(pair_names), len(TARGET_LABELS)), np.nan)
    letters = [["" for _ in TARGET_LABELS] for _ in pair_names]
    human = []
    for i, pn in enumerate(pair_names):
        rows = f4[f4["pair"] == pn]
        h = rows.iloc[0]
        human.append(f"{_fmt_rho(h['rho_human'])} [{_fmt_rho(h['rho_human_lo'])}, {_fmt_rho(h['rho_human_hi'])}]")
        for j, label in enumerate(TARGET_LABELS):
            r = rows[rows["run_id"] == label]
            if len(r) and r.iloc[0]["verdict"] != "not estimable":
                vals[i, j] = float(r.iloc[0]["delta_rho"])
                letters[i][j] = F4_LETTER[str(r.iloc[0]["verdict"])]
    return vals, letters, human


GRID46_SUB = {"CR_BASELINE_HUMAN_NUMERIC_CS46_S42": "Human\nnumeric", "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": "Human\nlexical",
              "CR_MAIN_SYNTH_NUMERIC_CS46_S42": "Synthetic\nnumeric", "CR_MAIN_SYNTH_LEXICAL_CS46_S42": "Synthetic\nlexical"}
SINGLE_RUN_HEAD = {"CR_MAIN_SYNTH_NUM_SONNET5_S42": "Claude\nSonnet 5", "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": "Gemini\n3.5 Flash",
                   "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": "DeepSeek\nV4-Pro"}


F4_ABBREV = {"Extraversion": "Extr.", "Agreeableness": "Agree.", "Conscientiousness": "Consc.", "Neuroticism": "Neuro.",
             "Openness": "Open."}   # appendix 28-pair figure only; keyed in its caption


def draw_delta_rho_grid(pair_names: list[str], vals: np.ndarray, letters, human, vmax: float, base: Path,
                        row_h: float, abbreviate: bool = False) -> None:
    """One Δρ grid (D-222(b)/(c)): optional human-ρ text column, seven run columns under family-coloured headers,
    Δρ on the shared diverging scale, the verdict letter in each cell's top-right corner, n/e hatched. No title, no
    key sentence; drawn at print width; every text >= 7 pt."""
    from matplotlib.gridspec import GridSpec
    n = len(pair_names)
    head_in, cbar_in = 0.62, (0.78 if letters is not None else 0.36)   # room for the verdict key (PI 14 Oct)
    fig_h = head_in + n * row_h + cbar_in
    fig = plt.figure(figsize=(5.1, fig_h))   # + 0.1 in tight-bbox pad each side ~ the 5.5 in text width
    labels = pair_names
    if abbreviate:
        labels = ["–".join(F4_ABBREV.get(x, x) for x in pn.split("–")) for pn in pair_names]
    left, right = (0.80 if abbreviate else 1.12) / 5.1, 0.995
    hw = 1.00 if abbreviate else 0.80
    gs = GridSpec(1, 2 if human else 1, figure=fig, width_ratios=([hw, 4.10 - hw - (0.32 if abbreviate else 0)]
                                                                   if human else [1.0]), wspace=0.03,
                  left=left, right=right, top=1 - head_in / fig_h, bottom=cbar_in / fig_h)
    ax = fig.add_subplot(gs[0, 1 if human else 0])
    im = ax.imshow(np.ma.masked_invalid(vals), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(n):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                _ne_cell(ax, j, i, fontsize=7)
                continue
            colour = "white" if abs(v) > 0.6 * vmax else "#111111"
            txt = "0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "−")
            ax.text(j - 0.13, i + 0.05, txt, ha="center", va="center", fontsize=7, color=colour)
            if letters is not None and letters[i][j]:
                ax.text(j + 0.46, i - 0.40, letters[i][j], ha="right", va="top", fontsize=7, color=colour,
                        weight="bold")
    ax.set_xticks([])
    ax.set_yticks(range(n))
    ax.set_yticklabels([] if human else labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if human:
        hx = fig.add_subplot(gs[0, 0])   # not sharey: a shared axis would print the row labels twice
        hx.set_xlim(0, 1)
        for i, t in enumerate(human):
            hx.text(0.5, i, t, ha="center", va="center", fontsize=7, color=HUMAN_INK)
        hx.set_xticks([])
        hx.set_yticks(range(n))
        hx.set_yticklabels(labels, fontsize=7)
        hx.tick_params(axis="y", length=0)
        for sp in hx.spines.values():
            sp.set_visible(False)
        hx.set_ylim(n - 0.5, -0.5)
    ax.set_ylim(n - 0.5, -0.5)
    fig.canvas.draw()
    # headers: family-hue strip and run names (level 1 model name; level 2 Sonnet 4.6 condition in words; no codes)
    pos = ax.get_position()
    cw = pos.width / vals.shape[1]
    top = pos.y1
    strip_h = 0.05 / fig_h
    for j, label in enumerate(TARGET_LABELS):
        x0 = pos.x0 + j * cw
        fig.add_artist(plt.Rectangle((x0 + 0.002, top + 0.008 / fig_h), cw - 0.004, strip_h, color=RUN_COLORS[label],
                                     transform=fig.transFigure, clip_on=False))
        t = SINGLE_RUN_HEAD.get(label) or GRID46_SUB[label]
        fig.text(x0 + cw / 2, top + 0.08 / fig_h, t, ha="center", va="bottom", fontsize=7, color=RUN_COLORS[label],
                 linespacing=1.0)
    x46 = pos.x0 + 3 * cw
    hue46 = RUN_COLORS["CR_BASELINE_HUMAN_NUMERIC_CS46_S42"]
    fig.text(x46 + 2 * cw, top + 0.40 / fig_h, "Claude Sonnet 4.6", ha="center", va="bottom", fontsize=7.5,
             weight="bold", color=hue46)
    fig.add_artist(plt.Line2D([x46 + 0.004, x46 + 4 * cw - 0.004], [top + 0.385 / fig_h] * 2, color=hue46, lw=0.8,
                              transform=fig.transFigure))
    if human:
        hp = fig.axes[1].get_position()
        fig.text((hp.x0 + hp.x1) / 2, top + 0.08 / fig_h, "Human ρ\n[95% CI]", ha="center", va="bottom",
                 fontsize=7, weight="bold", color=HUMAN_INK, linespacing=1.0)
    if letters is not None:
        # verdict key inside the figure (PI ruling 2026-10-14, amends D-222(b)): only the letters this figure uses
        used = {x for row in letters for x in row}
        names = [("P", "preserved"), ("E", "exaggerated"), ("W", "weakened"), ("R", "reversed"), ("I", "introduced"),
                 ("0", "no human target"), ("?", "indeterminate")]
        items = [(k, v) for k, v in names if k in used] + [("n/e", "not estimable")]
        per_row = 3 if len(items) <= 6 else 4
        x_left, x_right = 0.03, 0.97
        for k, (sym, word) in enumerate(items):
            r_, c_ = divmod(k, per_row)
            xk = x_left + (x_right - x_left) * (c_ + 0.5) / per_row
            yk = (0.62 - 0.16 * r_) / fig_h
            fig.text(xk, yk, (f"{sym} = " if sym else "") + word, ha="center", va="center", fontsize=7, color="#111111")
    cax = fig.add_axes([pos.x0 + pos.width * 0.25, 0.13 / fig_h, pos.width * 0.5, 0.06 / fig_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-vmax, 0, vmax])
    cb.ax.set_xticklabels([f"{-vmax:+.1f}".replace("-", "−"), "0", f"{vmax:+.1f}"], fontsize=7)
    cb.ax.tick_params(length=2, pad=1)
    cb.set_label("Δρ = ρ(run) − ρ(human)", fontsize=7, labelpad=1)
    save_figure(fig, base)


def write_f4_tables(f4: pd.DataFrame, pair_order: list[str], dirs: dict[str, Path], diag: Diagnostics) -> None:
    """D-222(d): table A (three non-Sonnet-4.6 runs, 28 rows) and table B (four Sonnet 4.6 conditions, 15 estimable
    rows); focal pairs first in the plan's order, then the Δρ figure's row order; read from the CR-27 CSV only."""
    order = F4_FOCAL_PAIRS + [p for p in pair_order if p not in F4_FOCAL_PAIRS]
    groups = [("a", list(TARGET_LABELS[:3]), "Trait-structure transfer, Claude Sonnet 5, Gemini 3.5 Flash and DeepSeek V4-Pro: per construct pair, the run's correlation, its difference from the human correlation, and the verdict.", ""),
              ("b", list(TARGET_LABELS[3:]), "Trait-structure transfer, the four Claude Sonnet 4.6 conditions: per construct pair, the run's correlation, its difference from the human correlation, and the verdict.",
               " The 13 pairs that involve CRT2 or bCRT are not estimable in these conditions (every agent scored 4/4 on both) and are omitted.")]
    rows_out = []
    for tag, runs, caption, extra in groups:
        sub = f4[f4["run_id"].isin(runs)]
        keep = [p for p in order if not (sub[sub["pair"] == p]["verdict"] == "not estimable").all()]
        head1 = ["", ""] + [f"\\multicolumn{{3}}{{c}}{{{tex_escape(condition_name(r))}}}" for r in runs]
        cmid = " ".join(f"\\cmidrule(lr){{{3 + 3 * k}-{5 + 3 * k}}}" for k in range(len(runs)))
        head2 = ["Pair", "Human $\\rho$ [95\\% CI]"] + ["$\\rho$", "$\\Delta\\rho$ [95\\% CI]", "V"] * len(runs)
        body = []
        for pn in keep:
            r0 = sub[sub["pair"] == pn].iloc[0]
            line = [tex_escape(pn).replace("–", "--"),
                    f"{_fmt_rho(r0['rho_human'])} [{_fmt_rho(r0['rho_human_lo'])}, {_fmt_rho(r0['rho_human_hi'])}]".replace("−", "$-$")]
            for rid in runs:
                r = sub[(sub["pair"] == pn) & (sub["run_id"] == rid)].iloc[0]
                d = f"{r['delta_rho']:+.2f} [{r['delta_rho_lo']:+.2f}, {r['delta_rho_hi']:+.2f}]".replace("-", "$-$")
                letter = F4_LETTER[str(r["verdict"])]
                line += [_fmt_rho(r["rho_run"]).replace("−", "$-$"), d, letter]
                rows_out.append({"table": tag, "pair": pn, "run_identifier": rid, "run": condition_name(rid),
                                 "rho_human": r0["rho_human"], "rho_run": r["rho_run"], "delta_rho": r["delta_rho"],
                                 "delta_rho_lo": r["delta_rho_lo"], "delta_rho_hi": r["delta_rho_hi"], "verdict": r["verdict"]})
            body.append(" & ".join(line) + " \\\\")
        colspec = "@{}ll" + "rlc" * len(runs) + "@{}"
        tex = "\n".join([
            "% Generated by scripts/build_neurips_camera_ready_assets.py (S-8, D-222(d)). Do not hand-edit.",
            "% Requires \\usepackage{booktabs,rotating}.",
            "\\begin{sidewaystable}",
            "\\centering", "\\scriptsize" if len(runs) > 3 else "\\footnotesize",   # table B has four runs
            "\\setlength{\\tabcolsep}{3pt}" if len(runs) > 3 else "\\setlength{\\tabcolsep}{3.5pt}",
            f"\\caption{{{caption}}}", f"\\label{{tab:f4-{tag}}}",
            f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(head1) + " \\\\", cmid, " & ".join(head2) + " \\\\",
            "\\midrule", *body, "\\bottomrule", "\\end{tabular}",
            "\\par\\smallskip", "\\begin{minipage}{\\linewidth}\\footnotesize\\raggedright", F4_TEXT_NOTE + extra + "\\par",
            "\\end{minipage}", "\\end{sidewaystable}", ""])
        (dirs["tables"] / f"table_f4_trait_structure_{tag}.tex").write_text(tex, encoding="utf-8")
        diag.table(f"table_f4_trait_structure_{tag}.tex")
    write_table(pd.DataFrame(rows_out), dirs["tables"] / "table_f4_trait_structure.csv", diag=diag)


def fig_main3_f4_bands(f4: pd.DataFrame, base: Path) -> None:
    """Main Figure 3 (PI ruling 2026-10-14): the six focal pairs as correlations. Per pair a band: the human rho as a
    black bar inside its grey 95% band; each estimable run as its marker (family hue, run shape, D-210) with its 95%
    interval and verdict symbol. Indeterminate runs are drawn grey so their position does not invite a reading. Pairs on
    which the four Claude Sonnet 4.6 conditions are not estimable carry an asterisk, explained in the caption. Values and
    verdicts are read from CR-27's CSV (D-222(e)). No title, no key sentence beyond the symbol key; >= 7 pt."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    grey = "#a8a8a8"
    step, gap, pad = 0.55, 0.45, 0.32
    y, ticks, labels, rows = 0.0, [], [], []
    for pn in F4_FOCAL_PAIRS:
        sub = f4[f4["pair"] == pn]
        est = [r for r in TARGET_LABELS if sub[sub["run_id"] == r].iloc[0]["verdict"] != "not estimable"]
        top, bottom = y, y + (len(est) - 1) * step
        rows.append((pn, sub, est, top, bottom))
        ticks.append((top + bottom) / 2)
        labels.append(pn + ("*" if len(est) < len(TARGET_LABELS) else ""))
        y = bottom + 2 * pad + gap
    n_sub = sum(len(r[2]) for r in rows)
    body_in = 0.118 * n_sub + 0.11 * len(rows)
    bottom_in = 0.98
    fig = plt.figure(figsize=(5.1, body_in + bottom_in + 0.05))
    H = body_in + bottom_in + 0.05
    ax = fig.add_axes([1.30 / 5.1, bottom_in / H, 1 - 1.30 / 5.1 - 0.01, body_in / H])
    for pn, sub, est, top, bottom in rows:
        h = sub.iloc[0]
        ax.axhspan(top - pad, bottom + pad, color="#f4f4f4", zorder=0, lw=0)
        ax.fill_betweenx([top - pad, bottom + pad], h["rho_human_lo"], h["rho_human_hi"], color="#d2d2d2", lw=0, zorder=1)
        ax.plot([h["rho_human"]] * 2, [top - pad, bottom + pad], color=HUMAN_INK, lw=2.0, zorder=2, solid_capstyle="butt")
        for k, r in enumerate(est):
            row = sub[sub["run_id"] == r].iloc[0]
            yy = top + k * step
            sym = F4_LETTER[str(row["verdict"])]
            c = grey if row["verdict"] == "indeterminate" else RUN_COLORS[r]
            ax.plot([row["rho_run_lo"], row["rho_run_hi"]], [yy, yy], color=c, lw=1.2, zorder=3, solid_capstyle="butt")
            ax.plot([row["rho_run"]], [yy], marker=RUN_MARKERS[r], color=c, ms=4.6, mec="white", mew=0.4, zorder=4)
            ax.text(row["rho_run_hi"] + 0.018, yy, sym, va="center", ha="left", fontsize=7, weight="bold", color=c, zorder=5)
    ax.axvline(0, color="#9a9a9a", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.22, 1.08)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["−0.2", "0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_xlabel("Spearman ρ", fontsize=7, labelpad=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(y - gap - pad + 0.02, -pad - 0.02)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, pad=1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    # legend: two rows of runs, then the symbol key; packed tight under the axis
    def h_run(r, label):
        return Line2D([], [], color=RUN_COLORS[r], marker=RUN_MARKERS[r], lw=1.2, ms=4.6, mec="white", mew=0.4, label=label)
    row1 = [Line2D([], [], color=HUMAN_INK, lw=2.0, label="human ρ"), Patch(color="#d2d2d2", lw=0, label="human 95% CI")]
    row1 += [h_run(r, condition_name(r)) for r in TARGET_LABELS[:3]]
    s46 = TARGET_LABELS[3:]
    row2 = [Line2D([], [], lw=0, label="Claude Sonnet 4.6:")]
    row2 += [h_run(r, condition_name(r).split(" · ", 1)[1]) for r in s46]
    y1 = (bottom_in - 0.36) / H
    fig.legend(handles=row1, loc="center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, y1),
               handlelength=1.1, handletextpad=0.3, columnspacing=0.55)
    fig.legend(handles=row2, loc="center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, y1 - 0.17 / H),
               handlelength=1.1, handletextpad=0.3, columnspacing=0.55)
    used = {F4_LETTER[str(v)] for v in f4[f4["pair"].isin(F4_FOCAL_PAIRS)]["verdict"]} - {""}
    names = [("P", "preserved"), ("E", "exaggerated"), ("W", "weakened"), ("R", "reversed"), ("I", "introduced"),
             ("0", "no human target"), ("?", "indeterminate (grey)")]
    key = "   ".join(f"{k} {v}" for k, v in names if k in used)
    fig.text(0.5, y1 - 0.36 / H, key, ha="center", va="center", fontsize=7, color="#111111")
    save_figure(fig, base)


def fig_main3_delta_r(selected: dict[str, ReportCandidate], dirs: dict[str, Path]) -> pd.DataFrame:
    quiet = Diagnostics()   # alias resolution for the same 28 core pairs Figure 5 already resolved
    pairs = [(a, b) for i, a in enumerate(CORE_ASSOCIATION_VARIABLES) for j, b in enumerate(CORE_ASSOCIATION_VARIABLES) if i > j]
    mat = np.full((len(pairs), len(TARGET_LABELS)), np.nan)
    for j, label in enumerate(TARGET_LABELS):
        df, path = load_corr_diff(selected[label])
        if df.empty or path is None:
            continue
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.rename(columns={df.columns[0]: "variable"}).set_index("variable")
        elif df.columns[0] not in df.index:
            df = df.set_index(df.columns[0])
        for i, (a, b) in enumerate(pairs):
            ra = resolve_matrix_name(list(df.index.astype(str)), a, label, path, quiet)
            cb_ = resolve_matrix_name(list(df.columns), b, label, path, quiet)
            if ra is not None and cb_ is not None:
                mat[i, j] = safe_float(df.loc[ra, cb_])
    out = pd.DataFrame(mat, columns=[RUN_SHORT[l] for l in TARGET_LABELS])
    out.insert(0, "pair", [f"{assoc_short(a)} – {assoc_short(b)}" for a, b in pairs])
    out.to_csv(dirs["tables"] / "main_fig3_delta_r_matrix_data.csv", index=False)
    f4 = load_f4(repo_root())
    if f4 is not None:
        names = set(f4["pair"])
        pair_names = [_f4_pair_name(b, a, names) for a, b in pairs]   # CR-27's names, figure's row order
        vals, letters, human = _f4_matrices(f4, pair_names)
        # one producer: the cells come from CR-27's Δρ, which equals the per-run difference files (checked at S-8)
        assert np.allclose(np.nan_to_num(vals, nan=9), np.nan_to_num(mat, nan=9), atol=5e-6), "F4 Δρ != diff files"
        vmax = math.ceil(float(np.nanmax(np.abs(f4.loc[f4["verdict"] != "not estimable", "delta_rho"]))) * 10) / 10
        draw_delta_rho_grid(pair_names, vals, letters, human, vmax, dirs["figures"] / "main_fig3_delta_r_matrix", row_h=0.205,
                            abbreviate=True)
        fi = [pair_names.index(p) for p in F4_FOCAL_PAIRS]
        draw_delta_rho_grid(F4_FOCAL_PAIRS, vals[fi], [letters[k] for k in fi], [human[k] for k in fi], vmax,
                            dirs["figures"] / "main_fig3_focal_delta_rho_grid", row_h=0.30)   # fallback if space runs out
        fig_main3_f4_bands(f4, dirs["figures"] / "main_fig3_f4_correlation_bands")
        write_f4_tables(f4, pair_names, dirs, Diagnostics())
    else:
        pair_names = [f"{clean_label(b)}–{clean_label(a)}" for a, b in pairs]
        vmax = math.ceil(float(np.nanmax(np.abs(mat))) * 10) / 10
        draw_delta_rho_grid(pair_names, mat, None, None, vmax, dirs["figures"] / "main_fig3_delta_r_matrix", row_h=0.205)
    return out


def fig_main4_brt_gap(dirs: dict[str, Path], diag: Diagnostics) -> None:
    t = pd.read_csv(dirs["tables"] / "table_brt_distributional_fidelity.csv")
    inv = {v: k for k, v in RUN_DISPLAY.items()}
    t["label"] = t["run_label"].map(inv)
    outcomes = [(fam, o) for fam, ms in BRT_PRIMARY_FAMILIES.items() for o in ms]
    ypos, y, prev = {}, 0.0, None
    for fam, o in outcomes:
        if prev is not None and fam != prev:
            y += 0.9
        ypos[clean_label(o)] = y
        y += 1.0
        prev = fam
    offsets = np.linspace(-0.33, 0.33, len(TARGET_LABELS))
    fig, ax = plt.subplots(figsize=(11.5, 9.4))
    ax.axvspan(-20, 20, color="#efefef", zorder=0)
    ax.axvline(0, color=HUMAN_INK, lw=1.2, zorder=1)
    for k, label in enumerate(TARGET_LABELS):
        sub = t[t["label"] == label]
        for _, r in sub.iterrows():
            yy = ypos[r["outcome"]] + offsets[k]
            gap = 100 * (float(r["run_prevalence"]) - float(r["human_prevalence"]))   # signed; `absolute_gap` is |gap|
            lo, hi = float(r["f1_gap_lo_pp"]), float(r["f1_gap_hi_pp"])
            ax.plot([lo, hi], [yy, yy], color=RUN_COLORS[label], lw=1.2, zorder=2)
            ax.scatter([gap], [yy], marker=RUN_MARKERS[label], s=30, color=RUN_COLORS[label], zorder=3,
                       edgecolor="white", linewidths=0.4)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()), fontsize=8.5)
    for fam in BRT_PRIMARY_FAMILIES:
        ys = [ypos[clean_label(o)] for o in BRT_PRIMARY_FAMILIES[fam]]
        ax.text(101, np.mean(ys), fam, fontsize=10, weight="bold", va="center", color="#333333")
    ax.set_xlim(-100, 100)
    ax.set_xlabel("Agent − human prevalence (percentage points; bootstrap 95% CI)", fontsize=9.5)
    ybot = max(ypos.values()) + 0.9
    ax.set_ylim(ybot + 0.4, -0.8)
    ax.text(0.8, ybot, "human prevalence", ha="left", va="center", fontsize=8, color=HUMAN_INK)
    ax.text(-19.5, ybot, "±20 pp band", ha="left", va="center", fontsize=8, color="#555555")
    ax.grid(axis="x", color="#e6e6e6", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=RUN_MARKERS[l], color=RUN_COLORS[l], ls="", markersize=6.5, label=RUN_SHORT[l]) for l in TARGET_LABELS]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, ncol=1, title="Run", title_fontsize=8.5)
    ax.set_title("Does human-like behavioural variation survive?\nN = 471 per run (DeepSeek V4-Pro BRT N = 467); human N = 473",
                 loc="left", fontsize=11.5, weight="bold")
    save_figure(fig, dirs["figures"] / "main_fig4a_brt_gap")
    diag.figure("main_fig4a_brt_gap.pdf")
    diag.figure("main_fig4a_brt_gap.png")


def fig_main5_lure_by_told_level(agent: pd.DataFrame, human: pd.DataFrame, dirs: dict[str, Path], diag: Diagnostics) -> None:
    """Lure rate by the reflection level each agent was told; people by their own CRT2 level (the frozen cutoffs).
    S-8d (PI 29 Oct): drawn at print width (5.5 in) for the appendix; run names in words; no internal title (the caption
    asks the question); the four Claude Sonnet 4.6 runs share one key entry (all at 0%)."""
    from matplotlib.lines import Line2D
    with plt.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75), sharey=True)
        lv = np.arange(5)
        for ax, inst in zip(axes, ("crt2", "bcrt")):
            h = human[(human["instrument"] == inst) & (human["response_category"] == "intuitive_incorrect")].set_index("told_level")
            hv = [100 * float(h["proportion"].get(k, 0.0)) for k in lv]
            ax.plot(lv, hv, color=HUMAN_INK, lw=1.8, zorder=4)
            ax.plot(lv, hv, color=HUMAN_INK, ls="", marker=HUMAN_MARKER_ALONG_Y, ms=9, mew=1.8, zorder=2.5)
            for label in TARGET_LABELS:
                a = agent[(agent["run_label"] == label) & (agent["instrument"] == inst)]
                vals = []
                for k in lv:
                    sub = a[(a["told_level"] == k) & (a["response_category"] == "intuitive_incorrect")]
                    vals.append(100 * float(sub["proportion"].iloc[0]) if len(sub) else 0.0)
                ax.plot(lv, vals, color=RUN_COLORS[label], lw=1.2, marker=RUN_MARKERS[label], ms=3.8, zorder=3, alpha=0.95)
            ax.set_xticks(lv)
            ax.set_xlim(-0.3, 4.3)
            ax.set_ylim(-4, 90)
            ax.set_xlabel("Told CRT2 level (people: own level)", fontsize=7)
            ax.tick_params(labelsize=7, length=2, pad=1)
            ax.grid(axis="y", color="#ececec", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title("A  CRT2 items" if inst == "crt2" else "B  bCRT items", loc="left", fontsize=7.5, weight="bold")
        axes[0].set_ylabel("Lure rate (%)", fontsize=7)
        s46 = [l for l in TARGET_LABELS if l.endswith("CS46_S42")]
        handles = [Line2D([], [], color=HUMAN_INK, marker=HUMAN_MARKER_ALONG_Y, ms=9, mew=1.8, lw=1.8, label="People")]
        handles += [Line2D([], [], marker=RUN_MARKERS[l], color=RUN_COLORS[l], ls="-", lw=1.2, ms=3.8,
                           label=CONDITIONS[l]["name"]) for l in TARGET_LABELS if l not in s46]
        handles.append(_MarkerRow([(RUN_MARKERS[l], RUN_COLORS[l]) for l in s46]))
        labels = [hd.get_label() for hd in handles[:-1]] + ["Claude Sonnet 4.6, four runs (0% at every level)"]
        fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3, fontsize=6.5, frameon=False,
                   bbox_to_anchor=(0.5, -0.04), handler_map={_MarkerRow: _MarkerRowHandler()}, columnspacing=1.2)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.3, wspace=0.12)
        save_figure(fig, dirs["figures"] / "main_fig5_lure_by_told_level")
    diag.figure("main_fig5_lure_by_told_level.pdf")
    diag.figure("main_fig5_lure_by_told_level.png")


# ------------------------------------------- Part II ---------------------------------------------

PARTII_FIGURES = [f"fig_s11_forest_H{i}.{ext}" for i in range(1, 5) for ext in ("pdf", "png")] + \
                 ["fig_s11_verdict_grid.pdf", "fig_s11_verdict_grid.png"]
PARTII_TABLES = [f"s11_family_H{i}_table.csv" for i in range(1, 5)] + ["s11_verdict_grid.csv"]



# S-8 (PI 19 Oct, mock 3 chosen): Figure 4 = (A) every primary BRT outcome x every run, people's prevalence against the
# run's, with the identity line and a +-20 pp band; (B) the associations people show at q < .05 within family (CR-20),
# each run's odds ratio per bCRT item against people's, verdict letters in run colour. The full verdict grid and the
# family tables stay in the appendix. Appendix companion: panel A split into one panel per run.
CR20_PRIMARY_CSV = Path("reports") / "cr20_human" / "tables" / "s5_primary_families.csv"
BRT_SIG_Q = 0.05
BRT_SHORT_NAME = {   # panel B labels; the appendix tables carry the full paper-facing names
    "BRT1_check_venue": "Phones the\nrestaurant", "BRT2_check_institutional_sources": "Consults regulators,\njournals, guidance",
    "BRT2_check_persons": "Asks friends or\nfamily (medication)", "BRT1_check_official": "Consults hygiene\nor official records",
    "BRT2_emo_worry": "Voices worry about\nthe medication"}
VERDICT_LETTER = {"transfers": "T", "fails to transfer": "F", "indeterminate": "?", "unstable": "U"}
BRT_BAND = "#ececec"


class _MarkerRow:
    """Legend handle: several run markers side by side (the four Claude Sonnet 4.6 conditions)."""
    def __init__(self, marks):
        self.marks = marks


class _MarkerRowHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.lines import Line2D as _L
        x0, y0, w, h = handlebox.xdescent, handlebox.ydescent, handlebox.width, handlebox.height
        n = len(orig_handle.marks)
        for i, (m, c) in enumerate(orig_handle.marks):
            handlebox.add_artist(_L([x0 + w * (i + 0.5) / n], [y0 + h / 2], marker=m, ls="", color=c, ms=3.2,
                                    transform=handlebox.get_transform()))
        return handlebox


class _Letter:
    def __init__(self, ch):
        self.ch = ch


class _LetterHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.text import Text
        t = Text(handlebox.xdescent + handlebox.width / 2, handlebox.ydescent + handlebox.height / 2, orig_handle.ch,
                 ha="center", va="center", fontsize=fontsize, weight="bold", color="#333333",
                 transform=handlebox.get_transform())
        handlebox.add_artist(t)
        return t


def _brt_family_tables(dirs: dict[str, Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(dirs["tables"] / f"s11_family_H{i}_table.csv") for i in range(1, 5)], ignore_index=True)


def _brt_sig_outcomes(root: Path) -> list[str]:
    h = read_required_csv(root, CR20_PRIMARY_CSV)
    h = h[(h["placement"] == "primary") & (h["predictor_block"] == "bCRT")]
    sig = h.loc[h["q_bh_within_family"] < BRT_SIG_Q, "outcome"].tolist()
    if len(sig) != 5:   # RESULTS_current §4: 15 primary outcomes, 5 at q < .05
        raise SystemExit(f"STOP: expected 5 human associations at q < .05, found {len(sig)}: {sig}")
    return sig


def _brt_scatter(ax, d: pd.DataFrame, runs: list[str], sig: list[str], ms: float = 16) -> None:
    xx = np.array([0.0, 100.0])
    ax.fill_between(xx, xx - 20, xx + 20, color=BRT_BAND, lw=0, zorder=0)
    ax.plot(xx, xx, color=HUMAN_INK, lw=0.9, zorder=1)
    for k in runs:
        sub = d[d["run_label"] == k]
        is_sig = sub["outcome"].isin(sig)
        for mask, filled in ((~is_sig, False), (is_sig, True)):
            ax.scatter(sub.loc[mask, "human_prev_pct"], sub.loc[mask, "run_prev_pct"], marker=RUN_MARKERS[k], s=ms,
                       facecolor=RUN_COLORS[k] if filled else "white", edgecolor=RUN_COLORS[k], linewidths=0.8,
                       zorder=3 if k not in SONNET46_RUNS else 2, alpha=0.95)
    ax.set_xlim(0, 100); ax.set_ylim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=7, length=2, pad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_main4_brt_scatter_forest(root: Path, dirs: dict[str, Path], diag: Diagnostics) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator
    d = _brt_family_tables(dirs)
    sig = _brt_sig_outcomes(root)
    est = [l for l in TARGET_LABELS if l not in SONNET46_RUNS]
    # guard: the plotted prevalences are the bundle's (table_brt_distributional_fidelity) values
    t = pd.read_csv(dirs["tables"] / "table_brt_distributional_fidelity.csv")
    if not np.isclose(np.sort(100 * t["run_prevalence"].astype(float)), np.sort(d["run_prev_pct"].astype(float)), atol=1e-9).all():
        raise SystemExit("STOP: Part II family-table prevalences differ from the bundle's BRT table.")
    W, H = 5.3, 3.45
    bot = 1.30 / H   # room under the panels for the x labels and the two keys
    keys_top = 0.83 / H   # both keys hang from the same line
    fig = plt.figure(figsize=(W, H))
    side = 0.38
    axA = fig.add_axes([0.085, bot, side, side * W / H])
    _brt_scatter(axA, d, TARGET_LABELS, sig)
    axA.set_xlabel("People's prevalence (%)", fontsize=7, weight="bold", labelpad=1)
    axA.set_ylabel("Run's prevalence (%)", fontsize=7, weight="bold", labelpad=1)
    axA.text(3, 101, "Run over-produces", fontsize=6.5, color="#555555", va="top")
    axA.text(97, 3, "Run under-produces", fontsize=6.5, color="#555555", ha="right", va="bottom")
    axA.text(90, 72, "Matches\npeople", fontsize=6.5, color="#555555", ha="center", rotation=45, rotation_mode="anchor")
    top_y = bot + side * W / H
    fig.text(0.01, top_y + 0.035, "A   Every outcome, every run", fontsize=7.5, weight="bold", ha="left", va="bottom")
    # B: the associations people show
    axB = fig.add_axes([0.685, bot, 0.245, side * W / H])
    for i, o in enumerate(sig):
        s = d[d["outcome"] == o]; h = s.iloc[0]
        axB.axhspan(i - 0.5, i + 0.5, color="#f4f4f4" if i % 2 == 0 else "white", lw=0, zorder=0)
        axB.plot([h["human_or_lo"], h["human_or_hi"]], [i - 0.30] * 2, color=HUMAN_INK, lw=1.2, zorder=2)
        axB.scatter([h["human_or"]], [i - 0.30], marker="|", s=40, color=HUMAN_INK, zorder=3, linewidths=1.6)
        for j, k in enumerate(est):
            r = s[s["run_label"] == k].iloc[0]
            yy = i - 0.10 + 0.19 * j
            v = str(r["f3_verdict"])
            if v == "unstable":   # C3: no reliable estimate; a dotted placeholder marks that the run is there
                axB.plot([0.5, 2.0], [yy] * 2, color=RUN_COLORS[k], lw=0.8, ls=(0, (1, 2)), alpha=0.7, zorder=2)
            else:
                axB.plot([r["f3_or_run_lo"], r["f3_or_run_hi"]], [yy] * 2, color=RUN_COLORS[k], lw=0.9, zorder=2)
                axB.scatter([r["f3_or_run"]], [yy], marker=RUN_MARKERS[k], s=13, color=RUN_COLORS[k], zorder=3,
                            edgecolor="white", linewidths=0.3)
            axB.text(1.06, yy, VERDICT_LETTER[v], transform=axB.get_yaxis_transform(), fontsize=6.8, weight="bold",
                     color=RUN_COLORS[k], ha="center", va="center", clip_on=False)
    axB.set_xscale("log"); axB.set_xlim(0.5, 2.0)
    axB.xaxis.set_minor_locator(NullLocator())
    axB.set_xticks([0.5, 0.75, 1, 1.5, 2]); axB.set_xticklabels(["0.5", "0.75", "1", "1.5", "2"])
    axB.axvline(1, color="#9e9e9e", lw=0.6, zorder=1)
    axB.set_ylim(len(sig) - 0.5, -0.5); axB.set_yticks(range(len(sig)))
    axB.set_yticklabels([BRT_SHORT_NAME[o] for o in sig], fontsize=7, linespacing=0.95)
    axB.tick_params(axis="y", length=0, pad=2); axB.tick_params(axis="x", labelsize=7, length=2, pad=1)
    for sp in ("top", "right", "left"):
        axB.spines[sp].set_visible(False)
    axB.set_xlabel("Odds ratio per bCRT item\n(log scale)", fontsize=7, weight="bold", labelpad=1, linespacing=1.0)
    fig.text(0.53, top_y + 0.035, "B   The five associations people show", fontsize=7.5, weight="bold", ha="left", va="bottom")
    axB.text(1.155, 0.5, "Verdict", transform=axB.transAxes, fontsize=7, weight="bold", ha="center", va="center",
             rotation=270, clip_on=False)   # a right-hand axis title, reading top to bottom, facing the data
    # Key 1 (under A): runs and encodings, every entry a mark then its label (PI 19 Oct).
    handles = [Line2D([], [], marker=RUN_MARKERS[k], ls="", color=RUN_COLORS[k], ms=4) for k in est]
    labels = [CONDITIONS[k]["name"] for k in est]
    handles.append(_MarkerRow([(RUN_MARKERS[l], RUN_COLORS[l]) for l in SONNET46_RUNS]))
    labels.append("Claude Sonnet 4.6, four conditions")
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="#777777", markeredgecolor="#777777", ms=4),
                Line2D([], [], color=HUMAN_INK, lw=1.0), Patch(facecolor=BRT_BAND)]
    labels += ["Filled: an association people show", "People (A: diagonal; B: odds ratio, 95% CI)", "Within ±20 pp of people"]
    fig.legend(handles, labels, loc="upper left", ncol=1, frameon=False, fontsize=6.8, handlelength=3.8,
               handletextpad=0.4, labelspacing=0.28, bbox_to_anchor=(0.0, keys_top), borderaxespad=0,
               handler_map={_MarkerRow: _MarkerRowHandler()})
    # Key 2 (under B): the verdict letters, fenced and titled, letter then meaning; the Sonnet 4.6 absence noted.
    vh = [_Letter(c) for c in ("T", "F", "?", "U")]
    vl = ["transfers", "fails to transfer", "indeterminate", "unstable: too few events to estimate (dotted)"]
    leg2 = fig.legend(vh, vl, loc="upper right", ncol=1, frameon=True, fancybox=False, edgecolor="#bdbdbd",
                      fontsize=6.8, handlelength=1.0, columnspacing=0.8, handletextpad=0.4, labelspacing=0.25,
                      borderpad=0.45, bbox_to_anchor=(0.985, keys_top), borderaxespad=0,
                      title="Verdict (letter in each run's colour)",
                      title_fontsize=6.8, handler_map={_Letter: _LetterHandler()})
    leg2.get_title().set_weight("bold")
    fig.text(0.985, 0.012, "Claude Sonnet 4.6 in B: not estimable (4/4 on bCRT)", fontsize=6.5, ha="right", va="bottom",
             color=RUN_COLORS[SONNET46_RUNS[0]])
    save_figure(fig, dirs["figures"] / "main_fig4_brt_scatter_forest")
    diag.figure("main_fig4_brt_scatter_forest.pdf"); diag.figure("main_fig4_brt_scatter_forest.png")
    # appendix: panel A split by run, three columns
    ncol = 3; nrow = int(np.ceil(len(TARGET_LABELS) / ncol))
    fig2, axes = plt.subplots(nrow, ncol, figsize=(5.3, 1.85 * nrow), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, TARGET_LABELS):
        _brt_scatter(ax, d, [k], sig, ms=14)
        ax.set_title(CONDITIONS[k]["name"].replace("Claude Sonnet 4.6 · ", "Claude Sonnet 4.6\n"), fontsize=7,
                     color=RUN_COLORS[k], weight="bold", pad=2, linespacing=1.0)
    used = len(TARGET_LABELS)
    for i, ax in enumerate(axes.flat):
        if i >= used:
            ax.axis("off")
            continue
        r, c = divmod(i, ncol)
        if (r + 1) * ncol + c >= used:   # the lowest panel in its column carries the x axis labels
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("People's prevalence (%)", fontsize=7, weight="bold", labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Run's prevalence (%)", fontsize=7, weight="bold", labelpad=1)
    fig2.tight_layout(h_pad=0.6, w_pad=0.4)
    save_figure(fig2, dirs["figures"] / "appendix_brt_prevalence_by_run")
    diag.figure("appendix_brt_prevalence_by_run.pdf"); diag.figure("appendix_brt_prevalence_by_run.png")


# S-8a (D-253(d), D-255, D-256(a)): the appendix per-level figure. Prevalence of the five behaviours people reliably link
# to reflection at each reflection level, Wilson 95% intervals as CR-30 stored them: the seven runs by the level each
# persona was told (crt2_level), people by their own level at the frozen cutoffs. Read, never recomputed.
CR30_RUNS_CSV = Path("reports") / "cr30_level_prevalence" / "level_prevalence_runs_cr30.csv"
CR30_HUMAN_CSV = Path("reports") / "cr30_level_prevalence" / "level_prevalence_human_cr30.csv"
CR30_LEVELS = (0, 1, 2, 3, 4)
CR30_SYNTH_TOLD_MIX = (109, 86, 103, 106, 67)
CR30_HUMDERIVED_TOLD_MIX = (94, 98, 106, 105, 68)
CR30_DSV4PRO_TOLD_MIX = (107, 85, 102, 106, 67)   # its 467 BRT respondents
CR30_HUMAN_LEVEL_MIX = (107, 79, 106, 111, 68)
CR30_COLUMN_TITLE = {   # BRT_SHORT_NAME's wording, wrapped for a fifth of the page width
    "BRT1_check_venue": "Phones the\nrestaurant", "BRT2_check_institutional_sources": "Consults\nregulators,\njournals, guidance",
    "BRT2_check_persons": "Asks friends\nor family\n(medication)", "BRT1_check_official": "Consults\nhygiene or\nofficial records",
    "BRT2_emo_worry": "Voices worry\nabout the\nmedication"}


def read_cr30_level_prevalence(root: Path, dirs: dict[str, Path], sig: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = read_required_csv(root, CR30_RUNS_CSV)
    hum = read_required_csv(root, CR30_HUMAN_CSV)
    if len(runs) != len(TARGET_LABELS) * len(sig) * len(CR30_LEVELS) or len(hum) != len(sig) * len(CR30_LEVELS):
        raise SystemExit(f"STOP: CR-30 row counts {len(runs)} / {len(hum)}, expected 175 / 25.")
    if set(runs["run_label"]) != set(TARGET_LABELS) or set(runs["outcome"]) != set(sig) or set(hum["outcome"]) != set(sig):
        raise SystemExit("STOP: CR-30 runs or behaviours differ from the seven runs and the five CR-20 behaviours.")
    for name, t in (("runs", runs), ("people", hum)):
        if not np.allclose(t["prevalence"], t["events"] / t["n"], atol=GUARD_ATOL):
            raise SystemExit(f"STOP: CR-30 {name}: prevalence is not events / n.")
        if ((t["wilson_lo"] > t["prevalence"]) | (t["wilson_hi"] < t["prevalence"])).any():
            raise SystemExit(f"STOP: CR-30 {name}: an interval does not contain its prevalence.")
    for k in TARGET_LABELS:
        want = (CR30_DSV4PRO_TOLD_MIX if k == DSV4PRO_LABEL else
                CR30_HUMDERIVED_TOLD_MIX if k.startswith("CR_BASELINE_HUMAN") else CR30_SYNTH_TOLD_MIX)
        for o in sig:
            got = tuple(runs[(runs["run_label"] == k) & (runs["outcome"] == o)].sort_values("level")["n"].astype(int))
            if got != want:
                raise SystemExit(f"STOP: CR-30 level mix {k} {o}: {got}, ruled {want}.")
    for o in sig:
        got = tuple(hum[hum["outcome"] == o].sort_values("level")["n"].astype(int))
        if got != CR30_HUMAN_LEVEL_MIX:
            raise SystemExit(f"STOP: CR-30 people's level mix {o}: {got}, ruled {CR30_HUMAN_LEVEL_MIX}.")
    fam = _brt_family_tables(dirs).set_index(["run_label", "outcome"])
    tot = runs.groupby(["run_label", "outcome"])[["events", "n"]].sum()
    for (k, o), r in tot.iterrows():
        f = fam.loc[(k, o)]
        if int(r["events"]) != int(f["run_events"]) or int(r["n"]) != int(f["run_n"]):
            raise SystemExit(f"STOP: CR-30 {k} {o}: {int(r['events'])}/{int(r['n'])} against the family table's "
                             f"{int(f['run_events'])}/{int(f['run_n'])}.")
    return runs, hum


def fig_appendix_brt_level_prevalence(root: Path, dirs: dict[str, Path], diag: Diagnostics) -> None:
    sig = _brt_sig_outcomes(root)
    runs, hum = read_cr30_level_prevalence(root, dirs, sig)
    lv = np.array(CR30_LEVELS)
    groups = ["human"] + list(TARGET_LABELS)
    fig, axes = plt.subplots(len(groups), len(sig), figsize=(5.3, 7.4), sharex=True, sharey=True)

    def series(t: pd.DataFrame, o: str) -> pd.DataFrame:
        return t[t["outcome"] == o].sort_values("level")

    def whiskers(t: pd.DataFrame) -> np.ndarray:
        return 100 * np.vstack([t["prevalence"] - t["wilson_lo"], t["wilson_hi"] - t["prevalence"]])

    for j, o in enumerate(sig):
        h = series(hum, o)
        hp = 100 * h["prevalence"].to_numpy()
        for i, g in enumerate(groups):
            ax = axes[i, j]
            if g == "human":
                ax.errorbar(lv, hp, yerr=whiskers(h), color=HUMAN_INK, lw=1.0, elinewidth=0.7, capsize=0, zorder=3)
                ax.plot(lv, hp, ls="", marker=HUMAN_MARKER_ALONG_Y, color=HUMAN_INK, ms=5, mew=1.2, zorder=4)
            else:
                ax.plot(lv, hp, color=HUMAN_INK, lw=0.7, alpha=0.35, zorder=1)   # people's rates, for direction
                r = series(runs[runs["run_label"] == g], o)
                rp = 100 * r["prevalence"].to_numpy()
                ax.errorbar(lv, rp, yerr=whiskers(r), color=RUN_COLORS[g], lw=1.0, elinewidth=0.7, capsize=0, zorder=3)
                ax.plot(lv, rp, ls="", marker=RUN_MARKERS[g], color=RUN_COLORS[g], ms=3.2, zorder=4)
            ax.set_xlim(-0.45, 4.45); ax.set_ylim(-4, 104)
            ax.set_xticks(lv); ax.set_yticks([0, 50, 100])
            ax.tick_params(labelsize=6, length=2, pad=1)
            ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0, j].set_title(CR30_COLUMN_TITLE[o], fontsize=6.3, weight="bold", pad=3, linespacing=1.0)
    for i, g in enumerate(groups):
        name = "People" if g == "human" else CONDITIONS[g]["name"].replace("Claude Sonnet 4.6 · ", "Claude Sonnet 4.6\n")
        axes[i, 0].set_ylabel(name, fontsize=6.3, weight="bold", color=HUMAN_INK if g == "human" else RUN_COLORS[g],
                              rotation=0, ha="right", va="center", labelpad=4, linespacing=1.0)
    # the caption carries the intervals, the grouping and the faint line; the figure keeps only its axis titles
    fig.text(0.175, 0.995, "Prevalence (%)", fontsize=6.8, weight="bold", ha="left", va="top")
    fig.text(0.58, 0.052, "Reflection level", fontsize=6.8, weight="bold", ha="center", va="bottom")

    def mix(t: pd.DataFrame) -> str:
        return "/".join(str(int(v)) for v in t[t["outcome"] == sig[0]].sort_values("level")["n"])
    synth = [k for k in TARGET_LABELS if k.startswith("CR_MAIN_SYNTH") and k != DSV4PRO_LABEL]
    derived = [k for k in TARGET_LABELS if k.startswith("CR_BASELINE_HUMAN")]
    fig.text(0.58, 0.0, f"n at levels 0–4: synthetic-profile runs {mix(runs[runs['run_label'] == synth[0]])} "
             f"(DeepSeek V4-Pro {mix(runs[runs['run_label'] == DSV4PRO_LABEL])});\nhuman-derived runs "
             f"{mix(runs[runs['run_label'] == derived[0]])}; people {mix(hum)}.", fontsize=6.0, ha="center", va="bottom", linespacing=1.2,
             color="#333333")
    fig.subplots_adjust(left=0.175, right=0.99, top=0.915, bottom=0.1, hspace=0.32, wspace=0.22)
    save_figure(fig, dirs["figures"] / "appendix_brt_level_prevalence")
    diag.figure("appendix_brt_level_prevalence.pdf"); diag.figure("appendix_brt_level_prevalence.png")
    data = pd.concat([hum, runs], ignore_index=True)
    data.to_csv(dirs["tables"] / "appendix_brt_level_prevalence_data.csv", index=False)   # CR-30's values, unchanged
    diag.table("appendix_brt_level_prevalence_data.csv")


def copy_partii_appendix(root: Path, dirs: dict[str, Path], diag: Diagnostics) -> list[dict[str, str]]:
    """CR-25 STEP 5: the Part II forests (H1-H4) and the verdict grid, copied byte-for-byte (D-200(a))."""
    import hashlib
    import shutil
    rows = []
    for name, sub, dest in [(n, "figures", dirs["figures"]) for n in PARTII_FIGURES] + \
                           [(n, "tables", dirs["tables"]) for n in PARTII_TABLES]:
        src = root / PARTII_SRC / sub / name
        if not src.exists():
            raise SystemExit(f"STOP: Part II source missing: {src}")
        shutil.copyfile(src, dest / name)
        a = hashlib.sha256(src.read_bytes()).hexdigest()
        b = hashlib.sha256((dest / name).read_bytes()).hexdigest()
        if a != b:
            raise SystemExit(f"STOP: Part II copy of {name} is not byte-identical.")
        rows.append({"asset": f"{sub}/{name}", "source": (PARTII_SRC / sub / name).as_posix(), "sha256": a})
        (diag.figure if sub == "figures" else diag.table)(name)
    return rows


# ------------------------------------------- captions --------------------------------------------

CR25_CAPTIONS = {
    "main_fig1_lure_ladder": (
        "**Figure 1. Which models produce intuitive-lure errors, and do they produce them where humans do?** "
        "Panel A places every full-battery run (filled markers, N = 471, Wilson 95% intervals) and every configuration of the "
        "150-profile ladder (hollow circles, persona-clustered bootstrap intervals; n/c where the source computed no per-test "
        "interval) on the CRT2 (A1) and bCRT (A2) lure axes, grouped by provider. One hue per model family, one shape per run "
        "(Claude Sonnet 5 blue plus, Gemini 3.5 Flash orange X, DeepSeek V4-Pro magenta downward triangle; Claude Sonnet 4.6 violet: "
        "human-derived numeric circle, human-derived lexical square, synthetic numeric upward triangle, synthetic "
        "lexical diamond); models with no full battery are grey. The human reference is black: on CRT2 the shaded band runs from "
        "the raw human rate (dashed) to the familiarity-adjusted rate (solid); on bCRT, which is not adjusted, one solid line. "
        "Lure production switches on with the thinking regime within a model (Claude Sonnet 5, DeepSeek V4-Pro) and is absent at "
        "both effort poles of GPT-5.6 and in every Claude Sonnet 4.6 run. Panel B shows where the errors fall: the per-item lure "
        "profile of the three lure-producing full batteries against the human profile (black; grey band raw to adjusted on CRT2)."),
    "main_fig2_distribution_calibration_heatmaps": (
        "**Figure 2. Are specified traits realised, and do the realised distributions match humans?** "
        "Rows are the eight constructs, columns the seven full-battery runs (colour strip = "
        "model-family hue; the four Claude Sonnet 4.6 columns share violet). Panel A: Jensen–Shannon "
        "divergence between each run's realised score distribution and the human distribution (darker = closer). Panel B: R² of "
        "realised on specified score (darker = better calibrated). NFC and the Big Five are realised and calibrated in every run; "
        "the two reflection tests are where the runs depart from humans. Hatched n/e cells: R² is not estimable because every "
        "Claude Sonnet 4.6 agent scores 4/4. N = 471 per run."),
    "main_fig3_delta_r_matrix": (
        "**Figure 3. Is the human correlation structure preserved?** "
        "Each cell is Δr = r(agent) − r(human) for one of the 28 pairs among CRT2, bCRT, NFC and the Big Five (rows) in one run "
        "(columns; colour strip = model-family hue), on one diverging scale: red, the agent association is stronger or more positive than "
        "the human one; blue, weaker or more negative; white, matched. Hatched n/e cells: the correlation is undefined because the "
        "agent's CRT2 or bCRT score is constant. N = 471 per run."),
    "main_fig4a_brt_gap": (
        "**Figure 4. Does human-like behavioural variation survive, and does the human association transfer?** "
        "Panel A: for each of the 15 primary BRT outcomes (grouped by hypothesis family H1–H4), the run's prevalence minus the "
        "human prevalence in percentage points with a bootstrap 95% interval; one hue per model family, one shape per run "
        "(the four Claude Sonnet 4.6 runs share violet); the shaded band is "
        "±20 pp, the compression threshold. Panel B (fig_s11_verdict_grid, copied unchanged from the Part II analysis): the "
        "transfer verdict per outcome and run. N = 471 per run (DeepSeek V4-Pro BRT N = 467: four personas returned no usable "
        "open-ended response, and the generating model snapshot has since been retired by the provider); human N = 473."),
    "main_fig5_lure_by_told_level": (
        "**Figure 5. Does specifying lower reflection produce more lure errors?** "
        "Intuitive-lure prevalence on CRT2 (A) and bCRT (B) items by the CRT2 level each agent was told (0–4), one line per run "
        "in its model-family hue and its own marker shape (one hue per model family, one shape per run; the four Claude Sonnet 4.6 runs share violet), beside the human rate by each respondent's own CRT2 level (black). bCRT was never specified, so its "
        "panel groups the same agents by their told CRT2 level. Claude Sonnet 5, Gemini 3.5 Flash and DeepSeek V4-Pro produce "
        "more lures at lower told levels; the four Claude Sonnet 4.6 runs are flat at 0.0% at every level. n per level is "
        "printed beneath the panels; N = 471 per run."),
    "appendix_crt_response_categories_by_crt_level": (
        "**Appendix Figure. Which response categories does each told CRT2 level produce?** "
        "One row of panels per run: response categories (green correct, orange intuitive incorrect, red other incorrect) by the "
        "CRT2 level each agent was told, 0–4 (`crt2_level`, the level written into the persona). bCRT was never specified, so "
        "the bCRT panels group the same agents by their told CRT2 level. Each bar carries its lure prevalence and n (agents and "
        "item responses); groups sum to 471 agents per run."),
    "appendix_brt_level_prevalence": (
        "**Appendix Figure. How often does each run produce the five behaviours at each reflection level?** "
        "Prevalence of the five behaviours people reliably link to bCRT, with Wilson 95% intervals as CR-30 stored them: "
        "people (top row, black) by their own CRT2 level at the frozen cutoffs, each run by the level its personas were told "
        "(`crt2_level`), in the run's hue and marker shape. The faint line in each run row repeats people's rates. n per level "
        "is printed beneath the panels. Data: `tables/appendix_brt_level_prevalence_data.csv` (CR-30, D-255)."),
    "table_item_exposure_and_unseen_lure": (
        "**Table. How familiar were the reflection items, and how much does familiarity lower the human lure rate?** "
        "Prior exposure per item and the human lure rate on all items, unseen items only and seen items only. CRT2 items had "
        "been seen before by far more respondents than bCRT items; the unseen-only rate is an upper bound on the exposure effect."),
    "table_configuration_ladder": (
        "**Table. Which configurations produce intuitive-lure errors?** The configuration ladder on the reasoning core: every "
        "model and thinking regime, each answering the eight reflection items for the same 150 profiles; lure rates and their "
        "intervals are on scored responses (blank answers excluded from both), by each source run's interval method."),
    "table_design_probes": (
        "**Table. Do decoding temperature, the retained history or a sanction framing explain the absence of lures in Claude "
        "Sonnet 4.6?** Temperature sweep (five settings), no-history ablation (paired, McNemar) and sanction probe."),
    "table_human_crt2_reference_by_item": (
        "**Table. What would the human response mix be if no one had seen the items before?** The human reference per item, raw "
        "and familiarity-adjusted (CRT2), and raw (bCRT, never adjusted)."),
}


def write_cr25_captions(dirs: dict[str, Path], diag: Diagnostics) -> None:
    for stem, text in CR25_CAPTIONS.items():
        (dirs["captions"] / f"{stem}.md").write_text(text + "\n", encoding="utf-8")


def write_appendix_manifest(partii: list[dict[str, str]], dirs: dict[str, Path], diag: Diagnostics) -> None:
    L = ["# Camera-ready figure and table plan (UPDATE_MAP §2b) — assets in this bundle", "",
         "Generated by `scripts/build_neurips_camera_ready_assets.py` (CR-25). Placement is S-8's; this lists what exists.", "",
         "## Main text", "",
         "| slot | asset | caption |", "|---|---|---|",
         "| Table 1 | `tables/table_fidelity_scorecard.tex` (parts A and B) | `captions/captions.md` (Table 7) |",
         "| Fig 1 | `figures/main_fig1_lure_ladder.pdf` | `captions/main_fig1_lure_ladder.md` |",
         "| Fig 2 | `figures/main_fig2_distribution_calibration_heatmaps.pdf` | `captions/main_fig2_distribution_calibration_heatmaps.md` |",
         "| Fig 3 | `figures/main_fig3_delta_r_matrix.pdf` (data `tables/main_fig3_delta_r_matrix_data.csv`) | `captions/main_fig3_delta_r_matrix.md` |",
         "| Fig 4 | A `figures/main_fig4a_brt_gap.pdf`; B `figures/fig_s11_verdict_grid.pdf` (Part II copy) | `captions/main_fig4a_brt_gap.md` |",
         "| Fig 5 | `figures/main_fig5_lure_by_told_level.pdf` (data `tables/table_crt_response_categories_by_specified_level.csv`) | `captions/main_fig5_lure_by_told_level.md` |",
         "", "## Appendix tables", "",
         "| table | asset |", "|---|---|",
         "| Experimental conditions | `tables/table_experimental_runs.tex` |",
         "| Configuration ladder | `tables/table_configuration_ladder.tex` (+ `.csv`) |",
         "| Design probes | `tables/table_design_probes.tex` (+ `.csv`) |",
         "| Human CRT2 reference, raw vs adjusted, per item | `tables/table_human_crt2_reference_by_item.tex` (+ `.csv`) |",
         "| Item exposure and unseen-only lure | `tables/table_item_exposure_and_unseen_lure.tex` (+ `.csv`) |",
         "| LIWC per-run tables (three estimable runs, PI-5) | `tables/table_liwc_theory_set_slope_fidelity_1_*.tex` … `_3_*.tex` |",
         "| BRT family tables H1–H4; verdict grid table (Part II) | `tables/s11_family_H1_table.csv` … `H4`, `tables/s11_verdict_grid.csv` |",
         "| BRT prevalence, steering and transfer by run (S-8b; read from the family tables) | `tables/table_brt_prevalence_by_run.tex`, `tables/table_brt_steering.tex`, `tables/table_brt_transfer.tex` |",
         "| Exploratory families (§6) | not copied: `reports/cr13b_partii/tables/s11_appendix_6_*_table.csv` |",
         "", "## Appendix figures", "",
         "| figure | asset |", "|---|---|",
         "| Distribution dashboard | `figures/fig2_distribution_fidelity_dashboard.pdf` |",
         "| ECDF overlays | `figures/fig2b_distribution_fidelity_ecdf_overlay.pdf` |",
         "| Calibration fingerprint | `figures/fig6_calibration_fingerprint.pdf` |",
         "| Relative calibration RMSE | `figures/appendix_calibration_relative_rmse.pdf` |",
         "| Seven association heatmaps | `figures/fig5_association_fidelity_heatmaps.pdf` |",
         "| CRT categories by told level (full; Fig 7, regrouped CR-25) | `figures/appendix_crt_response_categories_by_crt_level.pdf` |",
         "| BRT prevalence bars | `figures/fig4_brt_behavioural_compression.pdf` |",
         "| Part II forests H1–H4 | `figures/fig_s11_forest_H1.pdf` … `H4` |",
         "| Flagged BRT models | `figures/appendix_brt_transfer_models_flagged.pdf` |",
         "| BRT prevalence at each reflection level (CR-30, S-8a) | `figures/appendix_brt_level_prevalence.pdf` (data `tables/appendix_brt_level_prevalence_data.csv`) |",
         "| BRT categorical distributions | `figures/appendix_brt_categorical_distributions.pdf` |",
         "| LIWC slopes + heatmap | `figures/appendix_liwc_slope_fidelity_theory_set.pdf`, `figures/appendix_liwc_slope_difference_heatmap.pdf` |",
         "", "## Part II copies (byte-identical to the CR-13B sources)", "",
         "The figures were re-rendered by CR-25b and again by CR-25c with the paper palette (`scripts/paper_palette.py`, D-208(b), D-210), "
         "by CR-25d (human bar, decimal log-axis labels, D-212), and by S-8 (TrueType fonts, labelled ticks beside the human target, D-213(d)), through "
         "`scripts/cr13b/run_partii_figures.py`; the Part II data files are unchanged.", "",
         "| asset | source | sha256 |", "|---|---|---|"]
    L += [f"| `{r['asset']}` | `{r['source']}` | `{r['sha256']}` |" for r in partii]
    (dirs["captions"] / "appendix_figure_list.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# S-8b appendix BRT family tables (presentation only): prevalence, steering (F2) and transfer (F3) for the 15 coded
# behaviours x 7 runs, every value read from the bundle's copies of the Part II family tables (s11_family_H1-H4);
# nothing recomputed, formatting and rounding only. Guards: the paper's Table 1 counts (RESULTS_current R-T1).
# ---------------------------------------------------------------------------------------------------

# Display-only override of the stored paper_facing_name (the family-table CSVs keep the stored names).
BRT_DISPLAY_NAME = {"Phones the restaurant (primary source; H1a)": "Phones the restaurant (the primary source)",
                    # S-8e (D-291(c); PI 2 Nov Q7): words for the scenario's "Friend B", and the rubric's full target
                    # (brt_v3/prompt_v3.txt: "TOWARD FRIEND B OR TOWARD THE RESTAURANT"); Appendix A.7 says the same
                    "Appeals to fairness or loyalty toward Friend B": "Appeals to fairness or loyalty toward the birthday friend or the restaurant"}
BRT_FAMILY_NAMES = {"H1": "Sources the respondent plans to consult", "H2": "Decision taken", "H3": "Emotion expressed",
                    "H4": "Appeals to fairness or loyalty"}
# Column heads: the model name; for the four Sonnet 4.6 conditions a spanning "Claude Sonnet 4.6" over the condition in
# words, as in the run heads of the appendix level-prevalence figure and the Δρ grid (no run codes).
BRT_TABLE_HEAD = {
    "CR_MAIN_SYNTH_NUM_SONNET5_S42": ("Claude", "Sonnet 5"),
    "CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42": ("Gemini", "3.5 Flash"),
    "CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42": ("DeepSeek", "V4-Pro"),
    "CR_BASELINE_HUMAN_NUMERIC_CS46_S42": ("human", "numeric"),
    "CR_BASELINE_HUMAN_LEXICAL_CS46_S42": ("human", "lexical"),
    "CR_MAIN_SYNTH_NUMERIC_CS46_S42": ("synthetic", "numeric"),
    "CR_MAIN_SYNTH_LEXICAL_CS46_S42": ("synthetic", "lexical"),
}
# RESULTS_current R-T1 (Table 1): per run, compressed and unstable counts of 15; transfers / fails (incl. introduces) /
# indeterminate for the three runs whose bCRT varies.
BRT_TABLE_GUARD_COMPRESSED = dict(zip(TARGET_LABELS, (8, 8, 11, 9, 10, 10, 10)))
BRT_TABLE_GUARD_UNSTABLE = dict(zip(TARGET_LABELS, (5, 5, 4, 6, 6, 7, 6)))
BRT_TABLE_GUARD_VERDICTS = dict(zip(TARGET_LABELS[:3], ((3, 3, 4), (3, 4, 3), (1, 3, 7))))
BRT_TABLE_N = {**{lab: 471 for lab in TARGET_LABELS}, DSV4PRO_LABEL: 467}
BRT_TABLE_HUMAN_N = 473


def _brt_table_frame(dirs: dict[str, Path]) -> tuple[pd.DataFrame, list[str]]:
    """The 105 family-table rows, guarded; returns the frame and the behaviour order (H1..H4, family-table row order)."""
    d = _brt_family_tables(dirs)
    if len(d) != 105:
        raise SystemExit(f"STOP: expected 105 BRT family-table rows (15 x 7), read {len(d)}.")
    if sorted(d["run_label"].unique()) != sorted(TARGET_LABELS):
        raise SystemExit(f"STOP: family-table runs {sorted(d['run_label'].unique())} are not the seven target runs.")
    order = d.loc[d["run_label"] == TARGET_LABELS[0], "outcome"].tolist()
    for lab in TARGET_LABELS:
        sub = d[d["run_label"] == lab]
        if len(order) != 15 or sub["outcome"].tolist() != order:
            raise SystemExit(f"STOP: {lab} does not carry the same 15 behaviours in the same order.")
        for col in ("f1_compressed", "c3_unstable"):
            if sub[col].dtype != bool:
                raise SystemExit(f"STOP: {col} is not boolean in the family tables.")
        nc, nu = int(sub["f1_compressed"].sum()), int(sub["c3_unstable"].sum())
        if (nc, nu) != (BRT_TABLE_GUARD_COMPRESSED[lab], BRT_TABLE_GUARD_UNSTABLE[lab]):
            raise SystemExit(f"STOP: {lab} compressed/unstable {nc}/{nu}, Table 1 says "
                             f"{BRT_TABLE_GUARD_COMPRESSED[lab]}/{BRT_TABLE_GUARD_UNSTABLE[lab]}.")
        if set(sub["run_n"].astype(int)) != {BRT_TABLE_N[lab]}:
            raise SystemExit(f"STOP: {lab} run_n {sorted(set(sub['run_n']))}, the table note says {BRT_TABLE_N[lab]}.")
        # the notes say profile-likelihood intervals exactly where the model is Firth-penalised, i.e. the U rows
        if not ((sub["f2_estimator"] == "Firth logit") == sub["c3_unstable"]).all() or \
           not ((sub["f2_ci_type"] == "profile likelihood") == sub["c3_unstable"]).all():
            raise SystemExit(f"STOP: {lab} F2 Firth / profile-likelihood rows are not exactly the unstable rows.")
    if set(d["human_n"].astype(int)) != {BRT_TABLE_HUMAN_N}:
        raise SystemExit(f"STOP: human_n {sorted(set(d['human_n']))}, the table note says {BRT_TABLE_HUMAN_N}.")
    if (d.groupby("outcome")[["human_prev_pct", "human_prev_lo_pct", "human_prev_hi_pct", "human_or", "human_or_lo",
                              "human_or_hi"]].nunique() > 1).any().any():
        raise SystemExit("STOP: people's prevalence or OR differs between runs for one behaviour.")
    return d, order


def _brt_row(d: pd.DataFrame, lab: str, outcome: str) -> pd.Series:
    return d[(d["run_label"] == lab) & (d["outcome"] == outcome)].iloc[0]


def _brt_or_cell(o: object, lo: object, hi: object) -> str:
    vals = [safe_float(v) for v in (o, lo, hi)]
    if not np.isfinite(vals).all():
        return "n/e"
    return f"{vals[0]:.2f} [{vals[1]:.2f}, {vals[2]:.2f}]"


def _brt_mark(text: str) -> str:
    return f"\\textsuperscript{{{text}}}" if text else ""


def _brt_heads(runs: list[str], cols_per_run: int, first_col: int) -> tuple[str, str, list[str]]:
    """Run header cells: row 1 (the spanning "Claude Sonnet 4.6"), its cmidrule, and the row-2 cell per run."""
    row1, row2 = [], []
    s46 = [r for r in runs if r in SONNET46_RUNS]
    for r in runs:
        if r not in SONNET46_RUNS:
            row1.append(f"\\multicolumn{{{cols_per_run}}}{{c}}{{}}")
        elif r == s46[0]:
            row1.append(f"\\multicolumn{{{cols_per_run * len(s46)}}}{{c}}{{Claude Sonnet 4.6}}")
        row2.append(f"\\multicolumn{{{cols_per_run}}}{{c}}{{{tex_head(*BRT_TABLE_HEAD[r])}}}")
    cmid = ""
    if s46:
        start = first_col + cols_per_run * runs.index(s46[0])
        cmid = f"\\cmidrule(lr){{{start}-{start + cols_per_run * len(s46) - 1}}}"
    return " & ".join(row1), cmid, row2


def _brt_tex(env: str, colspec: str, caption: str, label: str, head1: str, cmid: str, head2: str, body: list[str],
             note: str, packages: str, tabcolsep: str) -> str:
    tab_env = "tabularx" if "X" in colspec else "tabular"
    width = "{\\linewidth}" if tab_env == "tabularx" else ""
    lines = [
        "% Generated by scripts/build_neurips_camera_ready_assets.py (S-8b appendix BRT family tables; values read from "
        "tables/s11_family_H1-H4_table.csv, formatting only). Do not hand-edit.",
        f"% Requires \\usepackage{{{packages}}}.",
        f"\\begin{{{env}}}" + ("[t]" if env == "table" else ""),
        "\\centering", "\\footnotesize", f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{{tab_env}}}{width}{{{colspec}}}", "\\toprule", head1 + " \\\\", cmid,
        head2 + " \\\\", "\\midrule", *body, "\\bottomrule", f"\\end{{{tab_env}}}",
        "\\par\\smallskip", "\\begin{minipage}{\\linewidth}\\footnotesize\\raggedright", note + "\\par",
        "\\end{minipage}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def _brt_body(d: pd.DataFrame, order: list[str], ncols: int, cells) -> list[str]:
    """Rows grouped by family under italic group rows; `cells(outcome)` returns the cells after the row label."""
    fam = dict(zip(d["outcome"], d["family"]))
    body, last = [], None
    for oc in order:
        if fam[oc] != last:
            if last is not None:
                body.append("\\addlinespace[2pt]")
            body.append(f"\\multicolumn{{{ncols}}}{{@{{}}l}}{{\\textit{{{tex_escape(BRT_FAMILY_NAMES[fam[oc]])}}}}} \\\\")
            last = fam[oc]
        name = d.loc[d["outcome"] == oc, "paper_facing_name"].iloc[0]
        name = BRT_DISPLAY_NAME.get(name, name)
        body.append(" & ".join([tex_escape(name)] + cells(oc)) + " \\\\")
    return body


def write_brt_prevalence_table(d: pd.DataFrame, order: list[str], dirs: dict[str, Path], diag: Diagnostics) -> None:
    runs = list(TARGET_LABELS)

    def cells(oc: str) -> list[str]:
        h = _brt_row(d, runs[0], oc)
        out = [f"{h['human_prev_pct']:.1f} [{h['human_prev_lo_pct']:.1f}, {h['human_prev_hi_pct']:.1f}]"]
        for r in runs:
            x = _brt_row(d, r, oc)
            out += [f"{x['run_prev_pct']:.1f}",
                    _brt_mark(("C" if bool(x["f1_compressed"]) else "") + ("U" if bool(x["c3_unstable"]) else ""))]
        return out
    head1, cmid, row2 = _brt_heads(runs, 2, 3)
    head1 = " & ".join(["", "", head1])
    head2 = " & ".join(["Behaviour", tex_head("People", "\\% [95\\% CI]")] + row2)
    colspec = "@{}" + tex_x(1.0) + "r" + "r@{}l" * len(runs) + "@{}"
    note = ("C: compressed, a gap of at least 20 percentage points with non-overlapping 95\\% intervals; U: unstable, "
            "fewer than 20 events or non-events (Section~\\ref{subsec:statistical_analyses}). N = 471 per run "
            "(DeepSeek V4-Pro 467); people N = 473.")
    tex = _brt_tex("table", colspec,
                   "How often does each coded BRT behaviour occur in people and in each run? Prevalence (\\%) of the 15 "
                   "coded behaviours.", "tab:brt-prevalence", head1, cmid, head2,
                   _brt_body(d, order, 2 + 2 * len(runs), cells), note, "booktabs,tabularx", "3pt")
    (dirs["tables"] / "table_brt_prevalence_by_run.tex").write_text(tex, encoding="utf-8")
    diag.table("table_brt_prevalence_by_run.tex")


def write_brt_steering_table(d: pd.DataFrame, order: list[str], dirs: dict[str, Path], diag: Diagnostics) -> None:
    runs = list(TARGET_LABELS)

    def cells(oc: str) -> list[str]:
        out = []
        for r in runs:
            x = _brt_row(d, r, oc)
            out += [_brt_or_cell(x["f2_or"], x["f2_or_lo"], x["f2_or_hi"]), _brt_mark("U" if bool(x["c3_unstable"]) else "")]
        return out
    head1, cmid, row2 = _brt_heads(runs, 2, 2)
    head1 = " & ".join(["", head1])
    head2 = " & ".join(["Behaviour"] + row2)
    # the label column wraps (1.5 in) so seven interval columns fit the 9 in text height at \\footnotesize
    colspec = "@{}>{\\raggedright\\arraybackslash}p{1.5in}" + "r@{}l" * len(runs) + "@{}"
    note = ("Logistic regression of each behaviour on the told level (0--4); Wald 95\\% intervals, profile-likelihood "
            "intervals where the model is Firth-penalised (U rows). U: unstable (fewer than 20 events or non-events); its "
            "estimate is shown but carries no verdict. People have no told level; their per-level rates for the five "
            "behaviours with a reliable bCRT association are in Figure~\\ref{fig:brt_level_prevalence}.")
    tex = _brt_tex("sidewaystable", colspec,
                   "Does each behaviour change with the reflection level the persona was told? Steering: odds ratio per "
                   "told level, by run.", "tab:brt-steering", head1, cmid, head2,
                   _brt_body(d, order, 1 + 2 * len(runs), cells), note, "booktabs,rotating,array", "3pt")
    (dirs["tables"] / "table_brt_steering.tex").write_text(tex, encoding="utf-8")
    diag.table("table_brt_steering.tex")


def brt_transfer_letter(x: pd.Series) -> str:
    """The verdict letter from f3_verdict and c3_unstable as stored; I (introduces) replaces F where people show no
    detectable association (human interval includes 1) and the run's own interval excludes 1."""
    v = str(x["f3_verdict"])
    if bool(x["c3_unstable"]) != (v == "unstable"):
        raise SystemExit(f"STOP: {x['run_label']} {x['outcome']}: f3_verdict {v!r} disagrees with c3_unstable.")
    if v == "unstable":
        return "U"
    if v.startswith("fails"):
        human_null = x["human_or_lo"] <= 1 <= x["human_or_hi"]
        run_excl = not (x["f3_or_run_lo"] <= 1 <= x["f3_or_run_hi"])
        return "I" if human_null and run_excl else "F"
    if v not in VERDICT_LETTER:
        raise SystemExit(f"STOP: {x['run_label']} {x['outcome']}: unexpected f3_verdict {v!r}.")
    return VERDICT_LETTER[v]


def write_brt_transfer_table(d: pd.DataFrame, order: list[str], dirs: dict[str, Path],
                             diag: Diagnostics) -> dict[str, dict[str, int]]:
    runs = list(TARGET_LABELS[:3])
    letters = {r: {oc: brt_transfer_letter(_brt_row(d, r, oc)) for oc in order} for r in runs}
    counts = {r: {k: list(letters[r].values()).count(k) for k in ("T", "F", "I", "?", "U")} for r in runs}
    for r in runs:
        c = counts[r]
        got = (c["T"], c["F"] + c["I"], c["?"])
        if got != BRT_TABLE_GUARD_VERDICTS[r] or c["U"] != BRT_TABLE_GUARD_UNSTABLE[r]:
            raise SystemExit(f"STOP: {r} transfers / fails / indeterminate / unstable {got} / {c['U']}, Table 1 says "
                             f"{BRT_TABLE_GUARD_VERDICTS[r]} / {BRT_TABLE_GUARD_UNSTABLE[r]}.")
        sub = d[d["run_label"] == r]
        if not ((sub["f3_ci_type"] == "profile likelihood") == sub["c3_unstable"]).all():
            raise SystemExit(f"STOP: {r} F3 profile-likelihood rows are not exactly the unstable rows.")
    for r in TARGET_LABELS[3:]:   # the note says the Sonnet 4.6 conditions are not estimable
        if not d.loc[d["run_label"] == r, "f3_verdict"].str.startswith("not estimable").all():
            raise SystemExit(f"STOP: {r} has an estimable F3 verdict; the table note no longer holds.")

    def cells(oc: str) -> list[str]:
        h = _brt_row(d, runs[0], oc)
        out = [_brt_or_cell(h["human_or"], h["human_or_lo"], h["human_or_hi"])]
        for r in runs:
            x = _brt_row(d, r, oc)
            out += [_brt_or_cell(x["f3_or_run"], x["f3_or_run_lo"], x["f3_or_run_hi"]), letters[r][oc]]
        return out
    head1 = " & ".join(["", ""] + [f"\\multicolumn{{2}}{{c}}{{{tex_head(*BRT_TABLE_HEAD[r])}}}" for r in runs])
    cmid = " ".join(f"\\cmidrule(lr){{{3 + 2 * k}-{4 + 2 * k}}}" for k in range(len(runs)))
    head2 = " & ".join(["Behaviour", tex_head("People", "OR [95\\% CI]")] + [tex_head("OR", "{[95\\% CI]}"), "V"] * len(runs))
    colspec = "@{}" + tex_x(1.0) + "r" + "rc" * len(runs) + "@{}"
    note = ("Logistic regression on realised bCRT (per item); Wald 95\\% intervals, profile-likelihood intervals for "
            "Firth-penalised (U) models. Verdicts test the run-minus-human slope difference "
            "(Section~\\ref{subsec:statistical_analyses}): T transfers, F fails to transfer, I introduces (fails where people "
            "show no detectable association and the run's own interval excludes 1), ? indeterminate, U unstable. The four "
            "Claude Sonnet 4.6 conditions are not estimable: every agent scored 4/4 on the bCRT.")
    tex = _brt_tex("table", colspec,
                   "Does people's link between bCRT and each behaviour carry over to the runs? Odds ratio per bCRT item in "
                   "people and in the three runs whose bCRT scores vary, with the transfer verdict.", "tab:brt-transfer",
                   head1, cmid, head2, _brt_body(d, order, 2 + 2 * len(runs), cells), note, "booktabs,tabularx", "3pt")
    (dirs["tables"] / "table_brt_transfer.tex").write_text(tex, encoding="utf-8")
    diag.table("table_brt_transfer.tex")
    return counts


def write_brt_family_appendix_tables(dirs: dict[str, Path], diag: Diagnostics) -> None:
    d, order = _brt_table_frame(dirs)
    write_brt_prevalence_table(d, order, dirs, diag)
    write_brt_steering_table(d, order, dirs, diag)
    counts = write_brt_transfer_table(d, order, dirs, diag)
    for r, c in counts.items():
        print(f"BRT transfer letters, {condition_name(r)}: " + ", ".join(f"{k} {v}" for k, v in c.items()))


# ---------------------------------------------------------------------------------------------------
# CR-26 (D-309, PI ruling 3 Nov 2026): public-release mode.
# The public release ships this builder with the aggregate inputs only; participant-level inputs stay under
# controlled access. A step that reads one is skipped, and its outputs are listed in
# diagnostics/controlled_access_outputs.md ("needs the controlled-access records"), instead of the run
# stopping. STEP_NEEDS gives, per step, the participant-level files it reads (from a traced full-input build)
# and the steps whose results it uses ("*" = every other step). With every input present no step is
# skipped, no extra file is written and the build is unchanged.
# ---------------------------------------------------------------------------------------------------
_HUMAN_DERIVED_RUNS = ("CR_BASELINE_HUMAN_NUMERIC_CS46_S42", "CR_BASELINE_HUMAN_LEXICAL_CS46_S42")
_P_PERSON = [HUMAN_PERSON_FILE]                                                   # people: one row per respondent
_P_CRT2_LONG = [HUMAN_V2_LONG]                                                    # people's CRT answers
_P_BRT_LABELS = [Path("reports") / "brt_recode_v3" / "brt_labels_v3_human.csv"]   # people's BRT codes
_P_PERSON_FRAME = [Path("reports") / "cr13_CR_MAIN_SYNTH_NUM_SONNET5_S42" / "tables" / "by_dataset" / "human"
                   / "00_person_frame.csv"]                                      # people's person frame
_P_AGENT_FRAMES = [Path("reports") / f"cr13_{lab}" / "tables" / "by_dataset" / AGENT_DATASET / "00_person_frame.csv"
                   for lab in _HUMAN_DERIVED_RUNS]          # agents carrying each participant's own profile
_P_CRT_LONG_ALL = [Path("reports") / f"cr13_{lab}" / "tables" / "comparisons" / "05_crt_response_categories_long.csv"
                   for lab in TARGET_LABELS]                # per-response long tables (people's rows included)
_P_CRT_LONG_LURE = [Path("reports") / f"cr13_{lab}" / "tables" / "comparisons" / "05_crt_response_categories_long.csv"
                    for lab in LURE_PRODUCING_RUNS]
STEP_NEEDS: dict[str, tuple[list[Path], list[str]]] = {
    "build_crt_assets": (_P_CRT2_LONG, []),
    "build_by_level_assets": (_P_AGENT_FRAMES + _P_CRT_LONG_ALL, []),
    "build_brt_assets": (_P_AGENT_FRAMES + _P_BRT_LABELS, []),
    "build_liwc_appendix_assets": (_P_AGENT_FRAMES + _P_PERSON_FRAME, []),
    "build_scorecard_table": ([], ["build_distribution_assets", "build_crt_assets", "build_calibration_assets",
                                   "build_association_assets", "build_brt_assets"]),
    "build_item_exposure_assets": (_P_PERSON + _P_CRT2_LONG, []),
    "build_ladder_and_probe_assets": (_P_PERSON + _P_CRT2_LONG, []),
    "build_human_reference_table": ([], ["build_ladder_and_probe_assets"]),
    "fig_main1_lure_ladder": (_P_CRT_LONG_LURE, ["build_ladder_and_probe_assets", "build_crt_assets"]),
    "fig_main2_distribution_calibration": ([], ["build_distribution_assets", "build_calibration_assets"]),
    "fig_main2_focal_grid": ([], ["build_distribution_assets", "build_calibration_assets"]),
    "fig_main4_brt_gap": ([], ["build_brt_assets"]),
    "human_level_rates": (_P_PERSON + _P_CRT2_LONG, []),
    "write_human_level_rates": ([], ["human_level_rates"]),
    "fig_main5_lure_by_told_level": ([], ["build_by_level_assets", "human_level_rates"]),
    "fig_main4_brt_scatter_forest": ([], ["build_brt_assets", "copy_partii_appendix"]),
    "fig_appendix_brt_level_prevalence": ([], ["copy_partii_appendix"]),
    "write_brt_family_appendix_tables": ([], ["copy_partii_appendix"]),
    "fig_appendix_brt_categories": (_P_AGENT_FRAMES + _P_BRT_LABELS, []),
    "write_appendix_manifest": ([], ["copy_partii_appendix"]),
    "write_compile_test_wrapper": ([], ["*"]),
    "write_brt_prevalence_recovery_report": ([], ["build_brt_assets"]),
    "write_diagnostics": ([], ["*"]),
}
_SKIPPED: dict[str, list[str]] = {}


def run_step(name: str, fn, *args, **kwargs):
    """Run one build step, or skip it when a participant-level input or an upstream step it uses is missing."""
    needs, after = STEP_NEEDS.get(name, ([], []))
    root = repo_root()
    why = [f"participant-level input `{p.as_posix()}`" for p in needs if not (root / p).exists()]
    why += [f"output of step `{s}`" for s in after if s != "*" and s in _SKIPPED]
    if "*" in after and _SKIPPED:
        why.append("the full build (some steps were skipped)")
    if why:
        _SKIPPED[name] = why
        print(f"needs the controlled-access records, skipped: {name}")
        return None
    return fn(*args, **kwargs)


def write_controlled_access_report(dirs: dict[str, Path]) -> None:
    """Release mode only (some step skipped): list the skipped steps and why. Never written on a full build."""
    if not _SKIPPED:
        return
    lines = ["# Outputs that need the controlled-access records", "",
             "These build steps were skipped because a participant-level input is not in the public release. "
             "Their outputs need the controlled-access records (see the release README).", ""]
    for name, why in _SKIPPED.items():
        lines.append(f"- `{name}`: needs " + "; ".join(why))
    (dirs["diagnostics"] / "controlled_access_outputs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cr25_assets(root: Path, selected: dict[str, ReportCandidate], dirs: dict[str, Path], diag: Diagnostics,
                      told_rates: pd.DataFrame | None) -> None:
    run_step("build_item_exposure_assets", build_item_exposure_assets, root, dirs, diag)
    lp = run_step("build_ladder_and_probe_assets", build_ladder_and_probe_assets, root, dirs, diag)
    run_step("build_human_reference_table", lambda: build_human_reference_table(lp["ref"], dirs, diag))
    run_step("fig_main1_lure_ladder", lambda: fig_main1_lure_ladder(selected, lp["ladder"], lp["ref"], dirs, diag))
    run_step("fig_main2_distribution_calibration", fig_main2_distribution_calibration, dirs, diag)
    run_step("fig_main2_focal_grid", fig_main2_focal_grid, dirs, diag)
    fig_main3_delta_r(selected, dirs)
    diag.figure("main_fig3_delta_r_matrix.pdf")
    diag.figure("main_fig3_delta_r_matrix.png")
    run_step("fig_main4_brt_gap", fig_main4_brt_gap, dirs, diag)
    human_levels = run_step("human_level_rates", human_level_rates, root)
    run_step("write_human_level_rates", lambda: write_human_level_rates(human_levels, dirs, diag))   # S-8e (D-280(c), D-293(c)): the people line of Appendix Figure 7
    run_step("fig_main5_lure_by_told_level", lambda: fig_main5_lure_by_told_level(told_rates, human_levels, dirs, diag))
    partii = copy_partii_appendix(root, dirs, diag)
    run_step("fig_main4_brt_scatter_forest", fig_main4_brt_scatter_forest, root, dirs, diag)   # S-8: reads the copied Part II family tables
    run_step("fig_appendix_brt_level_prevalence", fig_appendix_brt_level_prevalence, root, dirs, diag)   # S-8a: reads CR-30 (D-253(d), D-255)
    run_step("write_brt_family_appendix_tables", write_brt_family_appendix_tables, dirs, diag)   # S-8b: reads the copied Part II family tables
    run_step("fig_appendix_brt_categories", fig_appendix_brt_categories, root, dirs, diag)   # S-8d: design C (D-268(b)), one rubric for every group
    write_cr25_captions(dirs, diag)
    run_step("write_appendix_manifest", write_appendix_manifest, partii, dirs, diag)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    root = repo_root()
    out = output_root(root)
    dirs = mkdirs(out)
    diag = Diagnostics()
    selected = select_reports(root, diag)

    build_schematic(dirs)
    diag.figure("fig1_cognition_suite_schematic.pdf")
    diag.figure("fig1_cognition_suite_schematic.png")
    distribution = build_distribution_assets(selected, dirs, diag)
    crt = run_step("build_crt_assets", build_crt_assets, selected, dirs, diag)
    told_rates = run_step("build_by_level_assets", build_by_level_assets, root, selected, dirs, diag)
    brt, brt_flagged = run_step("build_brt_assets", build_brt_assets, root, selected, dirs, diag) or (None, None)
    association = build_association_assets(selected, dirs, diag)
    calibration = build_calibration_assets(selected, dirs, diag)
    run_step("build_liwc_appendix_assets", build_liwc_appendix_assets, root, selected, dirs, diag)
    build_experimental_runs_table(selected, dirs, diag)
    run_step("build_scorecard_table", build_scorecard_table, distribution, crt, calibration, association, brt,
             brt_flagged, dirs, diag)
    build_prompt_screening_table(root, dirs, diag)
    build_cr25_assets(root, selected, dirs, diag, told_rates)
    run_step("write_compile_test_wrapper", write_compile_test_wrapper, dirs, diag)
    write_captions(dirs, diag)
    run_step("write_brt_prevalence_recovery_report", write_brt_prevalence_recovery_report, root, selected, dirs, diag,
             brt)
    run_step("write_diagnostics", write_diagnostics, dirs, selected, diag)
    write_controlled_access_report(dirs)

    print(f"Wrote camera-ready assets to {rel(out, root)}")
    print(f"Selected {len(selected)} of {len(TARGET_LABELS)} target labels.")
    if diag.missing_labels:
        print("Missing labels: " + ", ".join(diag.missing_labels))


if __name__ == "__main__":
    main()
