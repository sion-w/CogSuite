"""Required input files and columns for the per-run analysis."""
from __future__ import annotations

from pathlib import Path

from socsim.common import load_yaml

REQUIRED_INPUT_FILES = [
    "person_with_liwc_brt_coded.csv",
    "synthetic_personas.csv",
]

RESOURCE_FILES = ["answer_keys.yaml", "config.yaml"]

ANSWER_KEY_TOP_LEVEL = ["crt2", "bcrt", "dcrt"]

HUMAN_REQUIRED = [
    "WritingTask_Q1_x",
    "WritingTask_Q2_x",
    "CRT2_Total",
    "BCRT_Total",
    "CRT_Combined",
    "NFC_Total",
    "BFI_Extraversion",
    "BFI_Agreeableness",
    "BFI_Conscientiousness",
    "BFI_Neuroticism",
    "BFI_Openness",
]

SYNTH_REQUIRED = [
    "CRT2_Total_adj",
    "BCRT_Total",
    "NFC_Total",
    "BFI_Extraversion",
    "BFI_Agreeableness",
    "BFI_Conscientiousness",
    "BFI_Neuroticism",
    "BFI_Openness",
]

COGSIM_PERSON_REQUIRED = [
    "run_id",
    "agent_id",
    "BCRT_Total_specified",
    "NFC_Total_specified",
    "CRT2_Total_adj_specified",
    "CRT_Combined_specified",
    "BFI_Extraversion_specified",
    "BFI_Agreeableness_specified",
    "BFI_Conscientiousness_specified",
    "BFI_Neuroticism_specified",
    "BFI_Openness_specified",
]

COGSIM_ITEMS_REQUIRED = ["run_id", "agent_id", "item_id", "item_type", "response_text_item", "response_numeric"]
SOCSIM_PERSON_REQUIRED = ["persona_id", "BCRT_Total", "NFC_Total", "CRT2_Total_adj"]
SOCSIM_ITEMS_REQUIRED = ["persona_id", "test_name", "item_id", "item_type", "response_text", "response_numeric"]


def validate_required_inputs(inputs_dir: Path) -> None:
    missing = [f for f in REQUIRED_INPUT_FILES if not (inputs_dir / f).exists()]
    if missing:
        raise ValueError(f"Missing required files in data/inputs: {missing}")


def validate_answer_keys(path: Path) -> None:
    payload = load_yaml(path)
    missing = [k for k in ANSWER_KEY_TOP_LEVEL if k not in payload]
    if missing:
        raise ValueError(f"Missing required keys in answer_keys.yaml: {missing}")
