from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from socsim.common import (
    AGENT_ITEM_TEXT_COLS,
    CRT_ITEM_TEXT_COLS,
    ensure_dir,
    read_csv_keep_text,
    read_csv_strict,
    require_columns,
)
from socsim.ingest.schema import (
    COGSIM_ITEMS_REQUIRED,
    COGSIM_PERSON_REQUIRED,
    HUMAN_REQUIRED,
    SOCSIM_ITEMS_REQUIRED,
    SOCSIM_PERSON_REQUIRED,
    SYNTH_REQUIRED,
    validate_answer_keys,
    validate_required_inputs,
)


def copy_resource_inputs(inputs_dir: Path, resources_dir: Path) -> None:
    ensure_dir(resources_dir)
    for name in ["answer_keys.yaml", "config.yaml"]:
        shutil.copy2(inputs_dir / name, resources_dir / name)
    validate_answer_keys(resources_dir / "answer_keys.yaml")


def load_required_inputs(inputs_dir: Path) -> dict:
    validate_required_inputs(inputs_dir)
    human = read_csv_strict(
        inputs_dir / "person_with_liwc_brt_coded.csv",
        HUMAN_REQUIRED,
        "person_with_liwc_brt_coded.csv",
        text_cols=CRT_ITEM_TEXT_COLS,
    )
    synth = read_csv_strict(inputs_dir / "synthetic_personas.csv", SYNTH_REQUIRED, "synthetic_personas.csv")
    require_columns(human, HUMAN_REQUIRED, "person_with_liwc_brt_coded.csv")
    require_columns(synth, SYNTH_REQUIRED, "synthetic_personas.csv")
    return {"human": human, "synthetic": synth}


def load_optional_cogsim(inputs_dir: Path) -> dict | None:
    base = inputs_dir / "cogsim_runs"
    if not base.exists():
        return None
    run_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not run_dirs:
        return None
    people_frames = []
    item_frames = []
    manifests = []
    for run in run_dirs:
        p_path = run / "wide_agents.csv"
        i_path = run / "long_responses.csv"
        m_path = run / "manifest.json"
        if not p_path.exists() or not i_path.exists() or not m_path.exists():
            continue
        p_df = pd.read_csv(p_path)
        i_df = read_csv_keep_text(i_path, AGENT_ITEM_TEXT_COLS)
        require_columns(p_df, COGSIM_PERSON_REQUIRED, f"{run.name}/wide_agents.csv")
        require_columns(i_df, COGSIM_ITEMS_REQUIRED, f"{run.name}/long_responses.csv")
        people_frames.append(p_df)
        item_frames.append(i_df)
        with m_path.open("r", encoding="utf-8") as f:
            manifests.append(json.load(f))
    if not people_frames:
        return None
    return {
        "person": pd.concat(people_frames, ignore_index=True),
        "items": pd.concat(item_frames, ignore_index=True),
        "manifests": manifests,
    }


def load_optional_socsim(inputs_dir: Path) -> dict | None:
    pairs = _discover_socsim_input_pairs(inputs_dir)
    if not pairs:
        return None
    if len(pairs) > 1:
        labels = ", ".join(label or "unlabelled" for label, _, _ in pairs)
        raise ValueError(
            "Multiple SOCSIM input pairs found in data/inputs. Keep only one pair for a pipeline run "
            f"or move the others aside. Detected labels: {labels}"
        )
    run_label, person_path, items_path = pairs[0]
    person = pd.read_csv(person_path)
    items = read_csv_keep_text(items_path, AGENT_ITEM_TEXT_COLS)
    require_columns(person, SOCSIM_PERSON_REQUIRED, person_path.name)
    require_columns(items, SOCSIM_ITEMS_REQUIRED, items_path.name)
    return {"person": person, "items": items, "run_label": run_label, "person_path": person_path, "items_path": items_path}


def _discover_socsim_input_pairs(inputs_dir: Path) -> list[tuple[str | None, Path, Path]]:
    person_re = re.compile(r"^socsim_per_persona(?:_results)?(?:_(?P<label>[^/]+))?\.csv$")
    item_re = re.compile(r"^socsim_per_item_results(?:_(?P<label>[^/]+))?\.csv$")
    people: dict[str | None, Path] = {}
    items: dict[str | None, Path] = {}
    for path in inputs_dir.glob("socsim_per_*.csv"):
        person_match = person_re.match(path.name)
        item_match = item_re.match(path.name)
        if person_match:
            people[person_match.group("label")] = path
        elif item_match:
            items[item_match.group("label")] = path
    labels = sorted(set(people).intersection(items), key=lambda value: value or "")
    return [(label, people[label], items[label]) for label in labels]
