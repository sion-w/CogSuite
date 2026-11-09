"""CRT item scoring for the per-run pipeline.

Every CRT2/bCRT/dCRT answer is scored with ``score/scoring_v2`` (``score_v2.score``): answer
keys first, then the adjudication cache, and only answers missing from the cache go to the
live three-provider adjudicator (the item stem reaches it via ``score_v2.item_meta``). Each
labelled row carries provenance columns: scorer_version, key_version, prompt_version,
adjudicator_models, run_id and date.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from socsim.common import require_columns
from socsim.scoring.crt_categories import add_item_response_categories

REPO = Path(__file__).resolve().parents[3]          # repository root
SCORE_DIR = REPO / "score"
sys.path.insert(0, str(SCORE_DIR))
from scoring_v2.score_v2 import (  # noqa: E402
    item_meta as sv2_item_meta,
    load_cache as sv2_load_cache,
    score as sv2_score,
)

_KEYS_PATH = SCORE_DIR / "scoring_v2" / "answer_keys_v2.yaml"
_PROMPT_PATH = SCORE_DIR / "scoring_v2" / "evaluator_prompt_v2.txt"
_CACHE_PATH = SCORE_DIR / "scoring_v2" / "cache" / "adjudications.jsonl"

_SV2_KEY_VERSION = hashlib.sha256(_KEYS_PATH.read_bytes()).hexdigest()[:12]
_SV2_PROMPT_VERSION = hashlib.sha256(_PROMPT_PATH.read_bytes()).hexdigest()[:12]

# Spend gate for live residual adjudication. Cache hits cost nothing.
CRT_ADJUDICATION_SPEND_GATE_USD = 10.0
_CRT_SPEND_USD = 0.0
_CRT_SPEND_PATH = _CACHE_PATH.parent / "crt_adjudication_spend_usd.txt"
_PANEL_GUARD_DONE = False


def _load_crt_spend() -> float:
    if _CRT_SPEND_PATH.exists():
        try:
            return float(_CRT_SPEND_PATH.read_text(encoding="utf-8").strip() or 0.0)
        except ValueError:
            return 0.0
    return 0.0


def _save_crt_spend() -> None:
    _CRT_SPEND_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CRT_SPEND_PATH.write_text(f"{_CRT_SPEND_USD:.6f}\n", encoding="utf-8")


# The DeepSeek seat is DeepSeek V4-Flash, Fireworks-hosted (the model behind the shipped caches).
# COGSUITE_DEEPSEEK_SEAT_MODEL names another Fireworks model for new coding, e.g. when that model is not served.
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_DEEPSEEK_V4_FLASH = os.environ.get("COGSUITE_DEEPSEEK_SEAT_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731")

PRICE = {
    "gpt-5.6-sol": {"in": 4.0, "out": 20.0},
    "gemini-3.5-flash": {"in": 1.50, "out": 9.00},
    FIREWORKS_DEEPSEEK_V4_FLASH: {"in": 0.22, "out": 0.66},
}

ANNOTATORS = [
    {
        "model_id": "gpt-5.6-sol",
        "api_style": "responses",
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "extra_params": {"reasoning": {"effort": "low"}},
        "extra_body": None,
        "setting": 'reasoning_effort="low"',
    },
    {
        "model_id": "gemini-3.5-flash",
        "api_style": "chat_completions",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "extra_params": {},
        "extra_body": {
            "extra_body": {"google": {"thinking_config": {"thinking_level": "minimal"}}}
        },
        "setting": 'thinking_level="minimal"',
    },
    {
        "model_id": FIREWORKS_DEEPSEEK_V4_FLASH,
        "api_style": "chat_completions",
        "base_url": FIREWORKS_BASE_URL,
        "key_env": "FIREWORKS_API_KEY",
        "extra_params": {},
        "extra_body": {"thinking": {"type": "disabled"}},
        "setting": 'thinking={"type":"disabled"}',
    },
]


def crt_adjudication_spend_usd() -> float:
    return _CRT_SPEND_USD


def panel_guard_deepseek_seat() -> None:
    """The deepseek seat must be Fireworks-hosted; STOP on the first-party host."""
    ds = next(a for a in ANNOTATORS if "deepseek" in a["model_id"])
    base = (ds.get("base_url") or "").rstrip("/")
    print(f"PANEL_GUARD (CRT) deepseek seat base_url={base}", flush=True)
    print(f"PANEL_GUARD (CRT) deepseek seat model_id={ds['model_id']}", flush=True)
    print(f"PANEL_GUARD (CRT) deepseek seat key_env={ds['key_env']}", flush=True)
    if "api.deepseek.com" in base or base.endswith("deepseek.com"):
        raise SystemExit(
            "STOP: PANEL GUARD — deepseek seat base_url is first-party DeepSeek "
            f"({base}); votes are only taken through the Fireworks-hosted seat."
        )
    if "api.fireworks.ai" not in base:
        raise SystemExit(
            f"STOP: PANEL GUARD — deepseek seat base_url is not Fireworks ({base})."
        )
    if ds["key_env"] != "FIREWORKS_API_KEY":
        raise SystemExit(
            f"STOP: PANEL GUARD — deepseek seat key_env must be FIREWORKS_API_KEY, got {ds['key_env']}"
        )
    if FIREWORKS_BASE_URL.rstrip("/") not in base:
        raise SystemExit(
            f"STOP: PANEL GUARD — expected {FIREWORKS_BASE_URL}, got {base}"
        )


def _ordinal_from_label(label: str | None) -> float:
    if label == "correct":
        return 2.0
    if label == "intuitive":
        return 1.0
    if label == "other":
        return 0.0
    return float("nan")


def _is_blank_raw_answer(raw_ans) -> bool:
    """Caller-side blank guard: scoring_v2 has no missing-data concept.

    Only an empty answer is missing. "none" is an answer (the correct one to CRT2 item 4)
    and goes to scoring_v2 like any other text.
    """
    if pd.isna(raw_ans):
        return True
    a = str(raw_ans).strip()
    return a == "" or a.lower() == "nan"


def _load_prompt_parts() -> tuple[str, str]:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    if not text.startswith("SYSTEM\n"):
        raise RuntimeError("evaluator_prompt_v2.txt must start with SYSTEM\\n")
    rest = text[len("SYSTEM\n") :]
    if "\nUSER\n" not in rest:
        raise RuntimeError("evaluator_prompt_v2.txt missing USER section")
    system, user_tmpl = rest.split("\nUSER\n", 1)
    return system.strip(), user_tmpl.strip()


def _load_api_adapter():
    """Load the simulator module, whose provider adapters make the adjudication calls."""
    sim = REPO / "simulate" / "output" / "data_analysis_odd_persona_trait_experiment" / "simulation_code_iter_1.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("simulator_api_adapter", sim)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_label(raw: str) -> str:
    import re

    s = (raw or "").strip()
    labels = ("correct", "intuitive", "other")
    try:
        m = re.search(r"\{[^{}]*\}", s, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            lab = str(obj.get("label", "")).strip().lower()
            if lab in labels:
                return lab
            if lab in {"other_incorrect", "other-incorrect"}:
                return "other"
            if lab in {"intuitive_incorrect", "intuitive-incorrect"}:
                return "intuitive"
    except Exception:
        pass
    low = s.lower()
    for lab in labels:
        if re.search(rf"\b{lab}\b", low):
            return lab
    if "other_incorrect" in low:
        return "other"
    if "intuitive_incorrect" in low:
        return "intuitive"
    raise ValueError(f"could not parse label from annotator output: {raw!r}")


def _call_one(adapter, ann: dict, system: str, user: str) -> tuple[str, dict]:
    if ann["api_style"] == "responses":
        result = adapter.call_gpt5_with_responses_api(
            system_prompt=system,
            user_prompt=user,
            model=ann["model_id"],
            max_output_tokens=256,
            base_url=ann["base_url"],
            key_env=ann["key_env"],
            extra_params=ann["extra_params"],
            max_retries=5,
        )
    else:
        result = adapter.call_openai_chat_completions_api(
            system_prompt=system,
            user_prompt=user,
            model=ann["model_id"],
            max_tokens=256,
            base_url=ann["base_url"],
            key_env=ann["key_env"],
            extra_params=ann["extra_params"] or None,
            extra_body=ann["extra_body"],
            max_retries=5,
        )
    label = _parse_label(result.text)
    in_est = max(1, (len(system) + len(user)) // 4)
    out_est = max(1, len(result.text or "") // 4)
    return label, {"input_tokens": in_est, "output_tokens": out_est}


def _make_adjudicator(cache: dict):
    """Live three-provider residual adjudicator. Mutates cache; appends jsonl.

    PANEL GUARD runs before the first live vote. Spend gate $10.
    """
    global _CRT_SPEND_USD, _PANEL_GUARD_DONE
    adapter = None
    system = user_tmpl = None

    def adjudicator(item_id: str, normalised: str) -> list[str]:
        global _CRT_SPEND_USD, _PANEL_GUARD_DONE
        nonlocal adapter, system, user_tmpl
        if not _PANEL_GUARD_DONE:
            panel_guard_deepseek_seat()
            _PANEL_GUARD_DONE = True
            _CRT_SPEND_USD = _load_crt_spend()
            print(f"PANEL_GUARD (CRT) loaded prior live-adjudication spend=${_CRT_SPEND_USD:.4f}", flush=True)
        if adapter is None:
            adapter = _load_api_adapter()
            system, user_tmpl = _load_prompt_parts()
        if _CRT_SPEND_USD >= CRT_ADJUDICATION_SPEND_GATE_USD:
            raise SystemExit(
                f"STOP: CRT residual adjudication spend ${_CRT_SPEND_USD:.4f} "
                f">= gate ${CRT_ADJUDICATION_SPEND_GATE_USD:.2f}"
            )
        meta = sv2_item_meta(item_id)
        user = (
            user_tmpl.replace("{item_stem}", meta.get("item_stem", ""))
            .replace("{correct_plain}", meta.get("correct_plain", ""))
            .replace("{intuitive_plain}", meta.get("intuitive_plain", ""))
            .replace("{normalised_answer}", normalised if normalised else "(blank)")
        )
        votes: list[str] = []
        vote_records: list[dict] = []
        for ann in ANNOTATORS:
            label, usage = _call_one(adapter, ann, system, user)
            votes.append(label)
            vote_records.append(
                {"model_id": ann["model_id"], "setting": ann["setting"], "label": label}
            )
            mid = ann["model_id"]
            _CRT_SPEND_USD += (
                usage["input_tokens"] / 1e6 * PRICE[mid]["in"]
                + usage["output_tokens"] / 1e6 * PRICE[mid]["out"]
            )
            if _CRT_SPEND_USD > CRT_ADJUDICATION_SPEND_GATE_USD:
                _save_crt_spend()
                raise SystemExit(
                    f"STOP: CRT residual adjudication spend ${_CRT_SPEND_USD:.4f} "
                    f"exceeded gate ${CRT_ADJUDICATION_SPEND_GATE_USD:.2f}"
                )
        counts = {lab: votes.count(lab) for lab in ("correct", "intuitive", "other")}
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_lab, top_n = ranked[0]
        n_at_top = sum(1 for _, n in ranked if n == top_n)
        if top_n == 3:
            final, agreement = top_lab, "unanimous"
        elif top_n == 2 and n_at_top == 1:
            final, agreement = top_lab, "majority"
        else:
            final, agreement = "other", "split"
        rec = {
            "item_id": item_id,
            "normalised": normalised,
            "votes": vote_records,
            "label": final,
            "agreement": agreement,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cache[(item_id, normalised)] = rec
        _save_crt_spend()
        print(
            f"[scoring_v2] adjudicated residual {item_id!r} {normalised[:60]!r} → {final} "
            f"({agreement}) spend=${_CRT_SPEND_USD:.4f}",
            flush=True,
        )
        return votes

    return adjudicator


def _adjudicator_models(votes) -> str:
    ids = [v["model_id"] for v in votes if isinstance(v, dict) and "model_id" in v]
    return ";".join(ids)


def score_crt_items_with_llm(
    items_df: pd.DataFrame,
    answer_keys: dict,
    prompt_col: str = "response_text",
    existing_scored: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score CRT2/bCRT/dCRT rows via scoring_v2. `answer_keys` is unused: scoring_v2 loads
    answer_keys_v2.yaml itself."""
    del answer_keys  # scoring_v2 loads answer_keys_v2.yaml itself
    require_columns(items_df, ["entity_id", "item_id", prompt_col], "agent_items")
    out = items_df.copy()
    mask = out["item_id"].astype(str).str.startswith(("crt2_", "bcrt_", "dcrt_"))
    crt = out.loc[mask].copy()
    result_cols = [
        "llm_label",
        "llm_correct",
        "llm_ordinal012",
        "llm_reason",
        "scorer_version",
        "key_version",
        "prompt_version",
        "adjudicator_models",
        "run_id",
        "date",
    ]
    for col in result_cols:
        crt[col] = np.nan

    run_id = f"cr13-crt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_date = datetime.now(timezone.utc).date().isoformat()
    global _CRT_SPEND_USD
    _CRT_SPEND_USD = _load_crt_spend()
    cache = sv2_load_cache()
    adjudicator = _make_adjudicator(cache)

    n_key = n_llm = n_missing = n_unadj = 0
    for idx, row in crt.iterrows():
        raw = row[prompt_col]
        item_id = str(row["item_id"])
        if _is_blank_raw_answer(raw):
            crt.at[idx, "llm_label"] = np.nan
            crt.at[idx, "llm_correct"] = np.nan
            crt.at[idx, "llm_ordinal012"] = np.nan
            crt.at[idx, "llm_reason"] = "item_missing"
            crt.at[idx, "scorer_version"] = "scoring_v2"
            crt.at[idx, "key_version"] = _SV2_KEY_VERSION
            crt.at[idx, "prompt_version"] = _SV2_PROMPT_VERSION
            crt.at[idx, "adjudicator_models"] = ""
            crt.at[idx, "run_id"] = run_id
            crt.at[idx, "date"] = run_date
            n_missing += 1
            continue
        res = sv2_score(item_id, raw, adjudicator=adjudicator, cache=cache)
        label = res["label"]
        method = res["method"]
        crt.at[idx, "llm_label"] = label
        crt.at[idx, "llm_correct"] = (
            1.0 if label == "correct" else 0.0 if label in {"intuitive", "other"} else np.nan
        )
        crt.at[idx, "llm_ordinal012"] = _ordinal_from_label(label)
        crt.at[idx, "llm_reason"] = method
        crt.at[idx, "scorer_version"] = "scoring_v2"
        crt.at[idx, "key_version"] = _SV2_KEY_VERSION
        crt.at[idx, "prompt_version"] = _SV2_PROMPT_VERSION
        crt.at[idx, "adjudicator_models"] = (
            _adjudicator_models(res.get("votes") or []) if method == "llm_majority" else ""
        )
        crt.at[idx, "run_id"] = run_id
        crt.at[idx, "date"] = run_date
        if method and str(method).startswith("key_"):
            n_key += 1
        elif method == "llm_majority":
            n_llm += 1
        elif method == "residual_unadjudicated":
            n_unadj += 1

    crt["final_correct"] = crt["llm_correct"]
    crt["final_ordinal012"] = crt["llm_ordinal012"]
    crt = add_item_response_categories(crt)
    labelled = int(crt["llm_label"].notna().sum())
    print(
        f"CRT scoring_v2 complete: labelled={labelled}/{len(crt)} "
        f"key={n_key} llm_majority={n_llm} item_missing={n_missing} "
        f"residual_unadjudicated={n_unadj} live_spend=${_CRT_SPEND_USD:.4f}",
        file=sys.stderr,
        flush=True,
    )
    if n_unadj:
        raise SystemExit(
            f"STOP: {n_unadj} residual_unadjudicated CRT rows after scoring_v2 "
            "(the adjudicator resolves every residual)."
        )
    return crt
