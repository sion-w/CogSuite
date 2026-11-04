"""v1 CRT LLM classifier (DEAD). Submitted-manuscript scorer.

CR-13 wired socsim.scoring.crt_score_items to scoring_v2.score(). This module
is retained as the archived v1 path and is not imported by the live scorer.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# D-27: keys live in a gitignored `.env` at THIS repo's root. Resolve relative
# to this file's location (never an absolute home-directory path, per D-18)
# so this module works whether it is imported via `python -m socsim.cli` or
# directly (e.g. rebuttal scripts). Idempotent / does not override an
# already-exported environment variable.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

_client = None


def _get_client():
    global _client
    try:
        from openai import OpenAI
    except Exception:
        return None, "import_openai_failed"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "no_api_key"
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client, "ok"


def _normalize_label(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    label = text.strip().split()[0].strip().lower()
    if label in {"correct", "intuitive", "other"}:
        return label
    return None


def _extract_output_text(resp) -> Optional[str]:
    raw = getattr(resp, "output_text", None)
    if raw:
        return raw
    try:
        return resp.output[0].content[0].text
    except Exception:
        return None


def classify_with_llm_debug(item_id: str, prompt_text: str, answer_text: str, rubric: Dict) -> Dict[str, Optional[str]]:
    """
    Return {"label": one of {correct,intuitive,other} or None, "reason": str}
    Uses OpenAI Responses API first, then chat-completions fallback.
    """
    client, reason = _get_client()
    if client is None:
        return {"label": None, "reason": reason}

    model = os.environ.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL_SNAPSHOT", "gpt-5-nano"))
    fallback_model = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
    sys_prompt = (
        "You are a strict classifier for CRT-style free-text answers.\n"
        "Use only these labels: correct, intuitive, other.\n"
        "Return valid JSON with a single key: label."
    )
    user_prompt = (
        f"ITEM_ID: {item_id}\n"
        f"QUESTION: {prompt_text}\n"
        f"ANSWER: {answer_text}\n"
        f"RUBRIC: {rubric}\n"
        "Choose label based on rubric and answer semantics."
    )

    def _do_call_responses(which_model: str):
        resp = client.responses.create(
            model=which_model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": sys_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            text={"format": {"type": "json_object"}},
        )
        raw = _extract_output_text(resp)
        try:
            payload = json.loads(raw) if raw else {}
            label = _normalize_label(payload.get("label"))
        except Exception:
            label = _normalize_label(raw)
        if label:
            return {"label": label, "reason": "ok", "model": which_model}
        return {"label": None, "reason": f"invalid_label:{str(raw)[:120]}", "model": which_model}

    def _do_call_chat(which_model: str):
        resp = client.chat.completions.create(
            model=which_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        try:
            payload = json.loads(raw) if raw else {}
            label = _normalize_label(payload.get("label"))
        except Exception:
            label = _normalize_label(raw)
        if label:
            return {"label": label, "reason": "ok", "model": which_model}
        return {"label": None, "reason": f"invalid_label:{str(raw)[:120]}", "model": which_model}

    for which_model in [model, fallback_model]:
        try:
            return _do_call_responses(which_model)
        except Exception as e:
            msg = str(e)
            try:
                return _do_call_chat(which_model)
            except Exception as e2:
                msg2 = str(e2)
                if which_model == fallback_model:
                    return {"label": None, "reason": f"api_exception:{e2.__class__.__name__}:{msg2[:200]}", "model": which_model}
                continue
    return {"label": None, "reason": "api_exception:unknown", "model": model}


def classify_with_llm(item_id: str, prompt_text: str, answer_text: str, rubric: Dict) -> Optional[str]:
    out = classify_with_llm_debug(item_id, prompt_text, answer_text, rubric)
    return out.get("label")
