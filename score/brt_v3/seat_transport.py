"""Seat transport for the BRT v3 panel coder.

``prompt_v3.txt`` keeps its SYSTEM/USER sections. At send time the coding rules (the file's
USER section up to the response placeholders) join the SYSTEM section, so the whole rubric is a
byte-stable system message and only the two scenario responses go in the user message.

Two transports:

* sync: one call per seat and entity through the simulator's provider adapters
  (``call_gpt5_with_responses_api`` for the OpenAI Responses seat,
  ``call_openai_chat_completions_api`` for the OpenAI-compatible Gemini and Fireworks seats);
* batch: the OpenAI Responses seat through the OpenAI Batch API (one job for every entity still
  to be coded); the other seats stay sync.

Batch results are written into the same per-call cache schema as sync results, with
``transport="batch"``.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_OUT = 8192


@dataclass
class SeatCallResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    transport: str = "sync"


def prompt_cache_key(prompt_sha: str) -> str:
    """Provider-side prompt-cache routing key: the prompt file's SHA-256."""
    return prompt_sha


def split_rubric_and_entity(system_file: str, user_file: str) -> tuple[str, str]:
    """Return (rubric system message, per-entity user template).

    The rubric is the file SYSTEM section plus the file USER section up to the
    ``BRT1_response`` placeholder; the user template keeps only ``{brt_1}``/``{brt_2}``.
    """
    marker = 'BRT1_response: """{brt_1}"""'
    if marker not in user_file:
        raise RuntimeError("prompt user template missing BRT1_response placeholder")
    head, _, tail = user_file.partition(marker)
    rubric = (system_file.rstrip() + "\n\n" + head).strip()
    entity_tmpl = (marker + tail).strip()
    if "{brt_1}" not in entity_tmpl or "{brt_2}" not in entity_tmpl:
        raise RuntimeError("entity template lost {brt_1}/{brt_2}")
    if "{brt_1}" in rubric or "{brt_2}" in rubric:
        raise RuntimeError("per-entity placeholders leaked into rubric system message")
    return rubric, entity_tmpl


def render_entity_user(user_tmpl: str, brt_1: str, brt_2: str) -> str:
    return user_tmpl.replace("{brt_1}", brt_1).replace("{brt_2}", brt_2)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _jsonable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def _extract_openai_text(payload: dict[str, Any]) -> str:
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks)


# --------------------------------------------------------------------------- #
# Sync transport: the simulator's provider adapters.
# --------------------------------------------------------------------------- #
def call_seat(
    ev: dict[str, Any],
    system: str,
    user: str,
    *,
    adapter: Any,
    prompt_sha: str,
    max_retries: int = 5,
) -> SeatCallResult:
    """One sync call for one seat. ``adapter`` is the loaded simulator module."""
    if ev["api_style"] == "responses":
        extra = dict(ev.get("extra_params") or {})
        extra["prompt_cache_key"] = prompt_cache_key(prompt_sha)
        result = adapter.call_gpt5_with_responses_api(
            system_prompt=system,
            user_prompt=user,
            model=ev["model_id"],
            max_output_tokens=MAX_OUT,
            base_url=ev.get("base_url"),
            key_env=ev["key_env"],
            extra_params=extra,
            max_retries=max_retries,
        )
    elif ev["api_style"] == "chat_completions":
        result = adapter.call_openai_chat_completions_api(
            system_prompt=system,
            user_prompt=user,
            model=ev["model_id"],
            max_tokens=MAX_OUT,
            base_url=ev.get("base_url"),
            key_env=ev["key_env"],
            extra_params=ev.get("extra_params") or None,
            extra_body=ev.get("extra_body"),
            max_retries=max_retries,
        )
    else:
        raise RuntimeError(f"unhandled seat {ev.get('evaluator_id')}")
    return SeatCallResult(text=result.text, usage={}, transport="sync")


# --------------------------------------------------------------------------- #
# Batch transport: OpenAI Batch API for the Responses seat.
# --------------------------------------------------------------------------- #
def openai_batch_body(system: str, user: str, *, model_id: str, prompt_sha: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system,
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user}],
            },
        ],
        "max_output_tokens": MAX_OUT,
        "reasoning": {"effort": "low"},
        "prompt_cache_key": prompt_cache_key(prompt_sha),
        "prompt_cache_options": {"mode": "explicit"},
    }


def _custom_id(pair_hash: str, evaluator_id: str, entity_id: str = "") -> str:
    """Unique batch row id. The entity id is included because two entities can share
    identical BRT texts (and therefore the same content-addressed pair hash)."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "", evaluator_id.split("@")[0])[:8]
    ent = re.sub(r"[^a-zA-Z0-9]+", "", str(entity_id))[:24]
    return f"{pair_hash[:16]}_{safe}_{ent}"[:64]


def submit_openai_batch(
    jobs: list[dict[str, Any]],
    *,
    api_key: str,
    work_dir: Path,
    prompt_sha: str,
) -> dict[str, Any]:
    from openai import OpenAI

    work_dir.mkdir(parents=True, exist_ok=True)
    jsonl = work_dir / "openai_batch_input.jsonl"
    id_map: dict[str, dict[str, str]] = {}
    with jsonl.open("w", encoding="utf-8") as fh:
        for job in jobs:
            cid = _custom_id(job["pair_hash"], job["evaluator_id"], job["entity_id"])
            id_map[cid] = {
                "pair_hash": job["pair_hash"],
                "evaluator_id": job["evaluator_id"],
                "entity_id": job["entity_id"],
            }
            line = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": openai_batch_body(
                    job["system"], job["user"], model_id=job["model_id"], prompt_sha=prompt_sha
                ),
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    (work_dir / "openai_custom_id_map.json").write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    client = OpenAI(api_key=api_key)
    with jsonl.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h"
    )
    rec = {
        "provider": "openai",
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "n": len(jobs),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    (work_dir / "openai_batch_job.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[batch] openai submitted {batch.id} n={len(jobs)}", flush=True)
    return rec


def poll_openai_batch(work_dir: Path, *, api_key: str, timeout_s: float = 6 * 3600) -> list[dict[str, Any]]:
    from openai import OpenAI

    rec = json.loads((work_dir / "openai_batch_job.json").read_text(encoding="utf-8"))
    id_map = json.loads((work_dir / "openai_custom_id_map.json").read_text(encoding="utf-8"))
    client = OpenAI(api_key=api_key)
    t0 = time.time()
    while True:
        batch = client.batches.retrieve(rec["batch_id"])
        status = batch.status
        print(f"[batch] openai {rec['batch_id']} status={status}", flush=True)
        if status in {"completed", "failed", "expired", "cancelled"}:
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"OpenAI batch {rec['batch_id']} still {status} after {timeout_s}s")
        time.sleep(20 if status in {"validating", "in_progress", "finalizing"} else 8)
    if batch.status != "completed":
        raise RuntimeError(f"OpenAI batch {rec['batch_id']} ended {batch.status}: {batch.errors}")
    out_id = batch.output_file_id
    if not out_id:
        raise RuntimeError(f"OpenAI batch {rec['batch_id']} completed with no output_file_id")
    content = client.files.content(out_id)
    text = content.text if hasattr(content, "text") else content.read().decode("utf-8")
    (work_dir / "openai_batch_output.jsonl").write_text(text, encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        meta = id_map.get(obj.get("custom_id")) or {}
        body = (obj.get("response") or {}).get("body") or {}
        rows.append(
            {
                **meta,
                "text": _extract_openai_text(body),
                "usage": _jsonable(body.get("usage") or {}),
                "transport": "batch",
                "error": obj.get("error"),
            }
        )
    return rows


def _openai_batch_is_resumable(job_path: Path, *, api_key: str) -> bool:
    """An existing job is reused while in flight or completed; failed, expired or
    cancelled jobs are not."""
    try:
        rec = json.loads(job_path.read_text(encoding="utf-8"))
        batch_id = rec.get("batch_id")
        if not batch_id:
            return False
        from openai import OpenAI

        batch = OpenAI(api_key=api_key).batches.retrieve(batch_id)
        status = getattr(batch, "status", "")
        print(f"[batch] openai {batch_id} prior status={status}", flush=True)
        return status in {"validating", "in_progress", "finalizing", "completed"}
    except Exception as exc:
        print(f"[batch] openai prior job not resumable: {exc}", flush=True)
        return False


def run_batches_for_remaining(
    jobs_by_seat: dict[str, list[dict[str, Any]]],
    evaluators: list[dict[str, Any]],
    *,
    prompt_sha: str,
    work_dir: Path,
) -> list[dict[str, Any]]:
    """Submit (or resume) one OpenAI batch per Responses seat, poll it, return its rows."""
    import os

    work_dir.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, Any]] = []
    for ev in evaluators:
        eid = ev["evaluator_id"]
        jobs = jobs_by_seat.get(eid) or []
        if not jobs:
            continue
        if ev["api_style"] != "responses":
            raise RuntimeError(f"no batch transport for {eid}")
        key = os.environ.get(ev["key_env"] or "") or ""
        if not key:
            raise RuntimeError(f"missing {ev['key_env']}")
        seat_dir = work_dir / re.sub(r"[^a-zA-Z0-9]+", "_", eid)
        job_path = seat_dir / "openai_batch_job.json"
        if job_path.exists() and _openai_batch_is_resumable(job_path, api_key=key):
            print(f"[batch] openai resume existing job in {seat_dir.name}", flush=True)
        else:
            if job_path.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                archived = seat_dir / f"openai_batch_job.failed_{stamp}.json"
                job_path.rename(archived)
                print(f"[batch] openai set aside unusable job as {archived.name}; submitting a fresh job", flush=True)
            submit_openai_batch(jobs, api_key=key, work_dir=seat_dir, prompt_sha=prompt_sha)
        collected.extend(poll_openai_batch(seat_dir, api_key=key))
    return collected
