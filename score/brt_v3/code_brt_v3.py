#!/usr/bin/env python3
"""BRT v3 three-seat panel coder.

Codes the two open-ended BRT scenario responses of each entity with three LLM seats
(``EVALUATORS``), using the rubric in ``prompt_v3.txt``, and takes the per-field majority
(a three-way split falls back to the conservative default in ``schema.SPLIT_DEFAULTS``).
Derived variables come from ``derived.compute_all``.

Entry point: ``code_brt_v3(rows, cache_dir=..., adjudications_path=...)``.

Caches, both under ``cache_dir``:

* ``adjudications_<label>.jsonl``: one line per entity with the three seats' labels and
  evidence spans, the consensus and the per-field agreement, keyed on
  (entity_id, scenario_pair_hash). An entity whose line matches its current text is not
  re-coded; when every entity has a matching line the run makes no API call and needs no keys.
* ``per_call/<pair_hash>__<evaluator>__<prompt_sha12>.json``: one seat's vote on one text pair,
  content-addressed, so identical texts are never sent twice.

The directory is stamped with the prompt SHA-256 (``.prompt_sha``); a cache made with a
different prompt is refused.

Live calls go through the simulator's provider adapters
(``simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py``);
``transport="batch"`` sends the OpenAI seat through the OpenAI Batch API instead.

CLI: ``python score/brt_v3/code_brt_v3.py --input-csv rows.csv --cache-dir DIR --out coded.csv``
(input columns: entity_id, brt_1, brt_2; optional source, run_label).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BRT_V3_DIR = Path(__file__).resolve().parent
REPO = BRT_V3_DIR.parents[1]
sys.path.insert(0, str(BRT_V3_DIR))

from derived import compute_all  # noqa: E402
from schema import (  # noqa: E402
    BOOL_FIELDS,
    CODING_FIELDS,
    PRESENT_STRING,
    SPLIT_DEFAULTS,
    STRING_FIELDS,
)
from seat_transport import (  # noqa: E402
    SeatCallResult,
    call_seat,
    prompt_cache_key as _prompt_cache_key,
    render_entity_user,
    run_batches_for_remaining,
    split_rubric_and_entity,
)

PROMPT_PATH = BRT_V3_DIR / "prompt_v3.txt"
SIMULATOR_PATH = (
    REPO / "simulate" / "output" / "data_analysis_odd_persona_trait_experiment" / "simulation_code_iter_1.py"
)
CACHE_STAMP_NAME = ".prompt_sha"

# The DeepSeek seat is DeepSeek V4-Flash, Fireworks-hosted (the model behind the shipped caches).
# COGSUITE_DEEPSEEK_SEAT_MODEL names another Fireworks model for new coding, e.g. when that model is not served.
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_DEEPSEEK_V4_FLASH = os.environ.get("COGSUITE_DEEPSEEK_SEAT_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731")
DEEPSEEK_EVALUATOR_ID = ("deepseek-v4-flash@thinking=disabled" if FIREWORKS_DEEPSEEK_V4_FLASH.endswith("deepseek-v4-flash-0731")
                         else FIREWORKS_DEEPSEEK_V4_FLASH.rsplit("/", 1)[-1] + "@thinking=disabled")

EVALUATORS = [
    {
        "evaluator_id": "gpt-5.6-sol@reasoning_effort=low",
        "model_id": "gpt-5.6-sol",
        "api_style": "responses",
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "extra_params": {"reasoning": {"effort": "low"}},
        "extra_body": None,
        "setting": 'reasoning_effort="low"',
    },
    {
        "evaluator_id": "gemini-3.5-flash@thinking_level=minimal",
        "model_id": "gemini-3.5-flash",
        "api_style": "chat_completions",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "extra_params": {},
        "extra_body": {"extra_body": {"google": {"thinking_config": {"thinking_level": "minimal"}}}},
        "setting": 'thinking_level="minimal"',
    },
    {
        "evaluator_id": DEEPSEEK_EVALUATOR_ID,
        "model_id": FIREWORKS_DEEPSEEK_V4_FLASH,
        "api_style": "chat_completions",
        "base_url": FIREWORKS_BASE_URL,
        "key_env": "FIREWORKS_API_KEY",
        "extra_params": {},
        "extra_body": {"thinking": {"type": "disabled"}},
        "setting": 'thinking={"type":"disabled"}',
    },
]

# List prices (US$ per million tokens) used for the char/4 spend estimate printed after live calls.
PRICE = {
    "gpt-5.6-sol": {"in": 4.0, "out": 20.0},
    "gemini-3.5-flash": {"in": 1.50, "out": 9.00},
    FIREWORKS_DEEPSEEK_V4_FLASH: {"in": 0.22, "out": 0.66},
}


@dataclass
class CellCounters:
    calls: int = 0
    hard_failures: int = 0
    empty_unparseable: int = 0
    cache_hits: int = 0
    input_tokens_est: int = 0
    output_tokens_est: int = 0


_ADAPTER: Any = None
_ADAPTER_LOCK = threading.Lock()


def load_simulator_adapter():
    """Import the simulator module, whose provider adapters make the seat calls."""
    spec = importlib.util.spec_from_file_location("simulator_api_adapter", SIMULATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _get_adapter():
    global _ADAPTER
    with _ADAPTER_LOCK:
        if _ADAPTER is None:
            _ADAPTER = load_simulator_adapter()
        return _ADAPTER


def load_prompt_parts() -> tuple[str, str]:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if not text.startswith("SYSTEM\n"):
        raise RuntimeError("prompt_v3.txt must start with SYSTEM\\n")
    rest = text[len("SYSTEM\n") :]
    if "\nUSER\n" not in rest:
        raise RuntimeError("prompt_v3.txt missing USER section")
    system, user_tmpl = rest.split("\nUSER\n", 1)
    return system.strip(), user_tmpl.strip()


def load_send_parts() -> tuple[str, str]:
    """Rubric-only system message + per-entity user template."""
    return split_rubric_and_entity(*load_prompt_parts())


def scenario_pair_hash(brt_1: str, brt_2: str) -> str:
    blob = (brt_1 or "").strip() + "\n\n---\n\n" + (brt_2 or "").strip()
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prompt_sha() -> str:
    return hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()


def assert_cache_version(cache_dir: Path) -> str:
    """Stamp ``cache_dir`` with the prompt SHA-256; STOP if it holds votes made with another prompt.

    Per-call cache keys hash the scenario texts, not the rubric, so the stamp is what ties a
    cache directory to this prompt.
    """
    cache_dir = Path(cache_dir)
    sha = prompt_sha()
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = cache_dir / CACHE_STAMP_NAME
    per_call = cache_dir / "per_call"
    has_calls = per_call.exists() and any(per_call.glob("*.json"))
    if stamp.exists():
        got = stamp.read_text(encoding="utf-8").strip()
        if got != sha:
            raise SystemExit(
                f"STOP: cache stamp mismatch in {cache_dir}: stamp={got[:16]}… prompt_sha={sha[:16]}…"
            )
    elif has_calls:
        raise SystemExit(f"STOP: cache dir {cache_dir} has per_call files but no {CACHE_STAMP_NAME} stamp")
    else:
        stamp.write_text(sha + "\n", encoding="utf-8")
    return sha


def _cache_path(pair_hash: str, evaluator_id: str, cache_dir: Path) -> Path:
    safe_eval = re.sub(r"[^a-zA-Z0-9@=_.,+-]+", "_", evaluator_id)
    return Path(cache_dir) / "per_call" / f"{pair_hash}__{safe_eval}__{prompt_sha()[:12]}.json"


def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before a closing bracket; labels are never altered."""
    prev = None
    out = s
    while out != prev:
        prev = out
        out = re.sub(r",(\s*[}\]])", r"\1", out)
    return out


def _extract_json(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty_annotator_output")
    candidates = [s, _strip_trailing_commas(s)]
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        blob = m.group(0)
        candidates.extend([blob, _strip_trailing_commas(blob)])
    last_err: Exception | None = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise ValueError(f"no_json_object:{s[:200]!r}")


def _validate_and_fix_evidence(
    payload: dict[str, Any], brt_1: str, brt_2: str
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Return (labels, evidence, problems); a non-empty ``problems`` means the output is invalid.

    Every field coded present must carry an evidence span that occurs in the response text
    (wrapping quotes ignored).
    """
    labels: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    problems: list[str] = []
    combined = (brt_1 or "") + "\n" + (brt_2 or "")

    for k, allowed in STRING_FIELDS.items():
        if k not in payload:
            problems.append(f"missing:{k}")
            continue
        v = str(payload[k]).strip()
        if v not in allowed:
            problems.append(f"invalid:{k}:{v}")
            continue
        labels[k] = v
        ev = str(payload.get(f"{k}_evidence", "NONE")).strip() or "NONE"
        evidence[k] = ev
        if v in PRESENT_STRING[k]:
            if ev == "NONE":
                problems.append(f"present_without_evidence:{k}")
            elif ev.strip('"').strip("'") not in combined and ev not in combined:
                stripped = ev.strip().strip('"').strip("'")
                if stripped not in combined:
                    problems.append(f"evidence_not_in_text:{k}")

    for k in BOOL_FIELDS:
        if k not in payload:
            problems.append(f"missing:{k}")
            continue
        try:
            iv = int(payload[k])
        except Exception:
            problems.append(f"invalid:{k}:{payload[k]}")
            continue
        if iv not in (0, 1):
            problems.append(f"invalid:{k}:{iv}")
            continue
        labels[k] = iv
        ev = str(payload.get(f"{k}_evidence", "NONE")).strip() or "NONE"
        evidence[k] = ev
        if iv == 1:
            if ev == "NONE":
                problems.append(f"present_without_evidence:{k}")
            else:
                stripped = ev.strip().strip('"').strip("'")
                if stripped not in combined and ev not in combined:
                    problems.append(f"evidence_not_in_text:{k}")

    return labels, evidence, problems


def _apply_evidence_coercion(labels: dict[str, Any], evidence: dict[str, str], problems: list[str]) -> dict[str, Any]:
    """After a failed re-request, set fields coded present without valid evidence to their default."""
    out = dict(labels)
    for p in problems:
        if p.startswith("present_without_evidence:") or p.startswith("evidence_not_in_text:"):
            field = p.split(":", 1)[1]
            out[field] = SPLIT_DEFAULTS[field]
            evidence[field] = "NONE"
    return out


def _should_stop_hard_failures(mid: str, ctr: CellCounters, pct: float) -> bool:
    """Hard-failure gate per seat. The Fireworks seat tolerates isolated 401/timeout failures."""
    if ctr.calls < 20:
        return False
    rate = ctr.hard_failures / max(1, ctr.calls)
    if "fireworks" in mid or "deepseek" in mid:
        return ctr.hard_failures >= 5 and rate > 0.10
    return rate > pct


def _is_fireworks_auth_flake(ev: dict, exc: BaseException) -> bool:
    if "fireworks.ai" not in str(ev.get("base_url") or ""):
        return False
    msg = str(exc)
    blob = f"{type(exc).__name__} {msg}".lower()
    return (
        "401" in msg
        or "unauthorized" in blob
        or "timeout" in blob
        or "timed out" in blob
        or "connection error" in blob
        or "connectionerror" in blob
        or "apiconnectionerror" in blob
        or "connecterror" in blob
        or "connection reset" in blob
        or "remotedisconnected" in blob
        or "temporarily unavailable" in blob
        or "503" in msg
        or "502" in msg
    )


def _call_evaluator(ev: dict, system: str, user: str, prompt_sha_val: str) -> SeatCallResult:
    result = call_seat(ev, system, user, adapter=_get_adapter(), prompt_sha=prompt_sha_val)
    if result.text is None:
        raise RuntimeError("annotator returned None text")
    return result


def _write_vote_record(
    *,
    pair_hash: str,
    ev: dict,
    expected_sha: str,
    labels: dict[str, Any],
    evidence: dict[str, str],
    raw: str,
    system: str,
    user: str,
    cpath: Path,
    ctr: CellCounters,
    transport: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    in_est = max(1, (len(system) + len(user)) // 4)
    out_est = max(1, len(raw or "") // 4)
    ctr.input_tokens_est += in_est
    ctr.output_tokens_est += out_est
    rec = {
        "scenario_pair_hash": pair_hash,
        "evaluator_id": ev["evaluator_id"],
        "model_id": ev["model_id"],
        "setting": ev["setting"],
        "prompt_sha": expected_sha,
        "prompt_cache_key": _prompt_cache_key(expected_sha),
        "labels": labels,
        "evidence": evidence,
        "raw": raw,
        "input_tokens_est": in_est,
        "output_tokens_est": out_est,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transport": transport,
        "usage": usage or {},
    }
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def _parse_or_retry(
    ev: dict,
    system: str,
    user: str,
    brt_1: str,
    brt_2: str,
    ctr: CellCounters,
    expected_sha: str,
    raw: str | None,
    usage: dict[str, Any],
    transport: str,
) -> tuple[dict[str, Any], dict[str, str], str, dict[str, Any], str]:
    """Parse one seat output. Unparseable JSON gets a re-request (and, if that also fails, one
    fresh request); invalid labels or evidence get one re-request, after which any field still
    coded present without valid evidence is set to its default."""
    if not (raw or "").strip():
        ctr.empty_unparseable += 1
        raise ValueError("empty_output")
    try:
        payload = _extract_json(raw or "")
    except (json.JSONDecodeError, ValueError) as parse_exc:
        ctr.calls += 1
        retry_user = (
            user
            + "\n\nPREVIOUS_OUTPUT_INVALID because: unparseable JSON "
            f"({parse_exc}). Re-emit a single valid JSON object with no trailing commas."
        )
        retry = _call_evaluator(ev, system, retry_user, expected_sha)
        raw, usage, transport = retry.text, retry.usage, retry.transport
        if not (raw or "").strip():
            ctr.empty_unparseable += 1
            raise ValueError("empty_output_parse_retry") from parse_exc
        try:
            payload = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as parse_exc2:
            ctr.calls += 1
            print(
                f"[json-retry] {ev['evaluator_id']} fresh request after parse retry failed ({parse_exc2})",
                flush=True,
            )
            retry2 = _call_evaluator(ev, system, user, expected_sha)
            raw, usage, transport = retry2.text, retry2.usage, retry2.transport
            if not (raw or "").strip():
                ctr.empty_unparseable += 1
                raise ValueError("empty_output_parse_retry2") from parse_exc2
            payload = _extract_json(raw)
    labels, evidence, problems = _validate_and_fix_evidence(payload, brt_1, brt_2)
    if problems:
        ctr.calls += 1
        retry_user = (
            user
            + "\n\nPREVIOUS_OUTPUT_INVALID because: "
            + "; ".join(problems)
            + ". Re-emit the full JSON. Any field coded present with NONE evidence is invalid; "
            "use UNCLEAR_OR_CONDITIONAL / ANCHOR_NONE / 0 with NONE evidence instead."
        )
        retry = _call_evaluator(ev, system, retry_user, expected_sha)
        raw, usage, transport = retry.text, retry.usage, retry.transport
        if not (raw or "").strip():
            ctr.empty_unparseable += 1
            raise ValueError("empty_output_retry")
        payload2 = _extract_json(raw)
        labels, evidence, problems2 = _validate_and_fix_evidence(payload2, brt_1, brt_2)
        if problems2:
            labels = _apply_evidence_coercion(labels, evidence, problems2)
            for k in CODING_FIELDS:
                if k not in labels:
                    labels[k] = SPLIT_DEFAULTS[k]
                    evidence[k] = "NONE"
    return labels, evidence, raw or "", usage, transport


def evaluate_one(
    ev: dict,
    system: str,
    user_tmpl: str,
    brt_1: str,
    brt_2: str,
    counters: dict[str, CellCounters],
    use_cache: bool,
    cache_dir: Path,
) -> dict[str, Any]:
    """One seat's vote on one text pair: per-call cache hit, else a live sync call."""
    pair_hash = scenario_pair_hash(brt_1, brt_2)
    cpath = _cache_path(pair_hash, ev["evaluator_id"], cache_dir)
    ctr = counters.setdefault(ev["model_id"], CellCounters())
    expected_sha = prompt_sha()

    if use_cache and cpath.exists():
        rec = json.loads(cpath.read_text(encoding="utf-8"))
        if rec.get("prompt_sha") != expected_sha:
            raise SystemExit(
                f"STOP: cache file {cpath.name} prompt_sha {rec.get('prompt_sha')!r} does not match "
                f"the current prompt ({expected_sha[:16]}…)"
            )
        ctr.cache_hits += 1
        return rec

    user = render_entity_user(user_tmpl, brt_1, brt_2)
    try:
        ctr.calls += 1
        result = _call_evaluator(ev, system, user, expected_sha)
        labels, evidence, raw, usage, used_transport = _parse_or_retry(
            ev, system, user, brt_1, brt_2, ctr, expected_sha, result.text, result.usage, result.transport
        )
    except Exception as exc:
        if "HardFailure" in exc.__class__.__name__:
            ctr.hard_failures += 1
        else:
            ctr.empty_unparseable += 1
        raise

    return _write_vote_record(
        pair_hash=pair_hash,
        ev=ev,
        expected_sha=expected_sha,
        labels=labels,
        evidence=evidence,
        raw=raw,
        system=system,
        user=user,
        cpath=cpath,
        ctr=ctr,
        transport=used_transport,
        usage=usage,
    )


def ingest_batch_row(
    ev: dict,
    system: str,
    user_tmpl: str,
    brt_1: str,
    brt_2: str,
    counters: dict[str, CellCounters],
    cache_dir: Path,
    raw: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    """Write one batch result into the per-call cache (``transport="batch"``)."""
    pair_hash = scenario_pair_hash(brt_1, brt_2)
    cpath = _cache_path(pair_hash, ev["evaluator_id"], cache_dir)
    expected_sha = prompt_sha()
    if cpath.exists():
        rec = json.loads(cpath.read_text(encoding="utf-8"))
        if rec.get("prompt_sha") == expected_sha:
            return rec
    ctr = counters.setdefault(ev["model_id"], CellCounters())
    user = render_entity_user(user_tmpl, brt_1, brt_2)
    ctr.calls += 1
    labels, evidence, raw, usage, used_transport = _parse_or_retry(
        ev, system, user, brt_1, brt_2, ctr, expected_sha, raw, usage, "batch"
    )
    return _write_vote_record(
        pair_hash=pair_hash,
        ev=ev,
        expected_sha=expected_sha,
        labels=labels,
        evidence=evidence,
        raw=raw,
        system=system,
        user=user,
        cpath=cpath,
        ctr=ctr,
        transport=used_transport,
        usage=usage,
    )


def _majority_per_field(votes: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Per-field consensus of three votes: (consensus, agreement_by_field, split_fields).

    Three equal labels are "unanimous", two "majority"; three different labels are a "split"
    and the field takes its conservative default.
    """
    consensus: dict[str, Any] = {}
    agreement: dict[str, str] = {}
    splits: list[str] = []
    for field in CODING_FIELDS:
        vals = [v["labels"][field] for v in votes]
        counts: dict[str, int] = {}
        for k in (str(x) for x in vals):
            counts[k] = counts.get(k, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top, top_n = ranked[0]
        n_at_top = sum(1 for _, n in ranked if n == top_n)
        if top_n == 3:
            agreement[field] = "unanimous"
            consensus[field] = int(float(top)) if field in BOOL_FIELDS else top
        elif top_n == 2 and n_at_top == 1:
            agreement[field] = "majority"
            consensus[field] = int(float(top)) if field in BOOL_FIELDS else top
        else:
            agreement[field] = "split"
            consensus[field] = SPLIT_DEFAULTS[field]
            splits.append(field)
    return consensus, agreement, splits


def panel_guard_deepseek_seat() -> None:
    """STOP unless the DeepSeek seat points at the Fireworks host it is defined on."""
    ds = next(e for e in EVALUATORS if e["evaluator_id"].startswith("deepseek-"))
    base = (ds.get("base_url") or "").rstrip("/")
    if "api.fireworks.ai" not in base:
        raise SystemExit(f"STOP: deepseek seat base_url is not Fireworks ({base}).")
    if ds["key_env"] != "FIREWORKS_API_KEY":
        raise SystemExit(f"STOP: deepseek seat key_env must be FIREWORKS_API_KEY, got {ds['key_env']}")


def verify_keys() -> None:
    """STOP unless every seat's API key is set (environment or the repository's ``.env``)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    missing = [e["key_env"] for e in EVALUATORS if not os.environ.get(e["key_env"])]
    if missing:
        raise SystemExit(f"STOP: BRT coding needs live calls but these API keys are not set: {missing}")
    panel_guard_deepseek_seat()


def _read_adjudication_lines(path: Path) -> list[tuple[str, dict | None]]:
    """Every non-blank line of an adjudications JSONL as (raw_line, parsed_or_None)."""
    out: list[tuple[str, dict | None]] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict) or "entity_id" not in obj:
                    obj = None
            except Exception:
                obj = None
            out.append((raw, obj))
    return out


def _current_pair_hashes(work: pd.DataFrame, id_col: str, text_col_1: str, text_col_2: str) -> dict[str, str]:
    """entity_id -> scenario_pair_hash of the current text. STOP if one id carries two texts."""
    hashes: dict[str, str] = {}
    for eid, b1, b2 in zip(work[id_col], work[text_col_1], work[text_col_2]):
        eid = str(eid)
        ph = scenario_pair_hash(str(b1), str(b2))
        if hashes.setdefault(eid, ph) != ph:
            raise SystemExit(f"STOP: entity_id {eid} appears with two different BRT texts in the input")
    return hashes


def _plan_resume(
    adj_lines: list[tuple[str, dict | None]], current: dict[str, str]
) -> tuple[set[tuple[str, str]], list[str]]:
    """(done set of (entity_id, scenario_pair_hash), entity_ids whose text has changed).

    An entity is done only when an adjudication line carries both its entity_id and the hash of
    its current text. An entity_id present only under other hashes (or with no hash recorded)
    is "changed": its stored codes were made on different text.
    """
    done: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    for _raw, adj in adj_lines:
        if adj is None:
            continue
        eid = str(adj["entity_id"])
        done.add((eid, str(adj.get("scenario_pair_hash", ""))))
        seen_ids.add(eid)
    changed = sorted(
        (e for e in current if e in seen_ids and (e, current[e]) not in done),
        key=lambda e: (len(e), e),
    )
    return done, changed


def _supersede_changed(active_adj: Path, adj_lines: list[tuple[str, dict | None]], current: dict[str, str]) -> Path:
    """Move every line that does not match its entity's current text to
    ``<adjudications>.superseded.jsonl`` (appended verbatim) and rewrite the live file with at
    most one line per entity. Lines for entity_ids not in the input are kept."""
    keep: list[str] = []
    superseded: list[str] = []
    for raw, adj in adj_lines:
        if adj is None:
            superseded.append(raw)
            continue
        eid = str(adj["entity_id"])
        if eid not in current or str(adj.get("scenario_pair_hash", "")) == current[eid]:
            keep.append(raw)
        else:
            superseded.append(raw)
    final: list[str] = []
    last_idx: dict[str, int] = {}
    for i, raw in enumerate(keep):
        last_idx[str(json.loads(raw)["entity_id"])] = i
    for i, raw in enumerate(keep):
        if last_idx[str(json.loads(raw)["entity_id"])] == i:
            final.append(raw)
        else:
            superseded.append(raw)
    sup_path = active_adj.with_name(active_adj.name + ".superseded.jsonl")
    with open(sup_path, "a", encoding="utf-8") as fh:
        for raw in superseded:
            fh.write(raw + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp = active_adj.with_name(active_adj.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for raw in final:
            fh.write(raw + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, active_adj)
    return sup_path


def _evidence_for_consensus(vote_recs: list[dict], consensus: dict[str, Any]) -> dict[str, str]:
    """Per field, the evidence span of the first seat whose label equals the consensus, else NONE."""
    out: dict[str, str] = {}
    for field in CODING_FIELDS:
        ev_span = "NONE"
        for vr in vote_recs:
            if str(vr["labels"][field]) == str(consensus[field]):
                ev_span = vr.get("evidence", {}).get(field, "NONE")
                break
        out[field] = ev_span
    return out


def _code_pending_entity(
    row: pd.Series,
    *,
    id_col: str,
    text_col_1: str,
    text_col_2: str,
    system: str,
    user_tmpl: str,
    counters: dict[str, CellCounters],
    use_cache: bool,
    cache_dir: Path,
) -> list[dict]:
    """The three seats' vote records for one entity (seats run in parallel; order of EVALUATORS).

    A seat that fails after its retries STOPs the run: no default vote is ever substituted for a
    missing seat.
    """
    eid = str(row[id_col])
    b1 = str(row[text_col_1])
    b2 = str(row[text_col_2])

    def _retry(ev: dict, exc: Exception, tag: str) -> dict:
        for extra in range(3):
            delay = min(20.0, 3.0 * (extra + 1))
            print(f"[{tag}] {eid} {ev['evaluator_id']} extra={extra + 1}/3 sleep {delay:.0f}s", flush=True)
            time.sleep(delay)
            try:
                return evaluate_one(ev, system, user_tmpl, b1, b2, counters, use_cache, cache_dir)
            except Exception as exc2:
                exc = exc2
        raise SystemExit(
            f"STOP: seat failed after extra retries ({ev['model_id']} {eid}): {str(exc)[:400]}"
        )

    ev_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(EVALUATORS)) as pool:
        futs = {
            pool.submit(evaluate_one, ev, system, user_tmpl, b1, b2, counters, use_cache, cache_dir): ev
            for ev in EVALUATORS
        }
        for fut in as_completed(futs):
            ev = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                if isinstance(exc, json.JSONDecodeError) or "Expecting" in str(exc):
                    rec = _retry(ev, exc, "json-retry-entity")
                elif ev["evaluator_id"].startswith("deepseek-") and _is_fireworks_auth_flake(ev, exc):
                    rec = _retry(ev, exc, "fireworks-retry-entity")
                else:
                    raise SystemExit(
                        f"STOP: seat failed without a vote ({ev['model_id']} {eid}): {str(exc)[:400]}"
                    )
            ev_results[ev["evaluator_id"]] = rec
    return [ev_results[ev["evaluator_id"]] for ev in EVALUATORS]


def code_brt_v3(
    rows: pd.DataFrame,
    *,
    text_col_1: str = "brt_1",
    text_col_2: str = "brt_2",
    id_col: str = "entity_id",
    source_col: str | None = "source",
    run_label_col: str | None = "run_label",
    adjudications_path: Path | None = None,
    cache_dir: Path | None = None,
    hard_failure_stop_pct: float = 0.01,
    max_rows: int | None = None,
    use_cache: bool = True,
    transport: str = "sync",
    recode_changed_text: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict[str, CellCounters]]:
    """Code ``rows`` (columns id_col, text_col_1, text_col_2) with the three-seat panel.

    Returns (consensus_wide, long_votes, split_rows, counters). ``consensus_wide`` has one row
    per entity with the 28 coded fields, their evidence spans and agreement, and the derived
    variables. ``transport`` is "sync" or "batch" (OpenAI seat through the Batch API).
    An entity whose text changed since it was adjudicated STOPs the run unless
    ``recode_changed_text`` is set, in which case its old lines move to
    ``<adjudications>.superseded.jsonl`` and it is coded afresh.
    """
    if transport not in {"sync", "batch"}:
        raise SystemExit(f"STOP: unknown transport {transport!r} (want sync|batch)")
    if cache_dir is None:
        raise SystemExit("STOP: code_brt_v3 requires an explicit cache_dir")
    system, user_tmpl = load_send_parts()
    active_cache = Path(cache_dir)
    assert_cache_version(active_cache)
    active_adj = Path(adjudications_path) if adjudications_path is not None else active_cache / "adjudications_brt.jsonl"
    active_adj.parent.mkdir(parents=True, exist_ok=True)

    work = rows.copy()
    if max_rows is not None:
        work = work.iloc[:max_rows].copy()
    work[id_col] = work[id_col].astype(str)
    work[text_col_1] = work[text_col_1].fillna("").astype(str)
    work[text_col_2] = work[text_col_2].fillna("").astype(str)

    counters: dict[str, CellCounters] = {e["model_id"]: CellCounters() for e in EVALUATORS}

    current_hashes = _current_pair_hashes(work, id_col, text_col_1, text_col_2)
    adj_lines = _read_adjudication_lines(active_adj)
    done_keys, changed_ids = _plan_resume(adj_lines, current_hashes)
    if changed_ids:
        listing = ", ".join(changed_ids)
        if not recode_changed_text:
            raise SystemExit(
                f"STOP: {len(changed_ids)} entity_id(s) in {active_adj} were adjudicated on different "
                f"BRT text than the current input (scenario_pair_hash differs): {listing}. "
                "Nothing was re-coded. Re-run with --recode-changed-text to re-code them (the old lines "
                f"move to {active_adj.name}.superseded.jsonl), or fix the input."
            )
        sup_path = _supersede_changed(active_adj, adj_lines, current_hashes)
        print(
            f"[resume] --recode-changed-text: {len(changed_ids)} entity_id(s) changed text ({listing}); "
            f"old adjudication lines appended to {sup_path}",
            flush=True,
        )
        adj_lines = _read_adjudication_lines(active_adj)
        done_keys, _ = _plan_resume(adj_lines, current_hashes)
    if done_keys:
        print(f"[resume] {len(done_keys)} adjudicated rows in {active_adj.name}", flush=True)

    pending_rows = [
        row for _, row in work.iterrows() if (str(row[id_col]), current_hashes[str(row[id_col])]) not in done_keys
    ]
    if pending_rows:
        print(f"[brt_v3] {len(pending_rows)} entities need live coding", flush=True)
        verify_keys()

    if transport == "batch" and pending_rows:
        batch_evaluators = [e for e in EVALUATORS if e.get("api_style") == "responses"]
        print(
            "[batch] batching " + ",".join(e["evaluator_id"] for e in batch_evaluators) + "; other seats sync",
            flush=True,
        )
        jobs_by_seat: dict[str, list[dict[str, Any]]] = {e["evaluator_id"]: [] for e in batch_evaluators}
        job_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in pending_rows:
            b1 = str(row[text_col_1])
            b2 = str(row[text_col_2])
            ph = scenario_pair_hash(b1, b2)
            user = render_entity_user(user_tmpl, b1, b2)
            for ev in batch_evaluators:
                job = {
                    "entity_id": str(row[id_col]),
                    "pair_hash": ph,
                    "evaluator_id": ev["evaluator_id"],
                    "model_id": ev["model_id"],
                    "system": system,
                    "user": user,
                    "brt_1": b1,
                    "brt_2": b2,
                }
                job_lookup[(ph, ev["evaluator_id"])] = job
                if use_cache and _cache_path(ph, ev["evaluator_id"], active_cache).exists():
                    continue
                jobs_by_seat[ev["evaluator_id"]].append(job)
        n_jobs = sum(len(v) for v in jobs_by_seat.values())
        print(f"[batch] remaining seat-calls={n_jobs}", flush=True)
        if n_jobs:
            run_tag = ""
            if run_label_col and run_label_col in pending_rows[0].index:
                run_tag = str(pending_rows[0][run_label_col] or "")
            safe_tag = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_tag) or "unlabelled"
            collected = run_batches_for_remaining(
                jobs_by_seat,
                batch_evaluators,
                prompt_sha=prompt_sha(),
                work_dir=active_cache / "_batch_work" / safe_tag,
            )
            ev_by_id = {e["evaluator_id"]: e for e in EVALUATORS}
            for item in collected:
                eid_key = item.get("evaluator_id")
                ph = item.get("pair_hash")
                job = job_lookup.get((ph, eid_key)) if ph and eid_key else None
                if job is None:
                    raise SystemExit(f"STOP: batch result unmatched {item.get('entity_id')} {eid_key}")
                if item.get("error") and not (item.get("text") or "").strip():
                    raise SystemExit(f"STOP: batch error {eid_key} {job['entity_id']}: {str(item.get('error'))[:400]}")
                ingest_batch_row(
                    ev_by_id[eid_key],
                    system,
                    user_tmpl,
                    job["brt_1"],
                    job["brt_2"],
                    counters,
                    active_cache,
                    item.get("text") or "",
                    item.get("usage") or {},
                )

    if transport == "sync" and pending_rows:
        first = pending_rows[0]
        print("[warmup] one sequential call per seat before the parallel loop", flush=True)
        for ev in EVALUATORS:
            evaluate_one(
                ev, system, user_tmpl, str(first[text_col_1]), str(first[text_col_2]),
                counters, use_cache, active_cache,
            )

    with open(active_adj, "a", encoding="utf-8") as adj_fh:
        for n_done, row in enumerate(pending_rows, start=1):
            eid = str(row[id_col])
            b1 = str(row[text_col_1])
            b2 = str(row[text_col_2])
            vote_recs = _code_pending_entity(
                row,
                id_col=id_col,
                text_col_1=text_col_1,
                text_col_2=text_col_2,
                system=system,
                user_tmpl=user_tmpl,
                counters=counters,
                use_cache=use_cache,
                cache_dir=active_cache,
            )
            consensus, agreement, splits = _majority_per_field(vote_recs)
            adj_rec = {
                "entity_id": eid,
                "source": str(row[source_col]) if source_col and source_col in row.index else "",
                "run_label": str(row[run_label_col]) if run_label_col and run_label_col in row.index else "",
                "scenario_pair_hash": scenario_pair_hash(b1, b2),
                "votes": [
                    {
                        "model_id": vr["model_id"],
                        "evaluator_id": vr["evaluator_id"],
                        "setting": vr["setting"],
                        "labels": vr["labels"],
                        "evidence": vr["evidence"],
                    }
                    for vr in vote_recs
                ],
                "consensus": consensus,
                "agreement_by_field": agreement,
                "split_fields": splits,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            adj_fh.write(json.dumps(adj_rec, ensure_ascii=False) + "\n")
            adj_fh.flush()
            for mid, ctr in counters.items():
                if _should_stop_hard_failures(mid, ctr, hard_failure_stop_pct):
                    raise SystemExit(f"STOP: hard failures exceed threshold for {mid}: {ctr.hard_failures}/{ctr.calls}")
            if n_done % 10 == 0 or n_done == len(pending_rows):
                print(f"[progress] {n_done}/{len(pending_rows)} newly coded", flush=True)

    if not active_adj.exists() or active_adj.stat().st_size == 0:
        raise SystemExit(f"STOP: no adjudications in {active_adj}")
    consensus_rows, long_rows, split_rows = _rebuild_from_adjudications(active_adj, current_hashes)
    if not consensus_rows:
        raise SystemExit(f"STOP: no consensus rows reconstructed from {active_adj}")
    consensus_df = compute_all(pd.DataFrame(consensus_rows))
    return consensus_df, pd.DataFrame(long_rows), split_rows, counters


def _select_adjudications(path: Path, current_hashes: dict[str, str] | None) -> list[dict]:
    """One adjudication per entity_id, in first-appearance order.

    When an entity has several lines, the one whose scenario_pair_hash matches the current text
    wins (the last such line if more than one); STOP if none matches. An entity with a single
    line and no current text to compare is returned as stored; several lines and no current text
    is ambiguous and STOPs.
    """
    by_eid: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            adj = json.loads(line)
            by_eid.setdefault(str(adj["entity_id"]), []).append(adj)
    selected: list[dict] = []
    for eid, lines in by_eid.items():
        if current_hashes is not None and eid in current_hashes:
            want = current_hashes[eid]
            matching = [a for a in lines if str(a.get("scenario_pair_hash", "")) == want]
            if not matching:
                raise SystemExit(
                    f"STOP: no adjudication line for entity_id {eid} in {path} matches the current "
                    f"BRT text (scenario_pair_hash {want[:12]}…; stored: "
                    f"{[str(a.get('scenario_pair_hash', ''))[:12] for a in lines]})"
                )
            selected.append(matching[-1])
        elif len(lines) == 1:
            selected.append(lines[0])
        else:
            raise SystemExit(
                f"STOP: {len(lines)} adjudication lines for entity_id {eid} in {path} and no current "
                "text to choose between them"
            )
    return selected


def _rebuild_from_adjudications(
    path: Path, current_hashes: dict[str, str] | None = None
) -> tuple[list[dict], list[dict], list[dict]]:
    """(consensus rows, long per-seat rows, split rows) from an adjudications JSONL."""
    consensus_rows: list[dict] = []
    long_rows: list[dict] = []
    split_rows: list[dict] = []
    for adj in _select_adjudications(path, current_hashes):
        eid = str(adj["entity_id"])
        source = adj.get("source", "")
        run_label = adj.get("run_label", "")
        pair_hash = adj.get("scenario_pair_hash", "")
        consensus = adj["consensus"]
        agreement = adj["agreement_by_field"]
        vote_recs = adj["votes"]
        evidence_out = _evidence_for_consensus(vote_recs, consensus)
        consensus_rows.append(
            {
                "entity_id": eid,
                "source": source,
                "run_label": run_label,
                "scenario_pair_hash": pair_hash,
                **consensus,
                **{f"{k}_evidence": evidence_out[k] for k in CODING_FIELDS},
                **{f"{k}_agreement": agreement[k] for k in CODING_FIELDS},
            }
        )
        for field in adj.get("split_fields", []):
            split_rows.append(
                {
                    "entity_id": eid,
                    "source": source,
                    "run_label": run_label,
                    "field": field,
                    "votes": [vr["labels"][field] for vr in vote_recs],
                    "model_ids": [vr["model_id"] for vr in vote_recs],
                }
            )
        for vr in vote_recs:
            for field in CODING_FIELDS:
                long_rows.append(
                    {
                        "entity_id": eid,
                        "source": source,
                        "run_label": run_label,
                        "scenario_pair_hash": pair_hash,
                        "field": field,
                        "evaluator_id": vr["evaluator_id"],
                        "model_id": vr["model_id"],
                        "label": vr["labels"][field],
                        "evidence": vr.get("evidence", {}).get(field, "NONE"),
                        "agreement": agreement[field],
                        "consensus_label": consensus[field],
                    }
                )
    return consensus_rows, long_rows, split_rows


def estimate_spend(counters: dict[str, CellCounters]) -> dict[str, float]:
    """char/4 token estimate x list price, per seat and total, for this run's live calls."""
    out: dict[str, float] = {}
    total = 0.0
    for mid, ctr in counters.items():
        usd = ctr.input_tokens_est / 1e6 * PRICE[mid]["in"] + ctr.output_tokens_est / 1e6 * PRICE[mid]["out"]
        out[mid] = usd
        total += usd
    out["total"] = total
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BRT v3 three-seat panel coder")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, help="default: <cache-dir>/adjudications_brt.jsonl")
    parser.add_argument("--out", type=Path, help="write the consensus table (one row per entity) here")
    parser.add_argument("--transport", choices=["sync", "batch"], default=os.environ.get("BRT_V3_TRANSPORT", "sync"))
    parser.add_argument("--id-col", default="entity_id")
    parser.add_argument("--text-col-1", default="brt_1")
    parser.add_argument("--text-col-2", default="brt_2")
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--run-label-col", default="run_label")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-cache", action="store_true", help="ignore per-call cache files")
    parser.add_argument(
        "--recode-changed-text",
        action="store_true",
        help="re-code entities whose BRT text changed since they were adjudicated; the old lines "
        "move to <adjudications>.superseded.jsonl (default: STOP and list them)",
    )
    args = parser.parse_args(argv)
    df = pd.read_csv(args.input_csv, dtype={args.id_col: str}, keep_default_na=False, na_values=[""])
    consensus_df, _long_df, split_rows, counters = code_brt_v3(
        df,
        text_col_1=args.text_col_1,
        text_col_2=args.text_col_2,
        id_col=args.id_col,
        source_col=args.source_col if args.source_col in df.columns else None,
        run_label_col=args.run_label_col if args.run_label_col in df.columns else None,
        adjudications_path=args.adjudications,
        cache_dir=args.cache_dir,
        max_rows=args.max_rows,
        use_cache=not args.no_cache,
        transport=args.transport,
        recode_changed_text=args.recode_changed_text,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        consensus_df.to_csv(args.out, index=False)
    spend = estimate_spend(counters)
    print(f"[done] n={len(consensus_df)} field_splits={len(split_rows)} live_spend_est=${spend['total']:.4f}", flush=True)


if __name__ == "__main__":
    main()
