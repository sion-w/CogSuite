from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from socsim.common import require_columns

_client = None

STRING_FIELDS = {
    "BRT1_decision": {
        "KEEP_BOOKING",
        "CHANGE_RESTAURANT",
        "CANCEL_OR_POSTPONE",
        "UNCLEAR_OR_CONDITIONAL",
    },
    "BRT1_anchor": {
        "ANCHOR_FOOD_CRITICS/PRO_REVIEWS",
        "ANCHOR_CROWD_REVIEWS",
        "ANCHOR_FRIEND_REPORT",
        "ANCHOR_BALANCED_CONFLICT",
        "ANCHOR_NONE_SPECIFIED",
    },
    "BRT2_decision": {"TAKE_MEDEX", "DELAY_AND_INVESTIGATE", "NOT_TAKE_MEDEX", "UNCLEAR_OR_CONDITIONAL"},
    "BRT2_anchor": {"ANCHOR_DOCTOR", "ANCHOR_SOCIAL_MEDIA", "ANCHOR_BALANCED_CONFLICT", "ANCHOR_NONE_SPECIFIED"},
    "BRT2_research_specificity": {
        "NO_ADDITIONAL_RESEARCH",
        "HIGH_QUALITY_RESEARCH",
        "VAGUE_RESEARCH",
        "SOCIAL_MEDIA_RESEARCH",
    },
}
BOOL_FIELDS = [
    "BRT1_deposit_salience",
    "BRT1_hygiene_risk_reasoning",
    "BRT1_moral_fairness_loyalty",
    "BRT1_emotional_outburst",
    "BRT2_emotional_reaction",
    "BRT2_social_media_sceptic",
]
CODING_FIELDS = list(STRING_FIELDS.keys()) + BOOL_FIELDS


def _build_prompts(q1_text: str, q2_text: str) -> tuple[str, str]:
    system_prompt = (
        "You are a strict coding assistant for Behavioural Reflection Task responses.\n"
        "Use ONLY the text provided for each scenario. Never invent details.\n"
        "If a category is ambiguous or missing, choose the UNCLEAR / NONE / VAGUE option rather than guessing.\n"
        "Return a single valid JSON object and nothing else. Use integers 0/1 for binary flags.\n"
    )
    user_prompt = f"""
Task: Code two open-ended responses. Pick exactly one label per field from the allowed options.

BRT1 scenario (restaurant / deposit):
 - Decision (BRT1_decision): KEEP_BOOKING | CHANGE_RESTAURANT | CANCEL_OR_POSTPONE | UNCLEAR_OR_CONDITIONAL
 - Anchor (BRT1_anchor):
     * ANCHOR_FOOD_CRITICS/PRO_REVIEWS — trusts professional critics / reputable review outlets.
     * ANCHOR_CROWD_REVIEWS — trusts mass customer reviews (Google/Yelp/TripAdvisor).
     * ANCHOR_FRIEND_REPORT — mainly leans on Friend A's report or similar anecdote.
     * ANCHOR_BALANCED_CONFLICT — explicitly weighs critics vs crowd vs friend without privileging one.
     * ANCHOR_NONE_SPECIFIED — no clear anchor.
 - Flags (0/1):
     * BRT1_deposit_salience — mentions the deposit/money at stake as a reason.
     * BRT1_hygiene_risk_reasoning — discusses plausibility/evidence of hygiene issues or base rates.
     * BRT1_moral_fairness_loyalty — mentions fairness/loyalty to Friend B.
     * BRT1_emotional_outburst — strong emotional language directed at Friend A's last-minute change.

BRT2 scenario (Medex / doctor vs social media):
 - Decision (BRT2_decision): TAKE_MEDEX | DELAY_AND_INVESTIGATE | NOT_TAKE_MEDEX | UNCLEAR_OR_CONDITIONAL
 - Anchor (BRT2_anchor):
     * ANCHOR_DOCTOR — trusts own doctor/medical professionals or guidelines over the social-media doctor.
     * ANCHOR_SOCIAL_MEDIA — treats the online doctor/video as the primary trusted source.
     * ANCHOR_BALANCED_CONFLICT — explicitly weighs both doctor and video without privileging one.
     * ANCHOR_NONE_SPECIFIED — no clear anchor.
 - Research specificity (BRT2_research_specificity):
     * NO_ADDITIONAL_RESEARCH — decides without further research.
     * HIGH_QUALITY_RESEARCH — plans evidence-based/professional checks (doctor follow-up, trials, journal articles, regulators).
     * VAGUE_RESEARCH — vague "look it up" / "do my own research" without source quality.
     * SOCIAL_MEDIA_RESEARCH — more influencer/social-media content (more videos, YouTube/TikTok, etc.).
 - Flags (0/1):
     * BRT2_emotional_reaction — explicit anxiety/fear/anger/worry.
     * BRT2_social_media_sceptic — calls the online doctor/video biased, misleading, clickbait, cherry-picked, agenda-driven, etc.

Rules:
- Base every label ONLY on the participant's wording.
- If uncertain, choose UNCLEAR_OR_CONDITIONAL, ANCHOR_NONE_SPECIFIED, or VAGUE_RESEARCH as applicable.
- Output strict JSON with keys: {list(STRING_FIELDS.keys()) + BOOL_FIELDS}.

BRT1_response: \"\"\"{q1_text.strip()}\"\"\"

BRT2_response: \"\"\"{q2_text.strip()}\"\"\"
"""
    return system_prompt, user_prompt


def _validate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    out = {}
    for k, allowed in STRING_FIELDS.items():
        if k not in payload:
            return None, f"missing_field:{k}"
        v = str(payload[k]).strip()
        if v not in allowed:
            return None, f"invalid_value:{k}:{v}"
        out[k] = v
    for k in BOOL_FIELDS:
        if k not in payload:
            return None, f"missing_field:{k}"
        try:
            iv = int(payload[k])
        except Exception:
            return None, f"invalid_value:{k}:{payload[k]}"
        if iv not in (0, 1):
            return None, f"invalid_value:{k}:{payload[k]}"
        out[k] = iv
    return out, None


def _max_workers() -> int:
    raw = os.environ.get("SOCSIM_OPENAI_CONCURRENCY", "10")
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


def _progress_interval() -> float:
    raw = os.environ.get("SOCSIM_PROGRESS_INTERVAL", "5")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 5.0


def _max_retries() -> int:
    raw = os.environ.get("SOCSIM_OPENAI_MAX_RETRIES", "5")
    try:
        return max(0, int(raw))
    except ValueError:
        return 5


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


def _extract_output_text(resp) -> str | None:
    raw = getattr(resp, "output_text", None)
    if raw:
        return raw
    try:
        return resp.output[0].content[0].text
    except Exception:
        return None


def _json_schema() -> dict:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, allowed in STRING_FIELDS.items():
        properties[key] = {"type": "string", "enum": sorted(list(allowed))}
        required.append(key)
    for key in BOOL_FIELDS:
        properties[key] = {"type": "integer", "enum": [0, 1]}
        required.append(key)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _api_error_text(exc: Exception) -> str:
    return str(exc)


def _is_retryable_api_error(exc: Exception) -> bool:
    text = _api_error_text(exc).lower()
    if "insufficient_quota" in text or "current quota" in text or "billing" in text:
        return False
    return "ratelimit" in exc.__class__.__name__.lower() or "rate limit" in text or "429" in text


def _retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass
    base = min(60.0, 2.0 * (2 ** attempt))
    return base + random.uniform(0.0, 1.0)


def _with_retries(call):
    attempts = _max_retries() + 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not _is_retryable_api_error(exc):
                raise
            time.sleep(_retry_delay(exc, attempt))
    raise last_exc if last_exc is not None else RuntimeError("unknown OpenAI retry failure")


def _call_openai(system_prompt: str, user_prompt: str) -> tuple[dict[str, Any] | None, str, str | None]:
    client, reason = _get_client()
    if client is None:
        return None, reason, None
    model = os.environ.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL_SNAPSHOT", "gpt-5-nano"))
    fallback_model = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
    schema = _json_schema()
    try:
        resp = _with_retries(
            lambda: client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                text={"format": {"type": "json_schema", "name": "brt_coding", "schema": schema, "strict": True}},
            )
        )
        raw = _extract_output_text(resp)
        payload = json.loads(raw) if raw else {}
        val, err = _validate_payload(payload)
        if err:
            return None, f"{err}|raw:{str(raw)[:200]}", raw
        return val, "ok", raw
    except Exception as e:
        first_err = str(e)
        try:
            resp = _with_retries(
                lambda: client.chat.completions.create(
                    model=fallback_model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_object"},
                )
            )
            raw = resp.choices[0].message.content
            payload = json.loads(raw) if raw else {}
            val, err = _validate_payload(payload)
            if err:
                return None, f"{err}|raw:{str(raw)[:200]}", raw
            return val, "ok", raw
        except Exception as e2:
            return None, f"api_exception:{e2.__class__.__name__}:{str(e2)[:200]}|first:{first_err[:120]}", None


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BRT2_reflective_accept"] = (
        (out["BRT2_decision"] == "TAKE_MEDEX")
        & out["BRT2_anchor"].isin(["ANCHOR_DOCTOR", "ANCHOR_BALANCED_CONFLICT"])
        & out["BRT2_research_specificity"].isin(["NO_ADDITIONAL_RESEARCH", "HIGH_QUALITY_RESEARCH"])
    ).astype(float)
    out["BRT2_reflective_delay"] = (
        (out["BRT2_decision"] == "DELAY_AND_INVESTIGATE")
        & (out["BRT2_research_specificity"] == "HIGH_QUALITY_RESEARCH")
    ).astype(float)
    out["BRT2_social_media_reject"] = (
        out["BRT2_decision"].isin(["NOT_TAKE_MEDEX", "DELAY_AND_INVESTIGATE"])
        & ((out["BRT2_anchor"] == "ANCHOR_SOCIAL_MEDIA") | (out["BRT2_research_specificity"] == "SOCIAL_MEDIA_RESEARCH"))
    ).astype(float)
    out["BRT1_reflective_keep"] = (
        (out["BRT1_decision"] == "KEEP_BOOKING")
        & ((out["BRT1_hygiene_risk_reasoning"] == 1) | out["BRT1_anchor"].isin(["ANCHOR_FOOD_CRITICS/PRO_REVIEWS", "ANCHOR_BALANCED_CONFLICT"]))
        & (out["BRT1_moral_fairness_loyalty"] == 1)
    ).astype(float)
    out["BRT1_emotional_change_cancel"] = (
        out["BRT1_decision"].isin(["CHANGE_RESTAURANT", "CANCEL_OR_POSTPONE"])
        & (out["BRT1_emotional_outburst"] == 1)
        & (out["BRT1_hygiene_risk_reasoning"] == 0)
    ).astype(float)
    out["BRT1_deposit_driven"] = (out["BRT1_deposit_salience"] == 1).astype(float)
    return out


def _code_one(row: pd.Series) -> dict[str, object]:
    sys_p, usr_p = _build_prompts(str(row.get("brt_1", "")), str(row.get("brt_2", "")))
    payload, reason, raw = _call_openai(sys_p, usr_p)
    record: dict[str, object] = {"entity_id": row["entity_id"], "BRT_llm_reason": reason, "BRT_llm_raw_response": raw}
    for col in CODING_FIELDS:
        record[col] = payload.get(col) if payload else np.nan
    return record


def _valid_cached_records(existing_coded: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if existing_coded is None or existing_coded.empty:
        return {}
    required = ["entity_id", "BRT_llm_reason", "BRT_llm_raw_response", *CODING_FIELDS]
    if any(col not in existing_coded.columns for col in required):
        return {}
    existing = existing_coded.copy()
    existing["entity_id"] = existing["entity_id"].astype(str)
    valid = existing[
        (existing["BRT_llm_reason"] == "ok")
        & existing[CODING_FIELDS].notna().all(axis=1)
    ].drop_duplicates(subset=["entity_id"], keep="last")
    return valid[required].set_index("entity_id").to_dict("index")


def validate_human_brt_coded(human_df: pd.DataFrame) -> None:
    require_columns(human_df, CODING_FIELDS, "human_person_brt_coded")


def _code_brt_for_agents_v1(  # DEAD v1 single-model coder (submitted manuscript). Not called.
    person_df: pd.DataFrame,
    items_df: pd.DataFrame,
    existing_coded: pd.DataFrame | None = None,
) -> pd.DataFrame:
    require_columns(person_df, ["entity_id"], "agent_person")
    require_columns(items_df, ["entity_id", "item_id", "response_text"], "agent_items")
    person = person_df.copy()
    person["entity_id"] = person["entity_id"].astype(str)
    items = items_df.copy()
    items["entity_id"] = items["entity_id"].astype(str)
    texts = (
        items.loc[items["item_id"].astype(str).isin(["brt_1", "brt_2"]), ["entity_id", "item_id", "response_text"]]
        .pivot_table(index="entity_id", columns="item_id", values="response_text", aggfunc="first")
        .reset_index()
    )
    if "brt_1" not in texts.columns:
        texts["brt_1"] = ""
    if "brt_2" not in texts.columns:
        texts["brt_2"] = ""
    cached_records = _valid_cached_records(existing_coded)
    coded: list[dict[str, object]] = []
    pending_rows = []
    for _, row in texts.iterrows():
        entity_id = str(row["entity_id"])
        cached = cached_records.get(entity_id)
        if cached:
            coded.append({"entity_id": entity_id, **cached})
        else:
            pending_rows.append(row)
    workers = min(_max_workers(), max(1, len(pending_rows)))
    progress_mode = os.environ.get("SOCSIM_PROGRESS", "compact").lower()
    use_tqdm = progress_mode == "tqdm"
    cached_count = len(coded)
    if len(pending_rows):
        print(
            f"Coding BRT text with OpenAI: {cached_count}/{len(texts)} cached, {len(pending_rows)} pending, concurrency={workers}.",
            file=sys.stderr,
            flush=True,
        )
    if use_tqdm:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_code_one, row) for row in pending_rows]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Coding BRT",
                mininterval=2.0,
                dynamic_ncols=False,
                ncols=80,
            ):
                coded.append(future.result())
    elif len(pending_rows):
        started = time.time()
        next_update = started
        completed = 0
        ok = cached_count
        failed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_code_one, row) for row in pending_rows]
            for future in as_completed(futures):
                record = future.result()
                coded.append(record)
                completed += 1
                if record.get("BRT_llm_reason") == "ok":
                    ok += 1
                else:
                    failed += 1
                now = time.time()
                if now >= next_update or completed == len(pending_rows):
                    elapsed = now - started
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = len(pending_rows) - completed
                    eta = remaining / rate if rate > 0 else float("nan")
                    eta_text = f"{eta/60:.1f}m" if eta == eta else "?"
                    msg = (
                        f"\rCoding BRT text: {ok}/{len(texts)} coded, "
                        f"{failed} failed, {remaining} pending, {rate:.2f}/s, ETA {eta_text}"
                    )
                    print(msg, end="", file=sys.stderr, flush=True)
                    next_update = now + _progress_interval()
        print(file=sys.stderr, flush=True)
    coded_df = pd.DataFrame(coded)
    coded_df["entity_id"] = coded_df["entity_id"].astype(str)
    coded_df = coded_df.drop_duplicates(subset=["entity_id"], keep="first")
    coded_df = _compute_derived(coded_df)
    ok_count = int((coded_df["BRT_llm_reason"] == "ok").sum()) if "BRT_llm_reason" in coded_df.columns else 0
    print(f"BRT coding complete: {ok_count}/{len(coded_df)} coded successfully.", file=sys.stderr, flush=True)
    return person.merge(coded_df, on="entity_id", how="left")


# ---------------------------------------------------------------------------
# ACTIVE: BRT v3.0 three-seat panel (CR-13 STEP 1/2). v1 `_code_brt_for_agents_v1`
# above is dead, present, not called.
# ---------------------------------------------------------------------------
# CR-13R 2026-09-23, PI RULING A. This gate now tests THIS RUN's live coding spend,
# not the cache cumulative.
#
# History: the gate used to compare token_est_spend_from_cache_dir() — the char/4
# token-estimate summed over every per-call file in the append-only cache, i.e. the
# whole campaign's history — against this constant. Two different quantities were
# being conflated. The cumulative estimate ignores cache hits (D-151d) and runs
# ≈2.9× hot: US$307.36 estimated against US$106.34 actually billed for CR-13.
# Setting the constant to the ruled A$260 ≈ US$172 (D-176 clause 3) therefore blocked
# every ingest outright, because the campaign's own history already exceeded it.
#
# The constant is now a per-run cap on usage objects, which is what its name says and
# what the job prompts mean by "BRT coding spend cap US$N". CR-13R's cap is US$5.
# The cumulative figure is still computed and printed each call, for the record.
BRT_CODING_SPEND_GATE_USD = 5.0
_BRT_SPEND_USD = 0.0
_V24_ALIASES = {
    "v24_BRT2_research_specificity": "BRT2_research_specificity",
    "v24_BRT1_reflective_keep": "BRT1_reflective_keep",
    "v24_BRT2_reflective_accept": "BRT2_reflective_accept",
    "v24_BRT2_high_quality_research": "BRT2_high_quality_research",
}


def brt_coding_spend_usd() -> float:
    return _BRT_SPEND_USD


def code_brt_for_agents(
    person_df: pd.DataFrame,
    items_df: pd.DataFrame,
    existing_coded: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Agent BRT coding via brt_v3.code_brt_v3 (rubric v3.0, three-seat panel)."""
    del existing_coded  # v3 cache is content-addressed; intermediate CSV is not the cache
    global _BRT_SPEND_USD
    require_columns(person_df, ["entity_id"], "agent_person")
    require_columns(items_df, ["entity_id", "item_id", "response_text"], "agent_items")
    person = person_df.copy()
    person["entity_id"] = person["entity_id"].astype(str)
    items = items_df.copy()
    items["entity_id"] = items["entity_id"].astype(str)
    texts = (
        items.loc[items["item_id"].astype(str).isin(["brt_1", "brt_2"]), ["entity_id", "item_id", "response_text"]]
        .pivot_table(index="entity_id", columns="item_id", values="response_text", aggfunc="first")
        .reset_index()
    )
    if "brt_1" not in texts.columns:
        texts["brt_1"] = ""
    if "brt_2" not in texts.columns:
        texts["brt_2"] = ""

    run_label = ""
    for col in ("run_label", "socsim_run_label", "dataset"):
        if col in person.columns and person[col].notna().any():
            run_label = str(person[col].dropna().iloc[0])
            break
    rows = pd.DataFrame(
        {
            "entity_id": texts["entity_id"].astype(str),
            "brt_1": texts["brt_1"],
            "brt_2": texts["brt_2"],
            "source": "agent",
            "run_label": run_label or "socsim_agent",
        }
    )

    from pathlib import Path
    from datetime import datetime, timezone

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "brt_v3"))
    from code_brt_v3 import (  # noqa: E402
        code_brt_v3,
        estimate_spend,
        prompt_sha,
        token_est_spend_from_cache_dir,
    )

    cache_dir = repo / "brt_v3" / "cache_cr13_agent"
    safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (run_label or "socsim_agent"))
    adj_path = cache_dir / f"adjudications_{safe_label or 'socsim_agent'}.jsonl"
    prior = token_est_spend_from_cache_dir(cache_dir)

    print(
        f"[brt_v3] coding {len(rows)} rows rubric=v3.0 cache_dir={cache_dir} "
        f"adjudications={adj_path.name} prior_cache_spend=${float(prior.get('total', 0.0)):.4f}",
        file=sys.stderr,
        flush=True,
    )
    consensus_df, _long_df, pi_review, counters = code_brt_v3(
        rows,
        text_col_1="brt_1",
        text_col_2="brt_2",
        id_col="entity_id",
        source_col="source",
        run_label_col="run_label",
        cache_dir=cache_dir,
        adjudications_path=adj_path,
        transport=os.environ.get("BRT_V3_TRANSPORT", "sync"),
    )
    live = estimate_spend(counters)
    cumulative = token_est_spend_from_cache_dir(cache_dir)
    # PI ruling A (CR-13R): the gate tracks this run's live spend, accumulated across
    # every code_brt_for_agents() call in this process. The cumulative cache estimate
    # is still printed, but it no longer gates.
    _BRT_SPEND_USD += float(live.get("total", 0.0))
    print(
        f"[brt_v3] this_call_live=${live.get('total', 0.0):.4f} "
        f"run_spend=${_BRT_SPEND_USD:.4f} gate=${BRT_CODING_SPEND_GATE_USD:.2f} "
        f"cumulative_cache=${float(cumulative.get('total', 0.0)):.4f} "
        f"n_cache_files={cumulative.get('n_files', 0)} "
        f"splits={len(pi_review)}",
        file=sys.stderr,
        flush=True,
    )
    if _BRT_SPEND_USD > BRT_CODING_SPEND_GATE_USD:
        raise SystemExit(
            f"STOP: BRT coding run spend ${_BRT_SPEND_USD:.4f} exceeded gate "
            f"${BRT_CODING_SPEND_GATE_USD:.2f}"
        )

    coded = consensus_df.copy()
    coded["entity_id"] = coded["entity_id"].astype(str)
    for src, dst in _V24_ALIASES.items():
        if src in coded.columns and dst not in coded.columns:
            coded[dst] = coded[src]
    coded["BRT_llm_reason"] = "ok"
    prompt_v = prompt_sha()[:12]
    coded["BRT_scoring_scorer_version"] = "brt_v3"
    coded["BRT_scoring_prompt_version"] = prompt_v
    coded["BRT_scoring_run_id"] = f"cr13-brt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    coded["BRT_scoring_date"] = datetime.now(timezone.utc).date().isoformat()
    n_split_entities = coded[[c for c in coded.columns if c.endswith("_agreement")]].eq("split").any(axis=1).sum() if any(
        c.endswith("_agreement") for c in coded.columns
    ) else 0
    print(
        f"BRT v3.0 coding complete: {len(coded)}/{len(rows)} coded; "
        f"panel-split entities={int(n_split_entities)} field-splits={len(pi_review)}",
        file=sys.stderr,
        flush=True,
    )
    overlap = [c for c in coded.columns if c in person.columns and c != "entity_id"]
    if overlap:
        person = person.drop(columns=overlap)
    return person.merge(coded, on="entity_id", how="left")

