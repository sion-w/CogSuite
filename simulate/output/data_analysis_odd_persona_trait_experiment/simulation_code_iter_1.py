#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import math
import os
import random
import re
import threading
import traceback
import time
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a required dependency
    load_dotenv = None


# ---------------------------------------------------------------------
# Global configuration and reproducibility
# ---------------------------------------------------------------------

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if PROJECT_ROOT is None:
    # This script lives in output/data_analysis_odd_persona_trait_experiment/,
    # two levels below the simulator root (the simulate/ folder), which holds
    # data_fitting/, simulation_outputs/ and the optional `.env` file.
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

# API keys may be kept in a git-ignored `.env` file in the simulator root. The
# file is resolved from PROJECT_ROOT, so the entry point finds it from any
# working directory.
if load_dotenv is not None:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DATA_PATH = os.environ.get("DATA_PATH")
if DATA_PATH is None:
    # Default data folder: data_fitting/persona_trait_experiment/ under PROJECT_ROOT.
    DATA_DIR = os.path.join(PROJECT_ROOT, "data_fitting", "persona_trait_experiment")
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

PROMPT_CONFIG_DIR = os.path.join(DATA_DIR, "prompt_config")

GLOBAL_RANDOM_SEED = 42
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)

DEFAULT_INCLUDED_BLOCKS = ["writing", "reasoning", "nfc", "bfi"]
BLOCK_TO_TEST_NAMES = {
    "writing": ["Writing"],
    "reasoning": ["CRT2", "bCRT", "dCRT"],
    "nfc": ["NFC"],
    "bfi": ["BFI"],
}


def resolve_included_blocks(tests_included: Optional[List[str]]) -> List[str]:
    requested = set(tests_included or [])
    if not requested:
        return list(DEFAULT_INCLUDED_BLOCKS)
    return [block_name for block_name in DEFAULT_INCLUDED_BLOCKS if block_name in requested]


def expand_included_test_names(tests_included: Optional[List[str]]) -> List[str]:
    test_names: List[str] = []
    for block_name in resolve_included_blocks(tests_included):
        test_names.extend(BLOCK_TO_TEST_NAMES[block_name])
    return test_names


def slugify_run_label(label: Optional[str]) -> str:
    if not label:
        return ""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("._- ")


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """
    Configuration for a simulation run.
    """

    personas_filename: str = "synthetic_personas.csv"
    human_mode: bool = False
    markers_filename: str = "serapio_goldberg_markers.csv"
    admin_mode: str = "per_test"
    shuffle_items_within_tests: bool = False
    shuffle_test_order: bool = False
    randomize_reasoning_block: bool = True
    holdout_fraction: float = 0.0
    max_personas: Optional[int] = None
    persona_ids: List[int] = field(default_factory=list)
    sampling_mode: str = "sequential"
    time_column: Optional[str] = None
    random_seed: int = GLOBAL_RANDOM_SEED
    output_dir: str = "simulation_outputs"
    max_parallel_personas: int = 4
    # Default to the real synthetic_personas.csv schema
    bfi_columns: Dict[str, str] = field(
        default_factory=lambda: {
            "extraversion": "BFI_Extraversion",
            "agreeableness": "BFI_Agreeableness",
            "conscientiousness": "BFI_Conscientiousness",
            "neuroticism": "BFI_Neuroticism",
            "openness": "BFI_Openness",
        }
    )
    nfc_column: str = "NFC_Total"
    crt_column: str = "CRT2_Total_adj"
    crt_level_column: str = "crt2_level"
    crt_cutoffs_filename: str = "crt2_level_cutoffs.yaml"
    use_crt_numeric_only: bool = False
    use_openai: bool = False
    openai_model: str = "gpt-5"
    openai_max_output_tokens: int = 4000
    # Provider settings for --use-openai: a base-URL / key-env override routes
    # calls to OpenAI-compatible endpoints (DeepSeek, Gemini, Together); the
    # API style selects the Responses API (native OpenAI models) or chat
    # completions (the compatible endpoints implement chat completions only);
    # the extra params/body carry each provider's reasoning-control setting.
    openai_base_url: Optional[str] = None
    openai_key_env: str = "OPENAI_API_KEY"
    openai_api_style: str = "responses"  # "responses" or "chat_completions"
    openai_extra_params: Dict[str, Any] = field(default_factory=dict)
    openai_extra_body: Dict[str, Any] = field(default_factory=dict)
    max_api_retries: int = 5
    use_anthropic: bool = False
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_output_tokens: int = 8000
    # None omits `temperature` from the Anthropic Messages API call (provider
    # default, 1.0). A set value, including 0.0, is passed verbatim. top_p is
    # never set.
    temperature: Optional[float] = None
    # None omits `thinking` from the Anthropic Messages API call (provider
    # default). A set value is passed verbatim, e.g. {"type": "disabled"}.
    anthropic_thinking: Optional[Dict[str, Any]] = None
    use_pseudo: bool = False
    # Optional final paragraph appended to the rendered persona system
    # message. None appends nothing.
    persona_system_suffix: Optional[str] = None
    # Persona prompt controls
    persona_spec_mode: str = "lexical_only"  # lexical_only, numeric_only, both
    bfi_ladder_granularity: int = 9  # 5 or 9
    include_user_context: bool = False
    multi_turn_conversation: bool = True
    # Persona prompt style for BFI portion: granular_serapio, coarse_numeric, coarse_descriptive
    persona_prompt_style: str = "granular_serapio"
    run_label: Optional[str] = None
    persona_prompt_version: Optional[str] = None
    task_prompt_version: Optional[str] = None
    crt_binning_method: str = "fixed_human_crt2_cutoffs"
    # Runtime provenance captured once input/output paths and sampled personas are known.
    personas_path: Optional[str] = None
    persona_bank_size: Optional[int] = None
    sample_n_requested: Optional[int] = None
    sample_n_selected: Optional[int] = None
    sampling_stratify_source: Optional[str] = None
    crt_cutoffs_path: Optional[str] = None
    crt_cutoffs_values: Optional[Dict[str, float]] = None
    tests_included: List[str] = field(default_factory=list)
    tests_included_explicitly_set: bool = False
    resolved_output_dir: Optional[str] = None

    def validate(self) -> None:
        """
        Validate configuration values.
        """
        if self.admin_mode not in {"per_item", "per_test", "all_tests"}:
            raise ValueError(
                f"Invalid admin_mode '{self.admin_mode}'. "
                "Must be one of: 'per_item', 'per_test', 'all_tests'."
            )
        if not (0.0 <= self.holdout_fraction < 1.0):
            raise ValueError(
                f"holdout_fraction must be in [0, 1), got "
                f"{self.holdout_fraction}."
            )
        if self.max_personas is not None and self.max_personas <= 0:
            raise ValueError("max_personas must be a positive integer or None.")
        if any(pid < 0 for pid in self.persona_ids):
            raise ValueError("persona_ids must be non-negative integers.")
        if self.sampling_mode not in {"sequential", "random", "stratified_crt"}:
            raise ValueError(
                "sampling_mode must be one of: sequential, random, stratified_crt."
            )
        if self.sampling_mode != "sequential" and self.max_personas is None:
            raise ValueError(
                "max_personas must be provided when sampling_mode is "
                "'random' or 'stratified_crt'."
            )
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if self.max_parallel_personas <= 0:
            raise ValueError("max_parallel_personas must be a positive integer.")
        if not self.bfi_columns:
            raise ValueError("bfi_columns mapping must not be empty.")
        if self.persona_prompt_style not in {
            "granular_serapio",
            "coarse_numeric",
            "coarse_descriptive",
        }:
            raise ValueError(
                "persona_prompt_style must be one of "
                "'granular_serapio', 'coarse_numeric', 'coarse_descriptive'."
            )
        if self.persona_spec_mode not in {"lexical_only", "numeric_only", "both"}:
            raise ValueError(
                "persona_spec_mode must be one of: lexical_only, numeric_only, both."
            )
        if self.bfi_ladder_granularity not in {5, 9}:
            raise ValueError("bfi_ladder_granularity must be 5 or 9.")
        if not isinstance(self.multi_turn_conversation, bool):
            raise ValueError("multi_turn_conversation must be a boolean.")
        invalid_tests_included = [
            name
            for name in self.tests_included
            if name not in set(DEFAULT_INCLUDED_BLOCKS)
        ]
        if invalid_tests_included:
            raise ValueError(
                "tests_included must only contain: "
                + ", ".join(DEFAULT_INCLUDED_BLOCKS)
                + f". Got: {invalid_tests_included}."
            )
        if self.tests_included:
            self.tests_included = resolve_included_blocks(self.tests_included)
        if not isinstance(self.tests_included_explicitly_set, bool):
            raise ValueError("tests_included_explicitly_set must be a boolean.")
        if not self.crt_cutoffs_filename:
            raise ValueError("crt_cutoffs_filename must not be empty.")
        if self.openai_api_style not in {"responses", "chat_completions"}:
            raise ValueError(
                "openai_api_style must be one of: responses, chat_completions."
            )
        if not self.openai_key_env:
            raise ValueError("openai_key_env must not be empty.")
        if self.max_api_retries < 0:
            raise ValueError("max_api_retries must be non-negative.")
        selected = [
            flag
            for flag in (self.use_openai, self.use_anthropic, self.use_pseudo)
            if flag
        ]
        if len(selected) > 1:
            raise ValueError(
                "Choose only one LLM provider: pseudo, OpenAI, or Anthropic."
            )


@dataclass
class Persona:
    """
    Structured representation of a single persona, derived from a row of
    synthetic_personas.csv.
    """

    persona_id: Any
    bfi_traits: Dict[str, float]
    nfc_score: float
    crt2_score: float
    crt2_level: int
    demographics: Dict[str, Any]

    @classmethod
    def from_series(cls, row: pd.Series, config: SimulationConfig) -> "Persona":
        """
        Build a Persona instance from a pandas Series using the schema in
        SimulationConfig.
        """
        persona_id = row.get("persona_id", None)
        bfi_traits = {
            domain: float(row[col])
            for domain, col in config.bfi_columns.items()
        }
        nfc_score = float(row[config.nfc_column])
        crt2_score = float(row[config.crt_column])
        crt2_level = int(row[config.crt_level_column])

        demo_cols = [
            "Country",
            "Employment Status",
            "Occupation_cat",
            "sex",
            "Highest education level completed",
            "Ethnicity Simplified",
            "Age",
        ]
        demographics: Dict[str, Any] = {}
        for col in demo_cols:
            if col in row:
                demographics[col] = row[col]

        return cls(
            persona_id=persona_id,
            bfi_traits=bfi_traits,
            nfc_score=nfc_score,
            crt2_score=crt2_score,
            crt2_level=crt2_level,
            demographics=demographics,
        )


@dataclass
class TestItem:
    """
    Single test item (question or writing task).
    """

    item_id: str
    test_name: str
    scale_name: Optional[str]
    item_text: str
    trait: Optional[str]
    keyed: int
    item_type: str
    correct_answer: Optional[str] = None
    lure_answers: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestBank:
    """
    Collection of all psychometric tests and writing tasks.
    """

    items: List[TestItem]

    def items_by_test(self) -> Dict[str, List[TestItem]]:
        grouped: Dict[str, List[TestItem]] = {}
        for item in self.items:
            grouped.setdefault(item.test_name, []).append(item)
        return grouped


@dataclass
class SimulationResult:
    """
    Container for simulation outputs.
    """

    per_persona: pd.DataFrame
    per_item: pd.DataFrame
    config: SimulationConfig
    api_calls: Optional[pd.DataFrame] = None
    failed_personas: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------
# API reliability: retry/backoff and failure accounting
# ---------------------------------------------------------------------
#
# API errors are retried with backoff and, once retries are exhausted, raised
# as LLMHardFailure; they are never returned as empty text. An empty response
# would be indistinguishable downstream from a persona that gave no parseable
# answer, and would enter the "missing" category that forms part of the
# denominators of the lure-prevalence estimates. Used by the OpenAI-family
# paths and the Anthropic Messages path.


class LLMHardFailure(Exception):
    """
    Raised when an LLM API call exhausts retries or hits a non-retryable
    error. Callers MUST record this as a hard failure (exception type +
    status code) and MUST NOT coerce it into an empty string.
    """

    def __init__(
        self,
        provider: str,
        exception_type: str,
        status_code: Optional[int],
        message: str,
    ):
        self.provider = provider
        self.exception_type = exception_type
        self.status_code = status_code
        super().__init__(
            f"[{provider}] hard failure after retries: {exception_type} "
            f"(status={status_code}): {message}"
        )


class RunCounters:
    """Thread-safe per-run counters for API reliability reporting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_calls = 0
        self.retried_calls = 0
        self.hard_failures = 0
        self.empty_or_unparseable_items = 0
        self.truncated_responses = 0

    def reset(self) -> None:
        with self._lock:
            self.total_calls = 0
            self.retried_calls = 0
            self.hard_failures = 0
            self.empty_or_unparseable_items = 0
            self.truncated_responses = 0

    def record_call(self, retried: bool = False) -> None:
        with self._lock:
            self.total_calls += 1
            if retried:
                self.retried_calls += 1

    def record_hard_failure(self) -> None:
        with self._lock:
            self.hard_failures += 1

    def record_empty_or_unparseable(self, n: int = 1) -> None:
        with self._lock:
            self.empty_or_unparseable_items += n

    def record_truncated(self) -> None:
        with self._lock:
            self.truncated_responses += 1

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            total = self.total_calls
            return {
                "total_api_calls": total,
                "retried_calls": self.retried_calls,
                "hard_failures": self.hard_failures,
                "hard_failure_rate": (self.hard_failures / total) if total else 0.0,
                "empty_or_unparseable_items": self.empty_or_unparseable_items,
                "truncated_responses": self.truncated_responses,
            }


# Module-level singleton: one simulator process = one run.
RUN_COUNTERS = RunCounters()


def _extract_status_code(exc: BaseException) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    return None


def _extract_retry_after(exc: BaseException) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    for key in ("retry-after", "Retry-After"):
        try:
            value = headers.get(key)
        except Exception:
            value = None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _is_retryable_status(status_code: Optional[int]) -> bool:
    if status_code is None:
        return False
    return status_code == 429 or 500 <= status_code < 600


def call_with_retry(
    fn,
    provider: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    counters: Optional["RunCounters"] = None,
):
    """
    Call `fn()` with bounded exponential backoff + jitter on HTTP 429/5xx,
    honouring `Retry-After` when present. Never converts an exception into
    an empty string: once retries are exhausted, or the error is
    non-retryable (e.g. 4xx other than 429), raises `LLMHardFailure` with the
    exception type and status code for the caller to record explicitly.
    """
    counters = counters if counters is not None else RUN_COUNTERS
    attempt = 0
    while True:
        try:
            result = fn()
            counters.record_call(retried=(attempt > 0))
            return result
        except Exception as exc:  # noqa: BLE001 - re-raised as LLMHardFailure below
            status_code = _extract_status_code(exc)
            retryable = _is_retryable_status(status_code)
            if not retryable or attempt >= max_retries:
                counters.record_call(retried=(attempt > 0))
                counters.record_hard_failure()
                raise LLMHardFailure(
                    provider=provider,
                    exception_type=type(exc).__name__,
                    status_code=status_code,
                    message=str(exc),
                ) from exc
            retry_after = _extract_retry_after(exc)
            delay = (
                retry_after
                if retry_after is not None
                else min(max_delay, base_delay * (2 ** attempt))
            )
            delay = max(0.0, delay) * (0.5 + random.random())  # jitter, +/-50%
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------
# OpenAI integration
# ---------------------------------------------------------------------


def get_openai_api_key(key_env: str = "OPENAI_API_KEY") -> str:
    api_key = os.environ.get(key_env)
    if api_key:
        return api_key
    raise ValueError(f"OpenAI-compatible API key not found in environment variable '{key_env}'")


@dataclass
class LLMCallResult:
    """Uniform return shape across the Responses API and OpenAI-compatible
    chat-completions paths, carrying the truncation indicators that flag a
    response cut off at the output-token limit."""

    text: str
    response_id: Optional[str] = None
    truncated: bool = False
    finish_reason: Optional[str] = None


def call_gpt5_with_responses_api(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-5",
    max_output_tokens: int = 4000,
    previous_response_id: Optional[str] = None,
    base_url: Optional[str] = None,
    key_env: str = "OPENAI_API_KEY",
    extra_params: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
) -> LLMCallResult:
    """
    Native OpenAI Responses API path (e.g. `gpt-5.6-sol`). `extra_params` is
    merged verbatim into the top-level `responses.create(...)` kwargs, e.g.
    `{"reasoning": {"effort": "none"}}`. Retries with bounded exponential
    backoff on 429/5xx; never returns "" on failure: raises `LLMHardFailure`
    once retries are exhausted.
    """
    api_key = get_openai_api_key(key_env)
    from openai import OpenAI

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    responses_kwargs: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }
    if previous_response_id:
        responses_kwargs["previous_response_id"] = previous_response_id
    if extra_params:
        responses_kwargs.update(extra_params)

    resp = call_with_retry(
        lambda: client.responses.create(**responses_kwargs),
        provider="openai_responses",
        max_retries=max_retries,
    )

    def extract_response(resp_obj: Any) -> str:
        if hasattr(resp_obj, "output_text") and isinstance(
            getattr(resp_obj, "output_text"), str
        ):
            return getattr(resp_obj, "output_text")
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list):
                content = (
                    output[0].get("content") if isinstance(output[0], dict) else None
                )
                if content and isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text")
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        return str(resp_obj)

    response_id = getattr(resp, "id", None)
    if response_id is not None and not isinstance(response_id, str):
        response_id = str(response_id)

    status = getattr(resp, "status", None)
    truncated = status == "incomplete"
    finish_reason: Optional[str] = None
    incomplete_details = getattr(resp, "incomplete_details", None)
    if incomplete_details is not None:
        if isinstance(incomplete_details, dict):
            finish_reason = incomplete_details.get("reason")
        else:
            finish_reason = getattr(incomplete_details, "reason", None)
    if truncated:
        RUN_COUNTERS.record_truncated()

    return LLMCallResult(
        text=extract_response(resp),
        response_id=response_id,
        truncated=truncated,
        finish_reason=finish_reason,
    )


def call_openai_chat_completions_api(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int = 4000,
    prior_messages: Optional[List[Dict[str, str]]] = None,
    base_url: Optional[str] = None,
    key_env: str = "OPENAI_API_KEY",
    extra_params: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
) -> LLMCallResult:
    """
    OpenAI-compatible chat-completions path (DeepSeek `deepseek-v4-flash` /
    `deepseek-v4-pro`; Google's Gemini OpenAI-compatibility layer for
    `gemini-3.5-flash`; Together-hosted models). Not the Responses API: these
    compatibility layers implement Chat Completions only.
    `extra_params` carries top-level fields; `extra_body` carries
    provider-specific nested fields via the SDK's `extra_body=` kwarg
    (DeepSeek `{"thinking": {"type": "enabled"|"disabled"}}`; Gemini
    `{"extra_body": {"google": {"thinking_config": {"thinking_level": ...}}}}`).
    Note the doubled `extra_body` key for Gemini: the openai-python SDK's
    own `extra_body=` kwarg merges its value directly into the outgoing
    JSON body, and Gemini's OpenAI-compatibility layer expects an
    `extra_body` field *within* that JSON body to carry Gemini-only fields,
    so the SDK kwarg must be given `{"extra_body": {...}}` to land there.
    A single-nested `{"google": {...}}` is rejected server-side with
    `Unknown name "google": Cannot find field.` (see the doubled-nesting
    Python examples at https://ai.google.dev/gemini-api/docs/openai#extra_body).
    Retries with bounded exponential backoff on 429/5xx; never returns ""
    on failure: raises `LLMHardFailure` once retries are exhausted.
    Maintains its own message history (Chat Completions has no
    `previous_response_id`).
    """
    api_key = get_openai_api_key(key_env)
    from openai import OpenAI

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if prior_messages:
        for msg in prior_messages:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", ""))
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    completion_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if extra_params:
        completion_kwargs.update(extra_params)
    if extra_body:
        completion_kwargs["extra_body"] = extra_body

    resp = call_with_retry(
        lambda: client.chat.completions.create(**completion_kwargs),
        provider="openai_compatible_chat_completions",
        max_retries=max_retries,
    )

    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    text = ""
    if choice is not None:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            text = content
    finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
    truncated = finish_reason == "length"
    if truncated:
        RUN_COUNTERS.record_truncated()

    return LLMCallResult(
        text=text or "",
        response_id=None,
        truncated=truncated,
        finish_reason=finish_reason,
    )


def get_anthropic_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    raise ValueError("Anthropic API key not found in environment")


def call_claude_messages_api(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 8000,
    prior_messages: Optional[List[Dict[str, str]]] = None,
    temperature: Optional[float] = None,
    thinking: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
) -> str:
    """
    Anthropic Messages API path. When `temperature` is None the parameter
    is omitted from the request (provider default, 1.0). When set,
    including 0.0, it is passed verbatim. `top_p` is never set.
    Optional `thinking` is passed to `messages.create` only when provided;
    when None the field is omitted (provider default).
    Retries with bounded exponential backoff on 429/5xx; never returns ""
    on failure: raises `LLMHardFailure` once retries are exhausted.
    """
    api_key = get_anthropic_api_key()
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    call_messages: List[Dict[str, str]] = []
    if prior_messages:
        for msg in prior_messages:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", ""))
            if role in {"user", "assistant"}:
                call_messages.append({"role": role, "content": content})
    call_messages.append({"role": "user", "content": user_prompt})

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "max_tokens": max_tokens,
        "messages": call_messages,
    }
    # Omitted when None, so the provider default applies; 0.0 is sent as a
    # literal 0.
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    # Sent only when provided.
    if thinking is not None:
        create_kwargs["thinking"] = thinking

    message = call_with_retry(
        lambda: client.messages.create(**create_kwargs),
        provider="anthropic_messages",
        max_retries=max_retries,
    )

    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        if parts:
            return "".join(parts)
    if hasattr(message, "text") and isinstance(message.text, str):
        return message.text
    return str(message)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.nunique() <= 1 or y.nunique() <= 1:
        return float("nan")
    return float(x.corr(y))


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    return pearson_corr(x.rank(method="average"), y.rank(method="average"))


def cronbach_alpha(df: pd.DataFrame) -> float:
    k = df.shape[1]
    if k < 2 or df.shape[0] < 2:
        return float("nan")

    item_vars = df.var(axis=0, ddof=1)
    total_score = df.sum(axis=1)
    total_var = total_score.var(ddof=1)
    if total_var == 0:
        return float("nan")

    alpha = (k / (k - 1.0)) * (1.0 - item_vars.sum() / total_var)
    return float(alpha)


def _resolve_data_dir_path(path_or_filename: str) -> str:
    """
    Resolve a file path relative to DATA_DIR unless an absolute path is given.
    """
    if os.path.isabs(path_or_filename):
        return path_or_filename
    return os.path.join(DATA_DIR, path_or_filename)


def load_crt_cutoff_config(path_or_filename: str) -> Tuple[str, Dict[str, float]]:
    """
    Load fixed CRT2 cutoff thresholds from YAML.
    """
    resolved_path = _resolve_data_dir_path(path_or_filename)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            "CRT cutoff YAML file not found: "
            f"'{path_or_filename}' (resolved to '{resolved_path}')."
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "CRT cutoff YAML must contain a mapping. Expected either top-level cutoff keys "
            "or a nested 'cutoffs' mapping."
        )

    if "cutoffs" in data:
        cutoff_container = data["cutoffs"]
    else:
        cutoff_container = data
    if not isinstance(cutoff_container, dict):
        raise ValueError(
            "CRT cutoff YAML 'cutoffs' entry must be a mapping containing "
            "'q1_20', 'q2_40', 'q3_60', and 'q4_80'. "
            f"Resolved path: '{resolved_path}'."
        )

    required_keys = ["q1_20", "q2_40", "q3_60", "q4_80"]
    missing_keys = [key for key in required_keys if key not in cutoff_container]
    if missing_keys:
        available_keys = sorted(str(key) for key in cutoff_container.keys())
        raise ValueError(
            "CRT cutoff YAML is missing required cutoff keys: "
            f"{missing_keys}. Resolved path: '{resolved_path}'. "
            f"Available keys: {available_keys}."
        )

    cutoffs = {key: float(cutoff_container[key]) for key in required_keys}
    if not (
        cutoffs["q1_20"]
        <= cutoffs["q2_40"]
        <= cutoffs["q3_60"]
        <= cutoffs["q4_80"]
    ):
        raise ValueError(
            "CRT cutoff YAML must satisfy q1_20 <= q2_40 <= q3_60 <= q4_80."
        )

    return os.path.abspath(resolved_path), cutoffs


def map_crt_total_to_level(
    scores: pd.Series, cutoffs: Dict[str, float]
) -> pd.Series:
    """
    Map continuous CRT total scores into discrete 0-4 levels using
    fixed cutoff thresholds.
    """
    scores = scores.astype(float)
    q1 = cutoffs["q1_20"]
    q2 = cutoffs["q2_40"]
    q3 = cutoffs["q3_60"]
    q4 = cutoffs["q4_80"]

    def to_level(val: float) -> int:
        if val <= q1:
            return 0
        if val <= q2:
            return 1
        if val <= q3:
            return 2
        if val <= q4:
            return 3
        return 4

    return scores.apply(to_level).astype(int)


def prepare_crt_levels(
    personas_df: pd.DataFrame, config: SimulationConfig
) -> pd.DataFrame:
    """
    Coerce the continuous CRT column to numeric and derive the discrete
    CRT level column used in persona prompting.
    """
    prepared = personas_df.copy()
    crt_vals = pd.to_numeric(prepared[config.crt_column], errors="coerce")
    missing_count = int(crt_vals.isna().sum())
    if missing_count:
        print(
            f"Error: {missing_count} persona(s) have no numeric CRT2 adjusted score. "
            "Every persona needs a value in the CRT column; complete the personas file and rerun.",
            flush=True,
        )
        raise ValueError(
            f"Column '{config.crt_column}' must be numeric and non-missing "
            "for all personas."
        )

    cutoffs_path, cutoffs = load_crt_cutoff_config(config.crt_cutoffs_filename)
    config.crt_cutoffs_path = cutoffs_path
    config.crt_cutoffs_values = cutoffs
    prepared[config.crt_column] = crt_vals
    prepared[config.crt_level_column] = map_crt_total_to_level(
        prepared[config.crt_column], cutoffs
    )
    return prepared


def sample_personas_for_run(
    personas_df: pd.DataFrame, config: SimulationConfig
) -> pd.DataFrame:
    """
    Select the personas used for this run according to the configured
    sampling mode.
    """
    if config.max_personas is None:
        return personas_df.copy()

    n = int(config.max_personas)
    if config.sampling_mode == "sequential":
        return personas_df.head(n).copy()

    if n > len(personas_df):
        raise ValueError(
            f"Requested {n} personas with sampling_mode='{config.sampling_mode}', "
            f"but only {len(personas_df)} rows are available."
        )

    rng = np.random.default_rng(config.random_seed)
    positions = np.arange(len(personas_df))

    if config.sampling_mode == "random":
        chosen_positions = rng.choice(positions, size=n, replace=False)
        return personas_df.iloc[chosen_positions].copy()

    if config.sampling_mode != "stratified_crt":
        raise ValueError(f"Unsupported sampling_mode: {config.sampling_mode}")

    level_series = personas_df[config.crt_level_column].astype(int)
    levels = list(range(5))
    base_allocation = n // len(levels)
    remainder = n % len(levels)
    allocations = {level: base_allocation for level in levels}

    if remainder:
        extra_levels = rng.permutation(levels)[:remainder]
        for level in extra_levels:
            allocations[int(level)] += 1

    available = {
        level: int((level_series == level).sum())
        for level in levels
    }
    insufficient = {
        level: {
            "requested": allocations[level],
            "available": available[level],
        }
        for level in levels
        if allocations[level] > available[level]
    }
    if insufficient:
        raise ValueError(
            "Unable to satisfy stratified_crt sampling without replacement. "
            f"Requested allocations by derived CRT bin: {allocations}; "
            f"available rows by derived CRT bin: {available}; "
            f"insufficient bins: {insufficient}."
        )

    selected_positions: List[np.ndarray] = []
    for level in levels:
        take = allocations[level]
        if take <= 0:
            continue
        level_positions = positions[level_series.to_numpy() == level]
        chosen = rng.choice(level_positions, size=take, replace=False)
        selected_positions.append(chosen)

    if not selected_positions:
        return personas_df.iloc[[]].copy()

    merged_positions = np.concatenate(selected_positions)
    rng.shuffle(merged_positions)
    return personas_df.iloc[merged_positions].copy()


def filter_personas_by_ids(
    personas_df: pd.DataFrame, config: SimulationConfig
) -> pd.DataFrame:
    """
    Keep only explicitly requested persona IDs, preserving CSV order.
    """
    if not config.persona_ids:
        return personas_df

    requested_ids = set(config.persona_ids)
    filtered = personas_df[personas_df["persona_id"].isin(requested_ids)].copy()
    found_ids = set(filtered["persona_id"].astype(int).tolist())
    missing_ids = sorted(requested_ids - found_ids)
    if missing_ids:
        raise ValueError(
            "Requested persona IDs were not found in the selected personas file: "
            f"{missing_ids}."
        )
    return filtered


# ---------------------------------------------------------------------
# Persona prompt construction
# ---------------------------------------------------------------------


class PersonaPromptBuilder:
    """
    Build persona prompts based on trait scores and blueprint specifications.
    """

    QUALIFIER_SCALE_1_TO_9: Dict[int, str] = {
        1: "extremely {low_adjective}",
        2: "very {low_adjective}",
        3: "{low_adjective}",
        4: "a bit {low_adjective}",
        5: "neither {low_adjective} nor {high_adjective}",
        6: "a bit {high_adjective}",
        7: "{high_adjective}",
        8: "very {high_adjective}",
        9: "extremely {high_adjective}",
    }

    def __init__(
        self,
        config: SimulationConfig,
        markers_df: Optional[pd.DataFrame] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.config = config
        self.markers_df = markers_df
        self.rng = rng if rng is not None else np.random.default_rng(GLOBAL_RANDOM_SEED)
        self._validate_markers_df()
        self.persona_template = self._load_yaml_config("persona_template.yaml")
        self.bfi_ladders_9 = self._load_yaml_config("bfi_ladders_9.yaml")
        self.bfi_ladders_5 = self._load_yaml_config("bfi_ladders_5.yaml")
        self.nfc_ladder_9 = self._load_yaml_config("nfc_ladder_9.yaml")
        self.crt2_ladder = self._load_yaml_config("crt2_ladder.yaml")

    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(PROMPT_CONFIG_DIR, filename)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _validate_markers_df(self) -> None:
        if self.markers_df is None:
            return
        required = {"domain", "pole", "marker"}
        if not required.issubset(self.markers_df.columns):
            warnings.warn(
                "Markers DataFrame is missing required columns "
                f"{required}; falling back to default descriptors.",
                UserWarning,
            )
            self.markers_df = None

    @staticmethod
    def _score_to_9_level(score: float) -> int:
        """
        Map a 1–5 trait score to a 1–9 lexical ladder using fixed,
        absolute thresholds. The 9 levels provide richer prompt
        specification; they do not imply that the original psychometric
        scales have 9 observed response levels.
        """
        if score < 1.0 or score > 5.0:
            raise ValueError(
                f"Score for 1–5 mapping must be in [1, 5], got {score}."
            )
        if score <= 1.25:
            return 1
        if score <= 1.75:
            return 2
        if score <= 2.25:
            return 3
        if score <= 2.75:
            return 4
        if score <= 3.25:
            return 5
        if score <= 3.75:
            return 6
        if score <= 4.25:
            return 7
        if score <= 4.75:
            return 8
        return 9

    @staticmethod
    def _score_to_5_level(score: float) -> int:
        if score < 1.0 or score > 5.0:
            raise ValueError(
                f"Score for 1–5 mapping must be in [1, 5], got {score}."
            )
        return max(1, min(5, int(round(score))))

    def _bfi_descriptor(self, domain: str, score: float) -> str:
        if self.config.bfi_ladder_granularity == 5 and self.bfi_ladders_5:
            ladders = self.bfi_ladders_5.get("traits", {})
            level = self._score_to_5_level(score)
        else:
            ladders = self.bfi_ladders_9.get("traits", {})
            level = self._score_to_9_level(score)
        domain_key = domain.capitalize()
        text = (
            ladders.get(domain_key, {}).get(level)
            if isinstance(ladders.get(domain_key, {}), dict)
            else None
        )
        if isinstance(text, str) and text.strip():
            return text.strip()
        # Without a ladder entry, use the marker-based granular description.
        return self.build_bfi_granular_prompt({domain: score}).replace(
            'For the following tasks, respond as a person described as: "', ""
        ).rstrip('"')

    def _nfc_descriptor(self, nfc_score: float) -> str:
        levels = self.nfc_ladder_9.get("levels", {})
        level = self._score_to_9_level(nfc_score)
        text = levels.get(level) if isinstance(levels, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
        return self.build_nfc_prompt(nfc_score).replace(
            'For the following tasks, respond as a person described as: "I ', ""
        ).rstrip('."')

    def _crt_descriptor(self, crt_level: int) -> str:
        levels = self.crt2_ladder.get("levels", {})
        text = levels.get(crt_level) if isinstance(levels, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
        return self.build_crt_prompt(crt_level, numeric_only=False).replace(
            'For the following CRT-style questions, respond as a person described as: "', ""
        ).rstrip('"')

    def _sample_markers(self, domain: str, pole: str, n: int = 3) -> List[str]:
        """
        Sample up to n markers for a given Big Five domain and pole
        ('low' or 'high') from the Serapio/Goldberg markers file if
        available, otherwise use a small default set.
        """
        default_markers: Dict[str, Dict[str, List[str]]] = {
            "extraversion": {
                "low": ["quiet", "reserved", "shy"],
                "high": ["outgoing", "talkative", "energetic"],
            },
            "agreeableness": {
                "low": ["critical", "stubborn", "cold"],
                "high": ["kind", "cooperative", "warm"],
            },
            "conscientiousness": {
                "low": ["careless", "disorganized", "impulsive"],
                "high": ["organized", "reliable", "hard-working"],
            },
            "neuroticism": {
                "low": ["calm", "emotionally stable"],
                "high": ["anxious", "moody", "easily upset"],
            },
            "openness": {
                "low": ["conventional", "unimaginative"],
                "high": ["curious", "imaginative", "open to new ideas"],
            },
        }

        if self.markers_df is not None:
            subset = self.markers_df[
                (self.markers_df["domain"].str.lower() == domain.lower())
                & (self.markers_df["pole"].str.lower() == pole.lower())
            ]
            if not subset.empty:
                markers = subset["marker"].dropna().astype(str).tolist()
                if not markers:
                    return default_markers.get(domain, {}).get(pole, [])
                return list(
                    self.rng.choice(
                        markers, size=min(n, len(markers)), replace=False
                    )
                )

        return default_markers.get(domain, {}).get(pole, [])

    def build_bfi_granular_prompt(self, traits: Dict[str, float]) -> str:
        """
        Granular 1–9 level BFI prompt using Serapio/Goldberg markers.
        """
        segments: List[str] = []
        for domain, score in traits.items():
            level = self._score_to_9_level(score)
            template = self.QUALIFIER_SCALE_1_TO_9[level]
            if level <= 4:
                low = ", ".join(self._sample_markers(domain, "low", n=2))
                high = ", ".join(self._sample_markers(domain, "high", n=2))
            elif level >= 6:
                low = ", ".join(self._sample_markers(domain, "low", n=2))
                high = ", ".join(self._sample_markers(domain, "high", n=2))
            else:
                low = ", ".join(self._sample_markers(domain, "low", n=1))
                high = ", ".join(self._sample_markers(domain, "high", n=1))
            descriptor = template.format(
                low_adjective=low if low else "low on this trait",
                high_adjective=high if high else "high on this trait",
            )
            domain_label = domain.capitalize()
            segments.append(f"In terms of {domain_label}, I am {descriptor}.")

        joined = " ".join(segments)
        return (
            "For the following tasks, respond as a person described as: "
            f"\"{joined}\""
        )

    def build_bfi_coarse_numeric_prompt(self, traits: Dict[str, float]) -> str:
        """
        Coarse three-bin numeric BFI prompt.

        Mapping:
            low: score <= 2.5
            neutral: 2.5 < score < 3.5
            high: score >= 3.5
        """
        segments: List[str] = []
        for domain, score in traits.items():
            if score <= 2.5:
                polarity = "low"
            elif score >= 3.5:
                polarity = "high"
            else:
                polarity = "neutral"
            trait_name = domain.lower()
            segments.append(
                f"You are a person with {polarity} {trait_name}."
            )
        return " ".join(segments)

    def build_bfi_coarse_descriptive_prompt(self, traits: Dict[str, float]) -> str:
        """
        Coarse three-bin descriptive BFI prompt using marker adjectives.
        """
        descriptors: List[str] = []

        for domain, score in traits.items():
            if score <= 2.5:
                # Emphasize low pole
                descriptors.extend(self._sample_markers(domain, "low", n=3))
            elif score >= 3.5:
                # Emphasize high pole
                descriptors.extend(self._sample_markers(domain, "high", n=3))
            else:
                # Neutral: mix of both poles
                descriptors.extend(self._sample_markers(domain, "low", n=1))
                descriptors.extend(self._sample_markers(domain, "high", n=1))

        # Deduplicate while preserving order
        seen: Dict[str, None] = {}
        unique_descriptors = [
            d for d in descriptors if not (d in seen or seen.setdefault(d, None))
        ]

        if not unique_descriptors:
            return (
                "You are a person whose personality traits are broadly average "
                "across the Big Five dimensions."
            )

        descriptor_list = ", ".join(unique_descriptors)
        return f"You are a person who is {descriptor_list}."

    def build_nfc_prompt(self, nfc_score: float) -> str:
        """
        Granular 9-level Need for Cognition specification based on
        high/low descriptors and the 1–9 qualifier scale.
        """
        level = self._score_to_9_level(nfc_score)

        high_descriptors = [
            "enjoys complex rather than simple problems",
            "likes responsibility for situations that require a lot of thinking",
            "finds satisfaction in deliberating hard and for long periods",
            "enjoys tasks that involve generating new solutions to problems",
            "prefers life to be filled with puzzles and challenging questions",
            "finds abstract thinking appealing",
            "prefers intellectual, difficult, and important tasks",
            "often deliberates about issues even when they do not affect them "
            "personally",
        ]
        low_descriptors = [
            "thinks only as hard as necessary",
            "prefers tasks that require little thought once learned",
            "would rather do something that requires little thought",
            "tries to avoid situations that demand deep thinking",
            "finds thinking effortful rather than fun",
            "feels relief rather than satisfaction after heavy mental effort",
            "is content when something works without understanding how or why",
        ]

        if level <= 4:
            chosen_low = self.rng.choice(
                low_descriptors, size=3, replace=False
            ).tolist()
            qualifier = self.QUALIFIER_SCALE_1_TO_9[level]
            clause = ", ".join(chosen_low)
            sentence = qualifier.format(
                low_adjective=clause, high_adjective=""
            )
            sentence = sentence.replace("  ", " ").strip(", ")
            return (
                "For the following tasks, respond as a person described as: "
                f"\"I {sentence}.\""
            )
        elif level >= 6:
            chosen_high = self.rng.choice(
                high_descriptors, size=3, replace=False
            ).tolist()
            qualifier = self.QUALIFIER_SCALE_1_TO_9[level]
            clause = ", ".join(chosen_high)
            sentence = qualifier.format(
                low_adjective="", high_adjective=clause
            )
            sentence = sentence.replace("  ", " ").strip(", ")
            return (
                "For the following tasks, respond as a person described as: "
                f"\"I {sentence}.\""
            )
        else:
            chosen_low = self.rng.choice(
                low_descriptors, size=2, replace=False
            ).tolist()
            chosen_high = self.rng.choice(
                high_descriptors, size=2, replace=False
            ).tolist()
            low_clause = ", ".join(chosen_low)
            high_clause = ", ".join(chosen_high)
            sentence = (
                f"neither strongly avoids thinking ({low_clause}) "
                f"nor strongly seeks out thinking ({high_clause})"
            )
            return (
                "For the following tasks, respond as a person described as: "
                f"\"I am {sentence}.\""
            )

    def build_crt_prompt(
        self, crt_level: int, numeric_only: bool = False
    ) -> str:
        """
        Construct CRT2-style thinking-style prompt, either numeric-only
        or descriptive.
        """
        if crt_level < 0 or crt_level > 4:
            raise ValueError(
                f"crt_level must be between 0 and 4 (inclusive), "
                f"got {crt_level}."
            )

        level_descriptions = {
            0: "I almost always trust my first impression, answer quickly "
            "without re-checking, and rarely notice when a question might be "
            "tricky.",
            1: "I often go with my first impression and only occasionally "
            "stop to reconsider whether it might be misleading.",
            2: "I sometimes pause to reconsider my first impression before "
            "answering, but I am inconsistent and often stick with the "
            "obvious answer.",
            3: "I usually pause to check whether an obvious answer could be "
            "a trap, and I am willing to change my mind after thinking "
            "things through.",
            4: "I almost always look for hidden assumptions, carefully check "
            "for tricks, and verify my answers with calculations before "
            "responding.",
        }

        numeric_tag = (
            f"This persona has a CRT2 ability level of {crt_level} on a 0–4 "
            "scale."
        )

        if numeric_only:
            return (
                "For the following CRT-style questions, respond as a person "
                f"with the following thinking style: \"{numeric_tag}\""
            )

        sentence = level_descriptions[crt_level]
        return (
            "For the following CRT-style questions, respond as a person "
            f"described as: \"{sentence}\""
        )

    def build_full_persona_prompt(
        self,
        persona: Persona,
        persona_row: pd.Series,
        style: str = "granular_serapio",
        numeric_crt_only: bool = False,
    ) -> str:
        """
        Build the full persona prompt (BFI + NFC + CRT) for a given
        Persona instance.
        """
        mode = self.config.persona_spec_mode

        bfi_desc: Dict[str, str] = {}
        for domain, score in persona.bfi_traits.items():
            lexical = self._bfi_descriptor(domain, score)
            if mode == "numeric_only":
                bfi_desc[domain] = f"{score:.1f}/5"
            elif mode == "both":
                bfi_desc[domain] = f"{lexical} (score: {score:.1f}/5)"
            else:
                bfi_desc[domain] = lexical

        nfc_lexical = self._nfc_descriptor(persona.nfc_score)
        if mode == "numeric_only":
            nfc_desc = f"{persona.nfc_score:.1f}/5"
        elif mode == "both":
            nfc_desc = f"{nfc_lexical} (score: {persona.nfc_score:.1f}/5)"
        else:
            nfc_desc = nfc_lexical

        crt_lexical = self._crt_descriptor(persona.crt2_level)
        if numeric_crt_only or mode == "numeric_only":
            crt_desc = f"CRT level {persona.crt2_level} on a 0-4 scale."
        elif mode == "both":
            crt_desc = f"{crt_lexical} (level: {persona.crt2_level}/4)"
        else:
            crt_desc = crt_lexical

        # Prefer the structured external persona template when available.
        tmpl_block = self.persona_template.get("persona_template", {})
        system_tmpl = tmpl_block.get("system_message")
        placeholder_map = tmpl_block.get("placeholder_mapping", {})

        if isinstance(system_tmpl, str) and isinstance(placeholder_map, dict):
            values: Dict[str, Any] = {}
            for placeholder, source in placeholder_map.items():
                if not isinstance(source, str):
                    values[placeholder] = ""
                    continue
                if source.startswith("derived:"):
                    key = source.split("derived:", 1)[1]
                    values[placeholder] = ""
                    if key == "BFI_Extraversion":
                        values[placeholder] = bfi_desc.get("extraversion", "")
                    elif key == "BFI_Agreeableness":
                        values[placeholder] = bfi_desc.get("agreeableness", "")
                    elif key == "BFI_Conscientiousness":
                        values[placeholder] = bfi_desc.get("conscientiousness", "")
                    elif key == "BFI_Neuroticism":
                        values[placeholder] = bfi_desc.get("neuroticism", "")
                    elif key == "BFI_Openness":
                        values[placeholder] = bfi_desc.get("openness", "")
                    elif key == "NFC_Total":
                        values[placeholder] = nfc_desc
                    elif key == "CRT_Level":
                        values[placeholder] = crt_desc
                else:
                    values[placeholder] = persona_row.get(source, "")
            try:
                return system_tmpl.format(**values)
            except Exception:
                pass

        # Without a usable persona template, concatenate the BFI, NFC and CRT
        # prompt styles.
        if style == "granular_serapio":
            bfi_prompt = self.build_bfi_granular_prompt(persona.bfi_traits)
        elif style == "coarse_numeric":
            bfi_prompt = self.build_bfi_coarse_numeric_prompt(persona.bfi_traits)
        elif style == "coarse_descriptive":
            bfi_prompt = self.build_bfi_coarse_descriptive_prompt(persona.bfi_traits)
        else:
            raise ValueError(f"Unknown persona prompt style: {style}")

        nfc_prompt = self.build_nfc_prompt(persona.nfc_score)
        crt_prompt = self.build_crt_prompt(
            persona.crt2_level, numeric_only=numeric_crt_only
        )
        return " ".join([bfi_prompt, nfc_prompt, crt_prompt])


# ---------------------------------------------------------------------
# YAML-based Test bank construction
# ---------------------------------------------------------------------


def _load_yaml_items_file(rel_filename: str) -> Any:
    """
    Load a YAML file from DATA_DIR and return its raw contents.
    """
    path = os.path.join(DATA_DIR, rel_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required YAML file '{rel_filename}' not found in DATA_DIR '{DATA_DIR}'."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def _coerce_items_list(yaml_data: Any, filename: str) -> List[Dict[str, Any]]:
    """
    Coerce YAML contents into a list of item dictionaries.
    """
    if isinstance(yaml_data, dict):
        if "items" in yaml_data and isinstance(yaml_data["items"], list):
            return yaml_data["items"]
        if "questions" in yaml_data and isinstance(yaml_data["questions"], list):
            return yaml_data["questions"]
        if "tasks" in yaml_data and isinstance(yaml_data["tasks"], list):
            return yaml_data["tasks"]
        raise ValueError(
            f"YAML file '{filename}' must contain an 'items', 'questions', "
            "or 'tasks' list."
        )
    if isinstance(yaml_data, list):
        return yaml_data
    raise ValueError(
        f"YAML file '{filename}' must be a list or contain an 'items' list."
    )


def load_likert_test_from_yaml(
    filename: str,
    default_test_name: str,
    default_trait: Optional[str] = None,
) -> List[TestItem]:
    """
    Load a Likert-scale questionnaire from a YAML file into TestItem
    objects. Validates keyed directions and, for BFI, enforces that
    items map to known Big Five domains.
    """
    yaml_data = _load_yaml_items_file(filename)
    items_raw = _coerce_items_list(yaml_data, filename)
    root_instructions = (
        yaml_data.get("instructions")
        if isinstance(yaml_data, dict)
        else None
    )
    root_response_scale = (
        yaml_data.get("response_scale")
        if isinstance(yaml_data, dict)
        else None
    )
    items: List[TestItem] = []

    for entry in items_raw:
        item_id = str(
            entry.get("id")
            or entry.get("item_id")
            or entry.get("name")
            or entry.get("code")
        )
        if not item_id:
            raise ValueError(
                f"Missing 'id'/'item_id' in entry of '{filename}': {entry}"
            )
        text = entry.get("text") or entry.get("prompt") or entry.get("question")
        if not text:
            raise ValueError(
                f"Missing 'text'/'prompt' in entry for item '{item_id}' "
                f"in '{filename}'."
            )

        test_name = entry.get("test_name") or default_test_name
        scale_name = (
            entry.get("scale_name")
            or entry.get("domain")
            or entry.get("scale")
            or default_trait
        )

        trait = entry.get("trait") or scale_name or default_trait

        if "keyed" in entry:
            keyed = int(entry["keyed"])
        else:
            reverse_flag = bool(
                entry.get("reverse_scored")
                or entry.get("reverse_keyed")
                or entry.get("reversed")
            )
            keyed = -1 if reverse_flag else 1

        if default_test_name.lower() in {"bfi", "nfc"}:
            if keyed not in {-1, 1}:
                raise ValueError(
                    f"Likert item '{item_id}' in '{filename}' must have "
                    f"keyed in {{-1, 1}}, got {keyed}."
                )

        # Enforce Big Five domain mapping for BFI items
        if default_test_name.lower() == "bfi":
            valid_domains = {
                "extraversion",
                "agreeableness",
                "conscientiousness",
                "neuroticism",
                "openness",
            }
            if trait is None:
                raise ValueError(
                    f"BFI item '{item_id}' in '{filename}' is missing a "
                    "'trait'/'domain' field."
                )
            trait_norm = str(trait).strip().lower()
            if trait_norm not in valid_domains:
                raise ValueError(
                    f"BFI item '{item_id}' in '{filename}' has trait/domain "
                    f"'{trait}', expected one of {sorted(valid_domains)}."
                )
            trait = trait_norm
            scale_name = trait_norm

        # For NFC-18 we force a single 'nfc' trait label
        if default_test_name.lower() == "nfc":
            trait = "nfc"
            scale_name = "nfc"

        metadata = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "id",
                "item_id",
                "text",
                "prompt",
                "question",
                "test_name",
                "scale_name",
                "domain",
                "scale",
                "trait",
                "keyed",
                "reverse_scored",
                "reverse_keyed",
                "reversed",
            }
        }
        if root_instructions:
            metadata["source_instructions"] = root_instructions
        if root_response_scale:
            metadata["source_response_scale"] = root_response_scale

        items.append(
            TestItem(
                item_id=item_id,
                test_name=str(test_name),
                scale_name=str(scale_name) if scale_name is not None else None,
                item_text=str(text),
                trait=str(trait) if trait is not None else None,
                keyed=keyed,
                item_type="likert",
                metadata=metadata,
            )
        )

    return items


def load_crt_test_from_yaml(
    filename: str,
    default_test_name: str,
    default_trait: Optional[str] = None,
) -> List[TestItem]:
    """
    Load CRT-style items (numeric response, correct answer) from YAML
    into TestItem objects.
    """
    yaml_data = _load_yaml_items_file(filename)
    items_raw = _coerce_items_list(yaml_data, filename)
    root_instructions = (
        yaml_data.get("instructions")
        if isinstance(yaml_data, dict)
        else None
    )
    items: List[TestItem] = []

    for entry in items_raw:
        item_id = str(
            entry.get("id")
            or entry.get("item_id")
            or entry.get("name")
            or entry.get("code")
        )
        if not item_id:
            raise ValueError(
                f"Missing 'id'/'item_id' in entry of '{filename}': {entry}"
            )
        text = entry.get("text") or entry.get("prompt") or entry.get("question")
        if not text:
            raise ValueError(
                f"Missing 'text'/'prompt' in entry for item '{item_id}' "
                f"in '{filename}'."
            )

        test_name = entry.get("test_name") or default_test_name
        scale_name = (
            entry.get("scale_name")
            or entry.get("domain")
            or entry.get("scale")
            or default_trait
        )
        trait = entry.get("trait") or scale_name or default_trait

        correct_answer = (
            entry.get("correct_answer")
            or entry.get("answer")
            or entry.get("solution")
        )
        if correct_answer is None or str(correct_answer).strip() == "":
            raise ValueError(
                f"Missing 'correct_answer'/'answer' for CRT item '{item_id}' "
                f"in '{filename}'."
            )

        lure_answers = (
            entry.get("lure_answers")
            or entry.get("lures")
            or entry.get("incorrect_answers")
            or entry.get("incorrect_options")
        )
        if lure_answers is not None and not isinstance(lure_answers, list):
            lure_answers = [lure_answers]

        metadata = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "id",
                "item_id",
                "text",
                "prompt",
                "question",
                "test_name",
                "scale_name",
                "domain",
                "scale",
                "trait",
                "correct_answer",
                "answer",
                "solution",
                "lure_answers",
                "lures",
                "incorrect_answers",
                "incorrect_options",
            }
        }
        if root_instructions:
            metadata["source_instructions"] = root_instructions

        items.append(
            TestItem(
                item_id=item_id,
                test_name=str(test_name),
                scale_name=str(scale_name) if scale_name is not None else None,
                item_text=str(text),
                trait=str(trait) if trait is not None else None,
                keyed=0,
                item_type="crt",
                correct_answer=str(correct_answer),
                lure_answers=[str(x) for x in lure_answers] if lure_answers else None,
                metadata=metadata,
            )
        )

    return items


def load_reasoning_test_from_yaml(
    filename: str,
    default_test_name: str,
    default_trait: Optional[str] = None,
) -> List[TestItem]:
    """
    Load distractor reasoning items (CRT-like free responses without
    correctness scoring) from YAML into TestItem objects.
    """
    yaml_data = _load_yaml_items_file(filename)
    items_raw = _coerce_items_list(yaml_data, filename)
    root_instructions = (
        yaml_data.get("instructions")
        if isinstance(yaml_data, dict)
        else None
    )
    items: List[TestItem] = []

    for entry in items_raw:
        item_id = str(
            entry.get("id")
            or entry.get("item_id")
            or entry.get("name")
            or entry.get("code")
        )
        if not item_id:
            raise ValueError(
                f"Missing 'id'/'item_id' in entry of '{filename}': {entry}"
            )
        text = entry.get("text") or entry.get("prompt") or entry.get("question")
        if not text:
            raise ValueError(
                f"Missing 'text'/'prompt' in entry for item '{item_id}' "
                f"in '{filename}'."
            )
        test_name = entry.get("test_name") or default_test_name
        scale_name = (
            entry.get("scale_name")
            or entry.get("domain")
            or entry.get("scale")
            or default_trait
            or "reasoning"
        )
        trait = entry.get("trait") or scale_name or default_trait or "reasoning"
        metadata = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "id",
                "item_id",
                "name",
                "code",
                "text",
                "prompt",
                "question",
                "test_name",
                "scale_name",
                "domain",
                "scale",
                "trait",
            }
        }
        if root_instructions:
            metadata["source_instructions"] = root_instructions
        items.append(
            TestItem(
                item_id=item_id,
                test_name=str(test_name),
                scale_name=str(scale_name) if scale_name is not None else None,
                item_text=str(text),
                trait=str(trait) if trait is not None else None,
                keyed=0,
                item_type="reasoning",
                metadata=metadata,
            )
        )

    return items


def load_writing_tasks_from_yaml(
    filename: str,
    default_test_name: str = "Writing",
) -> List[TestItem]:
    """
    Load open-ended writing tasks from YAML into TestItem objects.
    """
    yaml_data = _load_yaml_items_file(filename)
    items_raw = _coerce_items_list(yaml_data, filename)
    root_instructions = (
        yaml_data.get("instructions")
        if isinstance(yaml_data, dict)
        else None
    )
    items: List[TestItem] = []

    for entry in items_raw:
        item_id = str(
            entry.get("id")
            or entry.get("item_id")
            or entry.get("name")
            or entry.get("code")
        )
        if not item_id:
            raise ValueError(
                f"Missing 'id'/'item_id' in entry of '{filename}': {entry}"
            )
        text = entry.get("text") or entry.get("prompt") or entry.get("instruction")
        if not text:
            raise ValueError(
                f"Missing 'text'/'prompt'/'instruction' for writing task "
                f"'{item_id}' in '{filename}'."
            )

        test_name = entry.get("test_name") or default_test_name
        scale_name = entry.get("scale_name") or entry.get("domain") or "brt"
        trait = entry.get("trait") or scale_name

        metadata = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "id",
                "item_id",
                "text",
                "prompt",
                "instruction",
                "test_name",
                "scale_name",
                "domain",
                "trait",
            }
        }
        if root_instructions:
            metadata["source_instructions"] = root_instructions

        items.append(
            TestItem(
                item_id=item_id,
                test_name=str(test_name),
                scale_name=str(scale_name) if scale_name is not None else None,
                item_text=str(text),
                trait=str(trait) if trait is not None else None,
                keyed=0,
                item_type="open",
                metadata=metadata,
            )
        )

    return items


def build_test_bank(included_blocks: Optional[List[str]] = None) -> TestBank:
    """
    Construct the psychometric test bank and writing tasks from YAML
    specifications in DATA_DIR.
    """
    required_yaml_files = [
        "bfi10_items.yaml",
        "nfc18_items.yaml",
        "crt2_items.yaml",
        "bcrt_items.yaml",
        "dcrt_items.yaml",
        "writing_tasks.yaml",
    ]
    missing_files = [
        fname
        for fname in required_yaml_files
        if not os.path.exists(os.path.join(DATA_DIR, fname))
    ]
    if missing_files:
        raise FileNotFoundError(
            "The following required YAML files are missing in "
            f"DATA_DIR '{DATA_DIR}': {missing_files}. "
            "Please ensure all psychometric and writing task specifications "
            "are present."
        )

    items: List[TestItem] = []
    # BFI-10 from YAML
    items.extend(load_likert_test_from_yaml("bfi10_items.yaml", default_test_name="BFI"))
    # NFC-18 from YAML
    items.extend(
        load_likert_test_from_yaml(
            "nfc18_items.yaml", default_test_name="NFC", default_trait="nfc"
        )
    )
    # CRT2 items
    items.extend(
        load_crt_test_from_yaml(
            "crt2_items.yaml", default_test_name="CRT2", default_trait="crt2"
        )
    )
    # bCRT items
    items.extend(
        load_crt_test_from_yaml(
            "bcrt_items.yaml", default_test_name="bCRT", default_trait="crt2"
        )
    )
    # dCRT distractor reasoning items (no correctness scoring)
    items.extend(
        load_reasoning_test_from_yaml(
            "dcrt_items.yaml", default_test_name="dCRT", default_trait="crt2"
        )
    )
    # Writing tasks
    items.extend(load_writing_tasks_from_yaml("writing_tasks.yaml", "Writing"))

    if included_blocks:
        allowed_test_names = set(expand_included_test_names(included_blocks))
        items = [item for item in items if item.test_name in allowed_test_names]

    return TestBank(items=items)


# ---------------------------------------------------------------------
# Pseudo LLM simulator
# ---------------------------------------------------------------------


class PseudoLLM:
    """
    Simple, fully deterministic pseudo-LLM.
    """

    def __init__(
        self,
        likert_noise: float = 0.5,
        crt_slip: float = 0.2,
        random_seed: int = GLOBAL_RANDOM_SEED,
    ):
        self.likert_noise = likert_noise
        self.crt_slip = crt_slip
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def _latent_from_trait(trait_value: float) -> float:
        """
        Latent score centered around the 1–5 Likert midpoint (3).
        """
        return trait_value - 3.0

    def answer_likert(self, trait_value: float, keyed: int) -> int:
        """
        Generate a 1–5 Likert response aligned with the instrument's
        direction. Reverse-scoring is applied explicitly using
        response = 6 - response when keyed == -1.
        """
        latent = self._latent_from_trait(trait_value)
        noisy = latent + self.rng.normal(0.0, self.likert_noise)
        mapped = 3.0 + noisy
        response = int(round(mapped))
        response = max(1, min(5, response))

        # Explicit reverse scoring when keyed == -1
        if keyed == -1:
            response = 6 - response

        return response

    def answer_crt(self, crt_level: int, item: TestItem) -> Tuple[str, int]:
        if item.correct_answer is None:
            raise ValueError("CRT item must have a correct_answer defined.")

        p_correct_base = 0.1
        p_correct = min(
            1.0 - self.crt_slip,
            p_correct_base + (crt_level / 4.0) * (0.9 - self.crt_slip),
        )
        if self.rng.random() < p_correct:
            return item.correct_answer, 1

        if item.lure_answers:
            answer = str(
                self.rng.choice(
                    item.lure_answers, size=1, replace=True
                )[0]
            )
        else:
            try:
                val = float(item.correct_answer)
                val += self.rng.integers(-3, 4)
                answer = str(val)
            except ValueError:
                answer = "I am not sure."
        return answer, 0

    def answer_reasoning(self, crt_level: int, item: TestItem) -> str:
        _ = item
        if crt_level <= 1:
            return "I'm not sure, but my quick answer is 3."
        if crt_level == 2:
            return "I would estimate the answer as 4."
        if crt_level == 3:
            return "After checking quickly, my answer is 4."
        return "After reasoning it through carefully, my final answer is 4."

    def generate_writing(
        self,
        persona_prompt: str,
        traits: Dict[str, float],
        nfc_score: float,
        crt_level: int,
        item: TestItem,
    ) -> str:
        ext = traits.get("extraversion", 3.0)
        nfc = nfc_score
        crt = float(crt_level)

        social_words = [
            "friends",
            "conversation",
            "team",
            "together",
            "crowd",
            "talkative",
        ]
        analytic_words = [
            "analyze",
            "because",
            "therefore",
            "logic",
            "structure",
            "methodical",
        ]
        insight_words = [
            "realized",
            "understood",
            "noticed",
            "reflected",
            "insight",
        ]
        affect_words = [
            "happy",
            "anxious",
            "relieved",
            "frustrated",
            "excited",
        ]

        base_sentences = [
            "I remember this situation very clearly.",
            "At first, I reacted in a very intuitive way.",
            "As I thought more carefully, my perspective began to change.",
            "Step by step, I tried to understand what was really going on.",
            "Eventually I decided on a course of action.",
            "Looking back, the experience taught me a lot about myself.",
        ]

        num_sentences = 6 + int(round((nfc - 3.0) * 2)) + int(
            round((crt - 2.0))
        )
        num_sentences = max(4, min(10, num_sentences))

        sentences: List[str] = []
        for i in range(num_sentences):
            core = base_sentences[i % len(base_sentences)]

            extra_parts: List[str] = []

            if ext > 3.0 and self.rng.random() < 0.7:
                word = self.rng.choice(social_words)
                extra_parts.append(
                    f"I discussed it with my {word} to understand it better."
                )

            if nfc > 3.0 and self.rng.random() < 0.8:
                word = self.rng.choice(analytic_words)
                extra_parts.append(
                    f"I tried to {word} the problem before acting."
                )

            if crt >= 3 and self.rng.random() < 0.8:
                word = self.rng.choice(insight_words)
                extra_parts.append(
                    f"After some time, I {word} that my first impression "
                    "might be misleading."
                )

            if self.rng.random() < 0.6:
                word = self.rng.choice(affect_words)
                extra_parts.append(f"I felt quite {word}.")

            sentence = " ".join([core] + extra_parts)
            sentences.append(sentence)

        return " ".join(sentences)


# ---------------------------------------------------------------------
# Memory and Planning Agents for prompt construction
# ---------------------------------------------------------------------


class MemoryAgent:
    """
    Minimal Memory Agent abstraction to provide user and item context
    for LLM prompts.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def get_user_context(self, persona_row: pd.Series) -> str:
        if not self.config.include_user_context:
            return ""
        payload: Dict[str, Any] = {}
        persona_id = persona_row.get("persona_id", None)
        if persona_id is not None:
            payload["persona_id"] = persona_id
        return json.dumps(payload, ensure_ascii=True)

    @staticmethod
    def get_item_context(items: List[TestItem]) -> str:
        lines: List[str] = []
        for i, item in enumerate(items, start=1):
            lines.append(f"{i}. [{item.item_id}] {item.item_text}")
        return "\n".join(lines)


class PlanningAgent:
    """
    Minimal Planning Agent abstraction that decomposes the task into
    steps for the LLM.
    """

    def make_plan(self, admin_mode: str, items: List[TestItem]) -> str:
        _ = admin_mode
        _ = items
        return ""


# ---------------------------------------------------------------------
# Real LLM client using OpenAI Responses API
# ---------------------------------------------------------------------


def _load_response_schemas() -> Dict[str, Any]:
    path = os.path.join(PROMPT_CONFIG_DIR, "response_schemas.yaml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data.get("response_schemas", {})
    except Exception:
        pass
    return {}


def _schema_key_for_items(items: List[TestItem]) -> Optional[str]:
    kinds = {item.item_type for item in items}
    if kinds == {"likert"}:
        return "likert"
    if kinds == {"crt"}:
        return "crt"
    if kinds == {"open"}:
        return "brt"
    return None


def _schema_keys_for_items(items: List[TestItem]) -> List[str]:
    keys: List[str] = []
    kinds = {item.item_type for item in items}
    if "likert" in kinds:
        keys.append("likert")
    if "crt" in kinds or "reasoning" in kinds:
        keys.append("crt")
    if "open" in kinds:
        keys.append("brt")
    return keys


def _render_response_scale(response_scale: Any) -> List[str]:
    if not isinstance(response_scale, dict):
        return []
    min_v = response_scale.get("min")
    max_v = response_scale.get("max")
    labels = response_scale.get("labels")
    lines: List[str] = []
    if isinstance(min_v, int) and isinstance(max_v, int):
        lines.append(f"Respond with an integer between {min_v} and {max_v}.")
    if isinstance(labels, dict):
        # Keep labels in ascending numeric key order when possible.
        parsed: List[Tuple[int, str]] = []
        for k, v in labels.items():
            try:
                parsed.append((int(k), str(v)))
            except Exception:
                continue
        for k, v in sorted(parsed, key=lambda x: x[0]):
            lines.append(f"{k}: {v}")
    return lines


def build_user_prompt(
    items: List[TestItem],
    user_context: str,
    plan: str,
) -> Tuple[str, Dict[str, List[TestItem]]]:
    response_schemas = _load_response_schemas()
    lines: List[str] = []
    if user_context:
        lines.extend(["Additional context:", user_context, ""])

    # Group by test while preserving appearance order in this block.
    grouped: Dict[str, List[TestItem]] = {}
    order: List[str] = []
    for item in items:
        if item.test_name not in grouped:
            grouped[item.test_name] = []
            order.append(item.test_name)
        grouped[item.test_name].append(item)

    # Merge CRT2+bCRT+dCRT into one survey block.
    reasoning_names = {"CRT2", "bCRT", "dCRT"}
    surveys: List[List[TestItem]] = []
    reasoning_block: List[TestItem] = []
    for test_name in order:
        test_items = grouped[test_name]
        if test_name in reasoning_names:
            reasoning_block.extend(test_items)
        else:
            surveys.append(test_items)
    if reasoning_block:
        surveys.append(reasoning_block)

    survey_lookup: Dict[str, List[TestItem]] = {}
    for survey_idx, test_items in enumerate(surveys, start=1):
        survey_name = f"Survey {survey_idx}"
        survey_lookup[survey_name.lower()] = test_items
        first_item = test_items[0]
        lines.append(survey_name)
        lines.append("")

        instructions = first_item.metadata.get("source_instructions")
        if any(it.item_type in {"crt", "reasoning"} for it in test_items):
            instructions = "Read the following questions carefully and provide your answer."
        if instructions:
            lines.append(str(instructions).strip())
            lines.append("")

        scale_lines = _render_response_scale(first_item.metadata.get("source_response_scale"))
        if scale_lines:
            lines.extend(scale_lines)
            lines.append("")

        lines.append("Items:")
        for i, item in enumerate(test_items, start=1):
            # Hide IDs for CRT/reasoning/writing sections
            if item.item_type in {"crt", "reasoning", "open"}:
                lines.append(f"{i}. {item.item_text}")
            else:
                lines.append(f"{i}. ({item.item_id}) {item.item_text}")
        lines.append("")

        schema_keys = _schema_keys_for_items(test_items)
        if schema_keys:
            for schema_key in schema_keys:
                schema_block = response_schemas.get(schema_key, {})
                if not schema_block:
                    continue
                instruction = schema_block.get("instruction")
                json_schema = schema_block.get("json_schema")
                if instruction:
                    lines.append(str(instruction).strip())
                if json_schema is not None:
                    lines.append("JSON Schema object (machine-readable):")
                    lines.append(json.dumps(json_schema))
                lines.append("")

    if plan:
        lines.extend(["Task note:", plan, ""])
    return "\n".join(lines), survey_lookup


def parse_llm_answers(
    raw_text: str,
    items: List[TestItem],
    survey_lookup: Optional[Dict[str, List[TestItem]]] = None,
) -> Dict[str, str]:
    raw_text = raw_text.strip()
    json_candidates: List[str] = [raw_text]
    if "[" in raw_text and "]" in raw_text:
        start = raw_text.find("[")
        end = raw_text.rfind("]") + 1
        if end > start:
            json_candidates.append(raw_text[start:end])
    if "{" in raw_text and "}" in raw_text:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if end > start:
            json_candidates.append(raw_text[start:end])

    for json_text in json_candidates:
        try:
            data = json.loads(json_text)
            mapping: Dict[str, str] = {}
            entries: List[Any]
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict):
                entries = [data]
            else:
                entries = []
            if entries:
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    iid = entry.get("item_id") or entry.get("id")
                    if iid is None:
                        item_number = entry.get("item_number")
                        survey_name = (
                            entry.get("survey")
                            or entry.get("survey_id")
                            or entry.get("survey_name")
                        )
                        if (
                            isinstance(item_number, int)
                            and isinstance(survey_name, str)
                            and survey_lookup
                        ):
                            key = survey_name.strip().lower()
                            survey_items = survey_lookup.get(key)
                            if survey_items and 1 <= item_number <= len(survey_items):
                                iid = survey_items[item_number - 1].item_id
                        elif isinstance(item_number, int) and 1 <= item_number <= len(items):
                            iid = items[item_number - 1].item_id
                    resp = (
                        entry.get("response")
                        or entry.get("answer")
                        or entry.get("response_text")
                        or entry.get("text")
                        or ""
                    )
                    if iid is not None:
                        mapping[str(iid)] = str(resp)
            if mapping:
                return mapping
        except Exception:
            continue

    if len(items) == 1:
        return {items[0].item_id: raw_text}
    # A multi-item block with no extractable JSON yields no answers. Every
    # multi-item prompt asks for a JSON array/object (see the json_schema
    # block in build_user_prompt), so a reply without one (for example a
    # reasoning trace cut off at the output-token limit before the answer
    # JSON) leaves every item missing. The raw text is not copied onto each
    # item, because item-level parsers (e.g. the 1-5 digit regex in
    # _parse_likert_from_text) would then read the same spurious value for
    # every item. Missing items go through the caller's retry loop and are
    # counted by RUN_COUNTERS.record_empty_or_unparseable.
    return {}


class OpenAILLMClient:
    """
    Wrapper around OpenAI's API family to answer psychometric items and
    writing tasks given constructed prompts. Supports two API styles:
    `api_style="responses"` uses the native OpenAI Responses API
    with `previous_response_id` chaining (used for `gpt-5.6-sol`);
    `api_style="chat_completions"` uses OpenAI-compatible chat completions
    with a maintained message history (used for DeepSeek and Gemini's
    OpenAI-compatibility endpoint, neither of which implements the
    Responses API). `base_url`/`key_env` route to the correct provider;
    `extra_params`/`extra_body` carry each provider's reasoning-control
    setting: OpenAI `reasoning`, DeepSeek `thinking` (via extra_body),
    Gemini `thinking_level` (via extra_body's `google.thinking_config`).
    """

    def __init__(
        self,
        model: str = "gpt-5",
        max_output_tokens: int = 4000,
        multi_turn_conversation: bool = True,
        api_style: str = "responses",
        base_url: Optional[str] = None,
        key_env: str = "OPENAI_API_KEY",
        extra_params: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
    ):
        if api_style not in {"responses", "chat_completions"}:
            raise ValueError(
                f"api_style must be 'responses' or 'chat_completions', got {api_style!r}."
            )
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.multi_turn_conversation = multi_turn_conversation
        self.api_style = api_style
        self.base_url = base_url
        self.key_env = key_env
        self.extra_params = extra_params or {}
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self._previous_response_ids: Dict[str, str] = {}
        self._message_histories: Dict[str, List[Dict[str, str]]] = {}
        # Populated after every call; callers copy the truncation fields
        # into api_calls.csv.
        self.last_call_meta: Dict[str, Any] = {}

    def answer_block(
        self,
        system_prompt: str,
        user_prompt: str,
        items: List[TestItem],
        survey_lookup: Optional[Dict[str, List[TestItem]]] = None,
        persona_key: Optional[str] = None,
    ) -> Dict[str, str]:
        if self.api_style == "responses":
            previous_response_id: Optional[str] = None
            if self.multi_turn_conversation and persona_key:
                previous_response_id = self._previous_response_ids.get(persona_key)

            result = call_gpt5_with_responses_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                max_output_tokens=self.max_output_tokens,
                previous_response_id=previous_response_id,
                base_url=self.base_url,
                key_env=self.key_env,
                extra_params=self.extra_params,
                max_retries=self.max_retries,
            )
            if self.multi_turn_conversation and persona_key and result.response_id:
                self._previous_response_ids[persona_key] = result.response_id
        else:
            prior_messages: Optional[List[Dict[str, str]]] = None
            if self.multi_turn_conversation and persona_key:
                prior_messages = self._message_histories.get(persona_key, [])

            result = call_openai_chat_completions_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                max_tokens=self.max_output_tokens,
                prior_messages=prior_messages,
                base_url=self.base_url,
                key_env=self.key_env,
                extra_params=self.extra_params,
                extra_body=self.extra_body,
                max_retries=self.max_retries,
            )
            if self.multi_turn_conversation and persona_key:
                history = self._message_histories.setdefault(persona_key, [])
                history.append({"role": "user", "content": user_prompt})
                history.append({"role": "assistant", "content": result.text})

        self.last_call_meta = {
            "truncated": result.truncated,
            "finish_reason": result.finish_reason,
            "response_id": result.response_id,
        }
        return parse_llm_answers(result.text, items, survey_lookup=survey_lookup)


class AnthropicLLMClient:
    """
    Wrapper around Anthropic Messages API to answer psychometric items
    and writing tasks given constructed prompts.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8000,
        multi_turn_conversation: bool = True,
        temperature: Optional[float] = None,
        thinking: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.multi_turn_conversation = multi_turn_conversation
        self.temperature = temperature
        self.thinking = thinking
        self.max_retries = max_retries
        self._message_histories: Dict[str, List[Dict[str, str]]] = {}

    def answer_block(
        self,
        system_prompt: str,
        user_prompt: str,
        items: List[TestItem],
        survey_lookup: Optional[Dict[str, List[TestItem]]] = None,
        persona_key: Optional[str] = None,
    ) -> Dict[str, str]:
        prior_messages: Optional[List[Dict[str, str]]] = None
        if self.multi_turn_conversation and persona_key:
            prior_messages = self._message_histories.get(persona_key, [])
        raw = call_claude_messages_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            prior_messages=prior_messages,
            temperature=self.temperature,
            thinking=self.thinking,
            max_retries=self.max_retries,
        )
        if self.multi_turn_conversation and persona_key:
            history = self._message_histories.setdefault(persona_key, [])
            history.append({"role": "user", "content": user_prompt})
            history.append({"role": "assistant", "content": raw})
        return parse_llm_answers(raw, items, survey_lookup=survey_lookup)


# ---------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------


class PersonaSimulator:
    """
    Core simulator that rolls out test administrations for personas using
    either the PseudoLLM or a real LLM client.
    """

    def __init__(self, test_bank: TestBank, config: SimulationConfig):
        self.test_bank = test_bank
        self.config = config

    @staticmethod
    def _preferred_test_order(names: List[str]) -> List[str]:
        preferred = ["Writing", "CRT2", "bCRT", "dCRT", "NFC", "BFI"]
        rank = {name: idx for idx, name in enumerate(preferred)}
        return sorted(names, key=lambda n: (rank.get(n, 999), n))

    def _get_tests_order(
        self,
        items_by_test: Dict[str, List[TestItem]],
        persona_rng: np.random.Generator,
    ) -> List[str]:
        tests_order = self._preferred_test_order(list(items_by_test.keys()))
        if self.config.shuffle_test_order:
            persona_rng.shuffle(tests_order)
        return tests_order

    def _build_blocks(
        self,
        items_by_test: Dict[str, List[TestItem]],
        persona_rng: np.random.Generator,
    ) -> List[List[TestItem]]:
        if self.config.tests_included_explicitly_set:
            return self._build_blocks_for_explicit_selection(items_by_test, persona_rng)
        return self._build_blocks_default(items_by_test, persona_rng)

    def _build_blocks_default(
        self,
        items_by_test: Dict[str, List[TestItem]],
        persona_rng: np.random.Generator,
    ) -> List[List[TestItem]]:
        tests_order = self._get_tests_order(items_by_test, persona_rng)
        reasoning_names = {"CRT2", "bCRT", "dCRT"}
        blocks: List[List[TestItem]] = []

        if self.config.admin_mode == "per_item":
            reasoning_items: List[TestItem] = []
            for test_name in tests_order:
                if test_name in reasoning_names:
                    reasoning_items.extend(list(items_by_test[test_name]))
            if reasoning_items and (
                self.config.randomize_reasoning_block
                or self.config.shuffle_items_within_tests
            ):
                persona_rng.shuffle(reasoning_items)
            for item in reasoning_items:
                blocks.append([item])

            for test_name in tests_order:
                if test_name in reasoning_names:
                    continue
                test_items = list(items_by_test[test_name])
                if self.config.shuffle_items_within_tests:
                    persona_rng.shuffle(test_items)
                for item in test_items:
                    blocks.append([item])
            return blocks

        if self.config.admin_mode == "per_test":
            reasoning_items: List[TestItem] = []
            for test_name in tests_order:
                test_items = list(items_by_test[test_name])
                if test_name in reasoning_names:
                    reasoning_items.extend(test_items)
                else:
                    if self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                    blocks.append(test_items)
            if reasoning_items and (
                self.config.randomize_reasoning_block
                or self.config.shuffle_items_within_tests
            ):
                persona_rng.shuffle(reasoning_items)
            if reasoning_items:
                blocks.insert(0, reasoning_items)
            return blocks

        all_items: List[TestItem] = []
        reasoning_items = []
        for test_name in tests_order:
            test_items = list(items_by_test[test_name])
            if test_name in reasoning_names:
                reasoning_items.extend(test_items)
            else:
                if self.config.shuffle_items_within_tests:
                    persona_rng.shuffle(test_items)
                all_items.extend(test_items)
        if reasoning_items and (
            self.config.randomize_reasoning_block
            or self.config.shuffle_items_within_tests
        ):
            persona_rng.shuffle(reasoning_items)
        blocks.append(reasoning_items + all_items)
        return blocks

    def _build_blocks_for_explicit_selection(
        self,
        items_by_test: Dict[str, List[TestItem]],
        persona_rng: np.random.Generator,
    ) -> List[List[TestItem]]:
        section_items: List[Tuple[str, List[TestItem]]] = []

        writing_items = list(items_by_test.get("Writing", []))
        if writing_items:
            section_items.append(("writing", writing_items))

        reasoning_items: List[TestItem] = []
        for test_name in BLOCK_TO_TEST_NAMES["reasoning"]:
            reasoning_items.extend(list(items_by_test.get(test_name, [])))
        if reasoning_items:
            section_items.append(("reasoning", reasoning_items))

        nfc_items = list(items_by_test.get("NFC", []))
        if nfc_items:
            section_items.append(("nfc", nfc_items))

        bfi_items = list(items_by_test.get("BFI", []))
        if bfi_items:
            section_items.append(("bfi", bfi_items))

        if self.config.shuffle_test_order:
            persona_rng.shuffle(section_items)

        blocks: List[List[TestItem]] = []
        if self.config.admin_mode == "per_item":
            for section_name, test_items in section_items:
                if section_name == "reasoning":
                    if self.config.randomize_reasoning_block or self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                elif self.config.shuffle_items_within_tests:
                    persona_rng.shuffle(test_items)
                for item in test_items:
                    blocks.append([item])
            return blocks

        if self.config.admin_mode == "per_test":
            for section_name, test_items in section_items:
                if section_name == "reasoning":
                    if self.config.randomize_reasoning_block or self.config.shuffle_items_within_tests:
                        persona_rng.shuffle(test_items)
                elif self.config.shuffle_items_within_tests:
                    persona_rng.shuffle(test_items)
                blocks.append(test_items)
            return blocks

        all_items: List[TestItem] = []
        for section_name, test_items in section_items:
            if section_name == "reasoning":
                if self.config.randomize_reasoning_block or self.config.shuffle_items_within_tests:
                    persona_rng.shuffle(test_items)
            elif self.config.shuffle_items_within_tests:
                persona_rng.shuffle(test_items)
            all_items.extend(test_items)
        blocks.append(all_items)
        return blocks

    @staticmethod
    def _parse_likert_from_text(text: str) -> int:
        match = re.search(r"\b([1-5])\b", text)
        if match:
            try:
                val = int(match.group(1))
                if 1 <= val <= 5:
                    return val
            except ValueError:
                pass
        return 3

    @staticmethod
    def _score_crt_from_text(text: str, item: TestItem) -> bool:
        if item.correct_answer is None:
            return False
        correct_str = item.correct_answer.strip()
        if correct_str and correct_str in text:
            return True
        try:
            nums = re.findall(r"-?\d+\.?\d*", text)
            if nums:
                ans_val = float(nums[0])
                corr_val = float(correct_str)
                if abs(ans_val - corr_val) < 1e-6:
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _resolve_scale_and_trait(item: TestItem) -> Tuple[Optional[str], Optional[str]]:
        """
        Ensure scale/trait are always populated in outputs, even if source
        YAML omitted them.
        """
        default_scale = item.scale_name or item.trait
        if default_scale is None:
            if item.item_type == "open":
                default_scale = "brt"
            elif item.item_type in {"crt", "reasoning"}:
                default_scale = "crt2"
            elif item.test_name:
                default_scale = item.test_name.lower()
        default_trait = item.trait or default_scale
        return default_scale, default_trait

    @staticmethod
    def _missing_item_ids(items: List[TestItem], answers: Dict[str, str]) -> List[str]:
        missing: List[str] = []
        for item in items:
            text = str(answers.get(item.item_id, "")).strip()
            if not text:
                missing.append(item.item_id)
        return missing

    @staticmethod
    def _render_progress(prefix: str, current: int, total: int) -> str:
        total = max(1, total)
        width = 24
        filled = int((current / total) * width)
        bar = "#" * filled + "-" * (width - filled)
        return f"{prefix} [{bar}] {current}/{total}"

    @staticmethod
    def _render_parallel_status(
        done: int,
        total: int,
        in_flight: int,
        workers: int,
        elapsed_s: float,
        failed: int = 0,
    ) -> str:
        remaining = max(0, total - done)
        total = max(1, total)
        width = 24
        filled = int((done / total) * width)
        bar = "#" * filled + "-" * (width - filled)
        eta = "ETA=calculating"
        if done > 0:
            rate = done / max(elapsed_s, 0.001)
            eta_s = remaining / rate if rate > 0 else 0.0
            eta = f"ETA={eta_s / 60:.1f}m"
        succeeded = max(0, done - failed)
        return (
            f"Personas [{bar}] {done}/{total} "
            f"active={in_flight} remaining={remaining} workers={workers} "
            f"succeeded={succeeded} failed={failed} elapsed={elapsed_s / 60:.1f}m {eta}"
        )

    def rollout(
        self,
        personas_df: pd.DataFrame,
        params: Dict[str, Any],
        persona_prompts: pd.Series,
    ) -> SimulationResult:
        """
        Roll out a full set of test administrations for the given personas
        using either a pseudo-LLM or the OpenAI-backed LLM.
        """
        likert_noise = float(params.get("likert_noise", 0.5))
        crt_slip = float(params.get("crt_slip", 0.2))
        random_seed = int(params.get("random_seed", self.config.random_seed))

        api_call_rows: List[Dict[str, Any]] = []

        if self.config.use_openai:
            result = self._rollout_with_openai(
                personas_df=personas_df,
                persona_prompts=persona_prompts,
                random_seed=random_seed,
                api_call_rows=api_call_rows,
            )
            return result
        if self.config.use_anthropic:
            result = self._rollout_with_anthropic(
                personas_df=personas_df,
                persona_prompts=persona_prompts,
                random_seed=random_seed,
                api_call_rows=api_call_rows,
            )
            return result
        if self.config.use_pseudo or not (
            self.config.use_openai or self.config.use_anthropic
        ):
            pass
        else:
            raise RuntimeError(
                "Choose a single LLM provider: --use-pseudo, --use-openai, "
                "or --use-anthropic."
            )

        llm = PseudoLLM(
            likert_noise=likert_noise,
            crt_slip=crt_slip,
            random_seed=random_seed,
        )
        items_by_test = self.test_bank.items_by_test()

        memory_agent = MemoryAgent(self.config)
        planning_agent = PlanningAgent()
        persona_rows: List[Dict[str, Any]] = []
        item_rows: List[Dict[str, Any]] = []
        total_personas = len(personas_df)
        run_start_ts = time.time()

        for persona_idx, (idx, row) in enumerate(personas_df.iterrows()):
            persona_obj = Persona.from_series(row, self.config)
            persona_id = persona_obj.persona_id if persona_obj.persona_id is not None else idx
            bfi_traits = persona_obj.bfi_traits
            nfc_score = persona_obj.nfc_score
            crt_level = persona_obj.crt2_level
            prompt = persona_prompts.loc[idx]
            user_context = memory_agent.get_user_context(row)

            bfi_scores: Dict[str, List[int]] = {
                domain: [] for domain in self.config.bfi_columns
            }
            nfc_items: List[int] = []
            crt_corrects: List[int] = []

            writing_texts: Dict[str, str] = {}
            writing_metrics: Dict[str, Dict[str, float]] = {}

            persona_rng = np.random.default_rng(random_seed + persona_idx)

            blocks = self._build_blocks(items_by_test, persona_rng)

            elapsed_s = time.time() - run_start_ts
            print(
                f"{self._render_progress('Personas', persona_idx + 1, total_personas)} "
                f"(persona_id={persona_id}, elapsed={elapsed_s:.1f}s)",
                flush=True,
            )
            for block_index, block_items in enumerate(blocks):
                block_tests = ",".join(sorted({it.test_name for it in block_items}))
                print(
                    f"  {self._render_progress('Blocks', block_index + 1, len(blocks))} "
                    f"tests={block_tests} items={len(block_items)}",
                    flush=True,
                )
                plan = planning_agent.make_plan(self.config.admin_mode, block_items)
                system_prompt = prompt
                user_prompt, survey_lookup = build_user_prompt(
                    items=block_items,
                    user_context=user_context,
                    plan=plan,
                )

                api_call_rows.append(
                    {
                        "persona_id": persona_id,
                        "block_index": block_index,
                        "admin_mode": self.config.admin_mode,
                        "engine": "pseudo",
                        "test_names": ",".join(
                            sorted({it.test_name for it in block_items})
                        ),
                        "item_ids": [it.item_id for it in block_items],
                        "survey_item_ids": {
                            s: [it.item_id for it in its]
                            for s, its in survey_lookup.items()
                        },
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "prompt": "\n\n".join([system_prompt, user_prompt]),
                    }
                )

                for presented_index, item in enumerate(block_items, start=1):
                    self._simulate_item(
                        llm,
                        persona_id,
                        row,
                        bfi_traits,
                        nfc_score,
                        crt_level,
                        prompt,
                        item,
                        bfi_scores,
                        nfc_items,
                        crt_corrects,
                        item_rows,
                        writing_texts,
                        writing_metrics,
                        block_index=block_index,
                        presented_index=presented_index,
                    )

            persona_record: Dict[str, Any] = row.to_dict()
            persona_record["persona_id"] = persona_id

            for domain, scores in bfi_scores.items():
                if scores:
                    persona_record[f"sim_bfi_{domain}"] = float(
                        np.mean(scores)
                    )
                else:
                    persona_record[f"sim_bfi_{domain}"] = float("nan")

            persona_record["sim_nfc_score"] = (
                float(np.mean(nfc_items)) if nfc_items else float("nan")
            )
            persona_record["sim_crt2_score"] = (
                float(np.sum(crt_corrects)) if crt_corrects else float("nan")
            )

            for wt_id, text in writing_texts.items():
                persona_record[f"writing_{wt_id}_text"] = text
                metrics_ = writing_metrics.get(wt_id, {})
                for m_name, m_val in metrics_.items():
                    persona_record[f"writing_{wt_id}_{m_name}"] = m_val

            persona_rows.append(persona_record)

        per_persona_df = pd.DataFrame(persona_rows)
        per_item_df = pd.DataFrame(item_rows)

        api_calls_df = pd.DataFrame(api_call_rows) if api_call_rows else None

        return SimulationResult(
            per_persona=per_persona_df,
            per_item=per_item_df,
            config=self.config,
            api_calls=api_calls_df,
        )

    def _rollout_with_openai(
        self,
        personas_df: pd.DataFrame,
        persona_prompts: pd.Series,
        random_seed: int,
        api_call_rows: List[Dict[str, Any]],
    ) -> SimulationResult:
        """
        Rollout using the real OpenAI LLM through the Responses API.
        """
        items_by_test = self.test_bank.items_by_test()
        memory_agent = MemoryAgent(self.config)
        planning_agent = PlanningAgent()
        persona_entries = list(personas_df.iterrows())
        total_personas = len(persona_entries)
        workers = max(1, min(self.config.max_parallel_personas, total_personas))
        run_start_ts = time.time()

        def _build_blocks(persona_rng: np.random.Generator) -> List[List[TestItem]]:
            return self._build_blocks(items_by_test, persona_rng)

        def _process_persona(
            persona_idx: int,
            idx: Any,
            row: pd.Series,
        ) -> Tuple[
            int,
            Dict[str, Any],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
        ]:
            llm_client = OpenAILLMClient(
                model=self.config.openai_model,
                max_output_tokens=self.config.openai_max_output_tokens,
                multi_turn_conversation=self.config.multi_turn_conversation,
                api_style=self.config.openai_api_style,
                base_url=self.config.openai_base_url,
                key_env=self.config.openai_key_env,
                extra_params=self.config.openai_extra_params,
                extra_body=self.config.openai_extra_body,
                max_retries=self.config.max_api_retries,
            )
            persona_obj = Persona.from_series(row, self.config)
            persona_id = persona_obj.persona_id if persona_obj.persona_id is not None else idx
            persona_key = str(persona_id)
            persona_prompt = persona_prompts.loc[idx]
            user_context = memory_agent.get_user_context(row)

            bfi_scores: Dict[str, List[int]] = {
                domain: [] for domain in self.config.bfi_columns
            }
            nfc_items: List[int] = []
            crt_corrects: List[int] = []
            writing_texts: Dict[str, str] = {}
            writing_metrics: Dict[str, Dict[str, float]] = {}
            local_item_rows: List[Dict[str, Any]] = []
            local_api_rows: List[Dict[str, Any]] = []
            local_failure_rows: List[Dict[str, Any]] = []

            persona_rng = np.random.default_rng(random_seed + persona_idx)
            blocks = _build_blocks(persona_rng)

            for block_index, block_items in enumerate(blocks):
                block_tests = ",".join(sorted({it.test_name for it in block_items}))
                plan = planning_agent.make_plan(self.config.admin_mode, block_items)
                system_prompt = persona_prompt
                user_prompt, survey_lookup = build_user_prompt(
                    items=block_items,
                    user_context=user_context,
                    plan=plan,
                )
                local_api_rows.append(
                    {
                        "persona_id": persona_id,
                        "source_index": idx,
                        "block_index": block_index,
                        "admin_mode": self.config.admin_mode,
                        "engine": "openai",
                        "test_names": block_tests,
                        "item_ids": [it.item_id for it in block_items],
                        "survey_item_ids": {
                            s: [it.item_id for it in its]
                            for s, its in survey_lookup.items()
                        },
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "prompt": "\n\n".join([system_prompt, user_prompt]),
                    }
                )

                # A hard failure (retries exhausted or a non-retryable error)
                # is recorded with its exception type and status code, and
                # the block's items are treated as missing, never as empty
                # answers.
                try:
                    answers = llm_client.answer_block(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        items=block_items,
                        survey_lookup=survey_lookup,
                        persona_key=persona_key,
                    )
                    local_api_rows[-1]["finish_reason"] = llm_client.last_call_meta.get("finish_reason")
                    local_api_rows[-1]["truncated"] = llm_client.last_call_meta.get("truncated")
                except LLMHardFailure as hard_fail:
                    local_api_rows[-1]["finish_reason"] = None
                    local_api_rows[-1]["truncated"] = None
                    local_failure_rows.append(
                        {
                            "persona_idx": persona_idx,
                            "persona_id": persona_id,
                            "source_index": idx,
                            "engine": "openai",
                            "block_index": block_index,
                            "test_names": block_tests,
                            "missing_item_ids": json.dumps([it.item_id for it in block_items]),
                            "reason": f"hard_failure:{hard_fail.exception_type}:{hard_fail.status_code}",
                        }
                    )
                    answers = {}
                missing_ids = self._missing_item_ids(block_items, answers)
                retry_count = 0
                while missing_ids and retry_count < 2:
                    retry_count += 1
                    missing_items = [it for it in block_items if it.item_id in set(missing_ids)]
                    retry_prompt, retry_lookup = build_user_prompt(
                        items=missing_items,
                        user_context=user_context,
                        plan=(
                            "Retry only missing items. Return complete JSON for "
                            "all items shown in this retry block."
                        ),
                    )
                    try:
                        retry_answers = llm_client.answer_block(
                            system_prompt=system_prompt,
                            user_prompt=retry_prompt,
                            items=missing_items,
                            survey_lookup=retry_lookup,
                            persona_key=persona_key,
                        )
                    except LLMHardFailure as hard_fail:
                        local_failure_rows.append(
                            {
                                "persona_idx": persona_idx,
                                "persona_id": persona_id,
                                "source_index": idx,
                                "engine": "openai",
                                "block_index": block_index,
                                "test_names": block_tests,
                                "missing_item_ids": json.dumps(missing_ids),
                                "reason": f"hard_failure_on_retry:{hard_fail.exception_type}:{hard_fail.status_code}",
                            }
                        )
                        retry_answers = {}
                    for iid, txt in retry_answers.items():
                        if str(txt).strip():
                            answers[iid] = txt
                    missing_ids = self._missing_item_ids(block_items, answers)

                if missing_ids:
                    RUN_COUNTERS.record_empty_or_unparseable(len(missing_ids))
                    local_failure_rows.append(
                        {
                            "persona_idx": persona_idx,
                            "persona_id": persona_id,
                            "source_index": idx,
                            "engine": "openai",
                            "block_index": block_index,
                            "test_names": block_tests,
                            "missing_item_ids": json.dumps(missing_ids),
                            "reason": "missing_answers_after_retries",
                        }
                    )

                for presented_index, item in enumerate(block_items, start=1):
                    answer_text = answers.get(item.item_id, "").strip()
                    scale_name, trait = self._resolve_scale_and_trait(item)
                    record: Dict[str, Any] = {
                        "persona_id": persona_id,
                        "block_index": block_index,
                        "presented_index": presented_index,
                        "test_name": item.test_name,
                        "item_id": item.item_id,
                        "scale_name": scale_name,
                        "trait": trait,
                        "item_type": item.item_type,
                        "response_text": answer_text,
                    }
                    if item.item_type == "likert":
                        response_numeric = self._parse_likert_from_text(answer_text)
                        record["response_numeric"] = response_numeric
                        record["correct"] = None
                        if item.test_name == "BFI" and item.scale_name:
                            bfi_scores[item.scale_name].append(response_numeric)
                        elif item.test_name == "NFC":
                            nfc_items.append(response_numeric)
                    elif item.item_type == "crt":
                        record["response_numeric"] = None
                        if self.config.use_pseudo:
                            correct_flag = self._score_crt_from_text(answer_text, item)
                            record["correct"] = int(correct_flag)
                            crt_corrects.append(int(correct_flag))
                        else:
                            # CRT answers from real-LLM runs are not scored here;
                            # the CRT scorer (score/scoring_v2) scores them downstream.
                            record["correct"] = None
                    elif item.item_type == "reasoning":
                        record["response_numeric"] = None
                        record["correct"] = None
                    elif item.item_type == "open":
                        text = answer_text
                        record["response_numeric"] = None
                        record["correct"] = None
                        if item.metadata:
                            record["writing_condition"] = item.metadata.get("condition")
                            record["writing_tags"] = item.metadata.get("tags")
                        writing_texts[item.item_id] = text
                        metrics_ = compute_writing_metrics(text)
                        writing_metrics[item.item_id] = metrics_
                        for m_name, m_val in metrics_.items():
                            record[f"writing_{m_name}"] = m_val
                    else:
                        raise ValueError(f"Unknown item_type '{item.item_type}'.")
                    local_item_rows.append(record)

            persona_record: Dict[str, Any] = row.to_dict()
            persona_record["persona_id"] = persona_id
            for domain, scores in bfi_scores.items():
                persona_record[f"sim_bfi_{domain}"] = (
                    float(np.mean(scores)) if scores else float("nan")
                )
            persona_record["sim_nfc_score"] = (
                float(np.mean(nfc_items)) if nfc_items else float("nan")
            )
            if self.config.use_pseudo:
                persona_record["sim_crt2_score"] = (
                    float(np.sum(crt_corrects)) if crt_corrects else float("nan")
                )
            else:
                # CRT answers from real-LLM runs are not scored here; the CRT
                # scorer (score/scoring_v2) scores them downstream.
                persona_record["sim_crt2_score"] = np.nan
            for wt_id, text in writing_texts.items():
                persona_record[f"writing_{wt_id}_text"] = text
                metrics_ = writing_metrics.get(wt_id, {})
                for m_name, m_val in metrics_.items():
                    persona_record[f"writing_{wt_id}_{m_name}"] = m_val

            return (
                persona_idx,
                persona_record,
                local_item_rows,
                local_api_rows,
                local_failure_rows,
            )

        persona_rows_by_idx: Dict[int, Dict[str, Any]] = {}
        item_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
        api_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
        failure_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_meta = {
                executor.submit(_process_persona, persona_idx, idx, row): (
                    persona_idx,
                    idx,
                    row,
                )
                for persona_idx, (idx, row) in enumerate(persona_entries)
            }
            done_count = 0
            failed_count = 0
            for future in as_completed(future_to_meta):
                pidx, idx, row = future_to_meta[future]
                try:
                    (
                        pidx,
                        persona_record,
                        local_item_rows,
                        local_api_rows,
                        local_failure_rows,
                    ) = future.result()
                    persona_rows_by_idx[pidx] = persona_record
                    item_rows_by_idx[pidx] = local_item_rows
                    api_rows_by_idx[pidx] = local_api_rows
                    if local_failure_rows:
                        failure_rows_by_idx[pidx] = local_failure_rows
                        failed_count += 1
                except Exception as exc:
                    persona_id = row.get("persona_id", idx)
                    failure_rows_by_idx[pidx] = [
                        {
                            "persona_idx": pidx,
                            "persona_id": persona_id,
                            "source_index": idx,
                            "engine": "openai",
                            "block_index": None,
                            "test_names": None,
                            "missing_item_ids": "[]",
                            "reason": f"persona_exception: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    ]
                    failed_count += 1
                done_count += 1
                in_flight = min(workers, max(0, total_personas - done_count))
                print(
                    self._render_parallel_status(
                        done=done_count,
                        total=total_personas,
                        in_flight=in_flight,
                        workers=workers,
                        elapsed_s=time.time() - run_start_ts,
                        failed=failed_count,
                    ),
                    flush=True,
                )

        ordered_indices = sorted(persona_rows_by_idx.keys())
        persona_rows = [persona_rows_by_idx[i] for i in ordered_indices]
        item_rows: List[Dict[str, Any]] = []
        failure_rows: List[Dict[str, Any]] = []
        for i in ordered_indices:
            item_rows.extend(item_rows_by_idx.get(i, []))
            api_call_rows.extend(api_rows_by_idx.get(i, []))
        for i in sorted(failure_rows_by_idx.keys()):
            failure_rows.extend(failure_rows_by_idx.get(i, []))

        per_persona_df = pd.DataFrame(persona_rows)
        per_item_df = pd.DataFrame(item_rows)
        api_calls_df = pd.DataFrame(api_call_rows) if api_call_rows else None
        failed_personas_df = pd.DataFrame(failure_rows) if failure_rows else None
        return SimulationResult(
            per_persona=per_persona_df,
            per_item=per_item_df,
            config=self.config,
            api_calls=api_calls_df,
            failed_personas=failed_personas_df,
        )

    def _rollout_with_anthropic(
        self,
        personas_df: pd.DataFrame,
        persona_prompts: pd.Series,
        random_seed: int,
        api_call_rows: List[Dict[str, Any]],
    ) -> SimulationResult:
        """
        Rollout using the Anthropic Messages API.
        """
        items_by_test = self.test_bank.items_by_test()
        memory_agent = MemoryAgent(self.config)
        planning_agent = PlanningAgent()
        persona_entries = list(personas_df.iterrows())
        total_personas = len(persona_entries)
        workers = max(1, min(self.config.max_parallel_personas, total_personas))
        run_start_ts = time.time()

        def _build_blocks(persona_rng: np.random.Generator) -> List[List[TestItem]]:
            return self._build_blocks(items_by_test, persona_rng)

        def _process_persona(
            persona_idx: int,
            idx: Any,
            row: pd.Series,
        ) -> Tuple[
            int,
            Dict[str, Any],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
            List[Dict[str, Any]],
        ]:
            llm_client = AnthropicLLMClient(
                model=self.config.anthropic_model,
                max_tokens=self.config.anthropic_max_output_tokens,
                multi_turn_conversation=self.config.multi_turn_conversation,
                temperature=self.config.temperature,
                thinking=self.config.anthropic_thinking,
                max_retries=self.config.max_api_retries,
            )
            persona_obj = Persona.from_series(row, self.config)
            persona_id = persona_obj.persona_id if persona_obj.persona_id is not None else idx
            persona_key = str(persona_id)
            persona_prompt = persona_prompts.loc[idx]
            user_context = memory_agent.get_user_context(row)

            bfi_scores: Dict[str, List[int]] = {
                domain: [] for domain in self.config.bfi_columns
            }
            nfc_items: List[int] = []
            crt_corrects: List[int] = []
            writing_texts: Dict[str, str] = {}
            writing_metrics: Dict[str, Dict[str, float]] = {}
            local_item_rows: List[Dict[str, Any]] = []
            local_api_rows: List[Dict[str, Any]] = []
            local_failure_rows: List[Dict[str, Any]] = []

            persona_rng = np.random.default_rng(random_seed + persona_idx)
            blocks = _build_blocks(persona_rng)

            for block_index, block_items in enumerate(blocks):
                block_tests = ",".join(sorted({it.test_name for it in block_items}))
                plan = planning_agent.make_plan(self.config.admin_mode, block_items)
                system_prompt = persona_prompt
                user_prompt, survey_lookup = build_user_prompt(
                    items=block_items,
                    user_context=user_context,
                    plan=plan,
                )
                local_api_rows.append(
                    {
                        "persona_id": persona_id,
                        "source_index": idx,
                        "block_index": block_index,
                        "admin_mode": self.config.admin_mode,
                        "engine": "anthropic",
                        "test_names": block_tests,
                        "item_ids": [it.item_id for it in block_items],
                        "survey_item_ids": {
                            s: [it.item_id for it in its]
                            for s, its in survey_lookup.items()
                        },
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "prompt": "\n\n".join([system_prompt, user_prompt]),
                    }
                )

                # A hard failure (retries exhausted or a non-retryable error)
                # is recorded with its exception type and status code, and
                # the block's items are treated as missing, never as empty
                # answers.
                try:
                    answers = llm_client.answer_block(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        items=block_items,
                        survey_lookup=survey_lookup,
                        persona_key=persona_key,
                    )
                except LLMHardFailure as hard_fail:
                    local_failure_rows.append(
                        {
                            "persona_idx": persona_idx,
                            "persona_id": persona_id,
                            "source_index": idx,
                            "engine": "anthropic",
                            "block_index": block_index,
                            "test_names": block_tests,
                            "missing_item_ids": json.dumps(
                                [it.item_id for it in block_items]
                            ),
                            "reason": (
                                f"hard_failure:{hard_fail.exception_type}:"
                                f"{hard_fail.status_code}"
                            ),
                        }
                    )
                    answers = {}
                missing_ids = self._missing_item_ids(block_items, answers)
                retry_count = 0
                while missing_ids and retry_count < 2:
                    retry_count += 1
                    missing_items = [it for it in block_items if it.item_id in set(missing_ids)]
                    retry_prompt, retry_lookup = build_user_prompt(
                        items=missing_items,
                        user_context=user_context,
                        plan=(
                            "Retry only missing items. Return complete JSON for "
                            "all items shown in this retry block."
                        ),
                    )
                    try:
                        retry_answers = llm_client.answer_block(
                            system_prompt=system_prompt,
                            user_prompt=retry_prompt,
                            items=missing_items,
                            survey_lookup=retry_lookup,
                            persona_key=persona_key,
                        )
                    except LLMHardFailure as hard_fail:
                        local_failure_rows.append(
                            {
                                "persona_idx": persona_idx,
                                "persona_id": persona_id,
                                "source_index": idx,
                                "engine": "anthropic",
                                "block_index": block_index,
                                "test_names": block_tests,
                                "missing_item_ids": json.dumps(missing_ids),
                                "reason": (
                                    f"hard_failure_on_retry:"
                                    f"{hard_fail.exception_type}:"
                                    f"{hard_fail.status_code}"
                                ),
                            }
                        )
                        retry_answers = {}
                    for iid, txt in retry_answers.items():
                        if str(txt).strip():
                            answers[iid] = txt
                    missing_ids = self._missing_item_ids(block_items, answers)

                if missing_ids:
                    RUN_COUNTERS.record_empty_or_unparseable(len(missing_ids))
                    local_failure_rows.append(
                        {
                            "persona_idx": persona_idx,
                            "persona_id": persona_id,
                            "source_index": idx,
                            "engine": "anthropic",
                            "block_index": block_index,
                            "test_names": block_tests,
                            "missing_item_ids": json.dumps(missing_ids),
                            "reason": "missing_answers_after_retries",
                        }
                    )

                for presented_index, item in enumerate(block_items, start=1):
                    answer_text = answers.get(item.item_id, "").strip()
                    scale_name, trait = self._resolve_scale_and_trait(item)
                    record: Dict[str, Any] = {
                        "persona_id": persona_id,
                        "block_index": block_index,
                        "presented_index": presented_index,
                        "test_name": item.test_name,
                        "item_id": item.item_id,
                        "scale_name": scale_name,
                        "trait": trait,
                        "item_type": item.item_type,
                        "response_text": answer_text,
                    }
                    if item.item_type == "likert":
                        response_numeric = self._parse_likert_from_text(answer_text)
                        record["response_numeric"] = response_numeric
                        if item.scale_name in bfi_scores:
                            bfi_scores[item.scale_name].append(response_numeric)
                        elif item.trait == "nfc":
                            nfc_items.append(response_numeric)
                    elif item.item_type == "crt":
                        record["response_numeric"] = answer_text
                        if self.config.use_pseudo:
                            correct = self._score_crt_from_text(answer_text, item)
                            record["crt_correct"] = int(correct)
                            crt_corrects.append(int(correct))
                        else:
                            # CRT answers from real-LLM runs are not scored here;
                            # the CRT scorer (score/scoring_v2) scores them downstream.
                            record["crt_correct"] = None
                    elif item.item_type == "reasoning":
                        record["response_numeric"] = answer_text
                        record["crt_correct"] = None
                    else:
                        record["response_numeric"] = None
                        writing_texts[item.item_id] = answer_text
                        writing_metrics[item.item_id] = {
                            "length_words": len(answer_text.split()),
                            "length_chars": len(answer_text),
                        }
                    local_item_rows.append(record)

            persona_record: Dict[str, Any] = row.to_dict()
            persona_record["persona_id"] = persona_id
            for domain, scores in bfi_scores.items():
                persona_record[f"sim_bfi_{domain}"] = (
                    float(np.mean(scores)) if scores else float("nan")
                )
            persona_record["sim_nfc_score"] = (
                float(np.mean(nfc_items)) if nfc_items else float("nan")
            )
            if self.config.use_pseudo:
                persona_record["sim_crt2_score"] = (
                    float(np.sum(crt_corrects)) if crt_corrects else float("nan")
                )
            else:
                # CRT answers from real-LLM runs are not scored here; the CRT
                # scorer (score/scoring_v2) scores them downstream.
                persona_record["sim_crt2_score"] = np.nan
            for wt_id, text in writing_texts.items():
                persona_record[f"writing_{wt_id}_text"] = text
                metrics_ = writing_metrics.get(wt_id, {})
                for m_name, m_val in metrics_.items():
                    persona_record[f"writing_{wt_id}_{m_name}"] = m_val
            return (
                persona_idx,
                persona_record,
                local_item_rows,
                local_api_rows,
                local_failure_rows,
            )

        persona_rows_by_idx: Dict[int, Dict[str, Any]] = {}
        item_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
        api_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
        failure_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_meta = {
                executor.submit(_process_persona, persona_idx, idx, row): (
                    persona_idx,
                    idx,
                    row,
                )
                for persona_idx, (idx, row) in enumerate(persona_entries)
            }
            done_count = 0
            failed_count = 0
            for future in as_completed(future_to_meta):
                pidx, idx, row = future_to_meta[future]
                try:
                    (
                        pidx,
                        persona_record,
                        local_item_rows,
                        local_api_rows,
                        local_failure_rows,
                    ) = future.result()
                    persona_rows_by_idx[pidx] = persona_record
                    item_rows_by_idx[pidx] = local_item_rows
                    api_rows_by_idx[pidx] = local_api_rows
                    if local_failure_rows:
                        failure_rows_by_idx[pidx] = local_failure_rows
                        failed_count += 1
                except Exception as exc:
                    persona_id = row.get("persona_id", idx)
                    failure_rows_by_idx[pidx] = [
                        {
                            "persona_idx": pidx,
                            "persona_id": persona_id,
                            "source_index": idx,
                            "engine": "anthropic",
                            "block_index": None,
                            "test_names": None,
                            "missing_item_ids": "[]",
                            "reason": f"persona_exception: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    ]
                    failed_count += 1
                done_count += 1
                in_flight = min(workers, max(0, total_personas - done_count))
                print(
                    self._render_parallel_status(
                        done=done_count,
                        total=total_personas,
                        in_flight=in_flight,
                        workers=workers,
                        elapsed_s=time.time() - run_start_ts,
                        failed=failed_count,
                    ),
                    flush=True,
                )

        ordered_indices = sorted(persona_rows_by_idx.keys())
        persona_rows = [persona_rows_by_idx[i] for i in ordered_indices]
        item_rows: List[Dict[str, Any]] = []
        failure_rows: List[Dict[str, Any]] = []
        for i in ordered_indices:
            item_rows.extend(item_rows_by_idx.get(i, []))
            api_call_rows.extend(api_rows_by_idx.get(i, []))
        for i in sorted(failure_rows_by_idx.keys()):
            failure_rows.extend(failure_rows_by_idx.get(i, []))

        per_persona_df = pd.DataFrame(persona_rows)
        per_item_df = pd.DataFrame(item_rows)
        api_calls_df = pd.DataFrame(api_call_rows) if api_call_rows else None
        failed_personas_df = pd.DataFrame(failure_rows) if failure_rows else None
        return SimulationResult(
            per_persona=per_persona_df,
            per_item=per_item_df,
            config=self.config,
            api_calls=api_calls_df,
            failed_personas=failed_personas_df,
        )

    def _simulate_item(
        self,
        llm: PseudoLLM,
        persona_id: Any,
        row: pd.Series,
        bfi_traits: Dict[str, float],
        nfc_score: float,
        crt_level: int,
        persona_prompt: str,
        item: TestItem,
        bfi_scores: Dict[str, List[int]],
        nfc_items: List[int],
        crt_corrects: List[int],
        item_rows: List[Dict[str, Any]],
        writing_texts: Dict[str, str],
        writing_metrics: Dict[str, Dict[str, float]],
        block_index: int,
        presented_index: int,
    ) -> None:
        scale_name, trait = self._resolve_scale_and_trait(item)
        record: Dict[str, Any] = {
            "persona_id": persona_id,
            "block_index": block_index,
            "presented_index": presented_index,
            "test_name": item.test_name,
            "item_id": item.item_id,
            "scale_name": scale_name,
            "trait": trait,
            "item_type": item.item_type,
        }

        if item.item_type == "likert":
            if item.trait == "nfc":
                trait_val = nfc_score
            else:
                trait_key = item.trait
                if trait_key is None:
                    raise ValueError(
                        f"Likert item '{item.item_id}' missing trait mapping."
                    )
                if trait_key not in self.config.bfi_columns:
                    raise ValueError(
                        f"Likert item '{item.item_id}' references unknown "
                        f"trait '{trait_key}'. Expected one of "
                        f"{sorted(self.config.bfi_columns.keys())} or 'nfc'."
                    )
                trait_val = float(row[self.config.bfi_columns[trait_key]])
            response = llm.answer_likert(trait_val, item.keyed)
            record["response_text"] = str(response)
            record["response_numeric"] = response
            record["correct"] = None

            if item.test_name == "BFI" and item.scale_name:
                bfi_scores[item.scale_name].append(response)
            elif item.test_name == "NFC":
                nfc_items.append(response)

        elif item.item_type == "crt":
            ans_text, correct = llm.answer_crt(crt_level, item)
            record["response_text"] = ans_text
            record["response_numeric"] = None
            record["correct"] = int(correct)
            crt_corrects.append(int(correct))
        elif item.item_type == "reasoning":
            ans_text = llm.answer_reasoning(crt_level, item)
            record["response_text"] = ans_text
            record["response_numeric"] = None
            record["correct"] = None

        elif item.item_type == "open":
            text = llm.generate_writing(
                persona_prompt, bfi_traits, nfc_score, crt_level, item
            )
            record["response_text"] = text
            record["response_numeric"] = None
            record["correct"] = None
            if item.metadata:
                record["writing_condition"] = item.metadata.get("condition")
                record["writing_tags"] = item.metadata.get("tags")
            writing_texts[item.item_id] = text
            metrics_ = compute_writing_metrics(text)
            writing_metrics[item.item_id] = metrics_
            for m_name, m_val in metrics_.items():
                record[f"writing_{m_name}"] = m_val
        else:
            raise ValueError(f"Unknown item_type '{item.item_type}'.")

        item_rows.append(record)


# ---------------------------------------------------------------------
# Writing metrics (behavioural alignment proxies)
# ---------------------------------------------------------------------


def compute_writing_metrics(text: str) -> Dict[str, float]:
    """
    Compute simple proxy metrics for writing behaviour, intended as
    stand-ins for LIWC-like analytic, insight, and affect scores.
    """
    tokens = [t.lower() for t in text.split()]
    n_tokens = len(tokens) if tokens else 1

    analytic_vocab = {
        "because",
        "therefore",
        "logic",
        "analyze",
        "structure",
        "methodical",
        "reason",
        "reasoning",
        "evidence",
    }
    insight_vocab = {
        "realized",
        "understood",
        "noticed",
        "insight",
        "reflection",
        "reflected",
        "think",
        "thought",
        "considered",
        "aware",
    }
    affect_vocab = {
        "happy",
        "glad",
        "relieved",
        "sad",
        "anxious",
        "afraid",
        "frustrated",
        "angry",
        "excited",
        "upset",
    }

    analytic_count = sum(t in analytic_vocab for t in tokens)
    insight_count = sum(t in insight_vocab for t in tokens)
    affect_count = sum(t in affect_vocab for t in tokens)

    return {
        "length": float(n_tokens),
        "analytic": analytic_count / n_tokens,
        "insight": insight_count / n_tokens,
        "affect": affect_count / n_tokens,
    }


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------


class GridSearchCalibrator:
    """
    Simple grid-search calibrator for simulator parameters.
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        random_seed: int = GLOBAL_RANDOM_SEED,
    ):
        self.param_grid = param_grid
        self.random_seed = random_seed
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        simulator: PersonaSimulator,
        persona_prompts: pd.Series,
        config: SimulationConfig,
    ) -> None:
        """
        Perform grid search over simulator parameters, selecting the set
        that maximises trait fidelity on the training personas.
        """
        keys = sorted(self.param_grid.keys())

        best_score = -float("inf")
        best_params: Optional[Dict[str, Any]] = None

        for combo_index, combo in enumerate(
            itertools.product(*(self.param_grid[k] for k in keys))
        ):
            params = {k: v for k, v in zip(keys, combo)}
            params["random_seed"] = self.random_seed + combo_index

            result = simulator.rollout(train_df, params, persona_prompts)
            score = self._trait_fidelity_score(
                train_df, result.per_persona, config
            )

            if score > best_score:
                best_score = score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params if best_params is not None else {}

    @staticmethod
    def _trait_fidelity_score(
        true_df: pd.DataFrame,
        sim_df: pd.DataFrame,
        config: SimulationConfig,
    ) -> float:
        """
        Compute a scalar trait fidelity score by averaging Pearson
        correlations between target trait scores and simulated scores,
        aligning rows by persona_id.
        """
        if "persona_id" not in true_df.columns or "persona_id" not in sim_df.columns:
            raise ValueError(
                "Both true_df and sim_df must contain a 'persona_id' column "
                "for trait fidelity scoring."
            )

        merged = true_df.merge(sim_df, on="persona_id", how="inner")
        if merged.empty:
            return -float("inf")

        metrics: List[float] = []

        for domain, col_name in config.bfi_columns.items():
            sim_col = f"sim_bfi_{domain}"
            if col_name in merged.columns and sim_col in merged.columns:
                corr = pearson_corr(merged[col_name].astype(float), merged[sim_col].astype(float))
                if not math.isnan(corr):
                    metrics.append(corr)

        if (
            config.nfc_column in merged.columns
            and "sim_nfc_score" in merged.columns
        ):
            corr = pearson_corr(
                merged[config.nfc_column].astype(float),
                merged["sim_nfc_score"].astype(float),
            )
            if not math.isnan(corr):
                metrics.append(corr)

        if (
            config.crt_column in merged.columns
            and "sim_crt2_score" in merged.columns
        ):
            corr = pearson_corr(
                merged[config.crt_column].astype(float),
                merged["sim_crt2_score"].astype(float),
            )
            if not math.isnan(corr):
                metrics.append(corr)

        if not metrics:
            return -float("inf")

        return float(np.mean(metrics))


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


class Evaluator:
    """
    Evaluate simulation outputs according to blueprint metrics.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def compute_metrics(
        self,
        sim_result: SimulationResult,
        val_df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Compute trait fidelity, behavioural alignment, and response
        consistency metrics for the validation personas.
        """
        per_persona = sim_result.per_persona
        per_item = sim_result.per_item

        if "persona_id" not in val_df.columns or "persona_id" not in per_persona.columns:
            raise ValueError(
                "Both validation DataFrame and per_persona results must "
                "contain a 'persona_id' column."
            )

        # Align validation data and simulated per-persona data by persona_id
        merged_pp = val_df.merge(per_persona, on="persona_id", how="inner")

        metrics: Dict[str, Any] = {
            "trait_fidelity": {},
            "behavioural_alignment": {},
            "response_consistency": {},
        }

        # ----------------- Trait fidelity -----------------
        tf = metrics["trait_fidelity"]

        for domain, col_name in self.config.bfi_columns.items():
            sim_col = f"sim_bfi_{domain}"
            if col_name in merged_pp.columns and sim_col in merged_pp.columns:
                true_vals = merged_pp[col_name].astype(float)
                sim_vals = merged_pp[sim_col].astype(float)
                tf[domain] = {
                    "pearson": pearson_corr(true_vals, sim_vals),
                    "spearman": spearman_corr(true_vals, sim_vals),
                }

        if (
            self.config.nfc_column in merged_pp.columns
            and "sim_nfc_score" in merged_pp.columns
        ):
            true_vals = merged_pp[self.config.nfc_column].astype(float)
            sim_vals = merged_pp["sim_nfc_score"].astype(float)
            tf["nfc"] = {
                "pearson": pearson_corr(true_vals, sim_vals),
                "spearman": spearman_corr(true_vals, sim_vals),
            }

        if (
            self.config.crt_column in merged_pp.columns
            and "sim_crt2_score" in merged_pp.columns
        ):
            true_vals = merged_pp[self.config.crt_column].astype(float)
            sim_vals = merged_pp["sim_crt2_score"].astype(float)
            mae = float(np.mean(np.abs(true_vals - sim_vals)))
            tf["crt2"] = {
                "pearson": pearson_corr(true_vals, sim_vals),
                "spearman": spearman_corr(true_vals, sim_vals),
                "mae": mae,
            }

        tf["monotonicity_notes"] = (
            "Higher target traits should correspond to higher simulated "
            "scale scores; see trait_fidelity correlations."
        )

        # ----------------- Behavioural alignment -----------------
        ba = metrics["behavioural_alignment"]

        writing_cols = [
            col for col in merged_pp.columns if col.startswith("writing_")
        ]
        trait_pairs: List[Tuple[str, str]] = []
        for col in writing_cols:
            if col.endswith("_length") or col.endswith(
                ("_analytic", "_insight", "_affect")
            ):
                trait_pairs.append(("nfc", col))
                trait_pairs.append(("crt2", col))

        corrs: Dict[str, Any] = {}
        regressions: Dict[str, Any] = {}
        for trait, w_col in trait_pairs:
            if w_col not in merged_pp.columns:
                continue
            if trait == "nfc":
                t_col = self.config.nfc_column
            else:
                t_col = self.config.crt_column

            if t_col not in merged_pp.columns:
                continue

            true_vals = merged_pp[t_col].astype(float)
            sim_vals = merged_pp[w_col].astype(float)
            mask = true_vals.notna() & sim_vals.notna()
            if mask.sum() < 3:
                continue

            t = true_vals[mask]
            s = sim_vals[mask]
            key = f"{trait}_vs_{w_col}"

            corrs[key] = {
                "pearson": pearson_corr(t, s),
                "spearman": spearman_corr(t, s),
            }

            # Simple linear regression: s = a * t + b
            try:
                coef = np.polyfit(t.to_numpy(), s.to_numpy(), 1)
                regressions[key] = {
                    "slope": float(coef[0]),
                    "intercept": float(coef[1]),
                }
            except Exception:
                regressions[key] = {"slope": float("nan"), "intercept": float("nan")}

        ba["trait_writing_correlations"] = corrs
        ba["trait_writing_regressions"] = regressions

        # Stratified analyses by writing condition and tags (per writing item)
        writing_items = per_item[per_item["item_type"] == "open"].copy()
        by_condition: Dict[str, Any] = {}
        by_tag: Dict[str, Any] = {}

        if not writing_items.empty:
            # Merge trait scores from validation df
            trait_subset_cols = ["persona_id"]
            for col in [self.config.nfc_column, self.config.crt_column]:
                if col in val_df.columns:
                    trait_subset_cols.append(col)
            trait_df = val_df[trait_subset_cols].drop_duplicates("persona_id")
            writing_items = writing_items.merge(trait_df, on="persona_id", how="left")

            metric_cols = [
                c
                for c in writing_items.columns
                if c
                in {
                    "writing_length",
                    "writing_analytic",
                    "writing_insight",
                    "writing_affect",
                }
            ]

            # By condition
            if "writing_condition" in writing_items.columns:
                for cond in sorted(
                    set(
                        v
                        for v in writing_items["writing_condition"].dropna().unique()
                    )
                ):
                    cond_df = writing_items[
                        writing_items["writing_condition"] == cond
                    ]
                    group_metrics: Dict[str, Any] = {}
                    for m_col in metric_cols:
                        for trait, t_col in [
                            ("nfc", self.config.nfc_column),
                            ("crt2", self.config.crt_column),
                        ]:
                            if t_col not in cond_df.columns:
                                continue
                            t_vals = cond_df[t_col].astype(float)
                            s_vals = cond_df[m_col].astype(float)
                            mask = t_vals.notna() & s_vals.notna()
                            if mask.sum() < 3:
                                continue
                            t = t_vals[mask]
                            s = s_vals[mask]
                            key = f"{trait}_vs_{m_col}"
                            corr_info = {
                                "pearson": pearson_corr(t, s),
                                "spearman": spearman_corr(t, s),
                            }
                            try:
                                coef = np.polyfit(t.to_numpy(), s.to_numpy(), 1)
                                reg_info = {
                                    "slope": float(coef[0]),
                                    "intercept": float(coef[1]),
                                }
                            except Exception:
                                reg_info = {
                                    "slope": float("nan"),
                                    "intercept": float("nan"),
                                }
                            group_metrics[key] = {
                                "correlations": corr_info,
                                "regression": reg_info,
                            }
                    if group_metrics:
                        by_condition[str(cond)] = group_metrics

            # By tag (explode tags)
            if "writing_tags" in writing_items.columns:
                def split_tags(val: Any) -> List[str]:
                    if isinstance(val, list):
                        return [str(v).strip() for v in val if str(v).strip()]
                    if isinstance(val, str):
                        return [
                            t.strip()
                            for t in re.split(r"[;,]", val)
                            if t.strip()
                        ]
                    return []

                writing_items = writing_items.assign(
                    _tag_list=writing_items["writing_tags"].apply(split_tags)
                ).explode("_tag_list")

                tag_vals = writing_items["_tag_list"].dropna().unique()
                for tag in sorted(set(tag_vals)):
                    tag_df = writing_items[writing_items["_tag_list"] == tag]
                    group_metrics: Dict[str, Any] = {}
                    for m_col in metric_cols:
                        for trait, t_col in [
                            ("nfc", self.config.nfc_column),
                            ("crt2", self.config.crt_column),
                        ]:
                            if t_col not in tag_df.columns:
                                continue
                            t_vals = tag_df[t_col].astype(float)
                            s_vals = tag_df[m_col].astype(float)
                            mask = t_vals.notna() & s_vals.notna()
                            if mask.sum() < 3:
                                continue
                            t = t_vals[mask]
                            s = s_vals[mask]
                            key = f"{trait}_vs_{m_col}"
                            corr_info = {
                                "pearson": pearson_corr(t, s),
                                "spearman": spearman_corr(t, s),
                            }
                            try:
                                coef = np.polyfit(t.to_numpy(), s.to_numpy(), 1)
                                reg_info = {
                                    "slope": float(coef[0]),
                                    "intercept": float(coef[1]),
                                }
                            except Exception:
                                reg_info = {
                                    "slope": float("nan"),
                                    "intercept": float("nan"),
                                }
                            group_metrics[key] = {
                                "correlations": corr_info,
                                "regression": reg_info,
                            }
                    if group_metrics:
                        by_tag[str(tag)] = group_metrics

        ba["by_condition"] = by_condition
        ba["by_tag"] = by_tag

        # ----------------- Response consistency -----------------
        rc = metrics["response_consistency"]

        alpha_results: Dict[str, float] = {}
        for domain in self.config.bfi_columns.keys():
            subset = per_item[
                (per_item["test_name"] == "BFI")
                & (per_item["scale_name"] == domain)
            ]
            if subset.empty:
                alpha_results[domain] = float("nan")
                continue
            pivot = subset.pivot(
                index="persona_id",
                columns="item_id",
                values="response_numeric",
            )
            pivot = pivot.dropna(axis=0, how="any")
            alpha_results[domain] = cronbach_alpha(pivot)

        rc["cronbach_alpha_bfi"] = alpha_results

        subset_nfc = per_item[per_item["test_name"] == "NFC"]
        if not subset_nfc.empty:
            pivot_nfc = subset_nfc.pivot(
                index="persona_id",
                columns="item_id",
                values="response_numeric",
            )
            pivot_nfc = pivot_nfc.dropna(axis=0, how="any")
            rc["cronbach_alpha_nfc"] = cronbach_alpha(pivot_nfc)
        else:
            rc["cronbach_alpha_nfc"] = float("nan")

        rc["notes"] = (
            "Test–retest reliability and order effects are not explicitly "
            "simulated here but could be assessed by repeating rollouts "
            "with the same personas under different shuffling settings."
        )

        return metrics


# ---------------------------------------------------------------------
# Data loading, network building, and holdout split
# ---------------------------------------------------------------------


def load_data(config: SimulationConfig) -> Dict[str, pd.DataFrame]:
    """
    Load personas and markers from disk according to the provided
    SimulationConfig.
    """
    personas_path = os.path.join(DATA_DIR, config.personas_filename)
    markers_path = os.path.join(DATA_DIR, config.markers_filename)

    if not os.path.isabs(personas_path):
        personas_path = os.path.abspath(personas_path)
    if not os.path.exists(personas_path):
        raise FileNotFoundError(
            f"Personas file not found at '{personas_path}'. "
            f"Ensure PROJECT_ROOT ('{PROJECT_ROOT}') and DATA_DIR "
            f"('{DATA_DIR}') are set correctly and that the file exists."
        )

    config.personas_path = personas_path
    personas_df = pd.read_csv(personas_path)
    if personas_df.empty:
        raise ValueError(f"Personas file '{personas_path}' is empty.")

    if "persona_id" not in personas_df.columns:
        personas_df["persona_id"] = np.arange(len(personas_df))

    if os.path.exists(markers_path):
        markers_df = pd.read_csv(markers_path)
    else:
        markers_df = pd.DataFrame()

    return {"personas": personas_df, "markers": markers_df}


def build_network_and_agents(
    data: Dict[str, pd.DataFrame], config: SimulationConfig
) -> Tuple[pd.DataFrame, pd.Series, TestBank]:
    """
    Validate the personas schema, compute CRT levels, construct
    Persona-based prompts, and build the TestBank from YAML specs.
    """
    personas_df = data["personas"]
    markers_df = data.get("markers")

    # Schema validation
    expected_columns = [
        "Employment Status",
        "Occupation_cat",
        "sex",
        "Highest education level completed",
        "Ethnicity Simplified",
        "Age",
        config.crt_column,
        config.nfc_column,
    ] + list(config.bfi_columns.values())
    missing_schema = [c for c in expected_columns if c not in personas_df.columns]
    if missing_schema:
        raise ValueError(
            "personas file is missing required columns: "
            f"{missing_schema}. Expected schema includes demographic "
            "variables and trait scores."
        )
    if "Country" not in personas_df.columns:
        print("Note: 'Country' column not present; country will not be used.", flush=True)

    missing_cols = [
        col
        for col in config.bfi_columns.values()
        if col not in personas_df.columns
    ]
    if config.nfc_column not in personas_df.columns:
        missing_cols.append(config.nfc_column)
    if config.crt_column not in personas_df.columns:
        missing_cols.append(config.crt_column)

    if missing_cols:
        raise ValueError(
            "The following required trait columns are missing from "
            f"personas file: {missing_cols}. Please ensure the CSV "
            "contains these columns or adjust the configuration."
        )

    prepared_personas_df = prepare_crt_levels(personas_df, config)
    config.persona_bank_size = int(len(prepared_personas_df))
    prepared_personas_df = filter_personas_by_ids(prepared_personas_df, config)
    # Keep this null for full-bank runs so downstream analysis can distinguish
    # "no sampling cap requested" from an explicit capped sample size.
    config.sample_n_requested = (
        int(config.max_personas) if config.max_personas is not None else None
    )
    config.sampling_stratify_source = (
        f"derived_{config.crt_level_column}_from_{config.crt_column}"
        if config.sampling_mode == "stratified_crt"
        else None
    )

    if config.sampling_mode == "sequential":
        # Sequential mode samples from `prepared_personas_df`, which carries
        # the --persona-ids filter and the fixed-cutoff CRT level column, so a
        # --persona-ids run uses exactly the requested personas.
        personas_df = sample_personas_for_run(prepared_personas_df, config)
    else:
        # For explicit sampling modes, derive CRT bins on the full CSV first so
        # the same discretization drives both sampling and later persona prompts.
        personas_df = sample_personas_for_run(prepared_personas_df, config)

    # Range checks for BFI and NFC
    for col in config.bfi_columns.values():
        vals = personas_df[col].astype(float)
        if ((vals < 0.5) | (vals > 5.5)).any():
            raise ValueError(
                f"Column '{col}' appears to be out of expected [1,5] range. "
                "Please ensure BFI scores are on a 1–5 scale."
            )
    nfc_vals = personas_df[config.nfc_column].astype(float)
    if ((nfc_vals < 0.5) | (nfc_vals > 5.5)).any():
        raise ValueError(
            f"Column '{config.nfc_column}' appears to be out of expected [1,5] range. "
            "Please ensure NFC_Total is on a 1–5 scale."
        )

    # Always compute/discretize crt_level_column from the continuous CRT trait.
    # In explicit sampling modes this may already exist from the full-data pass;
    # recomputing is only needed for the default sequential path.
    if (
        config.crt_level_column not in personas_df.columns
        or config.sampling_mode == "sequential"
    ):
        personas_df = prepare_crt_levels(personas_df, config)

    config.sample_n_selected = int(len(personas_df))

    prompt_builder = PersonaPromptBuilder(
        config=config,
        markers_df=markers_df if markers_df is not None and not markers_df.empty else None,
        rng=np.random.default_rng(config.random_seed),
    )

    persona_prompts: Dict[Any, str] = {}

    for idx, row in personas_df.iterrows():
        persona = Persona.from_series(row, config)
        prompt = prompt_builder.build_full_persona_prompt(
            persona,
            row,
            style=config.persona_prompt_style,
            numeric_crt_only=config.use_crt_numeric_only,
        )
        # Append the optional --persona-system-suffix paragraph to the
        # rendered system message.
        if config.persona_system_suffix is not None:
            prompt = f"{prompt}\n\n{config.persona_system_suffix}"
        persona_prompts[idx] = prompt

    persona_prompts_series = pd.Series(persona_prompts)

    test_bank = build_test_bank(
        config.tests_included if config.tests_included_explicitly_set else None
    )
    config.tests_included = [
        block_name
        for block_name in DEFAULT_INCLUDED_BLOCKS
        if any(
            test_name in test_bank.items_by_test()
            for test_name in BLOCK_TO_TEST_NAMES[block_name]
        )
    ]

    return personas_df, persona_prompts_series, test_bank


def holdout_split(
    personas_df: pd.DataFrame, config: SimulationConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split personas into training and validation sets according to
    config.holdout_fraction, optionally respecting a temporal ordering
    column if provided.
    """
    if config.holdout_fraction <= 0.0:
        return personas_df.copy(), personas_df.copy()
    if config.time_column and config.time_column in personas_df.columns:
        sorted_df = personas_df.sort_values(config.time_column)
    else:
        sort_col = "persona_id" if "persona_id" in personas_df.columns else None
        if sort_col:
            sorted_df = personas_df.sort_values(sort_col)
        else:
            sorted_df = personas_df.sort_index()

    n_total = len(sorted_df)
    n_val = max(1, int(round(config.holdout_fraction * n_total)))
    n_train = max(1, n_total - n_val)

    # Preserve original indices to keep alignment with persona_prompts
    train_df = sorted_df.iloc[:n_train]
    val_df = sorted_df.iloc[n_train:]

    return train_df, val_df


# ---------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------


def save_results(
    metrics: Dict[str, Any],
    sim_result: SimulationResult,
    config: SimulationConfig,
    calibrator: GridSearchCalibrator,
) -> None:
    """
    Save simulation outputs (per-persona and per-item), API call logs,
    and a metrics/configuration report to disk.
    """
    output_dir = os.path.join(PROJECT_ROOT, config.output_dir)
    output_dir = os.path.abspath(output_dir)
    config.resolved_output_dir = output_dir
    ensure_dir(output_dir)

    per_persona_path = os.path.join(output_dir, "per_persona_results.csv")
    per_item_path = os.path.join(output_dir, "per_item_results.csv")
    sim_result.per_persona.to_csv(per_persona_path, index=False)
    sim_result.per_item.to_csv(per_item_path, index=False)

    if sim_result.api_calls is not None:
        api_calls_csv_path = os.path.join(output_dir, "api_calls.csv")
        sim_result.api_calls.to_csv(api_calls_csv_path, index=False)
        api_calls_jsonl_path = os.path.join(output_dir, "api_calls.jsonl")
        with open(api_calls_jsonl_path, "w", encoding="utf-8") as f_api:
            for _, row in sim_result.api_calls.iterrows():
                rec = row.to_dict()
                json.dump(rec, f_api)
                f_api.write("\n")

    if sim_result.failed_personas is not None:
        failed_csv_path = os.path.join(output_dir, "failed_personas.csv")
        sim_result.failed_personas.to_csv(failed_csv_path, index=False)
        failed_jsonl_path = os.path.join(output_dir, "failed_personas.jsonl")
        with open(failed_jsonl_path, "w", encoding="utf-8") as f_failed:
            for _, row in sim_result.failed_personas.iterrows():
                rec = row.to_dict()
                json.dump(rec, f_failed)
                f_failed.write("\n")

    _api_reliability = RUN_COUNTERS.as_dict()
    _api_reliability["hard_failure_rate_exceeds_1pct"] = (
        _api_reliability["hard_failure_rate"] > 0.01
    )
    report = {
        # Per-run API reliability counters (OpenAI-family and Anthropic
        # paths). When hard_failure_rate_exceeds_1pct is true, rerun the
        # personas in failed_personas.csv before scoring or analysis.
        "api_reliability": _api_reliability,
        "config": {
            "personas_filename": config.personas_filename,
            "human_mode": config.human_mode,
            "personas_path": config.personas_path,
            "markers_filename": config.markers_filename,
            "admin_mode": config.admin_mode,
            "shuffle_items_within_tests": config.shuffle_items_within_tests,
            "shuffle_test_order": config.shuffle_test_order,
            "randomize_reasoning_block": config.randomize_reasoning_block,
            "holdout_fraction": config.holdout_fraction,
            "max_personas": config.max_personas,
            "persona_ids": config.persona_ids,
            "sample_n_requested": config.sample_n_requested,
            "sample_n_selected": config.sample_n_selected,
            "persona_bank_size": config.persona_bank_size,
            "sampling_mode": config.sampling_mode,
            "sampling_without_replacement": True,
            "sampling_stratify_source": config.sampling_stratify_source,
            "time_column": config.time_column,
            "random_seed": config.random_seed,
            "output_dir": config.output_dir,
            "resolved_output_dir": config.resolved_output_dir,
            "max_parallel_personas": config.max_parallel_personas,
            "bfi_columns": config.bfi_columns,
            "nfc_column": config.nfc_column,
            "crt_column": config.crt_column,
            "crt_level_column": config.crt_level_column,
            "crt_cutoffs_filename": config.crt_cutoffs_filename,
            "crt_cutoffs_path": config.crt_cutoffs_path,
            "crt_binning_method": config.crt_binning_method,
            "crt_cutoffs_values": config.crt_cutoffs_values,
            "use_crt_numeric_only": config.use_crt_numeric_only,
            "use_openai": config.use_openai,
            "openai_model": config.openai_model,
            "openai_max_output_tokens": config.openai_max_output_tokens,
            "openai_base_url": config.openai_base_url,
            "openai_key_env": config.openai_key_env,
            "openai_api_style": config.openai_api_style,
            "openai_extra_params": config.openai_extra_params,
            "openai_extra_body": config.openai_extra_body,
            "max_api_retries": config.max_api_retries,
            "use_anthropic": config.use_anthropic,
            "anthropic_model": config.anthropic_model,
            "anthropic_max_output_tokens": config.anthropic_max_output_tokens,
            "temperature": config.temperature,
            "anthropic_thinking": config.anthropic_thinking,
            "use_pseudo": config.use_pseudo,
            "persona_system_suffix": config.persona_system_suffix,
            "persona_prompt_style": config.persona_prompt_style,
            "persona_spec_mode": config.persona_spec_mode,
            "persona_prompt_version": config.persona_prompt_version,
            "task_prompt_version": config.task_prompt_version,
            "bfi_ladder_granularity": config.bfi_ladder_granularity,
            "include_user_context": config.include_user_context,
            "multi_turn_conversation": config.multi_turn_conversation,
            "multi_turn_scope": (
                "persona_session"
                if config.multi_turn_conversation
                else "single_turn"
            ),
            "tests_included": config.tests_included,
            "run_label": config.run_label,
            "failed_persona_count": (
                int(len(sim_result.failed_personas))
                if sim_result.failed_personas is not None
                else 0
            ),
        },
    }

    metrics_path = os.path.join(output_dir, "metrics_and_config.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    jsonl_path = os.path.join(output_dir, "per_persona_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in sim_result.per_persona.iterrows():
            rec = row.to_dict()
            json.dump(rec, f)
            f.write("\n")


# ---------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------


def parse_persona_ids(values: Optional[List[str]]) -> List[int]:
    if not values:
        return []
    ids: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                ids.append(int(part))
    return ids


def parse_cli() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate LLM-based personas, administer psychometric tests "
            "and writing tasks, calibrate simulator parameters, and "
            "evaluate fidelity and reliability."
        )
    )

    parser.add_argument(
        "--personas-filename",
        type=str,
        default="synthetic_personas.csv",
        help="File name of synthetic personas CSV (relative to DATA_DIR).",
    )
    parser.add_argument(
        "--human-mode",
        action="store_true",
        help=(
            "Use human_personas.csv from DATA_DIR instead of the default "
            "synthetic_personas.csv."
        ),
    )
    parser.add_argument(
        "--markers-filename",
        type=str,
        default="serapio_goldberg_markers.csv",
        help="File name of Serapio/Goldberg markers CSV (relative to "
        "DATA_DIR).",
    )
    parser.add_argument(
        "--admin-mode",
        type=str,
        default="per_test",
        choices=["per_item", "per_test", "all_tests"],
        help="Administration granularity for API calls across the included survey blocks.",
    )
    parser.add_argument(
        "--tests-included",
        type=str,
        nargs="+",
        default=None,
        choices=DEFAULT_INCLUDED_BLOCKS,
        help=(
            "Optional subset of survey blocks to administer. "
            "Choices: writing, reasoning, nfc, bfi. "
            "If omitted, all blocks are included in the default order."
        ),
    )
    parser.add_argument(
        "--shuffle-items-within-tests",
        action="store_true",
        help="Shuffle item order within each test.",
    )
    parser.add_argument(
        "--shuffle-test-order",
        action="store_true",
        help="Shuffle the order of tests.",
    )
    parser.add_argument(
        "--no-randomize-reasoning-block",
        dest="randomize_reasoning_block",
        action="store_false",
        help="Disable randomization of the merged reasoning block (CRT2+bCRT+dCRT).",
    )
    parser.set_defaults(randomize_reasoning_block=True)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Fraction of personas used for validation (default: 0.0 / disabled).",
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        default=None,
        help="Optional cap on number of personas (useful for dry runs).",
    )
    parser.add_argument(
        "--persona-ids",
        nargs="+",
        default=None,
        help=(
            "Optional persona_id values to run (e.g. a profile panel, or failed personas to rerun). "
            "Accepts space-separated values or comma-separated groups."
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="sequential",
        choices=["sequential", "random", "stratified_crt"],
        help=(
            "How to select personas before simulation. "
            "'sequential' takes the first N rows when --max-personas is set "
            "(all rows otherwise). 'random' samples N rows uniformly "
            "without replacement. 'stratified_crt' samples approximately "
            "equally across fixed human-derived CRT bins."
        ),
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Optional column for temporal ordering (if present in "
        "personas CSV).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=GLOBAL_RANDOM_SEED,
        help="Random seed for deterministic simulations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory relative to PROJECT_ROOT. Defaults to a "
            "timestamped folder under simulation_outputs/."
        ),
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional human-readable label to save in metrics_and_config.json.",
    )
    parser.add_argument(
        "--max-parallel-personas",
        type=int,
        default=4,
        help=(
            "Maximum number of personas simulated concurrently. "
            "Each persona remains sequential across its own survey blocks."
        ),
    )
    parser.add_argument(
        "--use-crt-numeric-only",
        action="store_true",
        help="Use numeric-only CRT specification rather than descriptive "
        "sentence.",
    )
    parser.add_argument(
        "--crt-cutoffs-filename",
        type=str,
        default="crt2_level_cutoffs.yaml",
        help=(
            "YAML file containing fixed human-derived CRT2 cutoff thresholds "
            "(relative to DATA_DIR unless an absolute path is provided)."
        ),
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI LLM via Responses API instead of PseudoLLM.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5",
        help="OpenAI model to use with the Responses API (default: gpt-5).",
    )
    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=4000,
        help="Maximum number of output tokens for Responses API calls.",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help=(
            "Optional base URL override to route --use-openai at an "
            "OpenAI-compatible endpoint (e.g. DeepSeek "
            "https://api.deepseek.com, or Google's Gemini OpenAI-compat "
            "endpoint https://generativelanguage.googleapis.com/v1beta/openai/). "
            "Default (None) uses OpenAI's own endpoint."
        ),
    )
    parser.add_argument(
        "--openai-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help=(
            "Environment variable name to read the API key from for "
            "--use-openai (default: OPENAI_API_KEY; use GEMINI_API_KEY / "
            "DEEPSEEK_API_KEY for the corresponding OpenAI-compatible cells)."
        ),
    )
    parser.add_argument(
        "--openai-api-style",
        type=str,
        default="responses",
        choices=["responses", "chat_completions"],
        help=(
            "API style for --use-openai. 'responses' is the native OpenAI "
            "Responses API (default; used for gpt-5.6-sol). "
            "'chat_completions' is required for DeepSeek and Gemini's "
            "OpenAI-compatible endpoints, which do not implement the "
            "Responses API."
        ),
    )
    parser.add_argument(
        "--openai-extra-params-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object merged verbatim into the top-level API "
            "call kwargs, per cell. Example (OpenAI reasoning effort): "
            '\'{"reasoning": {"effort": "none"}}\'.'
        ),
    )
    parser.add_argument(
        "--openai-extra-body-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object passed as the extra_body= kwarg on "
            "chat-completions calls, per cell, to carry provider-specific "
            "reasoning-control fields. Examples: DeepSeek "
            '\'{"thinking": {"type": "enabled"}}\'; Gemini (note the doubled '
            "extra_body key -- required by Gemini's OpenAI-compat layer, "
            "see call_openai_chat_completions_api docstring) "
            '\'{"extra_body": {"google": {"thinking_config": '
            '{"thinking_level": "minimal"}}}}\'.'
        ),
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=5,
        help=(
            "Maximum retry attempts with exponential backoff + jitter on "
            "HTTP 429/5xx for --use-openai and --use-anthropic calls before raising a "
            "recorded hard failure."
        ),
    )
    parser.add_argument(
        "--use-anthropic",
        action="store_true",
        help="Use Anthropic LLM via Messages API instead of PseudoLLM.",
    )
    parser.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-sonnet-4-6",
        help="Anthropic model to use with the Messages API (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--anthropic-max-output-tokens",
        type=int,
        default=8000,
        help="Maximum number of output tokens for Anthropic calls.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Optional sampling temperature for Anthropic Messages API calls. "
            "Default None omits the parameter (provider default 1.0). "
            "When set, passed verbatim (0 is literal 0). Never sets top_p."
        ),
    )
    parser.add_argument(
        "--anthropic-thinking-json",
        type=str,
        default=None,
        help=(
            "Optional JSON object for Anthropic Messages API `thinking`. "
            "Default None omits the field (provider default). "
            'Example: \'{"type": "disabled"}\'.'
        ),
    )
    parser.add_argument(
        "--persona-system-suffix",
        type=str,
        default=None,
        help=(
            "Optional string appended as a final paragraph to the rendered "
            "persona system message. Default None appends nothing. "
            "Text is appended verbatim after a blank line."
        ),
    )
    parser.add_argument(
        "--use-pseudo",
        action="store_true",
        help="Use PseudoLLM for dry-run simulations.",
    )
    parser.add_argument(
        "--persona-prompt-style",
        type=str,
        default="granular_serapio",
        choices=["granular_serapio", "coarse_numeric", "coarse_descriptive"],
        help="Style for BFI persona prompts.",
    )
    parser.add_argument(
        "--persona-prompt-version",
        type=str,
        default=None,
        help="Optional human-readable version label for the persona prompt setup.",
    )
    parser.add_argument(
        "--persona-spec-mode",
        type=str,
        default="lexical_only",
        choices=["lexical_only", "numeric_only", "both"],
        help="Persona specification style: lexical descriptors, numeric only, or both.",
    )
    parser.add_argument(
        "--task-prompt-version",
        type=str,
        default=None,
        help="Optional human-readable version label for the task prompt setup.",
    )
    parser.add_argument(
        "--bfi-ladder-granularity",
        type=int,
        default=9,
        choices=[5, 9],
        help="Lexical ladder granularity for Big Five descriptors.",
    )
    parser.add_argument(
        "--include-user-context",
        action="store_true",
        help="Include extra context block in prompts (off by default).",
    )
    parser.add_argument(
        "--amnesiac-mode",
        action="store_true",
        help=(
            "Disable multi-turn chat memory across the included survey blocks for each agent. "
            "System prompt is still sent each call."
        ),
    )

    args = parser.parse_args()

    personas_filename = (
        "human_personas.csv" if args.human_mode else args.personas_filename
    )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        run_dir_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        slugified_label = slugify_run_label(args.run_label)
        if slugified_label:
            run_dir_name = f"{run_dir_name}_{slugified_label}"
        output_dir = os.path.join("simulation_outputs", run_dir_name)

    config = SimulationConfig(
        personas_filename=personas_filename,
        human_mode=args.human_mode,
        markers_filename=args.markers_filename,
        admin_mode=args.admin_mode,
        shuffle_items_within_tests=args.shuffle_items_within_tests,
        shuffle_test_order=args.shuffle_test_order,
        randomize_reasoning_block=args.randomize_reasoning_block,
        holdout_fraction=args.holdout_fraction,
        max_personas=args.max_personas,
        persona_ids=parse_persona_ids(args.persona_ids),
        sampling_mode=args.sampling_mode,
        time_column=args.time_column,
        random_seed=args.random_seed,
        output_dir=output_dir,
        max_parallel_personas=args.max_parallel_personas,
        use_crt_numeric_only=args.use_crt_numeric_only,
        crt_cutoffs_filename=args.crt_cutoffs_filename,
        use_openai=args.use_openai,
        openai_model=args.openai_model,
        openai_max_output_tokens=args.openai_max_output_tokens,
        openai_base_url=args.openai_base_url,
        openai_key_env=args.openai_key_env,
        openai_api_style=args.openai_api_style,
        openai_extra_params=(
            json.loads(args.openai_extra_params_json)
            if args.openai_extra_params_json
            else {}
        ),
        openai_extra_body=(
            json.loads(args.openai_extra_body_json)
            if args.openai_extra_body_json
            else {}
        ),
        max_api_retries=args.max_api_retries,
        use_anthropic=args.use_anthropic,
        anthropic_model=args.anthropic_model,
        anthropic_max_output_tokens=args.anthropic_max_output_tokens,
        temperature=args.temperature,
        anthropic_thinking=(
            json.loads(args.anthropic_thinking_json)
            if args.anthropic_thinking_json
            else None
        ),
        use_pseudo=args.use_pseudo,
        persona_system_suffix=args.persona_system_suffix,
        persona_prompt_style=args.persona_prompt_style,
        run_label=args.run_label,
        persona_prompt_version=args.persona_prompt_version,
        persona_spec_mode=args.persona_spec_mode,
        task_prompt_version=args.task_prompt_version,
        bfi_ladder_granularity=args.bfi_ladder_granularity,
        include_user_context=args.include_user_context,
        multi_turn_conversation=not args.amnesiac_mode,
        tests_included=args.tests_included or [],
        tests_included_explicitly_set=args.tests_included is not None,
    )
    config.validate()
    return config


# ---------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------


def main() -> None:
    config = parse_cli()
    RUN_COUNTERS.reset()

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    data = load_data(config)

    personas_df, persona_prompts, test_bank = build_network_and_agents(
        data, config
    )

    train_df, val_df = holdout_split(personas_df, config)
    train_prompts = persona_prompts.loc[train_df.index]
    val_prompts = persona_prompts.loc[val_df.index]

    simulator = PersonaSimulator(test_bank=test_bank, config=config)

    param_grid = {
        "likert_noise": [0.3, 0.5, 0.7],
        "crt_slip": [0.1, 0.2, 0.3],
    }
    calibrator = GridSearchCalibrator(
        param_grid=param_grid, random_seed=config.random_seed
    )

    if not config.use_openai and not config.use_anthropic:
        calibrator.fit(train_df, simulator, train_prompts, config)
        if calibrator.best_params_ is None:
            raise RuntimeError(
                "Calibration failed to identify any valid parameter set."
            )
        best_params = calibrator.best_params_
    else:
        best_params = {
            "likert_noise": 0.5,
            "crt_slip": 0.2,
            "random_seed": config.random_seed,
        }
        calibrator.best_params_ = best_params
        calibrator.best_score_ = None

    val_result = simulator.rollout(val_df, best_params, val_prompts)

    evaluator = Evaluator(config)
    metrics = evaluator.compute_metrics(val_result, val_df, train_df)

    save_results(metrics, val_result, config, calibrator)

    # Print the API reliability counters at the end of every run, so the
    # failure rate is visible without opening metrics_and_config.json.
    _counters = RUN_COUNTERS.as_dict()
    if _counters["total_api_calls"] > 0:
        print(
            "\nAPI reliability: "
            f"total_calls={_counters['total_api_calls']} "
            f"retried={_counters['retried_calls']} "
            f"hard_failures={_counters['hard_failures']} "
            f"hard_failure_rate={_counters['hard_failure_rate']:.4%} "
            f"empty_or_unparseable_items={_counters['empty_or_unparseable_items']} "
            f"truncated_responses={_counters['truncated_responses']}",
            flush=True,
        )
        if _counters["hard_failure_rate"] > 0.01:
            print(
                "WARNING: hard_failure_rate exceeds 1% for this run. "
                "Rerun the personas listed in failed_personas.csv before scoring or analysis.",
                flush=True,
            )


if __name__ == "__main__":
    main()