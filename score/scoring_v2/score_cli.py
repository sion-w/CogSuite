"""Batch CLI for the CRT scorer: score every CRT2 and bCRT answer in a simulator per_item_results.csv.

Reads a per_item_results.csv (columns persona_id, item_id, response_text), keeps the eight CRT2/bCRT items, and scores
each answer with score_v2: the answer is normalised, the answer keys are applied in the order correct -> intuitive ->
explicit_other, and an answer no key matches (a residual) is resolved from the adjudication cache, keyed on
(item_id, normalised answer). Residuals missing from the cache are adjudicated live (three annotator models, majority
vote, appended to the cache) only when --allow-api is given and every annotator's API key is set; otherwise they are
written with method residual_unadjudicated (label other) and listed on stderr. Writes one row per answer to --out with
columns source, run_label, entity_id, item_id, raw_response, normalised, label_v2, method, agreement.

response_text is read as typed: only an empty cell is a blank (unscored) answer. Words such as "None" are answers
("none" is the correct answer to CRT2 item 4) and are scored like any other text.

Example:
    python score/scoring_v2/score_cli.py --items simulation_outputs/<RUN>/per_item_results.csv --out crt_long.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_v2 import item_ids, item_meta, load_cache, score  # noqa: E402

DEFAULT_CACHE = Path(__file__).resolve().parent / "cache" / "adjudications.jsonl"

PROMPT_PATH = Path(__file__).resolve().parent / "evaluator_prompt_v2.txt"
OUTPUT_COLUMNS = ["source", "run_label", "entity_id", "item_id", "raw_response", "normalised", "label_v2", "method",
                  "agreement"]
LABELS = ("correct", "intuitive", "other")

# The three annotators of the residual vote: model, endpoint, and the reasoning setting recorded with each vote.
ANNOTATORS = [
    {"model_id": "gpt-5.6-sol", "api_style": "responses", "base_url": None, "key_env": "OPENAI_API_KEY",
     "extra_params": {"reasoning": {"effort": "low"}}, "extra_body": None, "setting": 'reasoning_effort="low"'},
    {"model_id": "gemini-3.5-flash", "api_style": "chat_completions",
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY",
     "extra_params": {}, "extra_body": {"extra_body": {"google": {"thinking_config": {"thinking_level": "minimal"}}}},
     "setting": 'thinking_level="minimal"'},
    {"model_id": os.environ.get("COGSUITE_DEEPSEEK_SEAT_MODEL", "accounts/fireworks/models/deepseek-v4-flash-0731"), "api_style": "chat_completions",
     "base_url": "https://api.fireworks.ai/inference/v1", "key_env": "FIREWORKS_API_KEY", "extra_params": {},
     "extra_body": {"thinking": {"type": "disabled"}}, "setting": 'thinking={"type":"disabled"}'},
]


# --------------------------------------------------------------------------- reading and scoring
def read_crt_items(path: Path) -> pd.DataFrame:
    """The CRT2/bCRT rows of a per_item_results.csv, in file order, with response_text blank-filled."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"persona_id", "item_id", "response_text"} - set(raw.columns)
    if missing:
        raise SystemExit(f"STOP: {path} lacks columns {sorted(missing)}.")
    raw = raw[raw["item_id"].isin(item_ids())].copy()
    raw["response_text"] = raw["response_text"].fillna("")
    return raw.reset_index(drop=True)


def score_items(items: pd.DataFrame, cache: Dict[tuple, dict], source: str, run_label: str) -> pd.DataFrame:
    """Score every row of read_crt_items(); agreement is the cached vote agreement for adjudicated residuals."""
    records = []
    for r in items.itertuples(index=False):
        out = score(r.item_id, r.response_text, adjudicator=None, cache=cache)
        rec = cache.get((r.item_id, out["normalised"])) if out["method"] == "llm_majority" else None
        records.append({"source": source, "run_label": run_label, "entity_id": str(r.persona_id),
                        "item_id": r.item_id, "raw_response": r.response_text, "normalised": out["normalised"],
                        "label_v2": out["label"], "method": out["method"],
                        "agreement": rec.get("agreement", "") if rec else ""})
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def unadjudicated(scored: pd.DataFrame) -> List[tuple]:
    """Distinct (item_id, normalised) residuals with no cached adjudication."""
    res = scored[scored["method"] == "residual_unadjudicated"]
    return sorted(set(zip(res["item_id"], res["normalised"])))


# --------------------------------------------------------------------------- live adjudication
def _prompt_parts() -> tuple:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    if not text.startswith("SYSTEM\n") or "\nUSER\n" not in text:
        raise SystemExit(f"STOP: {PROMPT_PATH.name} must hold a SYSTEM section followed by a USER section.")
    system, user = text[len("SYSTEM\n"):].split("\nUSER\n", 1)
    return system.strip(), user.strip()


def _parse_label(text: str) -> str:
    s = (text or "").strip()
    m = re.search(r"\{[^{}]*\}", s, flags=re.DOTALL)
    if m:
        try:
            lab = str(json.loads(m.group(0)).get("label", "")).strip().lower()
        except (ValueError, AttributeError):
            lab = ""
        lab = {"other_incorrect": "other", "intuitive_incorrect": "intuitive"}.get(lab, lab)
        if lab in LABELS:
            return lab
    low = s.lower()
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", low):
            return lab
    raise ValueError(f"no label in annotator output: {text!r}")


def _call(ann: dict, system: str, user: str, max_retries: int = 5) -> str:
    from openai import OpenAI

    kwargs = {"api_key": os.environ[ann["key_env"]]}
    if ann["base_url"]:
        kwargs["base_url"] = ann["base_url"]
    client = OpenAI(**kwargs)
    for attempt in range(max_retries + 1):
        try:
            if ann["api_style"] == "responses":
                resp = client.responses.create(
                    model=ann["model_id"], max_output_tokens=256, **ann["extra_params"],
                    input=[{"role": "system", "content": [{"type": "input_text", "text": system}]},
                           {"role": "user", "content": [{"type": "input_text", "text": user}]}])
                return resp.output_text
            extra = {"extra_body": ann["extra_body"]} if ann["extra_body"] else {}
            resp = client.chat.completions.create(
                model=ann["model_id"], max_tokens=256, **ann["extra_params"], **extra,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return resp.choices[0].message.content or ""
        except Exception:  # transient provider errors: bounded exponential backoff, then re-raise
            if attempt == max_retries:
                raise
            time.sleep(min(60.0, 2.0 ** attempt))
    return ""


def vote_agreement(votes: List[str]) -> tuple:
    """(label, agreement): three-way agreement is unanimous, two-way majority, otherwise split -> other."""
    counts = {lab: votes.count(lab) for lab in LABELS}
    top = max(counts.values())
    leaders = [lab for lab in LABELS if counts[lab] == top]
    if top == len(votes):
        return leaders[0], "unanimous"
    if top >= 2 and len(leaders) == 1:
        return leaders[0], "majority"
    return "other", "split"


def live_adjudicator() -> Optional[Callable[[str, str], dict]]:
    """A callable (item_id, normalised) -> cache record, or None when an annotator's API key is unset."""
    if not all(os.environ.get(a["key_env"]) for a in ANNOTATORS):
        return None
    system, user_tmpl = _prompt_parts()

    def adjudicate(item_id: str, normalised: str) -> dict:
        meta = item_meta(item_id)
        user = (user_tmpl.replace("{item_stem}", meta.get("item_stem", ""))
                .replace("{correct_plain}", meta.get("correct_plain", ""))
                .replace("{intuitive_plain}", meta.get("intuitive_plain", ""))
                .replace("{normalised_answer}", normalised if normalised else "(blank)"))
        votes = [{"model_id": a["model_id"], "setting": a["setting"], "label": _parse_label(_call(a, system, user))}
                 for a in ANNOTATORS]
        label, agreement = vote_agreement([v["label"] for v in votes])
        return {"item_id": item_id, "normalised": normalised, "votes": votes, "label": label, "agreement": agreement}

    return adjudicate


def adjudicate_into_cache(pairs: List[tuple], cache: Dict[tuple, dict], cache_path: Path,
                          adjudicator: Callable[[str, str], dict]) -> int:
    """Adjudicate each residual pair, appending one record per pair to the cache file and the in-memory cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as fh:
        for item_id, normalised in pairs:
            rec = adjudicator(item_id, normalised)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            cache[(item_id, normalised)] = rec
    return len(pairs)


def score_file(items_path: Path, cache_path: Path, source: str, run_label: str, allow_api: bool) -> pd.DataFrame:
    """Score one per_item_results.csv; with allow_api, adjudicate uncached residuals and rescore."""
    cache = load_cache(cache_path)
    items = read_crt_items(items_path)
    scored = score_items(items, cache, source, run_label)
    pending = unadjudicated(scored)
    if pending and allow_api:
        adjudicator = live_adjudicator()
        if adjudicator is None:
            print("API keys for the three annotators are not all set; residuals stay unadjudicated.", file=sys.stderr)
        else:
            adjudicate_into_cache(pending, cache, cache_path, adjudicator)
            scored = score_items(items, cache, source, run_label)
            pending = unadjudicated(scored)
    if pending:
        print(f"{len(pending)} distinct residual answer(s) unadjudicated in {items_path} "
              f"(method residual_unadjudicated, label other):", file=sys.stderr)
        for item_id, normalised in pending:
            print(f"  {item_id}\t{normalised!r}", file=sys.stderr)
    return scored


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", type=Path, required=True, help="per_item_results.csv to score")
    ap.add_argument("--out", type=Path, required=True, help="scored long CSV to write")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="adjudication cache (JSONL)")
    ap.add_argument("--source", default=None, help="source column value (default: the run folder name)")
    ap.add_argument("--run-label", default=None, help="run_label column value (default: the run folder name)")
    ap.add_argument("--allow-api", action="store_true",
                    help="adjudicate uncached residuals live (needs OPENAI_API_KEY, GEMINI_API_KEY, FIREWORKS_API_KEY)")
    args = ap.parse_args(argv)
    run = args.items.resolve().parent.name
    scored = score_file(args.items, args.cache, args.source or run, args.run_label or run, args.allow_api)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(scored)} rows; methods {scored['method'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
