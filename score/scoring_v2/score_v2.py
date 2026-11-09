"""CRT scorer: deterministic answer keys first, LLM adjudication for residuals, cached.

Every CRT label is produced by :func:`score`; there is no second implementation.

Steps:
  1. Normalise the answer string (``normalise.normalise``).
  2. Apply the answer keys deterministically, in the strict order
     correct -> intuitive -> explicit_other. The first class to match wins and
     no LLM call is made.
  3. Only strings that match no key are residuals. A residual is resolved by
     the LLM adjudicator (three annotators, majority vote), and the result is
     cached per ``(item_id, normalised_string)``, so the cache gives one label
     per distinct answer string.

Label space is ``correct`` / ``intuitive`` / ``other``. Mapping these to the
analysis vocabulary (``correct``, ``intuitive_incorrect``, ``other_incorrect``)
is done downstream, not here.

``method`` is one of:
  ``key_correct``  ``key_intuitive``  ``key_other``  ``llm_majority``
and, only when a residual has neither a cached adjudication nor an adjudicator
(deterministic-only mode), the placeholder ``residual_unadjudicated`` (label
defaults conservatively to ``other``). With the adjudication cache or a live
adjudicator, every residual resolves to ``llm_majority``.

``analyse/socsim/scoring/crt_score_items.py`` and ``score_cli.py`` call this
module.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:  # normal package-relative import
    from .normalise import normalise
except ImportError:  # allow flat-file execution / direct import
    from normalise import normalise  # type: ignore

import yaml

_KEYS_PATH = Path(__file__).resolve().parent / "answer_keys_v2.yaml"
_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "adjudications.jsonl"

_LABELS = ("correct", "intuitive", "other")
_CLASS_ORDER = (
    ("correct", "key_correct"),
    ("intuitive", "key_intuitive"),
    ("explicit_other", "key_other"),
)
# explicit_other resolves to the label "other"
_CLASS_LABEL = {"correct": "correct", "intuitive": "intuitive", "explicit_other": "other"}


def _load_keys() -> Dict[str, dict]:
    with open(_KEYS_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    items: Dict[str, dict] = {}
    for section in ("crt2", "bcrt"):
        for item_id, spec in (raw.get(section) or {}).items():
            compiled = {
                "item_stem": spec.get("item_stem", ""),
                "correct_plain": spec.get("correct_plain", ""),
                "intuitive_plain": spec.get("intuitive_plain", ""),
            }
            for cls in ("correct", "intuitive", "explicit_other"):
                compiled[cls] = [re.compile(p) for p in (spec.get(cls) or [])]
            items[item_id] = compiled
    return items


_KEYS = _load_keys()


def item_ids() -> List[str]:
    """The eight keyed CRT item ids, in file order."""
    return list(_KEYS.keys())


def item_meta(item_id: str) -> Dict[str, str]:
    """Return {item_stem, correct_plain, intuitive_plain} for an item (for the
    adjudicator prompt). Empty dict if the item is unknown."""
    spec = _KEYS.get(item_id)
    if not spec:
        return {}
    return {
        "item_stem": spec["item_stem"],
        "correct_plain": spec["correct_plain"],
        "intuitive_plain": spec["intuitive_plain"],
    }


def classify_deterministic(item_id: str, normalised: str) -> Dict[str, Optional[str]]:
    """Apply the answer keys in strict order to an already-normalised string.

    Returns ``{"label": <correct|intuitive|other>, "method": <key_*>}`` on a key
    hit, or ``{"label": None, "method": None}`` for a residual (no key matched).
    The input MUST already be normalised (callers use :func:`normalise`).
    """
    spec = _KEYS.get(item_id)
    if spec is None:
        # No answer key for this item (e.g. dcrt_*): everything is residual.
        return {"label": None, "method": None}
    for cls, method in _CLASS_ORDER:
        for pat in spec[cls]:
            if pat.search(normalised):
                return {"label": _CLASS_LABEL[cls], "method": method}
    return {"label": None, "method": None}


# --------------------------------------------------------------------------- #
# Adjudication cache: (item_id, normalised_string) -> {label, votes}
# --------------------------------------------------------------------------- #
def load_cache(path: Path = _CACHE_PATH) -> Dict[tuple, dict]:
    """Load the content-addressed adjudication cache. Missing file -> empty."""
    cache: Dict[tuple, dict] = {}
    if not Path(path).exists():
        return cache
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["item_id"], rec["normalised"])
            cache[key] = rec
    return cache


def _majority(votes: List[str]) -> Optional[str]:
    """Majority label over a list of votes; None on no strict majority."""
    counts = {lab: 0 for lab in _LABELS}
    for v in votes:
        if v in counts:
            counts[v] += 1
    best = max(counts, key=lambda k: counts[k])
    top = counts[best]
    if top == 0:
        return None
    if sum(1 for k in counts if counts[k] == top) > 1:
        return None  # tie -> no majority
    return best


def score(
    item_id: str,
    raw_answer: object,
    adjudicator: Optional[Callable[[str, str], List[str]]] = None,
    cache: Optional[Dict[tuple, dict]] = None,
) -> Dict[str, object]:
    """Score a single raw answer. The single scoring entry point.

    Parameters
    ----------
    item_id : str
    raw_answer : object
        The raw response string (or NaN/None); normalised internally.
    adjudicator : callable, optional
        ``adjudicator(item_id, normalised) -> [label, label, label]`` — the
        three-provider vote. If omitted, residuals not in the cache are left
        unadjudicated (deterministic-only mode).
    cache : dict, optional
        ``{(item_id, normalised): record}`` from :func:`load_cache`. A cache hit
        short-circuits the adjudicator so one string is never voted twice.

    Returns
    -------
    dict : ``{"label", "method", "votes", "normalised"}``
        ``method`` ∈ {key_correct, key_intuitive, key_other, llm_majority,
        residual_unadjudicated}. ``votes`` is the list of annotator votes for
        llm_majority (else []).
    """
    normalised = normalise(raw_answer)
    det = classify_deterministic(item_id, normalised)
    if det["label"] is not None:
        return {"label": det["label"], "method": det["method"], "votes": [], "normalised": normalised}

    # Residual: the content-addressed cache is consulted first, so a string has one label.
    key = (item_id, normalised)
    if cache is not None and key in cache:
        rec = cache[key]
        return {
            "label": rec["label"],
            "method": "llm_majority",
            "votes": rec.get("votes", []),
            "normalised": normalised,
        }

    if adjudicator is not None:
        votes = list(adjudicator(item_id, normalised))
        label = _majority(votes)
        if label is None:
            label = "other"  # no majority -> other (conservative)
        return {"label": label, "method": "llm_majority", "votes": votes, "normalised": normalised}

    # Deterministic-only mode: leave the residual unadjudicated, default
    # conservatively to "other" and flag it via method.
    return {"label": "other", "method": "residual_unadjudicated", "votes": [], "normalised": normalised}
