"""Unit tests for scoring_v2 — the strict evaluation order (JZ-RESCORE.md §3).

Order under test: normalise → correct → intuitive → explicit_other → LLM residual.
All assertions run in deterministic-only mode (no adjudicator, no network): a
residual therefore resolves to label "other" with method ``residual_unadjudicated``.

Run:  python -m pytest scoring_v2/test_score_v2.py -q
  or: python scoring_v2/test_score_v2.py
Any failure STOPS the Jz-A job.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalise import normalise  # noqa: E402
from score_v2 import (  # noqa: E402
    classify_deterministic,
    load_cache,
    score,
)


def _label(item_id: str, raw: str) -> str:
    return score(item_id, raw)["label"]


def _method(item_id: str, raw: str) -> str:
    return score(item_id, raw)["method"]


# --------------------------------------------------------------------------- #
# The explicit cases mandated by JZ-RESCORE.md §3 STEP 3.
# --------------------------------------------------------------------------- #
def test_crt2_q1_correct_forms():
    for s in ("second", "2nd", "second place", "2"):
        assert _label("crt2_q1", s) == "correct", s
        assert _method("crt2_q1", s) == "key_correct", s


def test_crt2_q1_intuitive_forms():
    for s in ("first", "1st"):
        assert _label("crt2_q1", s) == "intuitive", s
        assert _method("crt2_q1", s) == "key_intuitive", s


def test_crt2_q4_zero_cubic_feet_is_correct():
    assert _label("crt2_q4", "0 cubic feet") == "correct"


def test_crt2_q4_nine_is_other_not_correct():
    # v1 stored "9" as correct on this item (D-50); v2 must NOT.
    assert _label("crt2_q4", "9") == "other"
    assert _method("crt2_q4", "9") == "residual_unadjudicated"


def test_bcrt_q3_bare_zero_is_correct():
    assert _label("bcrt_q3", "0") == "correct"


def test_bcrt_q3_four_is_intuitive():
    assert _label("bcrt_q3", "4") == "intuitive"


def test_bcrt_q3_three_hours_is_other():
    assert _label("bcrt_q3", "3 hours") == "other"
    assert _method("bcrt_q3", "3 hours") == "key_other"  # explicit_other


def test_bcrt_q4_no_is_other_not_intuitive():
    # The one the old classifier got wrong: "no" is other_incorrect (D-50g).
    assert _label("bcrt_q4", "no") == "other"
    assert _method("bcrt_q4", "no") == "key_other"


def test_bcrt_q4_dontknow_family_is_intuitive():
    for s in ("no idea", "not sure"):
        assert _label("bcrt_q4", s) == "intuitive", s
        assert _method("bcrt_q4", s) == "key_intuitive", s


def test_bcrt_q4_yes_is_correct():
    assert _label("bcrt_q4", "yes") == "correct"


# --------------------------------------------------------------------------- #
# Order guarantees: intuitive must be checked BEFORE explicit_other, so the
# lure family is never swallowed by `^no$`.
# --------------------------------------------------------------------------- #
def test_bcrt_q4_order_no_idea_beats_no():
    assert _label("bcrt_q4", "no idea") == "intuitive"
    assert _label("bcrt_q4", "no.") == "other"       # trailing "." normalised away
    assert _label("bcrt_q4", "not enough information") == "intuitive"


def test_crt2_q4_twentyseven_is_intuitive():
    assert _label("crt2_q4", "27") == "intuitive"
    assert _label("crt2_q4", "27 cubic feet") == "intuitive"


# --------------------------------------------------------------------------- #
# Normalisation rule (JZ-RESCORE.md §1).
# --------------------------------------------------------------------------- #
def test_normalisation_rule():
    assert normalise("  Second Place. ") == "second place"
    assert normalise("$400") == "400"
    assert normalise("Yes!") == "yes"
    assert normalise("can’t") == "can't"   # curly apostrophe -> straight
    assert normalise("no   idea") == "no idea"  # whitespace run collapsed
    assert normalise(None) == ""
    # curly-apostrophe forms must reach the bcrt_q4 lure family after normalise
    assert _label("bcrt_q4", "I can’t tell") == "intuitive"


# --------------------------------------------------------------------------- #
# Structural fix for D-50: the cache makes one string -> one label, and a cache
# hit short-circuits (would-be) adjudication.
# --------------------------------------------------------------------------- #
def test_cache_hit_short_circuits_residual():
    # "banana" matches no crt2_q4 key -> residual. With a cache entry it must
    # return the cached label deterministically (method llm_majority).
    fake_cache = {
        ("crt2_q4", "banana"): {
            "item_id": "crt2_q4",
            "normalised": "banana",
            "label": "other",
            "votes": ["other", "other", "correct"],
        }
    }
    out = score("crt2_q4", "banana", cache=fake_cache)
    assert out["label"] == "other"
    assert out["method"] == "llm_majority"
    assert out["votes"] == ["other", "other", "correct"]


def test_adjudicator_majority_and_tiebreak():
    # majority
    out = score("crt2_q4", "somethingweird", adjudicator=lambda i, n: ["correct", "correct", "other"])
    assert out["label"] == "correct" and out["method"] == "llm_majority"
    # all-differ -> no majority -> other (conservative, D-53)
    out = score("crt2_q4", "somethingweird2", adjudicator=lambda i, n: ["correct", "intuitive", "other"])
    assert out["label"] == "other" and out["method"] == "llm_majority"


def test_deterministic_never_residual_for_keyed_hits():
    # classify_deterministic returns None only for genuine residuals.
    assert classify_deterministic("crt2_q1", "second")["label"] == "correct"
    assert classify_deterministic("crt2_q4", "9")["label"] is None
    assert load_cache.__name__ == "load_cache"  # module import smoke


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def _run_all() -> int:
    failures = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e!r}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
