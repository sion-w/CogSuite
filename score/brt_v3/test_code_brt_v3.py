"""Tests for the BRT v3 panel coder: prompt split, consensus, cache keys, resume (no API calls).

The seats are never called: ``evaluate_one`` is stubbed wherever an entity would need coding,
and the cache-only test unsets every API key. Fixtures are toy inputs in ``tmp_path``.

Run:  python -m pytest score/brt_v3 -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import code_brt_v3 as coder  # noqa: E402
from schema import BOOL_FIELDS, CODING_FIELDS, STRING_FIELDS  # noqa: E402
from seat_transport import split_rubric_and_entity  # noqa: E402

PROMPT_SHA = "472327bebe7616a0e3642ea3773490c87b95dd8f506e44d6eaa9d16050f1da4f"


# --------------------------------------------------------------------------- #
# Prompt and cache keys
# --------------------------------------------------------------------------- #
def test_prompt_file_sha():
    """The shipped agent cache is stamped with this prompt's SHA-256."""
    assert coder.prompt_sha() == PROMPT_SHA


def test_rubric_system_has_no_entity_placeholders():
    system, user = coder.load_send_parts()
    assert "{brt_1}" not in system and "{brt_2}" not in system
    assert "{brt_1}" in user and "{brt_2}" in user
    assert "BRT1_decision" in system
    assert "Global rules" in system
    assert user.strip().startswith("BRT1_response:")


def test_file_markers():
    raw = coder.PROMPT_PATH.read_text(encoding="utf-8")
    assert raw.startswith("SYSTEM\n")
    assert "\nUSER\n" in raw
    system_file, user_file = coder.load_prompt_parts()
    rubric, entity = split_rubric_and_entity(system_file, user_file)
    assert system_file in rubric
    assert entity.endswith('BRT2_response: """{brt_2}"""')


def test_cache_path_schema(tmp_path):
    p = coder._cache_path("abc", "gpt-5.6-sol@reasoning_effort=low", tmp_path)
    assert p.parent == tmp_path / "per_call"
    assert p.name == f"abc__gpt-5.6-sol@reasoning_effort=low__{PROMPT_SHA[:12]}.json"


def test_cache_stamp_mismatch_stops(tmp_path):
    (tmp_path / coder.CACHE_STAMP_NAME).write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="stamp mismatch"):
        coder.assert_cache_version(tmp_path)


def test_simulator_adapter_path_is_in_repo():
    assert coder.SIMULATOR_PATH.exists()
    assert coder.REPO in coder.SIMULATOR_PATH.parents


# --------------------------------------------------------------------------- #
# Consensus
# --------------------------------------------------------------------------- #
def _vote(*vals):
    return {"labels": {f: vals[i] if i < 4 else 0 for i, f in enumerate(CODING_FIELDS)}}


def test_majority_unanimous_and_split():
    keep = ["KEEP_BOOKING", "ANCHOR_NONE", "TAKE_MEDEX", "ANCHOR_NONE"] + [0] * 24
    consensus, agreement, splits = coder._majority_per_field([_vote(*keep) for _ in range(3)])
    assert consensus["BRT1_decision"] == "KEEP_BOOKING"
    assert agreement["BRT1_decision"] == "unanimous"
    assert splits == []

    a = _vote(*keep)
    b = _vote(*(["CHANGE_RESTAURANT"] + keep[1:]))
    c = _vote(*(["CANCEL_OR_POSTPONE"] + keep[1:]))
    consensus, agreement, splits = coder._majority_per_field([a, b, c])
    assert splits == ["BRT1_decision"]
    assert agreement["BRT1_decision"] == "split"
    assert consensus["BRT1_decision"] == "UNCLEAR_OR_CONDITIONAL"


def test_majority_two_of_three():
    keep = ["KEEP_BOOKING", "ANCHOR_NONE", "TAKE_MEDEX", "ANCHOR_NONE"] + [0] * 24
    flag = {"labels": dict(zip(CODING_FIELDS, keep))}
    flag["labels"][CODING_FIELDS[4]] = 1
    consensus, agreement, _ = coder._majority_per_field([flag, flag, _vote(*keep)])
    assert consensus[CODING_FIELDS[4]] == 1
    assert agreement[CODING_FIELDS[4]] == "majority"


def test_evidence_must_occur_in_text():
    payload = {f: sorted(v)[0] for f, v in STRING_FIELDS.items()}
    payload.update({f: 0 for f in BOOL_FIELDS})
    payload["BRT1_decision"] = "KEEP_BOOKING"
    payload["BRT1_decision_evidence"] = "we keep the booking"
    _, _, problems = coder._validate_and_fix_evidence(payload, "We will keep it.", "Take it.")
    assert "evidence_not_in_text:BRT1_decision" in problems
    payload["BRT1_decision_evidence"] = "keep it"
    _, _, problems = coder._validate_and_fix_evidence(payload, "We will keep it.", "Take it.")
    assert "evidence_not_in_text:BRT1_decision" not in problems


# --------------------------------------------------------------------------- #
# Resume keyed on (entity_id, scenario_pair_hash)
# --------------------------------------------------------------------------- #
def _labels() -> dict:
    out = {f: sorted(v)[0] for f, v in STRING_FIELDS.items()}
    out.update({f: 0 for f in BOOL_FIELDS})
    return out


def _rows(texts: dict[str, tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"entity_id": eid, "brt_1": b1, "brt_2": b2, "source": "agent", "run_label": "T"} for eid, (b1, b2) in texts.items()]
    )


def _adj_line(eid: str, b1: str, b2: str, marker: str = "stored") -> dict:
    """A stored adjudication line, shaped like the ones code_brt_v3 writes."""
    labels = _labels()
    votes = [
        {
            "model_id": e["model_id"],
            "evaluator_id": e["evaluator_id"],
            "setting": e["setting"],
            "labels": labels,
            "evidence": {f: "NONE" for f in CODING_FIELDS},
        }
        for e in coder.EVALUATORS
    ]
    return {
        "entity_id": eid,
        "source": "agent",
        "run_label": marker,  # marks which line was picked
        "scenario_pair_hash": coder.scenario_pair_hash(b1, b2),
        "votes": votes,
        "consensus": labels,
        "agreement_by_field": {f: "unanimous" for f in CODING_FIELDS},
        "split_fields": [],
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def _write(path: Path, lines: list[dict]) -> None:
    path.write_text("".join(json.dumps(d) + "\n" for d in lines), encoding="utf-8")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Stub the network seams; record which (brt_1, brt_2) pairs would have been sent to a seat."""
    calls: list[tuple[str, str]] = []

    def fake_evaluate_one(ev, system, user_tmpl, b1, b2, counters, use_cache, cache_dir):
        calls.append((b1, b2))
        return {
            "model_id": ev["model_id"],
            "evaluator_id": ev["evaluator_id"],
            "setting": ev["setting"],
            "labels": _labels(),
            "evidence": {f: "NONE" for f in CODING_FIELDS},
        }

    def no_adapter():
        raise AssertionError("the simulator adapter must not be loaded in these tests")

    monkeypatch.setattr(coder, "verify_keys", lambda: None)
    monkeypatch.setattr(coder, "load_simulator_adapter", no_adapter)
    monkeypatch.setattr(coder, "evaluate_one", fake_evaluate_one)

    cache = tmp_path / "cache"
    adj = cache / "adjudications_T.jsonl"
    cache.mkdir()

    def run(rows: pd.DataFrame, **kw):
        return coder.code_brt_v3(rows, cache_dir=cache, adjudications_path=adj, transport="sync", **kw)

    return run, calls, adj


TEXTS = {"1": ("a1", "a2"), "2": ("b1", "b2"), "3": ("c1", "c2")}


def test_cache_only_run_needs_no_keys(tmp_path, monkeypatch):
    """Every entity adjudicated on its current text: no key check, no adapter, no call."""
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(coder, "load_simulator_adapter", lambda: (_ for _ in ()).throw(AssertionError("adapter loaded")))
    cache = tmp_path / "cache"
    adj = cache / "adjudications_T.jsonl"
    cache.mkdir()
    _write(adj, [_adj_line(e, *t) for e, t in TEXTS.items()])
    consensus, long_df, _splits, counters = coder.code_brt_v3(_rows(TEXTS), cache_dir=cache, adjudications_path=adj)
    assert sorted(consensus["entity_id"]) == ["1", "2", "3"]
    assert len(long_df) == 3 * 3 * len(CODING_FIELDS)
    assert all(c.calls == 0 for c in counters.values())
    assert "BRT2_research_quality_v3" in consensus.columns


def test_missing_keys_stop_before_any_call(tmp_path, monkeypatch):
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(coder, "REPO", tmp_path)  # no .env to load
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(SystemExit, match="API keys are not set"):
        coder.code_brt_v3(_rows(TEXTS), cache_dir=cache, adjudications_path=cache / "adj.jsonl")


def test_unchanged_text_is_skipped(harness):
    run, calls, adj = harness
    _write(adj, [_adj_line(e, *t) for e, t in TEXTS.items()])
    before = adj.read_text(encoding="utf-8")
    consensus, *_ = run(_rows(TEXTS))
    assert calls == []
    assert adj.read_text(encoding="utf-8") == before
    assert not adj.with_name(adj.name + ".superseded.jsonl").exists()
    assert sorted(consensus["entity_id"]) == ["1", "2", "3"]
    assert set(consensus["run_label"]) == {"stored"}


def test_only_new_entities_are_coded(harness):
    run, calls, adj = harness
    _write(adj, [_adj_line(e, *t) for e, t in list(TEXTS.items())[:2]])
    consensus, *_ = run(_rows(TEXTS))
    assert set(calls) == {TEXTS["3"]}
    assert len(_read(adj)) == 3
    assert sorted(consensus["entity_id"]) == ["1", "2", "3"]


def test_changed_text_stops_by_default_and_recodes_nothing(harness):
    run, calls, adj = harness
    _write(adj, [_adj_line(e, *t) for e, t in TEXTS.items()])
    before = adj.read_text(encoding="utf-8")
    changed = dict(TEXTS, **{"2": ("b1 X", "b2"), "3": ("c1", "c2 X")})
    with pytest.raises(SystemExit) as exc:
        run(_rows(changed))
    msg = str(exc.value)
    assert msg.startswith("STOP")
    assert "2, 3" in msg and "--recode-changed-text" in msg
    assert calls == []
    assert adj.read_text(encoding="utf-8") == before


def test_changed_text_is_recoded_and_superseded_with_flag(harness):
    run, calls, adj = harness
    stored = [_adj_line(e, *t) for e, t in TEXTS.items()]
    _write(adj, stored)
    changed = dict(TEXTS, **{"2": ("b1 X", "b2 X")})
    consensus, *_ = run(_rows(changed), recode_changed_text=True)
    assert set(calls) == {("b1 X", "b2 X")}
    sup = adj.with_name(adj.name + ".superseded.jsonl")
    assert _read(sup) == [stored[1]]
    live = {line["entity_id"]: line for line in _read(adj)}
    assert sorted(live) == ["1", "2", "3"]
    assert live["2"]["scenario_pair_hash"] == coder.scenario_pair_hash("b1 X", "b2 X")
    assert live["1"]["run_label"] == live["3"]["run_label"] == "stored"
    assert live["2"]["run_label"] == "T"
    assert len(consensus) == 3
    calls.clear()
    run(_rows(changed), recode_changed_text=True)
    assert calls == []


def test_line_without_a_recorded_hash_counts_as_changed(harness):
    run, calls, adj = harness
    line = _adj_line("1", "a1", "a2")
    del line["scenario_pair_hash"]
    _write(adj, [line])
    with pytest.raises(SystemExit, match="STOP"):
        run(_rows({"1": ("a1", "a2")}))
    assert calls == []


def test_rebuild_picks_the_line_whose_hash_matches_current_text(tmp_path):
    adj = tmp_path / "adj.jsonl"
    a = _adj_line("7", "x1", "x2", marker="other")
    b = _adj_line("7", "y1", "y2", marker="current")
    c = _adj_line("8", "z1", "z2", marker="only")
    cur = {"7": coder.scenario_pair_hash("y1", "y2"), "8": coder.scenario_pair_hash("z1", "z2")}
    for order in ([a, b, c], [b, a, c]):
        _write(adj, order)
        consensus, long_rows, _ = coder._rebuild_from_adjudications(adj, cur)
        got = {r["entity_id"]: r for r in consensus}
        assert sorted(got) == ["7", "8"]
        assert got["7"]["run_label"] == "current"
        assert {r["scenario_pair_hash"] for r in long_rows if r["entity_id"] == "7"} == {cur["7"]}


def test_rebuild_stops_when_no_line_matches_current_text(tmp_path):
    adj = tmp_path / "adj.jsonl"
    _write(adj, [_adj_line("7", "x1", "x2"), _adj_line("7", "y1", "y2")])
    with pytest.raises(SystemExit, match="no adjudication line for entity_id 7"):
        coder._rebuild_from_adjudications(adj, {"7": coder.scenario_pair_hash("z1", "z2")})


def test_same_entity_id_with_two_texts_in_the_input_stops(harness):
    run, calls, _adj = harness
    rows = pd.concat([_rows({"1": ("a1", "a2")}), _rows({"1": ("other", "text")})], ignore_index=True)
    with pytest.raises(SystemExit, match="two different BRT texts"):
        run(rows)
    assert calls == []
