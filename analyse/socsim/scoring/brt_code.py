"""Agent BRT coding for the per-run analysis: the BRT v3 three-seat panel (score/brt_v3).

``code_brt_for_agents`` pivots an agent run's two BRT responses per persona, codes them with
``score/brt_v3/code_brt_v3.py`` and merges the consensus codes and derived variables onto the
person frame. The panel reads and writes its cache in ``agent_brt_cache_dir(root)``:
``$COGSUITE_BRT_CACHE`` if set, else ``<input root>/brt_cache/agent``. With the shipped
adjudication files in that directory the coding makes no API call; any entity without a
matching adjudication is coded live through the simulator's provider adapters
(OPENAI_API_KEY, GEMINI_API_KEY and FIREWORKS_API_KEY).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from socsim.common import require_columns

REPO = Path(__file__).resolve().parents[3]
BRT_V3_DIR = REPO / "score" / "brt_v3"

# Columns the human person file must carry for the human BRT analyses to run.
HUMAN_BRT_REQUIRED_COLUMNS = [
    "BRT1_decision",
    "BRT1_anchor",
    "BRT2_decision",
    "BRT2_anchor",
    "BRT2_research_specificity",
    "BRT1_deposit_salience",
    "BRT1_hygiene_risk_reasoning",
    "BRT1_moral_fairness_loyalty",
    "BRT1_emotional_outburst",
    "BRT2_emotional_reaction",
    "BRT2_social_media_sceptic",
]

# v2.4-category columns from derived.map_v3_to_v24 that the analysis reads under their plain names.
_V24_ALIASES = {
    "v24_BRT2_research_specificity": "BRT2_research_specificity",
    "v24_BRT1_reflective_keep": "BRT1_reflective_keep",
    "v24_BRT2_reflective_accept": "BRT2_reflective_accept",
    "v24_BRT2_high_quality_research": "BRT2_high_quality_research",
}


def validate_human_brt_coded(human_df: pd.DataFrame) -> None:
    require_columns(human_df, HUMAN_BRT_REQUIRED_COLUMNS, "human_person_brt_coded")


def agent_brt_cache_dir(root: Path | str | None = None) -> Path:
    """Agent BRT panel cache: $COGSUITE_BRT_CACHE, else <root>/brt_cache/agent."""
    env = os.environ.get("COGSUITE_BRT_CACHE")
    if env:
        return Path(env)
    return Path(root or os.environ.get("COGSUITE_INPUT_ROOT") or ".") / "brt_cache" / "agent"


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)


def code_brt_for_agents(
    person_df: pd.DataFrame,
    items_df: pd.DataFrame,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Code the agents' BRT responses with the v3 panel and merge the codes onto ``person_df``."""
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

    if str(BRT_V3_DIR) not in sys.path:
        sys.path.insert(0, str(BRT_V3_DIR))
    from code_brt_v3 import code_brt_v3, estimate_spend, prompt_sha  # noqa: E402

    cache_dir = Path(cache_dir) if cache_dir is not None else agent_brt_cache_dir()
    adj_path = cache_dir / f"adjudications_{_safe_label(run_label or 'socsim_agent')}.jsonl"
    print(
        f"[brt_v3] coding {len(rows)} rows cache_dir={cache_dir} adjudications={adj_path.name}",
        file=sys.stderr,
        flush=True,
    )
    consensus_df, _long_df, split_rows, counters = code_brt_v3(
        rows,
        text_col_1="brt_1",
        text_col_2="brt_2",
        id_col="entity_id",
        source_col="source",
        run_label_col="run_label",
        cache_dir=cache_dir,
        adjudications_path=adj_path,
        transport=os.environ.get("BRT_V3_TRANSPORT", "batch"),
    )
    live = estimate_spend(counters)
    print(f"[brt_v3] live_spend_est=${live.get('total', 0.0):.4f}", file=sys.stderr, flush=True)

    coded = consensus_df.copy()
    coded["entity_id"] = coded["entity_id"].astype(str)
    for src, dst in _V24_ALIASES.items():
        if src in coded.columns and dst not in coded.columns:
            coded[dst] = coded[src]
    coded["BRT_llm_reason"] = "ok"
    coded["BRT_scoring_scorer_version"] = "brt_v3"
    coded["BRT_scoring_prompt_version"] = prompt_sha()[:12]
    now = datetime.now(timezone.utc)
    coded["BRT_scoring_run_id"] = f"cr13-brt-{now.strftime('%Y%m%dT%H%M%SZ')}"
    coded["BRT_scoring_date"] = now.date().isoformat()
    agreement_cols = [c for c in coded.columns if c.endswith("_agreement")]
    n_split_entities = int(coded[agreement_cols].eq("split").any(axis=1).sum()) if agreement_cols else 0
    print(
        f"BRT v3 coding complete: {len(coded)}/{len(rows)} coded; "
        f"entities with a split field={n_split_entities} field-splits={len(split_rows)}",
        file=sys.stderr,
        flush=True,
    )
    overlap = [c for c in coded.columns if c in person.columns and c != "entity_id"]
    if overlap:
        person = person.drop(columns=overlap)
    return person.merge(coded, on="entity_id", how="left")
