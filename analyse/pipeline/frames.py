"""The human analytic frame and the agent runs' person frames.

Human frame (473 people): the human BRT labels (``reports/brt_labels/brt_labels_v3_human.csv``) joined on
the person ID to the trait scores of the human dataset (``data/inputs/person_with_liwc_brt_coded.csv``).
Every derived BRT column of the label file is first re-derived from the coded fields with
``score/brt_v3/derived.py`` and must agree with the shipped column. CRT2_Total_adj is the
familiarity-adjusted CRT2 score: CRT2_Total minus the fixed familiarity coefficient times the number of the
four CRT2 items the person had seen before.

Agent frames: ``reports/cr13_<run>/tables/by_dataset/socsim_agent/00_person_frame.csv``, the per-run person
frame the analysis pipeline writes (personas with no usable open-ended response carry no BRT value).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import BRT_LABELS_HUMAN, PERSON_FILE, require, run_dir, stop  # noqa: F401

DEMO = {"age": "Age", "gender": "sex", "education": "Highest education level completed"}
CRT2_SEEN_COLS = [f"Experience Check_{i}_seen" for i in range(1, 5)]


def add_crt2_familiarity(traits: pd.DataFrame) -> None:
    """Add CRT2_SeenCount (CRT2 items seen before, of four) and CRT2_Total_adj, in place."""
    from socsim.ingest.harmonise_human import CRT2_FAM_BETA_FIXED
    missing = [c for c in CRT2_SEEN_COLS if c not in traits.columns]
    if missing:
        stop(f"the person file lacks the CRT2 familiarity columns {missing}")
    traits["CRT2_SeenCount"] = traits[CRT2_SEEN_COLS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    traits["CRT2_Total_adj"] = (pd.to_numeric(traits["CRT2_Total"], errors="coerce")
                                - CRT2_FAM_BETA_FIXED * pd.to_numeric(traits["CRT2_SeenCount"], errors="coerce"))


def check_derived(labels: pd.DataFrame) -> None:
    """Every derived column in the label file equals the value derived.py computes from the coded fields."""
    import derived  # score/brt_v3, on sys.path via common
    redone = derived.map_v3_to_v24(derived.compute_v3_derived(labels.copy()))
    mismatches = {}
    for col in labels.columns:
        if col in redone.columns and col not in ("entity_id", "source", "run_label", "scenario_pair_hash"):
            a, b = labels[col], redone[col]
            if a.dtype.kind in "fi" or b.dtype.kind in "fi":
                diff = ~((pd.to_numeric(a, errors="coerce").fillna(-9e9))
                         == (pd.to_numeric(b, errors="coerce").fillna(-9e9)))
                if a.dtype == object and b.dtype == object:
                    diff = a.astype("string").fillna("<NA>") != b.astype("string").fillna("<NA>")
            else:
                diff = a.astype("string").fillna("<NA>") != b.astype("string").fillna("<NA>")
            if int(diff.sum()):
                mismatches[col] = int(diff.sum())
    if mismatches:
        stop(f"derived BRT columns in the label file disagree with derived.py: {mismatches}")


def human_frame(root: Path) -> pd.DataFrame:
    """The 473-person human frame: BRT labels, trait scores, response length and model covariates."""
    labels = pd.read_csv(require(root / BRT_LABELS_HUMAN, "human BRT labels (controlled-access human dataset)"),
                         low_memory=False)
    check_derived(labels)
    traits = pd.read_csv(require(root / PERSON_FILE, "human person file (controlled-access human dataset)"),
                         low_memory=False)
    add_crt2_familiarity(traits)

    keep = ["PROLIFIC_PID", "BCRT_Total", "CRT2_Total", "CRT2_Total_adj", "CRT2_SeenCount",
            "NFC_Total", "Country", DEMO["age"], DEMO["gender"], DEMO["education"], "WC"]
    keep = [c for c in keep if c in traits.columns]
    tr = traits[keep].copy().rename(columns={"PROLIFIC_PID": "entity_id"})
    tr["entity_id"] = tr["entity_id"].astype(str)
    tr = tr.rename(columns={DEMO["age"]: "age", DEMO["gender"]: "gender", DEMO["education"]: "education"})

    labels["entity_id"] = labels["entity_id"].astype(str)
    df = labels.merge(tr, on="entity_id", how="left", validate="one_to_one")

    wc = pd.to_numeric(df["WC"], errors="coerce")
    df["log_WC"] = np.log(wc.where(wc > 0))
    for c in ("BCRT_Total", "CRT2_Total", "CRT2_Total_adj", "NFC_Total"):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            df["z_" + c] = (s - s.mean()) / s.std(ddof=1)
    # covariates: numeric age; gender and education with levels of < 5 people collapsed
    df["age_c"] = pd.to_numeric(df["age"], errors="coerce")
    for c in ("gender", "education"):
        s = df[c].astype("string")
        vc = s.value_counts()
        rare = set(vc[vc < 5].index)
        df[c + "_c"] = s.where(~s.isin(rare) & s.notna(), "Other_or_missing")
    return df


def dummies(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Design block: numeric columns pass through, categoricals become drop-first dummies."""
    parts = []
    for c in cols:
        s = df[c]
        if s.dtype.kind in "fi":
            parts.append(pd.to_numeric(s, errors="coerce").rename(c).to_frame())
        else:
            parts.append(pd.get_dummies(s.astype("string"), prefix=c, drop_first=True, dtype=float))
    return pd.concat(parts, axis=1)


def agent_frame(root: Path, label: str) -> pd.DataFrame:
    p = run_dir(root, label) / "tables" / "by_dataset" / "socsim_agent" / "00_person_frame.csv"
    what = ("agent person frame of a human-derived run (controlled-access)" if "HUMAN" in label
            else "agent person frame (per-run analysis output)")
    return pd.read_csv(require(p, what), low_memory=False)


def brt_n(df: pd.DataFrame) -> int:
    """Realised BRT N for a run: rows that carry a coded BRT field."""
    return int(df["BRT1_decision"].notna().sum()) if "BRT1_decision" in df.columns else 0
