"""Statistics for the configuration runs (multi-provider, temperature and probe cells on the profile panel).

This module reads the report tables that the per-run pipeline (``socsim/pipeline.py``,
``socsim/analysis/*``, in particular ``socsim/analysis/crt_response_categories.py`` and
``socsim/analysis/distribution_divergence.py``) writes for each analysed run, and adds
statistics the per-run pipeline does not compute. It is not imported by the per-run
pipeline and never modifies its outputs. The statistics:

  - cluster-bootstrap 95% CIs over personas for intuitive-lure prevalence
    (items within a persona are not independent draws; resampling personas,
    not items, is what makes the CI valid)
  - the naive "rule of three" upper bound reported alongside the bootstrap CI
  - reweighting of pooled (i.e. collapsed across ``specified_level``)
    category proportions onto an external level-mix -- typically the
    reference run's level distribution, since a curated N=150 persona panel
    need not reproduce that run's level mix by chance
  - realised-score variance ratios, an exact McNemar test on paired any-lure
    flags, a paired proportion-difference CI and category-mix agreement
    between two cells
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from socsim.scoring.crt_categories import CRT_CATEGORY_LEVELS

RNG_SEED_DEFAULT = 42
CRT_INSTRUMENTS = ("crt2", "bcrt")


# ---------------------------------------------------------------------------
# Loaders -- thin wrappers around tables the per-run pipeline writes for an
# analysed report_dir. No scoring/statistics logic here.
# ---------------------------------------------------------------------------


def load_crt_long(report_dir: Path | str, dataset: str = "socsim_agent") -> pd.DataFrame:
    """Per-item, per-persona CRT response categories for one analysed report.

    Source: ``tables/comparisons/05_crt_response_categories_long.csv``,
    written by the per-run pipeline (``socsim.analysis.crt_response_categories
    .crt_response_category_long``). Filtered here to one dataset (default the
    simulated agent, ``socsim_agent``) purely for convenience.
    """
    path = Path(report_dir) / "tables" / "comparisons" / "05_crt_response_categories_long.csv"
    df = pd.read_csv(path)
    return df[df["dataset"] == dataset].copy()


def load_distribution_divergence(report_dir: Path | str) -> pd.DataFrame:
    """Per-run pipeline JS divergence + realised means for CRT2/BCRT/NFC/BFI.

    Source: ``tables/simulation_fidelity/distribution_divergence.csv``.
    """
    path = Path(report_dir) / "tables" / "simulation_fidelity" / "distribution_divergence.csv"
    return pd.read_csv(path)


def load_univariate_comparisons(report_dir: Path | str) -> pd.DataFrame:
    """Per-run pipeline human/synthetic/socsim_agent mean comparisons.

    Source: ``tables/comparisons/05_univariate_comparisons.csv``.
    """
    path = Path(report_dir) / "tables" / "comparisons" / "05_univariate_comparisons.csv"
    return pd.read_csv(path)


def load_run_metadata(report_dir: Path | str) -> dict:
    path = Path(report_dir) / "run_metadata.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_simulator_config(metrics_and_config_json: Path | str) -> dict:
    """Full simulator-side configuration record (the simulator's
    ``metrics_and_config.json``), including the API-reliability counters.
    Read verbatim -- this module never recomputes simulator-side metrics,
    only report-side statistics.
    """
    path = Path(metrics_and_config_json)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Level-mix reweighting (no per-run pipeline equivalent)
# ---------------------------------------------------------------------------


def persona_level_map(crt_long: pd.DataFrame, instrument: str) -> pd.Series:
    """One row per persona -> that persona's fixed ``specified_level`` for
    `instrument` (a designed property of the persona bank, not derived from
    this run's realised responses)."""
    inst = crt_long[crt_long["instrument"] == instrument]
    return inst.groupby("entity_id")["specified_level"].first()


def level_mix_weights(crt_long: pd.DataFrame, instrument: str) -> pd.Series:
    """Frequency of each ``specified_level`` among personas actually present
    in `crt_long`. Used both to describe a run's own level mix, and -- when
    computed on a DIFFERENT (e.g. the reference) run's `crt_long` --
    as the external reweighting target passed to
    :func:`pooled_category_proportions`."""
    levels = persona_level_map(crt_long, instrument).dropna()
    if levels.empty:
        return pd.Series(dtype=float)
    return levels.value_counts(normalize=True).sort_index()


def pooled_category_proportions(
    crt_long: pd.DataFrame,
    instrument: str,
    level_weights: pd.Series | None = None,
) -> pd.Series:
    """Pooled (collapsed-across-items-and-personas) response-category
    proportions for `instrument`.

    If `level_weights` is ``None``, this is a plain pooled proportion using
    this run's own item/persona mix (no reweighting).

    If `level_weights` is given (a Series indexed by ``specified_level``,
    summing to ~1 -- typically :func:`level_mix_weights` computed on a
    DIFFERENT run, e.g. the reference synthetic-numeric run), proportions
    are first computed WITHIN each ``specified_level`` present in THIS run,
    then recombined using the EXTERNAL weights. This answers "what would this
    cell's pooled category mix look like if its persona panel had the same
    level mix as the reference run" -- needed because a curated N=150 panel
    need not reproduce that mix by chance.
    """
    inst = crt_long[crt_long["instrument"] == instrument].copy()
    if inst.empty:
        return pd.Series(0.0, index=CRT_CATEGORY_LEVELS)

    if level_weights is None or level_weights.empty:
        counts = inst["response_category"].value_counts()
        counts = counts.reindex(CRT_CATEGORY_LEVELS, fill_value=0)
        return (counts / counts.sum()).astype(float)

    observed_levels = set(inst["specified_level"].dropna().unique())
    weighted_rows = []
    covered_weight = 0.0
    for level, weight in level_weights.items():
        if level not in observed_levels:
            continue
        level_rows = inst[inst["specified_level"] == level]
        counts = level_rows["response_category"].value_counts().reindex(CRT_CATEGORY_LEVELS, fill_value=0)
        within_level_props = counts / counts.sum()
        weighted_rows.append(within_level_props * weight)
        covered_weight += float(weight)
    if not weighted_rows or covered_weight <= 0:
        # This run has no personas at any level present in level_weights;
        # fall back to the unweighted pooled proportion rather than dividing
        # by zero.
        counts = inst["response_category"].value_counts().reindex(CRT_CATEGORY_LEVELS, fill_value=0)
        return (counts / counts.sum()).astype(float)
    combined = pd.concat(weighted_rows, axis=1).sum(axis=1)
    # Renormalise so the reported proportions still sum to 1 even if this
    # run is missing personas at some level(s) present in level_weights.
    return (combined / covered_weight).astype(float)


# ---------------------------------------------------------------------------
# Cluster bootstrap + rule of three
# ---------------------------------------------------------------------------


def rule_of_three_upper_bound(n_trials: int) -> float | None:
    """Classic naive "rule of three" approximate 95% upper bound on a
    proportion, 3/n. Most informative/valid when the observed event count is
    zero (the textbook case); reported here unconditionally as a simple
    reference bound ALONGSIDE the cluster-bootstrap CI (never in place of
    it). Returns ``None`` if n_trials <= 0."""
    if n_trials <= 0:
        return None
    return min(1.0, 3.0 / n_trials)


def cluster_bootstrap_proportion_ci(
    crt_long: pd.DataFrame,
    instrument: str | None,
    category: str = "intuitive_incorrect",
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = RNG_SEED_DEFAULT,
) -> dict:
    """Cluster-bootstrap 95% CI for the prevalence of `category` (default:
    the intuitive lure) among CRT items.

    Resamples PERSONAS (``entity_id``) with replacement, not individual
    items -- CRT2/bCRT items answered by the same persona are not
    independent draws, so an item-level (i.i.d.) bootstrap would understate
    the true sampling uncertainty. `instrument=None` pools CRT2 and bCRT
    items together (all items answered by each persona, both instruments).
    """
    df = crt_long if instrument is None else crt_long[crt_long["instrument"] == instrument]
    df = df.dropna(subset=["entity_id"])
    personas = df["entity_id"].unique()
    n_personas = len(personas)

    grouped = {pid: grp["response_category"].to_numpy() for pid, grp in df.groupby("entity_id")}
    all_categories = np.concatenate(list(grouped.values())) if grouped else np.array([])
    n_items = int(all_categories.size)
    n_events = int(np.sum(all_categories == category))
    point = float(n_events) / n_items if n_items else float("nan")

    rng = np.random.default_rng(seed)
    boot_props = np.empty(n_boot, dtype=float)
    persona_ids = list(grouped.keys())
    for b in range(n_boot):
        sample_ids = rng.choice(persona_ids, size=n_personas, replace=True)
        boot_cats = np.concatenate([grouped[pid] for pid in sample_ids]) if n_personas else np.array([])
        boot_props[b] = np.mean(boot_cats == category) if boot_cats.size else float("nan")

    lo, hi = (float(x) for x in np.nanquantile(boot_props, [alpha / 2, 1 - alpha / 2]))
    return {
        "instrument": instrument or "pooled",
        "category": category,
        "n_personas": int(n_personas),
        "n_items": n_items,
        "n_events": n_events,
        "proportion": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_boot": n_boot,
        "seed": seed,
        "rule_of_three_upper_bound": rule_of_three_upper_bound(n_items),
    }


# ---------------------------------------------------------------------------
# Per-cell tidy table assembly
# ---------------------------------------------------------------------------


def build_cell_tidy_table(
    report_dir: Path | str,
    model_label: str,
    original_run_level_weights: dict[str, pd.Series] | None = None,
    n_boot: int = 5000,
    seed: int = RNG_SEED_DEFAULT,
) -> pd.DataFrame:
    """Assemble one long/tidy table of configuration-cell statistics for a
    single analysed cell (report_dir). Combines per-run pipeline tables (read
    verbatim) with the statistics computed in this module.

    `original_run_level_weights`: optional ``{instrument: level_mix_weights
    Series}`` computed on the reference run, used to reweight this
    cell's pooled proportions per :func:`pooled_category_proportions`. If
    omitted, only the unweighted pooled proportions are reported.
    """
    crt_long = load_crt_long(report_dir)
    divergence = load_distribution_divergence(report_dir)
    univariate = load_univariate_comparisons(report_dir)
    original_run_level_weights = original_run_level_weights or {}

    rows: list[dict] = []

    def add(metric_family, **kw):
        row = {"model_label": model_label, "metric_family": metric_family}
        row.update(kw)
        rows.append(row)

    # --- CRT2/bCRT category proportions per specified_level (the per-run
    # pipeline computes this per item; here it is re-derived per persona-item,
    # pooled by level, directly from the long table, so one consistent source
    # feeds both this table and the reweighting below).
    for instrument in CRT_INSTRUMENTS:
        inst_df = crt_long[crt_long["instrument"] == instrument]
        if inst_df.empty:
            continue
        for level, level_df in inst_df.groupby("specified_level", dropna=True):
            counts = level_df["response_category"].value_counts().reindex(CRT_CATEGORY_LEVELS, fill_value=0)
            props = counts / counts.sum()
            n_personas = int(level_df["entity_id"].nunique())
            for category, proportion in props.items():
                add(
                    "category_proportion_by_level",
                    instrument=instrument,
                    specified_level=level,
                    category=category,
                    value=float(proportion),
                    n=int(counts.sum()),
                    n_personas=n_personas,
                )

        # Pooled, unweighted (this run's own level mix).
        pooled = pooled_category_proportions(crt_long, instrument, level_weights=None)
        for category, proportion in pooled.items():
            add(
                "category_proportion_pooled",
                instrument=instrument,
                category=category,
                value=float(proportion),
                n=int(inst_df.shape[0]),
            )

        # Pooled, reweighted onto the reference run's level mix.
        weights = original_run_level_weights.get(instrument)
        if weights is not None and not weights.empty:
            reweighted = pooled_category_proportions(crt_long, instrument, level_weights=weights)
            for category, proportion in reweighted.items():
                add(
                    "category_proportion_pooled_reweighted_to_original_run",
                    instrument=instrument,
                    category=category,
                    value=float(proportion),
                    n=int(inst_df.shape[0]),
                )

    # --- Intuitive-lure prevalence with cluster-bootstrap CI + naive bound,
    # per instrument and pooled across both.
    for instrument in (*CRT_INSTRUMENTS, None):
        result = cluster_bootstrap_proportion_ci(
            crt_long, instrument=instrument, category="intuitive_incorrect", n_boot=n_boot, seed=seed
        )
        add(
            "lure_prevalence",
            instrument=result["instrument"],
            category="intuitive_incorrect",
            value=result["proportion"],
            ci_low=result["ci95_low"],
            ci_high=result["ci95_high"],
            rule_of_three_upper_bound=result["rule_of_three_upper_bound"],
            n=result["n_items"],
            n_personas=result["n_personas"],
            n_events=result["n_events"],
        )

    # --- Category-mass JS divergence vs human, and realised NFC/BFI means +
    # JS, read from the per-run pipeline tables (no recomputation).
    div = divergence[divergence["comparison"] == "socsim_agent"]
    for _, r in div.iterrows():
        add(
            "js_divergence_vs_human",
            variable=r["variable"],
            value=float(r["js_divergence"]),
            mean_human=float(r["mean_human"]),
            mean_sim=float(r["mean_sim"]),
            cohens_d=float(r["cohens_d"]) if pd.notna(r["cohens_d"]) else None,
        )

    uni = univariate[(univariate["dataset_a"] == "human") & (univariate["dataset_b"] == "socsim_agent")]
    for _, r in uni.iterrows():
        add(
            "realised_mean_vs_human",
            variable=r["variable"],
            value=float(r["mean_b"]),
            mean_human=float(r["mean_a"]),
            mean_diff=float(r["mean_diff"]),
            p_value=float(r["p_value"]) if pd.notna(r["p_value"]) else None,
            n=int(r["N_b"]),
        )

    # --- Realised-score variance ratios vs human (the variance-by-temperature
    # trajectory; the same rows are added for every cell).
    for _, r in realised_score_variance_ratios(report_dir).iterrows():
        add(
            "variance_ratio_vs_human",
            variable=r["variable"],
            value=float(r["variance_ratio"]) if pd.notna(r["variance_ratio"]) else None,
            sd_human=float(r["sd_human"]),
            sd_sim=float(r["sd_sim"]),
            n=int(r["n_sim"]),
            n_human=int(r["n_human"]),
        )

    return pd.DataFrame(rows)


def original_run_level_weights_for_instruments(
    original_report_dir: Path | str,
    instruments: tuple[str, ...] = CRT_INSTRUMENTS,
) -> dict[str, pd.Series]:
    """Convenience wrapper: the reference run's level-mix weights for each
    CRT instrument, to pass as `original_run_level_weights` to
    :func:`build_cell_tidy_table`.

    Both CRT2 and bCRT are level-keyed on the CRT2 specified proxy (bCRT is
    not specified in the persona prompt), so every instrument's reweighting
    target is the reference run's CRT2 ``specified_level`` mix.
    """
    crt_long = load_crt_long(original_report_dir)
    crt2_weights = level_mix_weights(crt_long, "crt2")
    return {instrument: crt2_weights for instrument in instruments}


# ---------------------------------------------------------------------------
# Variance ratios, order-effects McNemar, category agreement
# ---------------------------------------------------------------------------


VARIANCE_RATIO_VARIABLES = (
    "CRT2_Total",
    "BCRT_Total",
    "NFC_Total",
    "BFI_Extraversion",
    "BFI_Agreeableness",
    "BFI_Conscientiousness",
    "BFI_Neuroticism",
    "BFI_Openness",
)


def load_descriptives_overview(report_dir: Path | str) -> pd.DataFrame:
    path = Path(report_dir) / "tables" / "00_descriptives_overview.csv"
    return pd.read_csv(path)


def realised_score_variance_ratios(
    report_dir: Path | str,
    variables: tuple[str, ...] = VARIANCE_RATIO_VARIABLES,
) -> pd.DataFrame:
    """Variance ratio = Var(socsim_agent) / Var(human) from the per-run
    descriptives overview (SD^2). Returns one row per variable with human/sim SD and
    the ratio. Empty-ratio when human SD is 0.
    """
    desc = load_descriptives_overview(report_dir)
    rows: list[dict] = []
    for variable in variables:
        human = desc[(desc["dataset"] == "human") & (desc["variable"] == variable)]
        sim = desc[(desc["dataset"] == "socsim_agent") & (desc["variable"] == variable)]
        if human.empty or sim.empty:
            continue
        sd_h = float(human.iloc[0]["SD"])
        sd_s = float(sim.iloc[0]["SD"])
        var_h = sd_h ** 2
        var_s = sd_s ** 2
        ratio = (var_s / var_h) if var_h > 0 else float("nan")
        rows.append(
            {
                "variable": variable,
                "sd_human": sd_h,
                "sd_sim": sd_s,
                "var_human": var_h,
                "var_sim": var_s,
                "variance_ratio": ratio,
                "n_human": int(human.iloc[0]["N_non_missing"]),
                "n_sim": int(sim.iloc[0]["N_non_missing"]),
            }
        )
    return pd.DataFrame(rows)


def persona_any_lure(
    crt_long: pd.DataFrame,
    instruments: tuple[str, ...] = CRT_INSTRUMENTS,
) -> pd.DataFrame:
    """One row per persona: any-lure flag over the named instruments, plus
    the persona's CRT2-proxy ``specified_level`` (from CRT2 rows)."""
    work = crt_long[crt_long["instrument"].isin(instruments)].copy()
    if work.empty:
        return pd.DataFrame(columns=["entity_id", "specified_level", "any_lure", "n_items"])
    crt2_levels = (
        crt_long[crt_long["instrument"] == "crt2"]
        .groupby("entity_id")["specified_level"]
        .first()
    )
    grouped = work.groupby("entity_id")
    out = pd.DataFrame(
        {
            "entity_id": grouped.size().index.astype(str),
            "any_lure": grouped["response_category"].apply(
                lambda s: bool((s == "intuitive_incorrect").any())
            ).to_numpy(),
            "n_items": grouped.size().to_numpy(),
        }
    )
    out["specified_level"] = out["entity_id"].map(crt2_levels)
    return out


def mcnemar_any_lure(
    paired: pd.DataFrame,
    col_a: str = "any_lure_a",
    col_b: str = "any_lure_b",
) -> dict:
    """Exact McNemar on paired any-lure flags (cols must be boolean).

    Returns discordant counts + mid-p / exact binomial p-value on the
    off-diagonal (b, c) with H0: P(lure_a) = P(lure_b).
    """
    from scipy.stats import binomtest

    a = paired[col_a].astype(bool)
    b = paired[col_b].astype(bool)
    n11 = int(((a) & (b)).sum())
    n10 = int(((a) & (~b)).sum())  # lure in A only
    n01 = int(((~a) & (b)).sum())  # lure in B only
    n00 = int(((~a) & (~b)).sum())
    discordant = n10 + n01
    if discordant == 0:
        p_value = 1.0
    else:
        # Exact McNemar: binomial test that P(n10) = 0.5 among discordant pairs
        p_value = float(binomtest(n10, discordant, 0.5).pvalue)
    return {
        "n_pairs": int(len(paired)),
        "n11_both_lure": n11,
        "n10_a_only": n10,
        "n01_b_only": n01,
        "n00_neither": n00,
        "discordant": discordant,
        "p_mcnemar_exact": p_value,
        "prop_lure_a": float(a.mean()) if len(paired) else float("nan"),
        "prop_lure_b": float(b.mean()) if len(paired) else float("nan"),
        "prop_diff_b_minus_a": float(b.mean() - a.mean()) if len(paired) else float("nan"),
    }


def paired_category_proportion_diff_ci(
    long_a: pd.DataFrame,
    long_b: pd.DataFrame,
    instrument: str | None,
    category: str = "intuitive_incorrect",
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = RNG_SEED_DEFAULT,
) -> dict:
    """Cluster-bootstrap CI for (prop_B - prop_A) on the *intersection* of
    personas present in both long tables. Resamples shared persona IDs.
    """
    a = long_a if instrument is None else long_a[long_a["instrument"] == instrument]
    b = long_b if instrument is None else long_b[long_b["instrument"] == instrument]
    a = a.copy()
    b = b.copy()
    a["entity_id"] = a["entity_id"].astype(str)
    b["entity_id"] = b["entity_id"].astype(str)
    shared = sorted(set(a["entity_id"]) & set(b["entity_id"]))
    if not shared:
        return {
            "instrument": instrument or "pooled",
            "category": category,
            "n_personas": 0,
            "prop_a": float("nan"),
            "prop_b": float("nan"),
            "diff_b_minus_a": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    grp_a = {pid: g["response_category"].to_numpy() for pid, g in a[a["entity_id"].isin(shared)].groupby("entity_id")}
    grp_b = {pid: g["response_category"].to_numpy() for pid, g in b[b["entity_id"].isin(shared)].groupby("entity_id")}

    def _prop(groups, ids):
        id_list = list(ids)
        cats = np.concatenate([groups[i] for i in id_list]) if id_list else np.array([])
        return float(np.mean(cats == category)) if cats.size else float("nan")

    prop_a = _prop(grp_a, shared)
    prop_b = _prop(grp_b, shared)
    point = prop_b - prop_a
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(shared, size=len(shared), replace=True)
        boots[i] = _prop(grp_b, sample) - _prop(grp_a, sample)
    lo, hi = (float(x) for x in np.nanquantile(boots, [alpha / 2, 1 - alpha / 2]))
    return {
        "instrument": instrument or "pooled",
        "category": category,
        "n_personas": len(shared),
        "prop_a": prop_a,
        "prop_b": prop_b,
        "diff_b_minus_a": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_boot": n_boot,
        "seed": seed,
    }


def category_proportion_agreement(
    long_a: pd.DataFrame,
    long_b: pd.DataFrame,
    instrument: str,
) -> dict:
    """L1 / max-abs agreement between two cells' pooled category mixes for
    one instrument (e.g. the default-temperature cell vs T=1.0)."""
    props_a = pooled_category_proportions(long_a, instrument)
    props_b = pooled_category_proportions(long_b, instrument)
    l1 = float(np.abs(props_a - props_b).sum())
    max_abs = float(np.abs(props_a - props_b).max())
    return {
        "instrument": instrument,
        "l1_distance": l1,
        "max_abs_diff": max_abs,
        **{f"prop_a_{c}": float(props_a[c]) for c in CRT_CATEGORY_LEVELS},
        **{f"prop_b_{c}": float(props_b[c]) for c in CRT_CATEGORY_LEVELS},
    }


def per_level_persona_counts(crt_long: pd.DataFrame, instrument: str) -> pd.Series:
    """Actual n personas per analysis ``specified_level`` (not the panel's
    crt2_level stratum size of 30)."""
    inst = crt_long[crt_long["instrument"] == instrument]
    return (
        inst.dropna(subset=["specified_level"])
        .groupby("specified_level")["entity_id"]
        .nunique()
        .sort_index()
    )
