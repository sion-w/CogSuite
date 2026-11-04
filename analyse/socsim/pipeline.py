from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

from socsim.analysis.binned_relationships import build_brt_liwc_binned_relationships, build_core_binned_relationships
from socsim.analysis.calibration import calibration_tables
from socsim.analysis.brt import brt_descriptives, run_brt_models
from socsim.analysis.comparisons import overlap_summary, pairwise_univariate_comparisons
from socsim.analysis.correlations import build_corr_matrix, build_corr_p_matrix, corr_diff_vs_human
from socsim.analysis.crt_response_categories import (
    crt_response_category_long,
    response_category_distribution,
    response_category_distribution_by_specified_level,
    response_category_divergence,
)
from socsim.analysis.descriptives import BFI_DISTRIBUTION_VARS, CORE_DISTRIBUTION_VARS, KEY_VARS, make_descriptives_overview
from socsim.analysis.distribution_divergence import compute_distribution_divergence
from socsim.analysis.models import run_fixed_models, run_liwc_family_models
from socsim.analysis.simulation_fidelity_scorecard import build_simulation_fidelity_scorecard
from socsim.analysis.slope_difference_tests import run_slope_difference_tests
from socsim.common import (
    AGENT_ITEM_TEXT_COLS,
    analysis_run_id,
    ensure_dir,
    json_write,
    load_yaml,
    read_csv_keep_text,
    require_columns,
)
from socsim.exclusions import EXCLUSION_NOTE, apply_brt_exclusions, exclusions_path
from socsim.ingest.harmonise_cogsim import harmonise_cogsim_items, harmonise_cogsim_person
from socsim.ingest.harmonise_human import harmonise_human
from socsim.ingest.harmonise_socsim import harmonise_socsim_items, harmonise_socsim_person
from socsim.ingest.harmonise_synth import harmonise_synthetic
from socsim.ingest.load_inputs import copy_resource_inputs, load_optional_cogsim, load_optional_socsim, load_required_inputs
from socsim.liwc.export_texts import export_agent_brt_texts
from socsim.liwc.families import compute_liwc_families
from socsim.liwc.import_join import join_liwc_outputs
from socsim.scoring.brt_code import code_brt_for_agents, validate_human_brt_coded
from socsim.scoring.crt_categories import add_item_response_categories
from socsim.scoring.crt_score_items import score_crt_items_with_llm
from socsim.scoring.crt_score_totals import compute_crt_totals
from socsim.scoring.likert_score_bfi_nfc import score_likert_bfi_nfc
from socsim.viz.calibration_figs import plot_abs_error_hists, plot_calibration_panel, plot_calibration_scatter
from socsim.viz.brt_figs import plot_brt_decision_distribution, plot_brt_primary_forest
from socsim.viz.comparisons_figs import plot_overlap_panel
from socsim.viz.corr_figs import plot_corr_diff, plot_corr_matrix
from socsim.viz.crt_category_figs import plot_response_category_by_specified_level, plot_response_category_distribution
from socsim.viz.descriptives_figs import plot_demographic_overview, plot_descriptives_overview
from socsim.viz.models_figs import plot_focused_relationships, plot_liwc_family_models, plot_primary_model_coefficients
from socsim.viz.plot_relationship_fidelity import plot_brt_liwc_relationship_fidelity, plot_core_relationship_fidelity


def _is_llm_failure_reason(value: object) -> bool:
    text = str(value) if value is not None else ""
    return text in {"import_openai_failed", "no_api_key"} or text.startswith("api_exception")


def _crt_cache_is_valid(df: pd.DataFrame) -> bool:
    if "llm_reason" not in df.columns or "llm_label" not in df.columns:
        return False
    if df.empty:
        return False
    missing_ok = df["llm_reason"].fillna("").astype(str).eq("item_missing")
    labels_complete = bool((df["llm_label"].notna() | missing_ok).all())
    any_failed = bool(df["llm_reason"].map(_is_llm_failure_reason).any())
    return labels_complete and not any_failed


def _crt_person_cache_has_category_outputs(df: pd.DataFrame) -> bool:
    required = [
        "crt2_correct_total",
        "crt2_intuitive_total",
        "crt2_other_total",
        "bcrt_correct_total",
        "bcrt_intuitive_total",
        "bcrt_other_total",
    ]
    return all(col in df.columns for col in required)


def _assert_crt_scoring_succeeded(scored_items: pd.DataFrame, dataset: str) -> None:
    if _crt_cache_is_valid(scored_items):
        return
    reason_counts = {}
    if "llm_reason" in scored_items.columns:
        reason_counts = scored_items["llm_reason"].fillna("missing_reason").astype(str).value_counts().head(5).to_dict()
    raise ValueError(
        f"{dataset} CRT scoring did not complete successfully for every CRT item. "
        f"Top failure reasons: {reason_counts}. "
        "Check OPENAI_API_KEY / model access, delete the failed *_items_scored.csv and *_person_scored.csv cache files, and rerun."
    )


def _assert_brt_coding_succeeded(coded_person: pd.DataFrame, dataset: str) -> None:
    if "BRT_llm_reason" not in coded_person.columns:
        raise ValueError(f"{dataset} BRT coding did not produce BRT_llm_reason.")
    attempted = coded_person["BRT_llm_reason"].notna()
    coded_count = int((coded_person.loc[attempted, "BRT_llm_reason"] == "ok").sum())
    total = int(attempted.sum())
    if coded_count == total:
        return
    reason_counts = coded_person["BRT_llm_reason"].fillna("missing_reason").astype(str).value_counts().head(5).to_dict()
    raise ValueError(
        f"{dataset} BRT coding did not complete successfully for every attempted agent "
        f"({coded_count}/{total} coded). Top failure reasons: {reason_counts}. "
        "Check OPENAI_API_KEY / model access, delete the failed *_person_brt_coded.csv cache file, and rerun."
    )


def _brt_cache_is_valid(df: pd.DataFrame) -> bool:
    if "BRT_llm_reason" not in df.columns:
        return False
    key_fields = [
        "BRT1_decision",
        "BRT1_anchor",
        "BRT2_decision",
        "BRT2_anchor",
        "BRT2_research_specificity",
    ]
    if any(c not in df.columns for c in key_fields):
        return False
    complete_rows = bool(df[key_fields].notna().all(axis=1).all()) if len(df) > 0 else False
    all_failed = bool(df["BRT_llm_reason"].map(_is_llm_failure_reason).all())
    return complete_rows and not all_failed


def _dataframe_fingerprint(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    cols = sorted(columns or list(df.columns))
    work = df.loc[:, [c for c in cols if c in df.columns]].copy()
    for col in work.columns:
        work[col] = work[col].where(work[col].notna(), "").astype(str)
    if list(work.columns):
        work = work.sort_values(list(work.columns), kind="mergesort").reset_index(drop=True)
    payload = work.to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _brt_text_fingerprint(items_df: pd.DataFrame) -> str:
    cols = ["entity_id", "item_id", "response_text"]
    if any(c not in items_df.columns for c in cols):
        return _dataframe_fingerprint(pd.DataFrame(columns=cols), cols)
    brt_rows = items_df.loc[items_df["item_id"].astype(str).isin(["brt_1", "brt_2"]), cols].copy()
    return _dataframe_fingerprint(brt_rows, cols)


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _write_cache_metadata(cache_path: Path, metadata: dict[str, object]) -> None:
    json_write(_cache_meta_path(cache_path), metadata)


def _cache_metadata_matches(cache_path: Path, expected: dict[str, object]) -> bool:
    meta_path = _cache_meta_path(cache_path)
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
    except Exception:
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def _agent_scoring_cache_metadata(dataset: str, person_df: pd.DataFrame, items_df: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset": dataset,
        "cache_kind": "agent_crt_likert_scoring",
        "scorer_version": "scoring_v2",
        "source_person_fingerprint": _dataframe_fingerprint(person_df),
        "source_items_fingerprint": _dataframe_fingerprint(items_df),
    }


def _agent_brt_cache_metadata(dataset: str, person_df: pd.DataFrame, items_df: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset": dataset,
        "cache_kind": "agent_brt_coding",
        "scorer_version": "brt_v3",
        "source_person_fingerprint": _dataframe_fingerprint(person_df),
        "source_brt_text_fingerprint": _brt_text_fingerprint(items_df),
    }


def _dataset_requested(dataset: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    if include_set and dataset not in include_set:
        return False
    return dataset not in exclude_set


def _clean_run_label(value: object) -> str | None:
    if value is None:
        return None
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return label or None


def _liwc_run_label(dataset: str, socsim_run_label: str | None) -> str | None:
    if dataset == "socsim_agent":
        return socsim_run_label
    return None


def _liwc_output_candidates(inputs_dir: Path, exports_dir: Path, dataset: str, run_label: str | None) -> list[Path]:
    stems = []
    if run_label:
        stems.append(f"{dataset}_{run_label}")
    stems.append(dataset)
    candidates: list[Path] = []
    for stem in stems:
        candidates.append(inputs_dir / f"{stem}_liwc_output.csv")
        candidates.append(exports_dir / f"{stem}_liwc_output.csv")
    return candidates


def _format_path_for_log(path: Path) -> str:
    return os.path.relpath(path)


def _summarize_brt_model_warnings(models: pd.DataFrame) -> str | None:
    if models.empty or "model_warning" not in models.columns:
        return None
    warning_mask = models["model_warning"].fillna("").astype(str).str.strip().ne("")
    if not bool(warning_mask.any()):
        return None
    flagged = models.loc[warning_mask].copy()
    pieces = []
    for dataset, ds_rows in flagged.groupby("dataset", sort=False):
        outcome_bits = []
        for outcome, out_rows in ds_rows.groupby("outcome", sort=False):
            prevalence = pd.to_numeric(out_rows.get("outcome_prevalence"), errors="coerce").dropna()
            events = pd.to_numeric(out_rows.get("event_count"), errors="coerce").dropna()
            n = pd.to_numeric(out_rows.get("N"), errors="coerce").dropna()
            if not prevalence.empty and not events.empty and not n.empty:
                outcome_bits.append(f"{outcome} ({int(events.iloc[0])}/{int(n.iloc[0])} events, {prevalence.iloc[0]:.1%})")
            else:
                outcome_bits.append(str(outcome))
        pieces.append(f"{dataset}: " + "; ".join(outcome_bits))
    return (
        f"BRT logistic model note: {len(flagged)}/{len(models)} coefficient rows were numerically unstable/non-converged, "
        "usually because a binary BRT outcome is very rare or very common. Affected outcomes: "
        + " | ".join(pieces)
    )


def _assert_openai_runtime_available() -> None:
    try:
        import openai  # noqa: F401
    except Exception as e:
        raise ValueError(
            "OpenAI SDK is not available. Install it before first-run CRT/BRT scoring: "
            "`python -m pip install openai`."
        ) from e


def _has_nonempty_text(values: pd.Series) -> bool:
    if values.empty:
        return False
    return bool(values.fillna("").astype(str).str.strip().ne("").any())


def _agent_items_have_brt_text(items_df: pd.DataFrame) -> bool:
    if "item_id" not in items_df.columns or "response_text" not in items_df.columns:
        return False
    brt_rows = items_df.loc[items_df["item_id"].astype(str).isin(["brt_1", "brt_2"]), "response_text"]
    return _has_nonempty_text(brt_rows)


def _human_has_brt_coded_columns(df: pd.DataFrame) -> bool:
    try:
        validate_human_brt_coded(df)
    except ValueError:
        return False
    return True


def _dataset_has_brt_outcomes(df: pd.DataFrame) -> bool:
    brt_cols = [
        "BRT1_decision",
        "BRT2_decision",
        "BRT2_research_specificity",
        "BRT1_emotional_outburst",
        "BRT1_moral_fairness_loyalty",
        "BRT2_reflective_accept",
        "BRT1_reflective_keep",
        "BRT2_high_quality_research",
    ]
    return any(col in df.columns and df[col].notna().any() for col in brt_cols)


def _create_report_dirs(report_dir: Path, datasets: list[str]) -> None:
    ensure_dir(report_dir / "tables")
    ensure_dir(report_dir / "tables" / "by_dataset")
    for ds in datasets:
        ensure_dir(report_dir / "tables" / "by_dataset" / ds)
    ensure_dir(report_dir / "tables" / "comparisons")
    ensure_dir(report_dir / "tables" / "calibration")
    ensure_dir(report_dir / "tables" / "simulation_fidelity")
    ensure_dir(report_dir / "figures" / "by_dataset")
    for ds in datasets:
        ensure_dir(report_dir / "figures" / "by_dataset" / ds)
    ensure_dir(report_dir / "figures" / "comparisons")
    ensure_dir(report_dir / "figures" / "calibration")
    ensure_dir(report_dir / "figures" / "relationship_fidelity")
    ensure_dir(report_dir / "figures" / "brt_liwc_relationships")


def _apply_dataset_filters(
    by_dataset: dict[str, pd.DataFrame],
    include: list[str] | None,
    exclude: list[str] | None,
) -> dict[str, pd.DataFrame]:
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    available = list(by_dataset.keys())
    selected = []
    for ds in available:
        if include_set and ds not in include_set:
            continue
        if ds in exclude_set:
            continue
        selected.append(ds)
    return {k: by_dataset[k] for k in selected}


def _write_notebooks(root: Path, default_analysis_run_id: str) -> None:
    ensure_dir(root)
    nb_template: dict[str, object] = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "from IPython.display import display, Image, Markdown\n",
                    f"analysis_run_id = '{default_analysis_run_id}'\n",
                    "reports_root = Path('../../reports')\n",
                    "run_dir = reports_root / analysis_run_id\n",
                    "if not run_dir.exists() and (reports_root / 'run_index.csv').exists():\n",
                    "    run_index = pd.read_csv(reports_root / 'run_index.csv', dtype=str).fillna('')\n",
                    "    matches = run_index[(run_index['analysis_run_id'] == analysis_run_id) | (run_index.get('socsim_run_label', '') == analysis_run_id)]\n",
                    "    if not matches.empty:\n",
                    "        run_dir = reports_root / matches.iloc[0]['report_path']\n",
                    "        analysis_run_id = matches.iloc[0]['analysis_run_id']\n",
                    "display(Markdown(f'## Report: {analysis_run_id}'))\n",
                    "meta_path = run_dir / 'run_metadata.json'\n",
                    "if meta_path.exists():\n",
                    "    meta = pd.read_json(meta_path, typ='series')\n",
                    "    display(Markdown(f\"Datasets selected: `{', '.join(meta.get('datasets_selected', []))}`\"))\n",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["This notebook is narrative-only and reads precomputed outputs from the report folder."]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebooks = [
        "00_overview.ipynb",
        "01_correlations.ipynb",
        "02_models.ipynb",
        "03_liwc_brt.ipynb",
        "04_simulator_comparison.ipynb",
        "simulation_fidelity_evaluation.ipynb",
    ]
    for name in notebooks:
        nb = json.loads(json.dumps(nb_template))
        if name == "00_overview.ipynb":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## Overview\nMain cross-dataset discrete distribution panels (core + BFI)."]})
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "for asset in ['tables/00_descriptives_overview.csv', 'tables/00_demographic_descriptives.csv', 'figures/00_descriptives_overview.png', 'figures/00_demographic_overview.png', 'figures/comparisons/05_core_distribution_panel.png', 'figures/comparisons/05_bfi_distribution_panel.png', 'tables/comparisons/05_crt_response_category_distribution.csv', 'tables/comparisons/05_crt_response_category_by_specified_level.csv', 'tables/comparisons/05_crt_response_category_divergence.csv', 'figures/comparisons/05_crt_response_category_distribution.png', 'figures/comparisons/05_crt_response_category_by_specified_level.png']:\n",
                        "    path = run_dir / asset\n",
                        "    if not path.exists():\n",
                        "        display(Markdown(f'Missing: `{path}`'))\n",
                        "    elif path.suffix == '.csv':\n",
                        "        display(pd.read_csv(path).head(30))\n",
                        "    else:\n",
                        "        display(Image(filename=str(path)))\n",
                    ],
                }
            )
        if name == "01_correlations.ipynb":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## Correlations\nSpearman heatmaps are shown in two clusters per dataset: FDR-adjusted first, then uncorrected. Significant cells are annotated with `r` and significance stars."]})
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "meta = pd.read_json(run_dir / 'run_metadata.json', typ='series') if (run_dir / 'run_metadata.json').exists() else pd.Series(dtype=object)\n",
                        "datasets = meta.get('datasets_selected', [])\n",
                        "for ds in datasets:\n",
                        "    display(Markdown(f'### {ds}'))\n",
                        "    display(Markdown('#### FDR-adjusted significance overlay'))\n",
                        "    heat_fdr = run_dir / 'figures' / 'by_dataset' / ds / '01_correlations_heatmap_fdr.png'\n",
                        "    if heat_fdr.exists():\n",
                        "        display(Image(filename=str(heat_fdr)))\n",
                        "    diff_fdr = run_dir / 'figures' / 'by_dataset' / ds / '01_correlations_diff_vs_human_fdr.png'\n",
                        "    if ds != 'human' and diff_fdr.exists():\n",
                        "        display(Image(filename=str(diff_fdr)))\n",
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "meta = pd.read_json(run_dir / 'run_metadata.json', typ='series') if (run_dir / 'run_metadata.json').exists() else pd.Series(dtype=object)\n",
                        "datasets = meta.get('datasets_selected', [])\n",
                        "for ds in datasets:\n",
                        "    display(Markdown(f'### {ds}'))\n",
                        "    display(Markdown('#### Uncorrected significance overlay'))\n",
                        "    heat = run_dir / 'figures' / 'by_dataset' / ds / '01_correlations_heatmap.png'\n",
                        "    if heat.exists():\n",
                        "        display(Image(filename=str(heat)))\n",
                        "    diff = run_dir / 'figures' / 'by_dataset' / ds / '01_correlations_diff_vs_human.png'\n",
                        "    if ds != 'human' and diff.exists():\n",
                        "        display(Image(filename=str(diff)))\n",
                    ],
                }
            )
        if name == "02_models.ipynb":
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["Primary inferential model: binomial GLM (`model_type == 'binomial_glm'`). OLS is shown only as sensitivity."],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "path = run_dir / 'tables/04_models_all_datasets.csv'\n",
                        "if path.exists():\n",
                        "    t = pd.read_csv(path)\n",
                        "    display(Markdown('### Primary binomial GLM results'))\n",
                        "    display(t[t['model_type'] == 'binomial_glm'].head(50))\n",
                        "    display(Markdown('### OLS sensitivity results'))\n",
                        "    display(t[t['model_type'] == 'ols_sensitivity'].head(50))\n",
                        "else:\n",
                        "    display(Markdown(f'Missing: `{path}`'))\n",
                        "meta = pd.read_json(run_dir / 'run_metadata.json', typ='series') if (run_dir / 'run_metadata.json').exists() else pd.Series(dtype=object)\n",
                        "for ds in meta.get('datasets_selected', []):\n",
                        "    figp = run_dir / 'figures' / 'by_dataset' / ds / '02_models_primary_coefficients.png'\n",
                        "    if figp.exists():\n",
                        "        display(Markdown(f'### {ds} coefficients'))\n",
                        "        display(Image(filename=str(figp)))\n",
                    ],
                }
            )
        if name == "03_liwc_brt.ipynb":
            nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## LIWC and BRT\nFront-loaded comparison of significant LIWC relationships across datasets, followed by full LIWC and BRT tables."]})
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "liwc_path = run_dir / 'tables/04_liwc_family_models_all.csv'\n",
                        "meta = pd.read_json(run_dir / 'run_metadata.json', typ='series') if (run_dir / 'run_metadata.json').exists() else pd.Series(dtype=object)\n",
                        "selected = list(meta.get('datasets_selected', []))\n",
                        "comparison_ds = ['human'] + [d for d in selected if d.endswith('_agent')]\n",
                        "if liwc_path.exists():\n",
                        "    liwc = pd.read_csv(liwc_path)\n",
                        "    liwc = liwc[liwc['dataset'].isin(comparison_ds)].copy()\n",
                        "    if 'predictor_group' not in liwc.columns:\n",
                        "        liwc['predictor_group'] = liwc['predictor'].astype(str).map(lambda s: 'CRT2' if 'CRT2' in s else ('BCRT' if 'BCRT' in s else s))\n",
                        "    q = liwc['q_fdr'] if 'q_fdr' in liwc.columns else pd.Series(1.0, index=liwc.index)\n",
                        "    w = liwc['warning'] if 'warning' in liwc.columns else pd.Series('', index=liwc.index)\n",
                        "    liwc['is_sig_fdr'] = (q.fillna(1.0) < 0.05) & (w.fillna('') == '')\n",
                        "    key_cols = ['family','liwc_var','predictor_group']\n",
                        "    sig_keys = liwc.groupby(key_cols, as_index=False)['is_sig_fdr'].max()\n",
                        "    sig_keys = sig_keys[sig_keys['is_sig_fdr']]\n",
                        "    if len(sig_keys) == 0:\n",
                        "        display(Markdown('No LIWC rows pass family-wise FDR < 0.05 in this run.'))\n",
                        "    else:\n",
                        "        keys = sig_keys[key_cols].drop_duplicates().copy()\n",
                        "        ds_df = pd.DataFrame({'dataset': comparison_ds})\n",
                        "        keys['_tmp'] = 1\n",
                        "        ds_df['_tmp'] = 1\n",
                        "        grid = keys.merge(ds_df, on='_tmp').drop(columns=['_tmp'])\n",
                        "        focus = grid.merge(liwc, on=key_cols + ['dataset'], how='left').copy()\n",
                        "        cols = [c for c in ['family','liwc_var','predictor_group','predictor','predictor_source_col','dataset','beta','q_fdr','p_raw','N','r2','is_sig_fdr'] if c in focus.columns]\n",
                        "        display(Markdown('### LIWC rows where at least one dataset is FDR-significant (counterpart rows included)'))\n",
                        "        display(focus[cols].sort_values(['family','liwc_var','predictor_group','dataset']))\n",
                        "        metrics = [c for c in ['beta','p_raw','q_fdr','N','r2'] if c in focus.columns]\n",
                        "        wide = focus.set_index(key_cols + ['dataset'])[metrics].unstack('dataset')\n",
                        "        wide.columns = [f'{m}__{d}' for m, d in wide.columns]\n",
                        "        wide = wide.reset_index()\n",
                        "        for ds in [d for d in comparison_ds if d != 'human']:\n",
                        "            hb, ab = f'beta__human', f'beta__{ds}'\n",
                        "            if hb in wide.columns and ab in wide.columns:\n",
                        "                wide[f'delta_{ds}_minus_human'] = wide[ab] - wide[hb]\n",
                        "        delta_cols = [c for c in wide.columns if c.startswith('delta_')]\n",
                        "        sort_col = delta_cols[0] if delta_cols else [c for c in wide.columns if c.startswith('beta__')][0]\n",
                        "        display(Markdown('### Human vs agent LIWC contrast (side-by-side coefficients, counterpart always shown)'))\n",
                        "        if delta_cols:\n",
                        "            display(wide.sort_values(sort_col, key=lambda s: s.abs(), ascending=False))\n",
                        "        else:\n",
                        "            display(wide.sort_values(sort_col, ascending=False))\n",
                        "else:\n",
                        "    display(Markdown(f'Missing: `{liwc_path}`'))\n",
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "brt_path = run_dir / 'tables/06_brt_primary_models_all_datasets.csv'\n",
                        "meta = pd.read_json(run_dir / 'run_metadata.json', typ='series') if (run_dir / 'run_metadata.json').exists() else pd.Series(dtype=object)\n",
                        "selected = list(meta.get('datasets_selected', []))\n",
                        "comparison_ds = ['human'] + [d for d in selected if d.endswith('_agent')]\n",
                        "if brt_path.exists():\n",
                        "    brt = pd.read_csv(brt_path)\n",
                        "    brt = brt[brt['dataset'].isin(comparison_ds)].copy()\n",
                        "    brt['is_sig'] = brt['p'].fillna(1.0) < 0.05\n",
                        "    key_cols = ['outcome','predictor','model_type']\n",
                        "    sig_keys = brt.groupby(key_cols, as_index=False)['is_sig'].max()\n",
                        "    sig_keys = sig_keys[sig_keys['is_sig']]\n",
                        "    if len(sig_keys) == 0:\n",
                        "        display(Markdown('No BRT primary model rows pass p < 0.05 in this run.'))\n",
                        "    else:\n",
                        "        focus = brt.merge(sig_keys[key_cols], on=key_cols, how='inner').copy()\n",
                        "        cols = [c for c in ['outcome','predictor','model_type','predictor_source_col','dataset','beta','p','CI95_low','CI95_high','N','is_sig'] if c in focus.columns]\n",
                        "        display(Markdown('### BRT rows where at least one dataset is significant (long/audit view; counterpart rows included)'))\n",
                        "        display(focus[cols].sort_values(['outcome','predictor','model_type','dataset']))\n",
                        "        metrics = [c for c in ['beta','p','N'] if c in focus.columns]\n",
                        "        wide = focus.set_index(key_cols + ['dataset'])[metrics].unstack('dataset')\n",
                        "        wide.columns = [f'{m}__{d}' for m, d in wide.columns]\n",
                        "        wide = wide.reset_index()\n",
                        "        for ds in [d for d in comparison_ds if d != 'human']:\n",
                        "            hb, ab = f'beta__human', f'beta__{ds}'\n",
                        "            if hb in wide.columns and ab in wide.columns:\n",
                        "                wide[f'delta_{ds}_minus_human'] = wide[ab] - wide[hb]\n",
                        "        delta_cols = [c for c in wide.columns if c.startswith('delta_')]\n",
                        "        sort_col = delta_cols[0] if delta_cols else [c for c in wide.columns if c.startswith('beta__')][0]\n",
                        "        display(Markdown('### Human vs agent BRT contrast (concise wide view of the same significant-key set)'))\n",
                        "        if delta_cols:\n",
                        "            display(wide.sort_values(sort_col, key=lambda s: s.abs(), ascending=False))\n",
                        "        else:\n",
                        "            display(wide.sort_values(sort_col, ascending=False))\n",
                        "else:\n",
                        "    display(Markdown(f'Missing: `{brt_path}`'))\n",
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "for asset in ['tables/04_liwc_family_models_all.csv', 'tables/06_brt_descriptives_all_datasets.csv', 'tables/06_brt_primary_models_all_datasets.csv']:\n",
                        "    path = run_dir / asset\n",
                        "    if path.exists():\n",
                        "        display(Markdown(f'### {asset}'))\n",
                        "        display(pd.read_csv(path).head(40))\n",
                    ],
                }
            )
        if name == "04_simulator_comparison.ipynb":
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "meta = pd.read_json(run_dir / 'run_metadata.json', typ='series') if (run_dir / 'run_metadata.json').exists() else pd.Series(dtype=object)\n",
                        "present = set(meta.get('datasets_selected', []))\n",
                        "if not {'cogsim_agent', 'socsim_agent'}.issubset(present):\n",
                        "    display(Markdown('This notebook is only meaningful when both `cogsim_agent` and `socsim_agent` are present.'))\n",
                        "else:\n",
                        "    comp = run_dir / 'tables/comparisons/05_univariate_comparisons.csv'\n",
                        "    if comp.exists():\n",
                        "        t = pd.read_csv(comp)\n",
                        "        keep = ((t['dataset_a'] == 'cogsim_agent') & (t['dataset_b'] == 'socsim_agent')) | ((t['dataset_a'] == 'socsim_agent') & (t['dataset_b'] == 'cogsim_agent'))\n",
                        "        display(t.loc[keep].head(80))\n",
                        "    for ds in ['cogsim_agent', 'socsim_agent']:\n",
                        "        s = run_dir / 'tables' / 'calibration' / f'calibration_summary_{ds}.csv'\n",
                        "        p = run_dir / 'figures' / 'calibration' / f'calibration_panel_{ds}.png'\n",
                        "        if s.exists():\n",
                        "            display(Markdown(f'### Calibration summary: {ds}'))\n",
                        "            display(pd.read_csv(s))\n",
                        "        if p.exists():\n",
                        "            display(Image(filename=str(p)))\n",
                    ],
                }
            )
        if name == "simulation_fidelity_evaluation.ipynb":
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 1. Introduction\n"
                        "This notebook reads the dedicated simulation-fidelity outputs without recomputing any analyses.\n\n"
                        "The framework covers marginal distribution fidelity, association fidelity, and functional relationship shape fidelity."
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 2. Distribution comparison plots"],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "for asset in ['figures/comparisons/05_core_distribution_panel.png', 'figures/comparisons/05_bfi_distribution_panel.png']:\n",
                        "    path = run_dir / asset\n",
                        "    if path.exists():\n",
                        "        display(Image(filename=str(path)))\n",
                        "    else:\n",
                        "        display(Markdown(f'Missing: `{path}`'))\n",
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 3. Distribution divergence metrics table"],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "path = run_dir / 'tables' / 'simulation_fidelity' / 'distribution_divergence.csv'\n",
                        "if path.exists():\n",
                        "    display(pd.read_csv(path))\n",
                        "else:\n",
                        "    display(Markdown(f'Missing: `{path}`'))\n",
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 4. Regression slope difference results"],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "for asset in ['core_slope_difference_tests.csv', 'brt_liwc_slope_difference_tests.csv']:\n",
                        "    path = run_dir / 'tables' / 'simulation_fidelity' / asset\n",
                        "    if path.exists():\n",
                        "        display(Markdown(f'### {asset}'))\n",
                        "        display(pd.read_csv(path))\n",
                        "    else:\n",
                        "        display(Markdown(f'Missing: `{path}`'))\n",
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 5. Relationship fidelity figures"],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "meta = pd.read_json(run_dir / 'run_metadata.json', typ='series') if (run_dir / 'run_metadata.json').exists() else pd.Series(dtype=object)\n",
                        "agents = [d for d in meta.get('datasets_selected', []) if d.endswith('_agent')]\n",
                        "for agent in agents:\n",
                        "    display(Markdown(f'### Core relationship fidelity: {agent}'))\n",
                        "    fig_dir = run_dir / 'figures' / 'relationship_fidelity' / agent\n",
                        "    if not fig_dir.exists():\n",
                        "        display(Markdown(f'Missing: `{fig_dir}`'))\n",
                        "        continue\n",
                        "    for path in sorted(fig_dir.glob('*.png')):\n",
                        "        display(Image(filename=str(path)))\n",
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 6. BRT-LIWC relationship figures"],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "meta = pd.read_json(run_dir / 'run_metadata.json', typ='series') if (run_dir / 'run_metadata.json').exists() else pd.Series(dtype=object)\n",
                        "agents = [d for d in meta.get('datasets_selected', []) if d.endswith('_agent')]\n",
                        "for agent in agents:\n",
                        "    display(Markdown(f'### BRT-LIWC relationship fidelity: {agent}'))\n",
                        "    fig_dir = run_dir / 'figures' / 'brt_liwc_relationships' / agent\n",
                        "    if not fig_dir.exists():\n",
                        "        display(Markdown(f'Missing: `{fig_dir}`'))\n",
                        "        continue\n",
                        "    for path in sorted(fig_dir.glob('*.png')):\n",
                        "        display(Image(filename=str(path)))\n",
                    ],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 7. Simulation fidelity scorecard"],
                }
            )
            nb["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "path = run_dir / 'tables' / 'simulation_fidelity' / 'simulation_fidelity_scorecard.csv'\n",
                        "if path.exists():\n",
                        "    display(pd.read_csv(path))\n",
                        "else:\n",
                        "    display(Markdown(f'Missing: `{path}`'))\n",
                    ],
                }
            )
        with (root / name).open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)


def run_pipeline(
    config_path: str,
    include_datasets: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
) -> Path:
    cfg = load_yaml(Path(config_path))
    paths = cfg["paths"]
    root = Path(".")
    inputs_dir = root / paths["inputs"]
    intermediate_dir = root / paths["intermediate"]
    exports_dir = root / paths["exports"]
    reports_root = root / paths["reports"]
    resources_dir = root / "socsim" / "resources"
    ensure_dir(intermediate_dir)
    ensure_dir(exports_dir)
    ensure_dir(reports_root)
    copy_resource_inputs(inputs_dir, resources_dir)
    answer_keys = load_yaml(resources_dir / "answer_keys.yaml")
    score_cfg = load_yaml(resources_dir / "config.yaml")
    have_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    datasets_cfg = cfg.get("datasets", {})
    cfg_include = datasets_cfg.get("include", []) or []
    cfg_exclude = datasets_cfg.get("exclude", []) or []
    effective_include = include_datasets if include_datasets is not None else cfg_include
    effective_exclude = exclude_datasets if exclude_datasets is not None else cfg_exclude

    required = load_required_inputs(inputs_dir)
    human = harmonise_human(required["human"])
    human = compute_liwc_families(human, score_cfg["liwc"])
    synthetic = harmonise_synthetic(required["synthetic"])
    human.to_csv(intermediate_dir / "human_person.csv", index=False)
    synthetic.to_csv(intermediate_dir / "synthetic_person.csv", index=False)
    socsim_run_label: str | None = None
    socsim_input_files: dict[str, str] = {}

    process_cogsim = _dataset_requested("cogsim_agent", effective_include, effective_exclude)
    cogsim_raw = load_optional_cogsim(inputs_dir) if process_cogsim else None
    if cogsim_raw is None:
        print("COGSIM dataset not selected. Skipping cogsim ingestion." if not process_cogsim else "COGSIM inputs absent. Skipping cogsim ingestion.")
        cogsim_person = None
        cogsim_items = None
    else:
        cogsim_person = harmonise_cogsim_person(cogsim_raw["person"])
        cogsim_items = harmonise_cogsim_items(cogsim_raw["items"])
        cogsim_person.to_csv(intermediate_dir / "cogsim_person_base.csv", index=False)
        cogsim_items.to_csv(intermediate_dir / "cogsim_items_base.csv", index=False)

    process_socsim = _dataset_requested("socsim_agent", effective_include, effective_exclude)
    socsim_raw = load_optional_socsim(inputs_dir) if process_socsim else None
    if socsim_raw is None:
        print("SOCSIM dataset not selected. Skipping socsim ingestion." if not process_socsim else "SOCSIM inputs absent. Skipping socsim ingestion.")
        socsim_person = None
        socsim_items = None
    else:
        socsim_run_label = _clean_run_label(socsim_raw.get("run_label"))
        socsim_input_files = {
            "person": str(socsim_raw["person_path"]),
            "items": str(socsim_raw["items_path"]),
        }
        socsim_person = harmonise_socsim_person(socsim_raw["person"])
        socsim_items = harmonise_socsim_items(socsim_raw["items"])
        socsim_person["socsim_run_label"] = socsim_run_label or "socsim_agent"
        socsim_person.to_csv(intermediate_dir / "socsim_person_base.csv", index=False)
        socsim_items.to_csv(intermediate_dir / "socsim_items_base.csv", index=False)

    agent_person = {}
    agent_items = {}
    if cogsim_person is not None and cogsim_items is not None:
        items_scored_path = intermediate_dir / "cogsim_items_scored.csv"
        person_scored_path = intermediate_dir / "cogsim_person_scored.csv"
        scoring_meta = _agent_scoring_cache_metadata("cogsim_agent", cogsim_person, cogsim_items)
        existing_scored_items = read_csv_keep_text(items_scored_path, AGENT_ITEM_TEXT_COLS) if items_scored_path.exists() and _cache_metadata_matches(items_scored_path, scoring_meta) else None
        if items_scored_path.exists() and person_scored_path.exists():
            scored_items_cached = existing_scored_items if existing_scored_items is not None else read_csv_keep_text(items_scored_path, AGENT_ITEM_TEXT_COLS)
            person_scored_cached = pd.read_csv(person_scored_path)
            if _crt_cache_is_valid(scored_items_cached) and _cache_metadata_matches(person_scored_path, scoring_meta):
                scored_items = add_item_response_categories(scored_items_cached)
                if not {"response_category", "final_response_category"}.issubset(scored_items_cached.columns):
                    scored_items.to_csv(items_scored_path, index=False)
                    _write_cache_metadata(items_scored_path, scoring_meta)
                if _crt_person_cache_has_category_outputs(person_scored_cached):
                    person_scored = person_scored_cached
                    print("Reusing cached COGSIM CRT/Likert scoring outputs from data/intermediate.")
                else:
                    print("Refreshing cached COGSIM person CRT totals with response-category columns (no OpenAI calls).")
                    person_scored = compute_crt_totals(scored_items, cogsim_person)
                    likert = score_likert_bfi_nfc(cogsim_items, score_cfg)
                    person_scored = person_scored.merge(likert, on="entity_id", how="left")
                    person_scored["dataset"] = "cogsim_agent"
                    person_scored.to_csv(person_scored_path, index=False)
                    _write_cache_metadata(person_scored_path, scoring_meta)
            else:
                print("Cached COGSIM CRT/Likert scoring is incomplete, invalid, or from different inputs; filling missing labels with OpenAI.")
                if not have_openai_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required to refresh invalid/stale COGSIM CRT cache. "
                        "Set OPENAI_API_KEY and rerun."
                    )
                _assert_openai_runtime_available()
                scored_items = score_crt_items_with_llm(cogsim_items, answer_keys, existing_scored=existing_scored_items)
                scored_items.to_csv(items_scored_path, index=False)
                _write_cache_metadata(items_scored_path, scoring_meta)
                _assert_crt_scoring_succeeded(scored_items, "COGSIM")
                person_scored = compute_crt_totals(scored_items, cogsim_person)
                likert = score_likert_bfi_nfc(cogsim_items, score_cfg)
                person_scored = person_scored.merge(likert, on="entity_id", how="left")
                person_scored["dataset"] = "cogsim_agent"
                person_scored.to_csv(person_scored_path, index=False)
                _write_cache_metadata(person_scored_path, scoring_meta)
        else:
            if not have_openai_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for first-run agent CRT scoring. "
                    "Set OPENAI_API_KEY or provide cached *_items_scored.csv and *_person_scored.csv in data/intermediate."
                )
            _assert_openai_runtime_available()
            scored_items = score_crt_items_with_llm(cogsim_items, answer_keys, existing_scored=existing_scored_items)
            scored_items.to_csv(items_scored_path, index=False)
            _write_cache_metadata(items_scored_path, scoring_meta)
            _assert_crt_scoring_succeeded(scored_items, "COGSIM")
            person_scored = compute_crt_totals(scored_items, cogsim_person)
            likert = score_likert_bfi_nfc(cogsim_items, score_cfg)
            person_scored = person_scored.merge(likert, on="entity_id", how="left")
            person_scored["dataset"] = "cogsim_agent"
            person_scored.to_csv(person_scored_path, index=False)
            _write_cache_metadata(person_scored_path, scoring_meta)
        agent_person["cogsim_agent"] = person_scored
        agent_items["cogsim_agent"] = cogsim_items

    if socsim_person is not None and socsim_items is not None:
        items_scored_path = intermediate_dir / "socsim_items_scored.csv"
        person_scored_path = intermediate_dir / "socsim_person_scored.csv"
        scoring_meta = _agent_scoring_cache_metadata("socsim_agent", socsim_person, socsim_items)
        existing_scored_items = read_csv_keep_text(items_scored_path, AGENT_ITEM_TEXT_COLS) if items_scored_path.exists() and _cache_metadata_matches(items_scored_path, scoring_meta) else None
        if items_scored_path.exists() and person_scored_path.exists():
            scored_items_cached = existing_scored_items if existing_scored_items is not None else read_csv_keep_text(items_scored_path, AGENT_ITEM_TEXT_COLS)
            person_scored_cached = pd.read_csv(person_scored_path)
            if _crt_cache_is_valid(scored_items_cached) and _cache_metadata_matches(person_scored_path, scoring_meta):
                scored_items = add_item_response_categories(scored_items_cached)
                if not {"response_category", "final_response_category"}.issubset(scored_items_cached.columns):
                    scored_items.to_csv(items_scored_path, index=False)
                    _write_cache_metadata(items_scored_path, scoring_meta)
                if _crt_person_cache_has_category_outputs(person_scored_cached):
                    person_scored = person_scored_cached
                    print("Reusing cached SOCSIM CRT/Likert scoring outputs from data/intermediate.")
                else:
                    print("Refreshing cached SOCSIM person CRT totals with response-category columns (no OpenAI calls).")
                    person_scored = compute_crt_totals(scored_items, socsim_person)
                    likert = score_likert_bfi_nfc(socsim_items, score_cfg)
                    person_scored = person_scored.merge(likert, on="entity_id", how="left")
                    person_scored["dataset"] = "socsim_agent"
                    person_scored.to_csv(person_scored_path, index=False)
                    _write_cache_metadata(person_scored_path, scoring_meta)
            else:
                print("Cached SOCSIM CRT/Likert scoring is incomplete, invalid, or from different inputs; filling missing labels with OpenAI.")
                if not have_openai_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required to refresh invalid/stale SOCSIM CRT cache. "
                        "Set OPENAI_API_KEY and rerun."
                    )
                _assert_openai_runtime_available()
                scored_items = score_crt_items_with_llm(socsim_items, answer_keys, existing_scored=existing_scored_items)
                scored_items.to_csv(items_scored_path, index=False)
                _write_cache_metadata(items_scored_path, scoring_meta)
                _assert_crt_scoring_succeeded(scored_items, "SOCSIM")
                person_scored = compute_crt_totals(scored_items, socsim_person)
                likert = score_likert_bfi_nfc(socsim_items, score_cfg)
                person_scored = person_scored.merge(likert, on="entity_id", how="left")
                person_scored["dataset"] = "socsim_agent"
                person_scored.to_csv(person_scored_path, index=False)
                _write_cache_metadata(person_scored_path, scoring_meta)
        else:
            if not have_openai_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for first-run agent CRT scoring. "
                    "Set OPENAI_API_KEY or provide cached *_items_scored.csv and *_person_scored.csv in data/intermediate."
                )
            _assert_openai_runtime_available()
            scored_items = score_crt_items_with_llm(socsim_items, answer_keys, existing_scored=existing_scored_items)
            scored_items.to_csv(items_scored_path, index=False)
            _write_cache_metadata(items_scored_path, scoring_meta)
            _assert_crt_scoring_succeeded(scored_items, "SOCSIM")
            person_scored = compute_crt_totals(scored_items, socsim_person)
            likert = score_likert_bfi_nfc(socsim_items, score_cfg)
            person_scored = person_scored.merge(likert, on="entity_id", how="left")
            person_scored["dataset"] = "socsim_agent"
            person_scored.to_csv(person_scored_path, index=False)
            _write_cache_metadata(person_scored_path, scoring_meta)
        agent_person["socsim_agent"] = person_scored
        agent_items["socsim_agent"] = socsim_items

    agent_brt_text_available = {ds: _agent_items_have_brt_text(items) for ds, items in agent_items.items()}
    human_has_brt = _human_has_brt_coded_columns(human)
    skipped_brt_steps: list[str] = []
    brt_exclusions_applied: dict[str, list[str]] = {}

    if human_has_brt:
        validate_human_brt_coded(human)
        human.to_csv(intermediate_dir / "human_person_brt_coded.csv", index=False)
    else:
        skipped_brt_steps.append("human_brt_validation")
        print("Human BRT-coded columns absent. Skipping human BRT-specific analyses.")

    for ds in list(agent_person.keys()):
        if not agent_brt_text_available.get(ds, False):
            skipped_brt_steps.append(f"{ds}:agent_brt_coding")
            print(f"{ds} has no BRT text/items. Skipping agent BRT coding.")
            continue
        coded_path = intermediate_dir / f"{ds}_person_brt_coded.csv"
        brt_meta = _agent_brt_cache_metadata(ds, agent_person[ds], agent_items[ds])
        if coded_path.exists():
            coded_cached = pd.read_csv(coded_path)
            require_columns(coded_cached, ["entity_id"], coded_path.name)
            brt_cache_meta_matches = _cache_metadata_matches(coded_path, brt_meta)
            if _brt_cache_is_valid(coded_cached) and brt_cache_meta_matches:
                coded = coded_cached
                print(f"Reusing cached {ds} BRT coding output from data/intermediate.")
            else:
                print(f"Cached {ds} BRT coding is invalid or from different inputs; recomputing with OpenAI.")
                if not have_openai_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required to refresh invalid/stale BRT cache. "
                        "Set OPENAI_API_KEY and rerun."
                    )
                _assert_openai_runtime_available()
                existing_coded = coded_cached if brt_cache_meta_matches else None
                coded = code_brt_for_agents(agent_person[ds], agent_items[ds], existing_coded=existing_coded)
                coded.to_csv(coded_path, index=False)
                _write_cache_metadata(coded_path, brt_meta)
                _assert_brt_coding_succeeded(coded, ds)
        else:
            if not have_openai_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for first-run agent BRT coding. "
                    "Set OPENAI_API_KEY or provide cached *_person_brt_coded.csv in data/intermediate."
                )
            _assert_openai_runtime_available()
            coded = code_brt_for_agents(agent_person[ds], agent_items[ds])
            coded.to_csv(coded_path, index=False)
            _write_cache_metadata(coded_path, brt_meta)
            _assert_brt_coding_succeeded(coded, ds)
        # CR-13B STEP 3b — PI ruling A (23 Sep 2026, CR-13R): drop the disclosed
        # exclusions from this run's BRT frame before ANY BRT analysis, table or LIWC
        # model. The rows stay (CRT analyses are unaffected); only BRT columns are voided.
        coded, excluded_ids = apply_brt_exclusions(coded, socsim_run_label or "")
        if excluded_ids:
            brt_exclusions_applied[ds] = excluded_ids
            print(
                f"{ds}: PI-ruled BRT exclusions applied for {socsim_run_label} - "
                f"{len(excluded_ids)} persona(s) voided: {', '.join(excluded_ids)}. {EXCLUSION_NOTE}"
            )
        agent_person[ds] = coded

    agent_liwc_available: dict[str, bool] = {}
    agent_liwc_export_paths: dict[str, str] = {}
    agent_liwc_import_paths: dict[str, str] = {}
    for ds in list(agent_person.keys()):
        if not agent_brt_text_available.get(ds, False):
            skipped_brt_steps.append(f"{ds}:agent_liwc_export_import")
            agent_liwc_available[ds] = False
            print(f"{ds} has no BRT text/items. Skipping LIWC text export/import.")
            continue
        liwc_label = _liwc_run_label(ds, socsim_run_label)
        export_path = export_agent_brt_texts(agent_items[ds], exports_dir, ds, run_label=liwc_label)
        agent_liwc_export_paths[ds] = str(export_path)
        liwc_candidates = _liwc_output_candidates(inputs_dir, exports_dir, ds, liwc_label)
        liwc_in = next((path for path in liwc_candidates if path.exists()), liwc_candidates[0])
        agent_liwc_import_paths[ds] = str(liwc_in)
        if liwc_in.exists():
            print(f"LIWC output detected for {ds}: {_format_path_for_log(liwc_in)}. Importing and computing LIWC family features.")
        else:
            expected = ", ".join(_format_path_for_log(path) for path in liwc_candidates[:2])
            print(
                f"LIWC output not found for {ds}. Exported text to {_format_path_for_log(export_path)}; "
                f"save LIWC results as {expected} and rerun to enable agent LIWC analyses."
            )
        liwc_joined = join_liwc_outputs(agent_person[ds], liwc_in, score_cfg["liwc"])
        liwc_joined.to_csv(intermediate_dir / f"{ds}_person_scored_liwc.csv", index=False)
        agent_person[ds] = liwc_joined
        agent_liwc_available[ds] = liwc_in.exists()

    run_id = cfg.get("analysis_run_id", "auto")
    if run_id == "auto":
        run_id = analysis_run_id()
        if socsim_run_label:
            run_id = f"{run_id}_{socsim_run_label}"
    report_dir = reports_root / run_id
    by_dataset = {"human": human, "synthetic": synthetic}
    by_dataset.update(agent_person)
    by_dataset = _apply_dataset_filters(by_dataset, effective_include, effective_exclude)
    if not by_dataset:
        raise ValueError("No datasets selected after applying include/exclude filters.")
    all_ds = list(by_dataset.keys())
    _create_report_dirs(report_dir, all_ds)

    # CR-13B STEP 4: persist each dataset's analysis-ready person frame, so the
    # camera-ready assets and the ANALYSIS_PLAN Part II models read the exact frame
    # this run scored (BRT exclusions already applied; LIWC already joined) instead
    # of the single rotating data/intermediate/ copy.
    for ds, frame in by_dataset.items():
        frame.to_csv(report_dir / "tables" / "by_dataset" / ds / "00_person_frame.csv", index=False)

    desc = make_descriptives_overview(by_dataset)
    desc.to_csv(report_dir / "tables" / "00_descriptives_overview.csv", index=False)
    plot_descriptives_overview(desc, report_dir / "figures" / "00_descriptives_overview.png")
    demographic_desc = desc[desc["section"] == "demographic"].copy() if not desc.empty and "section" in desc.columns else pd.DataFrame()
    demographic_desc.to_csv(report_dir / "tables" / "00_demographic_descriptives.csv", index=False)
    plot_demographic_overview(demographic_desc, report_dir / "figures" / "00_demographic_overview.png")

    corr_mats = {}
    corr_p_mats = {}
    corr_p_mats_fdr = {}
    corr_n_mats = {}
    for ds, frame in by_dataset.items():
        corr = build_corr_matrix(frame, ds, method="spearman")
        pmat, nmat = build_corr_p_matrix(frame, ds, method="spearman", fdr=False)
        pmat_fdr, _ = build_corr_p_matrix(frame, ds, method="spearman", fdr=True)
        corr_mats[ds] = corr
        corr_p_mats[ds] = pmat
        corr_p_mats_fdr[ds] = pmat_fdr
        corr_n_mats[ds] = nmat
        corr.to_csv(report_dir / "tables" / "by_dataset" / ds / "01_correlations.csv")
        pmat.to_csv(report_dir / "tables" / "by_dataset" / ds / "01_correlations_pvalues.csv")
        pmat_fdr.to_csv(report_dir / "tables" / "by_dataset" / ds / "01_correlations_pvalues_fdr.csv")
        nmat.to_csv(report_dir / "tables" / "by_dataset" / ds / "01_correlations_pairwise_n.csv")
        plot_corr_matrix(
            corr,
            ds,
            report_dir / "figures" / "by_dataset" / ds / "01_correlations_heatmap.png",
            p_matrix=pmat,
            n_matrix=nmat,
            correction_label="uncorrected p-values",
        )
        plot_corr_matrix(
            corr,
            ds,
            report_dir / "figures" / "by_dataset" / ds / "01_correlations_heatmap_fdr.png",
            p_matrix=pmat_fdr,
            n_matrix=nmat,
            correction_label="FDR-adjusted p-values",
        )
    if "human" in corr_mats:
        for ds in [k for k in corr_mats.keys() if k != "human"]:
            diff = corr_diff_vs_human(corr_mats["human"], corr_mats[ds])
            diff.to_csv(report_dir / "tables" / "by_dataset" / ds / "01_correlations_diff_vs_human.csv")
            plot_corr_diff(
                diff,
                ds,
                report_dir / "figures" / "by_dataset" / ds / "01_correlations_diff_vs_human.png",
                p_matrix=corr_p_mats.get(ds),
                n_matrix=corr_n_mats.get(ds),
                correction_label="uncorrected p-values",
            )
            plot_corr_diff(
                diff,
                ds,
                report_dir / "figures" / "by_dataset" / ds / "01_correlations_diff_vs_human_fdr.png",
                p_matrix=corr_p_mats_fdr.get(ds),
                n_matrix=corr_n_mats.get(ds),
                correction_label="FDR-adjusted p-values",
            )

    model_tables = []
    for ds, frame in by_dataset.items():
        mt = run_fixed_models(frame, ds)
        if not mt.empty:
            mt.to_csv(report_dir / "tables" / "by_dataset" / ds / "02_models_fixed.csv", index=False)
            model_tables.append(mt)
            plot_primary_model_coefficients(mt, ds, report_dir / "figures" / "by_dataset" / ds / "02_models_primary_coefficients.png")
        plot_focused_relationships(frame, ds, report_dir / "figures" / "by_dataset" / ds / "03_focused_relationships.png")
    if model_tables:
        pd.concat(model_tables, ignore_index=True).to_csv(report_dir / "tables" / "04_models_all_datasets.csv", index=False)

    liwc_tables = []
    liwc_families = score_cfg.get("liwc", {}).get("families", {})
    for ds in [k for k in by_dataset.keys() if k != "synthetic"]:
        lt = run_liwc_family_models(by_dataset[ds], ds, liwc_families=liwc_families)
        if not lt.empty:
            lt.to_csv(report_dir / "tables" / "by_dataset" / ds / "04_liwc_family_models.csv", index=False)
            liwc_tables.append(lt)
            plot_liwc_family_models(
                lt,
                ds,
                report_dir / "figures" / "by_dataset" / ds / "04_liwc_family_models.png",
                title="LIWC outcomes by family with cognitive predictors",
            )
            print(f"LIWC models completed for {ds}: {len(lt)} coefficient rows.")
        elif ds.endswith("_agent") and not agent_liwc_available.get(ds, False):
            print(f"LIWC models skipped for {ds}: no imported LIWC output was detected.")
        elif ds.endswith("_agent"):
            print(f"LIWC models skipped for {ds}: imported LIWC output did not contain usable configured LIWC variables.")
    if liwc_tables:
        pd.concat(liwc_tables, ignore_index=True).to_csv(report_dir / "tables" / "04_liwc_family_models_all.csv", index=False)

    brt_desc_tables = []
    brt_model_tables = []
    brt_report_datasets = [ds for ds in by_dataset.keys() if ds != "synthetic" and _dataset_has_brt_outcomes(by_dataset[ds])]
    skipped_brt_report_datasets = [ds for ds in by_dataset.keys() if ds != "synthetic" and ds not in brt_report_datasets]
    if skipped_brt_report_datasets:
        skipped_brt_steps.extend(f"{ds}:brt_reports" for ds in skipped_brt_report_datasets)
    for ds in brt_report_datasets:
        bdesc = brt_descriptives(by_dataset[ds], ds)
        if not bdesc.empty:
            bdesc.to_csv(report_dir / "tables" / "by_dataset" / ds / "06_brt_descriptives.csv", index=False)
            brt_desc_tables.append(bdesc)
        bmods = run_brt_models(by_dataset[ds], ds)
        if not bmods.empty:
            bmods.to_csv(report_dir / "tables" / "by_dataset" / ds / "06_brt_primary_models.csv", index=False)
            brt_model_tables.append(bmods)
            plot_brt_primary_forest(
                bmods,
                ds,
                report_dir / "figures" / "by_dataset" / ds / "06_brt_primary_forest.png",
            )
        plot_brt_decision_distribution(
            by_dataset[ds],
            ds,
            report_dir / "figures" / "by_dataset" / ds / "06_brt_decision_distribution.png",
        )
    if brt_desc_tables:
        pd.concat(brt_desc_tables, ignore_index=True).to_csv(report_dir / "tables" / "06_brt_descriptives_all_datasets.csv", index=False)
    if brt_model_tables:
        brt_models_all = pd.concat(brt_model_tables, ignore_index=True)
        brt_models_all.to_csv(report_dir / "tables" / "06_brt_primary_models_all_datasets.csv", index=False)
        brt_warning_summary = _summarize_brt_model_warnings(brt_models_all)
        if brt_warning_summary:
            print(brt_warning_summary)
        else:
            print("BRT logistic models completed without numerical stability warnings.")

    comp = pairwise_univariate_comparisons(by_dataset)
    comp.to_csv(report_dir / "tables" / "comparisons" / "05_univariate_comparisons.csv", index=False)
    core_vars = CORE_DISTRIBUTION_VARS
    bfi_vars = BFI_DISTRIBUTION_VARS
    overlap = overlap_summary(by_dataset, core_vars + bfi_vars)
    overlap.to_csv(report_dir / "tables" / "comparisons" / "05_overlap_summary.csv", index=False)
    plot_overlap_panel(by_dataset, core_vars, report_dir / "figures" / "comparisons" / "05_core_distribution_panel.png", "Core Discrete Distribution Panel")
    plot_overlap_panel(by_dataset, bfi_vars, report_dir / "figures" / "comparisons" / "05_bfi_distribution_panel.png", "BFI Discrete Distribution Panel")

    crt_category_long = crt_response_category_long(by_dataset)
    crt_category_dist = response_category_distribution(crt_category_long)
    crt_category_by_level = response_category_distribution_by_specified_level(crt_category_long)
    crt_category_divergence = response_category_divergence(crt_category_long)
    crt_category_long.to_csv(report_dir / "tables" / "comparisons" / "05_crt_response_categories_long.csv", index=False)
    crt_category_dist.to_csv(report_dir / "tables" / "comparisons" / "05_crt_response_category_distribution.csv", index=False)
    crt_category_by_level.to_csv(report_dir / "tables" / "comparisons" / "05_crt_response_category_by_specified_level.csv", index=False)
    crt_category_divergence.to_csv(report_dir / "tables" / "comparisons" / "05_crt_response_category_divergence.csv", index=False)
    plot_response_category_distribution(
        crt_category_dist,
        report_dir / "figures" / "comparisons" / "05_crt_response_category_distribution.png",
    )
    plot_response_category_by_specified_level(
        crt_category_by_level,
        report_dir / "figures" / "comparisons" / "05_crt_response_category_by_specified_level.png",
    )

    fidelity_dir = report_dir / "tables" / "simulation_fidelity"
    distribution_divergence = compute_distribution_divergence(by_dataset)
    distribution_divergence.to_csv(fidelity_dir / "distribution_divergence.csv", index=False)

    core_slope_tests, brt_liwc_slope_tests = run_slope_difference_tests(by_dataset, liwc_families=liwc_families)
    core_slope_tests.to_csv(fidelity_dir / "core_slope_difference_tests.csv", index=False)

    core_bin_table, core_rmse_table = build_core_binned_relationships(by_dataset)
    core_bin_table.to_csv(fidelity_dir / "core_binned_relationships.csv", index=False)
    core_rmse_table.to_csv(fidelity_dir / "core_binned_rmse.csv", index=False)

    selected_brt_agent_datasets = [
        ds
        for ds in by_dataset.keys()
        if ds.endswith("_agent") and agent_brt_text_available.get(ds, False) and _dataset_has_brt_outcomes(by_dataset[ds])
    ]
    run_brt_liwc_fidelity = bool(selected_brt_agent_datasets) and human_has_brt and "human" in by_dataset
    if run_brt_liwc_fidelity:
        brt_liwc_bin_table, brt_liwc_rmse_table = build_brt_liwc_binned_relationships(by_dataset, liwc_families=liwc_families)
    else:
        skipped_brt_steps.append("brt_liwc_relationship_fidelity")
        print("Skipping BRT-LIWC relationship fidelity analyses because no selected agent dataset has usable BRT text/outcomes.")
        brt_liwc_slope_tests = pd.DataFrame(
            columns=["predictor", "outcome", "comparison", "slope_human", "slope_sim", "interaction_beta", "interaction_p", "ci_low", "ci_high"]
        )
        brt_liwc_bin_table = pd.DataFrame(
            columns=["predictor", "outcome", "dataset", "bin", "bin_center", "mean_outcome", "ci_low", "ci_high"]
        )
        brt_liwc_rmse_table = pd.DataFrame(columns=["predictor", "outcome", "comparison", "rmse_bin"])

    brt_liwc_slope_tests.to_csv(fidelity_dir / "brt_liwc_slope_difference_tests.csv", index=False)
    brt_liwc_bin_table.to_csv(fidelity_dir / "brt_liwc_binned_relationships.csv", index=False)
    brt_liwc_rmse_table.to_csv(fidelity_dir / "brt_liwc_binned_rmse.csv", index=False)

    for agent_ds in [ds for ds in by_dataset.keys() if ds.endswith("_agent")]:
        plot_core_relationship_fidelity(core_bin_table, agent_ds, report_dir / "figures" / "relationship_fidelity")
        if run_brt_liwc_fidelity and agent_ds in selected_brt_agent_datasets:
            plot_brt_liwc_relationship_fidelity(brt_liwc_bin_table, agent_ds, report_dir / "figures" / "brt_liwc_relationships")

    scorecard = build_simulation_fidelity_scorecard(
        by_dataset=by_dataset,
        distribution_df=distribution_divergence,
        core_slopes_df=core_slope_tests,
        brt_liwc_slopes_df=brt_liwc_slope_tests,
        core_rmse_df=core_rmse_table,
        brt_liwc_rmse_df=brt_liwc_rmse_table,
    )
    scorecard.to_csv(fidelity_dir / "simulation_fidelity_scorecard.csv", index=False)

    for ds in [k for k in agent_person.keys() if k in by_dataset]:
        pair_df, summary_df, err_df = calibration_tables(by_dataset[ds], ds)
        pair_df.to_csv(report_dir / "tables" / "calibration" / f"calibration_pairs_{ds}.csv", index=False)
        summary_df.to_csv(report_dir / "tables" / "calibration" / f"calibration_summary_{ds}.csv", index=False)
        err_df.to_csv(report_dir / "tables" / "calibration" / f"calibration_per_agent_errors_{ds}.csv", index=False)
        plot_calibration_scatter(by_dataset[ds], ds, report_dir / "figures" / "calibration")
        plot_calibration_panel(by_dataset[ds], ds, report_dir / "figures" / "calibration" / f"calibration_panel_{ds}.png")
        plot_abs_error_hists(err_df, ds, report_dir / "figures" / "calibration")

    _write_notebooks(root / "socsim" / "notebooks", run_id)

    json_write(
        report_dir / "run_metadata.json",
        {
            "analysis_run_id": run_id,
            "datasets_present": ["human", "synthetic"] + list(agent_person.keys()),
            "datasets_selected": list(by_dataset.keys()),
            "datasets_include": effective_include,
            "datasets_exclude": effective_exclude,
            "socsim_run_label": socsim_run_label,
            "socsim_input_files": socsim_input_files,
            "agent_brt_text_available": agent_brt_text_available,
            "agent_liwc_export_import_enabled": agent_liwc_available,
            "agent_liwc_export_paths": agent_liwc_export_paths,
            "agent_liwc_import_paths": agent_liwc_import_paths,
            "dataset_brt_outcomes_available": {ds: _dataset_has_brt_outcomes(frame) for ds, frame in by_dataset.items()},
            "skipped_brt_steps": skipped_brt_steps,
            "brt_exclusions_applied": brt_exclusions_applied,
            "brt_exclusions_source": str(exclusions_path()),
            "brt_exclusions_note": EXCLUSION_NOTE if brt_exclusions_applied else "",
            "crt_response_category_columns": {
                "item_level_suffix": "_response_category",
                "values": ["correct", "intuitive_incorrect", "other_incorrect", "missing_or_unparseable"],
                "totals": [
                    "crt2_correct_total",
                    "crt2_intuitive_total",
                    "crt2_other_total",
                    "bcrt_correct_total",
                    "bcrt_intuitive_total",
                    "bcrt_other_total",
                ],
                "comparison_tables": [
                    "tables/comparisons/05_crt_response_category_distribution.csv",
                    "tables/comparisons/05_crt_response_category_by_specified_level.csv",
                    "tables/comparisons/05_crt_response_category_divergence.csv",
                ],
            },
            "key_variables": KEY_VARS,
            "demographic_descriptive_table": "tables/00_demographic_descriptives.csv",
        },
    )
    brt_skip_text = ""
    if skipped_brt_steps:
        brt_skip_text = "- BRT/LIWC gating this run: "
        brt_skip_text += ", ".join(skipped_brt_steps)
        brt_skip_text += "\n"
    analyses_text = (
        "# Analyses\n\n"
        "This run uses dataset filtering (`datasets.include` / `datasets.exclude`) before all outputs.\n\n"
        f"- Datasets selected: {', '.join(by_dataset.keys())}\n"
        f"- SOCSIM run label: {socsim_run_label or '-'}\n"
        "- Primary CRT outcome modelling: **binomial GLM** for CRT2/BCRT totals (successes=k, failures=4-k).\n"
        "- OLS models are retained as **sensitivity-only** output.\n"
        "- Main comparison figures use **stacked discrete proportions + mean ±95% CI** panels.\n"
        "- CRT2/bCRT response categories are surfaced from item-level evaluator labels as `correct`, `intuitive_incorrect`, `other_incorrect`, or `missing_or_unparseable`; per-person totals are exported as `crt2_correct_total`, `crt2_intuitive_total`, `crt2_other_total`, `bcrt_correct_total`, `bcrt_intuitive_total`, and `bcrt_other_total`.\n"
        "- CRT2/bCRT category comparison outputs include source/run_id distributions, distributions by specified CRT level when available, and human-vs-simulation divergence metrics in `tables/comparisons/05_crt_response_category_*.csv`.\n"
        "- Correlation matrices use **Spearman** associations; both uncorrected and FDR-adjusted significance overlays are exported.\n"
        "- LIWC analyses model **LIWC category outcomes** within each family using cognitive predictors (human CRT2 adjusted + BCRT observed; agent CRT2/BCRT realised), with family-wise FDR correction.\n"
        "- Simulation fidelity exports add **distribution divergence**, **pooled interaction slope-difference tests**, **binned relationship RMSE**, dedicated **relationship fidelity figures**, and a **simulation fidelity scorecard**.\n"
        "- Calibration keeps specified->realised fidelity, including BCRT proxy calibration where `CRT2_Total_adj_specified` is used as proxy specified cognitive-reflection parameter for `BCRT_Total_realised` (BCRT is not directly specified).\n"
        f"{brt_skip_text}"
    )
    (report_dir / "ANALYSES.md").write_text(analyses_text, encoding="utf-8")
    return report_dir
