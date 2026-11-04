# CogSuite

Code for the paper *Beyond High and Low: Evaluating Graded Cognitive Diversity in LLM Persona Simulations* (NeurIPS 2026, Evaluations & Datasets track).

Data release (OSF): https://osf.io/h8d5s/ · Archived release (Zenodo): https://doi.org/10.5281/zenodo.23910425 · Code licence: MIT (`LICENSE`) · Data licence: CC BY 4.0

## What the benchmark tests

CogSuite asks whether LLM agents that are given a person's profile (demographics and trait scores) reproduce the *graded* variation in how people reason, rather than collapsing toward generic competence. Agents and people answer the same cognition suite:

- two cognitive reflection tests: CRT2, and a lower-exposure bespoke CRT (bCRT), with decoy items;
- Need for Cognition (NFC-18) and Big Five personality (BFI-10);
- two open-ended Behavioural Reflection Test (BRT) scenarios, a restaurant booking and a medication decision, coded with a fixed rubric.

Agents are conditioned on profiles taken from people, either each person's own profile or one drawn from a synthetic bank fitted to them, written either as numbers (numeric format) or as words (lexical format). Their answers are compared with people's on five fidelity targets: marginal distributions, calibration (are specified traits realised?), associations among constructs, transfer to downstream BRT behaviour, and error types (correct, intuitive-lure, other incorrect, missing).

The human sample, the bCRT and the BRT are presented in the companion study (arXiv 2607.07961), and participant-level data stay with that study under its ethics approval.

## Layout

| Folder | What it holds |
|---|---|
| `simulate/` | The simulator (`output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py`, with its runbook `README_simulation_code_iter_1.md`), the survey item files, the prompt configuration (`data_fitting/persona_trait_experiment/prompt_config/`), the CRT2 level cutoffs, and the stored configuration of every run in the paper (`run_configs/`, with one command per run in `run_configs/RUNS.md`). |
| `score/` | CRT scoring (`scoring_v2/`: the complete answer keys, normalisation and the deterministic-first scorer with its tests) and the BRT rubric and coding prompt (`brt_v3/prompt_v3.txt`, field definitions in `brt_v3/schema.py`). |
| `analyse/` | The analysis package (`socsim/`), the script that rebuilds every figure and table (`scripts/build_neurips_camera_ready_assets.py`), and the interval, trait-structure and per-level prevalence scripts it depends on. |
| `data/synthetic/` | The code that generated the synthetic profile bank (`05_synth_personas.py`, with `utils.py` and `config/config.yaml`) and the code that drew the configuration-run panel (`generate_rebuttal_persona_subsets.py`). |

Data files are not in this repository. They are in the data release on OSF (https://osf.io/h8d5s/), which has its own README.

## Quickstart: a smoke test with no API key

Use Python 3.11. `environment.yml` is the reference environment, with the exact builds that made the paper's figures and tables; `requirements.txt` installs the same versions with pip.

```bash
git clone https://github.com/sion-w/CogSuite && cd CogSuite
conda env create -f environment.yml && conda activate cogsuite
# or: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Download the data release from OSF (https://osf.io/h8d5s/) and unpack it into data/release/ (git-ignored).
# The simulator reads the synthetic profile bank under its contract name:
cp data/release/synthetic/synthetic_personas_v2_20260729_s42_n5000.csv \
   simulate/data_fitting/persona_trait_experiment/synthetic_personas.csv

# Pseudo mode: deterministic placeholder answers, no API calls, no key needed.
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py \
  --use-pseudo --max-personas 1 --tests-included writing \
  --output-dir simulation_outputs/smoke_test
```

The run writes `per_persona_results.csv`, `per_item_results.csv`, `api_calls.csv` and `metrics_and_config.json` to `simulate/simulation_outputs/smoke_test/`.

## Running the experiments

`simulate/run_configs/RUNS.md` gives the exact command for each run in the paper: the seven full-battery runs and the configuration runs. Each command was written out from that run's stored configuration (`simulate/run_configs/<run>.json`). Real runs make paid API calls. Put keys in the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or the variable named by `--openai-key-env`), or in a `.env` file in `simulate/`, which git ignores. Never commit a key.

The settings that define a run:

- **Model and provider.** Use `--use-anthropic --anthropic-model <id>` for the Anthropic Messages API. Use `--use-openai --openai-model <id>` for OpenAI and for OpenAI-compatible providers. For the compatible providers, add `--openai-base-url <url> --openai-api-style chat_completions` and `--openai-key-env <VAR>`. `--openai-max-output-tokens` and `--anthropic-max-output-tokens` set the output limit.
- **Thinking or reasoning setting.** Use `--anthropic-thinking-json '{"type": "disabled"}'` for Anthropic models. For OpenAI models, use `--openai-extra-params-json` (for example `'{"reasoning": {"effort": "none"}}'`). For providers that take provider-specific fields in the request body, use `--openai-extra-body-json` (for example the Gemini `thinking_level` and DeepSeek `thinking` objects in `RUNS.md`). Leaving these out keeps the provider default.
- **Prompt format.** Use `--persona-spec-mode numeric_only` with `--persona-prompt-version winner_v2_numeric_only`, or `lexical_only` with `winner_v2_lexical_only`. `--task-prompt-version task_v1` throughout.
- **Persona source.** By default the simulator reads the synthetic bank (`synthetic_personas.csv`, see the quickstart). `--persona-ids <id> ...` reruns specific profiles; the configuration runs on the profile panel use the ids in `REBUTTAL_PANEL_150.csv` from the data release, and every run's exact ids are in its stored configuration. `--human-mode` reads `human_personas.csv`, people's own profiles, which is not public (see "Access to the human data").
- **Battery and sampling.** `--tests-included writing reasoning nfc bfi` selects blocks (`writing` is the BRT). `--random-seed 42` and `--sampling-mode` fix which profiles are drawn. Temperature is left at the provider default unless `--temperature` is given.

The runbook `simulate/output/data_analysis_odd_persona_trait_experiment/README_simulation_code_iter_1.md` lists every option.

## Scoring

CRT answers are scored by `score/scoring_v2/score_v2.py`. It applies the answer keys in `answer_keys_v2.yaml` deterministically first, and only answers the keys do not resolve go to a three-model panel, with the majority label kept and cached per distinct answer string. The panel's votes for every such string, and the corrections to our own annotations, are in the data release (`adjudications/`). Run the scorer's tests with `python -m pytest score/scoring_v2/test_score_v2.py` (pytest is not in `requirements.txt`).

BRT answers were coded with the rubric in `score/brt_v3/prompt_v3.txt` by majority vote of three LLMs from different providers, the same rubric for people and agents. The BRT coding prompt contains short example phrases, some taken from participants’ answers; they are not linked to any person or to any other data. Please cite the companion study (arXiv 2607.07961) when using the rubric. Worked coding examples on agent answers are in the data release (`brt/brt_coding_examples.md`).

**BRT agreement.** The agreement available is the machine panel's own: two coding passes over the same responses, compared field by field (`brt/brt_machine_panel_agreement_per_field.csv` and `brt/brt_machine_panel_agreement_suppressed_fields.csv` in the data release). The fields that fall below their agreement gate are reported with their agreement: what the restaurant decision rested on, the medication decision, and what the medication decision rested on. Agreement between human coders and the panel has not been measured yet, so no human inter-rater statistic is released.

## Regenerating the figures and tables

The builder reads its inputs from a folder laid out like the analysis repository it came from. In the data release that folder is `builder_inputs/soc-sim/`; the folders beside it, `builder_inputs/cognition-suite-sim/` and `builder_inputs/brt_final/`, hold the profile panel and the CRT2 level cutoffs it also reads.

```bash
export COGSUITE_INPUT_ROOT="$PWD/data/release/builder_inputs/soc-sim"
PYTHONPATH=score:score/scoring_v2 python analyse/scripts/build_neurips_camera_ready_assets.py
```

Outputs go to `$COGSUITE_INPUT_ROOT/reports/neurips_paper_assets_v2/` (`figures/`, `tables/`, `captions/`, `diagnostics/`). Compare them with the paper's tables of record in the data release (`paper_tables/`).

People's individual-level records are not public. A build step that needs them is skipped rather than stopping the run, and `diagnostics/controlled_access_outputs.md` lists the skipped steps and the inputs each one needs. With the controlled-access records placed at the paths that file names, the same command rebuilds everything.

**Rebuilt from the public release.** These outputs rebuild byte-identical (tables) or render-identical (figures) to the paper's from the released inputs alone:

- Figures (`.pdf` and `.png`): `appendix_brt_level_prevalence`, `fig1_cognition_suite_schematic`, `fig2_distribution_fidelity_dashboard`, `fig2b_distribution_fidelity_ecdf_overlay`, `fig5_association_fidelity_heatmaps`, `fig6_calibration_fingerprint`, `fig_s11_forest_H1`, `fig_s11_forest_H2`, `fig_s11_forest_H3`, `fig_s11_forest_H4`, `fig_s11_verdict_grid`, `main_fig2_distribution_calibration_heatmaps`, `main_fig2_focal_distribution_grid`, `main_fig3_delta_r_matrix`, `main_fig3_f4_correlation_bands`, `main_fig3_focal_delta_rho_grid`
- Tables: `appendix_brt_level_prevalence_data.csv`, `main_fig3_delta_r_matrix_data.csv`, `s11_family_H1_table.csv`, `s11_family_H2_table.csv`, `s11_family_H3_table.csv`, `s11_family_H4_table.csv`, `s11_verdict_grid.csv`, `table_association_fidelity_metrics.csv`, `table_brt_prevalence_by_run.tex`, `table_brt_steering.tex`, `table_brt_transfer.tex`, `table_calibration_fidelity.csv`, `table_distribution_fidelity_metrics.csv`, `table_experimental_runs.csv`, `table_experimental_runs.tex`, `table_f4_trait_structure.csv`, `table_f4_trait_structure_a.tex`, `table_f4_trait_structure_b.tex`, `table_run_identifiers.csv`, `table_run_identifiers.tex`
- Captions and diagnostics: `captions/appendix_brt_level_prevalence.md`, `captions/appendix_crt_response_categories_by_crt_level.md`, `captions/appendix_figure_list.md`, `captions/captions.md`, `captions/main_fig1_lure_ladder.md`, `captions/main_fig2_distribution_calibration_heatmaps.md`, `captions/main_fig3_delta_r_matrix.md`, `captions/main_fig4a_brt_gap.md`, `captions/main_fig5_lure_by_told_level.md`, `captions/table_configuration_ladder.md`, `captions/table_design_probes.md`, `captions/table_human_crt2_reference_by_item.md`, `captions/table_item_exposure_and_unseen_lure.md`, `diagnostics/prompt_screening_unavailable.md`

The builder also writes a retired diagnostic figure and its data table that the paper does not print; they are left out of these lists.

**Need the controlled-access records.** These outputs use people's individual-level records (their CRT answers, BRT codes, person-level frame, or the agent files of the two human-derived runs, which carry each participant's own profile):

- Figures (`.pdf` and `.png`): `appendix_brt_categorical_distributions`, `appendix_brt_categories_difference`, `appendix_brt_prevalence_by_run`, `appendix_brt_transfer_models_flagged`, `appendix_crt_response_categories_by_crt_level`, `appendix_liwc_dots_slopes_bcrt`, `appendix_liwc_dots_slopes_crt2`, `appendix_liwc_slope_difference_heatmap`, `appendix_liwc_slope_fidelity_theory_set`, `fig3_crt_bcrt_intuitive_lure_fidelity`, `fig4_brt_behavioural_compression`, `main_fig1_lure_ladder`, `main_fig1a_lure_ladder_rates`, `main_fig1b_lure_ladder_items`, `main_fig4_brt_scatter_forest`, `main_fig4a_brt_gap`, `main_fig5_lure_by_told_level`
- Tables: `WHAT_MOVED_CR25_LADDER_PROBES.md`, `_cr23_compile_test.tex`, `appendix_brt_category_shares.csv`, `appendix_liwc_transfer_verdicts.csv`, `table_brt_distributional_fidelity.csv`, `table_brt_transfer_models_flagged.csv`, `table_configuration_ladder.csv`, `table_configuration_ladder.tex`, `table_crt_bcrt_response_category_fidelity.csv`, `table_crt_response_categories_by_specified_level.csv`, `table_design_probes.csv`, `table_design_probes.tex`, `table_fidelity_scorecard.csv`, `table_fidelity_scorecard.tex`, `table_human_crt2_reference_by_item.csv`, `table_human_crt2_reference_by_item.tex`, `table_human_lure_by_own_crt2_level.csv`, `table_intuitive_lure_prevalence_by_run_task.csv`, `table_item_exposure_and_unseen_lure.csv`, `table_item_exposure_and_unseen_lure.tex`, `table_ladder_probe_js_what_moved.csv`, `table_ladder_probe_per_item_lure.csv`, `table_liwc_theory_set_slope_fidelity.csv`, `table_liwc_theory_set_slope_fidelity.tex`, `table_liwc_theory_set_slope_fidelity_1_CR_MAIN_SYNTH_NUM_SONNET5_S42.tex`, `table_liwc_theory_set_slope_fidelity_2_CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42.tex`, `table_liwc_theory_set_slope_fidelity_3_CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42.tex`, `table_liwc_theory_set_slope_fidelity_4_CR_BASELINE_HUMAN_NUMERIC_CS46_S42.tex`, `table_liwc_theory_set_slope_fidelity_5_CR_BASELINE_HUMAN_LEXICAL_CS46_S42.tex`, `table_liwc_theory_set_slope_fidelity_6_CR_MAIN_SYNTH_NUMERIC_CS46_S42.tex`, `table_liwc_theory_set_slope_fidelity_7_CR_MAIN_SYNTH_LEXICAL_CS46_S42.tex`
- Captions and diagnostics: `captions/liwc_captions.md`, `diagnostics/asset_quality_report.md`, `diagnostics/brt_prevalence_recovery_report.md`, `diagnostics/ladder_intervals_scored_denominator_cr25b.csv`, `diagnostics/missing_assets_report.md`, `diagnostics/variable_aliases_used.md`

**Figure rendering.** Figures were made with matplotlib 3.10.0 on macOS with matplotlib's default (macOS) backend, and are render-identical under those conditions. With the Agg backend (the default on Linux and in headless sessions, or with `MPLBACKEND=Agg`), several figures come out a few pixels wider or narrower because their tight bounding box is measured differently. Their content and text are identical. Tables are unaffected by the backend.

**Table digits.** Tables rebuild byte-identical in the conda environment (`environment.yml`). With the pip install, pandas reads a few numbers from the input CSV files with a different last binary digit, so a few values in `table_distribution_fidelity_metrics.csv` differ in their final printed digit (about the sixteenth significant figure); nothing else changes. With the controlled-access records added, every output rebuilds identical in the conda environment except `appendix_liwc_transfer_verdicts.csv`, whose regression slopes can differ in their last digits because the linear-algebra library build differs from the one used for the paper.

## Access to the human data

Raw human free-text and individual-level human data are not part of this release. Participants consented to their information being shared only in a format that does not identify them, for future research only within the permissions each gave, and to anonymised data being released publicly. De-identified individual-level records are available from the authors now and will be released publicly with the companion study's updated dataset (arXiv 2607.07961; https://osf.io/xuv7e/).

## Acknowledgement

Agent-simulation code was generated using the SOCIA-EVO simulator framework.

## Licence

Code: MIT (`LICENSE`), copyright 2026 Sion Weatherhead, Ben Newell, Aaron Belbasis and Flora Salim. Data release: CC BY 4.0.
