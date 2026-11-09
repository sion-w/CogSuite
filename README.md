# CogSuite

Code for the paper *Imperfectly Imperfect: Evaluating Graded Cognitive Diversity in LLM Persona Simulations* (NeurIPS 2026, Evaluations & Datasets track).

Data release (OSF): https://osf.io/h8d5s/ · Archived on Zenodo (the DOI is in the GitHub release notes and the paper) · Code licence: MIT (`LICENSE`) · Data licence: CC BY 4.0

## What the benchmark tests

CogSuite asks whether LLM agents that are given a person's profile (demographics and trait scores) reproduce the *graded* variation in how people reason, rather than collapsing toward generic competence. Agents and people answer the same cognition suite:

- two cognitive reflection tests: CRT2, and a bespoke CRT (bCRT), with decoy items;
- Need for Cognition (NFC-18) and Big Five personality (BFI-10);
- two open-ended Behavioural Reflection Test (BRT) scenarios, a restaurant booking and a medication decision, coded with a fixed rubric.

Agents are conditioned on profiles taken from people, either each person's own profile or one drawn from a synthetic bank fitted to them, written either as numbers (numeric format) or as words (lexical format). Their answers are compared with people's on five fidelity targets: marginal distributions, calibration (are specified traits realised?), associations among constructs, transfer to downstream BRT behaviour, and error types (correct, intuitive lure, other incorrect, missing).

The human sample, the bCRT and the BRT are presented in the companion study (arXiv 2607.07961).

## The pipeline

| Stage | Code | Reads | Writes |
|---|---|---|---|
| 1. Simulate | `simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py` | a profile file, the item files, the prompt configuration | `simulate/simulation_outputs/<RUN>/` (per-item and per-persona results, run metadata) |
| 2. Score CRT answers | `score/scoring_v2/` (`score_cli.py` for one run; `analyse/pipeline/score_config_runs.py` for the configuration runs) | a run's `per_item_results.csv` | one labelled row per CRT2/bCRT answer |
| 3. Code BRT answers | `score/brt_v3/` (rubric `prompt_v3.txt`, panel coder `code_brt_v3.py`) | a run's BRT answers | coded BRT fields per persona |
| 4. Per-run analysis | `analyse/pipeline/run_per_run_analysis.py` (the `analyse/socsim` package) | a run's raw outputs, the human person file, the run's LIWC-22 output | `reports/cr13_<RUN>/` |
| 5. Cross-run analyses | `analyse/pipeline/run_chain.py` | the per-run reports, the scored configuration runs | `reports/{crt_tables, brt_models, liwc_models, human_brt_families, trait_structure, ladder_intervals, run_intervals, level_prevalence}/` |
| 6. Figures and tables | `analyse/scripts/build_paper_assets.py` | everything above | `reports/paper_assets/{figures, tables, captions, diagnostics}/` |

Stages 2 to 6 score and code through caches shipped with the release, so the paper's runs reanalyse without any API key. Stages 4 and 5 compare agents with people and so also need the human data (see "Human data"); the data release carries their outputs, so stage 6 runs from the release alone. Stages 4 to 6 read every path from one analysis folder, set with `--root` or `COGSUITE_INPUT_ROOT`. The data release's `analysis/` folder is laid out this way.

**Names.** The simulator file and folder names (`simulation_code_iter_1.py`, `data_analysis_odd_persona_trait_experiment/`) are those of the SOCIA-EVO framework that generated the simulation code. Version labels name instrument versions: CRT scoring `scoring_v2` (answer keys and adjudication), BRT rubric `brt_v3` (v3.0), persona prompts `persona_v2_*`, task prompt `task_v1`. Per-run analysis reports are named `cr13_<run>`. In the BRT model tables, `s11_` and `f1_`/`f2_`/`f3_` prefixes follow the paper's appendix on BRT transfer, and `f4_` the trait-structure analysis.

## Layout

| Folder | What it holds |
|---|---|
| `simulate/` | The simulator and its runbook (`output/data_analysis_odd_persona_trait_experiment/`), the survey item files and prompt configuration (`data_fitting/persona_trait_experiment/`), the synthetic profile bank (`synthetic_personas.csv`) and its first release (`synthetic_personas_first_release.csv`) with their CRT2 level cutoffs, and the stored configuration of every run in the paper (`run_configs/`, one command per run in `run_configs/RUNS.md`). |
| `score/` | CRT scoring (`scoring_v2/`: answer keys, normalisation, the deterministic-first scorer, its adjudication prompt and cache, a batch CLI and tests) and BRT coding (`brt_v3/`: the rubric, field definitions, the three-model panel coder, derived variables and tests). |
| `analyse/` | The per-run analysis package (`socsim/`), the cross-run analyses (`pipeline/`), and the figure and table builder (`scripts/build_paper_assets.py`, with the colour system in `scripts/paper_palette.py`). |
| `data/synthetic/` | The code that generated the synthetic profile bank (`05_synth_personas.py`), the code that drew the configuration-run panels (`generate_config_panels.py`), and the configuration-run panels (`CONFIG_PANEL_150.csv`, `CONFIG_PROBE_100.csv`). |

## Install

Use Python 3.11. `environment.yml` is the reference environment, with the exact builds that made the paper's figures and tables; `requirements.txt` installs the same versions with pip.

```bash
git clone https://github.com/sion-w/CogSuite && cd CogSuite
conda env create -f environment.yml && conda activate cogsuite
# or: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

API keys go in the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `FIREWORKS_API_KEY`, or the variable named by `--openai-key-env`) or in a `.env` file in `simulate/` or at the repository root; git ignores both. Never commit a key.

## 1. Simulate

A smoke test with no API key (pseudo mode gives deterministic placeholder answers and makes no calls):

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py \
  --use-pseudo --max-personas 1 --tests-included writing \
  --output-dir simulation_outputs/smoke_test
```

The run writes `per_persona_results.csv`/`.jsonl`, `per_item_results.csv`, `api_calls.csv`/`.jsonl` and `metrics_and_config.json` to `simulate/simulation_outputs/smoke_test/`.

`simulate/run_configs/RUNS.md` gives the exact command for each run in the paper: the seven full-battery runs, the configuration runs, and the with-history reference run. Each command was written out from that run's stored configuration (`simulate/run_configs/<run>.json`). Real runs make paid API calls. The settings that define a run:

- **Model and provider.** `--use-anthropic --anthropic-model <id>` for the Anthropic Messages API; `--use-openai --openai-model <id>` for OpenAI and OpenAI-compatible providers, with `--openai-base-url <url> --openai-api-style chat_completions` and `--openai-key-env <VAR>` for the compatible providers. `--openai-max-output-tokens` and `--anthropic-max-output-tokens` set the output limit.
- **Thinking or reasoning setting.** `--anthropic-thinking-json '{"type": "disabled"}'` for Anthropic models; `--openai-extra-params-json` for OpenAI models (for example `'{"reasoning": {"effort": "none"}}'`); `--openai-extra-body-json` for providers that take provider-specific fields in the request body (for example the Gemini `thinking_level` and DeepSeek `thinking` objects in `RUNS.md`). Leaving these out keeps the provider default.
- **Prompt format.** `--persona-spec-mode numeric_only` with `--persona-prompt-version persona_v2_numeric_only`, or `lexical_only` with `persona_v2_lexical_only`; `--task-prompt-version task_v1` throughout.
- **Profiles.** By default the simulator reads the synthetic bank `simulate/data_fitting/persona_trait_experiment/synthetic_personas.csv`, the bank of the seven full-battery runs; a profile's `persona_id` is its 0-based row number. The configuration runs, the probes and the with-history reference run use the first release of the bank (`--personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml`), and `--persona-ids <id> ...` selects profiles: the 150 ids in `data/synthetic/CONFIG_PANEL_150.csv`, or the 100 in `CONFIG_PROBE_100.csv`. A profile's told CRT2 level comes from its CRT2 score and the cutoffs file. `--human-mode` reads `human_personas.csv`, people's own profiles, from the companion study's dataset (see "Human data").
- **Battery and sampling.** `--tests-included writing reasoning nfc bfi` selects blocks (`writing` is the BRT). `--random-seed 42` and `--sampling-mode` fix which profiles are drawn. Temperature stays at the provider default unless `--temperature` is given.

The runbook `simulate/output/data_analysis_odd_persona_trait_experiment/README_simulation_code_iter_1.md` lists every option. At the end of a run that makes API calls the simulator prints its API reliability counters; if more than 1% of calls failed, rerun the personas listed in `failed_personas.csv` before analysis.

## 2. Score CRT answers

CRT answers are scored by `score/scoring_v2/score_v2.py`. Answer keys (`answer_keys_v2.yaml`) are applied first; an answer no key resolves goes to a three-model panel, and the majority label is kept and cached per distinct answer string (`score/scoring_v2/cache/adjudications.jsonl`, which holds every panel vote behind the paper). The scorer labels every answer `correct`, `intuitive` or `other`; an empty answer is labelled `other` and the analysis counts it as unscored (excluded from lure-rate denominators). A typed word such as "None" is an answer ("none" is correct on CRT2 item 4).

```bash
# a released run (see section 4 for the data release), or your own run under simulate/simulation_outputs/<RUN>/
python score/scoring_v2/score_cli.py --items data/release/analysis/simulation_outputs/<RUN>/per_item_results.csv --out <folder>/crt_long_<RUN>.csv
```

Answers missing from the cache are listed and labelled `residual_unadjudicated`; with `--allow-api` and the three keys set they are put to the panel and added to the cache. The panel: OpenAI `gpt-5.6-sol` (reasoning effort low), Google `gemini-3.5-flash` (thinking level minimal) and DeepSeek V4-Flash on Fireworks (`accounts/fireworks/models/deepseek-v4-flash-0731`, thinking disabled). Fireworks no longer serves DeepSeek V4-Flash; to score or code new answers, set `COGSUITE_DEEPSEEK_SEAT_MODEL` to an available Fireworks model (for example `accounts/fireworks/models/deepseek-v4p1-flash`). Every vote records the model and endpoint that cast it; the cache's DeepSeek votes carry the Fireworks ids current when they were cast. The caches hold every vote behind the paper, so reproducing the paper needs no seat at all. Run the tests with `python -m pytest score/scoring_v2/test_score_v2.py` (pytest is not in `requirements.txt`).

## 3. Code BRT answers

BRT answers are coded with rubric v3.0 (`score/brt_v3/prompt_v3.txt`) by three seats, the same rubric for people and agents; each field takes the majority, and a three-way split takes the conservative default. The seats: OpenAI `gpt-5.6-sol` (reasoning effort low), Google `gemini-3.5-flash` (thinking level minimal) and DeepSeek V4-Flash on Fireworks (thinking disabled), each with up to 8,192 output tokens. A field coded present must quote evidence found in the response; otherwise the seat is asked once more and the field then falls back to its default. Coding runs inside the per-run analysis (stage 4). To code answers on their own, without the human data, give the coder a CSV with one row per persona and columns `entity_id`, `brt_1` and `brt_2` (the two scenario answers from a run's `per_item_results.csv`):

```bash
python score/brt_v3/code_brt_v3.py --input-csv brt_answers.csv --cache-dir <cache folder> --out brt_codes.csv
```

 The coding cache for the paper's synthetic-profile runs is in the data release (`analysis/brt_cache/agent/`), so they recode with no key; new runs need `OPENAI_API_KEY`, `GEMINI_API_KEY` and `FIREWORKS_API_KEY`, and `COGSUITE_DEEPSEEK_SEAT_MODEL` as for CRT scoring (the GPT seat uses the OpenAI Batch API by default, `BRT_V3_TRANSPORT=batch`). The prompt is released exactly as it was sent to the panel; it contains short example phrases, some taken from participants' answers, not linked to any person or to any other data. Please cite the companion study (arXiv 2607.07961) when using the rubric. Worked coding examples on agent answers are in the data release (`brt/brt_coding_examples.md`). Run the tests with `python -m pytest score/brt_v3`.

**BRT agreement.** The agreement reported is the machine panel's own: two coding passes over the same responses, compared field by field (`brt/brt_machine_panel_agreement_per_field.csv` and `brt/brt_machine_panel_agreement_suppressed_fields.csv` in the data release). The 90% gate applies to the fields the analyses use (`primary` or `contributing` in the per-field table); the two unspecified-check fields are used in their suppressed form (`_sup`: an unspecified check counts only when no specific source is coded anywhere in the answer) and are gated on that form (`brt_machine_panel_agreement_suppressed_fields.csv`). The gated fields below the gate are reported with their agreement: what the restaurant decision rested on, the medication decision, and what the medication decision rested on.

## 4. Per-run analysis

Download the data release from OSF and unpack it into `data/release/` (git-ignored), so that the analysis folder is `data/release/analysis/`.

```bash
export COGSUITE_INPUT_ROOT="$PWD/data/release/analysis"
python analyse/pipeline/run_per_run_analysis.py --root "$COGSUITE_INPUT_ROOT" --run CR_MAIN_SYNTH_NUM_SONNET5_S42   # or --all
```

For a new run, first copy its folder from `simulate/simulation_outputs/<RUN>/` to `simulation_outputs/<RUN>/` in the analysis folder; the driver creates the run's analysis configuration from `analyse/socsim/config/analysis.yaml`. For each run it reads `simulation_outputs/<RUN>/`, the run's LIWC-22 output in `liwc/`, the agent BRT cache `brt_cache/agent/` and the human person file `data/inputs/person_with_liwc_brt_coded.csv` (see "Human data"), scores and codes the run, and writes `reports/cr13_<RUN>/` (tables, figures, `run_metadata.json`). Four personas of the DeepSeek V4-Pro run returned no usable open-ended response and are excluded from its BRT analyses (`analyse/socsim/resources/brt_exclusions.csv`).

**LIWC.** LIWC-22 is commercial software, so it runs outside this code. The paper's LIWC outputs ship in the data release (`analysis/liwc/`). For a new run:

1. Run the per-run analysis (this needs the human person file, as above). It writes `data/exports/socsim_agent_<RUN>_liwc_text_export.csv` (columns `entity_id`, `brt_1`, `brt_2`, `brt_concat`) under the analysis folder and carries on with the LIWC columns empty.
2. In LIWC-22, analyse that file as a spreadsheet: text column `brt_concat`, the default LIWC-22 dictionary, input columns kept (so the output carries `entity_id`).
3. Save the result as `liwc/socsim_agent_<RUN>_liwc_output.csv` in the analysis folder.
4. Rerun the same command. The output is joined on `entity_id` and the LIWC family models run.

The pipeline prints these steps, with the actual paths, when a run's LIWC output is missing.

## 5. Cross-run analyses

```bash
python analyse/pipeline/run_chain.py --root "$COGSUITE_INPUT_ROOT"
python analyse/pipeline/score_config_runs.py --root "$COGSUITE_INPUT_ROOT"    # the configuration runs' CRT scoring
```

`run_chain.py` runs, in order: the human CRT2 reference, the human BRT families, the CRT distribution tables, the LIWC family models, the BRT transfer models and their figures, the trait-structure transfer, and the interval and per-level prevalence tables. Each step also runs alone (`python analyse/pipeline/<step>.py --root ...`; `--only` and `--from` select steps). A step whose inputs need the human data is skipped with a message naming the missing file. Run the tests with `python -m pytest analyse/pipeline/tests`.

## 6. Figures and tables

```bash
export COGSUITE_INPUT_ROOT="$PWD/data/release/analysis"   # as in section 4
python analyse/scripts/build_paper_assets.py
```

Outputs go to `$COGSUITE_INPUT_ROOT/reports/paper_assets/` (`figures/`, `tables/`, `captions/`, `diagnostics/`). The data release carries the outputs of stages 4 and 5, so this command rebuilds from the release directly; compare the results with the paper's tables in the data release (`paper_tables/`). A build step that needs the human data is skipped rather than stopping the run, and `diagnostics/controlled_access_outputs.md` lists the skipped steps and the inputs each needs. With the human data placed at the paths that file names, the same command rebuilds everything.

**From the data release alone** the builder writes these outputs, identical to the paper's (tables byte-identical, figures render-identical; see the notes below):

- Figures (`.pdf` and `.png`): `appendix_brt_level_prevalence`, `appendix_calibration_relative_rmse`, `fig1_cognition_suite_schematic`, `fig2_distribution_fidelity_dashboard`, `fig2b_distribution_fidelity_ecdf_overlay`, `fig5_association_fidelity_heatmaps`, `fig6_calibration_fingerprint`, `fig_s11_forest_H1` to `H4`, `fig_s11_verdict_grid`, `main_fig2_distribution_calibration_heatmaps`, `main_fig2_focal_distribution_grid`, `main_fig3_delta_r_matrix`, `main_fig3_f4_correlation_bands`, `main_fig3_focal_delta_rho_grid`
- Tables: `appendix_brt_level_prevalence_data.csv`, `appendix_calibration_relative_rmse_data.csv`, `main_fig3_delta_r_matrix_data.csv`, `s11_family_H1_table.csv` to `H4`, `s11_verdict_grid.csv`, `table_association_fidelity_metrics.csv`, `table_brt_prevalence_by_run.tex`, `table_brt_steering.tex`, `table_brt_transfer.tex`, `table_calibration_fidelity.csv`, `table_distribution_fidelity_metrics.csv`, `table_experimental_runs.csv`/`.tex`, `table_f4_trait_structure.csv`, `table_f4_trait_structure_a.tex`/`_b.tex`, `table_run_identifiers.csv`/`.tex`

The other figures and tables need the human data. Their outputs, as the paper prints them, are in the data release's `paper_tables/`.

**Reproducibility notes.** Figures were made with matplotlib 3.10.0 on macOS with matplotlib's default (macOS) backend, and are render-identical under those conditions. With the Agg backend (the default on Linux and in headless sessions, or with `MPLBACKEND=Agg`), several figures come out a few pixels wider or narrower because their tight bounding box is measured differently; their content and text are identical. Tables rebuild byte-identical in the conda environment. With the pip install, pandas reads a few numbers from the input files with a different last binary digit, so a few values in `table_distribution_fidelity_metrics.csv` differ in their final printed digit. Model fits that use linear algebra (the LIWC slope tables, the calibration summaries and the BRT transfer models) can differ in their last digits between linear-algebra library builds; no printed value changes.

## Synthetic profiles

The bank in `simulate/data_fitting/persona_trait_experiment/synthetic_personas.csv` (5,000 profiles, seed 42) is the bank of the seven full-battery runs; `synthetic_personas_first_release.csv` (5,000 profiles, seed 42) is the bank of the configuration runs, the probes and the with-history reference run, shipped only so that those runs can be reproduced exactly; for new work, use `synthetic_personas.csv`. `data/synthetic/05_synth_personas.py` generated it from the companion study's scored person file (`--input`); run on the companion study's released dataset it gives a bank fitted to that file. Free-text occupations answered "Other" are mapped to occupation groups with a title map that quotes participants' answers; it ships with the companion study's controlled-access records (`--title-map`). `data/synthetic/generate_config_panels.py` draws the configuration-run panels (`CONFIG_PANEL_150.csv`, `CONFIG_PROBE_100.csv`) from the levels of the with-history reference run in the data release, and reproduces them exactly:

```bash
python data/synthetic/generate_config_panels.py --root "$COGSUITE_INPUT_ROOT" --out <folder>
```

## Human data

Raw human free-text and individual-level human data are not part of this release. Participants consented to their information being shared only in a format that does not identify them, for future research only within the permissions each gave, and to anonymised data being released publicly. De-identified individual-level records are available from the authors now and will be released publicly with the companion study's updated dataset (arXiv 2607.07961; https://osf.io/xuv7e/). The stages that compare agents with people read these files from the analysis folder:

| File (under the analysis folder) | Used by |
|---|---|
| `data/inputs/person_with_liwc_brt_coded.csv` | per-run analysis, cross-run analyses, figures and tables |
| `reports/brt_labels/brt_labels_v3_human.csv` | cross-run analyses, figures and tables |
| `simulation_outputs/CR_BASELINE_HUMAN_{NUMERIC,LEXICAL}_CS46_S42/`, `liwc/socsim_agent_CR_BASELINE_HUMAN_*_liwc_output.csv`, `brt_cache/agent/adjudications_CR_BASELINE_HUMAN_*.jsonl` | the two human-derived runs (each agent carries a participant's own profile) |
| `simulate/data_fitting/persona_trait_experiment/human_personas.csv` (placed in the repository folder; git-ignored) | simulating with people's own profiles (`--human-mode`) |

## Citation

Please cite the paper (`.zenodo.json` carries the release metadata) and, when using the bCRT, the BRT or the rubric, the companion study (arXiv 2607.07961).

## Acknowledgement

Agent-simulation code was generated using the SOCIA-EVO simulator framework.

## Licence

Code: MIT (`LICENSE`), copyright 2026 Sion Weatherhead, Ben Newell, Aaron Belbasis and Flora Salim. Data release: CC BY 4.0.
