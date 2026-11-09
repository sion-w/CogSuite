# `simulation_code_iter_1.py` Runbook

This runbook documents the simulator entry point:

- `simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py`

Run the commands below from the repository root.

## Environment

Install the dependencies from the repository root (`environment.yml` is the reference environment; `requirements.txt` installs the same versions with pip):

```bash
pip install -r requirements.txt
```

For local verification without external calls, use `--use-pseudo`. For real LLM runs, set the relevant provider key in the environment or in a git-ignored `simulate/.env` file:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

OpenAI-compatible providers read the key from the variable named by `--openai-key-env` (for example `GEMINI_API_KEY`, `DEEPSEEK_API_KEY` or `TOGETHER_API_KEY`).

## Data Layout

The simulator resolves paths from the simulator root, `simulate/` (two levels above the script; override with the `PROJECT_ROOT` environment variable).

- Persona and survey data: `simulate/data_fitting/persona_trait_experiment/` (override with `DATA_PATH`, relative to the simulator root)
- Prompt configuration: `simulate/data_fitting/persona_trait_experiment/prompt_config/`
- Default output root: `simulate/simulation_outputs/`

Required runtime files include:

- `synthetic_personas.csv` (the synthetic profile bank)
- `human_personas.csv` (only for `--human-mode`; people's own profiles, available through controlled access and not part of this repository)
- `crt2_level_cutoffs.yaml`
- `bfi10_items.yaml`
- `nfc18_items.yaml`
- `crt2_items.yaml`
- `bcrt_items.yaml`
- `dcrt_items.yaml`
- `writing_tasks.yaml`
- `prompt_config/persona_template.yaml`
- `prompt_config/bfi_ladders_5.yaml`
- `prompt_config/bfi_ladders_9.yaml`
- `prompt_config/nfc_ladder_9.yaml`
- `prompt_config/crt2_ladder.yaml`
- `prompt_config/response_schemas.yaml`

## Default Behavior

- `admin_mode=per_test`
- persona source: `synthetic_personas.csv`
- `--human-mode` switches to `human_personas.csv`
- survey blocks included by default: `writing`, `reasoning`, `nfc`, `bfi`
- default block order: Writing, merged reasoning (`CRT2+bCRT+dCRT`), NFC, BFI
- `max_parallel_personas=4`
- `holdout_fraction=0.0`
- `random_seed=42`
- `sampling_mode=sequential`
- persona spec mode: `lexical_only`
- BFI ladder granularity: 9-level
- multi-turn conversation memory is enabled across blocks for each persona
- outputs are written under `simulate/simulation_outputs/run_YYYYMMDD_HHMMSS` (with the slugified `--run-label` appended when given)

The simulator derives `crt2_level` from `CRT2_Total_adj` using `crt2_level_cutoffs.yaml`; the source CSV does not need a pre-existing discrete CRT column.

`--persona-prompt-version` and `--task-prompt-version` are labels recorded in `metrics_and_config.json`; they are not sent to the model. The persona prompt itself is set by `--persona-spec-mode`, `--bfi-ladder-granularity` and the files in `prompt_config/`.

## No-API Smoke Test

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py \
  --use-pseudo \
  --max-personas 1 \
  --tests-included writing \
  --output-dir "simulation_outputs/smoke_test"
```

`--output-dir` is relative to the simulator root, so this run writes to `simulate/simulation_outputs/smoke_test/`.

## Example Commands

Random local dry run:

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py \
  --use-pseudo \
  --max-personas 20 \
  --sampling-mode random \
  --random-seed 42
```

Human-derived baseline with numeric persona specifications (`CR_BASELINE_HUMAN_NUMERIC_CS46_S42`, 471 profiles; needs `human_personas.csv`):

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py \
  --human-mode \
  --use-anthropic \
  --anthropic-model "claude-sonnet-4-6" \
  --anthropic-max-output-tokens 8000 \
  --tests-included writing reasoning nfc bfi \
  --max-parallel-personas 10 \
  --persona-spec-mode numeric_only \
  --sampling-mode sequential \
  --random-seed 42 \
  --run-label "CR_BASELINE_HUMAN_NUMERIC_CS46_S42" \
  --persona-prompt-version "persona_v2_numeric_only" \
  --task-prompt-version "task_v1"
```

Synthetic main run with random sampling and lexical persona specifications (`CR_MAIN_SYNTH_LEXICAL_CS46_S42`, 471 profiles drawn from the synthetic bank):

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py \
  --use-anthropic \
  --anthropic-model "claude-sonnet-4-6" \
  --anthropic-max-output-tokens 8000 \
  --tests-included writing reasoning nfc bfi \
  --max-personas 471 \
  --sampling-mode random \
  --random-seed 42 \
  --max-parallel-personas 10 \
  --persona-spec-mode lexical_only \
  --run-label "CR_MAIN_SYNTH_LEXICAL_CS46_S42" \
  --persona-prompt-version "persona_v2_lexical_only" \
  --task-prompt-version "task_v1"
```

`simulate/run_configs/RUNS.md` gives the command for every run in the paper, and `simulate/run_configs/<run>.json` holds each run's stored configuration.

## Common Options

- `--admin-mode {per_item,per_test,all_tests}`
- `--tests-included {writing,reasoning,nfc,bfi} [{writing,reasoning,nfc,bfi} ...]`
- `--amnesiac-mode`
- `--human-mode`
- `--personas-filename <csv>`
- `--markers-filename <csv>`
- `--max-personas <N>`
- `--persona-ids <id> [<id> ...]`
- `--sampling-mode {sequential,random,stratified_crt}`
- `--max-parallel-personas <N>`
- `--random-seed <N>`
- `--holdout-fraction <float>`
- `--time-column <column>`
- `--output-dir <relative/path>`
- `--run-label <label>`
- `--shuffle-items-within-tests`
- `--shuffle-test-order`
- `--no-randomize-reasoning-block`
- `--persona-prompt-version <label>`
- `--task-prompt-version <label>`
- `--persona-spec-mode {lexical_only,numeric_only,both}`
- `--persona-prompt-style {granular_serapio,coarse_numeric,coarse_descriptive}`
- `--persona-system-suffix <text>`
- `--bfi-ladder-granularity {5,9}`
- `--include-user-context`
- `--use-crt-numeric-only`
- `--crt-cutoffs-filename <yaml>`
- `--use-pseudo`
- `--use-openai --openai-model <model> --openai-max-output-tokens <N>`
- `--openai-base-url <url> --openai-api-style {responses,chat_completions} --openai-key-env <VAR>`
- `--openai-extra-params-json <json> --openai-extra-body-json <json>`
- `--use-anthropic --anthropic-model <model> --anthropic-max-output-tokens <N>`
- `--temperature <float> --anthropic-thinking-json <json>`
- `--max-api-retries <N>`

## Output Files

Each run writes:

- `per_persona_results.csv`
- `per_persona_results.jsonl`
- `per_item_results.csv`
- `metrics_and_config.json`
- `api_calls.csv`
- `api_calls.jsonl`
- `failed_personas.csv` and `failed_personas.jsonl` when failures occur

`api_calls` includes the prompt text and `survey_item_ids` mapping needed to decode generic prompt survey names back to internal item IDs.

`metrics_and_config.json` includes an `api_reliability` block (calls, retries, hard failures, unparseable items, truncated responses). API errors are retried with backoff and then recorded as hard failures; items without a usable answer are left missing and listed in `failed_personas.csv`. When `hard_failure_rate_exceeds_1pct` is true, rerun the listed personas with `--persona-ids` before scoring.
