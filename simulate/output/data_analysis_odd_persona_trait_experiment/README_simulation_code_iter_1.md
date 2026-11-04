# `simulation_code_iter_1.py` Runbook

This runbook documents the standalone simulator entry point:

- `output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py`

Run commands from the root of the standalone repository.

## Environment

Install the simulator dependencies:

```bash
pip install -r requirements.txt
```

For local verification without external calls, use `--use-pseudo`. For real LLM runs, set the relevant provider key:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

## Data Layout

The simulator resolves paths relative to the standalone repository root by default.

- Persona and survey data: `data_fitting/persona_trait_experiment/`
- Prompt configuration: `data_fitting/persona_trait_experiment/prompt_config/`
- Default output root: `simulation_outputs/`

Required runtime files include:

- `synthetic_personas.csv`
- `human_personas.csv`
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
- outputs are written under `simulation_outputs/run_YYYYMMDD_HHMMSS`

The simulator derives `crt2_level` from `CRT2_Total_adj` using `crt2_level_cutoffs.yaml`; the source CSV does not need a pre-existing discrete CRT column.

## No-API Smoke Test

```bash
python "output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py" \
  --use-pseudo \
  --max-personas 1 \
  --tests-included writing \
  --output-dir "simulation_outputs/smoke_test"
```

## Example Commands

Random local dry run:

```bash
python "output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py" \
  --use-pseudo \
  --max-personas 20 \
  --sampling-mode random \
  --random-seed 42
```

Full human baseline with numeric persona specifications:

```bash
python "output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py" \
  --human-mode \
  --use-anthropic \
  --anthropic-model "claude-sonnet-4-6" \
  --max-parallel-personas 8 \
  --persona-spec-mode numeric_only \
  --run-label "BASELINE_HUMAN467_NUMERIC_FULL_S42" \
  --persona-prompt-version "winner_v2_numeric_only" \
  --task-prompt-version "task_v1"
```

Synthetic main run with random sampling and lexical persona specifications:

```bash
python "output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py" \
  --use-anthropic \
  --anthropic-model "claude-sonnet-4-6" \
  --max-personas 467 \
  --sampling-mode random \
  --random-seed 42 \
  --max-parallel-personas 15 \
  --persona-spec-mode lexical_only \
  --run-label "MAIN_SYNTH470_LEXICAL_RANDOM_S42" \
  --persona-prompt-version "winner_v2_lexical_only" \
  --task-prompt-version "task_v1"
```

## Common Options

- `--admin-mode {per_item,per_test,all_tests}`
- `--tests-included {writing,reasoning,nfc,bfi} [{writing,reasoning,nfc,bfi} ...]`
- `--amnesiac-mode`
- `--human-mode`
- `--personas-filename <csv>`
- `--max-personas <N>`
- `--persona-ids <id> [<id> ...]`
- `--sampling-mode {sequential,random,stratified_crt}`
- `--max-parallel-personas <N>`
- `--random-seed <N>`
- `--holdout-fraction <float>`
- `--output-dir <relative/path>`
- `--run-label <label>`
- `--shuffle-items-within-tests`
- `--shuffle-test-order`
- `--no-randomize-reasoning-block`
- `--persona-prompt-version <label>`
- `--task-prompt-version <label>`
- `--persona-spec-mode {lexical_only,numeric_only,both}`
- `--bfi-ladder-granularity {5,9}`
- `--include-user-context`
- `--use-crt-numeric-only`
- `--use-pseudo`
- `--use-openai --openai-model <model> --openai-max-output-tokens <N>`
- `--use-anthropic --anthropic-model <model> --anthropic-max-output-tokens <N>`

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
