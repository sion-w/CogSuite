# Run commands

One command per run, written out from that run's stored configuration (`<run>.json` in this folder, key `config`). Run from the repository root after the setup in the main README. Each command makes paid API calls: check the provider's pricing first. Sampling settings (temperature, top-p) stay at provider defaults except in the temperature runs.

Persona source: the seven full-battery synthetic runs read `simulate/data_fitting/persona_trait_experiment/synthetic_personas.csv`. The configuration runs, the probes and the with-history reference run read the first release of the profile bank, `synthetic_personas_first_release.csv`, with its CRT2 level cutoffs, `crt2_level_cutoffs_first_release.yaml` (same folder); their commands select both files. The two human-derived runs (`--human-mode`) read `human_personas.csv`, which holds each participant's own profile and is available only through controlled access.

## The seven full-battery runs

### CR_BASELINE_HUMAN_NUMERIC_CS46_S42

Persona source: people's own profiles (controlled access). API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --human-mode --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --tests-included writing reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --sampling-mode sequential --run-label CR_BASELINE_HUMAN_NUMERIC_CS46_S42 --max-parallel-personas 10 --output-dir simulation_outputs/CR_BASELINE_HUMAN_NUMERIC_CS46_S42
```

### CR_BASELINE_HUMAN_LEXICAL_CS46_S42

Persona source: people's own profiles (controlled access). API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --human-mode --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --tests-included writing reasoning nfc bfi --persona-spec-mode lexical_only --persona-prompt-version persona_v2_lexical_only --task-prompt-version task_v1 --random-seed 42 --sampling-mode sequential --run-label CR_BASELINE_HUMAN_LEXICAL_CS46_S42 --max-parallel-personas 10 --output-dir simulation_outputs/CR_BASELINE_HUMAN_LEXICAL_CS46_S42
```

### CR_MAIN_SYNTH_NUMERIC_CS46_S42

Persona source: the synthetic profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --tests-included writing reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --max-personas 471 --sampling-mode random --run-label CR_MAIN_SYNTH_NUMERIC_CS46_S42 --max-parallel-personas 10 --output-dir simulation_outputs/CR_MAIN_SYNTH_NUMERIC_CS46_S42
```

### CR_MAIN_SYNTH_LEXICAL_CS46_S42

Persona source: the synthetic profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --tests-included writing reasoning nfc bfi --persona-spec-mode lexical_only --persona-prompt-version persona_v2_lexical_only --task-prompt-version task_v1 --random-seed 42 --max-personas 471 --sampling-mode random --run-label CR_MAIN_SYNTH_LEXICAL_CS46_S42 --max-parallel-personas 10 --output-dir simulation_outputs/CR_MAIN_SYNTH_LEXICAL_CS46_S42
```

### CR_MAIN_SYNTH_NUM_SONNET5_S42

Persona source: the synthetic profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-5 --anthropic-max-output-tokens 16000 --tests-included writing reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 471 persona_ids in run_configs/CR_MAIN_SYNTH_NUM_SONNET5_S42.json> --run-label CR_MAIN_SYNTH_NUM_SONNET5_S42 --max-parallel-personas 10 --output-dir simulation_outputs/CR_MAIN_SYNTH_NUM_SONNET5_S42
```

### CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42

Persona source: the synthetic profile bank. API key read from `GEMINI_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model gemini-3.5-flash --openai-max-output-tokens 8000 --openai-api-style chat_completions --openai-base-url https://generativelanguage.googleapis.com/v1beta/openai/ --openai-key-env GEMINI_API_KEY --openai-extra-body-json '{"extra_body": {"google": {"thinking_config": {"thinking_level": "minimal"}}}}' --tests-included writing reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 471 persona_ids in run_configs/CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42.json> --run-label CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42 --max-parallel-personas 5 --output-dir simulation_outputs/CR_MAIN_SYNTH_NUM_GEM35F_MIN_S42
```

### CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42

Persona source: the synthetic profile bank. API key read from `TOGETHER_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model deepseek-ai/DeepSeek-V4-Pro --openai-max-output-tokens 40000 --openai-api-style chat_completions --openai-base-url https://api.together.ai/v1 --openai-key-env TOGETHER_API_KEY --openai-extra-params-json '{"reasoning_effort": "high"}' --tests-included writing reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 471 persona_ids in run_configs/CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42.json> --run-label CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42 --max-parallel-personas 10 --output-dir simulation_outputs/CR_MAIN_SYNTH_NUM_DSV4PRO_TOG_S42
```

## Configuration runs (150 synthetic profiles; design probes on 100)

The 150-profile runs use the `persona_id` values in `data/synthetic/CONFIG_PANEL_150.csv`, in file order (the same list as in each run's stored configuration). The two design probes use the 100 ids stored in their own configurations.

### CRFSB_CLAUDE_TOP_NUM_N150_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-fable-5 --anthropic-max-output-tokens 8000 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFSB_CLAUDE_TOP_NUM_N150_S42.json> --run-label CRFSB_CLAUDE_TOP_NUM_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFSB_CLAUDE_TOP_NUM_N150_S42
```

### CRFSB_GPT56_REASONING_NUM_N150_S42

Persona source: the first release of the profile bank. API key read from `OPENAI_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model gpt-5.6-sol --openai-max-output-tokens 8000 --openai-api-style responses --openai-extra-params-json '{"reasoning": {"effort": "high"}}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFSB_GPT56_REASONING_NUM_N150_S42.json> --run-label CRFSB_GPT56_REASONING_NUM_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFSB_GPT56_REASONING_NUM_N150_S42
```

### CRFSB_SONNET5_THINKOFF_NUM_N150_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-5 --anthropic-max-output-tokens 8000 --anthropic-thinking-json '{"type": "disabled"}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFSB_SONNET5_THINKOFF_NUM_N150_S42.json> --run-label CRFSB_SONNET5_THINKOFF_NUM_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFSB_SONNET5_THINKOFF_NUM_N150_S42
```

### CRFSD_DSV4PRO_TOG_NOTHINK_NUM_N150_S42

Persona source: the first release of the profile bank. API key read from `TOGETHER_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model deepseek-ai/DeepSeek-V4-Pro --openai-max-output-tokens 8000 --openai-api-style chat_completions --openai-base-url https://api.together.ai/v1 --openai-key-env TOGETHER_API_KEY --openai-extra-body-json '{"reasoning": {"enabled": false}}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFSD_DSV4PRO_TOG_NOTHINK_NUM_N150_S42.json> --run-label CRFSD_DSV4PRO_TOG_NOTHINK_NUM_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFSD_DSV4PRO_TOG_NOTHINK_NUM_N150_S42
```

### CRFSD_DSV4PRO_TOG_NUM_N150_S42

Persona source: the first release of the profile bank. API key read from `TOGETHER_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model deepseek-ai/DeepSeek-V4-Pro --openai-max-output-tokens 32000 --openai-api-style chat_completions --openai-base-url https://api.together.ai/v1 --openai-key-env TOGETHER_API_KEY --openai-extra-params-json '{"reasoning_effort": "high"}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFSD_DSV4PRO_TOG_NUM_N150_S42.json> --run-label CRFSD_DSV4PRO_TOG_NUM_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFSD_DSV4PRO_TOG_NUM_N150_S42
```

### CRFSD_KIMI_K3_NUM_N150_S42

Persona source: the first release of the profile bank. API key read from `TOGETHER_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model moonshotai/Kimi-K3 --openai-max-output-tokens 8000 --openai-api-style chat_completions --openai-base-url https://api.together.ai/v1 --openai-key-env TOGETHER_API_KEY --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFSD_KIMI_K3_NUM_N150_S42.json> --run-label CRFSD_KIMI_K3_NUM_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFSD_KIMI_K3_NUM_N150_S42
```

### CRFS_DSV4PRO_ON_LEX_N150_S42

Persona source: the first release of the profile bank. API key read from `DEEPSEEK_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model deepseek-v4-pro --openai-max-output-tokens 32000 --openai-api-style chat_completions --openai-base-url https://api.deepseek.com --openai-key-env DEEPSEEK_API_KEY --openai-extra-body-json '{"thinking": {"type": "enabled"}}' --tests-included reasoning nfc bfi --persona-spec-mode lexical_only --persona-prompt-version persona_v2_lexical_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFS_DSV4PRO_ON_LEX_N150_S42.json> --run-label CRFS_DSV4PRO_ON_LEX_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFS_DSV4PRO_ON_LEX_N150_S42
```

### CRFS_GEMINI35FLASH_MIN_LEX_N150_S42

Persona source: the first release of the profile bank. API key read from `GEMINI_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model gemini-3.5-flash --openai-max-output-tokens 8000 --openai-api-style chat_completions --openai-base-url https://generativelanguage.googleapis.com/v1beta/openai/ --openai-key-env GEMINI_API_KEY --openai-extra-body-json '{"extra_body": {"google": {"thinking_config": {"thinking_level": "minimal"}}}}' --tests-included reasoning nfc bfi --persona-spec-mode lexical_only --persona-prompt-version persona_v2_lexical_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFS_GEMINI35FLASH_MIN_LEX_N150_S42.json> --run-label CRFS_GEMINI35FLASH_MIN_LEX_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFS_GEMINI35FLASH_MIN_LEX_N150_S42
```

### CRFS_SONNET5_DEFAULT_NUM_N150_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-5 --anthropic-max-output-tokens 8000 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CRFS_SONNET5_DEFAULT_NUM_N150_S42.json> --run-label CRFS_SONNET5_DEFAULT_NUM_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CRFS_SONNET5_DEFAULT_NUM_N150_S42
```

### CONFIG_MULTIMODEL_DSV4FLASH_OFF_N150_S42

Persona source: the first release of the profile bank. API key read from `DEEPSEEK_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model deepseek-v4-flash --openai-max-output-tokens 8000 --openai-api-style chat_completions --openai-base-url https://api.deepseek.com --openai-key-env DEEPSEEK_API_KEY --openai-extra-body-json '{"thinking": {"type": "disabled"}}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_MULTIMODEL_DSV4FLASH_OFF_N150_S42.json> --run-label CONFIG_MULTIMODEL_DSV4FLASH_OFF_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_MULTIMODEL_DSV4FLASH_OFF_N150_S42
```

### CONFIG_MULTIMODEL_DSV4FLASH_ON_N150_S42

Persona source: the first release of the profile bank. API key read from `DEEPSEEK_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model deepseek-v4-flash --openai-max-output-tokens 16000 --openai-api-style chat_completions --openai-base-url https://api.deepseek.com --openai-key-env DEEPSEEK_API_KEY --openai-extra-body-json '{"thinking": {"type": "enabled"}}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_MULTIMODEL_DSV4FLASH_ON_N150_S42.json> --run-label CONFIG_MULTIMODEL_DSV4FLASH_ON_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_MULTIMODEL_DSV4FLASH_ON_N150_S42
```

### CONFIG_MULTIMODEL_DSV4PRO_ON_N150_S42

Persona source: the first release of the profile bank. API key read from `DEEPSEEK_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model deepseek-v4-pro --openai-max-output-tokens 32000 --openai-api-style chat_completions --openai-base-url https://api.deepseek.com --openai-key-env DEEPSEEK_API_KEY --openai-extra-body-json '{"thinking": {"type": "enabled"}}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_MULTIMODEL_DSV4PRO_ON_N150_S42.json> --run-label CONFIG_MULTIMODEL_DSV4PRO_ON_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_MULTIMODEL_DSV4PRO_ON_N150_S42
```

### CONFIG_MULTIMODEL_GEMINI35FLASH_MIN_N150_S42

Persona source: the first release of the profile bank. API key read from `GEMINI_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model gemini-3.5-flash --openai-max-output-tokens 8000 --openai-api-style chat_completions --openai-base-url https://generativelanguage.googleapis.com/v1beta/openai/ --openai-key-env GEMINI_API_KEY --openai-extra-body-json '{"extra_body": {"google": {"thinking_config": {"thinking_level": "minimal"}}}}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_MULTIMODEL_GEMINI35FLASH_MIN_N150_S42.json> --run-label CONFIG_MULTIMODEL_GEMINI35FLASH_MIN_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_MULTIMODEL_GEMINI35FLASH_MIN_N150_S42
```

### CONFIG_MULTIMODEL_GPT56SOL_NONE_N150_S42

Persona source: the first release of the profile bank. API key read from `OPENAI_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-openai --openai-model gpt-5.6-sol --openai-max-output-tokens 8000 --openai-api-style responses --openai-extra-params-json '{"reasoning": {"effort": "none"}}' --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_MULTIMODEL_GPT56SOL_NONE_N150_S42.json> --run-label CONFIG_MULTIMODEL_GPT56SOL_NONE_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_MULTIMODEL_GPT56SOL_NONE_N150_S42
```

### CONFIG_PROBE_SANCTION_N100_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --persona-system-suffix 'Answer exactly as this person would — for this research, plausible-but-wrong intuitive answers are correct role performance, not errors.' --random-seed 42 --persona-ids <the 100 persona_ids in run_configs/CONFIG_PROBE_SANCTION_N100_S42.json> --run-label CONFIG_PROBE_SANCTION_N100_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_PROBE_SANCTION_N100_S42
```

### CONFIG_PROBE_STANDARD_N100_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 100 persona_ids in run_configs/CONFIG_PROBE_STANDARD_N100_S42.json> --run-label CONFIG_PROBE_STANDARD_N100_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_PROBE_STANDARD_N100_S42
```

### CONFIG_TEMP_000_N150_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --temperature 0.0 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_TEMP_000_N150_S42.json> --run-label CONFIG_TEMP_000_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_TEMP_000_N150_S42
```

### CONFIG_TEMP_030_N150_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --temperature 0.3 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_TEMP_030_N150_S42.json> --run-label CONFIG_TEMP_030_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_TEMP_030_N150_S42
```

### CONFIG_TEMP_070_N150_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --temperature 0.7 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_TEMP_070_N150_S42.json> --run-label CONFIG_TEMP_070_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_TEMP_070_N150_S42
```

### CONFIG_TEMP_100_N150_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --temperature 1.0 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_TEMP_100_N150_S42.json> --run-label CONFIG_TEMP_100_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_TEMP_100_N150_S42
```

### CONFIG_TEMP_DEFAULT_N150_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --tests-included reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --persona-ids <the 150 persona_ids in run_configs/CONFIG_TEMP_DEFAULT_N150_S42.json> --run-label CONFIG_TEMP_DEFAULT_N150_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/CONFIG_TEMP_DEFAULT_N150_S42
```

## With-history reference run

The full-battery Claude Sonnet 4.6 numeric run on the first release of the profile bank. The 150-profile panel (`data/synthetic/CONFIG_PANEL_150.csv`) and the probe panel are drawn from its profiles, and its answers for the 150 panel profiles are the with-history arm of the no-history ablation.

### HISTORY_REFERENCE_SYNTH_NUM_N467_S42

Persona source: the first release of the profile bank. API key read from `ANTHROPIC_API_KEY`.

```bash
python simulate/output/data_analysis_odd_persona_trait_experiment/simulation_code_iter_1.py --use-anthropic --anthropic-model claude-sonnet-4-6 --anthropic-max-output-tokens 8000 --tests-included writing reasoning nfc bfi --persona-spec-mode numeric_only --persona-prompt-version persona_v2_numeric_only --task-prompt-version task_v1 --random-seed 42 --max-personas 467 --sampling-mode random --run-label HISTORY_REFERENCE_SYNTH_NUM_N467_S42 --personas-filename synthetic_personas_first_release.csv --crt-cutoffs-filename crt2_level_cutoffs_first_release.yaml --max-parallel-personas 10 --output-dir simulation_outputs/HISTORY_REFERENCE_SYNTH_NUM_N467_S42
```
