# Code README — Refusal Direction Pipeline

> Paper: **"Refusal in Language Models Is Mediated by a Single Direction"** (Arditi et al., 2024)

This codebase implements the **difference-of-means** method for finding a single "refusal direction" in a language model's residual stream, plus several interventions (directional ablation, activation addition, weight orthogonalization) to remove or induce refusal behavior.

---

## Top-Level Files

### `pipeline/run_pipeline.py`
**Main entry point.** Runs the full 5-step pipeline:
1. Generate candidate refusal directions via difference-of-means on harmful vs. harmless activations
2. Select the best direction by evaluating ablation refusal scores, steering scores, and KL divergence
3. Generate completions on harmful benchmarks (baseline, ablation, actadd)
4. Generate completions on harmless benchmarks (baseline, actadd with +1 coeff to induce refusal)
5. Evaluate cross-entropy loss on Pile/Alpaca datasets for each intervention

**Usage:**
```bash
python -m pipeline.run_pipeline --model_path unsloth/Meta-Llama-3.1-8B-Instruct
```

### `orthogonalize_and_save.py`
**Weight orthogonalization script (we created this).** Applies Equation 5 from the paper — a permanent, rank-1 weight modification that removes the refusal direction from all residual-stream-writing matrices:
- Embedding matrix (`embed_tokens`)
- Attention output projections (`o_proj`) in every layer
- MLP down projections (`down_proj`) in every layer

Loads the pre-computed `direction.pt` from a pipeline run and saves a full modified model to disk.

**Usage:**
```bash
python orthogonalize_and_save.py \
    --model_path unsloth/Meta-Llama-3.1-8B-Instruct \
    --direction_path pipeline/runs/Meta-Llama-3.1-8B-Instruct/direction.pt \
    --output_path ortho_model
```

### `chat.py`
**Interactive chat interface (we created this).** Uses HuggingFace `generate()` (not vLLM) for fast startup. Three modes:
- `--mode original` — Chat with the safety-aligned model (will refuse harmful prompts)
- `--mode ortho` — Chat with the weight-orthogonalized model (refusal removed)
- `--mode compare` — Side-by-side comparison (loads both models in 8-bit to fit on a single GPU)

**Usage:**
```bash
python chat.py --mode ortho --ortho_path ortho_model
python chat.py --mode compare --ortho_path ortho_model --load_in_8bit
```

### `requirements.txt`
Full pinned dependency list (156 lines). Key packages: `transformers`, `accelerate`, `torch`, `vllm`, `jaxtyping`, `einops`, `litellm`, `datasets`.

### `setup.sh`
Environment setup script. Checks Python ≥ 3.10, optionally configures HuggingFace and Together AI tokens, creates a venv, and installs requirements.

### `COMMANDS.md`
Quick-reference command sheet for running the pipeline, orthogonalization, and chat.

### `LICENSE`
MIT License.

### `README.md`
Original project README with overview and instructions.

---

## `pipeline/` — Core Pipeline

### `pipeline/config.py`
**Dataclass configuration** for the pipeline. Key defaults:
- `n_train = 128` — training examples per split
- `n_val = 32` — validation examples per split
- `n_test = 100` — test examples
- `filter_train = True` / `filter_val = True` — filter examples by refusal scores
- `evaluation_datasets = ("jailbreakbench",)` — which benchmark to evaluate on
- `max_new_tokens = 512`
- `jailbreak_eval_methodologies = ("substring_matching",)` — ASR evaluation method
- `ce_loss_batch_size = 2`, `ce_loss_n_batches = 2048` — loss evaluation settings

Artifacts are saved under `pipeline/runs/{model_alias}/`.

### `pipeline/__init__.py`
Empty package init.

---

## `pipeline/submodules/` — Pipeline Stages

### `pipeline/submodules/generate_directions.py`
**Step 1: Difference-of-means.** Computes mean activations for harmful and harmless instruction sets across all layers and token positions, then takes the difference. Uses forward pre-hooks to accumulate activations at specified positions (the end-of-instruction token positions).

Key function: `generate_directions(model_base, harmful_instructions, harmless_instructions, artifact_dir)` → saves `mean_diffs.pt` (shape: `[n_positions, n_layers, d_model]`).

### `pipeline/submodules/select_direction.py`
**Step 2: Direction selection.** Evaluates every candidate direction (each layer × each position) on three criteria:
1. **Ablation refusal score** — Does ablating this direction reduce refusal on harmful prompts?
2. **Steering refusal score** — Does adding this direction induce refusal on harmless prompts?
3. **KL divergence** — How much does ablating this direction change the output distribution on harmless prompts?

Filters out directions from the last 20% of layers, those above a KL threshold (0.1), and those that can't induce refusal. Selects the best remaining direction. Produces diagnostic plots (`ablation_scores.png`, `actadd_scores.png`, `kl_div_scores.png`).

Key function: `select_direction(model_base, harmful_val, harmless_val, candidate_directions, ...)` → returns `(pos, layer, direction)`.

Also contains helper functions: `get_refusal_scores()` (computes log-odds of refusal tokens), `kl_div_fn()`, `plot_refusal_scores()`.

### `pipeline/submodules/evaluate_jailbreak.py`
**Step 3: Jailbreak ASR evaluation.** Three evaluation methodologies:
1. **`substring_matching`** — Checks if the model's response starts with known refusal prefixes ("I'm sorry", "I cannot", "As an AI", etc.). No refusal prefix = jailbreak success.
2. **`llamaguard2`** — Uses Together AI API to query Meta-Llama/LlamaGuard-2-8B for safety classification.
3. **`harmbench`** — Uses a local `cais/HarmBench-Llama-2-13b-cls` classifier via vLLM to judge whether the completion is harmful.

Reports per-category and average ASR.

### `pipeline/submodules/evaluate_loss.py`
**Step 5: Cross-entropy loss evaluation.** Measures model utility degradation by computing perplexity on three datasets:
- **Pile** (general language, streamed from `monology/pile-uncopyrighted`)
- **Alpaca** (instruction-following, from `tatsu-lab/alpaca`)
- **Alpaca with custom completions** (using model's own baseline completions)

Uses a loss mask to only compute loss on the response portion (after end-of-instruction tokens).

---

## `pipeline/model_utils/` — Model Adapters

Each supported model family has its own adapter implementing the `ModelBase` abstract class. The adapter handles model-specific details: chat template formatting, tokenization, refusal token IDs, weight orthogonalization, and activation addition.

### `pipeline/model_utils/model_base.py`
**Abstract base class.** Defines the interface:
- `_load_model()` / `_load_tokenizer()` — loading
- `_get_tokenize_instructions_fn()` — returns a function that formats + tokenizes instructions
- `_get_eoi_toks()` — end-of-instruction token IDs (used for activation extraction positions)
- `_get_refusal_toks()` — token IDs that indicate refusal (e.g., "I" for "I cannot")
- `_get_model_block_modules()` / `_get_attn_modules()` / `_get_mlp_modules()` — module accessors for hooks
- `_get_orthogonalization_mod_fn()` / `_get_act_add_mod_fn()` — weight modification functions
- `generate_completions()` — batch generation with optional forward hooks

### `pipeline/model_utils/model_factory.py`
**Factory function.** Routes `model_path` to the correct adapter:
- `"llama-3"` → `Llama3Model`
- `"llama"` → `Llama2Model`
- `"gemma"` → `GemmaModel`
- `"qwen"` → `QwenModel`
- `"yi"` → `YiModel`

### `pipeline/model_utils/llama3_model.py`
**Llama 3 / 3.1 adapter.** Uses the `<|start_header_id|>...<|eot_id|>` chat template. Loads 8B models in 8-bit quantization. Refusal token: `[40]` (token for "I").

### `pipeline/model_utils/llama2_model.py`
**Llama 2 adapter.** Uses the `[INST] ... [/INST]` chat template with optional `<<SYS>>` system prompt. Refusal token: `[306]` (token for "I" in Llama-2 vocabulary).

### `pipeline/model_utils/gemma_model.py`
**Gemma adapter.** Uses the `<start_of_turn>user ... <end_of_turn>` template. No system prompt support. Refusal token: `[235285]`.

### `pipeline/model_utils/qwen_model.py`
**Qwen adapter.** Uses the `<|im_start|> ... <|im_end|>` template. Supports flash attention. Refusal tokens: `[40, 2121]` ("I", "As").

### `pipeline/model_utils/yi_model.py`
**Yi adapter.** Uses the same `<|im_start|>` template as Qwen. Refusal tokens: `[59597]` ("I").

### `pipeline/model_utils/__init__.py`
Empty package init.

---

## `pipeline/utils/` — Utility Functions

### `pipeline/utils/hook_utils.py`
**Forward hook utilities.** Core mechanism for intervening on model activations at runtime:
- `add_hooks()` — Context manager that temporarily registers forward pre-hooks and forward hooks on specified modules
- `get_direction_ablation_input_pre_hook(direction)` — Pre-hook that projects out a direction from the input activation (removes the component along `direction`)
- `get_direction_ablation_output_hook(direction)` — Same but applied to module output
- `get_all_direction_ablation_hooks(model_base, direction)` — Creates ablation hooks for every layer's block input, attention output, and MLP output
- `get_activation_addition_input_pre_hook(vector, coeff)` — Pre-hook that adds `coeff * vector` to the activation (for steering / activation addition)
- `get_directional_patching_input_pre_hook()` — Hook for directional patching experiments

### `pipeline/utils/utils.py`
**Matrix orthogonalization utility.** Single function `get_orthogonalized_matrix(matrix, vec)` — projects out the component along `vec` from every row of `matrix` using einops.

### `pipeline/utils/__init__.py`
Empty package init.

---

## `pipeline/runs/` — Pipeline Results

Pre-computed results for multiple models. Each subdirectory contains the full pipeline output.

### `pipeline/runs/Meta-Llama-3.1-8B-Instruct/`
Our primary run. Contains:
- **`direction.pt`** — The selected refusal direction tensor (shape `[4096]`)
- **`direction_metadata.json`** — Selected position (`pos: -2`) and layer (`layer: 11`)
- **`generate_directions/mean_diffs.pt`** — All candidate directions (shape `[n_positions, n_layers, d_model]`)
- **`select_direction/`** — Diagnostic plots and score JSONs:
  - `ablation_scores.png` — Refusal scores when ablating each candidate direction
  - `actadd_scores.png` — Refusal scores when adding each candidate direction
  - `kl_div_scores.png` — KL divergence for each ablation
  - `direction_evaluations.json` — All scores for all candidates
  - `direction_evaluations_filtered.json` — Filtered and sorted candidates
- **`completions/`** — Generated completions and evaluations:
  - `jailbreakbench_{baseline,ablation,actadd}_completions.json` — Model responses
  - `jailbreakbench_{baseline,ablation,actadd}_evaluations.json` — ASR results
  - `harmless_{baseline,actadd}_completions.json` — Harmless prompt responses
  - `harmless_{baseline,actadd}_evaluations.json` — Refusal-on-harmless results
- **`loss_evals/`** — Perplexity results:
  - `baseline_loss_eval.json` — Pile PPL ≈ 8.76
  - `ablation_loss_eval.json` — Pile PPL ≈ 8.80 (minimal degradation)
  - `actadd_loss_eval.json` — Pile PPL ≈ 10.19

Other model runs: `gemma-2b-it/`, `llama-2-7b-chat-hf/`, `meta-llama-3-8b-instruct/`, `qwen-1_8b-chat/`, `yi-6b-chat/`.

---

## `ortho_model/` — Saved Orthogonalized Model

The weight-orthogonalized Llama-3.1-8B-Instruct model (~16 GB). Created by `orthogonalize_and_save.py`. Contains:
- 4 safetensor shards (`model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors`)
- `model.safetensors.index.json` — shard index
- `config.json`, `generation_config.json` — model configuration
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja` — tokenizer files
- `ortho_metadata.json` — records source model, direction path, method, and direction layer/position

---

## `dataset/` — Datasets

### `dataset/load_dataset.py`
**Dataset loading functions:**
- `load_dataset_split(harmtype, split)` — Loads harmful/harmless train/val/test splits from `splits/`
- `load_dataset(dataset_name)` — Loads a specific processed benchmark dataset from `processed/`

### `dataset/generate_datasets.ipynb`
**Jupyter notebook** for downloading, processing, and splitting the raw datasets into the standard format.

### `dataset/__init__.py`
Empty package init.

### `dataset/raw/`
Raw benchmark datasets as downloaded:
- `advbench.csv` — AdvBench harmful behavior dataset
- `harmbench_val.csv`, `harmbench_test.csv` — HarmBench validation and test sets
- `jailbreakbench.csv` — JailbreakBench dataset
- `malicious_instruct.txt` — Malicious instructions dataset
- `strongreject.csv` — StrongREJECT dataset
- `tdc2023_dev_behaviors.json`, `tdc2023_test_behaviors.json` — TDC 2023 behaviors

### `dataset/processed/`
Processed datasets in unified JSON format (`[{"instruction": "...", "category": "..."}, ...]`):
- `advbench.json`, `harmbench_val.json`, `harmbench_test.json`
- `jailbreakbench.json`, `malicious_instruct.json`, `strongreject.json`, `tdc2023.json`
- `alpaca.json` — Harmless instruction-following dataset

### `dataset/splits/`
Train/validation/test splits used by the pipeline:
- `harmful_train.json`, `harmful_val.json`, `harmful_test.json`
- `harmless_train.json`, `harmless_val.json`, `harmless_test.json`

---

## Key Results (Llama-3.1-8B-Instruct)

| Intervention | JailbreakBench ASR | Pile PPL |
|---|---|---|
| Baseline | 0.21 | 8.76 |
| Directional ablation | 1.00 | 8.80 |
| Activation addition | 1.00 | 10.19 |

- **Selected direction**: position = -2, layer = 11
- **Minimal utility impact**: PPL only increases from 8.76 → 8.80 with ablation
