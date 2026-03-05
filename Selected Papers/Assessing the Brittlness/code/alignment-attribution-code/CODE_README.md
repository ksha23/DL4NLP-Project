# Code README — Assessing the Brittleness (ActSVD)

**Paper**: "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications" (Wei et al., 2024)

This directory contains the code for Activation-aware SVD (ActSVD) — a method that identifies and removes safety-critical weight subspaces from aligned LLMs using activation-weighted singular value decomposition.

---

## Top-Level Scripts

### `main_low_rank.py`
**The primary script we used.** Runs the naive ActSVD pipeline:
1. Loads the model and calibration data (safety-aligned SFT prompts)
2. Wraps every Linear layer with `ActLinear` to record activations
3. Runs calibration samples through the model to collect activation statistics
4. Performs activation-weighted SVD on each weight matrix
5. Removes the top-k singular vectors (the safety-critical subspace)
6. Evaluates perplexity (WikiText-2) and attack success rate (vLLM + AdvBench)

```bash
python main_low_rank.py --model llama3.1-8b-instruct --prune_method low_rank \
    --prune_data align_short --rank 10 --nsamples 128 --top_remove \
    --eval_attack --save_attack_res --save results/llama3_actsvd --dump_U
```

### `main_low_rank_diff_llama3.py`
**Orthogonal projection variant for Llama 3.** Implements the paper's orthogonal projection method (Section 4) adapted for Llama-3.1-8B-Instruct:
- Collects activations from two datasets: "positive" (general utility, e.g. Alpaca) and "negative" (safety-aligned SFT)
- Computes ActSVD on each, identifies safety-specific subspace via set difference
- Projects out the safety subspace while preserving the utility subspace
- Handles Llama-3's GQA (grouped query attention) with proportional rank scaling for k_proj/v_proj

```bash
python main_low_rank_diff_llama3.py --model llama3.1-8b-instruct \
    --prune_data_pos alpaca_cleaned_no_safety --prune_data_neg align_short \
    --rank_pos 4000 --rank_neg 4090 --nsamples 128 \
    --eval_attack --save_attack_res --save results/llama3_ortho_proj
```

### `main_low_rank_diff.py`
**Original orthogonal projection script** from the paper authors. Designed for Llama-2. The `_llama3.py` variant above is our adaptation with GQA handling and Llama-3 chat template fixes.

### `main.py`
**Original paper script** supporting multiple pruning methods (Wanda, SparseGPT, magnitude pruning, etc.) and various experiments. We didn't use this directly — `main_low_rank.py` is the focused low-rank version.

### `_run_attack_eval.py`
**Subprocess helper for vLLM attack evaluation.** Launched as a separate process by `main_low_rank.py` and `main_low_rank_diff_llama3.py` so that vLLM gets a completely fresh CUDA context (avoids GPU OOM from leftover PyTorch memory). Loads the saved modified model with vLLM and generates responses to AdvBench prompts across multiple attack configurations (with/without system prompt, single/multiple templates).

### `run_baseline_eval.py`
**Baseline evaluation script.** Evaluates the original unmodified model on perplexity and attack success rate for comparison. Not yet run for our experiments.

### `chat.py`
**Interactive chat interface using vLLM.** Supports three modes:
- `--mode original` — Chat with the safety-aligned model
- `--mode jailbroken` — Chat with the modified model (specify path with `--jailbroken-path`)
- `--mode compare` — Side-by-side comparison of both models

```bash
python chat.py --mode jailbroken --jailbroken-path temp/tmp_vllm_model_ortho
```

### `plot_results.py`
**Visualization script.** Parses result log files and generates 4 plots:
- ASR comparison (bar chart: baseline vs ActSVD)
- PPL comparison (bar chart)
- ASR heatmap by attack type and configuration
- ASR by attack configuration (grouped bars)

Output saved to `results/plots/`.

### `rewind_ft_model.py`
**Weight rewinding script** from the original paper. Takes a fine-tuned (jailbroken) model and selectively "rewinds" specific weights back to the original aligned model using a binary mask. Used for the paper's fine-tuning experiments — not used in our ActSVD experiments.

---

## `lib/` — Core Library

### `lib/data.py`
**Data loading and tokenization.** Loads calibration datasets (safety-aligned SFT, Alpaca, AdvBench) and tokenizes them with the appropriate chat template. Contains Llama-2 and Llama-3 specific paths. Fixed to handle Llama-3's `apply_chat_template` BOS token correctly (no double-BOS).

### `lib/eval.py`
**Evaluation functions:**
- `eval_ppl()` — Perplexity evaluation on WikiText-2 (4096-token chunks)
- `eval_attack()` — Attack success rate using vLLM generation + substring matching on AdvBench prompts (3 configs: basic, basic_nosys, multiple_nosys)
- `eval_zero_shot()` — Zero-shot benchmark evaluation (not used in our experiments)

### `lib/model_wrapper_low.py`
**Activation recording wrapper (full activations).** Contains:
- `ActLinear` — Drop-in `nn.Linear` replacement that records full activation tensors (offloaded to CPU)
- `make_Act()` — Wraps all Linear layers in a model with `ActLinear`
- `make_low_rank()` — Performs activation-weighted SVD: computes `S = (X^T X)^{1/2}`, then SVD on `W·S`, removes top-k components
- `no_act_recording` / `set_mask` — Context managers for controlling activation recording

### `lib/model_wrapper.py`
**Activation recording wrapper (norm-based).** Similar to `model_wrapper_low.py` but stores activation *norms* rather than full activations. Used by the pruning methods in `main.py`. Contains Wanda v2 and WandG implementations.

### `lib/prune.py`
**Pruning methods library (2161 lines).** Implements multiple structured/unstructured pruning algorithms:
- Wanda (weight × activation magnitude)
- SparseGPT (second-order pruning)
- Magnitude pruning
- Ablation-based pruning
- Attention head pruning
- Wanda with set-difference for disentangled pruning

### `lib/ablate.py`
**Ablation-based weight modification.** Implements `AblateGPT` class that collects Hessian information and performs weight ablation (setting specific weight elements to zero based on importance scores).

### `lib/sparsegpt.py`
**SparseGPT implementation.** Second-order pruning method that uses Hessian information for optimal weight removal. Based on the original SparseGPT paper.

### `lib/layerwrapper.py`
**Layer wrapper for activation collection.** `WrappedGPT` class wraps individual Linear layers to collect activation statistics during calibration, used by the pruning methods.

### `lib/prompt_utils.py`
**Prompt template utilities.** Defines chat templates for Llama-2 and Llama-3, system prompts for different evaluation modes (base, pure_bad, alpaca, etc.), and the `apply_prompt_template()` function used by `eval.py`.

---

## `data/` — Datasets

| File | Description |
|---|---|
| `SFT_aligned_llama2-7b-chat-hf_train.csv` | Full safety-aligned SFT training data (prompt-response pairs from Llama-2 chat alignment) |
| `SFT_aligned_llama2-7b-chat-hf_train_short.csv` | Shortened version (used with `--prune_data align_short`) — same data, fewer samples |
| `alpaca_cleaned_no_safety_train.csv` | Cleaned Alpaca dataset with safety-related examples removed — used as "positive" (utility) calibration data |
| `advbench.txt` | AdvBench harmful instructions (520 prompts) — used for attack evaluation |
| `probing_result_7b.json` | Pre-computed probing results for Llama-2 7B |
| `probing_result_13b.json` | Pre-computed probing results for Llama-2 13B |

---

## `results/` — Experiment Outputs

### `results/llama3_actsvd/`
Naive ActSVD results (rank=10, top_remove):
- `log_low_rank.txt` — PPL and ASR summary
- `align_short/` — Saved SVD decomposition artifacts (U matrices)
- `attack_10/` — Per-prompt attack results (.jsonl files)

### `results/llama3_ortho_proj/`
Orthogonal projection results:
- `log_ortho_proj.txt` — PPL and ASR for all runs (3996/3996, 4000/4090 buggy, 4000/4090 fixed)
- `attack_3996_3996/` and `attack_4000_4090/` — Per-prompt attack results

### `results/plots/`
Generated visualizations: `asr_comparison.png`, `ppl_comparison.png`, `asr_by_config.png`, `summary_table.png`

---

## `temp/` — Temporary Model Checkpoints

| Directory | Description |
|---|---|
| `temp/tmp_vllm_model/` | Naive ActSVD modified model (~15 GB) — PPL=1.8M (destroyed utility), used for chat |
| `temp/tmp_vllm_model_ortho/` | Orthogonal projection model (~15 GB) — PPL=10.43, ASR≈0.96, good utility |

---

## Config Files

| File | Description |
|---|---|
| `default_config.yaml` | Accelerate/FSDP config for distributed training (not used in our single-GPU setup) |
| `environment.yml` | Conda environment spec from original authors |
| `COMMANDS.md` | Reference of all commands we ran and their results |
| `LICENSE` | MIT License |
| `README.md` | Original paper authors' README |

---

## Key Results Summary

| Method | Params | PPL | ASR (basic) | ASR (nosys) | ASR (multi) |
|---|---|---|---|---|---|
| **Naive ActSVD** | rank=10, top_remove | 1,818,062 | 0.00 | 0.72 | 1.00 |
| **Ortho Projection** | r_u=4000, r_s=4090 | **10.43** | **0.96** | **0.91** | **0.91** |
