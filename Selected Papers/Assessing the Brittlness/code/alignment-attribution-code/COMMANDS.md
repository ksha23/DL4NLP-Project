# Common Commands Reference

All commands should be run from:
```
cd "Selected Papers/Assessing the Brittlness/code/alignment-attribution-code"
conda activate entangle
```

---

## 1. Naive ActSVD (Top-Rank Removal)

Removes the top-r safety-critical singular value ranks. **Destroys utility** (PPL explodes).

```bash
python main_low_rank.py \
  --model llama3.1-8b-instruct \
  --prune_method low_rank \
  --prune_data align_short \
  --rank 10 \
  --nsamples 128 \
  --top_remove \
  --eval_attack \
  --save_attack_res \
  --save results/llama3_actsvd \
  --dump_U
```

**Key arguments:**
- `--rank N` — Number of top singular value ranks to remove (default: 10)
- `--top_remove` — Required flag to remove top ranks (without this, removes bottom ranks)
- `--prune_data` — Dataset for activation collection: `align_short`, `align`, `alpaca_cleaned_no_safety`
- `--dump_U` — Save the U matrices (singular vectors) to disk for later analysis
- `--nsamples` — Number of calibration samples (default: 128)

**Results saved to:** `results/llama3_actsvd/`

---

## 2. ActSVD with Orthogonal Projection

Removes safety ranks that are **orthogonal to utility** ranks: ΔW = (I - Π_u) · Π_s · W.  
**Preserves utility** while breaking safety.

```bash
python main_low_rank_diff_llama3.py \
  --model llama3.1-8b-instruct \
  --prune_data_pos alpaca_cleaned_no_safety \
  --prune_data_neg align_short \
  --rank_pos 4000 \
  --rank_neg 4090 \
  --nsamples 128 \
  --eval_attack \
  --save_attack_res \
  --save results/llama3_ortho_proj
```

**Key arguments:**
- `--rank_pos N` — Number of LOW utility ranks to remove (keeps top R−rank_pos utility ranks)
- `--rank_neg N` — Number of LOW safety ranks to remove (keeps top R−rank_neg safety ranks)
- `--prune_data_pos` — Utility dataset (default: `alpaca_cleaned_no_safety`)
- `--prune_data_neg` — Safety dataset (default: `align_short`)
- `--niter N` — SVD iterations (default: 20)

**Paper's best hyperparameters (Llama-2-7B):**
- `rank_pos=4000, rank_neg=4090` → keeps 96 utility ranks, 6 safety ranks, rank(ΔW) ≤ 6
- This gives PPL ≈ 9.89, ASR ≈ 0.90 on Llama-3.1-8B-Instruct

**Understanding the parameters:**
- Higher `rank_pos` = more low-utility ranks removed = fewer utility ranks kept (more aggressive)
- Higher `rank_neg` = more low-safety ranks removed = fewer safety ranks kept in Π_s
- The perturbation rank is bounded by: rank(ΔW) ≤ min(rank_pos, R − rank_neg)

**Results saved to:** `results/llama3_ortho_proj/`

---

## 3. Baseline Evaluation (Original Model)

Evaluate the **unmodified** model's PPL and attack success rate for comparison.

```bash
python run_baseline_eval.py \
  --model llama3.1-8b-instruct \
  --save results/llama3_baseline
```

**Results saved to:** `results/llama3_baseline/`

---

## 4. Plot Generation

Generate comparison plots from the results.

```bash
python plot_results.py \
  --actsvd_dir results/llama3_actsvd \
  --baseline_dir results/llama3_baseline \
  --output_dir results/plots
```

**Generates:**
- `results/plots/asr_comparison.png`
- `results/plots/ppl_comparison.png`
- `results/plots/asr_by_config.png`
- `results/plots/summary_table.png`

> If baseline results don't exist, only ActSVD results are plotted.

---

## 5. Interactive Chat

### Chat with the original (safety-aligned) model:
```bash
python chat.py --mode original
```

### Chat with the naive ActSVD jailbroken model (gibberish — utility destroyed):
```bash
python chat.py --mode jailbroken
```

### Chat with the orthogonal projection jailbroken model (coherent + jailbroken):
```bash
python chat.py --mode jailbroken --jailbroken-path temp/tmp_vllm_model_ortho
```

### Chat without system prompt (higher ASR):
```bash
python chat.py --mode jailbroken --jailbroken-path temp/tmp_vllm_model_ortho --no-system
```

### Side-by-side comparison (original vs. jailbroken):
```bash
python chat.py --mode compare --jailbroken-path temp/tmp_vllm_model_ortho --no-system
```

**Chat options:**
- `--no-system` — Disable the safety system prompt (increases attack success)
- `--system "..."` — Use a custom system prompt
- `--max-tokens N` — Max tokens to generate (default: 512)

---

## 6. View Results Logs

```bash
# Naive ActSVD results
cat results/llama3_actsvd/log_low_rank.txt

# Orthogonal projection results
cat results/llama3_ortho_proj/log_ortho_proj.txt

# Baseline results
cat results/llama3_baseline/log_baseline.txt
```

---

## Notes

- **Saved models** are in `temp/`:
  - `temp/tmp_vllm_model/` — Naive ActSVD model
  - `temp/tmp_vllm_model_ortho/` — Orthogonal projection model
- **Model weights cache:** `llm_weights/`
- **All pipelines** save the modified model, evaluate PPL, then run attack eval in a subprocess (needed for RTX 3090's 24GB VRAM).
- **Environment variable** `VLLM_ENABLE_V1_MULTIPROCESSING=0` is set automatically by the scripts for vLLM 0.16.0 compatibility.
