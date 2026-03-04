# Entanglement Gap Investigation

## Overview

This project investigates the **Entanglement Gap Hypothesis**: whether newer language models (Llama-3) have safety mechanisms more deeply entangled in their weights compared to earlier models (Llama-2), despite both appearing modular in activation space.

Three complementary analysis methods are ported from their original codebases and applied to `meta-llama/Meta-Llama-3.1-8B-Instruct`:

| Task | Method | Paper | Measures |
|------|--------|-------|----------|
| A | ActSVD (Activation SVD) | Wei et al. 2024 - Assessing Brittleness | Safety in activation space |
| B | Difference-of-Means | Arditi et al. 2024 - Refusal in LLMs | Refusal direction in residual stream |
| C | Weight SVD / MSO | Ponkshe et al. 2026 - Safety Subspaces | Safety entanglement in weight space |

## Prerequisites

- **GPU**: NVIDIA RTX 5070 Ti (16 GB VRAM) or equivalent
- **CUDA**: 12.x+
- **HuggingFace Token**: Required for gated Llama-3.1 model access

## Setup

```bash
# 1. Environment (already created)
conda activate entangle

# 2. Set HuggingFace token
huggingface-cli login --token YOUR_TOKEN_HERE
# or
export HF_TOKEN=YOUR_TOKEN_HERE
```

## Running Experiments

### Quick Start (All experiments sequentially)
```bash
./run_experiments.sh
```

### Individual Tasks

**Task A: ActSVD**
```bash
cd "Selected Papers/Assessing the Brittlness/code/alignment-attribution-code"
conda run -n entangle python main_low_rank.py \
    --model llama3.1-8b-instruct \
    --prune_method low_rank \
    --prune_data align_short \
    --rank 10 --top_remove --eval_attack --save_attack_res \
    --save results/llama3_actsvd --dump_U
```

**Task B: Refusal Direction**
```bash
cd "Selected Papers/Refusal in Language Models/code/refusal_direction"
conda run -n entangle python -m pipeline.run_pipeline \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Task C: Weight SVD**
```bash
cd "Selected Papers/Safety Subspaces/code/safety-subspaces"
conda run -n entangle python utils/post_processing_subspace.py \
    --base_model_path meta-llama/Meta-Llama-3.1-8B \
    --aligned_model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --finetuned_model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --frac 0.10 --method_type same \
    --dump_basis results/llama3_basis
```

**Cross-method Analysis**
```bash
python entanglement_analysis.py \
    --actsvd_dir results/llama3_actsvd \
    --refusal_dir "Selected Papers/Refusal in Language Models/code/refusal_direction/pipeline/runs/Meta-Llama-3.1-8B-Instruct" \
    --weight_basis results/llama3_basis/alignment_basis.pt
```

**Final Report**
```bash
python generate_report.py --results_dir results/
```

### Fine-tuning for MSO Analysis (Optional, compute-intensive)
```bash
./run_finetune_mso.sh
```

## Key Code Changes (from original papers)

### Task A modifications
- `main_low_rank.py`: Added Llama-3.1 to `modeltype2path`, 8-bit loading for VRAM
- `lib/data.py`: `get_align()` uses `tokenizer.apply_chat_template()` for Llama-3
- `lib/prompt_utils.py`: Added `apply_prompt_template_llama3()` with proper header tokens
- `lib/eval.py`: `eval_attack()` accepts `model_family` parameter

### Task B modifications
- `pipeline/model_utils/llama3_model.py`: Added `load_in_8bit=True` for 8B models

### Task C modifications
- `utils/post_processing_subspace.py`: Load to CPU (not GPU), added `--dump_basis` flag

## VRAM Strategy

All 8B models use 8-bit quantization (`load_in_8bit=True`) to fit in 16 GB VRAM.
Weight SVD loads state_dicts to CPU and processes per-layer on GPU.

## Output Structure

```
results/
├── llama3_actsvd/          # Task A: PPL, ASR, SVD vectors
│   ├── log_low_rank.txt
│   ├── attack_10/
│   └── align_short/proj_mat/
├── llama3_basis/           # Task C: alignment basis vectors
│   └── alignment_basis.pt
├── mso_llama3.csv          # MSO analysis (after fine-tuning)
├── entanglement_gap/       # Cross-method comparison
│   ├── entanglement_gap_report.txt
│   └── entanglement_gap_results.json
└── FINAL_REPORT.txt        # Comprehensive report
```
