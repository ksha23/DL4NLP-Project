# Agent Prompt: Run ActSVD on RTX 3090

**Objective:** Run the ActSVD (Activation SVD) analysis from the "Assessing the Brittleness" paper on Llama-3.1-8B-Instruct, on a machine with an **NVIDIA RTX 3090 (24 GB VRAM)**. The codebase has already been ported for Llama-3; you need to set up the environment from scratch and execute the pipeline.

---

## Prerequisites

- **Hardware:** NVIDIA RTX 3090 (24 GB VRAM)
- **OS:** Linux (Ubuntu recommended)
- **HuggingFace account** with access to Llama-3.1 models (or use the ungated mirror below)

---

## Step 1: Clone/Navigate to the Project

Ensure you have the DL4NLP-Project repository. The ActSVD code lives in:

```
Selected Papers/Assessing the Brittlness/code/alignment-attribution-code/
```

---

## Step 2: Create Conda Environment

```bash
conda create -n entangle python=3.10 -y
conda activate entangle
```

Install dependencies (use CUDA 12.x for RTX 3090):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes peft datasets
pip install vllm jaxtyping einops lm-eval pandas tqdm safetensors
pip install matplotlib  # required by refusal-direction submodule imports
```

---

## Step 3: Model Access

The code uses **`unsloth/Meta-Llama-3.1-8B-Instruct`** (ungated mirror of Meta's model). No HuggingFace token is required for this mirror.

If you prefer the official Meta model (`meta-llama/Meta-Llama-3.1-8B-Instruct`), you must:
1. Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Log in: `huggingface-cli login` (or set `HF_TOKEN`)

The codebase is configured for `unsloth/Meta-Llama-3.1-8B-Instruct` in `main_low_rank.py` (`modeltype2path`).

---

## Step 4: Verify Data Files Exist

The ActSVD pipeline uses calibration data from:

```
Selected Papers/Assessing the Brittlness/code/alignment-attribution-code/data/
```

Required files:
- `SFT_aligned_llama2-7b-chat-hf_train_short.csv` (for `--prune_data align_short`)

If missing, the dataset may need to be downloaded or the path adjusted in `lib/data.py`.

---

## Step 5: Run ActSVD Pipeline

From the project root:

```bash
cd "Selected Papers/Assessing the Brittlness/code/alignment-attribution-code"

conda activate entangle

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

**Flags:**
- `--top_remove`: Remove top ranks (safety-critical directions)
- `--eval_attack`: Run jailbreak/attack evaluation on AdvBench
- `--save_attack_res`: Save attack completions to JSONL
- `--dump_U`: Save SVD vectors for cross-method comparison

**Expected runtime:** ~12–20 minutes for make_low_rank (32 layers), ~30–60 min for attack eval depending on dataset size.

---

## Step 6: Expected Outputs

After completion:

1. **`results/llama3_actsvd/log_low_rank.txt`  
   - PPL (Wikitext perplexity)  
   - ASR (Attack Success Rate) for various configs

2. **`results/llama3_actsvd/align_short/proj_mat/10/`**  
   - SVD `V` matrices for each layer (e.g. `V_model.layers.0.self_attn.q_proj_align_short.pkl`)

3. **`results/llama3_actsvd/attack_10/`**  
   - Attack completions (JSONL files)

---

## Step 7: Success Criteria

- **PPL** should remain reasonable after top-rank removal (e.g. < 30)
- **ASR** should increase significantly after top-rank removal (model should jailbreak)

If both hold, the ActSVD intervention successfully removes safety while preserving utility.

---

## Troubleshooting

- **CUDA OOM:** If you still hit OOM on 24 GB, reduce `--nsamples` to 64.
- **vLLM:** Ensure vLLM is built for your CUDA version. Reinstall if needed: `pip install vllm --no-cache-dir`
- **Tokenizer:** Llama-3 uses `use_fast=True`; the code already sets this.

---

## Notes

- The RTX 3090 has 24 GB VRAM; the full 8B model in fp16 (~16 GB) fits with room for activations. No `max_memory` or CPU offloading hacks are needed.
- The codebase is already configured for Llama-3.1 (chat templates, model paths, tokenizer).
