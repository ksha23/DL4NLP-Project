#!/bin/bash
# Entanglement Gap Experiment Runner
# Requires: conda env 'entangle', HF_TOKEN set
set -e

export CONDA_ENV="entangle"
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PROJ_ROOT/results"
mkdir -p "$RESULTS_DIR"

# Paths to codebases
ACTSVD_DIR="$PROJ_ROOT/Selected Papers/Assessing the Brittlness/code/alignment-attribution-code"
REFUSAL_DIR="$PROJ_ROOT/Selected Papers/Refusal in Language Models/code/refusal_direction"
SUBSPACE_DIR="$PROJ_ROOT/Selected Papers/Safety Subspaces/code/safety-subspaces"

MODEL_INSTRUCT="unsloth/Meta-Llama-3.1-8B-Instruct"
MODEL_BASE="unsloth/Meta-Llama-3.1-8B"

echo "==========================================="
echo "Phase 1: Task A - ActSVD on Llama-3.1"
echo "==========================================="
cd "$ACTSVD_DIR"
conda run --no-banner -n $CONDA_ENV python main_low_rank.py \
    --model llama3.1-8b-instruct \
    --prune_method low_rank \
    --prune_data align_short \
    --rank 10 \
    --nsamples 64 \
    --top_remove \
    --eval_attack \
    --save_attack_res \
    --save "$RESULTS_DIR/llama3_actsvd" \
    --dump_U

echo ""
echo "==========================================="
echo "Phase 2: Task B - Refusal Direction"
echo "==========================================="
cd "$REFUSAL_DIR"
conda run --no-banner -n $CONDA_ENV python -m pipeline.run_pipeline \
    --model_path "$MODEL_INSTRUCT"

echo ""
echo "==========================================="
echo "Phase 3: Task C - Weight SVD"
echo "==========================================="
cd "$SUBSPACE_DIR"
mkdir -p "$RESULTS_DIR/llama3_basis"
conda run --no-banner -n $CONDA_ENV python utils/post_processing_subspace.py \
    --base_model_path "$MODEL_BASE" \
    --aligned_model_path "$MODEL_INSTRUCT" \
    --finetuned_model_path "$MODEL_INSTRUCT" \
    --frac 0.10 \
    --method_type same \
    --dump_basis "$RESULTS_DIR/llama3_basis"

echo ""
echo "==========================================="
echo "Phase 4: Cross-Method Analysis"
echo "==========================================="
cd "$PROJ_ROOT"
conda run --no-banner -n $CONDA_ENV python entanglement_analysis.py \
    --actsvd_dir "$RESULTS_DIR/llama3_actsvd" \
    --refusal_dir "$REFUSAL_DIR/pipeline/runs/Meta-Llama-3.1-8B-Instruct" \
    --weight_basis "$RESULTS_DIR/llama3_basis/alignment_basis.pt" \
    --output_dir "$RESULTS_DIR/entanglement_gap"

echo ""
echo "==========================================="
echo "All experiments complete!"
echo "Results in: $RESULTS_DIR"
echo "==========================================="
