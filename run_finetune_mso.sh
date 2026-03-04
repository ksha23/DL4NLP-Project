#!/bin/bash
# Fine-tune Llama-3.1-8B-Instruct on harmful + useful datasets for MSO analysis
set -e

export CONDA_ENV="entangle"
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PROJ_ROOT/results"
SUBSPACE_DIR="$PROJ_ROOT/Selected Papers/Safety Subspaces/code/safety-subspaces"

MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

echo "==========================================="
echo "Fine-tuning: Harmful (BeaverTails)"
echo "==========================================="
cd "$SUBSPACE_DIR"
conda run --no-banner -n $CONDA_ENV python finetune.py \
    --model "$MODEL" \
    --data_mode harmful \
    --harmful_size 4000 \
    --data_path_contaminated "PKU-Alignment/BeaverTails" \
    --dataset_split_contaminated "30k_train" \
    --batch_size 1 \
    --epochs 1 \
    --lr 1e-4 \
    --max_seq_length 512

echo ""
echo "==========================================="
echo "Fine-tuning: Useful (MetaMathQA)"
echo "==========================================="
conda run --no-banner -n $CONDA_ENV python finetune.py \
    --model "$MODEL" \
    --data_mode pure \
    --data_path "meta-math/MetaMathQA" \
    --dataset_split "train[:20000]" \
    --dataset_field query response \
    --batch_size 1 \
    --epochs 1 \
    --lr 1e-4 \
    --max_seq_length 512

echo ""
echo "==========================================="
echo "Computing MSO between useful and harmful updates"
echo "==========================================="
# After fine-tuning, run MSO analysis
# Note: update paths below to point to actual fine-tuned model outputs
HARMFUL_MODEL="$SUBSPACE_DIR/experiments/arithmetic/Meta-Llama-3.1-8B-Instruct/latest_harmful/final_model"
USEFUL_MODEL="$SUBSPACE_DIR/experiments/arithmetic/Meta-Llama-3.1-8B-Instruct/latest_useful/final_model"

if [ -d "$HARMFUL_MODEL" ] && [ -d "$USEFUL_MODEL" ]; then
    conda run --no-banner -n $CONDA_ENV python exp-3-update_spaces/update_spaces_functions.py \
        --instruct-path "$MODEL" \
        --useful-path "$USEFUL_MODEL" \
        --harmful-path "$HARMFUL_MODEL" \
        --granularity tensor \
        --output-csv "$RESULTS_DIR/mso_llama3.csv" \
        --energy-fracs 0.1 0.3 0.5 0.7 0.9
    echo "MSO results saved to $RESULTS_DIR/mso_llama3.csv"
else
    echo "[WARN] Fine-tuned models not found. Check experiment directories manually."
    echo "Expected harmful: $HARMFUL_MODEL"
    echo "Expected useful: $USEFUL_MODEL"
fi

echo "Done."
