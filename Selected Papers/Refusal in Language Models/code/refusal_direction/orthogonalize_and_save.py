"""
Apply weight orthogonalization from "Refusal in Language Models Is Mediated by a Single Direction"
to create a permanently jailbroken model.

This script:
1. Loads the original model
2. Loads the pre-computed refusal direction (from the pipeline run)
3. Applies weight orthogonalization (Equation 5 from the paper)
4. Saves the modified model to disk

Usage:
    python orthogonalize_and_save.py \
        --model_path unsloth/Meta-Llama-3.1-8B-Instruct \
        --direction_path pipeline/runs/Meta-Llama-3.1-8B-Instruct/direction.pt \
        --output_path ortho_model
"""

import torch
import argparse
import os
import json
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_orthogonalized_matrix(matrix, vec):
    """
    Orthogonalize the rows/columns of matrix with respect to vec.
    W' = W - r_hat @ r_hat^T @ W  (for column vectors)
    """
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix.device, dtype=matrix.dtype)
    proj = (matrix @ vec.unsqueeze(-1)) * vec.unsqueeze(0)
    return matrix - proj


def orthogonalize_weights(model, direction):
    """
    Apply weight orthogonalization (Section 4.1 / Equation 5).
    
    Modifies all matrices that write to the residual stream:
    - Embedding matrix
    - Attention output projection (o_proj) 
    - MLP down projection (down_proj)
    
    This prevents the model from ever writing the refusal direction
    to its residual stream.
    """
    direction = direction.to(dtype=torch.float32)
    direction = direction / direction.norm()

    print("Orthogonalizing embedding matrix...")
    W = model.model.embed_tokens.weight.data.float()
    # embed_tokens: [vocab_size, d_model] — each row is a token embedding
    # We want to remove the component along direction from each row
    proj = (W @ direction.unsqueeze(-1)) * direction.unsqueeze(0)
    model.model.embed_tokens.weight.data = (W - proj).to(model.model.embed_tokens.weight.dtype)

    n_layers = len(model.model.layers)
    for layer_idx in range(n_layers):
        block = model.model.layers[layer_idx]
        
        # o_proj: [d_model, num_heads * head_dim] — output is o_proj @ attn_output
        # The output goes to the residual stream, so we orthogonalize the output rows
        # W_out' = W_out - r_hat @ r_hat^T @ W_out  (applied to W_out^T then transposed)
        W_o = block.self_attn.o_proj.weight.data.float()  # [d_model, d_inner]
        proj_o = direction.unsqueeze(-1) @ (direction.unsqueeze(0) @ W_o)
        block.self_attn.o_proj.weight.data = (W_o - proj_o).to(block.self_attn.o_proj.weight.dtype)

        # down_proj: [d_model, intermediate_size] — output goes to residual stream
        W_d = block.mlp.down_proj.weight.data.float()  # [d_model, intermediate_size]
        proj_d = direction.unsqueeze(-1) @ (direction.unsqueeze(0) @ W_d)
        block.mlp.down_proj.weight.data = (W_d - proj_d).to(block.mlp.down_proj.weight.dtype)

        if (layer_idx + 1) % 8 == 0 or layer_idx == n_layers - 1:
            print(f"  Orthogonalized layers 1-{layer_idx + 1}/{n_layers}")

    print("Weight orthogonalization complete.")


def main():
    parser = argparse.ArgumentParser(description="Apply weight orthogonalization and save model")
    parser.add_argument("--model_path", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model path or local path")
    parser.add_argument("--direction_path", type=str, 
                        default="pipeline/runs/Meta-Llama-3.1-8B-Instruct/direction.pt",
                        help="Path to the saved refusal direction tensor")
    parser.add_argument("--output_path", type=str, default="ortho_model",
                        help="Directory to save the orthogonalized model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"],
                        help="Model dtype for loading")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Load direction
    print(f"Loading refusal direction from {args.direction_path}...")
    direction = torch.load(args.direction_path, map_location="cpu", weights_only=True)
    print(f"  Direction shape: {direction.shape}, norm: {direction.norm():.4f}")

    # Load metadata if available
    metadata_path = os.path.join(os.path.dirname(args.direction_path), "direction_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"  Direction metadata: pos={metadata['pos']}, layer={metadata['layer']}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cpu",  # Load on CPU for orthogonalization to save GPU memory
        trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)
    print(f"  Model loaded: {model.config.num_hidden_layers} layers, d_model={model.config.hidden_size}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Apply orthogonalization
    print("\nApplying weight orthogonalization...")
    direction = direction.to(dtype=torch.float32)
    orthogonalize_weights(model, direction)

    # Save model and tokenizer
    print(f"\nSaving orthogonalized model to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Save metadata about the orthogonalization
    ortho_metadata = {
        "source_model": args.model_path,
        "direction_path": args.direction_path,
        "method": "weight_orthogonalization",
        "paper": "Refusal in Language Models Is Mediated by a Single Direction (Arditi et al., 2024)",
        "direction_norm": direction.norm().item(),
    }
    if os.path.exists(metadata_path):
        ortho_metadata["direction_pos"] = metadata["pos"]
        ortho_metadata["direction_layer"] = metadata["layer"]

    with open(os.path.join(args.output_path, "ortho_metadata.json"), "w") as f:
        json.dump(ortho_metadata, f, indent=2)

    print(f"\nDone! Orthogonalized model saved to: {args.output_path}")
    print(f"  Total files: {len(os.listdir(args.output_path))}")
    total_size = sum(os.path.getsize(os.path.join(args.output_path, f)) 
                     for f in os.listdir(args.output_path) 
                     if os.path.isfile(os.path.join(args.output_path, f)))
    print(f"  Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
