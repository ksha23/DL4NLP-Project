"""
Orthogonal-projection ActSVD pipeline for Llama-3.1-8B-Instruct.

Adapted from main_low_rank_diff.py to:
  1. Support Llama-3.1-8B-Instruct (model paths, chat templates)
  2. Use subprocess isolation for vLLM attack eval (RTX 3090 VRAM constraint)
  3. Use cuDevicePrimaryCtxReset to fully free GPU memory before subprocess

The orthogonal projection method removes safety-critical ranks that are
ORTHOGONAL to utility ranks:
    ΔW(r_u, r_s) = (I - Π_u) · Π_s · W
This preserves utility while breaking safety alignment.

Arguments:
    --rank_pos  : number of LOW ranks to REMOVE from the UTILITY subspace (keep top (R - rank_pos) utility ranks)
    --rank_neg  : number of LOW ranks to REMOVE from the SAFETY subspace (keep top (R - rank_neg) safety ranks)
    
    In the paper's notation:
      - prune_data_pos = utility dataset (alpaca_cleaned_no_safety)
      - prune_data_neg = safety dataset (align_short)
      - Π_u projects onto the top (R - rank_pos) utility ranks
      - Π_s projects onto the top (R - rank_neg) safety ranks
      - ΔW = (I - Π_u) · Π_s · W  removes safety ranks orthogonal to utility

    Paper's optimal for Llama-2-7B-chat: rank_pos ∈ [50..4000], rank_neg ∈ [50..4000]
    with best results around rank_pos ~ 3996 (keep top 100 utility), rank_neg ~ 3996 (keep top 100 safety)
    i.e., the paper's (r_u, r_s) correspond to keeping ~100 ranks of each.
"""

import argparse
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from importlib.metadata import version
from lib.data import get_loaders
from lib.eval import eval_ppl
from lib.model_wrapper_low import (
    ActLinear, make_Act, revert_Act_to_Linear, clear_act_buffer, 
    no_act_recording, set_mask
)
from functools import reduce
import pickle

print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())

SAVE_PATH = "temp"

modeltype2path = {
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "llama3.1-8b-base": "unsloth/Meta-Llama-3.1-8B",
}


def is_llama3(model_name):
    return "llama3" in model_name


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        modeltype2path[model_name],
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = min(model.config.max_position_embeddings, 4096)
    return model


def make_low_rank_diff(
    args,
    model,
    tokenizer,
    device=torch.device("cuda:0"),
    prune_data_pos="alpaca_cleaned_no_safety",
    prune_data_neg="align_short",
    model_family="llama3",
):
    """
    Orthogonal projection rank removal.
    
    prune_data_pos: UTILITY dataset - retain most useful (total_rank - rank_pos) ranks
    prune_data_neg: SAFETY dataset - identify most safety-relevant (total_rank - rank_neg) ranks

    Modified weight: W_new = W - (I - Π_u) · Π_s · W
    This removes safety-critical ranks that are orthogonal to utility ranks.
    
    rank(ΔW) ≤ min(rank_pos, total_rank - rank_neg)
    """
    model = make_Act(model, verbose=False)
    model.requires_grad_(False)
    clear_act_buffer(model)

    # globally disable recording.
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            module.record_activation = False

    # load datasets
    print(f"loading calibration data: utility={prune_data_pos}, safety={prune_data_neg}")
    dataloader_pos, _ = get_loaders(
        prune_data_pos,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
        model_family=model_family,
    )
    dataloader_neg, _ = get_loaders(
        prune_data_neg,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
        model_family=model_family,
    )
    print("dataset loading complete")

    num_hidden_layers = model.config.num_hidden_layers

    for layer in range(num_hidden_layers):
        layer_filter_fn = lambda x, l=layer: f"layers.{l}." in x

        # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.record_activation = True

        activation_norms_pos = {}
        activation_norms_neg = {}

        # Forward pass with UTILITY data (positive)
        with torch.no_grad():
            for batch in dataloader_pos:
                inp, tar = batch[0].to(device), batch[1].to(device)
                assert args.disentangle, "should run in disentangle mode"
                mask = tar.ne(-100)
                with set_mask(model, mask):
                    model(inp)

        # save utility activations & clear
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                activation_norms_pos[name] = module.activation_norms
                module.activation_norms = []

        # Forward pass with SAFETY data (negative)
        with torch.no_grad():
            for batch in dataloader_neg:
                inp, tar = batch[0].to(device), batch[1].to(device)
                assert args.disentangle, "should run in disentangle mode"
                mask = tar.ne(-100)
                with set_mask(model, mask):
                    model(inp)

        # save safety activations & clear
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                activation_norms_neg[name] = module.activation_norms
                module.activation_norms = []

        # make low_rank with orthogonal projection
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                d_out, d_in = module.base.weight.data.shape
                total_rank = min(d_out, d_in)

                # Clamp rank params: can't remove more low ranks than exist
                # rank_pos/neg are #low ranks to remove; we keep (total_rank - rank_pos/neg) top ranks
                # Scale proportionally for layers with different dimensions (e.g. GQA k_proj/v_proj)
                # The paper's default is for 4096-rank layers; scale for smaller layers
                ref_rank = 4096  # reference total rank (Llama's hidden dim)
                scale = total_rank / ref_rank
                eff_rank_pos = min(int(args.rank_pos * scale), total_rank - 1)
                eff_rank_neg = min(int(args.rank_neg * scale), total_rank - 1)
                q_pos = total_rank - eff_rank_pos  # #top utility ranks to keep
                q_neg = total_rank - eff_rank_neg  # #top safety ranks to keep

                print(f"[Layer {layer}] orthogonal projection: {name} "
                      f"(total_rank={total_rank}, keep_util={q_pos}, keep_safety={q_neg})")

                # --- Utility subspace (positive) ---
                activation_norms_p = torch.cat(activation_norms_pos[name], dim=0).to(device)
                score_p = activation_norms_p @ module.base.weight.data.T
                _, _, Vp = torch.svd_lowrank(
                    score_p.float(), q=q_pos, niter=args.niter
                )
                Vp_proj = (Vp @ Vp.T).type(module.base.weight.data.dtype)
                del activation_norms_p, score_p

                # --- Safety subspace (negative) ---
                activation_norms_n = torch.cat(activation_norms_neg[name], dim=0).to(device)
                score_n = activation_norms_n @ module.base.weight.data.T
                _, _, Vn = torch.svd_lowrank(
                    score_n.float(), q=q_neg, niter=args.niter
                )
                Vn_proj = (Vn @ Vn.T).type(module.base.weight.data.dtype)
                del activation_norms_n, score_n

                # --- Orthogonal projection: ΔW = (I - Π_u) · Π_s · W ---
                Vp_proj_ortho = (torch.eye(d_out, device=device) - Vp_proj).type(
                    module.base.weight.data.dtype
                )
                module.base.weight.data.sub_(
                    Vp_proj_ortho @ (Vn_proj @ module.base.weight.data)
                )
                
                rank_delta = min(eff_rank_pos, q_neg)
                print(f"  rank(ΔW) ≤ {rank_delta}")
                
                del Vp_proj, Vn_proj, Vp_proj_ortho, Vp, Vn
                torch.cuda.empty_cache()

        # disable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.record_activation = False
                module.clear_act_buffer()

        # clear references
        activation_norms_pos.clear()
        activation_norms_neg.clear()

        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"[Layer {layer}/{num_hidden_layers}] GPU mem: {mem_gb:.2f} GiB")

    model = revert_Act_to_Linear(model)
    model.zero_grad()


def main():
    parser = argparse.ArgumentParser(description="Orthogonal-projection ActSVD for Llama-3.1")
    parser.add_argument("--model", type=str, default="llama3.1-8b-instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--save", type=str, default="results/llama3_ortho_proj")

    # Rank parameters
    parser.add_argument("--rank_pos", type=int, default=100,
                        help="Number of LOW utility ranks to remove (keep top R-rank_pos utility ranks)")
    parser.add_argument("--rank_neg", type=int, default=100,
                        help="Number of LOW safety ranks to remove (keep top R-rank_neg safety ranks)")
    parser.add_argument("--niter", type=int, default=20, help="SVD iterations")

    # Data
    parser.add_argument("--prune_data_pos", type=str, default="alpaca_cleaned_no_safety",
                        help="Utility dataset (positive)")
    parser.add_argument("--prune_data_neg", type=str, default="align_short",
                        help="Safety dataset (negative)")

    # Flags
    parser.add_argument("--eval_attack", action="store_true")
    parser.add_argument("--save_attack_res", action="store_true")
    parser.add_argument("--entangle_prompt_feat", dest="disentangle", action="store_false")

    args = parser.parse_args()

    # Force disentangle mode
    if not hasattr(args, 'disentangle') or args.disentangle is None:
        args.disentangle = True
    print(f"Disentangle: {args.disentangle}")

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_family = "llama3" if is_llama3(args.model) else "llama2"
    print(f"Model: {args.model}, family: {model_family}")
    print(f"Rank pos (utility): {args.rank_pos}, Rank neg (safety): {args.rank_neg}")
    print(f"Utility data: {args.prune_data_pos}, Safety data: {args.prune_data_neg}")

    # Load model
    print(f"Loading model {args.model}...")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        modeltype2path[args.model], use_fast=True, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        if is_llama3(args.model):
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda:0")

    # Run orthogonal projection
    print("=" * 60)
    print("Running orthogonal-projection ActSVD...")
    print("=" * 60)
    make_low_rank_diff(
        args, model, tokenizer, device,
        prune_data_pos=args.prune_data_pos,
        prune_data_neg=args.prune_data_neg,
        model_family=model_family,
    )

    # Evaluate perplexity
    print("=" * 60)
    print("Evaluating perplexity...")
    print("=" * 60)
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity: {ppl_test}")

    # Save results
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, "log_ortho_proj.txt")

    rank_label = f"{args.rank_pos}_{args.rank_neg}"
    if not os.path.exists(save_filepath):
        with open(save_filepath, "w") as f:
            print("method\trank_pos_neg\tmetric\tscore", file=f, flush=True)
            print(f"ortho_proj\t{rank_label}\tPPL\t{ppl_test:.4f}", file=f, flush=True)
    else:
        with open(save_filepath, "a") as f:
            print(f"ortho_proj\t{rank_label}\tPPL\t{ppl_test:.4f}", file=f, flush=True)

    # Attack evaluation via subprocess
    if args.eval_attack:
        save_attackpath = os.path.join(args.save, f"attack_{rank_label}")
        if not os.path.exists(save_attackpath):
            os.makedirs(save_attackpath)

        # Save modified model for vLLM
        pruned_path = os.path.join(SAVE_PATH, "tmp_vllm_model_ortho")
        print(f"Saving modified model to {pruned_path}...")
        model.save_pretrained(pruned_path)
        tokenizer.save_pretrained(pruned_path)

        # ---- Aggressively free ALL GPU memory ----
        import gc

        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except Exception:
            pass

        for param in model.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad = param.grad.cpu()
        for buf in model.buffers():
            buf.data = buf.data.cpu()

        del model, tokenizer
        if isinstance(ppl_test, torch.Tensor):
            ppl_test = ppl_test.item()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        gc.collect()

        free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
        print(f"[Parent] GPU after cleanup: free={free_mem:.2f} GiB")

        # Reset CUDA context
        import ctypes
        try:
            libcuda = ctypes.CDLL("libcuda.so.1")
            result = libcuda.cuDevicePrimaryCtxReset(ctypes.c_int(0))
            print(f"[Parent] cuDevicePrimaryCtxReset: {result} (0=success)")
        except Exception as e:
            print(f"[Parent] Warning: Could not reset CUDA context: {e}")

        # Launch attack eval in subprocess
        import subprocess, json
        attack_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_run_attack_eval.py"
        )
        attack_args = json.dumps({
            "pruned_path": os.path.abspath(pruned_path),
            "save_attackpath": os.path.abspath(save_attackpath),
            "model_family": model_family,
            "save_attack_res": args.save_attack_res,
            "save_filepath": os.path.abspath(save_filepath),
            "prune_method": "ortho_proj",
            "rank": rank_label,
        })
        print(f"[Parent] Launching attack eval subprocess...")
        ret = subprocess.run(
            [sys.executable, attack_script, attack_args],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if ret.returncode != 0:
            print(f"ERROR: attack eval subprocess exited with code {ret.returncode}")
        else:
            print("Attack evaluation completed successfully.")

    print("=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
