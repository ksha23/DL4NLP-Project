#!/usr/bin/env python
"""
Run baseline (original, non-jailbroken) evaluation on the unmodified model.

This runs the same attack evaluation as the ActSVD pipeline but on the
ORIGINAL Llama-3.1-8B-Instruct model (no SVD intervention), so we can
compare baseline safety vs. post-intervention (jailbroken) safety.

Usage:
    conda activate entangle
    python run_baseline_eval.py --model llama3.1-8b-instruct \
        --save results/llama3_baseline
"""

import argparse
import os
import sys
import json
import subprocess
import gc
import ctypes

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model paths ──────────────────────────────────────────────────────────
modeltype2path = {
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
}

SAVE_PATH = "temp"


def is_llama3(model_name):
    return "llama3" in model_name


def main():
    parser = argparse.ArgumentParser(description="Baseline (no intervention) evaluation")
    parser.add_argument("--model", type=str, default="llama3.1-8b-instruct",
                        choices=list(modeltype2path.keys()))
    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    parser.add_argument("--save", type=str, default="results/llama3_baseline",
                        help="Directory to save baseline results")
    parser.add_argument("--save_attack_res", action="store_true", default=True)
    args = parser.parse_args()

    model_family = "llama3" if is_llama3(args.model) else "llama2"
    hf_path = modeltype2path[args.model]

    # ── Create save directories ──────────────────────────────────────────
    os.makedirs(args.save, exist_ok=True)
    save_attackpath = os.path.join(args.save, "attack_baseline")
    os.makedirs(save_attackpath, exist_ok=True)
    save_filepath = os.path.join(args.save, "log_baseline.txt")

    # ── Load model for PPL eval ──────────────────────────────────────────
    print(f"Loading {hf_path} for PPL evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    model.seqlen = min(model.config.max_position_embeddings, 4096)

    tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── PPL eval ─────────────────────────────────────────────────────────
    from lib.eval import eval_ppl
    device = torch.device("cuda:0")

    # eval_ppl expects args.model
    class FakeArgs:
        pass
    fake_args = FakeArgs()
    fake_args.model = args.model

    ppl_test = eval_ppl(fake_args, model, tokenizer, device)
    print(f"Baseline wikitext perplexity: {ppl_test}")

    with open(save_filepath, "w") as f:
        print("method\trank_removed\tmetric\tscore", file=f, flush=True)
        print(f"baseline\t0\tPPL\t{ppl_test:.4f}", file=f, flush=True)

    # ── Save model for vLLM ──────────────────────────────────────────────
    pruned_path = os.path.join(SAVE_PATH, "tmp_vllm_model_baseline")
    print(f"Saving model to {pruned_path} for vLLM evaluation...")
    model.save_pretrained(pruned_path)
    tokenizer.save_pretrained(pruned_path)

    # ── Aggressively free GPU memory ─────────────────────────────────────
    if isinstance(ppl_test, torch.Tensor):
        ppl_test = ppl_test.item()

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
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    gc.collect()

    # Destroy CUDA context to free ALL GPU memory
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        result = libcuda.cuDevicePrimaryCtxReset(ctypes.c_int(0))
        print(f"[Baseline] cuDevicePrimaryCtxReset result: {result} (0=success)")
    except Exception as e:
        print(f"[Baseline] Warning: Could not reset CUDA context: {e}")

    # ── Launch vLLM attack eval subprocess ───────────────────────────────
    attack_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "_run_attack_eval.py")
    attack_args = json.dumps({
        "pruned_path": os.path.abspath(pruned_path),
        "save_attackpath": os.path.abspath(save_attackpath),
        "model_family": model_family,
        "save_attack_res": args.save_attack_res,
        "save_filepath": os.path.abspath(save_filepath),
        "prune_method": "baseline",
        "rank": 0,
    })
    print("Launching vLLM attack evaluation subprocess...")
    ret = subprocess.run(
        [sys.executable, attack_script, attack_args],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if ret.returncode != 0:
        print(f"ERROR: attack eval subprocess exited with code {ret.returncode}")
    else:
        print("Baseline evaluation completed successfully!")


if __name__ == "__main__":
    main()
