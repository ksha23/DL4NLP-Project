#!/usr/bin/env python
"""
Standalone subprocess script for vLLM attack evaluation.

Launched by main_low_rank.py so that the vLLM engine gets a completely fresh
CUDA context (no leftover PyTorch memory from the parent process).

Usage (called automatically by main_low_rank.py):
    python _run_attack_eval.py '<json_config>'

The JSON config must contain:
    pruned_path      – path to the saved HF model directory
    save_attackpath  – directory where per-attack .jsonl results are saved
    model_family     – "llama2" or "llama3"
    save_attack_res  – bool, whether to save per-prompt results
    save_filepath    – path to the summary TSV log file
    prune_method     – e.g. "low_rank"
    rank             – int, rank removed
"""

import sys
import os
import json

# Disable V1 multiprocessing so EngineCore runs in-process.
# Without this, vLLM 0.16 spawns a child process via multiprocessing.spawn
# which (a) re-executes this script causing infinite recursion, and
# (b) splits GPU memory between parent and child processes.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main():
    # ---- parse args ----
    cfg = json.loads(sys.argv[1])
    pruned_path      = cfg["pruned_path"]
    save_attackpath  = cfg["save_attackpath"]
    model_family     = cfg["model_family"]
    save_attack_res  = cfg["save_attack_res"]
    save_filepath    = cfg["save_filepath"]
    prune_method     = cfg["prune_method"]
    rank             = cfg["rank"]

    # ---- imports (heavy, after arg parse so fast-fail on bad args) ----
    from transformers import AutoTokenizer
    from vllm import LLM
    from lib.eval import eval_attack

    # ---- load tokenizer + vLLM model ----
    tokenizer = AutoTokenizer.from_pretrained(pruned_path)
    print(f"[_run_attack_eval] Loading vLLM model from {pruned_path} ...")
    vllm_model = LLM(
        model=pruned_path,
        tokenizer=pruned_path,
        dtype="bfloat16",
        swap_space=32,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    print("[_run_attack_eval] vLLM model loaded.")

    # ---- attack configs ----
    # For each include_inst setting, run three prompt variants:
    #   basic            – add_sys_prompt=True,  do_sample=False, num_sampled=1
    #   basic_no_sys     – add_sys_prompt=False, do_sample=False, num_sampled=1
    #   multiple_no_sys  – add_sys_prompt=False, do_sample=True,  num_sampled=5

    attack_configs = [
        # (label,            add_sys_prompt, do_sample, num_sampled, filename_suffix)
        ("basic",            True,           False,     1,           "basic.jsonl"),
        ("basic_no_sys",     False,          False,     1,           "basic_no_sys.jsonl"),
        ("multiple_no_sys",  False,          True,      5,           "multiple_no_sys.jsonl"),
    ]

    for include_inst in [True, False]:
        suffix = "" if include_inst else "no_inst_"
        for label, add_sys_prompt, do_sample, num_sampled, fname_suffix in attack_configs:
            fname = os.path.join(save_attackpath, f"{suffix}{fname_suffix}")
            print(f"[_run_attack_eval] Running {suffix}{label} ...")
            score = eval_attack(
                vllm_model,
                tokenizer,
                num_sampled=num_sampled,
                add_sys_prompt=add_sys_prompt,
                do_sample=do_sample,
                save_attack_res=save_attack_res,
                include_inst=include_inst,
                filename=fname,
                model_family=model_family,
            )
            # Determine the metric name (matches the original log format)
            if label == "basic":
                metric = f"{suffix}ASR_basic"
            elif label == "basic_no_sys":
                metric = f"{suffix}ASR_basic_nosys"
            elif label == "multiple_no_sys":
                metric = f"{suffix}ASR_multiple_nosys"
            else:
                metric = f"{suffix}ASR_{label}"

            print(f"[_run_attack_eval] {metric}: {score:.4f}")
            with open(save_filepath, "a") as f:
                print(f"{prune_method}\t{rank}\t{metric}\t{score:.4f}", file=f, flush=True)
            print("********************************")

    # ---- GCG attack (llama2 only) ----
    if model_family == "llama2":
        print("[_run_attack_eval] Running GCG attack ...")
        score = eval_attack(
            vllm_model,
            tokenizer,
            num_sampled=1,
            add_sys_prompt=False,
            gcg=True,
            do_sample=False,
            save_attack_res=save_attack_res,
            include_inst=True,
            filename=os.path.join(save_attackpath, "gcg.jsonl"),
            model_family=model_family,
        )
        print(f"[_run_attack_eval] ASR_gcg: {score:.4f}")
        with open(save_filepath, "a") as f:
            print(f"{prune_method}\t{rank}\tASR_gcg\t{score:.4f}", file=f, flush=True)

    # ---- cleanup ----
    del vllm_model
    print("[_run_attack_eval] Done.")


if __name__ == "__main__":
    main()
