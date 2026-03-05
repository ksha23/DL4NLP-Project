import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import (
    prune_wanda,
    prune_random,
    prune_magnitude,
    prune_sparsegpt,
    prune_ablate,
    check_sparsity,
    find_layers,
    prune_wanda_decouple_activations,
    get_mask,
    prune_wandg_set_difference,
)
from lib.model_wrapper import prune_wanda_v2, prune_wandg
from lib.model_wrapper_low import make_low_rank
from lib.eval import eval_ppl, eval_zero_shot

print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())

SAVE_PATH = "temp"

modeltype2path = {
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "llama2-13b-hf": "meta-llama/Llama-2-13b-hf",
    "llama3.1-8b-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "llama3.1-8b-base": "unsloth/Meta-Llama-3.1-8B",
}


def is_llama3(model_name):
    return "llama3" in model_name


def get_llm(model_name, cache_dir="llm_weights"):
    load_kwargs = dict(
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained(
        modeltype2path[model_name], **load_kwargs
    )

    model.seqlen = min(model.config.max_position_embeddings, 4096)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )
    parser.add_argument("--prune_method", type=str, choices=["low_rank"])
    parser.add_argument(
        "--prune_data",
        type=str,
        choices=[
            "wikitext",
            "alpaca",
            "alpaca_cleaned",
            "alpaca_cleaned_no_safety",
            "align",
            "align_short",
            "misalign",
            "align_misalign",
            "misalign_align",
            "align_short_misalign",
            "none",
        ],
        default="alpaca_cleaned_no_safety",
    )

    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--save", type=str, default=None, help="Path to save results.")
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save the pruned model."
    )
    parser.add_argument(
        "--top_remove", action="store_true", help="Remove the top ranks."
    )
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--eval_attack", action="store_true")
    parser.add_argument("--save_attack_res", action="store_true")
    parser.add_argument(
        "--entangle_prompt_feat",
        dest="disentangle",
        action="store_false",
        help="entangle the prompt and response when computing the wanda score",
    )
    parser.add_argument(
        "--dump_U", action="store_true", help="dump the U matrix for analysis"
    )

    # low rank
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--niter", type=int, default=20)

    args = parser.parse_args()

    print("Disentangle:", args.disentangle)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        modeltype2path[args.model], use_fast=True
    )

    if tokenizer.pad_token is None:
        if is_llama3(args.model):
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda:0")
    if (
        "30b" in args.model or "65b" in args.model
    ):  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.prune_method == "low_rank":
        model_family = "llama3" if is_llama3(args.model) else "llama2"
        make_low_rank(args, model, tokenizer, device, prune_data=args.prune_data, model_family=model_family)

    ################################################################
    print("*" * 30)

    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    if args.save_attack_res:
        save_attackpath = os.path.join(args.save, f"attack_{args.rank}")
        print(save_attackpath)
        if not os.path.exists(save_attackpath):
            os.makedirs(save_attackpath)
    else:
        save_attackpath = ""
    if not os.path.exists(save_filepath):
        with open(save_filepath, "w") as f:
            print("method\trank_removed\tmetric\tscore", file=f, flush=True)
            print(
                f"{args.prune_method}\t{args.rank}\tPPL\t{ppl_test:.4f}",
                file=f,
                flush=True,
            )
    else:
        with open(save_filepath, "a") as f:
            print(
                f"{args.prune_method}\t{args.rank}\tPPL\t{ppl_test:.4f}",
                file=f,
                flush=True,
            )

    if args.eval_attack:
        model_family = "llama3" if is_llama3(args.model) else "llama2"
        pruned_path = os.path.join(SAVE_PATH, f"tmp_vllm_model")
        model.save_pretrained(pruned_path)
        tokenizer.save_pretrained(pruned_path)
        
        # ---- Aggressively free ALL GPU memory ----
        import gc
        
        # Remove accelerate dispatch hooks if present (device_map="auto")
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except Exception:
            pass
        
        # Move ALL parameters and buffers to CPU individually
        # (model.cpu() may not work properly with accelerate's device_map)
        for param in model.parameters():
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad = param.grad.cpu()
        for buf in model.buffers():
            buf.data = buf.data.cpu()
        
        del model
        del tokenizer
        # Also convert ppl_test to plain float if it's a CUDA tensor
        if isinstance(ppl_test, torch.Tensor):
            ppl_test = ppl_test.item()
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        gc.collect()
        
        # Report GPU memory state after initial cleanup
        free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
        total_mem = torch.cuda.mem_get_info(0)[1] / 1024**3
        alloc_mem = torch.cuda.memory_allocated(0) / 1024**3
        reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
        print(f"[Parent] GPU after empty_cache: free={free_mem:.2f} GiB, total={total_mem:.2f} GiB, "
              f"allocated={alloc_mem:.2f} GiB, reserved={reserved_mem:.2f} GiB")
        
        # Destroy the CUDA primary context to release ALL GPU memory
        # (including CUDA context overhead). After this, parent must NOT use CUDA.
        import ctypes
        try:
            torch.cuda.empty_cache()
            libcuda = ctypes.CDLL("libcuda.so.1")
            result = libcuda.cuDevicePrimaryCtxReset(ctypes.c_int(0))
            print(f"[Parent] cuDevicePrimaryCtxReset result: {result} (0=success)")
        except Exception as e:
            print(f"[Parent] Warning: Could not reset CUDA context: {e}")

        # Launch vLLM attack evaluation in a fresh subprocess so that the
        # parent process's CUDA context (which holds ~15 GiB) does not
        # interfere.  The subprocess gets a clean GPU.
        import subprocess, sys, json

        attack_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "_run_attack_eval.py")
        attack_args = json.dumps({
            "pruned_path": os.path.abspath(pruned_path),
            "save_attackpath": os.path.abspath(save_attackpath),
            "model_family": model_family,
            "save_attack_res": args.save_attack_res,
            "save_filepath": os.path.abspath(save_filepath),
            "prune_method": args.prune_method,
            "rank": args.rank,
        })
        ret = subprocess.run(
            [sys.executable, attack_script, attack_args],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if ret.returncode != 0:
            print(f"ERROR: attack eval subprocess exited with code {ret.returncode}")
        else:
            print("Attack evaluation subprocess completed successfully.")

    if args.eval_zero_shot:
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True

        task_list = [
            "boolq",
            "rte",
            "hellaswag",
            "winogrande",
            "arc_challenge",
            "openbookqa",
        ]
        num_shot = 0
        results = eval_zero_shot(
            modeltype2path[args.model],
            model,
            tokenizer,
            task_list,
            num_shot,
            accelerate,
            limit=200,
        )
        print("********************************")
        print("zero_shot evaluation results")
        sum_acc = 0
        with open(save_filepath, "a") as f:
            for k, v in results["results"].items():
                print(
                    f"{args.prune_method}\t{args.rank}\t{k}\t{v['acc']:.4f}",
                    file=f,
                    flush=True,
                )
                sum_acc += v["acc"]
            print(
                f"{args.prune_method}\t{args.rank}\taveraged\t{sum_acc/len(task_list):.4f}",
                file=f,
                flush=True,
            )

        print(results)


if __name__ == "__main__":
    main()
