#!/usr/bin/env python
"""
Interactive chat with Llama-3.1-8B-Instruct — baseline or jailbroken (ActSVD).

Usage:
    # Chat with the ORIGINAL (safety-aligned) model:
    python chat.py --mode original

    # Chat with the JAILBROKEN (ActSVD top-10 removed) model:
    python chat.py --mode jailbroken

    # Chat with BOTH side-by-side (same prompt, two responses):
    python chat.py --mode compare

Options:
    --no-system     Disable the default safety system prompt
    --system "..."  Use a custom system prompt
    --max-tokens N  Max tokens to generate (default: 512)
"""

import os
import sys
import json
import argparse
import subprocess

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
JAILBROKEN_MODEL = os.path.join(SCRIPT_DIR, "temp", "tmp_vllm_model")
CACHE_DIR = os.path.join(SCRIPT_DIR, "llm_weights")

DEFAULT_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe. Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature. If a question "
    "does not make any sense, or is not factually coherent, explain why instead of "
    "answering something not correct. If you don't know the answer to a question, "
    "please don't share false information."
)

# ── ANSI colors ──────────────────────────────────────────────────────────
C_RED     = "\033[91m"
C_GREEN   = "\033[92m"
C_YELLOW  = "\033[93m"
C_CYAN    = "\033[96m"
C_BOLD    = "\033[1m"
C_RESET   = "\033[0m"


def format_prompt(user_msg, system_prompt=None):
    """Format using Llama-3 chat template."""
    parts = ["<|begin_of_text|>"]
    if system_prompt:
        parts.append(
            "<|start_header_id|>system<|end_header_id|>\n\n"
            + system_prompt
            + "<|eot_id|>"
        )
    parts.append(
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + user_msg
        + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-MODEL MODE: loads vLLM in this process, interactive loop
# ═══════════════════════════════════════════════════════════════════════

def run_single_mode(model_path, label, system_prompt, max_tokens):
    """Load one model with vLLM and run an interactive chat loop."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    color = C_GREEN if "original" in label.lower() else C_RED

    print(f"\n{'='*60}")
    print(f"Loading {label}: {model_path}")
    print(f"{'='*60}")
    model = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        swap_space=32,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        download_dir=CACHE_DIR,
    )
    print(f"✓ {label} loaded.\n")

    print(f"{color}{'='*60}{C_RESET}")
    print(f"{C_BOLD}  Chatting with: {label}{C_RESET}")
    if system_prompt:
        print(f"{color}  System prompt: ON{C_RESET}")
    else:
        print(f"{color}  System prompt: OFF{C_RESET}")
    print(f"{color}  Type 'quit' or 'exit' to stop{C_RESET}")
    print(f"{color}{'='*60}{C_RESET}\n")

    params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    while True:
        try:
            user_input = input(f"{C_BOLD}You: {C_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        prompt = format_prompt(user_input, system_prompt)
        outputs = model.generate([prompt], params)
        response = outputs[0].outputs[0].text.strip()

        print(f"\n{color}{label}:{C_RESET}")
        print(response)
        print()


# ═══════════════════════════════════════════════════════════════════════
#  COMPARE MODE: each generation runs in a SUBPROCESS
#  (vLLM holds ~15GB CUDA memory that can't be freed in-process,
#   so we must spawn a fresh process for each model swap)
# ═══════════════════════════════════════════════════════════════════════

_CHILD_CODE = r'''
import os, sys, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

cfg = json.loads(sys.argv[1])
model_path = cfg["model_path"]
prompt = cfg["prompt"]
max_tokens = cfg["max_tokens"]
cache_dir = cfg["cache_dir"]

from vllm import LLM, SamplingParams

model = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    swap_space=32,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    download_dir=cache_dir,
)

params = SamplingParams(
    temperature=0.7,
    max_tokens=max_tokens,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)

outputs = model.generate([prompt], params)
response = outputs[0].outputs[0].text.strip()

# Write result as JSON on a tagged line so parent can find it
print("\n__CHAT_RESULT__" + json.dumps({"response": response}))
'''


def _subprocess_generate(model_path, prompt, max_tokens):
    """
    Spawn a child process that loads vLLM, generates one response, exits.
    The child gets a completely clean GPU (no leftover CUDA context).
    """
    cfg = json.dumps({
        "model_path": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "cache_dir": CACHE_DIR,
    })

    try:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE, cfg],
            capture_output=True,
            text=True,
            timeout=180,
        )

        # Find the __CHAT_RESULT__ line in stdout
        for line in result.stdout.split("\n"):
            if line.startswith("__CHAT_RESULT__"):
                data = json.loads(line[len("__CHAT_RESULT__"):])
                return data["response"]

        # No result found — show what went wrong
        stderr_lines = result.stderr.strip().split("\n")
        # Filter to just the error, skip vLLM info spam
        err_lines = [l for l in stderr_lines if "Error" in l or "error" in l
                     or "Traceback" in l or l.startswith(" ")]
        if not err_lines:
            err_lines = stderr_lines[-5:]
        return f"[ERROR: no response]\n" + "\n".join(err_lines)

    except subprocess.TimeoutExpired:
        return "[ERROR: generation timed out after 180s]"
    except Exception as e:
        return f"[ERROR: {e}]"


def run_compare_mode(jailbroken_path, system_prompt, max_tokens):
    """Compare mode: each prompt → two subprocess generations."""
    print(f"\n{C_BOLD}{'='*60}{C_RESET}")
    print(f"{C_BOLD}  COMPARE MODE: Same prompt → Original & Jailbroken{C_RESET}")
    print(f"{C_BOLD}  Each model loads in a fresh subprocess (clean GPU){C_RESET}")
    if system_prompt:
        print(f"{C_CYAN}  System prompt: ON{C_RESET}")
    else:
        print(f"{C_CYAN}  System prompt: OFF{C_RESET}")
    print(f"{C_CYAN}  ~10-15s per model to load+generate. Be patient.{C_RESET}")
    print(f"{C_CYAN}  Type 'quit' or 'exit' to stop{C_RESET}")
    print(f"{C_BOLD}{'='*60}{C_RESET}\n")

    while True:
        try:
            user_input = input(f"{C_BOLD}You: {C_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        prompt = format_prompt(user_input, system_prompt)

        # ── Original model (subprocess) ──────────────────────────────
        print(f"\n{C_YELLOW}  ⏳ Loading & generating with original model...{C_RESET}")
        resp_orig = _subprocess_generate(ORIGINAL_MODEL, prompt, max_tokens)
        print(f"\n{C_GREEN}┌─ 🛡️  Original (safety-aligned):{C_RESET}")
        for line in resp_orig.split("\n"):
            print(f"{C_GREEN}│  {line}{C_RESET}")

        # ── Jailbroken model (subprocess) ────────────────────────────
        print(f"\n{C_YELLOW}  ⏳ Loading & generating with jailbroken model...{C_RESET}")
        resp_jail = _subprocess_generate(jailbroken_path, prompt, max_tokens)
        print(f"\n{C_RED}┌─ 💥 Jailbroken (ActSVD top-10 removed):{C_RESET}")
        for line in resp_jail.split("\n"):
            print(f"{C_RED}│  {line}{C_RESET}")

        print()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with original or jailbroken Llama-3.1-8B-Instruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py --mode original              # Chat with the safe model
  python chat.py --mode jailbroken             # Chat with the jailbroken model
  python chat.py --mode jailbroken --no-system # No system prompt (more vulnerable)
  python chat.py --mode compare                # Side-by-side comparison
        """,
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["original", "jailbroken", "compare"],
        help="Which model to chat with",
    )
    parser.add_argument(
        "--no-system", action="store_true",
        help="Disable the default safety system prompt",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        help="Custom system prompt (overrides default)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--jailbroken-path", type=str, default=JAILBROKEN_MODEL,
        help="Path to the jailbroken model (default: temp/tmp_vllm_model)",
    )
    args = parser.parse_args()

    # Determine system prompt
    if args.no_system:
        system_prompt = None
    elif args.system:
        system_prompt = args.system
    else:
        system_prompt = DEFAULT_SYSTEM

    # Validate jailbroken model exists
    if args.mode in ("jailbroken", "compare"):
        if not os.path.isdir(args.jailbroken_path):
            print(f"ERROR: Jailbroken model not found at: {args.jailbroken_path}")
            print("Run the ActSVD pipeline first (main_low_rank.py --top_remove)")
            print("The model is saved to temp/tmp_vllm_model/ during the pipeline run.")
            sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────
    if args.mode == "original":
        run_single_mode(ORIGINAL_MODEL, "🛡️  Original", system_prompt, args.max_tokens)

    elif args.mode == "jailbroken":
        run_single_mode(args.jailbroken_path, "💥 Jailbroken", system_prompt, args.max_tokens)

    elif args.mode == "compare":
        run_compare_mode(args.jailbroken_path, system_prompt, args.max_tokens)


if __name__ == "__main__":
    main()
