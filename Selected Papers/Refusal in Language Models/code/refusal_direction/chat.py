"""
Interactive chat with the refusal-direction orthogonalized model.

Modes:
  --mode original    : Chat with the original (safety-aligned) model
  --mode ortho       : Chat with the weight-orthogonalized model (refusal removed)
  --mode compare     : Chat with both models side-by-side

Usage:
    python chat.py --mode ortho --ortho_path ortho_model
    python chat.py --mode compare --ortho_path ortho_model
    python chat.py --mode original
"""

import torch
import argparse
import sys
import os
import json

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, device="cuda", load_in_8bit=False):
    """Load a model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    load_kwargs = dict(
        trust_remote_code=True,
    )

    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, system_prompt=None, max_new_tokens=512, temperature=0.0):
    """Generate a response using the chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        if temperature == 0.0:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Decode only the new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def chat_loop_single(model, tokenizer, model_label, system_prompt=None, max_new_tokens=512, temperature=0.0):
    """Single model chat loop."""
    print(f"\n{'='*60}")
    print(f"  Chatting with: {model_label}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    if system_prompt:
        print(f"  System prompt: {system_prompt[:80]}...")
    print(f"{'='*60}")
    print("Type 'quit' or 'exit' to stop. Type 'clear' to reset.\n")

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            os.system("clear")
            continue

        response = generate_response(
            model, tokenizer, user_input,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(f"\n\033[92m{model_label}:\033[0m {response}\n")


def chat_loop_compare(model_orig, tokenizer_orig, model_ortho, tokenizer_ortho, 
                       system_prompt=None, max_new_tokens=512, temperature=0.0):
    """Side-by-side comparison chat loop."""
    print(f"\n{'='*60}")
    print(f"  Side-by-side comparison mode")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"{'='*60}")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print(f"\n\033[93m--- Original (aligned) ---\033[0m")
        resp_orig = generate_response(
            model_orig, tokenizer_orig, user_input,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(resp_orig)

        print(f"\n\033[91m--- Orthogonalized (refusal removed) ---\033[0m")
        resp_ortho = generate_response(
            model_ortho, tokenizer_ortho, user_input,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(resp_ortho)
        print()


def main():
    parser = argparse.ArgumentParser(description="Chat with original or orthogonalized model")
    parser.add_argument("--mode", type=str, default="ortho",
                        choices=["original", "ortho", "compare"],
                        help="Chat mode")
    parser.add_argument("--model_path", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct",
                        help="Path to the original model")
    parser.add_argument("--ortho_path", type=str, default="ortho_model",
                        help="Path to the orthogonalized model")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="System prompt to use")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load models in 8-bit quantization")
    args = parser.parse_args()

    if args.mode == "original":
        model, tokenizer = load_model(args.model_path, load_in_8bit=args.load_in_8bit)
        chat_loop_single(model, tokenizer, "Original Model",
                         system_prompt=args.system_prompt,
                         max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature)

    elif args.mode == "ortho":
        if not os.path.exists(args.ortho_path):
            print(f"Error: Orthogonalized model not found at '{args.ortho_path}'")
            print("Run orthogonalize_and_save.py first to create it.")
            sys.exit(1)
        model, tokenizer = load_model(args.ortho_path, load_in_8bit=args.load_in_8bit)

        # Show ortho metadata if available
        meta_path = os.path.join(args.ortho_path, "ortho_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  Method: {meta.get('method', 'N/A')}")
            print(f"  Source: {meta.get('source_model', 'N/A')}")
            print(f"  Direction: layer={meta.get('direction_layer', '?')}, pos={meta.get('direction_pos', '?')}")

        chat_loop_single(model, tokenizer, "Orthogonalized Model (refusal removed)",
                         system_prompt=args.system_prompt,
                         max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature)

    elif args.mode == "compare":
        if not os.path.exists(args.ortho_path):
            print(f"Error: Orthogonalized model not found at '{args.ortho_path}'")
            print("Run orthogonalize_and_save.py first to create it.")
            sys.exit(1)

        # For compare mode on a single GPU, we need 8-bit quantization for both
        print("Note: Loading two models. Using 8-bit quantization to fit on single GPU.\n")
        model_orig, tok_orig = load_model(args.model_path, load_in_8bit=True)
        model_ortho, tok_ortho = load_model(args.ortho_path, load_in_8bit=True)

        chat_loop_compare(model_orig, tok_orig, model_ortho, tok_ortho,
                          system_prompt=args.system_prompt,
                          max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature)


if __name__ == "__main__":
    main()
