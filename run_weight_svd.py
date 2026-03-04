"""
Task C: Extract alignment subspace basis vectors from weight delta
(Instruct - Base) using SVD.

This is a simplified version of post_processing_subspace.py that only
extracts and saves the basis vectors, without producing a projected model.
"""
import gc, os, torch
from pathlib import Path
from collections import OrderedDict
from transformers import AutoModelForCausalLM
from tqdm.auto import tqdm

BASE_PATH = "unsloth/Meta-Llama-3.1-8B"
INSTRUCT_PATH = "unsloth/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "results/weight_svd_basis"
FRAC = 0.1
DEVICE = "cuda"


def load_state_dict(model_path: str):
    print(f"  Loading {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return sd


def left_singular_basis(mat: torch.Tensor, frac: float, device: torch.device):
    out_dim, in_dim = mat.shape
    k = max(1, int(round(frac * min(out_dim, in_dim))))
    mat_gpu = mat.to(device)
    try:
        U, S, _ = torch.linalg.svd_lowrank(mat_gpu, q=k, niter=4)
    except Exception:
        U, S, _ = torch.linalg.svd(mat_gpu, full_matrices=False)
        U = U[:, :k]
        S = S[:k]
    result_U = U.cpu()
    result_S = S.cpu()
    del mat_gpu, U, S
    torch.cuda.empty_cache()
    return result_U, result_S


def main():
    device = torch.device(DEVICE)

    print("Loading state dicts ...")
    base_sd = load_state_dict(BASE_PATH)
    instruct_sd = load_state_dict(INSTRUCT_PATH)

    basis_dict = {}
    singular_values = {}
    delta_norms = {}

    for name in tqdm(base_sd.keys(), desc="Computing weight SVD"):
        w_base = base_sd[name]
        w_inst = instruct_sd[name]

        if not torch.is_floating_point(w_base):
            continue
        if w_base.ndim != 2:
            continue

        delta = (w_inst.float() - w_base.float())
        delta_norms[name] = torch.norm(delta).item()

        if delta.shape[0] < 2 or delta.shape[1] < 2:
            continue

        U, S = left_singular_basis(delta, FRAC, device)
        basis_dict[name] = U
        singular_values[name] = S

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    basis_path = os.path.join(OUTPUT_DIR, "alignment_basis.pt")
    torch.save(basis_dict, basis_path)
    print(f"Saved {len(basis_dict)} alignment basis vectors to {basis_path}")

    sv_path = os.path.join(OUTPUT_DIR, "singular_values.pt")
    torch.save(singular_values, sv_path)
    print(f"Saved singular values to {sv_path}")

    norms_path = os.path.join(OUTPUT_DIR, "delta_norms.pt")
    torch.save(delta_norms, norms_path)
    print(f"Saved delta norms to {norms_path}")

    print("\nTop 10 layers by delta norm:")
    sorted_norms = sorted(delta_norms.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, norm in sorted_norms:
        print(f"  {name}: {norm:.4f}")

    del base_sd, instruct_sd, basis_dict, singular_values
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
