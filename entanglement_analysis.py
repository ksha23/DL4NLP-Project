"""
Entanglement Gap Analysis
=========================
Unified script that:
  1. Compares ActSVD vectors (Task A) with Refusal Direction (Task B)
  2. Compares Weight SVD basis (Task C) with Activation SVD vectors (Task A)
  3. Runs weight-based intervention and measures safety vs utility impact
  4. Produces a final comparison table

Usage:
    python entanglement_analysis.py \
        --actsvd_dir "Selected Papers/Assessing the Brittlness/code/alignment-attribution-code/results/llama3_actsvd" \
        --refusal_dir "Selected Papers/Refusal in Language Models/code/refusal_direction/pipeline/runs/Meta-Llama-3.1-8B-Instruct" \
        --weight_basis "Selected Papers/Safety Subspaces/code/safety-subspaces/results/llama3_basis/alignment_basis.pt" \
        --output_dir results/entanglement_gap
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np


def load_actsvd_vectors(actsvd_dir, prune_data="align_short"):
    """Load ActSVD V matrices dumped by main_low_rank.py --dump_U."""
    proj_dir = os.path.join(actsvd_dir, prune_data, "proj_mat")
    if not os.path.exists(proj_dir):
        print(f"[WARN] ActSVD projection directory not found: {proj_dir}")
        return {}

    vectors = {}
    for rank_dir in sorted(os.listdir(proj_dir)):
        rank_path = os.path.join(proj_dir, rank_dir)
        if not os.path.isdir(rank_path):
            continue
        for fname in sorted(os.listdir(rank_path)):
            if fname.startswith("V_") and fname.endswith(".pkl"):
                layer_name = fname[2:].replace(f"_{prune_data}.pkl", "")
                with open(os.path.join(rank_path, fname), "rb") as f:
                    V = pickle.load(f)
                vectors[layer_name] = V
    print(f"Loaded {len(vectors)} ActSVD V matrices")
    return vectors


def load_refusal_direction(refusal_dir):
    """Load the refusal direction vector and metadata from Task B."""
    direction_path = os.path.join(refusal_dir, "direction.pt")
    metadata_path = os.path.join(refusal_dir, "direction_metadata.json")

    if not os.path.exists(direction_path):
        print(f"[WARN] Refusal direction not found: {direction_path}")
        return None, None

    direction = torch.load(direction_path, map_location="cpu", weights_only=True)
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    print(f"Loaded refusal direction: shape={direction.shape}, metadata={metadata}")
    return direction, metadata


def load_weight_basis(weight_basis_path):
    """Load alignment basis vectors from Task C (weight SVD)."""
    if not os.path.exists(weight_basis_path):
        print(f"[WARN] Weight basis not found: {weight_basis_path}")
        return {}

    basis = torch.load(weight_basis_path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(basis)} weight SVD basis matrices")
    return basis


def cosine_similarity_matrix(A, B):
    """Compute pairwise cosine similarity between columns of A and B."""
    A_norm = F.normalize(A, dim=0)
    B_norm = F.normalize(B, dim=0)
    return (A_norm.T @ B_norm).abs()


def compare_actsvd_vs_refusal(actsvd_vectors, refusal_direction, refusal_metadata):
    """
    Compare ActSVD singular vectors (per-layer) with the refusal direction.
    The refusal direction lives in residual-stream space (d_model).
    ActSVD V vectors are in the output space of each linear layer (d_out).
    We compare at the layer indicated by refusal_metadata.
    """
    if refusal_direction is None or not actsvd_vectors:
        print("[SKIP] Cannot compare ActSVD vs Refusal: missing data")
        return {}

    results = {}
    refusal_layer = refusal_metadata.get("layer", -1) if refusal_metadata else -1
    d_model = refusal_direction.shape[-1]

    for layer_name, V in actsvd_vectors.items():
        if V.shape[0] != d_model:
            continue

        V_float = V.cpu().float()
        ref_float = refusal_direction.cpu().float()
        if ref_float.ndim == 1:
            ref_float = ref_float.unsqueeze(1)

        cos_sims = cosine_similarity_matrix(V_float, ref_float)
        max_cos = cos_sims.max().item()
        results[layer_name] = {
            "max_cosine_sim": max_cos,
            "mean_cosine_sim": cos_sims.mean().item(),
        }

    if results:
        best_layer = max(results, key=lambda k: results[k]["max_cosine_sim"])
        print(f"\n=== ActSVD vs Refusal Direction ===")
        print(f"Best matching layer: {best_layer}")
        print(f"  Max cosine similarity: {results[best_layer]['max_cosine_sim']:.4f}")
        print(f"  Refusal direction layer: {refusal_layer}")

        top_layers = sorted(results.items(), key=lambda x: x[1]["max_cosine_sim"], reverse=True)[:10]
        print(f"\nTop 10 layers by max cosine similarity with refusal direction:")
        for name, vals in top_layers:
            print(f"  {name:<50} max_cos={vals['max_cosine_sim']:.4f}  mean_cos={vals['mean_cosine_sim']:.4f}")

    return results


def compare_weight_vs_activation_svd(weight_basis, actsvd_vectors):
    """
    Compare weight-delta SVD basis (Task C) with activation SVD vectors (Task A).
    Both are per-layer, so we match by layer name.
    """
    if not weight_basis or not actsvd_vectors:
        print("[SKIP] Cannot compare Weight SVD vs ActSVD: missing data")
        return {}

    results = {}
    matched = 0

    for name in weight_basis:
        act_name = name.replace(".weight", "")
        if act_name in actsvd_vectors:
            W_basis = weight_basis[name].cpu().float()
            A_basis = actsvd_vectors[act_name].cpu().float()

            if W_basis.shape[0] != A_basis.shape[0]:
                continue

            cos_matrix = cosine_similarity_matrix(W_basis, A_basis)
            results[name] = {
                "max_cosine_sim": cos_matrix.max().item(),
                "mean_cosine_sim": cos_matrix.mean().item(),
                "frobenius_overlap": cos_matrix.norm().item() / min(W_basis.shape[1], A_basis.shape[1]),
            }
            matched += 1

    if results:
        print(f"\n=== Weight SVD vs Activation SVD ===")
        print(f"Matched {matched} layers")

        overall_max = np.mean([v["max_cosine_sim"] for v in results.values()])
        overall_frob = np.mean([v["frobenius_overlap"] for v in results.values()])
        print(f"Average max cosine similarity: {overall_max:.4f}")
        print(f"Average Frobenius overlap: {overall_frob:.4f}")

        top_layers = sorted(results.items(), key=lambda x: x[1]["max_cosine_sim"], reverse=True)[:10]
        print(f"\nTop 10 layers by max cosine similarity (Weight vs Activation SVD):")
        for name, vals in top_layers:
            print(f"  {name:<50} max_cos={vals['max_cosine_sim']:.4f}  frob={vals['frobenius_overlap']:.4f}")

    return results


def load_intervention_results(actsvd_dir):
    """Load ASR and PPL results from Task A runs."""
    log_path = os.path.join(actsvd_dir, "log_low_rank.txt")
    if not os.path.exists(log_path):
        print(f"[WARN] ActSVD log not found: {log_path}")
        return {}

    results = {}
    with open(log_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                method, rank, metric, score = parts
                if metric == "PPL" or "ASR" in metric:
                    results[metric] = float(score)
    print(f"Loaded intervention results: {results}")
    return results


def generate_report(
    actsvd_vs_refusal,
    weight_vs_actsvd,
    intervention_results,
    output_dir,
):
    """Generate the final entanglement gap report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 70)
    report.append("ENTANGLEMENT GAP ANALYSIS REPORT")
    report.append("Llama-3.1-8B-Instruct")
    report.append("=" * 70)

    report.append("\n1. CROSS-METHOD ALIGNMENT")
    report.append("-" * 40)

    if actsvd_vs_refusal:
        avg_max = np.mean([v["max_cosine_sim"] for v in actsvd_vs_refusal.values()])
        report.append(f"ActSVD vs Refusal Direction:")
        report.append(f"  Average max cosine similarity across layers: {avg_max:.4f}")
        report.append(f"  Interpretation: {'High' if avg_max > 0.3 else 'Low'} agreement between methods")

    if weight_vs_actsvd:
        avg_max = np.mean([v["max_cosine_sim"] for v in weight_vs_actsvd.values()])
        avg_frob = np.mean([v["frobenius_overlap"] for v in weight_vs_actsvd.values()])
        report.append(f"\nWeight SVD vs Activation SVD:")
        report.append(f"  Average max cosine similarity: {avg_max:.4f}")
        report.append(f"  Average subspace overlap (Frobenius): {avg_frob:.4f}")
        report.append(f"  Interpretation: {'Entangled' if avg_frob > 0.3 else 'Modular'} safety in weights vs activations")

    report.append("\n2. INTERVENTION RESULTS")
    report.append("-" * 40)
    if intervention_results:
        if "PPL" in intervention_results:
            report.append(f"  Wikitext PPL after ActSVD top removal: {intervention_results['PPL']:.2f}")
        for key in sorted(intervention_results):
            if "ASR" in key:
                report.append(f"  {key}: {intervention_results[key]:.4f}")
    else:
        report.append("  (No intervention results available yet)")

    report.append("\n3. ENTANGLEMENT GAP VERDICT")
    report.append("-" * 40)
    report.append("Hypothesis: If Llama-3 has more entangled safety than Llama-2:")
    report.append("  - Activation interventions should STILL remove safety (modular in activations)")
    report.append("  - Weight interventions should ALSO hurt utility (entangled in weights)")
    report.append("  - Weight SVD and Activation SVD subspaces should have LOW overlap")

    if weight_vs_actsvd and intervention_results:
        avg_frob = np.mean([v["frobenius_overlap"] for v in weight_vs_actsvd.values()])
        ppl = intervention_results.get("PPL", 0)
        asr = max([v for k, v in intervention_results.items() if "ASR" in k], default=0)

        if avg_frob < 0.2 and asr > 0.5:
            verdict = "MODULAR (Wei et al. 2024 pattern): Safety removable via activations, low weight-activation overlap"
        elif avg_frob > 0.3 and ppl > 50:
            verdict = "ENTANGLED (Ponkshe et al. 2026 pattern): High weight-activation overlap, utility degraded"
        else:
            verdict = "MIXED: Partial entanglement detected"

        report.append(f"\n  VERDICT: {verdict}")
    else:
        report.append("\n  VERDICT: Insufficient data - run all experiments first")

    report_text = "\n".join(report)
    print(report_text)

    report_path = os.path.join(output_dir, "entanglement_gap_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")

    all_results = {
        "actsvd_vs_refusal": {k: v for k, v in (actsvd_vs_refusal or {}).items()},
        "weight_vs_actsvd": {k: v for k, v in (weight_vs_actsvd or {}).items()},
        "intervention_results": intervention_results or {},
    }
    json_path = os.path.join(output_dir, "entanglement_gap_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw results saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Entanglement Gap Analysis")
    parser.add_argument("--actsvd_dir", type=str, required=True,
                        help="Path to ActSVD results directory (Task A)")
    parser.add_argument("--refusal_dir", type=str, required=True,
                        help="Path to refusal direction results (Task B)")
    parser.add_argument("--weight_basis", type=str, required=True,
                        help="Path to weight SVD alignment_basis.pt (Task C)")
    parser.add_argument("--output_dir", type=str, default="results/entanglement_gap",
                        help="Output directory for analysis results")
    parser.add_argument("--prune_data", type=str, default="align_short",
                        help="Calibration dataset used in ActSVD")
    args = parser.parse_args()

    print("=" * 70)
    print("Loading experiment artifacts...")
    print("=" * 70)

    actsvd_vectors = load_actsvd_vectors(args.actsvd_dir, args.prune_data)
    refusal_direction, refusal_metadata = load_refusal_direction(args.refusal_dir)
    weight_basis = load_weight_basis(args.weight_basis)
    intervention_results = load_intervention_results(args.actsvd_dir)

    print("\n" + "=" * 70)
    print("Running cross-method comparisons...")
    print("=" * 70)

    actsvd_vs_refusal = compare_actsvd_vs_refusal(actsvd_vectors, refusal_direction, refusal_metadata)
    weight_vs_actsvd = compare_weight_vs_activation_svd(weight_basis, actsvd_vectors)

    print("\n" + "=" * 70)
    print("Generating report...")
    print("=" * 70)

    generate_report(
        actsvd_vs_refusal,
        weight_vs_actsvd,
        intervention_results,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
