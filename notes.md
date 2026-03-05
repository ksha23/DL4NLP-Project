# Project Notes

## Comparing the Two Papers' Approaches to Removing Safety Alignment

Both papers aim to remove safety alignment (refusal behavior) from instruction-tuned LLMs, but they operate at different levels of granularity and use different techniques.

### Paper 1: "Assessing the Brittleness of Safety Alignment" (ActSVD)

**Core idea:** Safety alignment lives in a low-rank subspace of the model's weight matrices. Find that subspace via activation-aware SVD on safety-related data, then remove it.

**Method:**
1. Collect activations on safety-aligned data (refusal responses) → run SVD → get safety subspace $U_s$
2. Collect activations on general utility data (Alpaca) → run SVD → get utility subspace $U_u$
3. Orthogonal projection: remove overlap between the two subspaces so that removing safety doesn't destroy utility: $U_s' = U_s - U_u U_u^T U_s$
4. Reconstruct weight matrices without the pure-safety directions

**Intervention type:** High-rank (removes thousands of dimensions from each weight matrix)

**What "orthogonalization" means here:** Making the safety subspace orthogonal to the utility subspace, so they can be separated cleanly.

**Results (Llama-3.1-8B-Instruct):**
- Naive ActSVD (rank 10, no orthogonal projection): PPL = 1,818,062 (model destroyed), ASR mixed
- Orthogonal projection (rank_pos=4000, rank_neg=4090): PPL = 10.43, ASR ≈ 0.91–0.96

---

### Paper 2: "Refusal in Language Models Is Mediated by a Single Direction"

**Core idea:** Refusal behavior is mediated by a single direction in activation space. Find it via difference-of-means, then remove it.

**Method:**
1. Compute mean activations on harmful prompts vs. harmless prompts at each layer
2. Take the difference → candidate refusal directions (one per layer × position)
3. Select the best direction by evaluating: ablation refusal score, steering score, KL divergence
4. Remove the direction via:
   - **Directional ablation** (inference-time hook: project out the direction from every layer's activations)
   - **Activation addition** (inference-time hook: subtract the direction at the selected layer)
   - **Weight orthogonalization** (permanent: modify embed, o_proj, down_proj so they can never write along this direction)

**Intervention type:** Rank-1 (removes a single vector in $\mathbb{R}^{4096}$)

**What "orthogonalization" means here:** Making weight matrices orthogonal to the refusal direction vector, so the model physically cannot represent refusal in its residual stream.

**Results (Llama-3.1-8B-Instruct):**
- Baseline: ASR = 0.21, Pile PPL = 8.76
- Directional ablation: ASR = 1.00, Pile PPL = 8.80
- Activation addition: ASR = 1.00, Pile PPL = 10.19

---

### Side-by-Side Comparison

| Aspect | ActSVD (Paper 1) | Refusal Direction (Paper 2) |
|---|---|---|
| **Granularity** | High-rank subspace | Single direction (rank-1) |
| **Where it operates** | Weight matrices (all linear layers) | Activations (all layers) or weights (embed, o_proj, down_proj) |
| **How direction is found** | Activation-aware SVD on safety data | Difference-of-means on harmful vs. harmless |
| **Orthogonalization target** | Safety subspace vs. utility subspace | Weight matrices vs. refusal direction |
| **Utility preservation** | Explicit: project out utility before removing safety | Implicit: single direction has minimal impact on general behavior |
| **PPL after intervention** | 10.43 (ortho proj) | 8.80 (ablation) |
| **ASR after intervention** | 0.91–0.96 | 1.00 |
| **Permanent modification?** | Yes (weight reconstruction) | Both options: hooks (temporary) or weight ortho (permanent) |
| **Requires two datasets?** | Yes (safety + utility) | Yes (harmful + harmless) |

### Key Takeaways

1. **Refusal direction is simpler and more effective** — a single direction achieves ASR = 1.0 with almost no PPL degradation (8.76 → 8.80). ActSVD requires careful tuning of rank parameters and the orthogonal projection step.

2. **ActSVD is more general** — it doesn't assume safety lives in a single direction. It could capture more complex safety representations that aren't rank-1.

3. **Both confirm safety alignment is surprisingly fragile** — whether you model it as a subspace or a single direction, it can be cleanly separated from the model's general capabilities.

4. **Orthogonalization serves different purposes** — in ActSVD it's a technique for disentangling two subspaces; in the refusal direction paper it's a technique for permanently modifying weights.
