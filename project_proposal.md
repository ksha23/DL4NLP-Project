# Project Proposal: The Entanglement Gap—Reconciling Safety in Training Dynamics vs. Inference Mechanics

## 1. Problem + Motivation
A critical contradiction has emerged in LLM safety research regarding the **separability** of safety mechanisms:
*   **The "Entangled" View:** The recent paper *Safety Subspaces are Not Linearly Distinct* (Ponkshe et al., 2026) argues that safety cannot be separated from utility. They show that removing the "Safety Update" (weights learned during alignment) damages general capabilities, implying safety is deeply fused with intelligence.
*   **The "Modular" View:** Conversely, *Refusal in Language Models* (Arditi et al., 2024) and *Assessing the Brittleness...* (ActSVD, Wei et al., 2024) demonstrate that precise interventions on **activations** can disable refusal while preserving utility.

**Our Hypothesis:** We propose that **Safety is historically entangled (during training)** but **mechanistically distinct (during inference).** If true, Ponkshe et al. are measuring the *learning process*, while Arditi/Wei are measuring the *final mechanism*.

## 2. Methods Overview
We will compare three methods on the same model to isolate where the entanglement occurs.

*   **Method A: Weight Update SVD (Measuring "Training Entanglement")**
    *   *Source:* Ponkshe et al. (2026); Code: `Safety Subspaces/`.
    *   *Technique:* We compute the SVD of the alignment weight update ($\Delta W = W_{aligned} - W_{base}$) and project out its principal components.
    *   *Hypothesis:* This will degrade utility because training updates mix safety and general capability learning.

*   **Method B: ActSVD (Measuring "Mechanism Disentanglement")**
    *   *Source:* Wei et al. (2024); Code: `Assessing the Brittlness/`.
    *   *Technique:* We perform SVD on **activations** from safety prompts to identify specific "Safety Ranks" in the weights.
    *   *Hypothesis:* This method uses runtime evidence to find modular safety components that Method A misses.

*   **Method C: Difference-of-Means (The "Refusal Direction")**
    *   *Source:* Arditi et al. (2024); Code: `Refusal in Language Models/`.
    *   *Technique:* We compute the steering vector $\vec{v}_{refusal} = \mu_{harmful} - \mu_{harmless}$ and ablate it during inference.

## 3. Experiments + Analysis
We will use **Llama-3.1-8B-Instruct** and the **JailbreakBench** evaluation suite. We choose the *Instruct* model because base models lack the specific safety alignment (refusal behavior) we aim to analyze.

*   **The "Entanglement Gap" Experiment:**
    1.  **Run All 3 Interventions:** Apply each method to remove "Safety."
    2.  **Measure Safety Loss:** Drop in refusal rate on `AdvBench`.
    3.  **Measure Utility Retention:** Zero-shot accuracy on `GSM8K` (Math) and `MMLU` (Knowledge).

*   **Expected Result:**
    *   **Method A (Weights)** $\to$ **Utility Drops** (Confirming Ponkshe et al.).
    *   **Method B (ActSVD) & C (Activations)** $\to$ **Utility Stays High** (Confirming Arditi/Wei).
    *   **Conclusion:** This proves that while you cannot "unlearn" safety (training is entangled), you *can* surgically disable it (inference is modular).

## 4. Timeline + Next Steps
*   **Completed:** Codebases for ActSVD (`main_low_rank.py`), Diff-Means (`generate_directions.py`), and Weight-SVD (`update_spaces`) are acquired and reviewed.
*   **Next Steps:**
    *   Run Method A (Weight Update SVD) to replicate the negative result from Ponkshe et al.
    *   Run the full comparative benchmark on GSM8K/AdvBench.
    *   (Stretch Goal) Implement **Refusal Direction Optimization (RDO)** from *The Geometry of Refusal* (code: `rdo.py`) to visualize the specific "refusal cone" and explain *why* Method C works but might be less precise than Method B.
