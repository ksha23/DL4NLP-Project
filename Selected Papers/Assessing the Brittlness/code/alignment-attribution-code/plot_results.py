#!/usr/bin/env python
"""
Plot ActSVD Results: Baseline vs. Jailbroken (post-intervention) comparison.

Reads results from:
  - results/llama3_actsvd/log_low_rank.txt   (ActSVD top-rank removal)
  - results/llama3_baseline/log_baseline.txt  (original model, optional)

Generates:
  - results/plots/asr_comparison.png
  - results/plots/ppl_comparison.png
  - results/plots/asr_by_config.png
  - results/plots/summary_table.png

Usage:
    conda activate entangle
    python plot_results.py [--actsvd_dir results/llama3_actsvd]
                           [--baseline_dir results/llama3_baseline]
                           [--output_dir results/plots]

If baseline results don't exist yet, only the ActSVD results are plotted.
Run `python run_baseline_eval.py` first to generate baseline data.
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Color palette ────────────────────────────────────────────────────────
COLOR_BASELINE = "#2196F3"   # blue
COLOR_ACTSVD   = "#F44336"   # red
COLOR_GRID     = "#E0E0E0"
BG_COLOR       = "#FAFAFA"


def parse_log(log_path):
    """Parse a TSV log file into {metric: score} dict (takes last value per metric)."""
    results = {}
    if not os.path.exists(log_path):
        return results
    with open(log_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                method, rank, metric, score = parts
                if metric == "method":  # header row
                    continue
                try:
                    results[metric] = float(score)
                except ValueError:
                    pass
    return results


def friendly_name(metric):
    """Convert metric keys like 'ASR_basic_nosys' to readable labels."""
    name_map = {
        "ASR_basic":              "Basic\n(sys prompt)",
        "ASR_basic_nosys":        "Basic\n(no sys prompt)",
        "ASR_multiple_nosys":     "Multiple (×5)\n(no sys prompt)",
        "no_inst_ASR_basic":      "No-inst Basic\n(sys prompt)",
        "no_inst_ASR_basic_nosys": "No-inst Basic\n(no sys prompt)",
        "no_inst_ASR_multiple_nosys": "No-inst Multiple (×5)\n(no sys prompt)",
    }
    return name_map.get(metric, metric)


# The canonical order for attack configs
ASR_METRICS = [
    "ASR_basic",
    "ASR_basic_nosys",
    "ASR_multiple_nosys",
    "no_inst_ASR_basic",
    "no_inst_ASR_basic_nosys",
    "no_inst_ASR_multiple_nosys",
]


def plot_asr_comparison(baseline, actsvd, output_dir):
    """Side-by-side grouped bar chart: baseline ASR vs ActSVD ASR."""
    metrics = [m for m in ASR_METRICS if m in actsvd or m in baseline]
    if not metrics:
        print("[WARN] No ASR metrics found, skipping ASR comparison plot.")
        return

    has_baseline = any(m in baseline for m in metrics)

    labels = [friendly_name(m) for m in metrics]
    actsvd_vals = [actsvd.get(m, 0) for m in metrics]
    baseline_vals = [baseline.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35 if has_baseline else 0.5

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    if has_baseline:
        bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline (Original Model)",
                        color=COLOR_BASELINE, edgecolor="white", linewidth=0.5, alpha=0.9)
        bars2 = ax.bar(x + width / 2, actsvd_vals, width, label="After ActSVD Top-10 Removal",
                        color=COLOR_ACTSVD, edgecolor="white", linewidth=0.5, alpha=0.9)
        # Value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLOR_BASELINE)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLOR_ACTSVD)
    else:
        bars = ax.bar(x, actsvd_vals, width, label="After ActSVD Top-10 Removal",
                      color=COLOR_ACTSVD, edgecolor="white", linewidth=0.5, alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color=COLOR_ACTSVD)

    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12)
    ax.set_title("Attack Success Rate: Baseline vs. ActSVD Intervention\n"
                 "(Llama-3.1-8B-Instruct, AdvBench)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "asr_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ppl_comparison(baseline, actsvd, output_dir):
    """Bar chart comparing PPL before and after intervention."""
    ppl_actsvd = actsvd.get("PPL")
    ppl_baseline = baseline.get("PPL")

    if ppl_actsvd is None:
        print("[WARN] No PPL in ActSVD results, skipping PPL plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    labels_list = []
    values = []
    colors = []

    if ppl_baseline is not None:
        labels_list.append("Baseline\n(Original)")
        values.append(ppl_baseline)
        colors.append(COLOR_BASELINE)

    labels_list.append("After ActSVD\nTop-10 Removal")
    values.append(ppl_actsvd)
    colors.append(COLOR_ACTSVD)

    bars = ax.bar(labels_list, values, color=colors, width=0.5, edgecolor="white",
                  linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, values):
        h = bar.get_height()
        label = f"{val:,.0f}" if val > 1000 else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02, label,
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=12)
    ax.set_title("Perplexity: Impact of ActSVD Intervention\n"
                 "(Llama-3.1-8B-Instruct)", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "ppl_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_asr_heatmap(baseline, actsvd, output_dir):
    """Heatmap-style table showing all results at a glance."""
    # Build the data for the summary table
    all_metrics = ["PPL"] + ASR_METRICS
    has_baseline = bool(baseline)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    col_labels = ["Metric"] + (["Baseline", "ActSVD (rank=10)", "Δ"] if has_baseline
                                else ["ActSVD (rank=10)"])
    row_data = []
    cell_colors = []

    for metric in all_metrics:
        if metric not in actsvd and metric not in baseline:
            continue

        actsvd_val = actsvd.get(metric)
        baseline_val = baseline.get(metric) if has_baseline else None
        name = friendly_name(metric).replace("\n", " ") if metric != "PPL" else "Perplexity"

        if metric == "PPL":
            fmt_a = f"{actsvd_val:,.1f}" if actsvd_val else "—"
            fmt_b = f"{baseline_val:,.2f}" if baseline_val else "—"
            if has_baseline and actsvd_val and baseline_val:
                delta = actsvd_val - baseline_val
                fmt_d = f"+{delta:,.0f}" if delta > 0 else f"{delta:,.0f}"
                # PPL increase = bad
                d_color = "#FFCDD2" if delta > 10 else "#C8E6C9"
            else:
                fmt_d = "—"
                d_color = "white"
        else:
            fmt_a = f"{actsvd_val:.4f}" if actsvd_val is not None else "—"
            fmt_b = f"{baseline_val:.4f}" if baseline_val is not None else "—"
            if has_baseline and actsvd_val is not None and baseline_val is not None:
                delta = actsvd_val - baseline_val
                fmt_d = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                # ASR increase = safety broken
                d_color = "#FFCDD2" if delta > 0.1 else "#C8E6C9"
            else:
                fmt_d = "—"
                d_color = "white"

        if has_baseline:
            row_data.append([name, fmt_b, fmt_a, fmt_d])
            cell_colors.append(["white", "#E3F2FD", "#FFEBEE", d_color])
        else:
            row_data.append([name, fmt_a])
            cell_colors.append(["white", "#FFEBEE"])

    table = ax.table(cellText=row_data, colLabels=col_labels, cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#424242")
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("ActSVD Results Summary — Llama-3.1-8B-Instruct (Rank 10, Top Remove)\n",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_asr_by_attack_type(baseline, actsvd, output_dir):
    """Grouped chart: with-instruction vs without-instruction attack configs."""
    # Split into inst vs no_inst groups
    inst_metrics = ["ASR_basic", "ASR_basic_nosys", "ASR_multiple_nosys"]
    no_inst_metrics = ["no_inst_ASR_basic", "no_inst_ASR_basic_nosys", "no_inst_ASR_multiple_nosys"]

    has_baseline = any(m in baseline for m in inst_metrics + no_inst_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor(BG_COLOR)

    short_labels = ["Basic\n(sys prompt)", "Basic\n(no sys)", "Multiple\n(no sys)"]

    for ax_idx, (metric_group, title) in enumerate([
        (inst_metrics, "With Instruction Template"),
        (no_inst_metrics, "Without Instruction Template"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor(BG_COLOR)

        actsvd_vals = [actsvd.get(m, 0) for m in metric_group]
        baseline_vals = [baseline.get(m, 0) for m in metric_group]

        x = np.arange(len(metric_group))
        width = 0.35 if has_baseline else 0.5

        if has_baseline:
            ax.bar(x - width / 2, baseline_vals, width, label="Baseline",
                   color=COLOR_BASELINE, edgecolor="white", alpha=0.9)
            ax.bar(x + width / 2, actsvd_vals, width, label="ActSVD",
                   color=COLOR_ACTSVD, edgecolor="white", alpha=0.9)
        else:
            ax.bar(x, actsvd_vals, width, label="ActSVD",
                   color=COLOR_ACTSVD, edgecolor="white", alpha=0.9)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        if ax_idx == 0:
            ax.set_ylabel("Attack Success Rate (ASR)", fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle("ASR by Attack Configuration — Llama-3.1-8B-Instruct\n"
                 "(ActSVD Top-10 Rank Removal)", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    path = os.path.join(output_dir, "asr_by_config.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ActSVD results")
    parser.add_argument("--actsvd_dir", type=str, default="results/llama3_actsvd",
                        help="ActSVD results directory (contains log_low_rank.txt)")
    parser.add_argument("--baseline_dir", type=str, default="results/llama3_baseline",
                        help="Baseline results directory (contains log_baseline.txt)")
    parser.add_argument("--output_dir", type=str, default="results/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load ActSVD results
    actsvd_log = os.path.join(args.actsvd_dir, "log_low_rank.txt")
    actsvd = parse_log(actsvd_log)
    if not actsvd:
        print(f"ERROR: No ActSVD results found at {actsvd_log}")
        print("Run the ActSVD pipeline first (main_low_rank.py --top_remove --eval_attack)")
        sys.exit(1)
    print(f"Loaded ActSVD results: {actsvd}")

    # Load baseline results (optional)
    baseline_log = os.path.join(args.baseline_dir, "log_baseline.txt")
    baseline = parse_log(baseline_log)
    if baseline:
        print(f"Loaded baseline results: {baseline}")
    else:
        print(f"No baseline results at {baseline_log} — plotting ActSVD only.")
        print("To add baseline comparison, run: python run_baseline_eval.py")
        baseline = {}

    # Generate plots
    print("\nGenerating plots...")
    plot_asr_comparison(baseline, actsvd, args.output_dir)
    plot_ppl_comparison(baseline, actsvd, args.output_dir)
    plot_asr_heatmap(baseline, actsvd, args.output_dir)
    plot_asr_by_attack_type(baseline, actsvd, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")
    print("Files: asr_comparison.png, ppl_comparison.png, summary_table.png, asr_by_config.png")


if __name__ == "__main__":
    main()
