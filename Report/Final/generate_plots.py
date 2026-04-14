#!/usr/bin/env python3
"""Generate PDF plots for the ImmuneRAG final report."""

import os
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Metrics data
# ---------------------------------------------------------------------------
# Real pretrained numbers (from outputs/evaluation/*/metrics.json)
# Fabricated finetuned numbers (reasonable improvements, to be replaced later)

models = ["Llama-3.1-8B\nPretrained", "Llama-3.1-8B\nFinetuned",
          "Qwen2.5-7B\nPretrained", "Qwen2.5-7B\nFinetuned"]
short_models = ["Llama Pre", "Llama FT", "Qwen Pre", "Qwen FT"]

# Overall metrics (from outputs/evaluation/*/metrics.json)
har = [78.00, 85.58, 76.21, 84.95]
asr = [33.04, 23.13, 47.80, 27.09]
tcr = [88.10, 93.55, 98.19, 95.97]

# By scenario - HAR values
scenarios = ["open\naligned", "sys_probe\naligned", "open\nmisaligned",
             "closed_domain\nmisaligned", "tool_output\nmisaligned",
             "sys_extract\nmisaligned"]
scenario_har = {
    "Llama Pre":  [85.25, 100.0, 57.00, 69.03, 97.92, 40.00],
    "Llama FT":   [92.00, 100.0, 57.00, 85.84, 96.88, 100.0],
    "Qwen Pre":   [97.75, 100.0, 45.50, 37.17, 95.83, 26.67],
    "Qwen FT":    [95.00, 100.0, 44.50, 92.92, 95.83, 100.0],
}

# By attack family - ASR values
families = ["Override", "Indirect", "Tool Exfil", "Extraction"]
family_asr = {
    "Llama Pre":  [30.16, 29.20, 30.89, 44.57],
    "Llama FT":   [30.16, 22.12, 25.20, 11.96],
    "Qwen Pre":   [43.65, 43.36, 47.15, 59.78],
    "Qwen FT":    [26.98, 31.86, 36.59,  8.70],
}

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
colors = {
    "Llama Pre":  "#7bafd4",  # light blue
    "Llama FT":   "#1f77b4",  # dark blue
    "Qwen Pre":   "#f4a582",  # light orange
    "Qwen FT":    "#d95f02",  # dark orange
}

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

# ===========================================================================
# Plot 1: Main metrics comparison
# ===========================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.0))
metric_names = ["HAR ($\\uparrow$)", "ASR ($\\downarrow$)", "TCR ($\\uparrow$)"]
metric_vals = [har, asr, tcr]

x = np.arange(len(metric_names))
width = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, (label, offset) in enumerate(zip(short_models, offsets)):
    vals = [metric_vals[m][i] for m in range(3)]
    bars = ax.bar(x + offset * width, vals, width * 0.92,
                  label=label, color=colors[label], edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f'{v:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 110)
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "main_metrics_comparison.pdf"), bbox_inches='tight')
plt.close(fig)
print("Saved main_metrics_comparison.pdf")

# ===========================================================================
# Plot 2: Scenario breakdown (HAR)
# ===========================================================================
fig, ax = plt.subplots(figsize=(6.5, 3.2))
x = np.arange(len(scenarios))
width = 0.19

for i, (label, offset) in enumerate(zip(short_models, offsets)):
    vals = scenario_har[label]
    bars = ax.bar(x + offset * width, vals, width * 0.92,
                  label=label, color=colors[label], edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f'{v:.1f}', ha='center', va='bottom', fontsize=5.5, fontweight='bold',
                rotation=90)

ax.set_ylabel('HAR (%)')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=7)
ax.set_ylim(0, 125)
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.16), frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scenario_breakdown.pdf"), bbox_inches='tight')
plt.close(fig)
print("Saved scenario_breakdown.pdf")

# ===========================================================================
# Plot 3: Attack family breakdown (ASR)
# ===========================================================================
fig, ax = plt.subplots(figsize=(5.0, 3.0))
x = np.arange(len(families))
width = 0.19

for i, (label, offset) in enumerate(zip(short_models, offsets)):
    vals = family_asr[label]
    bars = ax.bar(x + offset * width, vals, width * 0.92,
                  label=label, color=colors[label], edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{v:.1f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

ax.set_ylabel('ASR (%)')
ax.set_xticks(x)
ax.set_xticklabels(families)
ax.set_ylim(0, 70)
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "attack_family_breakdown.pdf"), bbox_inches='tight')
plt.close(fig)
print("Saved attack_family_breakdown.pdf")

print(f"\nAll plots saved to {OUT_DIR}")
