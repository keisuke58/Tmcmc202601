#!/usr/bin/env python3
"""
plot_3tooth_comparison.py
=========================
Cross-tooth comparison: T23, T30, T31 × 4 conditions
"""

import os
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEETH = {
    "T23": os.path.join(os.path.dirname(os.path.abspath(__file__))),
    "T30": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_growth_job_runs_t30"),
    "T31": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_growth_job_runs_t31"),
}
CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]
LABELS = ["Comm.\nStatic", "Comm.\nHOBIC", "DH\nBaseline", "Dysbiotic\nStatic"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
TOOTH_MARKERS = {"T23": "o", "T30": "s", "T31": "^"}
TOOTH_COLORS = {"T23": "#E07B39", "T30": "#3A7EBF", "T31": "#5BA85A"}


def load_summary(tooth_dir):
    path = os.path.join(tooth_dir, "growth_summary.csv")
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["condition"]] = {k: float(v) for k, v in row.items() if k != "condition"}
    return rows


all_summaries = {t: load_summary(d) for t, d in TEETH.items()}

# ── Fig: 3-tooth × 4-condition bar chart (max_mises_load, max_U) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("3-Tooth Growth Load Comparison (Patient 1: T23/T30/T31)", fontsize=13)

n_cond = len(CONDITIONS)
n_teeth = len(TEETH)
w = 0.22
xs = np.arange(n_cond)

for ax_i, (metric, ylabel, scale, title) in enumerate(
    [
        (
            "max_mises_load_Pa",
            "max σ_Mises_LOAD [Pa]",
            1.0,
            "Max von Mises (LOAD step, GCF 100 Pa)",
        ),
        ("max_U_mm", "max |U| [mm]", 1.0, "Max Displacement (LOAD step)"),
    ]
):
    ax = axes[ax_i]
    for ti, (tooth, summ) in enumerate(all_summaries.items()):
        offset = (ti - 1) * w
        vals = [summ[c][metric] * scale for c in CONDITIONS]
        bars = ax.bar(
            xs + offset,
            vals,
            w,
            label=tooth,
            color=TOOTH_COLORS[tooth],
            alpha=0.85,
            edgecolor="k",
            lw=0.6,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(title="Tooth", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = os.path.join(TEETH["T23"], "fig_3tooth_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", out)

# ── Fig: GROWTH σ comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.set_title("Max σ_Mises_GROWTH vs Condition (3 teeth)", fontsize=12)
for ti, (tooth, summ) in enumerate(all_summaries.items()):
    offset = (ti - 1) * w
    vals = [summ[c]["max_mises_growth_Pa"] for c in CONDITIONS]
    ax.bar(
        xs + offset,
        vals,
        w,
        label=tooth,
        color=TOOTH_COLORS[tooth],
        alpha=0.85,
        edgecolor="k",
        lw=0.6,
    )
ax.set_xticks(xs)
ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylabel("max σ_Mises_GROWTH [Pa]", fontsize=10)
ax.legend(title="Tooth", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out2 = os.path.join(TEETH["T23"], "fig_3tooth_growth_stress.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", out2)

# Print summary table
print("\n=== 3-TOOTH × 4-CONDITION SUMMARY (Pa / mm) ===")
print(
    "%-6s  %-22s  %10s  %10s  %10s"
    % ("Tooth", "Condition", "σG_max[Pa]", "σL_max[Pa]", "U_max[mm]")
)
print("-" * 65)
for tooth, summ in all_summaries.items():
    for cond in CONDITIONS:
        s = summ[cond]
        print(
            "%-6s  %-22s  %10.4g  %10.4g  %10.6f"
            % (tooth, cond, s["max_mises_growth_Pa"], s["max_mises_load_Pa"], s["max_U_mm"])
        )
    print()
