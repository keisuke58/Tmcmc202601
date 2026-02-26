#!/usr/bin/env python3
"""
Generate paper-quality Fig 21: NUTS vs HMC vs RW TMCMC comparison.

Creates a compact 2-panel figure:
  (a) Acceptance rate bar chart across 4 conditions
  (b) Improvement ratio vs free dimensionality

Uses saved results from nuts_4condition_comparison.json.
"""
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAPER_FIG_DIR = PROJECT_ROOT / "FEM" / "figures" / "paper_final"

# Load results
data = json.load(open(SCRIPT_DIR / "nuts_4condition_comparison.json"))

conditions = ["Commensal_Static", "Commensal_HOBIC", "Dysbiotic_Static", "Dysbiotic_HOBIC"]
short_labels = ["CS\n(d=9)", "CH\n(d=13)", "DS\n(d=15)", "DH\n(d=20)"]

rw_accept = [data["conditions"][c]["RW"]["avg_accept"] for c in conditions]
hmc_accept = [data["conditions"][c]["HMC"]["avg_accept"] for c in conditions]
nuts_accept = [data["conditions"][c]["NUTS"]["avg_accept"] for c in conditions]
free_dims = [data["conditions"][c]["free_dims"] for c in conditions]

# Improvement ratios
nuts_improvement = [n / r for n, r in zip(nuts_accept, rw_accept)]
hmc_improvement = [h / r for h, r in zip(hmc_accept, rw_accept)]

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 2, 2]})

# Colors
c_rw = "#E53935"  # red
c_hmc = "#1E88E5"  # blue
c_nuts = "#43A047"  # green

# (a) Acceptance rate bar chart
ax = axes[0]
x = np.arange(len(conditions))
w = 0.25
ax.bar(x - w, rw_accept, w, color=c_rw, alpha=0.85, label="RW", edgecolor="white", linewidth=0.5)
ax.bar(x, hmc_accept, w, color=c_hmc, alpha=0.85, label="HMC", edgecolor="white", linewidth=0.5)
ax.bar(
    x + w, nuts_accept, w, color=c_nuts, alpha=0.85, label="NUTS", edgecolor="white", linewidth=0.5
)

# Value labels
for i in range(len(conditions)):
    ax.text(
        x[i] - w,
        rw_accept[i] + 0.015,
        f"{rw_accept[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=c_rw,
    )
    ax.text(
        x[i],
        hmc_accept[i] + 0.015,
        f"{hmc_accept[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=c_hmc,
    )
    ax.text(
        x[i] + w,
        nuts_accept[i] + 0.015,
        f"{nuts_accept[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=c_nuts,
    )

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("Acceptance Rate", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("(a) Mutation Acceptance Rate", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
ax.text(3.5, 0.52, "50% baseline", fontsize=7, color="gray", ha="right")

# (b) Improvement ratio vs free dims
ax = axes[1]
ax.plot(free_dims, hmc_improvement, "s-", color=c_hmc, ms=8, lw=2, label="HMC/RW", zorder=3)
ax.plot(free_dims, nuts_improvement, "o-", color=c_nuts, ms=8, lw=2, label="NUTS/RW", zorder=3)

# Annotate
for i, d in enumerate(free_dims):
    ax.annotate(
        f"{nuts_improvement[i]:.2f}×",
        (d, nuts_improvement[i]),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=8,
        color=c_nuts,
    )
    ax.annotate(
        f"{hmc_improvement[i]:.2f}×",
        (d, hmc_improvement[i]),
        textcoords="offset points",
        xytext=(8, -10),
        fontsize=8,
        color=c_hmc,
    )

ax.set_xlabel("Free Parameters (d)", fontsize=11)
ax.set_ylabel("Accept Ratio vs RW", fontsize=11)
ax.set_title("(b) Improvement vs Dimensionality", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.8, 2.5)

# (c) Wall time comparison
ax = axes[2]
rw_time = [data["conditions"][c]["RW"]["time_s"] for c in conditions]
hmc_time = [data["conditions"][c]["HMC"]["time_s"] for c in conditions]
nuts_time = [data["conditions"][c]["NUTS"]["time_s"] for c in conditions]

ax.bar(x - w, rw_time, w, color=c_rw, alpha=0.85, label="RW", edgecolor="white", linewidth=0.5)
ax.bar(x, hmc_time, w, color=c_hmc, alpha=0.85, label="HMC", edgecolor="white", linewidth=0.5)
ax.bar(
    x + w, nuts_time, w, color=c_nuts, alpha=0.85, label="NUTS", edgecolor="white", linewidth=0.5
)

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("Wall Time [s]", fontsize=11)
ax.set_title("(c) Computational Cost", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = PAPER_FIG_DIR / "Fig21_nuts_comparison.png"
plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# Also save as PDF
out_pdf = PAPER_FIG_DIR / "Fig21_nuts_comparison.pdf"
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={"width_ratios": [3, 2, 2]})

# Recreate for PDF (same code)
ax = axes2[0]
ax.bar(x - w, rw_accept, w, color=c_rw, alpha=0.85, label="RW", edgecolor="white", linewidth=0.5)
ax.bar(x, hmc_accept, w, color=c_hmc, alpha=0.85, label="HMC", edgecolor="white", linewidth=0.5)
ax.bar(
    x + w, nuts_accept, w, color=c_nuts, alpha=0.85, label="NUTS", edgecolor="white", linewidth=0.5
)
for i in range(len(conditions)):
    ax.text(
        x[i] - w,
        rw_accept[i] + 0.015,
        f"{rw_accept[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=c_rw,
    )
    ax.text(
        x[i],
        hmc_accept[i] + 0.015,
        f"{hmc_accept[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=c_hmc,
    )
    ax.text(
        x[i] + w,
        nuts_accept[i] + 0.015,
        f"{nuts_accept[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=c_nuts,
    )
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("Acceptance Rate", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("(a) Mutation Acceptance Rate", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

ax = axes2[1]
ax.plot(free_dims, hmc_improvement, "s-", color=c_hmc, ms=8, lw=2, label="HMC/RW", zorder=3)
ax.plot(free_dims, nuts_improvement, "o-", color=c_nuts, ms=8, lw=2, label="NUTS/RW", zorder=3)
for i, d in enumerate(free_dims):
    ax.annotate(
        f"{nuts_improvement[i]:.2f}×",
        (d, nuts_improvement[i]),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=8,
        color=c_nuts,
    )
    ax.annotate(
        f"{hmc_improvement[i]:.2f}×",
        (d, hmc_improvement[i]),
        textcoords="offset points",
        xytext=(8, -10),
        fontsize=8,
        color=c_hmc,
    )
ax.set_xlabel("Free Parameters (d)", fontsize=11)
ax.set_ylabel("Accept Ratio vs RW", fontsize=11)
ax.set_title("(b) Improvement vs Dimensionality", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.8, 2.5)

ax = axes2[2]
ax.bar(x - w, rw_time, w, color=c_rw, alpha=0.85, label="RW", edgecolor="white", linewidth=0.5)
ax.bar(x, hmc_time, w, color=c_hmc, alpha=0.85, label="HMC", edgecolor="white", linewidth=0.5)
ax.bar(
    x + w, nuts_time, w, color=c_nuts, alpha=0.85, label="NUTS", edgecolor="white", linewidth=0.5
)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("Wall Time [s]", fontsize=11)
ax.set_title("(c) Computational Cost", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(str(out_pdf), bbox_inches="tight")
plt.close()
print(f"Saved: {out_pdf}")
