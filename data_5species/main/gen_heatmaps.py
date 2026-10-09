#!/usr/bin/env python3
"""Generate MAP interaction matrix heatmaps for paper Figure 3."""
import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPECIES = ["So", "An", "Vei", "Fn", "Pg"]

# theta index -> (i, j) in 5x5 A matrix
IDX_MAP = {
    0:(0,0), 1:(0,1), 2:(1,1), 3:(0,1),  # b1,b2 -> skip
    5:(2,2), 6:(2,3), 7:(3,3),
    10:(0,2), 11:(0,3), 12:(1,2), 13:(1,3),
    14:(4,4),
    16:(0,4), 17:(1,4), 18:(2,4), 19:(3,4),
}
# Correct mapping: theta -> A[i,j]
def theta_to_A(theta_dict):
    A = np.zeros((5,5))
    mapping = {
        0:(0,0), 1:(0,1), 2:(1,1),
        5:(2,2), 6:(2,3), 7:(3,3),
        10:(0,2), 11:(0,3), 12:(1,2), 13:(1,3),
        14:(4,4),
        16:(0,4), 17:(1,4), 18:(2,4), 19:(3,4),
    }
    for ti, (i,j) in mapping.items():
        val = float(theta_dict.get(str(ti), 0))
        A[i,j] = val
        A[j,i] = val  # symmetric
    return A

# Find results
base = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs")
dirs = {
    "CS": "jax_ode_nuts_Commensal_Static_20260319_231727",
    "CH": "jax_ode_nuts_Commensal_HOBIC_20260319_231640",
    "DS": "jax_ode_nuts_Dysbiotic_Static_20260319_232305",
}
# Latest DH
for d in sorted(glob.glob(base + "/jax_ode_nuts_Dysbiotic_HOBIC_20260319_23*")):
    if os.path.exists(d + "/config.json"):
        dirs["DH"] = os.path.basename(d)

# Also check for unlocked CS/CH
for d in sorted(glob.glob(base + "/jax_ode_nuts_Commensal_Static_20260320_*")):
    if os.path.exists(d + "/config.json"):
        dirs["CS"] = os.path.basename(d)
for d in sorted(glob.glob(base + "/jax_ode_nuts_Commensal_HOBIC_20260320_*")):
    if os.path.exists(d + "/config.json"):
        dirs["CH"] = os.path.basename(d)

# Single figure: 2x2 heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
order = ["DH", "DS", "CS", "CH"]
titles = {
    "DH": "Dysbiotic HOBIC (DH)",
    "DS": "Dysbiotic Static (DS)",
    "CS": "Commensal Static (CS)",
    "CH": "Commensal HOBIC (CH)",
}

# Get global vmin/vmax
all_A = []
for label in order:
    d = os.path.join(base, dirs[label])
    theta = json.load(open(d + "/theta_MAP.json"))
    A = theta_to_A(theta)
    all_A.append(A)
vmax = max(np.abs(a).max() for a in all_A)
vmin = -vmax

for idx, (label, A) in enumerate(zip(order, all_A)):
    ax = axes[idx // 2, idx % 2]
    d = os.path.join(base, dirs[label])
    cfg = json.load(open(d + "/config.json"))
    rmse = cfg.get("rmse", 0)

    im = ax.imshow(A, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(5))
    ax.set_xticklabels(SPECIES, fontsize=11)
    ax.set_yticks(range(5))
    ax.set_yticklabels(SPECIES, fontsize=11)
    ax.set_title("{}\nRMSE = {:.4f}".format(titles[label], rmse),
                 fontsize=12, fontweight="bold")

    # Annotate values
    for i in range(5):
        for j in range(5):
            val = A[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            if abs(val) > 0.01:
                ax.text(j, i, "{:.2f}".format(val), ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
            else:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=8, color="gray")

# Colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("$A_{ij}$ (interaction strength)", fontsize=12)

fig.suptitle("MAP Interaction Matrices $\\mathbf{A}$ (fix-$\\psi$, all 20 params estimated)",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 0.89, 0.95])

outpath = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/figures_fixpsi/heatmap_A_4cond.png")
fig.savefig(outpath, dpi=300, bbox_inches="tight")
print("Saved:", outpath)

# Also save individual condition heatmaps for paper
for idx, (label, A) in enumerate(zip(order, all_A)):
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    d = os.path.join(base, dirs[label])
    cfg = json.load(open(d + "/config.json"))
    rmse = cfg.get("rmse", 0)

    im2 = ax2.imshow(A, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(SPECIES, fontsize=13)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(SPECIES, fontsize=13)
    ax2.set_title("{} — RMSE = {:.4f}".format(titles[label], rmse),
                  fontsize=13, fontweight="bold")
    for i in range(5):
        for j in range(5):
            val = A[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            if abs(val) > 0.01:
                ax2.text(j, i, "{:.2f}".format(val), ha="center", va="center",
                         fontsize=11, color=color, fontweight="bold")
            else:
                ax2.text(j, i, "0", ha="center", va="center",
                         fontsize=9, color="gray")
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="$A_{ij}$")
    plt.tight_layout()
    out2 = os.path.expanduser(
        "~/Tmcmc202601/data_5species/main/_runs/figures_fixpsi/heatmap_A_{}.png".format(label))
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("Saved:", out2)
