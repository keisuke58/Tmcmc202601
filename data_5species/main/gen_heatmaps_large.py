#!/usr/bin/env python3
"""Generate MAP interaction matrix heatmaps — paper quality.
Layout: CS(top-left) CH(top-right) DS(bot-left) DH(bot-right)
"""
import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paper style
rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 22,
    "axes.labelsize": 24,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif",
})

SPECIES = ["So", "An", "Vei", "Fn", "Pg"]

def theta_to_A(theta_dict):
    A = np.zeros((5, 5))
    mapping = {
        0:(0,0), 1:(0,1), 2:(1,1),
        5:(2,2), 6:(2,3), 7:(3,3),
        10:(0,2), 11:(0,3), 12:(1,2), 13:(1,3),
        14:(4,4),
        16:(0,4), 17:(1,4), 18:(2,4), 19:(3,4),
    }
    for ti, (i, j) in mapping.items():
        val = float(theta_dict.get(str(ti), 0))
        A[i, j] = val
        A[j, i] = val
    return A

base = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs")

def find_latest(cond, cult):
    """Find run with best RMSE."""
    candidates = []
    for pat in ["20260319", "20260320"]:
        candidates += [d for d in glob.glob(base + "/jax_ode_nuts_{}_{}_{}_*".format(cond, cult, pat))
                       if os.path.exists(d + "/config.json")]
    best, best_rmse = None, 999
    for c in candidates:
        try:
            cfg = json.load(open(c + "/config.json"))
            r = cfg.get("rmse", 999)
            np = cfg.get("n_particles", 0)
            # Prefer 10000p runs, then best RMSE
            score = r - (10.0 if np >= 10000 else 0.0)
            if score < best_rmse:
                best_rmse = score
                best = c
        except:
            pass
    return best

# Layout: CS(TL) CH(TR) DS(BL) DH(BR)
panels = [
    ("CS", "Commensal", "Static",  "Commensal Static (CS)"),
    ("CH", "Commensal", "HOBIC",   "Commensal HOBIC (CH)"),
    ("DS", "Dysbiotic", "Static",  "Dysbiotic Static (DS)"),
    ("DH", "Dysbiotic", "HOBIC",   "Dysbiotic HOBIC (DH)"),
]

# Load all
data = {}
for label, cond, cult, title in panels:
    d = find_latest(cond, cult)
    theta = json.load(open(d + "/theta_MAP.json"))
    cfg = json.load(open(d + "/config.json"))
    A = theta_to_A(theta)
    data[label] = {"A": A, "rmse": cfg.get("rmse", 0), "title": title, "dir": d}
    print("{}: {} RMSE={:.4f}".format(label, os.path.basename(d), cfg.get("rmse", 0)))

# Global color scale
vmax = max(np.abs(d["A"]).max() for d in data.values())
vmax = np.ceil(vmax)  # round up

# ===== Figure 3: 2x2 heatmaps =====
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for idx, (label, _, _, _) in enumerate(panels):
    ax = axes[idx]
    d = data[label]
    A = d["A"]

    im = ax.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(5))
    ax.set_xticklabels(SPECIES)
    ax.set_yticks(range(5))
    ax.set_yticklabels(SPECIES)

    # Title with condition and RMSE
    ax.set_title("{}\nRMSE={:.4f}".format(d["title"], d["rmse"]),
                 fontsize=20, fontweight="bold", pad=6)
    if idx > 0:
        ax.set_yticklabels([])

    # a_ij labels
    A_LABELS = [
        ["$a_{11}$","$a_{12}$","$a_{13}$","$a_{14}$","$a_{15}$"],
        ["$a_{12}$","$a_{22}$","$a_{23}$","$a_{24}$","$a_{25}$"],
        ["$a_{13}$","$a_{23}$","$a_{33}$","$a_{34}$","$a_{35}$"],
        ["$a_{14}$","$a_{24}$","$a_{34}$","$a_{44}$","$a_{45}$"],
        ["$a_{15}$","$a_{25}$","$a_{35}$","$a_{45}$","$a_{55}$"],
    ]

    # Annotate
    for i in range(5):
        for j in range(5):
            val = A[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            if abs(val) > 0.005:
                txt = "{:.1f}".format(val) if abs(val) >= 1.0 else "{:.2f}".format(val)
                ax.text(j, i+0.15, txt, ha="center", va="center",
                        fontsize=16, color=color, fontweight="bold", fontfamily="serif")
            # a_ij label (top of cell, bold)
            ax.text(j, i-0.22, A_LABELS[i][j], ha="center", va="center",
                    fontsize=16, color=color, fontweight="bold", fontfamily="serif", alpha=0.7)

    # Grid lines
    for x in np.arange(-0.5, 5, 1):
        ax.axhline(x, color="white", linewidth=0.5)
        ax.axvline(x, color="white", linewidth=0.5)

# Shared colorbar
fig.subplots_adjust(right=0.92, wspace=0.08)
cbar_ax = fig.add_axes([0.935, 0.15, 0.012, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r"$A_{ij}$", fontsize=22)

out1 = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/figures_fixpsi/heatmap_A_4cond.png")
fig.savefig(out1, dpi=300, bbox_inches="tight")
print("Saved:", out1)
plt.close(fig)
