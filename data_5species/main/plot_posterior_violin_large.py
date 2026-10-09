#!/usr/bin/env python3
"""Paper Figure: Posterior parameter distributions (violin) for all 4 conditions."""
import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 19,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

base = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs")

# theta indices for interaction matrix only (exclude b_i = theta[3,4,8,9,15])
PARAM_INDICES = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]

PARAM_LABELS = [
    "$a_{11}$", "$a_{12}$", "$a_{22}$",
    "$a_{33}$", "$a_{34}$", "$a_{44}$",
    "$a_{13}$", "$a_{14}$", "$a_{23}$", "$a_{24}$",
    "$a_{55}$",
    "$a_{15}$", "$a_{25}$", "$a_{35}$", "$a_{45}$",
]

BLOCK_LABELS = [
    "So--An", "So--An", "So--An",
    "Vei--Fn", "Vei--Fn", "Vei--Fn",
    "Cross", "Cross", "Cross", "Cross",
    "Pg",
    "Pg cross", "Pg cross", "Pg cross", "Pg cross",
]

COLORS = {"CS": "#2166AC", "CH": "#4393C3", "DS": "#D6604D", "DH": "#B2182B"}

def find_best(cond, cult):
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

panels = [
    ("CS", "Commensal", "Static"),
    ("CH", "Commensal", "HOBIC"),
    ("DS", "Dysbiotic", "Static"),
    ("DH", "Dysbiotic", "HOBIC"),
]

# Load all samples
data = {}
for label, cond, cult in panels:
    d = find_best(cond, cult)
    samples = np.load(d + "/samples.npy")
    data[label] = samples
    print("{}: {} ({} samples)".format(label, os.path.basename(d), len(samples)))

# Figure: 3 rows x 5 cols = 15 params
fig, axes = plt.subplots(3, 5, figsize=(16, 8))

for plot_idx, p_idx in enumerate(PARAM_INDICES):
    row, col = plot_idx // 5, plot_idx % 5
    ax = axes[row, col]

    vp_data = []
    colors_list = []

    for i, label in enumerate(["CS", "CH", "DS", "DH"]):
        s = data[label][:, p_idx]
        vp_data.append(s)
        colors_list.append(COLORS[label])

    vp = ax.violinplot(vp_data, positions=range(4), showmedians=True,
                        showextrema=False, widths=0.7)

    for j, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors_list[j])
        body.set_alpha(0.6)
        body.set_edgecolor(colors_list[j])
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.5)

    # MAP values as dots
    for i, label in enumerate(["CS", "CH", "DS", "DH"]):
        d = find_best(*[p[1:] for p in panels if p[0] == label][0])
        theta_map = json.load(open(d + "/theta_MAP.json"))
        map_val = float(theta_map[str(p_idx)])
        ax.plot(i, map_val, "k*", markersize=8, zorder=5)

    ax.set_title(PARAM_LABELS[plot_idx], fontsize=20, fontweight="bold")
    ax.set_xticks(range(4))
    if row == 2:
        ax.set_xticklabels(["CS", "CH", "DS", "DH"], fontsize=17)
    else:
        ax.set_xticklabels([])
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.tick_params(direction="in", length=2)

# Row block labels
row_labels = ["So--An / Vei--Fn", "Cross-block", "Pg self + cross"]
for r in range(3):
    axes[r, 0].set_ylabel(row_labels[r], fontsize=17, fontweight="bold")

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor=COLORS["CS"], alpha=0.6, label="CS"),
    Patch(facecolor=COLORS["CH"], alpha=0.6, label="CH"),
    Patch(facecolor=COLORS["DS"], alpha=0.6, label="DS"),
    Patch(facecolor=COLORS["DH"], alpha=0.6, label="DH"),
    Line2D([0], [0], marker="*", color="k", linewidth=0, markersize=8, label="MAP"),
]
fig.legend(handles=legend_els, loc="upper center", ncol=5, fontsize=17,
           bbox_to_anchor=(0.5, 1.03))

plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.5, w_pad=0.3)
plt.subplots_adjust(top=0.93)

out = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/figures_fixpsi/paper_posterior_violin.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("Saved:", out)
out2 = out.replace(".png", ".pdf")
fig.savefig(out2, bbox_inches="tight")
print("Saved:", out2)
