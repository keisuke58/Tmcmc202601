#!/usr/bin/env python3
"""Generate corner plot for dh_baseline posterior (paper figure)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_5species'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Load data ---
samples = np.load('../data_5species/_runs/dh_baseline/samples.npy')  # (300, 20)
print(f"Samples shape: {samples.shape}")

PARAM_NAMES = [
    r"$a_{11}$", r"$a_{12}$", r"$a_{22}$", r"$b_1$", r"$b_2$",
    r"$a_{33}$", r"$a_{34}$", r"$a_{44}$", r"$b_3$", r"$b_4$",
    r"$a_{13}$", r"$a_{14}$", r"$a_{23}$", r"$a_{24}$",
    r"$a_{55}$", r"$b_5$",
    r"$a_{15}$", r"$a_{25}$", r"$a_{35}$", r"$a_{45}$"
]

# Select key parameters for a readable corner plot
# Focus on: bridge params (a35, a45), growth rates (b1..b5), key interactions
key_idx = [3, 4, 8, 9, 15, 18, 19]  # b1, b2, b3, b4, b5, a35, a45
key_names = [PARAM_NAMES[i] for i in key_idx]
key_samples = samples[:, key_idx]

n = len(key_idx)
fig, axes = plt.subplots(n, n, figsize=(12, 12))

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if j > i:
            ax.axis('off')
            continue
        if i == j:
            ax.hist(key_samples[:, i], bins=25, density=True,
                    color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.set_yticks([])
        else:
            ax.scatter(key_samples[:, j], key_samples[:, i],
                       s=3, alpha=0.3, color='steelblue')

        if i == n - 1:
            ax.set_xlabel(key_names[j], fontsize=11)
        else:
            ax.set_xticklabels([])
        if j == 0 and i > 0:
            ax.set_ylabel(key_names[i], fontsize=11)
        elif j > 0:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

fig.suptitle('DH Posterior: Key Parameters (300 samples)', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

outdir = 'figures/paper_final'
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'Fig19_corner_plot_dh_key_params.png')
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"Saved: {outpath}")
plt.close()

# --- Parameter correspondence table (LaTeX) ---
print("\n=== Parameter Index Table (LaTeX) ===")
print(r"\begin{tabular}{cllc}")
print(r"\hline")
print(r"Index & Symbol & Physical meaning & DH bounds \\")
print(r"\hline")

meanings = [
    "So self-interaction", "So-An interaction", "An self-interaction",
    "So growth rate", "An growth rate",
    "Vei self-interaction", "Vei-Fn interaction", "Fn self-interaction",
    "Vei growth rate", "Fn growth rate",
    "So-Vei interaction", "So-Fn interaction", "An-Vei interaction", "An-Fn interaction",
    "Pg self-interaction", "Pg growth rate",
    "So-Pg interaction", "An-Pg interaction", "Vei-Pg interaction (bridge)", "Fn-Pg interaction (bridge)"
]

dh_bounds = {
    0: "[0, 5]", 1: "[-0.5, 3]", 2: "[0, 5]", 3: "[0, 3]", 4: "[0, 3]",
    5: "[0, 5]", 6: "[-0.5, 3]", 7: "[0, 5]", 8: "[0, 5]", 9: "[0, 3]",
    10: "[-3, 3]", 11: "[-0.5, 3]", 12: "[0, 5]", 13: "[-0.5, 3]",
    14: "[0, 5]", 15: "[0, 0.5]",
    16: "[-0.5, 3]", 17: "[-0.5, 3]", 18: "[0, 5]", 19: "[0, 5]"
}

raw_names = [
    "a_{11}", "a_{12}", "a_{22}", "b_1", "b_2",
    "a_{33}", "a_{34}", "a_{44}", "b_3", "b_4",
    "a_{13}", "a_{14}", "a_{23}", "a_{24}",
    "a_{55}", "b_5",
    "a_{15}", "a_{25}", "a_{35}", "a_{45}"
]

for i in range(20):
    print(f"  {i} & ${raw_names[i]}$ & {meanings[i]} & {dh_bounds[i]} \\\\")
print(r"\hline")
print(r"\end{tabular}")

# --- Per-condition free parameter count ---
print("\n=== Free parameters per condition ===")
conditions = {
    "Commensal Static (CS)": {"locks": [9, 15, 6, 7, 11, 13, 14, 16, 17, 18, 19], "n_free": 9},
    "Commensal HOBIC (CH)": {"locks": [6, 12, 13, 16, 17, 15, 18], "n_free": 12},  # 20-8=12, actually need to check
    "Dysbiotic Static (DS)": {"locks": [6, 12, 13, 16, 17], "n_free": 15},
    "Dysbiotic HOBIC (DH)": {"locks": [], "n_free": 20},
}

for cond, info in conditions.items():
    n_free = 20 - len(info["locks"])
    print(f"  {cond}: {n_free} free, {len(info['locks'])} locked")
    locked_names = [f"${raw_names[i]}$" for i in sorted(info["locks"])]
    print(f"    Locked: {', '.join(locked_names) if locked_names else 'none'}")

print("\nDone.")
