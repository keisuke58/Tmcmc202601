#!/usr/bin/env python3
"""Paper Figure 2: φ̄ posterior predictive fits, 4 conditions × 5 species.
Heine-consistent colors, publication quality.
Layout: CS(row1) CH(row2) DS(row3) DH(row4), columns = 5 species."""
import numpy as np
import json
import os
import sys
import glob
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif",
})

# Heine 2025 colors
# So=blue, An=green, Vd=orange(D)/yellow(C), Fn=purple, Pg=red
SPECIES = ["So", "An", "Vd", "Fn", "Pg"]
SPECIES_ITALIC = [r"$\it{S. oralis}$", r"$\it{A. naeslundii}$",
                  r"$\it{V. dispar}$", r"$\it{F. nucleatum}$",
                  r"$\it{P. gingivalis}$"]

def get_colors(condition):
    """Heine-consistent colors. Vd changes by condition."""
    if condition.startswith("C"):  # Commensal → V. parvula → yellow
        vd_color = "#DAA520"  # goldenrod
    else:  # Dysbiotic → V. dispar → orange
        vd_color = "#FF8C00"  # dark orange
    return ["#2166AC", "#1B7837", vd_color, "#7B3294", "#B2182B"]
    #        So blue   An green  Vd         Fn purple  Pg red

SPECIES_MAP_CSV = {
    "S. oralis": 0, "A. naeslundii": 1, "V. dispar": 2,
    "F. nucleatum": 3, "P. gingivalis_20709": 4,
}

DATA_DIR = os.path.expanduser("~/Tmcmc202601/data_5species/experiment_data")

def load_exp_data(condition, cultivation):
    fname = os.path.join(DATA_DIR, "fig3_species_distribution_summary.csv")
    raw = {}
    with open(fname) as f:
        for row in csv.DictReader(f):
            if row["condition"] == condition and row["cultivation"] == cultivation:
                day = int(row["day"])
                si = SPECIES_MAP_CSV.get(row["species"])
                if si is None:
                    continue
                if day not in raw:
                    raw[day] = np.zeros(5)
                raw[day][si] = float(row["mean"])
    days = sorted(raw.keys())
    data_pct = np.array([raw[d] for d in days])
    sums = data_pct.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    data_frac = data_pct / sums
    return np.array(days), data_frac

# ODE simulation (replicator)
def hill(x, K=0.05, n=4.0):
    xn = np.abs(x)**n
    return xn / (K**n + xn)

def ode_rhs(phi, theta):
    a = np.zeros((5, 5))
    b = np.zeros(5)
    a[0,0]=theta[0]; a[0,1]=theta[1]; a[1,1]=theta[2]; b[0]=theta[3]; b[1]=theta[4]
    a[2,2]=theta[5]; a[2,3]=theta[6]; a[3,3]=theta[7]; b[2]=theta[8]; b[3]=theta[9]
    a[0,2]=theta[10]; a[0,3]=theta[11]; a[1,2]=theta[12]; a[1,3]=theta[13]
    a[4,4]=theta[14]; b[4]=theta[15]
    a[0,4]=theta[16]; a[1,4]=theta[17]; a[2,4]=theta[18]; a[3,4]=theta[19]
    fitness = b.copy()
    for i in range(5):
        for j in range(5):
            if a[i,j] != 0:
                fitness[i] += a[i,j] * hill(phi[j])
    mean_f = np.dot(phi, fitness)
    return phi * (fitness - mean_f)

def simulate(theta, ic, n_steps=2500, dt=1e-4):
    x = np.array(ic, dtype=np.float64)
    traj = [x.copy()]
    for _ in range(n_steps):
        dx = ode_rhs(x, theta)
        x = x + dt * dx
        x = np.clip(x, 1e-12, None)
        x = x / x.sum()
        traj.append(x.copy())
    return np.array(traj)

# Find latest results
base = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs")

def find_latest(cond, cult):
    candidates = []
    for pat in ["20260319", "20260320"]:
        candidates += sorted([d for d in glob.glob(
            base + "/jax_ode_nuts_{}_{}_{}_*".format(cond, cult, pat))
            if os.path.exists(d + "/config.json")])
    return candidates[-1] if candidates else None

panels = [
    ("CS", "Commensal", "Static"),
    ("CH", "Commensal", "HOBIC"),
    ("DS", "Dysbiotic", "Static"),
    ("DH", "Dysbiotic", "HOBIC"),
]

# ===== Main figure =====
fig, axes = plt.subplots(4, 5, figsize=(14, 11))

for row_idx, (label, cond, cult) in enumerate(panels):
    d = find_latest(cond, cult)
    theta_j = json.load(open(d + "/theta_MAP.json"))
    theta = np.array([theta_j[str(i)] for i in range(20)])
    cfg = json.load(open(d + "/config.json"))
    samples = np.load(d + "/samples.npy")

    days_exp, data_exp = load_exp_data(cond, cult)
    ic = data_exp[0]
    n_steps = 2500
    t_days = days_exp[0] + np.arange(n_steps+1) * (days_exp[-1] - days_exp[0]) / n_steps

    # MAP trajectory
    traj_map = simulate(theta, ic, n_steps)

    # Posterior predictive (50 samples)
    n_sub = min(50, len(samples))
    np.random.seed(42)
    idx_sub = np.random.choice(len(samples), n_sub, replace=False)
    trajs = np.zeros((n_sub, n_steps+1, 5))
    for k, si_idx in enumerate(idx_sub):
        trajs[k] = simulate(samples[si_idx], ic, n_steps)
    q10 = np.percentile(trajs, 10, axis=0)
    q25 = np.percentile(trajs, 25, axis=0)
    q75 = np.percentile(trajs, 75, axis=0)
    q90 = np.percentile(trajs, 90, axis=0)

    colors = get_colors(label)
    rmse = cfg.get("rmse", 0)

    for si in range(5):
        ax = axes[row_idx, si]
        col = colors[si]

        # CI bands
        ax.fill_between(t_days, q10[:, si], q90[:, si], color=col, alpha=0.12)
        ax.fill_between(t_days, q25[:, si], q75[:, si], color=col, alpha=0.25)

        # MAP line
        ax.plot(t_days, traj_map[:, si], "-", color=col, linewidth=1.8)

        # Experimental data
        ax.plot(days_exp, data_exp[:, si], "o", color=col,
                markersize=6, markeredgecolor="k", markeredgewidth=0.6, zorder=5)

        # Formatting
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([1, 5, 10, 15, 21])

        if row_idx == 0:
            ax.set_title(SPECIES_ITALIC[si], fontsize=13, pad=8)
        if row_idx == 3:
            ax.set_xlabel("Day", fontsize=12)
        else:
            ax.set_xticklabels([])

        if si == 0:
            ax.set_ylabel(r"$\bar{\varphi}_i$", fontsize=13)
            # Row label
            ax.annotate(label, xy=(-0.4, 0.5), xycoords="axes fraction",
                        fontsize=14, fontweight="bold", ha="center", va="center",
                        rotation=90)
        else:
            ax.set_yticklabels([])

        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(direction="in", length=3)

    # RMSE annotation on rightmost panel
    axes[row_idx, 4].annotate(
        "RMSE={:.3f}".format(rmse),
        xy=(0.95, 0.92), xycoords="axes fraction",
        fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"))

plt.tight_layout(h_pad=0.3, w_pad=0.2)
plt.subplots_adjust(left=0.07)

outpath = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/figures_fixpsi/paper_fig2_phibar.png")
fig.savefig(outpath, dpi=300, bbox_inches="tight")
print("Saved:", outpath)

# Also save PDF for LaTeX
outpdf = outpath.replace(".png", ".pdf")
fig.savefig(outpdf, bbox_inches="tight")
print("Saved:", outpdf)
