#!/usr/bin/env python3
"""Paper Figure 2 — final version.
Uses correct Newton solver via plot_gpu_tmcmc_results.py functions.
Vd color: goldenrod (Commensal) / orange (Dysbiotic), matching Heine 2025."""
import sys, os
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# Import from existing script
sys.path.insert(0, str(Path(__file__).parent))
from plot_gpu_tmcmc_results import (
    load_run, load_experiment_data, simulate_ode_map,
    compute_posterior_predictive, convert_days_to_idx,
    load_replicate_data, SPECIES, SPECIES_MAP,
)

rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Heine 2025 colors — Vd changes by condition
COLOR_MAP = {
    "CS": ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#B2182B"],  # Vd=goldenrod
    "CH": ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#B2182B"],  # Vd=goldenrod
    "DS": ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#B2182B"],  # Vd=orange
    "DH": ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#B2182B"],  # Vd=orange
}

SPECIES_ITALIC = [
    r"$\it{S.\/ oralis}$",
    r"$\it{A.\/ naeslundii}$",
    r"$\it{V.\/ dispar}$",
    r"$\it{F.\/ nucleatum}$",
    r"$\it{P.\/ gingivalis}$",
]

COND_FULL = {
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
    "DS": "Dysbiotic Static",
    "DH": "Dysbiotic HOBIC",
}

# ===== Main =====
base = Path(__file__).parent / "_runs"
data_dir = Path(__file__).parent.parent

# Run dirs (CS/CH/DS/DH order)
import glob
def find_best(cond, cult):
    """Find run with lowest RMSE."""
    candidates = []
    for pat in ["20260319", "20260320"]:
        candidates += [Path(d) for d in glob.glob(
            str(base / "jax_ode_nuts_{}_{}_{}_*".format(cond, cult, pat)))
            if os.path.exists(str(Path(d) / "config.json"))]
    if not candidates:
        return None
    best = None
    best_score = 999
    for c in candidates:
        try:
            cfg = json.load(open(str(c / "config.json")))
            rmse = cfg.get("rmse", 999)
            np_ = cfg.get("n_particles", 0)
            score = rmse - (10.0 if np_ >= 10000 else 0.0)
            if score < best_score:
                best_score = score
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

runs = {}
for label, cond, cult in panels:
    runs[label] = find_best(cond, cult)
    print(f"  {label}: {runs[label].name}")

rep_df = load_replicate_data(data_dir)

fig, axes = plt.subplots(4, 5, figsize=(17, 13))

for row_idx, (label, cond, cult) in enumerate(panels):
    run = load_run(runs[label])
    cfg = run["config"]
    theta_map = run["theta_MAP"]
    colors = COLOR_MAP[label]

    days, data_frac, total_vol, data_abs, phi_init_abs = load_experiment_data(
        data_dir, cond, cult)
    phi_init_frac = data_frac[0, :]

    # MAP trajectory (φ̄ = φ × ψ)
    traj = simulate_ode_map(theta_map, cond, cult, phi_init=phi_init_frac)
    idx = convert_days_to_idx(days)
    phi_raw = traj[idx, 0:5]
    psi_raw = traj[idx, 6:11]
    phibar_map = phi_raw * psi_raw
    phibar_map = np.clip(phibar_map, 1e-10, 1.0)
    phibar_map = phibar_map / np.maximum(phibar_map.sum(axis=1, keepdims=True), 1e-12)

    # Posterior predictive
    print(f"  Posterior predictive for {label}...")
    posterior_pred = compute_posterior_predictive(
        run["samples"], phi_init_frac, cond, cult, days, n_max=100)

    # Replicate data
    cond_rep = rep_df[
        (rep_df["condition"] == cond) & (rep_df["cultivation"] == cult)]

    rmse = cfg.get("rmse", 0)
    logL = cfg.get("max_logL", 0)

    sp_name_to_idx = {
        "S. oralis": 0, "A. naeslundii": 1,
        "V. dispar": 2, "V. parvula": 2,
        "F. nucleatum": 3,
        "P. gingivalis_W83": 4, "P. gingivalis_20709": 4,
    }

    for sp_idx in range(5):
        ax = axes[row_idx, sp_idx]
        col = colors[sp_idx]

        # Replicate boxplots (IQR)
        sp_names = [k for k, v in sp_name_to_idx.items() if v == sp_idx]
        box_data, box_pos = [], []
        for day in days:
            vals = cond_rep[
                (cond_rep["day"] == day) & (cond_rep["species"].isin(sp_names))
            ]["distribution_pct"].values / 100.0
            if len(vals) > 0:
                box_data.append(vals)
                box_pos.append(day)
        if box_data:
            bp = ax.boxplot(
                box_data, positions=box_pos, widths=1.8,
                patch_artist=True, showfliers=False, manage_ticks=False, zorder=1,
                medianprops=dict(color="k", linewidth=1),
                whiskerprops=dict(color="gray", linewidth=0.7),
                capprops=dict(color="gray", linewidth=0.7),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_alpha(0.2)
                patch.set_edgecolor(col)
                patch.set_linewidth(0.6)

        # CI bands
        sp_samples = posterior_pred[:, :, sp_idx]
        q10 = np.percentile(sp_samples, 10, axis=0)
        q25 = np.percentile(sp_samples, 25, axis=0)
        q75 = np.percentile(sp_samples, 75, axis=0)
        q90 = np.percentile(sp_samples, 90, axis=0)
        ax.fill_between(days, q10, q90, alpha=0.12, color=col)
        ax.fill_between(days, q25, q75, alpha=0.25, color=col)

        # MAP line
        ax.plot(days, phibar_map[:, sp_idx], "-", color=col, linewidth=2.0,
                label="MAP" if sp_idx == 0 and row_idx == 0 else None)

        # Experimental data
        ax.plot(days, data_frac[:, sp_idx], "o", color=col,
                markersize=5.5, markeredgecolor="k", markeredgewidth=0.5, zorder=5)

        # Formatting
        ax.set_xlim(-0.5, 22.5)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 5, 10, 15, 21])

        if row_idx == 0:
            ax.set_title(SPECIES_ITALIC[sp_idx], fontsize=14, pad=8)
        if row_idx == 3:
            ax.set_xlabel("Day")
        else:
            ax.set_xticklabels([])

        if sp_idx == 0:
            ax.set_ylabel(r"$\bar{\varphi}_i$ (species fraction)")
        else:
            ax.set_yticklabels([])

        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.tick_params(direction="in", length=3)

    # Row label (left side)
    axes[row_idx, 0].annotate(
        f"{label}\n{COND_FULL[label]}",
        xy=(-0.45, 0.5), xycoords="axes fraction",
        fontsize=13, fontweight="bold", ha="center", va="center",
        rotation=90, linespacing=1.4)

    # RMSE + logL (right side)
    axes[row_idx, 4].annotate(
        "RMSE {:.3f}\nlogL {:.1f}".format(rmse, logL),
        xy=(0.97, 0.95), xycoords="axes fraction",
        fontsize=11, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="#cccccc", linewidth=0.5))

# Legend in top-left panel
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], color="gray", linewidth=2, label="MAP"),
    Patch(facecolor="gray", alpha=0.25, label="50% CI"),
    Patch(facecolor="gray", alpha=0.12, label="80% CI"),
    Line2D([0], [0], marker="o", color="gray", markersize=5,
           markeredgecolor="k", linewidth=0, label="Exp. data"),
]
axes[0, 0].legend(handles=legend_elements, loc="upper left",
                   fontsize=10, framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout(h_pad=0.4, w_pad=0.15)
plt.subplots_adjust(left=0.06)

outpng = base / "figures_fixpsi" / "paper_fig2_final.png"
outpdf = base / "figures_fixpsi" / "paper_fig2_final.pdf"
fig.savefig(outpng, dpi=300, bbox_inches="tight")
fig.savefig(outpdf, bbox_inches="tight")
print(f"Saved: {outpng}")
print(f"Saved: {outpdf}")
