#!/usr/bin/env python3
"""Paper Figure 2 — TRANSPOSED: columns=conditions, rows=species.
Large fonts matching 11pt body text. Legend outside."""
import sys, os
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

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
    "font.size": 19,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 20,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_MAP = {
    "CS": ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#B2182B"],
    "CH": ["#2166AC", "#1B7837", "#DAA520", "#7B3294", "#B2182B"],
    "DS": ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#B2182B"],
    "DH": ["#2166AC", "#1B7837", "#FF8C00", "#7B3294", "#B2182B"],
}

SPECIES_ITALIC = [
    r"$\it{S.\ oralis}$",
    r"$\it{A.\ naeslundii}$",
    r"$\it{V.\ dispar}$",
    r"$\it{F.\ nucleatum}$",
    r"$\it{P.\ gingivalis}$",
]

COND_FULL = {
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
    "DS": "Dysbiotic Static",
    "DH": "Dysbiotic HOBIC",
}

import glob
base = Path(__file__).parent / "_runs"
data_dir = Path(__file__).parent.parent

def find_best(cond, cult):
    candidates = []
    for pat in ["20260319", "20260320"]:
        candidates += [Path(d) for d in glob.glob(
            str(base / "jax_ode_nuts_{}_{}_{}_*".format(cond, cult, pat)))
            if os.path.exists(str(Path(d) / "config.json"))]
    best, best_score = None, 999
    for c in candidates:
        try:
            cfg = json.load(open(str(c / "config.json")))
            rmse = cfg.get("rmse", 999)
            np_ = cfg.get("n_particles", 0)
            score = rmse - (10.0 if np_ >= 10000 else 0.0)
            if score < best_score:
                best_score, best = score, c
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

sp_name_to_idx = {
    "S. oralis": 0, "A. naeslundii": 1,
    "V. dispar": 2, "V. parvula": 2,
    "F. nucleatum": 3,
    "P. gingivalis_W83": 4, "P. gingivalis_20709": 4,
}

# Precompute
all_data = {}
for label, cond, cult in panels:
    run = load_run(runs[label])
    cfg = run["config"]
    theta_map = run["theta_MAP"]
    days, data_frac, total_vol, data_abs, phi_init_abs = load_experiment_data(
        data_dir, cond, cult)
    phi_init_frac = data_frac[0, :]
    traj = simulate_ode_map(theta_map, cond, cult, phi_init=phi_init_frac)
    idx = convert_days_to_idx(days)
    phi_raw = traj[idx, 0:5]
    psi_raw = traj[idx, 6:11]
    phibar_map = phi_raw * psi_raw
    phibar_map = np.clip(phibar_map, 1e-10, 1.0)
    phibar_map = phibar_map / np.maximum(phibar_map.sum(axis=1, keepdims=True), 1e-12)
    print(f"  Posterior predictive for {label}...")
    posterior_pred = compute_posterior_predictive(
        run["samples"], phi_init_frac, cond, cult, days, n_max=100)
    cond_rep = rep_df[
        (rep_df["condition"] == cond) & (rep_df["cultivation"] == cult)]
    all_data[label] = {
        "days": days, "data_frac": data_frac, "phibar_map": phibar_map,
        "posterior_pred": posterior_pred, "cond_rep": cond_rep,
        "rmse": cfg.get("rmse", 0), "logL": cfg.get("max_logL", 0),
    }

# 5 rows (species) x 4 columns (conditions)
fig, axes = plt.subplots(5, 4, figsize=(15, 17), sharex=True, sharey=True)

for col_idx, (label, cond, cult) in enumerate(panels):
    d = all_data[label]
    colors = COLOR_MAP[label]
    days = d["days"]

    for sp_idx in range(5):
        ax = axes[sp_idx, col_idx]
        col = colors[sp_idx]

        # Replicate boxplots
        sp_names = [k for k, v in sp_name_to_idx.items() if v == sp_idx]
        box_data, box_pos = [], []
        for day in days:
            vals = d["cond_rep"][
                (d["cond_rep"]["day"] == day) & (d["cond_rep"]["species"].isin(sp_names))
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
        sp_samples = d["posterior_pred"][:, :, sp_idx]
        q10 = np.percentile(sp_samples, 10, axis=0)
        q25 = np.percentile(sp_samples, 25, axis=0)
        q75 = np.percentile(sp_samples, 75, axis=0)
        q90 = np.percentile(sp_samples, 90, axis=0)
        ax.fill_between(days, q10, q90, alpha=0.12, color=col)
        ax.fill_between(days, q25, q75, alpha=0.25, color=col)

        # MAP line
        ax.plot(days, d["phibar_map"][:, sp_idx], "-", color=col, linewidth=2.0)

        # Experimental data
        ax.plot(days, d["data_frac"][:, sp_idx], "o", color=col,
                markersize=6, markeredgecolor="k", markeredgewidth=0.5, zorder=5)

        ax.set_xlim(-0.5, 22.5)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 3, 6, 10, 15, 21])
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.tick_params(direction="in", length=4)

        # Column title (top row only): condition label + RMSE/logL
        if sp_idx == 0:
            ax.set_title(
                f"{label} — {COND_FULL[label]}\n"
                f"RMSE = {d['rmse']:.3f},  logL = {d['logL']:.1f}",
                fontsize=18, fontweight="bold", pad=12)

        # Row label (left column): species name as ylabel
        if col_idx == 0:
            ax.set_ylabel(
                SPECIES_ITALIC[sp_idx] + r"    $\bar{\varphi}_i$",
                fontsize=17)

        # X-axis: "Day" on ALL bottom row panels
        if sp_idx == 4:
            ax.set_xlabel("Day", fontsize=18)

        # Y tick labels only on left column
        if col_idx > 0:
            ax.set_yticklabels([])

# Legend outside (top center)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], color="#2166AC", linewidth=2.5, label="MAP"),
    Patch(facecolor="#2166AC", alpha=0.35, label="50% CI"),
    Patch(facecolor="#2166AC", alpha=0.15, label="80% CI"),
    Line2D([0], [0], marker="o", color="#2166AC", markersize=9,
           markeredgecolor="k", linewidth=0, label="Exp. data"),
    Patch(facecolor="#2166AC", alpha=0.25, edgecolor="#2166AC", linewidth=0.8, label="Replicates (IQR)"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=5,
           fontsize=20, framealpha=0.95, edgecolor="#999999",
           bbox_to_anchor=(0.5, 1.03), handlelength=2.0, handletextpad=0.5)

plt.tight_layout(h_pad=0.4, w_pad=0.3, rect=[0, 0, 1, 0.97])

outpdf = base / "figures_fixpsi" / "paper_fig2_transposed.pdf"
outpng = base / "figures_fixpsi" / "paper_fig2_transposed.png"
fig.savefig(outpdf, bbox_inches="tight")
fig.savefig(outpng, dpi=300, bbox_inches="tight")
print(f"Saved: {outpdf}")
print(f"Saved: {outpng}")
