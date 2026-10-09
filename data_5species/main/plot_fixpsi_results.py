#!/usr/bin/env python3
"""Visualize fix-psi + narrow prior TMCMC results for all 4 conditions."""
import numpy as np
import json
import os
import glob
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Result directories
RUNS = {
    "DH": os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_HOBIC_20260319_224322"),
    "DS": os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_Static_20260319_232305"),
    "CS": os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Commensal_Static_20260319_231727"),
    "CH": os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Commensal_HOBIC_20260319_231640"),
}

# Override DH if newer narrow result exists
for d in sorted(glob.glob(os.path.expanduser(
        "~/Tmcmc202601/data_5species/main/_runs/jax_ode_nuts_Dysbiotic_HOBIC_20260319_23*"))):
    if os.path.exists(os.path.join(d, "config.json")):
        RUNS["DH"] = d

SPECIES = ["So", "An", "Vd", "Fn", "Pg"]
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
COND_MAP = {
    "DH": ("Dysbiotic", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
}
COND_LABELS = {
    "DH": "Dysbiotic HOBIC",
    "DS": "Dysbiotic Static",
    "CS": "Commensal Static",
    "CH": "Commensal HOBIC",
}

DATA_DIR = os.path.expanduser("~/Tmcmc202601/data_5species/experiment_data")

def load_exp_data(condition, cultivation):
    fname = os.path.join(DATA_DIR, "fig3_species_distribution_summary.csv")
    rows = []
    with open(fname) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["condition"] == condition and row["cultivation"] == cultivation:
                rows.append(row)
    days = sorted(set(int(r["day"]) for r in rows))
    data = np.zeros((len(days), 5))
    for r in rows:
        di = days.index(int(r["day"]))
        si = SPECIES.index(r["species"])
        data[di, si] = float(r["fraction"])
    return np.array(days), data

def hill(x, K=0.05, n=4.0):
    xn = np.abs(x)**n
    return xn / (K**n + xn)

def ode_rhs(phi, theta, K=0.05, n=4.0):
    """5-species replicator ODE."""
    a = np.zeros((5, 5))
    b = np.zeros(5)
    # M1: a11, a12, a22, b1, b2
    a[0,0] = theta[0]; a[0,1] = theta[1]; a[1,1] = theta[2]
    b[0] = theta[3]; b[1] = theta[4]
    # M2: a33, a34, a44, b3, b4
    a[2,2] = theta[5]; a[2,3] = theta[6]; a[3,3] = theta[7]
    b[2] = theta[8]; b[3] = theta[9]
    # M3: cross a13, a14, a23, a24
    a[0,2] = theta[10]; a[0,3] = theta[11]; a[1,2] = theta[12]; a[1,3] = theta[13]
    # M4: a55, b5
    a[4,4] = theta[14]; b[4] = theta[15]
    # M5: cross a15, a25, a35, a45
    a[0,4] = theta[16]; a[1,4] = theta[17]; a[2,4] = theta[18]; a[3,4] = theta[19]

    fitness = np.zeros(5)
    for i in range(5):
        f_i = b[i]
        for j in range(5):
            if a[i,j] != 0:
                f_i += a[i,j] * hill(phi[j], K, n)
        fitness[i] = f_i

    mean_fitness = np.dot(phi, fitness)
    dphi = phi * (fitness - mean_fitness)
    return dphi

def simulate(theta, ic, n_steps=2500, dt=1e-4):
    x = np.array(ic, dtype=np.float64)
    traj = [x.copy()]
    for _ in range(n_steps):
        dx = ode_rhs(x, theta)
        x = x + dt * dx
        x = np.clip(x, 0, None)
        s = x.sum()
        if s > 0:
            x = x / s  # normalize to simplex
        traj.append(x.copy())
    return np.array(traj)

# ============ MAIN ============
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
summary = []

for idx, label in enumerate(["DH", "DS", "CS", "CH"]):
    ax = axes[idx]
    run_dir = RUNS[label]
    cond, cult = COND_MAP[label]

    theta_path = os.path.join(run_dir, "theta_MAP.json")
    if not os.path.exists(theta_path):
        ax.set_title(f"{COND_LABELS[label]} — NO RESULTS YET")
        summary.append(None)
        continue

    theta_map = json.load(open(theta_path))
    theta = np.array([theta_map[str(i)] for i in range(20)])
    cfg = json.load(open(os.path.join(run_dir, "config.json")))

    days_exp, data_exp = load_exp_data(cond, cult)
    ic = data_exp[0]
    n_steps = 2500
    traj = simulate(theta, ic, n_steps)
    t_days = days_exp[0] + np.arange(n_steps + 1) * (days_exp[-1] - days_exp[0]) / n_steps

    # RMSE at experimental timepoints (skip IC)
    pred_at_exp = []
    for di in range(1, len(days_exp)):
        day = days_exp[di]
        idx_t = np.argmin(np.abs(t_days - day))
        pred_at_exp.append(traj[idx_t])
    pred_at_exp = np.array(pred_at_exp)
    obs = data_exp[1:]
    residuals = pred_at_exp - obs
    rmse_total = np.sqrt(np.mean(residuals**2))
    rmse_per = np.sqrt(np.mean(residuals**2, axis=0))

    # Plot
    for si, (sp, col) in enumerate(zip(SPECIES, COLORS)):
        ax.plot(t_days, traj[:, si], color=col, linewidth=2, label=sp)
        ax.scatter(days_exp, data_exp[:, si], color=col, s=80, zorder=5,
                   edgecolors="k", linewidth=0.8)

    ax.set_title("{}\nlogL={:.1f}  RMSE={:.4f}  stages={}".format(
        COND_LABELS[label], cfg.get("max_logL", 0), rmse_total, cfg.get("n_stages", 0)),
        fontsize=11, fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Species fraction")
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    summary.append({
        "condition": COND_LABELS[label],
        "label": label,
        "logL": cfg.get("max_logL", 0),
        "log_evidence": cfg.get("log_evidence", 0),
        "rmse_total": rmse_total,
        "rmse_per": {sp: float(r) for sp, r in zip(SPECIES, rmse_per)},
        "stages": cfg.get("n_stages", 0),
        "accept": cfg.get("mean_accept", 0),
    })

plt.suptitle("Fix-ψ + Narrow Prior TMCMC (1000 particles, RW)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
outpath = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/fixpsi_narrow_4cond.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print("Saved:", outpath)

# Summary table
print("\n" + "="*95)
print("{:>20s} {:>7s} {:>9s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>5s}".format(
    "Condition", "logL", "logZ", "RMSE", "So", "An", "Vd", "Fn", "Pg", "stg"))
print("="*95)
for s in summary:
    if s is None:
        continue
    rp = s["rmse_per"]
    print("{:>20s} {:>7.2f} {:>9.2f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f} {:>5d}".format(
        s["condition"], s["logL"], s["log_evidence"], s["rmse_total"],
        rp["So"], rp["An"], rp["Vd"], rp["Fn"], rp["Pg"], s["stages"]))

# RMSE bar chart
fig2, (ax_rmse, ax_sp) = plt.subplots(1, 2, figsize=(14, 5))

valid = [s for s in summary if s is not None]
labels = [s["label"] for s in valid]
rmses = [s["rmse_total"] for s in valid]
colors_bar = {"DH": "#d62728", "DS": "#ff7f0e", "CS": "#1f77b4", "CH": "#2ca02c"}

bars = ax_rmse.bar(labels, rmses, color=[colors_bar[l] for l in labels], edgecolor="k", linewidth=0.5)
for bar, val in zip(bars, rmses):
    ax_rmse.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
ax_rmse.set_ylabel("RMSE")
ax_rmse.set_title("Total RMSE by Condition", fontweight="bold")
ax_rmse.grid(axis="y", alpha=0.3)

# Per-species RMSE grouped
x = np.arange(5)
width = 0.2
for i, s in enumerate(valid):
    vals = [s["rmse_per"][sp] for sp in SPECIES]
    ax_sp.bar(x + i*width, vals, width, label=s["label"],
              color=colors_bar[s["label"]], edgecolor="k", linewidth=0.3)
ax_sp.set_xticks(x + width*1.5)
ax_sp.set_xticklabels(SPECIES)
ax_sp.set_ylabel("RMSE")
ax_sp.set_title("Per-Species RMSE", fontweight="bold")
ax_sp.legend()
ax_sp.grid(axis="y", alpha=0.3)

plt.tight_layout()
outpath2 = os.path.expanduser("~/Tmcmc202601/data_5species/main/_runs/fixpsi_narrow_rmse.png")
plt.savefig(outpath2, dpi=150, bbox_inches="tight")
print("Saved:", outpath2)
