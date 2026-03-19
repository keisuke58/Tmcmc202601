#!/usr/bin/env python3
"""Plot 5-panel species fit for DH MAP estimate."""
import sys
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimate_reduced_nishioka import load_experimental_data, convert_days_to_model_time

# Load experimental data
DATA_DIR = Path(__file__).resolve().parent.parent.parent  # data_5species/
data, t_days, sigma_obs_est, phi_init_exp, metadata = load_experimental_data(
    DATA_DIR, "Dysbiotic", "HOBIC", start_from_day=1, normalize=True
)

# Load MAP theta
run_dir = Path(__file__).resolve().parent / "jax_ode_nuts_Dysbiotic_HOBIC_20260319_170643"
with open(run_dir / "theta_MAP.json") as f:
    theta_map = json.load(f)
theta = np.array([theta_map[str(i)] for i in range(20)])

# Run ODE (numpy version via core)
sys.path.insert(0, str(DATA_DIR / "main"))
from hamilton_ode_jax import theta_to_matrices

A, b = theta_to_matrices(theta)
A = np.array(A)
b = np.array(b)

# Simple Euler ODE
dt = 1e-4
n_steps = 2500
phi = phi_init_exp / phi_init_exp.sum()
phi = np.clip(phi, 0.001, 0.99)
K_hill = 0.05
n_hill = 4.0

traj = np.zeros((n_steps + 1, 5))
traj[0] = phi.copy()
for i in range(n_steps):
    # Hill gate for Pg (species 4)
    hill = phi[4] ** n_hill / (K_hill**n_hill + phi[4] ** n_hill)
    A_eff = A.copy()
    A_eff[:, 4] *= hill
    A_eff[4, :] *= hill

    dphi = A_eff @ phi + b
    phi_new = phi + dt * dphi
    phi_new = np.clip(phi_new, 1e-12, None)
    phi_new = phi_new / phi_new.sum()
    phi = phi_new
    traj[i + 1] = phi

# Get model predictions at observation times
t_model, idx_sparse = convert_days_to_model_time(t_days, dt, n_steps, day_scale=None)
idx_sparse = np.clip(idx_sparse, 0, n_steps)
phi_pred = traj[idx_sparse]
phi_pred = phi_pred / phi_pred.sum(axis=1, keepdims=True)

# Dense trajectory for smooth curves
days_dense = np.linspace(0, 21, 200)
day_scale = (n_steps * dt * 0.95) / 21.0
idx_dense = np.clip((days_dense * day_scale / dt).astype(int), 0, n_steps)
phi_dense = traj[idx_dense]
phi_dense = phi_dense / phi_dense.sum(axis=1, keepdims=True)

# Plot
species = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)
fig.suptitle("DH (Dysbiotic HOBIC) — GPU TMCMC MAP fit (RMSE=0.062, R²=0.54)", fontsize=13, y=1.02)

for i, (ax, sp, col) in enumerate(zip(axes, species, colors)):
    # Dense trajectory
    ax.plot(days_dense, phi_dense[:, i], color=col, lw=2, label="MAP prediction")
    # Experimental data with error bars
    sigma_i = (
        sigma_obs_est[i]
        if hasattr(sigma_obs_est, "__len__") and len(sigma_obs_est) == 5
        else sigma_obs_est
    )
    ax.errorbar(
        t_days,
        data[:, i],
        yerr=float(sigma_i) * 0.5,
        fmt="o",
        color="k",
        markersize=6,
        capsize=3,
        label="Heine 2025",
    )
    # Per-species R²
    ss_res = np.sum((data[:, i] - phi_pred[:, i]) ** 2)
    ss_tot = np.sum((data[:, i] - np.mean(data[:, i])) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    ax.set_title(f"{sp}\nR²={r2:.2f}", fontsize=10)
    ax.set_xlabel("Day")
    if i == 0:
        ax.set_ylabel("Species fraction φ")
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

plt.tight_layout()
out = run_dir / "dh_5panel_fit.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
