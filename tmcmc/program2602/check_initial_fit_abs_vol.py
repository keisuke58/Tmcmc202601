
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Ensure we can import from tmcmc
sys.path.append(str(Path.cwd() / "tmcmc"))

from improved_5species_jit import get_theta_true, BiofilmNewtonSolver5S
from run_M4_joint_estimation_abs_vol import convert_csv_to_npy, VOLUME_SCALE_FACTOR

# Configuration
TEMP_DIR = Path("temp_check_fit")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir()

print(f"=== 1. Converting Data (Scale Factor: {VOLUME_SCALE_FACTOR}) ===")
convert_csv_to_npy(TEMP_DIR)

print("\n=== 2. Loading Data ===")
exp_data = np.load(TEMP_DIR / "data.npy") # Shape: (N_days, 5)
t_idx = np.load(TEMP_DIR / "t_idx.npy")   # Shape: (N_days,)
days = np.load(TEMP_DIR / "t_obs.npy")    # Shape: (N_days,)

print(f"Exp Data Shape: {exp_data.shape}")
print(f"Time Indices: {t_idx}")

print("\n=== 3. Running Simulation with Initial Parameters ===")
# Parameters from generate_5species_data.py
# dt=1e-4, maxtimestep=750, c_const=25.0, phi_init=0.02
solver = BiofilmNewtonSolver5S(
    dt=1e-4,
    maxtimestep=750,
    active_species=[0, 1, 2, 3, 4],
    c_const=25.0,
    alpha_const=0.0, # As per generation script
    phi_init=0.02
)

theta_true = get_theta_true()
print(f"Theta True: {theta_true}")

t_arr, g_arr = solver.solve(theta_true)
# g_arr shape: (751, 12)

print("\n=== 4. Comparison & Plotting ===")

# Simulation Absolute Volume
# phi (0-4) * gamma (11)
phi_sim = g_arr[:, 0:5]
gamma_sim = g_arr[:, 11:12]
abs_vol_sim = phi_sim * gamma_sim

# Extract at observed times
sim_at_obs = abs_vol_sim[t_idx]

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

species_names = ["S1 (S. gordonii)", "S2 (A. naeslundii)", "S3 (Veillonella)", "S4 (F. nucleatum)", "S5 (P. gingivalis)"]

# Plot per species
for i in range(5):
    ax = axes[i]
    # Simulation trajectory
    ax.plot(t_arr, abs_vol_sim[:, i], label='Sim (Initial)', color='blue', alpha=0.7)
    # Experimental points
    ax.scatter(days * 1e-4 * (750/days.max()) if days.max() > 0 else 0, # Map days to time? No, plot against indices or map time?
               # t_arr is time (0 to 0.075). days are 1, 3, 7.
               # t_idx maps days to steps. t_arr[t_idx] is the time.
               exp_data[:, i], label='Exp Data (Scaled)', color='red', marker='o')
    
    # Correct x-axis for Scatter:
    # We should plot against Simulation Time
    sim_times_at_obs = t_arr[t_idx]
    ax.scatter(sim_times_at_obs, exp_data[:, i], color='red', zorder=5)

    ax.set_title(species_names[i])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute Volume")
    ax.legend()
    ax.grid(True)

# Plot Total Volume (Sum of Absolute Volumes)
ax = axes[5]
total_vol_sim = np.sum(abs_vol_sim, axis=1)
total_vol_exp = np.sum(exp_data, axis=1)

ax.plot(t_arr, total_vol_sim, label='Sim Total', color='black', linestyle='--')
ax.scatter(sim_times_at_obs, total_vol_exp, label='Exp Total', color='green', marker='s')
ax.set_title("Total Volume (Gamma approx)")
ax.set_xlabel("Time (s)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plot_path = TEMP_DIR / "initial_fit.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# Quantitative Check
print("\n=== Metrics ===")
for i in range(5):
    max_sim = np.max(abs_vol_sim[:, i])
    max_exp = np.max(exp_data[:, i])
    print(f"Species {i}: Max Sim={max_sim:.4f}, Max Exp={max_exp:.4f}, Ratio={max_sim/max_exp if max_exp>0 else 0:.2f}")

total_sim_max = np.max(total_vol_sim)
total_exp_max = np.max(total_vol_exp)
print(f"Total Vol: Max Sim={total_sim_max:.4f}, Max Exp={total_exp_max:.4f}, Ratio={total_sim_max/total_exp_max if total_exp_max>0 else 0:.2f}")

print("\nDone.")
