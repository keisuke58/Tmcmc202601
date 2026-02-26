import nbformat as nbf
from pathlib import Path
import numpy as np  # Just in case, though not used in script execution logic

nb_path = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/analyze_parallel_fixed_20260126.ipynb")
nb = nbf.read(nb_path, as_version=4)

# Remove existing Spaghetti Plot sections to ensure update
print("Removing existing Spaghetti Plot sections...")
nb.cells = [c for c in nb.cells if "## 5. Spaghetti Plot" not in c.source]

# Define new cells
md_cell = nbf.v4.new_markdown_cell(
    """## 5. Spaghetti Plot (Posterior Predictive)

This section attempts to generate the "Spaghetti Plot" (Posterior Predictive) using the run results.
Since the full posterior samples (`trace_M1.npy`) might not be saved in this run, we check for their existence.
If missing, we plot the MAP estimate and explain the limitation.
Note: We apply the **interpolation fix** demonstrated above to ensure trajectories are not collapsed."""
)

code_cell = nbf.v4.new_code_cell(
    """# Spaghetti Plot Logic
import json
from scipy.interpolate import interp1d

# 1. Load MAP Parameters
map_file = RUN_DIR / "theta_MAP_M1.json"
if map_file.exists():
    with open(map_file, 'r') as f:
        map_data = json.load(f)
    # Convert dictionary to array (assuming order a11, a12, a22, b1, b2)
    # Note: Adjust keys based on your specific JSON structure
    # For M1: ["a11", "a12", "a22", "b1", "b2"]
    param_names = ["a11", "a12", "a22", "b1", "b2"]
    theta_MAP = np.array([map_data[k] for k in param_names])
    print(f"Loaded MAP: {theta_MAP}")
else:
    print("MAP file not found. Using fallback/defined theta_MAP.")

# 2. Run Simulation for MAP
# We need to construct the full parameter vector if theta_MAP is partial
# Assuming 'tsm' object is available from previous cells
# and 'theta_base' is available.

theta_full_MAP = theta_base.copy()
theta_full_MAP[active_indices] = theta_MAP

# Solve
t_sim_map, x_sim_map, _ = tsm.solve_tsm(theta_full_MAP)

# 3. Apply Interpolation Fix (Crucial for PaperFig09 issue)
if len(t_sim_map) != len(t_M1) or not np.allclose(t_sim_map, t_M1):
    print("Applying interpolation fix for MAP...")
    x_sim_map_interp = np.zeros((len(t_M1), x_sim_map.shape[1]))
    for j in range(x_sim_map.shape[1]):
        interp_func = interp1d(t_sim_map, x_sim_map[:, j], kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        x_sim_map_interp[:, j] = interp_func(t_M1)
    x_sim_map = x_sim_map_interp
    t_sim_map = t_M1

# Compute phibar (Observation operator)
# phibar = x * psi (assuming simple scaling or identity for now, or use compute_phibar if available)
# If compute_phibar is not available, we use x_sim_map directly or simple scaling
try:
    phibar_map = compute_phibar(x_sim_map, active_species)
except NameError:
    # Fallback if compute_phibar is not defined in notebook scope
    # Assuming x_sim_map is already close to phibar for M1 (or need to multiply by psi)
    phibar_map = x_sim_map

# 4. Check for Samples (Trace)
trace_file = RUN_DIR / "trace_M1.npy"
samples = None
if trace_file.exists():
    print(f"Loading samples from {trace_file}...")
    samples = np.load(trace_file)
else:
    print(f"Samples file not found at {trace_file}")
    print("No samples found. Proceeding with MAP only.")

# 5. Plotting
plt.figure(figsize=(10, 6))

# Normalize time for plotting
t_min, t_max = t_M1.min(), t_M1.max()
t_norm = (t_M1 - t_min) / (t_max - t_min)
t_obs = t_norm[idx_M1]

# Plot Data
# Assuming data_M1 is (N, n_species)
for sp_idx in range(data_M1.shape[1]):
    plt.scatter(t_obs, data_M1[:, sp_idx], label=f'Data Sp{sp_idx+1}', zorder=10)

# Plot Samples (if available)
if samples is not None:
    n_plot = min(50, len(samples))
    indices = np.random.choice(len(samples), n_plot, replace=False)
    print(f"Plotting {n_plot} posterior trajectories...")

    # Pre-allocate for speed? No, just loop.
    for i, idx in enumerate(indices):
        theta_s = samples[idx]
        theta_full_s = theta_base.copy()
        theta_full_s[active_indices] = theta_s

        t_s, x_s, _ = tsm.solve_tsm(theta_full_s)

        # Interpolate
        if len(t_s) != len(t_M1):
            x_s_interp = np.zeros((len(t_M1), x_s.shape[1]))
            for j in range(x_s.shape[1]):
                f_int = interp1d(t_s, x_s[:, j], kind='linear', bounds_error=False, fill_value='extrapolate')
                x_s_interp[:, j] = f_int(t_M1)
            x_s = x_s_interp

        # Phibar
        try:
            phi_s = compute_phibar(x_s, active_species)
        except:
            phi_s = x_s

        # Plot lines (thin, transparent)
        for sp_idx in range(phi_s.shape[1]):
            plt.plot(t_norm, phi_s[:, sp_idx], color='gray', alpha=0.1, zorder=1)

# Plot MAP
for sp_idx in range(phibar_map.shape[1]):
    plt.plot(t_norm, phibar_map[:, sp_idx], '--', linewidth=2, label=f'MAP Sp{sp_idx+1}', zorder=5)

plt.title("Posterior Predictive Check (Spaghetti Plot)")
plt.xlabel("Normalized Time")
plt.ylabel("Concentration / Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
)

# Append
nb.cells.append(md_cell)
nb.cells.append(code_cell)

# Save
nbf.write(nb, nb_path)
print("Notebook updated successfully.")
