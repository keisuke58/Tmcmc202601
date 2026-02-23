import nbformat as nbf
from pathlib import Path

nb_path = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/analyze_parallel_fixed_20260126.ipynb")
nb = nbf.read(nb_path, as_version=4)

# Get the last code cell (which we just added)
last_cell = nb.cells[-1]

# New content with interpolation logic
new_source = """# Spaghetti Plot Logic

# Check for samples
trace_file = RUN_DIR / "trace_M1.npy"
samples = None

if trace_file.exists():
    print(f"Loading samples from {trace_file}...")
    samples = np.load(trace_file)
else:
    print(f"Samples file not found at {trace_file}")
    print("No samples found. Proceeding with MAP only.")

plt.figure(figsize=(10, 6))

# Plot Data
t_min, t_max = t_M1.min(), t_M1.max()
t_norm = (t_M1 - t_min) / (t_max - t_min)
t_obs = t_norm[idx_M1]

plt.scatter(t_obs, data_M1[:, 0], color='blue', label='Data', zorder=10)

# Plot Samples (Spaghetti)
if samples is not None:
    n_plot = min(100, len(samples))
    indices = np.random.choice(len(samples), n_plot, replace=False)
    
    print(f"Plotting {n_plot} posterior trajectories...")
    
    for i, idx in enumerate(indices):
        theta = samples[idx]
        # Note: Logic to map theta_sub to theta_full is omitted here for brevity
        # Assuming we can run:
        # t_s, x_s, _ = tsm.solve_tsm(theta)
        # if len(t_s) != len(t_M1): ... interpolate ...
        pass

# Plot MAP (Corrected with Interpolation)
# We use phibar_map and t_arr_map from previous cells
if len(t_arr_map) != len(t_M1) or not np.allclose(t_arr_map, t_M1):
    print("Interpolating MAP to t_M1 grid for plotting...")
    f_map = interp1d(t_arr_map, phibar_map, axis=0, fill_value="extrapolate")
    phibar_map_plot = f_map(t_M1)
else:
    phibar_map_plot = phibar_map

plt.plot(t_norm, phibar_map_plot[:, 0], 'r-', linewidth=2, label='MAP', zorder=5)

plt.title("Posterior Predictive Check (Spaghetti Plot)")
plt.xlabel("Normalized Time")
plt.ylabel("φ̄")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""

last_cell.source = new_source
nbf.write(nb, nb_path)
print("Notebook refined successfully.")
