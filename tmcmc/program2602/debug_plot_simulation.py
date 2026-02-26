import sys
from pathlib import Path

# Add project root to path (one level up from this file)
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import json
from tmcmc.improved_5species_jit import BiofilmNewtonSolver5S
from tmcmc.visualization.helpers import compute_phibar


def main():
    base_dir = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc")
    run_dir = base_dir / "_runs/manual_M2_high_precision"

    # Load Data and Params
    data = np.load(run_dir / "data.npy")
    with open(run_dir / "theta_MAP.json", "r") as f:
        params = json.load(f)
    theta_true = np.array(params["theta_full"])

    print(f"Loaded Data: {data.shape}")
    print(f"Loaded Theta: {theta_true.shape}")

    # Setup Solver (5 Species)
    solver = BiofilmNewtonSolver5S(
        dt=1e-5,
        maxtimestep=5000,
        active_species=[0, 1, 2, 3, 4],  # FULL MODEL
        phi_init=0.02,
        c_const=25.0,
        alpha_const=0.0,
    )

    # Run Simulation
    print("Running simulation (5 species)...")
    t, x = solver.solve(theta_true)
    print(f"Simulation done. x shape: {x.shape}")

    # Compute Observables
    phibar = compute_phibar(x, [0, 1, 2, 3, 4])

    # Print Debug Info
    print("\n--- Debug Info ---")
    print(f"Time (t) first 5: {t[:5]}")
    print(f"Data (t=0): {data[0]}")
    print(f"Sim (t=0): {phibar[0]}")
    print(f"Data (t=10*dt_sim = dt_data): {data[1]}")
    print(f"Sim (t=10): {phibar[10]}")

    # Plot
    plt.figure(figsize=(12, 8))

    # Plot Simulation Lines
    colors = ["b", "g", "r", "c", "m"]
    labels = ["S1", "S2", "S3", "S4", "S5"]

    # Interpolate simulation to data points for RMSE
    # Data is 501 points. Simulation is 5001 points.
    # We take every 10th point from simulation.
    idx_sparse = np.arange(0, len(t), 10)
    if len(idx_sparse) > len(data):
        idx_sparse = idx_sparse[: len(data)]

    # Plot Data Points
    t_data = t[idx_sparse]
    for i in range(5):
        plt.scatter(
            t_data, data[:, i], color=colors[i], marker="o", alpha=0.3, label=f"Data {labels[i]}"
        )

    # Plot Simulation
    for i in range(5):
        plt.plot(
            t, phibar[:, i], color=colors[i], linestyle="-", linewidth=2, label=f"Sim {labels[i]}"
        )

    plt.legend()
    plt.title("Debug Plot: 5-Species Simulation vs Data")
    plt.xlabel("Time")
    plt.ylabel("Active Biomass (phi*psi)")
    plt.grid(True)

    output_path = base_dir / "debug_plot_5species.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    # Compute RMSE
    sim_sparse = phibar[idx_sparse]
    rmse = np.sqrt(np.mean((sim_sparse - data[: len(idx_sparse)]) ** 2))
    print(f"RMSE: {rmse}")


if __name__ == "__main__":
    main()
