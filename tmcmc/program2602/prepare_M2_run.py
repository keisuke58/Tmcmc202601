
import numpy as np
import json
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))
from config import MODEL_CONFIGS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="manual_M2_high_precision", help="Run ID for output directory")
    args = parser.parse_args()

    # 1. Define Correct True Parameters (from improved_5species_jit.py)
    theta_true = [
        0.8, 2.0, 1.0, 0.1, 0.2,  # M1: a11, a12, a22, b1, b2
        1.5, 1.0, 2.0, 0.3, 0.4,  # M2: a33, a34, a44, b3, b4
        2.0, 1.0, 2.0, 1.0,       # M3: a13, a14, a23, a24
        1.2, 0.25,                # M4: a55, b5
        1.0, 1.0, 1.0, 1.0        # M5: a15, a25, a35, a45
    ]

    # 2. Setup Output Directory
    base_dir = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc")
    output_dir = base_dir / "_runs" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing run in: {output_dir}")

    # 3. Save theta_MAP.json (True values for locking)
    theta_map_data = {
        "theta_full": theta_true,
        "model": "M2_setup",
        "note": "Created by prepare_M2_run.py with True Values"
    }
    with open(output_dir / "theta_MAP.json", "w") as f:
        json.dump(theta_map_data, f, indent=2)
    print(f"Saved theta_MAP.json")

    # 4. Prepare Data and Indices
    # Load Ground Truth
    data_dir = base_dir / "data_5species"
    g_arr = np.load(data_dir / "g_arr.npy")
    t_arr = np.load(data_dir / "t_arr.npy")
    
    # M2 Configuration (from config.py)
    config = MODEL_CONFIGS["M2"]
    dt_sim = config["dt"]
    steps_sim = config["maxtimestep"]
    T_end_sim = dt_sim * steps_sim
    
    print(f"Config: dt={dt_sim}, steps={steps_sim}, T_end={T_end_sim:.4f}")

    # Data Configuration
    dt_data = 1e-4
    
    # We need to slice data to match T_end_sim
    n_data_points = int(T_end_sim / dt_data) + 1 
    
    # Check bounds
    if n_data_points > len(t_arr):
        print(f"WARNING: Requested data points {n_data_points} > Available {len(t_arr)}. Clipping.")
        n_data_points = len(t_arr)
        steps_sim = int(t_arr[-1] / dt_sim) # Adjust sim steps to match data end
        print(f"Adjusted steps_sim to {steps_sim}")

    print(f"Using {n_data_points} data points (T=0 to {t_arr[n_data_points-1]:.4f})")

    # Extract Observables (phi1..phi5 * psi)
    raw_data_subset = g_arr[:n_data_points, :]
    obs_data = np.zeros((n_data_points, 5))
    for i in range(5):
        # phi_i * psi_i
        obs_data[:, i] = raw_data_subset[:, i] * raw_data_subset[:, 6+i]

    # Add Noise
    sigma_obs = 0.001
    np.random.seed(42)
    noise = np.random.normal(0, sigma_obs, obs_data.shape)
    obs_data_noisy = obs_data + noise

    # Save Data
    np.save(output_dir / "data.npy", obs_data_noisy)
    print(f"Saved data.npy shape: {obs_data_noisy.shape}")

    # 5. Create Indices (idx_M2.npy)
    # Mapping: Simulation step -> Data index
    # We want indices of simulation that correspond to data points
    # Sim: 0, 1, ... steps_sim
    # Data[k] corresponds to t = k * dt_data
    # Sim[j] corresponds to t = j * dt_sim
    # j = k * (dt_data / dt_sim)
    
    ratio = int(round(dt_data / dt_sim))
    idx_m2 = np.arange(0, steps_sim + 1, ratio)
    
    # Ensure length match
    if len(idx_m2) > n_data_points:
        idx_m2 = idx_m2[:n_data_points]
    elif len(idx_m2) < n_data_points:
        # If simulation is slightly shorter due to rounding?
        # Just clip data
        print(f"Trimming data from {n_data_points} to {len(idx_m2)} to match indices")
        obs_data_noisy = obs_data_noisy[:len(idx_m2)]
        np.save(output_dir / "data.npy", obs_data_noisy)

    np.save(output_dir / "idx_M2.npy", idx_m2)
    print(f"Saved idx_M2.npy shape: {idx_m2.shape}")
    print(f"Index content sample: {idx_m2[:10]} ... {idx_m2[-10:]}")

if __name__ == "__main__":
    main()
