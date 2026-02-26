import numpy as np
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tmcmc.improved_5species_jit import BiofilmNewtonSolver5S, get_theta_true
from tmcmc.config import MODEL_CONFIGS


def main():
    # Setup
    run_id = "20260123_134436_debug_seed42"
    run_dir = Path("tmcmc/tmcmc/_runs") / run_id
    if not run_dir.exists():
        # Try absolute path if relative fails
        run_dir = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs") / run_id
        if not run_dir.exists():
            print(f"Run dir not found: {run_dir}")
            return

    print(f"Generating data in {run_dir}")

    # Load idx
    idx_file = run_dir / "idx_M3.npy"
    if not idx_file.exists():
        print(f"Idx file not found: {idx_file}")
        return
    idx = np.load(idx_file)
    print(f"Loaded idx with {len(idx)} points")

    # Config (using M5 config)
    if "M5" not in MODEL_CONFIGS:
        print("Error: M5 config not found in MODEL_CONFIGS")
        return

    config = MODEL_CONFIGS["M5"]
    dt = config["dt"]
    maxtimestep = config["maxtimestep"]

    # Params
    theta_true = get_theta_true()
    print(f"True Theta: {theta_true}")

    # Run Simulation
    solver = BiofilmNewtonSolver5S(
        dt=dt,
        maxtimestep=maxtimestep,
        c_const=config["c_const"],
        alpha_const=config["alpha_const"],
        phi_init=config["phi_init"],
    )

    # active_mask: all 5 species active
    active_mask = np.array([1, 1, 1, 1, 1], dtype=np.int32)

    # Solve
    print("Solving forward model...")
    t_arr, g_arr = solver.solve(theta_true)

    # Extract Data (5 species)
    # g_arr columns 0-4 are phi1-phi5
    data_clean = g_arr[idx, 0:5]

    # Add Noise
    sigma_obs = 0.001
    np.random.seed(42)
    noise = np.random.normal(0, sigma_obs, data_clean.shape)
    data_noisy = data_clean + noise

    # Save
    np.save(run_dir / "data.npy", data_noisy)
    np.save(run_dir / "t_idx.npy", t_arr[idx])
    print(f"Saved data.npy shape: {data_noisy.shape}")

    # Create theta_MAP.json with true values for M1-M3
    # We create a theta_base with M1-M3 true, and M4-M5 defaults (or zeros, handled by script)
    theta_base = np.zeros(20)
    theta_base[:14] = theta_true[:14]

    # M4/M5 defaults in script are:
    # M4: 1.2, 0.25
    # M5: 1.0, 1.0, 1.0, 1.0
    # Let's explicitly set them to "bad" values to prove estimation works?
    # Or just let them be 0?
    # If 0, log-likelihood might be NaN if log(param) is used?
    # No, params are usually positive.
    # refine_5species_linearized.py fills defaults if loaded length < 20.
    # If I save full 20, I should provide safe start values.
    # I'll use the script's defaults as start values.
    theta_base[14:16] = [1.2, 0.25]
    theta_base[16:20] = [1.0, 1.0, 1.0, 1.0]

    map_data = {
        "theta_full": theta_base.tolist(),
        "active_indices": list(range(14)),  # Just metadata
    }
    with open(run_dir / "theta_MAP.json", "w") as f:
        json.dump(map_data, f, indent=2)
    print("Saved theta_MAP.json")


if __name__ == "__main__":
    main()
