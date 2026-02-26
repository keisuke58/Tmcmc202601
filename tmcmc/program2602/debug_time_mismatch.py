import numpy as np
import sys
from pathlib import Path
import json

# Set paths
RUN_DIR = Path(
    "/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs/parallel_fixed_M1M2M3_20260126_210657"
)
ROOT_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc")
sys.path.append(str(ROOT_DIR))

# Import project modules
from config import MODEL_CONFIGS
from improved1207_paper_jit import BiofilmNewtonSolver, BiofilmTSM, HAS_NUMBA
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical


def check_samples():
    print("Loading data...")
    t_M1 = np.load(RUN_DIR / "t_M1.npy")

    results_path = RUN_DIR / "results_MAP_linearization.npz"
    if not results_path.exists():
        print("No results npz found")
        return

    data = np.load(results_path)
    samples_M1 = data["samples_M1"]
    print(f"Loaded {len(samples_M1)} samples")

    # Load base theta
    with open(RUN_DIR / "theta_MAP_M1.json", "r") as f:
        theta_map_data = json.load(f)
    theta_base = np.array(theta_map_data["theta_prior_mean"])  # Or appropriate base
    # Actually, we should construct full theta from samples.
    # The config has active indices.

    config = MODEL_CONFIGS["M1"]
    active_indices = config["active_indices"]

    # Setup Solver
    solver_kwargs = {
        k: v
        for k, v in config.items()
        if k not in ["active_species", "active_indices", "param_names"]
    }
    solver = BiofilmNewtonSolver(
        **solver_kwargs,
        active_species=config["active_species"],
        use_numba=HAS_NUMBA,
    )
    tsm = BiofilmTSM_Analytical(
        solver,
        active_theta_indices=active_indices,
        use_complex_step=False,
        use_analytical=False,
    )

    # Use theta_prior_mean as base? Or theta_MAP?
    # Usually we start with a base vector and update active indices.
    # Let's use the one from json if available, or just zeros and update.
    # The code usually uses `theta_base_M1`.
    # Let's assume theta_base is what was used.
    # We can reconstruct it from theta_full in json if we know which are active.
    theta_full_map = np.array(theta_map_data["theta_full"])

    mismatches = 0
    print("Checking first 20 samples...")
    for i in range(min(20, len(samples_M1))):
        theta_sample = samples_M1[i]
        theta_curr = theta_full_map.copy()
        theta_curr[active_indices] = theta_sample

        t_arr, _, _ = tsm.solve_tsm(theta_curr)

        if len(t_arr) != len(t_M1) or not np.allclose(t_arr, t_M1):
            diff = np.max(np.abs(t_arr - t_M1)) if len(t_arr) == len(t_M1) else float("inf")
            print(
                f"Sample {i}: Mismatch! len(t_arr)={len(t_arr)}, len(t_M1)={len(t_M1)}, max_diff={diff}"
            )
            mismatches += 1
        else:
            pass
            # print(f"Sample {i}: Match")

    if mismatches > 0:
        with open("debug_result.txt", "w") as f:
            f.write(f"Found {mismatches} mismatches in checked samples.\n")
        print(f"Found {mismatches} mismatches in checked samples.")
    else:
        with open("debug_result.txt", "w") as f:
            f.write("No mismatches found in checked samples.\n")
        print("No mismatches found in checked samples.")


if __name__ == "__main__":
    try:
        check_samples()
    except Exception as e:
        with open("debug_error.txt", "w") as f:
            f.write(str(e))
        print(e)
