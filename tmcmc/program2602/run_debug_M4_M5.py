import os
import shutil
import json
import numpy as np
import subprocess
import sys
import argparse
from pathlib import Path

# Config defaults
DEFAULT_RUN_ID = "debug_M4_M5_100p"
BASE_RUNS_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/_runs")
SOURCE_DIR = Path(
    "/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs/20260123_134436_debug_seed42"
)
SCRIPT_PATH = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/main/refine_5species_linearized.py")
N_PARTICLES = 500
SIGMA_OBS = 0.001
N_JOBS = 24


def parse_args():
    parser = argparse.ArgumentParser(description="Run M4/M5 estimation debug")
    parser.add_argument(
        "--run-id", type=str, default=DEFAULT_RUN_ID, help="Run ID for the output directory"
    )
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES, help="Number of particles")
    return parser.parse_args()


def setup_run_dir(run_id, n_particles):
    base_dir = BASE_RUNS_DIR / run_id

    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)

    # Copy files
    files_to_copy = ["data.npy", "idx_M3.npy", "idx_M4.npy", "idx_M5.npy", "t_idx.npy"]
    for f in files_to_copy:
        src = SOURCE_DIR / f
        dst = base_dir / f
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied {f}")
        else:
            print(f"Warning: {f} not found in source")

    # Create theta_MAP.json with M1-M3 fixed to true values
    theta_true = np.array(
        [
            0.8,
            2.0,
            1.0,
            0.1,
            0.2,  # M1
            1.5,
            1.0,
            2.0,
            0.3,
            0.4,  # M2
            2.0,
            1.0,
            2.0,
            1.0,  # M3
            1.2,
            0.25,  # M4 (True)
            1.0,
            1.0,
            1.0,
            1.0,  # M5 (True)
        ]
    )

    # Initial guess for M4/M5 (different from true to test estimation)
    theta_init = theta_true.copy()
    # Overwrite M2 decay parameters (b3, b4) to 0.5 for stability - REMOVED for M4/M5 estimation with True M2
    # theta_init[8] = 0.5
    # theta_init[9] = 0.5

    # Use a more reasonable guess closer to Order of Magnitude 1.0
    # True: a55=1.2, b5=0.25
    # Guess: a55=1.0, b5=0.3
    theta_init[14] = 1.0  # a55
    theta_init[15] = 0.3  # b5
    theta_init[16:] = 1.0  # M5 defaults

    map_data = {
        "theta_full": theta_init.tolist(),
        "model": "M3_locked",
        "note": "Debug run for M4/M5 with M1-M3 fixed to true values",
    }

    with open(base_dir / "theta_MAP.json", "w") as f:
        json.dump(map_data, f, indent=4)
    print("Created theta_MAP.json")
    return base_dir


def run_estimation(stage, base_dir, run_id, n_particles):
    print(f"\n=== Running {stage} Estimation ===")

    log_file = base_dir / f"run_{stage}.log"

    cmd = [
        "python3",
        str(SCRIPT_PATH),
        "--run-id",
        run_id,
        "--stage",
        stage,
        "--mode",
        "stage",
        "--n-particles",
        str(n_particles),
        "--n-jobs",
        str(N_JOBS),
        "--n-stages",
        "20",
        "--n-chains",
        "1",
        "--refine-range",
        "3.0",
        "--sigma-obs",
        str(SIGMA_OBS),
        "--debug-level",
        "INFO",
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")

    # Run command with logging
    with open(log_file, "w") as f:
        ret = subprocess.run(
            cmd, cwd="/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc", stdout=f, stderr=subprocess.STDOUT
        )

    if ret.returncode != 0:
        print(f"Error running {stage} estimation. Check log at {log_file}")
        # Print last few lines of log
        print("Last 10 lines of log:")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.strip())
        except (OSError, IOError) as e:
            print(f"Could not read log: {e}")
        sys.exit(1)

    # After run, update theta_MAP.json for next stage
    refined_json_path = base_dir / "refined" / "theta_MAP_refined.json"
    if refined_json_path.exists():
        # Rename the refined output to avoid overwriting by next stage
        stage_json_path = base_dir / f"theta_MAP_{stage}_refined.json"
        shutil.copy(refined_json_path, stage_json_path)

        # Update main theta_MAP.json for next stage
        with open(refined_json_path, "r") as f:
            data = json.load(f)

        # Save as new theta_MAP.json
        with open(base_dir / "theta_MAP.json", "w") as f:
            json.dump(data, f, indent=4)
        print(f"Updated theta_MAP.json with {stage} results")
    else:
        print(f"Warning: {refined_json_path} not found")


def generate_extra_plots(base_dir):
    print("\n=== Generating Extra Plots ===")

    sys.path.append("/home/nishioka/IKM_Hiwi/Tmcmc202601")
    from tmcmc.visualization.plot_manager import PlotManager
    from tmcmc.improved_5species_jit import BiofilmNewtonSolver5S

    # Initialize PlotManager
    plot_mgr = PlotManager(str(base_dir / "refined"))

    # Load latest theta_MAP.json
    with open(base_dir / "theta_MAP.json", "r") as f:
        map_data = json.load(f)

    theta_full = np.array(map_data["theta_full"])

    # Load data
    data = np.load(base_dir / "data.npy")
    idx_sparse = np.load(base_dir / "idx_M5.npy")  # Use M5 index for full view

    # Run simulation with final parameters
    print("Running final simulation...")
    # Setup solver (using M5 config)
    solver = BiofilmNewtonSolver5S(
        dt=0.0001, maxtimestep=750, c_const=25.0, alpha_const=0.0, phi_init=0.02
    )

    t_fit, x_fit = solver.solve(theta_full)

    # Plot TSM
    active_species = [0, 1, 2, 3, 4]  # All 5 species
    plot_mgr.plot_TSM_simulation(
        t_fit, x_fit, active_species, "Final_M4_M5_Fit", data=data, idx_sparse=idx_sparse
    )
    print("Generated Final_M4_M5_Fit.png")

    print("Done generating plots.")


if __name__ == "__main__":
    args = parse_args()
    base_dir = setup_run_dir(args.run_id, args.n_particles)
    run_estimation("M4", base_dir, args.run_id, args.n_particles)
    run_estimation("M5", base_dir, args.run_id, args.n_particles)
    generate_extra_plots(base_dir)
