import os
import shutil
import json
import numpy as np
import subprocess
import sys
import argparse
from pathlib import Path

# Config defaults
DEFAULT_RUN_ID = "manual_M4_M5_joint_estimation_1000p_true_init"
BASE_RUNS_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/_runs")
SOURCE_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/data_5species_sequential")
SCRIPT_PATH = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/main/refine_5species_linearized.py")
N_PARTICLES = 1000
SIGMA_OBS = 0.001
N_JOBS = 24
REFINE_RANGE = 0.5  # +/- 50% from initial (true) value

def parse_args():
    parser = argparse.ArgumentParser(description="Run M4/M5 JOINT estimation with True Init")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID, help="Run ID for the output directory")
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES, help="Number of particles")
    return parser.parse_args()

def setup_run_dir(run_id, n_particles):
    base_dir = BASE_RUNS_DIR / run_id

    if base_dir.exists():
        print(f"Directory {base_dir} exists. Cleaning up...")
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)
    
    # Copy Data Files
    files_to_copy = ["data_M5.npy", "idx_M3.npy", "t_M5.npy"]
    for f in files_to_copy:
        src = SOURCE_DIR / f
        dst = base_dir / f
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied {f}")
        else:
            print(f"Warning: {f} not found in source")
            
    # Rename data_M5.npy to data.npy and data_M4.npy for compatibility
    if (base_dir / "data_M5.npy").exists():
        shutil.copy(base_dir / "data_M5.npy", base_dir / "data.npy")
        shutil.copy(base_dir / "data_M5.npy", base_dir / "data_M4.npy")

    # Rename t_M5.npy to t_idx.npy
    if (base_dir / "t_M5.npy").exists():
        shutil.copy(base_dir / "t_M5.npy", base_dir / "t_idx.npy")

    # PROCESS DATA: Convert raw state (12 cols) to observables (5 cols)
    # Observable = phi_i * psi_i
    if (base_dir / "data.npy").exists():
        raw_data = np.load(base_dir / "data.npy")
        if raw_data.shape[1] == 12:
            print("Converting raw data (12 cols) to observables (5 cols)...")
            n_obs = raw_data.shape[0]
            obs_data = np.zeros((n_obs, 5))
            # phi indices: 0-4, psi indices: 6-10
            for i in range(5):
                obs_data[:, i] = raw_data[:, i] * raw_data[:, 6+i]
            
            np.save(base_dir / "data.npy", obs_data)
            np.save(base_dir / "data_M4.npy", obs_data)
            print(f"Saved processed data.npy with shape {obs_data.shape}")
        else:
            print(f"Data already has shape {raw_data.shape}, skipping conversion.")

    # Create TIME index file for M4 stage (idx_M4.npy)
    # This must match the data rows. Since we use full data (751 rows), indices are 0..750.
    if (base_dir / "data.npy").exists():
        data = np.load(base_dir / "data.npy")
        n_obs = data.shape[0]
        time_indices = np.arange(n_obs, dtype=int)
        np.save(base_dir / "idx_M4.npy", time_indices)
        print(f"Created idx_M4.npy with {n_obs} indices (0..{n_obs-1})")

    # Create theta_MAP.json with TRUE VALUES
    theta_true = np.array([
        0.8, 2.0, 1.0, 0.1, 0.2,   # M1
        1.5, 1.0, 2.0, 0.3, 0.4,   # M2
        2.0, 1.0, 2.0, 1.0,        # M3
        1.2, 0.25,                 # M4 (True)
        1.0, 1.0, 1.0, 1.0         # M5 (True)
    ])
    
    # We use True Values as initial center for refinement
    map_data = {
        "theta_full": theta_true.tolist(),
        "model": "M3_locked",
        "note": "Joint M4+M5 estimation initialized at True Values"
    }
    
    with open(base_dir / "theta_MAP.json", "w") as f:
        json.dump(map_data, f, indent=4)
    print("Created theta_MAP.json with True Values")
    
    return base_dir

def run_estimation(stage, base_dir, run_id, n_particles):
    print(f"\n=== Running {stage} (Joint M4+M5) Estimation ===")
    
    log_file = base_dir / f"run_{stage}.log"
    
    cmd = [
        "python3", str(SCRIPT_PATH),
        "--run-id", run_id,
        "--stage", stage,
        "--mode", "stage",  # Lock non-target parameters
        "--n-particles", str(n_particles),
        "--n-jobs", str(N_JOBS),
        "--n-stages", "30",
        "--n-chains", "1",
        "--refine-range", str(REFINE_RANGE), # +/- 50%
        "--sigma-obs", str(SIGMA_OBS),
        "--debug-level", "INFO"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")
    
    # Run command with logging
    with open(log_file, "w") as f:
        ret = subprocess.run(cmd, cwd="/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc", stdout=f, stderr=subprocess.STDOUT)
        
    if ret.returncode == 0:
        print(f"SUCCESS: {stage} estimation completed.")
    else:
        print(f"FAILED: {stage} estimation failed. Check log: {log_file}")
        sys.exit(1)

def main():
    args = parse_args()
    run_id = args.run_id
    n_particles = args.n_particles
    
    print(f"Starting Joint M4/M5 Estimation Run: {run_id}")
    print(f"Particles: {n_particles}, Range: +/- {REFINE_RANGE*100}%")
    
    base_dir = setup_run_dir(run_id, n_particles)
    
    # We run 'M4' stage, but we have hacked idx_M4.npy to include M5 parameters too.
    run_estimation("M4", base_dir, run_id, n_particles)
    
    print("\nAll done.")

if __name__ == "__main__":
    main()
