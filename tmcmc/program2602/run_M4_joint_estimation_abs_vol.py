import os
import shutil
import json
import numpy as np
import pandas as pd
import subprocess
import sys
import argparse
from pathlib import Path

# Config defaults
DEFAULT_RUN_ID = "abs_vol_analysis_v1"
BASE_RUNS_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/_runs")
CSV_SOURCE_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species")
SCRIPT_PATH = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/main/refine_5species_linearized.py")

# Estimation Parameters
N_PARTICLES = 1000
SIGMA_OBS = 2.0  # Scaled for absolute volume (approx 5-10% of max value ~30)
N_JOBS = 24
REFINE_RANGE = 0.5

# Data Conversion Parameters
# Scale factor to align Experimental Volume with Simulation Gamma
# Based on check_scale.py: Experimental Max ~0.64, Simulation Target ~30.0 -> Factor ~47.0
VOLUME_SCALE_FACTOR = 47.0 

# Simulation Time Parameters (Must match M4/M5 config)
DT = 1e-4
MAX_TIMESTEP = 750 # Steps
# T_MAX = 750 * 1e-4 = 0.075

def parse_args():
    parser = argparse.ArgumentParser(description="Run Estimation with Absolute Volume Data")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID, help="Run ID")
    parser.add_argument("--n-particles", type=int, default=N_PARTICLES, help="Number of particles")
    parser.add_argument("--convert-only", action="store_true", help="Only convert data, do not run estimation")
    return parser.parse_args()

def convert_csv_to_npy(base_dir):
    """
    Convert absolute_volume_analysis.ipynb compatible CSVs to TMCMC .npy format.
    Applies scale adjustment and formats for Absolute Volume (phi * gamma).
    """
    print("\n=== Converting Experimental Data to TMCMC Format ===")
    
    vol_file = CSV_SOURCE_DIR / 'biofilm_boxplot_data.csv'
    sp_file = CSV_SOURCE_DIR / 'species_distribution_data.csv'
    
    if not vol_file.exists() or not sp_file.exists():
        print(f"Error: Data files not found in {CSV_SOURCE_DIR}")
        sys.exit(1)
        
    df_vol = pd.read_csv(vol_file)
    df_sp = pd.read_csv(sp_file)
    
    # 1. Clean and Rename
    df_vol = df_vol[['condition', 'cultivation', 'day', 'median']].rename(columns={'median': 'total_volume'})
    df_sp = df_sp[['condition', 'cultivation', 'species', 'day', 'median']].rename(columns={'median': 'species_ratio'})
    
    # 2. Merge
    df_merged = pd.merge(df_sp, df_vol, on=['condition', 'cultivation', 'day'], how='left')
    
    # 3. Calculate Absolute Volume (Experiment Scale)
    # Ratio is %, so divide by 100
    df_merged['absolute_volume'] = df_merged['total_volume'] * (df_merged['species_ratio'] / 100.0)
    
    # 4. Filter for specific condition
    # Mapping based on analysis of reproduce_composition_figures.py and paper text:
    species_map_color = {
        'Blue': 0, 
        'Green': 1,
        'Yellow': 2,
        'Orange': 2, # Merge Veillonella strains
        'Purple': 3,
        'Red': 4
    }
    
    species_map_name = {
        'S. gordonii': 0, 'S1': 0, 'S. oralis': 0,
        'A. naeslundii': 1, 'S2': 1,
        'Veillonella': 2, 'V. dispar': 2, 'V. parvula': 2,
        'F. nucleatum': 3, 'S4': 3,
        'P. gingivalis': 4, 'S5': 4
    }
    
    species_map = {**species_map_color, **species_map_name}
    
    # Check species names in CSV
    unique_species = df_merged['species'].unique()
    print(f"Species found in CSV: {unique_species}")
    
    # Apply mapping
    df_merged['species_idx'] = df_merged['species'].map(species_map)
    
    # Check for unmapped species
    if df_merged['species_idx'].isnull().any():
        unmapped = df_merged[df_merged['species_idx'].isnull()]['species'].unique()
        print(f"Warning: Unmapped species found: {unmapped}. Dropping them.")
        df_merged = df_merged.dropna(subset=['species_idx'])
    
    df_merged['species_idx'] = df_merged['species_idx'].astype(int)
    
    # Filter Condition
    conds = df_merged['condition'].unique()
    target_cond = 'Dysbiotic' if 'Dysbiotic' in conds else conds[0]
    
    cults = df_merged['cultivation'].unique()
    target_cult = 'Static' if 'Static' in cults else cults[0]
    
    print(f"Using Condition: {target_cond}, Cultivation: {target_cult}")
    
    df_target = df_merged[(df_merged['condition'] == target_cond) & (df_merged['cultivation'] == target_cult)]
    
    # Aggregate if multiple species map to same index (e.g. Yellow+Orange -> S3)
    # Sum absolute volumes
    df_agg = df_target.groupby(['day', 'species_idx'])['absolute_volume'].sum().reset_index()
    
    # Pivot to shape (n_days, n_species)
    pivot_df = df_agg.pivot(index='day', columns='species_idx', values='absolute_volume')
    
    # Ensure all indices 0-4 exist
    for i in range(5):
        if i not in pivot_df.columns:
            print(f"Warning: Species Index {i} missing from data. Filling with 0.")
            pivot_df[i] = 0.0
            
    # Sort columns
    pivot_df = pivot_df.sort_index(axis=1)
    
    mapped_data = pivot_df.values
    days = pivot_df.index.values
    
    print(f"Experimental Days: {days}")

    # 5. Apply Scale Factor
    print(f"Applying Volume Scale Factor: {VOLUME_SCALE_FACTOR}")
    scaled_data = mapped_data * VOLUME_SCALE_FACTOR
    
    # 6. Save Data
    np.save(base_dir / "data.npy", scaled_data)
    np.save(base_dir / "data_M4.npy", scaled_data) # Use same data for M4 stage
    
    # 7. Generate idx_M4.npy (Time Index Mapping)
    # Map 'Day' to Simulation Steps.
    # Assumption: Max Day corresponds to MAX_TIMESTEP (End of simulation).
    # Linear mapping: step = (day / max_day) * MAX_TIMESTEP
    
    max_day = days.max()
    print(f"Max Experimental Day: {max_day}")
    print(f"Mapping Days to Simulation Steps (Max Step: {MAX_TIMESTEP})...")
    
    # Calculate indices
    # We use round() to get nearest integer step
    # Avoid index 0 if day > 0, but day 0 is not usually in data.
    # If day=0 exists, it maps to step 0.
    
    t_indices = (days / max_day * MAX_TIMESTEP).round().astype(int)
    
    # Ensure indices are within bounds [0, MAX_TIMESTEP]
    t_indices = np.clip(t_indices, 0, MAX_TIMESTEP)
    
    # Ensure unique and sorted (though days are sorted)
    # If multiple days map to same step (unlikely with 750 steps), we keep them.
    
    print(f"Mapped Indices: {t_indices}")
    
    np.save(base_dir / "idx_M4.npy", t_indices)
    np.save(base_dir / "t_idx.npy", t_indices) # For generic loader
    
    # Also save t_obs (Days) for reference
    np.save(base_dir / "t_obs.npy", days)

    print("Conversion Completed.")

def setup_run_dir(run_id, n_particles):
    base_dir = BASE_RUNS_DIR / run_id
    if base_dir.exists():
        print(f"Directory {base_dir} exists. Cleaning up...")
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)
    
    # Convert Data
    convert_csv_to_npy(base_dir)

    # Create theta_MAP.json with TRUE VALUES (as starting point)
    theta_true = np.array([
        0.8, 2.0, 1.0, 0.1, 0.2,   # M1
        1.5, 1.0, 2.0, 0.3, 0.4,   # M2
        2.0, 1.0, 2.0, 1.0,        # M3
        1.2, 0.25,                 # M4 (True)
        1.0, 1.0, 1.0, 1.0         # M5 (True)
    ])
    
    map_data = {
        "theta_full": theta_true.tolist(),
        "model": "M3_locked",
        "note": "Absolute Volume Estimation Initialized at True Values"
    }
    
    with open(base_dir / "theta_MAP.json", "w") as f:
        json.dump(map_data, f, indent=4)
    print("Created theta_MAP.json")
    
    return base_dir

def run_estimation(stage, base_dir, run_id, n_particles):
    print(f"\n=== Running {stage} Estimation with Absolute Volume Model ===")
    
    log_file = base_dir / f"run_{stage}.log"
    
    cmd = [
        "python3", str(SCRIPT_PATH),
        "--run-id", run_id,
        "--stage", stage,
        "--mode", "stage",
        "--n-particles", str(n_particles),
        "--n-jobs", str(N_JOBS),
        "--n-stages", "30",
        "--n-chains", "1",
        "--refine-range", str(REFINE_RANGE),
        "--sigma-obs", str(SIGMA_OBS),
        "--use-absolute-volume",
        "--debug-level", "INFO"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")
    
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
    
    print(f"Starting Absolute Volume Estimation Run: {run_id}")
    
    base_dir = setup_run_dir(run_id, n_particles)
    
    if args.convert_only:
        print("Data conversion complete. Exiting (--convert-only).")
        return

    # Run estimation
    run_estimation("M4", base_dir, run_id, n_particles)
    
    print("\nAll done.")

if __name__ == "__main__":
    main()
