#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import MODEL_CONFIGS, DebugConfig, PRIOR_BOUNDS_DEFAULT
    from visualization import PlotManager, compute_fit_metrics
    from improved1207_paper_jit import BiofilmNewtonSolver, HAS_NUMBA
    from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
    from main.case2_main_fixed import compute_MAP_with_uncertainty, ExperimentConfig
    from utils import save_json
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

from scipy.interpolate import interp1d

def recover_run(run_dir_path):
    run_dir = Path(run_dir_path)
    if not run_dir.exists():
        print(f"Error: Directory {run_dir} does not exist.")
        return

    print(f"Recovering plots for: {run_dir}")
    
    # Load config
    config_path = run_dir / "config.json"
    
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_mgr = PlotManager(str(figures_dir))
    
    # Identify models present
    models = []
    # Check for trace_*.npy files
    for f in run_dir.glob("trace_M*.npy"):
        model = f.stem.split("_")[1] # trace_M2.npy -> M2
        models.append(model)
    
    if not models:
        print(f"No trace files found in {run_dir}")
        return

    print(f"Found models: {models}")
    
    for model in models:
        print(f"Processing model {model}...")
        
        # Load meta info
        meta_path = run_dir / f"likelihood_meta_{model}.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                sigma_obs = meta.get("sigma_obs", 0.001)
                cov_rel = meta.get("cov_rel", 0.005)
                active_indices = meta.get("active_indices", MODEL_CONFIGS[model]["active_indices"])
        else:
            print(f"Warning: Meta file for {model} not found. Using defaults.")
            sigma_obs = 0.001
            cov_rel = 0.005
            active_indices = MODEL_CONFIGS[model]["active_indices"]
            
        # Load data
        try:
            data = np.load(run_dir / f"data_{model}.npy")
            idx_sparse = np.load(run_dir / f"idx_{model}.npy")
            t_arr = np.load(run_dir / f"t_{model}.npy")
            samples = np.load(run_dir / f"trace_{model}.npy")
        except FileNotFoundError as e:
            print(f"Error loading files for {model}: {e}")
            continue

        theta_map_path = run_dir / f"theta_MAP_{model}.json"
        theta_mean_path = run_dir / f"theta_MEAN_{model}.json"
        
        MAP_sub = None
        MEAN_sub = None
        
        if theta_map_path.exists():
            with open(theta_map_path, 'r') as f:
                d = json.load(f)
                MAP_sub = np.array(d["theta_sub"])
                print(f"Loaded MAP from {theta_map_path}")
        
        if theta_mean_path.exists():
             with open(theta_mean_path, 'r') as f:
                d = json.load(f)
                MEAN_sub = np.array(d["theta_sub"])
                print(f"Loaded MEAN from {theta_mean_path}")
                
        # If MAP/MEAN not found (because script crashed before saving), estimate from samples
        if MAP_sub is None:
            print("MAP file not found. Using Mean from samples as fallback.")
            MEAN_sub = np.mean(samples, axis=0)
            MAP_sub = MEAN_sub 
            
        if MEAN_sub is None:
            MEAN_sub = np.mean(samples, axis=0)

        # Prepare full parameter vectors
        prior_mean = (PRIOR_BOUNDS_DEFAULT[0] + PRIOR_BOUNDS_DEFAULT[1]) / 2.0
        theta_full_MAP = np.full(20, prior_mean)
        theta_full_MEAN = np.full(20, prior_mean)
        
        theta_full_MAP[active_indices] = MAP_sub
        theta_full_MEAN[active_indices] = MEAN_sub
        
        # Plotting
        solver_kwargs = {
            k: v for k, v in MODEL_CONFIGS[model].items()
            if k not in ["active_species", "active_indices", "param_names"]
        }
        
        solver = BiofilmNewtonSolver(
            **solver_kwargs,
            active_species=MODEL_CONFIGS[model]["active_species"],
            use_numba=HAS_NUMBA,
        )
        
        tsm = BiofilmTSM_Analytical(
            solver,
            active_theta_indices=active_indices,
            cov_rel=cov_rel,
            use_complex_step=True, # Robust fallback
            use_analytical=False,
            theta_linearization=theta_full_MAP, # Just for shape
            paper_mode=False,
        )
        
        print(f"Running simulation for {model} MAP...")
        try:
            t_fit, x0_fit_MAP, _ = tsm.solve_tsm(theta_full_MAP)
            
            # Interpolate if needed
            if len(t_fit) != len(t_arr) or not np.allclose(t_fit, t_arr):
                 x0_fit_MAP_interp = np.zeros((len(t_arr), x0_fit_MAP.shape[1]))
                 for j in range(x0_fit_MAP.shape[1]):
                    f = interp1d(t_fit, x0_fit_MAP[:, j], kind='linear', bounds_error=False, fill_value='extrapolate')
                    x0_fit_MAP_interp[:, j] = f(t_arr)
                 x0_fit_MAP = x0_fit_MAP_interp
            
            # Re-generate data plot with fit
            # We need to pass 'phibar' if we want to show the ground truth line if available
            # But here we just plot the fit against the noisy data
            plot_mgr.plot_TSM_simulation(t_arr, x0_fit_MAP, MODEL_CONFIGS[model]["active_species"], f"{model}_MAP_fit", data, idx_sparse)
            print(f"Generated {model}_MAP_fit plot.")
        except Exception as e:
            print(f"Failed to generate MAP plot: {e}")
            import traceback
            traceback.print_exc()

        # Also plot trace if samples exist
        if len(samples) > 0:
             print(f"Generating trace plot for {model}...")
             fig, axes = plt.subplots(len(active_indices), 1, figsize=(10, 2*len(active_indices)), sharex=True)
             if len(active_indices) == 1: axes = [axes]
             
             param_names = MODEL_CONFIGS[model]["param_names"]
             
             for i, ax in enumerate(axes):
                 ax.plot(samples[:, i])
                 ax.set_ylabel(param_names[i] if i < len(param_names) else f"Param {i}")
                 ax.grid(True, alpha=0.3)
                 
             axes[-1].set_xlabel("Sample")
             plt.tight_layout()
             plt.savefig(figures_dir / f"trace_plot_{model}.png")
             plt.close()
             print(f"Generated trace_plot_{model}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Run directories to recover")
    args = parser.parse_args()
    
    for d in args.dirs:
        recover_run(d)
