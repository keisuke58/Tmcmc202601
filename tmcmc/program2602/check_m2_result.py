
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path
import json

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from improved_5species_jit import BiofilmNewtonSolver5S, HAS_NUMBA
from tmcmc_5species_tsm import BiofilmTSM5S
from visualization.helpers import compute_phibar
from config import MODEL_CONFIGS

def load_run_data(run_dir):
    run_path = Path(run_dir)
    
    # Load samples
    samples_path = run_path / "samples.npy"
    if not samples_path.exists():
        # Try to find temporary samples
        npy_files = sorted(run_path.glob("tmcmc_samples_*.npy"))
        if npy_files:
            samples_path = npy_files[-1]
            print(f"Using intermediate samples: {samples_path}")
        else:
            raise FileNotFoundError(f"No samples found in {run_dir}")
            
    samples = np.load(samples_path)
    
    # Load log likelihoods if available to find MAP
    loglik_path = run_path / "log_likelihoods.npy"
    if loglik_path.exists():
        logliks = np.load(loglik_path)
        map_idx = np.argmax(logliks)
        theta_map = samples[map_idx]
        print(f"Using MAP estimate (log-likelihood: {logliks[map_idx]:.4f})")
    else:
        # Fallback to mean
        theta_map = np.mean(samples, axis=0)
        print("Using Mean estimate (log likelihoods not found)")
        
    return theta_map, samples

def load_ground_truth():
    data_dir = Path(__file__).parent / "data_5species"
    t_arr = np.load(data_dir / "t_arr.npy")
    g_arr = np.load(data_dir / "g_arr.npy")
    return t_arr, g_arr

def run_simulation(theta, model_config_name="M2"):
    # M2 config: active_species=[2, 3] (Species 3, 4)
    # The parameter vector 'theta' from M2 run contains only the active parameters.
    # We need to construct the full 20D parameter vector for simulation, 
    # but BiofilmTSM5S handles the mapping if we provide indices.
    
    # However, to compare with full 5-species ground truth, we might want to run 
    # the simulation with the same conditions as the ground truth but with estimated parameters replaced.
    
    # Actually, M2 run estimates parameters for Species 3 and 4.
    # Species 1, 2, 5 are inactive or fixed?
    # In the project memory, M2 is "Species 3, 4 active".
    
    # Let's look at how case2_main_fixed.py sets up the solver.
    # It uses config["active_species"] and config["active_indices"].
    
    config = MODEL_CONFIGS[model_config_name]
    active_species = config["active_species"] # e.g. [2, 3]
    active_indices = config["active_indices"]
    
    # Load true parameters to fill in the non-estimated ones
    data_dir = Path(__file__).parent / "data_5species"
    theta_true = np.load(data_dir / "theta_true.npy")
    
    # Construct full theta vector with estimated values
    theta_full = theta_true.copy()
    
    # If theta has shape (n_active_params,), map it to full vector
    if theta.shape[0] == len(active_indices):
        theta_full[active_indices] = theta
    else:
        print(f"Warning: Theta shape {theta.shape} does not match active indices length {len(active_indices)}. Assuming full vector or mismatch.")
        if theta.shape[0] == 20:
            theta_full = theta
    
    # Setup solver
    solver = BiofilmNewtonSolver5S(
        active_species=active_species,
        use_numba=HAS_NUMBA
    )
    
    # Setup TSM
    # We use theta_full as linearization point if needed, but TSM solve uses the passed theta
    tsm = BiofilmTSM5S(
        solver,
        active_theta_indices=active_indices,
        cov_rel=0.005, # Default
        theta_linearization=theta_full 
    )
    
    # Run simulation
    # Note: solve_tsm returns (t_arr, x0, sig2)
    # But we want the full trajectory. solve_tsm calls solver.solve internally.
    # Let's call solver.solve directly to get the trajectory
    
    t_arr, g_arr = solver.solve(theta_full)
    
    return t_arr, g_arr

def main():
    parser = argparse.ArgumentParser(description="Verify M2 run results")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID or path to run directory")
    parser.add_argument("--model", type=str, default="M2", help="Model name (M2, etc.)")
    args = parser.parse_args()
    
    # Resolve run directory
    if os.path.exists(args.run_id):
        run_dir = args.run_id
    else:
        run_dir = os.path.join(os.path.dirname(__file__), "_runs", args.run_id)
        
    if not os.path.exists(run_dir):
        print(f"Error: Run directory {run_dir} not found")
        return

    print(f"Analyzing run: {run_dir}")
    
    try:
        theta_est, samples = load_run_data(run_dir)
    except Exception as e:
        print(f"Failed to load run data: {e}")
        return

    print("Running simulation with estimated parameters...")
    t_est, g_est = run_simulation(theta_est, args.model)
    
    print("Loading ground truth...")
    t_true, g_true = load_ground_truth()
    
    # Compute RMSE for active species
    config = MODEL_CONFIGS[args.model]
    active_species = config["active_species"] # [2, 3] for M2
    
    # Interpolate estimated result to match ground truth time points if needed
    # Usually t_arr is same if parameters are similar and solver is deterministic with fixed steps
    # But adaptive steps might differ. BiofilmNewtonSolver5S usually uses fixed steps?
    # Let's assume fixed steps or interpolate.
    
    from scipy.interpolate import interp1d
    
    rmse_list = []
    
    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for i, sp_idx in enumerate(active_species):
        # Ground truth
        # g_arr structure: 0-4 (Phi), 5 (Phi0), 6-10 (Psi), 11 (Gamma)
        # Species index sp_idx (0-4) maps directly to g_arr column sp_idx
        
        y_true = g_true[:, sp_idx]
        
        # Estimated
        y_est_raw = g_est[:, sp_idx]
        
        # Interpolate
        f_est = interp1d(t_est, y_est_raw, kind='linear', fill_value="extrapolate")
        y_est_interp = f_est(t_true)
        
        rmse = np.sqrt(np.mean((y_true - y_est_interp)**2))
        rmse_list.append(rmse)
        
        plt.plot(t_true, y_true, '-', color=colors[sp_idx], label=f'True S{sp_idx+1}', alpha=0.5, linewidth=3)
        plt.plot(t_true, y_est_interp, '--', color=colors[sp_idx], label=f'Est S{sp_idx+1} (RMSE={rmse:.4f})')

    plt.title(f"Model {args.model} Estimation Result\nTotal RMSE: {np.mean(rmse_list):.4f}")
    plt.xlabel("Time")
    plt.ylabel("Volume Fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(run_dir, "verification_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
