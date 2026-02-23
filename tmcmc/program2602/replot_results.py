import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append("/home/nishioka/IKM_Hiwi/Tmcmc202601")

from tmcmc.improved_5species_jit import BiofilmNewtonSolver5S
from tmcmc.visualization.plot_manager import PlotManager
from tmcmc.visualization import compute_fit_metrics
from tmcmc.utils import save_json
from tmcmc.config import MODEL_CONFIGS

def main():
    run_id = "manual_M4_M5_joint_estimation_1000p_true_init"
    base_dir = Path(f"/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/_runs/{run_id}")
    
    print(f"Checking run directory: {base_dir}")
    
    # 1. Load MAP parameters
    map_file = base_dir / "refined" / "theta_MAP_refined.json"
    if not map_file.exists():
        print(f"Error: MAP file not found at {map_file}")
        return

    with open(map_file) as f:
        map_data = json.load(f)
    
    theta_full = np.array(map_data["theta_full"])
    print("Loaded theta_full.")
    
    # 2. Load Data and Indices
    data_file = base_dir / "data.npy"
    idx_file = base_dir / "idx_M4.npy"
    
    if not data_file.exists():
        print("Error: data.npy not found")
        return
    if not idx_file.exists():
        print("Error: idx_M4.npy not found")
        return
        
    data = np.load(data_file)
    idx_sparse = np.load(idx_file)
    
    print(f"Data shape: {data.shape}")
    print(f"Idx shape: {idx_sparse.shape}")
    print(f"Idx min/max: {idx_sparse.min()}, {idx_sparse.max()}")
    
    # 3. Setup Solver
    # Replicate logic from refine_5species_linearized.py
    # Use M4 config which has phi_init=0.02
    base_config = MODEL_CONFIGS.get("M4") 
    
    solver_kwargs = {
        k: v for k, v in base_config.items()
        if k not in ["active_species", "active_indices", "param_names"]
    }
    
    # Force M4/M5 active species
    active_species = [0, 1, 2, 3, 4]
    
    print(f"Solver kwargs: {solver_kwargs}")
    
    solver = BiofilmNewtonSolver5S(**solver_kwargs)
    
    # 4. Run Simulation
    print("Running solver...")
    t_fit, x_fit = solver.solve(theta_full)
    print(f"Solver output t_fit shape: {t_fit.shape}")
    print(f"Solver output x_fit shape: {x_fit.shape}")
    
    # 5. Calculate Metrics
    print("Calculating metrics...")
    try:
        metrics = compute_fit_metrics(t_fit, x_fit, active_species, data, idx_sparse)
        save_json(base_dir / "refined" / "metrics_recalc.json", metrics)
        print("Metrics recalculated and saved.")
    except Exception as e:
        print(f"Metrics calculation failed: {e}")
    
    # 6. Plotting with Debug
    print("Attempting plotting...")
    plot_dir = base_dir / "refined"
    plot_mgr = PlotManager(str(plot_dir))
    
    try:
        plot_mgr.plot_TSM_simulation(
            t_fit, x_fit, active_species, "Refined_MAP_Fit_Replot", 
            data, idx_sparse
        )
        print("Plotting successful!")
    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Detailed Debug for Shape Mismatch
        t_normalized = (t_fit - t_fit.min()) / (t_fit.max() - t_fit.min())
        t_obs = t_normalized[idx_sparse]
        print(f"DEBUG: t_normalized shape: {t_normalized.shape}")
        print(f"DEBUG: t_obs (sliced) shape: {t_obs.shape}")
        print(f"DEBUG: data shape: {data.shape}")
        
        if t_obs.shape[0] != data.shape[0]:
            print("MISMATCH DETECTED: t_obs rows != data rows")

if __name__ == "__main__":
    main()
