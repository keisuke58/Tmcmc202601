
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append("/home/nishioka/IKM_Hiwi/Tmcmc202601")

from tmcmc.visualization.plot_manager import PlotManager
from tmcmc.improved_5species_jit import BiofilmNewtonSolver5S

BASE_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/_runs/debug_M4_M5_100p")

def generate_plots():
    print("Generating plots from", BASE_DIR)
    
    # Check if files exist
    if not (BASE_DIR / "theta_MAP.json").exists():
        print("theta_MAP.json not found")
        return

    # Load theta_MAP.json
    with open(BASE_DIR / "theta_MAP.json", "r") as f:
        map_data = json.load(f)
    
    theta_full = np.array(map_data["theta_full"])
    print("Loaded theta_full:", theta_full)
    
    # Load data
    data_path = BASE_DIR / "data.npy"
    idx_path = BASE_DIR / "idx_M5.npy" # Use M5 index for full view
    
    if not data_path.exists() or not idx_path.exists():
        print("Data or Index file not found")
        return
        
    data = np.load(data_path)
    idx_sparse = np.load(idx_path)
    
    # Setup plot manager
    # Output to 'figures' dir
    output_dir = BASE_DIR / "figures"
    plot_mgr = PlotManager(str(output_dir))
    
    # Run simulation
    print("Running simulation...")
    solver = BiofilmNewtonSolver5S(
        dt=0.0001, maxtimestep=750, c_const=25.0, alpha_const=0.0, phi_init=0.02
    )
    
    t_fit, x_fit = solver.solve(theta_full)
    
    # Plot TSM
    active_species = [0, 1, 2, 3, 4]
    plot_mgr.plot_TSM_simulation(
        t_fit, x_fit, active_species, "Debug_M4_M5_Fit",
        data=data, idx_sparse=idx_sparse
    )
    print("Saved TSM simulation plot")
    
    # Separate plots for each species
    for sp in active_species:
        plt.figure(figsize=(10, 6))
        # x_fit shape: (time, species) or (species, time)?
        # BiofilmNewtonSolver5S returns (t, x). x is (time, species).
        plt.plot(t_fit, x_fit[:, sp], label=f"Species {sp+1} (Sim)", linewidth=2)
        
        # Plot data if available
        # data shape: (time, species) or flattened?
        # data.npy usually (time_idx, species)
        # idx_sparse matches data rows to t_arr indices
        if idx_sparse is not None:
             t_data = t_fit[idx_sparse.astype(int)]
             # Check data shape
             if data.ndim == 2 and data.shape[1] >= 5:
                 plt.scatter(t_data, data[:, sp], label=f"Species {sp+1} (Data)", marker='x', color='red')
        
        plt.title(f"Species {sp+1} Fit")
        plt.xlabel("Time")
        plt.ylabel("Volume Fraction")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"species_{sp+1}_fit.png")
        plt.close()
    print("Saved individual species plots")

    # Load traces if available (from refined/trace_refined.npy)
    # Check M4 and M5 refined folders?
    # The script outputs to 'refined' subdir in run_dir.
    # But run_estimation("M4") runs in RUN_ID directory.
    # So results are in RUN_ID/refined.
    # M5 overwrites M4 in 'refined'?
    # My script renames theta_MAP_refined.json but not trace_refined.npy.
    # So trace_refined.npy will be from M5 (last run).
    
    trace_path = BASE_DIR / "refined" / "trace_refined.npy"
    if trace_path.exists():
        trace = np.load(trace_path)
        print(f"Loaded trace from {trace_path}: shape {trace.shape}")
        
        # Plot trace (simple)
        plt.figure(figsize=(12, 8))
        for i in range(trace.shape[1]):
            plt.subplot(trace.shape[1], 1, i+1)
            plt.plot(trace[:, i])
            plt.ylabel(f"Param {i}")
        plt.xlabel("Sample")
        plt.tight_layout()
        plt.savefig(output_dir / "trace_plot_M5.png")
        print("Saved trace plot")
        
        # Corner plot?
        try:
            import corner
            figure = corner.corner(trace)
            figure.savefig(output_dir / "corner_plot_M5.png")
            print("Saved corner plot")
        except ImportError:
            print("corner package not found, skipping corner plot")
            
        # Posterior Predictive Bands
        print("Generating posterior predictive bands...")
        n_samples = min(100, trace.shape[0])
        indices = np.random.choice(trace.shape[0], n_samples, replace=False)
        selected_thetas = trace[indices]
        
        phibar_samples = []
        for i, theta_sample in enumerate(selected_thetas):
            # theta_sample might be partial if mode=stage? 
            # But trace_refined.npy usually stores the active parameters?
            # Or the full theta?
            # refine_5species_linearized.py saves what?
            # If it returns samples from TMCMC, they are active params.
            # We need to reconstruct full theta.
            
            # Map samples to full theta
            # Assuming trace contains ACTIVE parameters only
            # We need active_indices from M5 (since this is likely M5 trace)
            # M5 active_indices = [16, 17, 18, 19] if mode=stage, or [0..19] if mode=refine?
            # The run uses mode="stage" for M4 and M5.
            # So trace has 2 columns for M4, 4 columns for M5.
            # Wait, if we run fix_plots.py after M5, trace_refined.npy will be M5 trace.
            # M5 mode=stage means active_indices = [16, 17, 18, 19].
            # So trace has 4 columns.
            # We need to merge with base theta.
            
            theta_full_sample = theta_full.copy()
            
            # How do we know which indices? 
            # We can infer from trace shape or just assume M5 stage indices for now since we know the context.
            # M5 stage indices: 16, 17, 18, 19
            if trace.shape[1] == 4:
                theta_full_sample[16:20] = theta_sample
            elif trace.shape[1] == 20:
                theta_full_sample = theta_sample
            elif trace.shape[1] == 2: # M4
                theta_full_sample[14:16] = theta_sample
            else:
                # Fallback: use MAP (don't update) or try to guess?
                # If we can't map, skip bands
                pass
            
            try:
                _, x_sim = solver.solve(theta_full_sample)
                phibar_samples.append(x_sim)
            except Exception as e:
                print(f"Solver failed for sample {i}: {e}")
        
        if phibar_samples:
            phibar_samples = np.array(phibar_samples)
            # phibar_samples shape: (n_samples, n_time, n_species)
            # plot_posterior_predictive_band expects (n_samples, n_time, n_species)
            
            plot_mgr.plot_posterior_predictive_band(
                t_fit, phibar_samples, active_species, "Posterior_Bands_M5",
                data=data, idx_sparse=idx_sparse
            )
            print("Saved posterior predictive bands")

    else:
        print("Trace file not found")

if __name__ == "__main__":
    generate_plots()
