#!/usr/bin/env python3
"""
Verify Dysbiotic HOBIC Surge Capture
------------------------------------
This script loads the experimental data for Dysbiotic HOBIC condition
and simulates the biofilm dynamics using the MAP parameters from the latest run.
It produces a plot comparing the experimental data with the simulation result,
focusing on P. gingivalis (Red) and F. nucleatum (Purple) to verify the Surge phenomenon.

Usage:
    python verify_dysbiotic_hobic_surge.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # /home/.../Tmcmc202601
DATA_5SPECIES_ROOT = PROJECT_ROOT / "data_5species"
TMCMC_PROGRAM_ROOT = PROJECT_ROOT / "tmcmc" / "program2602"

sys.path.insert(0, str(DATA_5SPECIES_ROOT))
sys.path.insert(0, str(DATA_5SPECIES_ROOT / "main"))
sys.path.insert(0, str(TMCMC_PROGRAM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tmcmc"))

# Import necessary modules
from improved_5species_jit import BiofilmNewtonSolver5S, get_theta_true, theta_to_matrices_numpy
from estimate_commensal_static import load_experimental_data, SPECIES_MAP

def load_map_parameters(run_dir: Path):
    """Load MAP parameters from theta_MAP.json"""
    map_file = run_dir / "theta_MAP.json"
    if not map_file.exists():
        raise FileNotFoundError(f"MAP file not found: {map_file}")
    
    with open(map_file, "r") as f:
        data = json.load(f)
    
    return np.array(data["theta_sub"])

def run_simulation(theta, phi_init, t_eval):
    """Run deterministic simulation with given parameters"""
    solver = BiofilmNewtonSolver5S(
        dt=0.001,  # Use fine timestep for accuracy
        maxtimestep=2500, # Sufficient for 96h? 2500*0.001 = 2.5 (too short?)
                          # Wait, typical dt is 0.01 or 0.1?
                          # improved_5species_jit default is dt=0.01, steps=5000 -> T=50.
                          # Experimental data is in hours (0-96h).
                          # Need to check time scaling.
        # Check config.py or estimate script for time scaling.
    )
    
    # Check time scaling in estimate_commensal_static.py
    # Usually data time is normalized or model time is scaled.
    # Let's assume t=1.0 corresponds to 24h or similar.
    # Actually, let's check estimate_commensal_static.py for time handling.
    
    pass

# We need to check time scaling first.
# estimate_commensal_static.py:
# t_span = (0, days[-1]) -> 0 to 4 (96h = 4 days)
# t_eval = days
# solver.solve(..., t_eval=t_eval)

def main():
    # 1. Configuration
    condition = "Dysbiotic"
    cultivation = "HOBIC"
    # run_dir = DATA_5SPECIES_ROOT / "_runs" / "Dysbiotic_HOBIC_20260208_002100"
    # Try the 'abs' run which might have better volume tracking?
    # run_dir = DATA_5SPECIES_ROOT / "_runs" / "Dysbiotic_HOBIC_abs_20260208_011000"
    run_dir = DATA_5SPECIES_ROOT / "_runs" / "Dysbiotic_HOBIC_20260207_023107"
    
    if not run_dir.exists():
        print(f"Run directory {run_dir} does not exist.")
        sys.exit(1)
    
    print(f"Loading data for {condition} {cultivation}...")
    
    # 2. Load Experimental Data
    data_abs, t_days, sigma_obs, phi_init_exp, metadata = load_experimental_data(
        DATA_5SPECIES_ROOT,
        condition=condition,
        cultivation=cultivation
    )
    
    # Normalize data to fractions for comparison with TSM (which models fractions)
    row_sums = data_abs.sum(axis=1, keepdims=True)
    data_frac = np.where(row_sums > 0, data_abs / row_sums, 0.0)
    
    print(f"Timepoints: {t_days}")
    print(f"Initial conditions (Exp Abs): {phi_init_exp}")
    
    # 3. Load MAP Parameters
    print(f"Loading MAP parameters from {run_dir}...")
    try:
        theta_map = load_map_parameters(run_dir)
        print("MAP parameters loaded successfully.")
    except FileNotFoundError:
        print("MAP file not found. Using default/true parameters for testing.")
        theta_map = get_theta_true()

    # 4. Simulation
    print("Running simulation...")
    
    # Configuration from config.json of the run
    # "dt": 0.0001, "maxtimestep": 2500, "c_const": 25.0
    dt = 0.0001
    maxtimestep = 2500
    c_const = 25.0
    alpha_const = 0.0
    
    # Calculate day_scale matching estimate_commensal_static.py logic
    t_max_model = maxtimestep * dt
    t_max_days = float(t_days[-1])
    day_scale = (t_max_model * 0.95) / t_max_days
    print(f"Calculated day_scale: {day_scale:.6f}")
    
    # Convert days to model time
    t_model_eval = t_days * day_scale
    idx_sparse = np.round(t_model_eval / dt).astype(int)
    print(f"Evaluation indices: {idx_sparse}")
    
    # Initialize solver
    solver = BiofilmNewtonSolver5S(
        dt=dt,
        maxtimestep=maxtimestep,
        c_const=c_const,
        alpha_const=alpha_const,
        phi_init=phi_init_exp,
        active_species=[0, 1, 2, 3, 4]
    )
    
    # 6. Run simulation with MAP parameters
    print("Running simulation (MAP)...")
    t_arr, g_arr = solver.solve(theta_map)
    
    # 6b. Run simulation with HYPOTHETICAL "Hidden Cooperation" parameters
    print("Running simulation (Hypothetical: Negative Interactions)...")
    theta_hyp = theta_map.copy()
    # Force negative interactions (Cooperation) from Veillonella (S2) to P.g. (S4)?
    # Wait, Veillonella is S2 (Index 2). P.g. is S4 (Index 4).
    # Interaction S2->S4 is a25 (Index 17).
    # Interaction S3->S4 is a35 (Index 18).
    # Interaction S4->S4 is a45 (Index 19).
    
    # Let's try making theta[18] (Vei->Pg) POSITIVE (Growth?).
    # Note: Indices in code logic: 
    # a15 (S0->S4) = theta[16]
    # a25 (S1->S4) = theta[17]
    # a35 (S2->S4) = theta[18]  <- Veillonella to Pg
    # a45 (S3->S4) = theta[19]  <- Fn to Pg
    
    theta_hyp[18] = 15.0   # EXTREME POSITIVE interaction from Veillonella
    theta_hyp[19] = 10.0   # EXTREME POSITIVE interaction from F. nucleatum
    
    # Also reduce Decay (b5) and Self-Inhibition (a55)
    theta_hyp[14] = 0.1   # a55 (S5->S5) Minimal self-inhibition
    theta_hyp[15] = 0.0   # b5 (Decay) NO decay
    
    t_arr_hyp, g_arr_hyp = solver.solve(theta_hyp)
    
    # --- Visualization ---
    
    # Calculate indices for evaluation
    t_eval_days = t_days
    idx_eval = np.searchsorted(t_arr, t_eval_days * day_scale)
    # Clamp indices
    idx_eval = np.clip(idx_eval, 0, len(t_arr) - 1)
    
    print(f"Calculated day_scale: {day_scale:.6f}")
    print(f"Evaluation indices: {idx_eval}")

    # Extract simulation results at timepoints
    phi_sim = g_arr[idx_eval, 0:5]
    phi_sim_hyp = g_arr_hyp[idx_eval, 0:5]
    
    # Full trajectory for plotting
    t_plot_days = t_arr / day_scale
    g_arr_eval = g_arr[:, 0:5]
    g_arr_eval_hyp = g_arr_hyp[:, 0:5]

    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    
    species_colors = ['blue', 'green', 'gold', 'purple', 'red']
    species_names = ['S. oralis', 'A. naeslundii', 'Veillonella', 'F. nucleatum', 'P. gingivalis']
    
    # Plot 1: MAP
    plt.subplot(1, 2, 1)
    for i in range(5):
        # Plot experimental data (scatter)
        # Use data_frac for comparison
        plt.scatter(t_days, data_frac[:, i], color=species_colors[i], label=f"{species_names[i]} (Exp)", alpha=0.6)
        
        # Plot simulation (line)
        plt.plot(t_plot_days, g_arr_eval[:, i], color=species_colors[i], linestyle='-', linewidth=2)

    plt.xlabel("Time (Days)")
    plt.ylabel("Relative Abundance")
    plt.title(f"MAP (Current Run)\nNo Surge")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Hypothetical
    plt.subplot(1, 2, 2)
    for i in range(5):
        # Plot experimental data (scatter)
        plt.scatter(t_days, data_frac[:, i], color=species_colors[i], alpha=0.6)
        
        # Plot simulation (line)
        plt.plot(t_plot_days, g_arr_eval_hyp[:, i], color=species_colors[i], linestyle='--', linewidth=2)

    plt.xlabel("Time (Days)")
    plt.title(f"Hypothetical (Positive Interaction)\nVei->Pg: {theta_hyp[18]}, Fn->Pg: {theta_hyp[19]}")
    plt.grid(True, alpha=0.3)
    plt.legend(species_names, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("verify_dysbiotic_hobic_surge_comparison.png")
    print("Plot saved to verify_dysbiotic_hobic_surge_comparison.png")
    
    # 7. Check Surge at final timepoint (Hypothetical)
    final_idx_exp = -1
    final_idx_sim = -1 # End of trajectory
    
    # P. gingivalis (Index 4, Red)
    pg_exp = data_frac[final_idx_exp, 4]
    pg_sim = g_arr_eval_hyp[-1, 4]
    
    # F. nucleatum (Index 3, Purple)
    fn_exp = data_frac[final_idx_exp, 3]
    fn_sim = g_arr_eval_hyp[-1, 3]
    
    # Veillonella (Index 2, Gold)
    vei_exp = data_frac[final_idx_exp, 2]
    vei_sim = g_arr_eval_hyp[-1, 2]
    
    print("\n=== Surge Verification Results (Hypothetical) ===")
    print(f"Timepoint: Day 21")
    print(f"P. gingivalis (Red): Exp = {pg_exp:.4f}, Sim = {pg_sim:.4f}, Diff = {pg_sim - pg_exp:.4f}")
    print(f"F. nucleatum (Purple): Exp = {fn_exp:.4f}, Sim = {fn_sim:.4f}, Diff = {fn_sim - fn_exp:.4f}")
    print(f"Veillonella (Gold):  Exp = {vei_exp:.4f}, Sim = {vei_sim:.4f}, Diff = {vei_sim - vei_exp:.4f}")
    
    if pg_sim > 0.8 * pg_exp:
        print("SUCCESS: P. gingivalis Surge CAPTURED with negative parameters!")
    else:
        print("WARNING: P. gingivalis Surge NOT captured even with negative parameters.")
        
    print("\nM5 Block Parameters (P.g. Cross-Interactions):")
    print(f"a15 (S1->P.g): {theta_map[16]:.4f}")
    print(f"a25 (S2->P.g): {theta_map[17]:.4f}")
    print(f"a35 (S3->P.g): {theta_map[18]:.4f}")
    print(f"a45 (S4->P.g): {theta_map[19]:.4f}")
    
    print("\nM4 Block Parameters (P.g. Self):")
    print(f"a55 (S5->S5): {theta_map[14]:.4f}")
    print(f"b5 (Decay):   {theta_map[15]:.4f}")

if __name__ == "__main__":
    main()
