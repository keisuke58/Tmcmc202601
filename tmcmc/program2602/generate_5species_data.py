
import numpy as np
import os
from improved_5species_jit import BiofilmNewtonSolver5S, get_theta_true

def generate_data():
    print("Generating synthetic data for 5-species model...")
    
    # 1. Get True Parameters
    theta_true = get_theta_true()
    print(f"True Parameters: {theta_true}")
    
    # 2. Setup Solver
    # Using settings matched with M3 (4-species) from config.py
    # dt=1e-4, maxtimestep=750, c_const=25.0
    solver = BiofilmNewtonSolver5S(
        dt=1e-4,
        maxtimestep=750,
        active_species=[0, 1, 2, 3, 4], # All 5 species active
        c_const=25.0,
        alpha_const=0.0,
        phi_init=0.02
    )
    
    # 3. Run Simulation
    t_arr, g_arr = solver.solve(theta_true)
    
    # 4. Save Data
    # output_dir = "data_5species"
    output_dir = os.path.join(os.path.dirname(__file__), "data_5species")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "t_arr.npy"), t_arr)
    np.save(os.path.join(output_dir, "g_arr.npy"), g_arr)
    np.save(os.path.join(output_dir, "theta_true.npy"), theta_true)
    
    print(f"Data saved to {output_dir}/")
    print(f"Time steps: {len(t_arr)}")
    print(f"State shape: {g_arr.shape}")
    
    # Check if species 5 grew
    phi5_final = g_arr[-1, 4]
    print(f"Final Phi5: {phi5_final}")
    if phi5_final < 1e-6:
        print("WARNING: Species 5 did not grow! Adjust parameters.")
    else:
        print("Species 5 grew successfully.")

if __name__ == "__main__":
    generate_data()
