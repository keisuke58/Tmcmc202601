
import numpy as np
import os
from improved_5species_jit import BiofilmNewtonSolver5S, get_theta_true

def generate_sequential_data():
    print("Generating sequential synthetic data for 5-species model...")
    
    theta_true = get_theta_true()
    data_dir = os.path.join(os.path.dirname(__file__), "data_5species_sequential")
    os.makedirs(data_dir, exist_ok=True)
    
    # Common settings (M3-like)
    dt = 1e-4
    maxtimestep = 750
    c_const = 25.0
    alpha_const = 0.0
    phi_init = 0.02

    scenarios = [
        ("M1", [0, 1]),             # S1, S2
        ("M2", [2, 3]),             # S3, S4
        ("M3", [0, 1, 2, 3]),       # S1, S2, S3, S4
        ("M4", [4]),                # S5 (Self)
        ("M5", [0, 1, 2, 3, 4]),    # All (S5 Cross)
    ]
    
    for name, active_species in scenarios:
        print(f"Generating data for {name} (Active: {active_species})...")
        
        solver = BiofilmNewtonSolver5S(
            dt=dt,
            maxtimestep=maxtimestep,
            active_species=active_species,
            c_const=c_const,
            alpha_const=alpha_const,
            phi_init=phi_init,
            use_numba=True
        )
        
        t_arr, g_arr = solver.solve(theta_true)
        
        # Save
        np.save(os.path.join(data_dir, f"t_{name}.npy"), t_arr)
        np.save(os.path.join(data_dir, f"data_{name}.npy"), g_arr)
        
        # Verify growth
        final_biomass = np.sum(g_arr[-1, active_species])
        print(f"  Final biomass (active): {final_biomass:.4f}")
        
    np.save(os.path.join(data_dir, "theta_true.npy"), theta_true)
    print(f"All data saved to {data_dir}/")

if __name__ == "__main__":
    generate_sequential_data()
