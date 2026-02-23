
import time
import numpy as np
from improved1207_paper_jit import BiofilmNewtonSolver, get_theta_true as get_theta_4s
from improved_5species_jit import BiofilmNewtonSolver5S, get_theta_true as get_theta_5s

def benchmark():
    print("Benchmarking 4-species vs 5-species solver...")
    
    # Common settings (M3-like)
    dt = 1e-4
    maxtimestep = 750
    c_const = 25.0
    phi_init = 0.02
    
    # 1. Run 4-Species Solver
    print("\n--- 4-Species Solver ---")
    theta_4s = get_theta_4s() # 14 params
    # M3 active species: [0, 1, 2, 3] (all 4)
    solver4 = BiofilmNewtonSolver(
        dt=dt, maxtimestep=maxtimestep,
        active_species=[0, 1, 2, 3],
        c_const=c_const, alpha_const=0.0, phi_init=phi_init
    )
    
    # Warmup
    solver4.solve(theta_4s)
    
    start_time = time.time()
    for _ in range(10):
        solver4.solve(theta_4s)
    end_time = time.time()
    avg_time_4s = (end_time - start_time) / 10.0
    print(f"Avg time (4S): {avg_time_4s:.4f} s")
    
    # 2. Run 5-Species Solver (configured as 4-species)
    print("\n--- 5-Species Solver (Emulating 4S) ---")
    theta_5s = get_theta_5s() # 20 params
    # 5th species params are at the end, but solver uses active_species to mask
    # We set active_species=[0, 1, 2, 3] to disable 5th species
    solver5 = BiofilmNewtonSolver5S(
        dt=dt, maxtimestep=maxtimestep,
        active_species=[0, 1, 2, 3],
        c_const=c_const, alpha_const=0.0, phi_init=phi_init
    )
    
    # Warmup
    solver5.solve(theta_5s)
    
    start_time = time.time()
    for _ in range(10):
        solver5.solve(theta_5s)
    end_time = time.time()
    avg_time_5s = (end_time - start_time) / 10.0
    print(f"Avg time (5S): {avg_time_5s:.4f} s")
    
    print(f"\nSlowdown factor: {avg_time_5s / avg_time_4s:.2f}x")

if __name__ == "__main__":
    benchmark()
