
import numpy as np
import sys
from pathlib import Path
import logging

# Add project root to sys.path
sys.path.insert(0, str(Path.cwd()))

from tmcmc.case2_tmcmc_refined_JIT import LogLikelihoodEvaluator, MODEL_CONFIGS, select_sparse_data_indices
from tmcmc.improved1207_paper_jit import get_theta_true, BiofilmNewtonSolver, BiofilmTSM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_m2_integration")

def test_m2_integration():
    print("=== Testing M2 Integration with Locked Inactive Species ===")

    # 1. Setup M2 Config
    config = MODEL_CONFIGS["M2"]
    print(f"M2 Config loaded: active_species={config['active_species']}")
    
    active_species = config["active_species"] # [2, 3]
    active_indices = config["active_indices"] # [5, 6, 7, 8, 9]
    
    theta_true = get_theta_true()
    
    # 2. Generate Synthetic Data (using True M2 params)
    # We need data to evaluate likelihood. 
    # We'll generate it using the same solver logic.
    print("Generating synthetic data...")
    solver_kwargs = {
        k: v for k, v in config.items()
        if k not in ["active_species", "active_indices", "param_names"]
    }
    
    solver = BiofilmNewtonSolver(
        **solver_kwargs,
        active_species=active_species,
        use_numba=True,
    )
    
    tsm = BiofilmTSM(solver, active_theta_indices=active_indices)
    t_arr, x0_true, _ = tsm.solve_tsm(theta_true)
    
    # Create sparse data
    n_total = len(t_arr)
    n_obs = 20
    idx_sparse = select_sparse_data_indices(n_total, n_obs)
    
    # Calculate phi_bar (product of phi * psi) for active species
    # active_species are [2, 3]. In state vector x0 (10 dim):
    # phi1..4 are indices 0..3
    # psi1..4 are indices 5..8
    # So for sp=2 (species 3): phi is x0[:, 2], psi is x0[:, 7]
    # For sp=3 (species 4): phi is x0[:, 3], psi is x0[:, 8]
    
    data = np.zeros((n_obs, len(active_species)))
    for i, sp in enumerate(active_species):
        phi = x0_true[idx_sparse, sp]
        psi = x0_true[idx_sparse, 5 + sp]
        data[:, i] = phi * psi
        
    print(f"Generated data shape: {data.shape}")

    # 3. Create LogLikelihoodEvaluator
    evaluator = LogLikelihoodEvaluator(
        solver_kwargs=solver_kwargs,
        active_species=active_species,
        active_indices=active_indices,
        theta_base=theta_true,
        data=data,
        idx_sparse=idx_sparse,
        sigma_obs=0.001,
        cov_rel=0.005
    )
    
    # 4. Evaluate Baseline (True Params)
    theta_m2_true = theta_true[active_indices]
    logL_base = evaluator(theta_m2_true)
    print(f"Baseline LogL (True Params): {logL_base:.6f}")
    
    # 5. Test M1 Parameter Sensitivity (Inactive)
    print("\n--- Testing M1 Parameter Sensitivity (Should be 0) ---")
    # We can't directly pass M1 params to evaluator because it only takes theta_sub (M2 params).
    # But evaluator uses self.theta_base. We can modify self.theta_base temporarily.
    
    m1_indices = [0, 1, 2, 3, 4]
    pass_all = True
    for idx in m1_indices:
        original_val = evaluator.theta_base[idx]
        perturbed_val = original_val * 1.5 # 50% change
        
        evaluator.theta_base[idx] = perturbed_val
        logL_pert = evaluator(theta_m2_true)
        
        diff = abs(logL_pert - logL_base)
        print(f"Param idx {idx}: {original_val} -> {perturbed_val}")
        print(f"  LogL Change: {diff:.6e}")
        
        if diff > 1e-10:
             print("  FAIL: Significant sensitivity to inactive parameter!")
             pass_all = False
        else:
             print("  PASS: No sensitivity.")
             
        # Restore
        evaluator.theta_base[idx] = original_val

    if pass_all:
        print("\nSUCCESS: M2 model is correctly isolated from M1 parameters.")
    else:
        print("\nFAILURE: M2 model is still coupled to M1 parameters.")

    # 6. Test M2 Parameter Sensitivity (Active)
    print("\n--- Testing M2 Parameter Sensitivity (Should be > 0) ---")
    for i, idx in enumerate(active_indices):
        theta_m2_pert = theta_m2_true.copy()
        theta_m2_pert[i] *= 1.1 # 10% change
        
        logL_pert = evaluator(theta_m2_pert)
        diff = abs(logL_pert - logL_base)
        
        print(f"Param idx {idx} (M2 param {i}): {theta_m2_true[i]} -> {theta_m2_pert[i]}")
        print(f"  LogL Change: {diff:.6f}")
        
        if diff < 1e-6:
            print("  WARNING: Low sensitivity to active parameter!")
        else:
            print("  PASS: Sensitivity detected.")

if __name__ == "__main__":
    test_m2_integration()
