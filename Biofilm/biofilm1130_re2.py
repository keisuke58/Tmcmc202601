import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, root
import sys
import time
import pandas as pd
import os

# 設定: 数値計算の警告抑制とグラフスタイル
np.seterr(all='ignore')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 100,
    'lines.linewidth': 2
})

# Create output folders
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"results_{timestamp}"
figures_folder = os.path.join(output_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

# Log file
log_file = os.path.join(output_folder, "optimization_log.txt")
results_file = os.path.join(output_folder, "estimation_results.csv")

def log_message(message):
    """Print and save to log file"""
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# =============================================================================
# 1. Physics Engine (Solver) - OPTIMIZED
# =============================================================================
class HierarchicalBiofilmSolver:
    def __init__(self, tolerance=1e-4):
        self.Kp1 = 1e-4
        self.Eta_val = 1.0
        self.tolerance = tolerance  # Adjustable tolerance

    def _run_simulation(self, n_species, params_A, params_B,
                        initial_phi, alpha_val, c_val, dt, n_steps, n_samples=10):

        A = np.array(params_A)
        b_diag = np.array(params_B)

        phi_vec = np.array([initial_phi] * n_species)
        phi0 = 1.0 - np.sum(phi_vec)
        psi_vec = np.array([0.999] * n_species)
        gamma = 1e-3
        g_current = np.concatenate([phi_vec, [phi0], psi_vec, [gamma]])

        traj_data = []
        # OPTIMIZED: Reduced sampling (10 points instead of 20)
        sample_indices = np.linspace(n_steps // n_samples, n_steps, n_samples, dtype=int)

        for step in range(1, n_steps + 1):
            # OPTIMIZED: Use relaxed tolerance
            sol = root(self._residual_func, g_current,
                       args=(g_current, n_species, dt, A, b_diag, alpha_val, c_val),
                       method='lm', tol=self.tolerance)

            if not sol.success:
                sol = root(self._residual_func, g_current,
                           args=(g_current, n_species, dt, A, b_diag, alpha_val, c_val),
                           method='hybr', tol=self.tolerance)
                if not sol.success: return None

            g_new = sol.x
            g_new[0:n_species+1] = np.clip(g_new[0:n_species+1], 1e-6, 1.0-1e-6)
            g_new[n_species+1:2*n_species+1] = np.clip(g_new[n_species+1:2*n_species+1], 0.1, 5.0)
            g_current = g_new.copy()

            if step in sample_indices:
                phi = g_current[0:n_species]
                psi = g_current[n_species+1:2*n_species+1]
                traj_data.append(phi * psi)

        return np.array(traj_data)

    def _residual_func(self, g_new, g_old, n, dt, A, b_diag, alpha, c_val):
        phi = g_new[0:n]; phi0 = g_new[n]; psi = g_new[n+1 : 2*n+1]; gamma = g_new[-1]
        phidot = (phi - g_old[0:n]) / dt
        phi0dot = (phi0 - g_old[n]) / dt
        psidot = (psi - g_old[n+1 : 2*n+1]) / dt

        Q = np.zeros_like(g_new)
        Eta_vec = np.full(n, self.Eta_val)
        CapitalPhi = phi * psi
        Interaction_dot = A @ CapitalPhi

        denom_phi = np.sign((phi-1)**3 * phi**3) * np.maximum(np.abs((phi-1)**3 * phi**3), 1e-12)
        Q[0:n] = (self.Kp1 * (2. - 4.*phi)) / denom_phi + (1./Eta_vec)*(gamma + (Eta_vec + Eta_vec*psi**2)*phidot + Eta_vec*phi*psi*psidot) - (c_val/Eta_vec) * psi * Interaction_dot

        denom_phi0 = np.sign((phi0-1)**3 * phi0**3) * np.maximum(np.abs((phi0-1)**3 * phi0**3), 1e-12)
        Q[n] = gamma + (self.Kp1*(2.-4.*phi0))/denom_phi0 + phi0dot

        denom_psiA = np.sign((psi-1)**2 * psi**3) * np.maximum(np.abs((psi-1)**2 * psi**3), 1e-12)
        denom_psiB = np.sign((psi-1)**3 * psi**2) * np.maximum(np.abs((psi-1)**3 * psi**2), 1e-12)
        Q[n+1 : 2*n+1] = (-2.*self.Kp1)/denom_psiA - (2.*self.Kp1)/denom_psiB + (b_diag * alpha / Eta_vec) * psi + phi*psi*phidot + phi**2*psidot - (c_val/Eta_vec) * phi * Interaction_dot

        Q[-1] = np.sum(phi) + phi0 - 1.0
        return Q

    # --- Wrappers (Table 3) ---
    def run_M1(self, params):
        p_a11, p_a12, p_a22, p_b1, p_b2 = params
        A = [[p_a11, p_a12], [p_a12, p_a22]]
        B = [p_b1, p_b2]
        return self._run_simulation(2, A, B, 0.2, 100.0, 100.0, 1e-5, 2500)

    def run_M2(self, params):
        p_a33, p_a34, p_a44, p_b3, p_b4 = params
        A = [[p_a33, p_a34], [p_a34, p_a44]]
        B = [p_b3, p_b4]
        return self._run_simulation(2, A, B, 0.2, 10.0, 100.0, 1e-5, 5000)

    def run_M3(self, params, known_M1, known_M2):
        a13, a14, a23, a24 = params
        a11, a12, a22, b1, b2 = known_M1
        a33, a34, a44, b3, b4 = known_M2
        A = [[a11, a12, a13, a14], [a12, a22, a23, a24], [a13, a23, a33, a34], [a14, a24, a34, a44]]
        B = [b1, b2, b3, b4]
        return self._run_simulation(4, A, B, 0.02, 0.0, 25.0, 1e-4, 750)

    def run_M3_val(self, est_M1, est_M2, est_M3):
        a11, a12, a22, b1, b2 = est_M1
        a33, a34, a44, b3, b4 = est_M2
        a13, a14, a23, a24 = est_M3
        A = [[a11, a12, a13, a14], [a12, a22, a23, a24], [a13, a23, a33, a34], [a14, a24, a34, a44]]
        B = np.array([b1, b2, b3, b4])

        dt = 1e-4; n_steps = 1500; c_val = 25.0
        g_current = np.concatenate([[0.02]*4, [0.92], [0.999]*4, [1e-3]])
        traj = []; t_axis = []

        for step in range(1, n_steps + 1):
            alpha = 50.0 if step > 750 else 0.0
            sol = root(self._residual_func, g_current, args=(g_current, 4, dt, A, B, alpha, c_val), 
                      method='lm', tol=self.tolerance)
            if not sol.success: return None, None
            g_new = sol.x
            g_new[0:9] = np.clip(g_new[0:9], 1e-6, 5.0)
            g_current = g_new.copy()
            if step % 10 == 0:
                traj.append(g_current[0:4] * g_current[5:9])
                t_axis.append(step/1500.0)
        return np.array(t_axis), np.array(traj)

# =============================================================================
# 2. OPTIMIZED Optimization Logic with TWO-STAGE approach
# =============================================================================
def obj_M1(params, data, tolerance=1e-4):
    local_solver = HierarchicalBiofilmSolver(tolerance=tolerance)
    sim = local_solver.run_M1(params)
    if sim is None: return 1e15
    return np.mean((sim - data)**2)

def obj_M2(params, data, tolerance=1e-4):
    local_solver = HierarchicalBiofilmSolver(tolerance=tolerance)
    sim = local_solver.run_M2(params)
    if sim is None: return 1e15
    return np.mean((sim - data)**2)

def obj_M3(params, data, m1, m2, tolerance=1e-4):
    local_solver = HierarchicalBiofilmSolver(tolerance=tolerance)
    sim = local_solver.run_M3(params, m1, m2)
    if sim is None: return 1e15
    return np.mean((sim - data)**2)

def two_stage_optimization(obj_func, bounds, args_tuple, name):
    """Two-stage optimization: coarse then refined"""
    log_message(f"\n  Stage 1 (Coarse): Broad search...")
    t_start = time.time()
    
    # Stage 1: Coarse search
    res_coarse = differential_evolution(
        obj_func, bounds, args=args_tuple,
        strategy='randtobest1bin',
        maxiter=15, popsize=10,
        workers=-1, updating='immediate',
        disp=False, polish=False
    )
    
    t_stage1 = time.time() - t_start
    log_message(f"  Stage 1 completed in {t_stage1:.1f}s, Loss: {res_coarse.fun:.6f}")
    
    # Stage 2: Refined search around best solution
    log_message(f"  Stage 2 (Refined): Local optimization...")
    t_start = time.time()
    
    x_best = res_coarse.x
    bounds_refined = [(max(bounds[i][0], x-0.3), min(bounds[i][1], x+0.3)) 
                      for i, x in enumerate(x_best)]
    
    res_fine = differential_evolution(
        obj_func, bounds_refined, args=args_tuple,
        strategy='best1bin',
        maxiter=20, popsize=8,
        workers=-1, updating='immediate',
        disp=False, polish=True
    )
    
    t_stage2 = time.time() - t_start
    log_message(f"  Stage 2 completed in {t_stage2:.1f}s, Loss: {res_fine.fun:.6f}")
    
    # Save timing data
    res_fine.stage1_time = t_stage1
    res_fine.stage2_time = t_stage2
    res_fine.total_time = t_stage1 + t_stage2
    
    return res_fine

def plot_fit(data, fit, title, species_indices, filename):
    t_norm = np.linspace(0, 1, len(data))
    all_colors = ['blue', 'orange', 'green', 'red']
    
    plt.figure(figsize=(8, 5))
    for i, sp_idx in enumerate(species_indices):
        color = all_colors[sp_idx - 1]
        plt.plot(t_norm, data[:, i], 'o', color=color, alpha=0.4, label=f'Data Species {sp_idx}')
        plt.plot(t_norm, fit[:, i], '-', color=color, linewidth=2, label=f'Realizations Species {sp_idx}')
    plt.title(title)
    plt.xlabel("Normalized Time $t$")
    plt.ylabel("Living Biomass $\overline{\Phi}(t)$")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, filename), dpi=300, bbox_inches='tight')
    plt.show()
    log_message(f"Saved: {filename}")

# =============================================================================
# MAIN EXECUTION - OPTIMIZED
# =============================================================================
if __name__ == "__main__":
    overall_start = time.time()
    
    log_message("="*70)
    log_message("=== OPTIMIZED Case II: Two-Stage Parallel Optimization ===")
    log_message(f"=== Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    log_message(f"=== Output folder: {output_folder} ===")
    log_message("="*70)

    # OPTIMIZED: Relaxed tolerance for faster simulation
    solver = HierarchicalBiofilmSolver(tolerance=1e-4)

    # True Parameters
    TRUE_M1 = [0.8, 2.0, 1.0, 0.1, 0.2]
    TRUE_M2 = [1.5, 1.0, 2.0, 0.3, 0.4]
    TRUE_M3 = [2.0, 1.0, 2.0, 1.0]

    log_message("\n[Step 0] Generating Data...")
    d_M1 = solver.run_M1(TRUE_M1) + np.random.normal(0, 0.002, (10, 2))
    d_M2 = solver.run_M2(TRUE_M2) + np.random.normal(0, 0.002, (10, 2))
    d_M3 = solver.run_M3(TRUE_M3, TRUE_M1, TRUE_M2) + np.random.normal(0, 0.002, (10, 4))
    
    # Save data
    np.savetxt(os.path.join(output_folder, "data_M1.csv"), d_M1, delimiter=',', header='Species1,Species2', comments='')
    np.savetxt(os.path.join(output_folder, "data_M2.csv"), d_M2, delimiter=',', header='Species3,Species4', comments='')
    np.savetxt(os.path.join(output_folder, "data_M3.csv"), d_M3, delimiter=',', header='Species1,Species2,Species3,Species4', comments='')

    # --- Stage 1: M1 (Two-stage optimization) ---
    log_message("\n" + "="*70)
    log_message("[Step 1] Estimating M1 with Two-Stage Optimization...")
    log_message("="*70)
    t_total = time.time()
    
    res1 = two_stage_optimization(obj_M1, [(0, 3)]*5, (d_M1, 1e-4), "M1")
    est_M1 = res1.x
    
    t_M1 = time.time() - t_total
    log_message(f"\n[M1] Total time: {t_M1:.1f}s")
    plot_fit(d_M1, solver.run_M1(est_M1), "Fig 9: M1 Fit (Species 1 & 2)", [1, 2], "fig9_M1_fit.png")

    # --- Stage 2: M2 ---
    log_message("\n" + "="*70)
    log_message("[Step 2] Estimating M2 with Two-Stage Optimization...")
    log_message("="*70)
    t_total = time.time()
    
    res2 = two_stage_optimization(obj_M2, [(0, 3)]*5, (d_M2, 1e-4), "M2")
    est_M2 = res2.x
    
    t_M2 = time.time() - t_total
    log_message(f"\n[M2] Total time: {t_M2:.1f}s")
    plot_fit(d_M2, solver.run_M2(est_M2), "Fig 11: M2 Fit (Species 3 & 4)", [3, 4], "fig11_M2_fit.png")

    # --- Stage 3: M3 ---
    log_message("\n" + "="*70)
    log_message("[Step 3] Estimating M3 with Two-Stage Optimization...")
    log_message("="*70)
    t_total = time.time()
    
    res3 = two_stage_optimization(obj_M3, [(0, 3)]*4, (d_M3, est_M1, est_M2, 1e-4), "M3")
    est_M3 = res3.x
    
    t_M3 = time.time() - t_total
    log_message(f"\n[M3] Total time: {t_M3:.1f}s")
    plot_fit(d_M3, solver.run_M3(est_M3, est_M1, est_M2), "Fig 13: M3 Fit (All Species)", 
             [1, 2, 3, 4], "fig13_M3_fit.png")

    # --- Final Results ---
    log_message("\n" + "="*70)
    log_message("=== ESTIMATION RESULT ===")
    log_message("="*70)

    raw_est = np.concatenate([est_M1, est_M2, est_M3])
    raw_true = np.concatenate([TRUE_M1, TRUE_M2, TRUE_M3])

    raw_labels = np.array([
        "a11","a12","a22","b1","b2",
        "a33","a34","a44","b3","b4",
        "a13","a14","a23","a24"
    ])

    new_order_indices = [0, 1, 10, 11, 2, 12, 13, 5, 6, 7, 3, 4, 8, 9]
    all_est = raw_est[new_order_indices]
    all_true = raw_true[new_order_indices]
    labels = raw_labels[new_order_indices]

    # --- Print Table ---
    log_message("\n=== FINAL ESTIMATED PARAMETERS (Sorted) ===")
    df = pd.DataFrame({"Parameter": labels, "True": all_true, "Estimated": np.round(all_est, 4)})
    df["Error"] = np.round(df["Estimated"] - df["True"], 4)
    df["Error %"] = np.round(100 * df["Error"] / df["True"], 2)
    log_message("\n" + df.to_string(index=False))
    
    # Save results to CSV
    df.to_csv(results_file, index=False)
    log_message(f"\nResults saved to: {results_file}")

    # --- Plot Fig 14 (Sorted Bar Chart) ---
    plt.figure(figsize=(14, 6))
    x = np.arange(len(all_true))

    plt.bar(x + 0.2, all_true, 0.4, label='True Mean', color='orange', alpha=0.8)
    plt.bar(x - 0.2, all_est, 0.4, label='Posterior Mean', color='blue', alpha=0.8)

    plt.xticks(x, labels, fontsize=11)
    plt.title("Fig 14: Parameters Comparison", fontsize=16)
    plt.ylabel("Parameter Values", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    plt.axvline(x=9.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(4.5, max(all_true)*1.05, "Interaction Parameters ($A$)", ha='center', fontsize=12)
    plt.text(11.5, max(all_true)*1.05, "Sensitivity ($B$)", ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "fig14_parameters_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()
    log_message("Saved: fig14_parameters_comparison.png")

    # Plot Fig 15 (Validation with Normalized Time)
    log_message("\n" + "="*70)
    log_message("Running Validation (M3val)...")
    log_message("="*70)
    t_val_start = time.time()
    t, val = solver.run_M3_val(est_M1, est_M2, est_M3)
    t_val = time.time() - t_val_start

    plt.figure(figsize=(10, 6))
    all_colors = ['blue', 'orange', 'green', 'red']

    plt.axvspan(0.0, 0.5, color='blue', alpha=0.05, label='Updating Phase')
    plt.axvspan(0.5, 1.0, color='red', alpha=0.05, label='Prediction Phase')

    for i in range(4):
        plt.plot(t, val[:, i], color=all_colors[i], linewidth=2, label=f'Realizations Species {i+1}')
    plt.axvline(x=0.5, color='k', linestyle='--', label='Antibiotics ON')
    plt.title("Fig 15: Validation with Antibiotics Shock")
    plt.xlabel("Normalized Time $t$")
    plt.ylabel("Living Biomass $\overline{\Phi}(t)$")
    plt.xlim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, "fig15_validation.png"), dpi=300, bbox_inches='tight')
    plt.show()
    log_message("Saved: fig15_validation.png")
    log_message(f"Validation time: {t_val:.1f}s")
    
    # Save validation data
    val_data = np.column_stack([t, val])
    np.savetxt(os.path.join(output_folder, "validation_M3.csv"), val_data, delimiter=',', 
               header='Time,Species1,Species2,Species3,Species4', comments='')
    
    # --- Summary Report ---
    overall_time = time.time() - overall_start
    
    log_message("\n" + "="*70)
    log_message("=== COMPUTATION TIME SUMMARY ===")
    log_message("="*70)
    log_message(f"M1 Optimization:  {t_M1:.1f}s  (Stage1: {res1.stage1_time:.1f}s, Stage2: {res1.stage2_time:.1f}s)")
    log_message(f"M2 Optimization:  {t_M2:.1f}s  (Stage1: {res2.stage1_time:.1f}s, Stage2: {res2.stage2_time:.1f}s)")
    log_message(f"M3 Optimization:  {t_M3:.1f}s  (Stage1: {res3.stage1_time:.1f}s, Stage2: {res3.stage2_time:.1f}s)")
    log_message(f"Validation:       {t_val:.1f}s")
    log_message(f"-" * 70)
    log_message(f"TOTAL TIME:       {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    log_message(f"Finished at:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    log_message("\n" + "="*70)
    log_message("=== ALL OPTIMIZATIONS COMPLETED ===")
    log_message(f"=== All results saved in: {output_folder} ===")
    log_message("="*70)
    
    # Save timing summary to separate file
    timing_summary = {
        'Task': ['M1 Stage1', 'M1 Stage2', 'M1 Total', 'M2 Stage1', 'M2 Stage2', 'M2 Total', 
                 'M3 Stage1', 'M3 Stage2', 'M3 Total', 'Validation', 'Overall'],
        'Time (s)': [res1.stage1_time, res1.stage2_time, t_M1, 
                     res2.stage1_time, res2.stage2_time, t_M2,
                     res3.stage1_time, res3.stage2_time, t_M3, t_val, overall_time]
    }
    pd.DataFrame(timing_summary).to_csv(os.path.join(output_folder, "timing_summary.csv"), index=False)
    log_message(f"\nTiming summary saved to: timing_summary.csv")