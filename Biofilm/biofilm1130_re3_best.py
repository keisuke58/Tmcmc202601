import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, root
from scipy.stats import pearsonr
import pandas as pd
import os
import time as time_module

# 設定: 数値計算の警告抑制とグラフスタイル
np.seterr(all='ignore')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 100,
    'lines.linewidth': 2
})

# Create output folder
output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

# =============================================================================
# CONVERGENCE MONITOR (Early Stopping)
# =============================================================================
class ConvergenceMonitor:
    """Monitor convergence and implement early stopping"""
    def __init__(self, patience=5, tol=1e-4):
        self.patience = patience
        self.tol = tol
        self.best_loss = np.inf
        self.counter = 0
        self.history = []
        
    def check(self, current_loss):
        self.history.append(current_loss)
        
        if current_loss < self.best_loss - self.tol:
            self.best_loss = current_loss
            self.counter = 0
            return False  # Continue
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop if patience exceeded
    
    def reset(self):
        self.best_loss = np.inf
        self.counter = 0
        self.history = []

# =============================================================================
# 1. Physics Engine (Solver) - OPTIMIZED
# =============================================================================
class HierarchicalBiofilmSolver:
    def __init__(self, tolerance=1e-4):
        self.Kp1 = 1e-4
        self.Eta_val = 1.0
        self.tolerance = tolerance

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
        sample_indices = np.linspace(n_steps // n_samples, n_steps, n_samples, dtype=int)

        for step in range(1, n_steps + 1):
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
# 2. BAYESIAN LIKELIHOOD with Summary Statistics (Paper Eq. 29)
# =============================================================================
def likelihood_with_uncertainty(params, data, solver_func, tolerance=1e-4, n_mc_samples=50):
    """
    Likelihood function using summary statistics (mean and variance)
    Based on Paper Section 2.1.2, Equation (29)
    """
    # Generate stochastic samples with aleatory uncertainty (CoV = 0.5%)
    CoV = 0.005
    mc_outputs = []
    
    for _ in range(n_mc_samples):
        # Add aleatory uncertainty to parameters
        noisy_params = params * (1 + np.random.normal(0, CoV, len(params)))
        sim = solver_func(noisy_params)
        if sim is None:
            return -1e15
        mc_outputs.append(sim)
    
    mc_outputs = np.array(mc_outputs)  # Shape: (n_mc_samples, n_time_steps, n_species)
    
    # Compute mean and variance at each time step
    mu = np.mean(mc_outputs, axis=0)  # Shape: (n_time_steps, n_species)
    sigma2 = np.var(mc_outputs, axis=0) + 1e-8  # Avoid division by zero
    
    # Compute log-likelihood using Eq. (29)
    log_L = 0.0
    n_time, n_species = data.shape
    
    for k in range(n_time):
        for l in range(n_species):
            log_L += -0.5 * np.log(2 * np.pi * sigma2[k, l])
            log_L += -0.5 * (data[k, l] - mu[k, l])**2 / sigma2[k, l]
    
    return log_L

# Simplified objective for DE (maximize log-likelihood = minimize negative log-likelihood)
def obj_M1_bayesian(params, data, tolerance=1e-4):
    local_solver = HierarchicalBiofilmSolver(tolerance=tolerance)
    log_L = likelihood_with_uncertainty(params, data, local_solver.run_M1, tolerance, n_mc_samples=30)
    return -log_L  # Minimize negative log-likelihood

def obj_M2_bayesian(params, data, tolerance=1e-4):
    local_solver = HierarchicalBiofilmSolver(tolerance=tolerance)
    log_L = likelihood_with_uncertainty(params, data, local_solver.run_M2, tolerance, n_mc_samples=30)
    return -log_L

def obj_M3_bayesian(params, data, m1, m2, tolerance=1e-4):
    local_solver = HierarchicalBiofilmSolver(tolerance=tolerance)
    solver_func = lambda p: local_solver.run_M3(p, m1, m2)
    log_L = likelihood_with_uncertainty(params, data, solver_func, tolerance, n_mc_samples=30)
    return -log_L

# =============================================================================
# 3. TWO-STAGE OPTIMIZATION with Posterior Sampling
# =============================================================================
def two_stage_optimization_with_posterior(obj_func, bounds, args_tuple, name, n_posterior_samples=500):
    """
    Two-stage optimization that returns posterior samples
    Stage 1: Coarse global search
    Stage 2: Refined local search + Posterior sampling around optimum
    """
    print(f"\n  Stage 1 (Coarse): Broad search...")
    t_start = time_module.time()
    
    res_coarse = differential_evolution(
        obj_func, bounds, args=args_tuple,
        strategy='randtobest1bin',
        maxiter=10, popsize=8,
        workers=-1, updating='immediate',
        disp=False, polish=False
    )
    
    print(f"  Stage 1 completed in {time_module.time()-t_start:.1f}s, Loss: {res_coarse.fun:.6f}")
    
    print(f"  Stage 2 (Refined): Local optimization...")
    t_start = time_module.time()
    
    x_best = res_coarse.x
    bounds_refined = [(max(bounds[i][0], x-0.3), min(bounds[i][1], x+0.3)) 
                      for i, x in enumerate(x_best)]
    
    res_fine = differential_evolution(
        obj_func, bounds_refined, args=args_tuple,
        strategy='best1bin',
        maxiter=15, popsize=6,
        workers=-1, updating='immediate',
        disp=False, polish=True
    )
    
    print(f"  Stage 2 completed in {time_module.time()-t_start:.1f}s, Loss: {res_fine.fun:.6f}")
    
    # Stage 3: Generate posterior samples around the optimum
    print(f"  Stage 3 (Posterior Sampling): Generating {n_posterior_samples} samples...")
    t_start = time_module.time()
    
    posterior_samples = generate_posterior_samples(
        obj_func, res_fine.x, bounds_refined, args_tuple, n_posterior_samples
    )
    
    print(f"  Stage 3 completed in {time_module.time()-t_start:.1f}s")
    
    return res_fine, posterior_samples

def generate_posterior_samples(obj_func, map_estimate, bounds, args_tuple, n_samples):
    """
    Generate posterior samples using random walk around MAP estimate
    Simulates MCMC-like sampling
    """
    samples = []
    current = map_estimate.copy()
    current_loss = obj_func(current, *args_tuple)
    
    # Adaptive step size based on parameter ranges
    step_sizes = np.array([b[1] - b[0] for b in bounds]) * 0.05
    
    accept_count = 0
    
    for i in range(n_samples * 3):  # Generate more to ensure n_samples accepted
        # Propose new sample
        proposal = current + np.random.normal(0, step_sizes, len(current))
        
        # Check bounds
        in_bounds = all(bounds[j][0] <= proposal[j] <= bounds[j][1] for j in range(len(proposal)))
        
        if in_bounds:
            proposal_loss = obj_func(proposal, *args_tuple)
            
            # Metropolis-Hastings acceptance criterion
            # Accept if better, or with probability exp(-delta_loss)
            delta_loss = proposal_loss - current_loss
            accept_prob = np.exp(-delta_loss) if delta_loss > 0 else 1.0
            
            if np.random.rand() < accept_prob:
                current = proposal.copy()
                current_loss = proposal_loss
                accept_count += 1
        
        # Store sample (after burn-in)
        if i >= n_samples and len(samples) < n_samples:
            samples.append(current.copy())
        
        if len(samples) >= n_samples:
            break
    
    print(f"    Acceptance rate: {100*accept_count/(n_samples*3):.1f}%")
    
    return np.array(samples)

# =============================================================================
# 4. POSTERIOR ANALYSIS (Correlation, Visualization)
# =============================================================================
def analyze_posterior_correlation(samples, param_names):
    """Compute Pearson correlation coefficients like in Paper Fig 3"""
    n_params = len(param_names)
    corr_matrix = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(samples[:, i], samples[:, j])
                corr_matrix[i, j] = corr
    
    return corr_matrix

def plot_posterior_with_correlation(samples, param_names, true_values, filename):
    """Visualize posterior samples with correlation (Paper Fig 3, 8, 10, 12)"""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    corr_matrix = analyze_posterior_correlation(samples, param_names)
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: Histogram
                ax.hist(samples[:, i], bins=30, color='blue', alpha=0.6, density=True)
                ax.axvline(true_values[i], color='red', linestyle='--', linewidth=2, label='True')
                ax.set_ylabel('Density' if j == 0 else '')
                
            elif i > j:
                # Lower triangle: Scatter plot
                ax.scatter(samples[:, j], samples[:, i], alpha=0.3, s=1)
                ax.axvline(true_values[j], color='red', linestyle='--', alpha=0.5)
                ax.axhline(true_values[i], color='red', linestyle='--', alpha=0.5)
                
            else:
                # Upper triangle: Correlation coefficient
                corr = corr_matrix[i, j]
                ax.text(0.5, 0.5, f'ρ={corr:.3f}', 
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # Labels
            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticklabels([])
            
            if j == 0 and i != j:
                ax.set_ylabel(param_names[i])
            elif j != 0:
                ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")
    
    return corr_matrix

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
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# =============================================================================
# MAIN EXECUTION - FULL BAYESIAN UPDATING
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("=== FULL BAYESIAN UPDATING with Paper Methods ===")
    print("="*70)

    solver = HierarchicalBiofilmSolver(tolerance=1e-4)

    # True Parameters (Paper Case II)
    TRUE_M1 = [0.8, 2.0, 1.0, 0.1, 0.2]
    TRUE_M2 = [1.5, 1.0, 2.0, 0.3, 0.4]
    TRUE_M3 = [2.0, 1.0, 2.0, 1.0]

    print("\n[Step 0] Generating Data with Aleatory Uncertainty (CoV=0.5%)...")
    # Generate data with aleatory uncertainty
    CoV = 0.005
    d_M1 = solver.run_M1(TRUE_M1) + np.random.normal(0, 0.002, (10, 2))
    d_M2 = solver.run_M2(TRUE_M2) + np.random.normal(0, 0.002, (10, 2))
    d_M3 = solver.run_M3(TRUE_M3, TRUE_M1, TRUE_M2) + np.random.normal(0, 0.002, (10, 4))

    # --- Stage 1: M1 (Bayesian Updating with Posterior) ---
    print("\n" + "="*70)
    print("[Step 1] Bayesian Updating for M1 (Species 1 & 2)")
    print("="*70)
    t_total = time_module.time()
    
    res1, posterior_M1 = two_stage_optimization_with_posterior(
        obj_M1_bayesian, [(0, 3)]*5, (d_M1, 1e-4), "M1", n_posterior_samples=500
    )
    est_M1 = res1.x
    
    print(f"\n[M1] Total time: {time_module.time()-t_total:.1f}s")
    print(f"[M1] MAP Estimate: {np.round(est_M1, 4)}")
    print(f"[M1] True: {TRUE_M1}")
    
    # Visualize posterior with correlation
    param_names_M1 = ['a11', 'a12', 'a22', 'b1', 'b2']
    corr_M1 = plot_posterior_with_correlation(
        posterior_M1, param_names_M1, TRUE_M1, "fig8_M1_posterior.png"
    )
    print("\n[M1] Correlation Matrix:")
    print(pd.DataFrame(corr_M1, index=param_names_M1, columns=param_names_M1).round(3))
    
    plot_fit(d_M1, solver.run_M1(est_M1), "Fig 9: M1 Fit (Species 1 & 2)", [1, 2], "fig9_M1_fit.png")

    # --- Stage 2: M2 ---
    print("\n" + "="*70)
    print("[Step 2] Bayesian Updating for M2 (Species 3 & 4)")
    print("="*70)
    t_total = time_module.time()
    
    res2, posterior_M2 = two_stage_optimization_with_posterior(
        obj_M2_bayesian, [(0, 3)]*5, (d_M2, 1e-4), "M2", n_posterior_samples=500
    )
    est_M2 = res2.x
    
    print(f"\n[M2] Total time: {time_module.time()-t_total:.1f}s")
    print(f"[M2] MAP Estimate: {np.round(est_M2, 4)}")
    print(f"[M2] True: {TRUE_M2}")
    
    # Visualize posterior with correlation
    param_names_M2 = ['a33', 'a34', 'a44', 'b3', 'b4']
    corr_M2 = plot_posterior_with_correlation(
        posterior_M2, param_names_M2, TRUE_M2, "fig10_M2_posterior.png"
    )
    print("\n[M2] Correlation Matrix:")
    print(pd.DataFrame(corr_M2, index=param_names_M2, columns=param_names_M2).round(3))
    
    plot_fit(d_M2, solver.run_M2(est_M2), "Fig 11: M2 Fit (Species 3 & 4)", [3, 4], "fig11_M2_fit.png")

    # --- Stage 3: M3 ---
    print("\n" + "="*70)
    print("[Step 3] Bayesian Updating for M3 (Cross-Interactions)")
    print("="*70)
    t_total = time_module.time()
    
    res3, posterior_M3 = two_stage_optimization_with_posterior(
        obj_M3_bayesian, [(0, 3)]*4, (d_M3, est_M1, est_M2, 1e-4), "M3", n_posterior_samples=500
    )
    est_M3 = res3.x
    
    print(f"\n[M3] Total time: {time_module.time()-t_total:.1f}s")
    print(f"[M3] MAP Estimate: {np.round(est_M3, 4)}")
    print(f"[M3] True: {TRUE_M3}")
    
    # Visualize posterior with correlation
    param_names_M3 = ['a13', 'a14', 'a23', 'a24']
    corr_M3 = plot_posterior_with_correlation(
        posterior_M3, param_names_M3, TRUE_M3, "fig12_M3_posterior.png"
    )
    print("\n[M3] Correlation Matrix:")
    print(pd.DataFrame(corr_M3, index=param_names_M3, columns=param_names_M3).round(3))
    
    plot_fit(d_M3, solver.run_M3(est_M3, est_M1, est_M2), "Fig 13: M3 Fit (All Species)", 
             [1, 2, 3, 4], "fig13_M3_fit.png")

    # --- Final Results ---
    print("\n" + "="*70)
    print("=== ESTIMATION RESULT ===")
    print("="*70)

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

    print("\n=== FINAL ESTIMATED PARAMETERS (Sorted) ===")
    df = pd.DataFrame({"Parameter": labels, "True": all_true, "Estimated": np.round(all_est, 4)})
    df["Error"] = np.round(df["Estimated"] - df["True"], 4)
    df["Error %"] = np.round(100 * df["Error"] / (df["True"] + 1e-10), 2)
    print(df)

    # --- Plot Fig 14 ---
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
    plt.savefig(os.path.join(output_folder, "fig14_parameters_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: fig14_parameters_comparison.png")

    # --- Validation (Fig 15) ---
    print("\n" + "="*70)
    print("Running Validation (M3val) - Antibiotic Shock at t=0.5")
    print("="*70)
    t, val = solver.run_M3_val(est_M1, est_M2, est_M3)

    if val is not None:
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
        plt.savefig(os.path.join(output_folder, "fig15_validation.png"), dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: fig15_validation.png")
    
    print("\n" + "="*70)
    print("=== ALL BAYESIAN UPDATING COMPLETED ===")
    print("="*70)