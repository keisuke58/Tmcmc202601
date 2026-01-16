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

# Create output folder
output_folder = "figures"
os.makedirs(output_folder, exist_ok=True)

# =============================================================================
# 1. Physics Engine (Solver)
# =============================================================================
class HierarchicalBiofilmSolver:
    def __init__(self):
        self.Kp1 = 1e-4
        self.Eta_val = 1.0

    def _run_simulation(self, n_species, params_A, params_B,
                        initial_phi, alpha_val, c_val, dt, n_steps):

        A = np.array(params_A)
        b_diag = np.array(params_B)

        phi_vec = np.array([initial_phi] * n_species)
        phi0 = 1.0 - np.sum(phi_vec)
        psi_vec = np.array([0.999] * n_species)
        gamma = 1e-3
        g_current = np.concatenate([phi_vec, [phi0], psi_vec, [gamma]])

        traj_data = []
        # Sampling: Ensure exactly 20 points
        sample_indices = np.linspace(n_steps // 20, n_steps, 20, dtype=int)

        for step in range(1, n_steps + 1):
            # Solver
            sol = root(self._residual_func, g_current,
                       args=(g_current, n_species, dt, A, b_diag, alpha_val, c_val),
                       method='lm', tol=1e-5) # Optimized tolerance

            if not sol.success:
                sol = root(self._residual_func, g_current,
                           args=(g_current, n_species, dt, A, b_diag, alpha_val, c_val),
                           method='hybr', tol=1e-5)
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
            alpha = 50.0 if step > 750 else 0.0 # t=0.5
            sol = root(self._residual_func, g_current, args=(g_current, 4, dt, A, B, alpha, c_val), method='lm', tol=1e-5)
            if not sol.success: return None, None
            g_new = sol.x
            g_new[0:9] = np.clip(g_new[0:9], 1e-6, 5.0)
            g_current = g_new.copy()
            if step % 10 == 0:
                traj.append(g_current[0:4] * g_current[5:9])
                t_axis.append(step/1500.0)
        return np.array(t_axis), np.array(traj)

# =============================================================================
# 2. Optimization Logic (Parallel Ready)
# =============================================================================
solver = HierarchicalBiofilmSolver()

def obj_M1(params, data):
    local_solver = HierarchicalBiofilmSolver()
    sim = local_solver.run_M1(params)
    if sim is None: return 1e15
    return np.mean((sim - data)**2)

def obj_M2(params, data):
    local_solver = HierarchicalBiofilmSolver()
    sim = local_solver.run_M2(params)
    if sim is None: return 1e15
    return np.mean((sim - data)**2)

def obj_M3(params, data, m1, m2):
    local_solver = HierarchicalBiofilmSolver()
    sim = local_solver.run_M3(params, m1, m2)
    if sim is None: return 1e15
    return np.mean((sim - data)**2)

def plot_fit(data, fit, title, species_indices, filename):
    # Normalized time axis: 0 to 1
    t_norm = np.linspace(0, 1, len(data))
    # All species colors: 1=blue, 2=orange, 3=green, 4=red
    all_colors = ['blue', 'orange', 'green', 'red']

    plt.figure(figsize=(8, 5))
    for i, sp_idx in enumerate(species_indices):
        color = all_colors[sp_idx - 1]  # species_indices are 1-indexed
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
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=== Case II: Parallel Optimization (Full Figures) ===")

    # True Parameters
    TRUE_M1 = [0.8, 2.0, 1.0, 0.1, 0.2]
    TRUE_M2 = [1.5, 1.0, 2.0, 0.3, 0.4]
    TRUE_M3 = [2.0, 1.0, 2.0, 1.0]

    print("\n[Step 0] Generating Data...")
    d_M1 = solver.run_M1(TRUE_M1) + np.random.normal(0, 0.002, (20, 2))
    d_M2 = solver.run_M2(TRUE_M2) + np.random.normal(0, 0.002, (20, 2))
    d_M3 = solver.run_M3(TRUE_M3, TRUE_M1, TRUE_M2) + np.random.normal(0, 0.002, (20, 4))

    # --- Parallel Settings ---
    STRATEGY = 'best1bin'
    MAX_ITER = 10
    POP_SIZE = 10
    WORKERS = -1

    # --- Stage 1: M1 ---
    print("\n[Step 1] Estimating M1 (Parallel)...")
    res1 = differential_evolution(obj_M1, [(0,3)]*5, args=(d_M1,),
                                  strategy=STRATEGY, maxiter=MAX_ITER, popsize=POP_SIZE,
                                  workers=WORKERS, updating='deferred', disp=True)
    est_M1 = res1.x

    # Plot Fig 9
    plot_fit(d_M1, solver.run_M1(est_M1), "Fig 9: M1 Fit (Species 1 & 2)", [1, 2], "fig9_M1_fit.png")

    # --- Stage 2: M2 ---
    print("\n[Step 2] Estimating M2 (Parallel)...")
    res2 = differential_evolution(obj_M2, [(0,3)]*5, args=(d_M2,),
                                  strategy=STRATEGY, maxiter=MAX_ITER, popsize=POP_SIZE,
                                  workers=WORKERS, updating='deferred', disp=True)
    est_M2 = res2.x

    # Plot Fig 11
    plot_fit(d_M2, solver.run_M2(est_M2), "Fig 11: M2 Fit (Species 3 & 4)", [3, 4], "fig11_M2_fit.png")

    # --- Stage 3: M3 ---
    print("\n[Step 3] Estimating M3 (Parallel)...")
    res3 = differential_evolution(obj_M3, [(0,3)]*4, args=(d_M3, est_M1, est_M2),
                                  strategy=STRATEGY, maxiter=MAX_ITER, popsize=POP_SIZE,
                                  workers=WORKERS, updating='deferred', disp=True)
    est_M3 = res3.x

    # Plot Fig 13
    plot_fit(d_M3, solver.run_M3(est_M3, est_M1, est_M2), "Fig 13: M3 Fit (All Species)", [1, 2, 3, 4], "fig13_M3_fit.png")

    # --- Final Results ---
    print("\n=== ESTIMATION RESULT ===")

    raw_est = np.concatenate([est_M1, est_M2, est_M3])
    raw_true = np.concatenate([TRUE_M1, TRUE_M2, TRUE_M3])

    raw_labels = np.array([
        "a11","a12","a22","b1","b2",
        "a33","a34","a44","b3","b4",
        "a13","a14","a23","a24"
    ])

    new_order_indices = [
        0, 1, 10, 11, 2, 12, 13, 5, 6, 7, 3, 4, 8, 9
    ]

    all_est = raw_est[new_order_indices]
    all_true = raw_true[new_order_indices]
    labels = raw_labels[new_order_indices]

    # --- Print Table ---
    print("\n=== FINAL ESTIMATED PARAMETERS (Sorted) ===")
    df = pd.DataFrame({"Parameter": labels, "True": all_true, "Estimated": np.round(all_est, 4)})
    df["Error"] = df["Estimated"] - df["True"]
    print(df)

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
    plt.savefig(os.path.join(output_folder, "fig14_parameters_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: fig14_parameters_comparison.png")

    # Plot Fig 15 (Validation with Normalized Time)
    print("\nRunning Validation (M3val)...")
    t, val = solver.run_M3_val(est_M1, est_M2, est_M3)

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