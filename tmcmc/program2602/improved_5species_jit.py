# -*- coding: utf-8 -*-
"""
improved_5species_jit.py

✅ Goal
- Extension of "improved1207_paper_jit.py" to support 5 species.
- Used for generating synthetic data for the missing 5th species.

✅ State g (12,):
    [phi1, phi2, phi3, phi4, phi5, phi0,  psi1, psi2, psi3, psi4, psi5, gamma]
    Indices:
    0-4: phi1..phi5
    5:   phi0
    6-10: psi1..psi5
    11:  gamma

Active species: subset of {0,1,2,3,4}  (phi0, gamma are always present)

Theta (20,) order:
    M1 (5): a11, a12, a22, b1, b2
    M2 (5): a33, a34, a44, b3, b4
    M3 (4): a13, a14, a23, a24
    M4 (2): a55, b5
    M5 (4): a15, a25, a35, a45
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    njit = None


# =============================================================================
# 0. TRUE THETA (Extended for 5 species)
# =============================================================================

def get_theta_true() -> np.ndarray:
    """
    True parameter vector used for synthetic data generation (Case II + 5th Species).
    Order follows THETA_NAMES.
    """
    return np.array([
        0.8, 2.0, 1.0, 0.1, 0.2,  # M1 block (S1, S2)
        1.5, 1.0, 2.0, 0.3, 0.4,  # M2 block (S3, S4)
        2.0, 1.0, 2.0, 1.0,       # M3 cross-interaction block (S1-S2 <-> S3-S4)
        1.2, 0.25,                # M4 block (S5 self: a55, b5) - Hypothetical values
        1.0, 1.0, 1.0, 1.0        # M5 block (S5 cross: a15, a25, a35, a45) - Hypothetical values
    ], dtype=np.float64)


# =============================================================================
# 1. RESIDUAL + JACOBIAN KERNELS (5 Species)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _compute_Q_vector_numba(Q, phi_new, phi0_new, psi_new, gamma_new,
                                phi_old, phi0_old, psi_old,
                                dt, Kp1, Eta_vec, Eta_phi_vec,
                                c_val, alpha_val, K_hill, n_hill, A, b_diag, active_mask):
        # --- Stabilization: clip to [eps, 1-eps] ---
        # And strict locking for inactive species
        eps = 1e-12
        for i in range(5):
            if active_mask[i] == 1:
                if phi_new[i] < eps:
                    phi_new[i] = eps
                elif phi_new[i] > 1.0 - eps:
                    phi_new[i] = 1.0 - eps

                if psi_new[i] < eps:
                    psi_new[i] = eps
                elif psi_new[i] > 1.0 - eps:
                    psi_new[i] = 1.0 - eps
            else:
                phi_new[i] = 0.0
                psi_new[i] = 0.0

        if phi0_new < eps:
            phi0_new = eps
        elif phi0_new > 1.0 - eps:
            phi0_new = 1.0 - eps

        # Q is pre-allocated
        
        phidot = (phi_new - phi_old) / dt
        phi0dot = (phi0_new - phi0_old) / dt
        psidot = (psi_new - psi_old) / dt

        Interaction = A @ (phi_new * psi_new)

        # --- HILL FUNCTION GATING FOR P. GINGIVALIS (Index 4) ---
        # Gate interactions based on F. nucleatum (Index 3) concentration
        if K_hill > 1e-9:
            fn_conc = phi_new[3] * psi_new[3]
            if fn_conc < 0.0: fn_conc = 0.0
            # Use safe power calculation
            num = fn_conc**n_hill
            den = K_hill**n_hill + num
            if den < 1e-12:
                h_val = 0.0
            else:
                h_val = num / den
            
            # Apply to P. gingivalis interaction term
            Interaction[4] *= h_val
        # --------------------------------------------------------

        # Q[0..4] for phi_i
        for i in range(5):
            if active_mask[i] == 1:
                term1 = (Kp1 * (2.0 - 4.0 * phi_new[i])) / ((phi_new[i] - 1.0)**3 * phi_new[i]**3)
                term2 = (1.0 / Eta_vec[i]) * (gamma_new
                                              + (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) * phidot[i]
                                              + Eta_vec[i] * phi_new[i] * psi_new[i] * psidot[i])
                term3 = (c_val / Eta_vec[i]) * psi_new[i] * Interaction[i]
                Q[i] = term1 + term2 - term3
            else:
                # Inactive species: equation is phi = 0
                Q[i] = phi_new[i]

        # Q[5] for phi0
        Q[5] = gamma_new + (Kp1 * (2.0 - 4.0 * phi0_new)) / ((phi0_new - 1.0)**3 * phi0_new**3) + phi0dot

        # Q[6..10] for psi_i
        for i in range(5):
            if active_mask[i] == 1:
                term1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0)**2 * psi_new[i]**3) \
                        - (2.0 * Kp1) / ((psi_new[i] - 1.0)**3 * psi_new[i]**2)
                term2 = (b_diag[i] * alpha_val / Eta_vec[i]) * psi_new[i]
                term3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i]**2 * psidot[i]
                term4 = (c_val / Eta_vec[i]) * phi_new[i] * Interaction[i]
                Q[6+i] = term1 + term2 + term3 - term4
            else:
                # Inactive species: equation is psi = 0
                Q[6+i] = psi_new[i]

        # constraint
        Q[11] = np.sum(phi_new) + phi0_new - 1.0
        # return Q # Modified to modify in-place


    @njit(cache=True, fastmath=True)
    def _compute_jacobian_numba(K, phi_new, phi0_new, psi_new, gamma_new,
                                phi_old, psi_old,
                                dt, Kp1, Eta_vec, Eta_phi_vec,
                                c_val, alpha_val, K_hill, n_hill, A, b_diag, active_mask):
        # K is pre-allocated
        K[:] = 0.0

        # --- Stabilization ---
        eps = 1e-12
        for i in range(5):
            if active_mask[i] == 1:
                if phi_new[i] < eps:
                    phi_new[i] = eps
                elif phi_new[i] > 1.0 - eps:
                    phi_new[i] = 1.0 - eps

                if psi_new[i] < eps:
                    psi_new[i] = eps
                elif psi_new[i] > 1.0 - eps:
                    psi_new[i] = 1.0 - eps
            else:
                phi_new[i] = 0.0
                psi_new[i] = 0.0

        if phi0_new < eps:
            phi0_new = eps
        elif phi0_new > 1.0 - eps:
            phi0_new = 1.0 - eps

        phidot = (phi_new - phi_old) / dt
        psidot = (psi_new - psi_old) / dt
        Interaction = A @ (phi_new * psi_new)

        # --- HILL FUNCTION GATING FOR P. GINGIVALIS (Index 4) ---
        # Same logic as Q vector to keep Interaction consistent
        h_val_pg = 1.0
        I_raw_4 = Interaction[4]  # Save before gating for derivative terms
        dh_dphi3 = 0.0
        dh_dpsi3 = 0.0
        if K_hill > 1e-9:
            fn_conc = phi_new[3] * psi_new[3]
            if fn_conc < 0.0: fn_conc = 0.0
            num = fn_conc**n_hill
            den = K_hill**n_hill + num
            if den < 1e-12:
                h_val_pg = 0.0
            else:
                h_val_pg = num / den
                # Hill derivatives: dh/dx = n * K^n * x^(n-1) / (K^n + x^n)^2
                if fn_conc > 1e-20:
                    Kn = K_hill**n_hill
                    dh_dx = n_hill * Kn * fn_conc**(n_hill - 1.0) / (den * den)
                    dh_dphi3 = dh_dx * psi_new[3]
                    dh_dpsi3 = dh_dx * phi_new[3]

            Interaction[4] *= h_val_pg
        # --------------------------------------------------------

        phi_p = np.zeros(5)
        psi_p = np.zeros(5)
        for i in range(5):
            if active_mask[i] == 1:
                v = phi_new[i]
                phi_p[i] = (Kp1*(-4.+8.*v))/((v**3)*(v-1.)**3) \
                           - (Kp1*(2.-4.*v))*(3./((v**4)*(v-1.)**3)
                                              +3./((v**3)*(v-1.)**4))
                v = psi_new[i]
                psi_p[i] = (4.*Kp1*(3.-5.*v+5.*v**2))/((v**4)*(v-1.)**4)

        v0 = phi0_new
        phi0_p = (Kp1*(-4.+8.*v0))/((v0**3)*(v0-1.)**3) \
                 - (Kp1*(2.-4.*v0))*(3./((v0**4)*(v0-1.)**3)
                                     +3./((v0**3)*(v0-1.)**4))

        # dQ/dphi, dQ/dpsi, dQ/dgamma
        for i in range(5):
            # Hill scaling for row 4
            current_h = 1.0
            if i == 4:
                current_h = h_val_pg

            if active_mask[i] == 1:
                for j in range(5):
                    # Scale A[i,j] by current_h
                    K[i, j] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * current_h * psi_new[j])
                
                # A[i,i] also scaled
                K[i, i] += phi_p[i] \
                           + (1./Eta_vec[i])*((Eta_phi_vec[i] + Eta_vec[i]*psi_new[i]**2)/dt
                                              + Eta_vec[i]*psi_new[i]*psidot[i]) \
                           - (c_val/Eta_vec[i])*(psi_new[i]*(Interaction[i] + A[i,i]*current_h*psi_new[i]))

                for j in range(5):
                    K[i, j+6] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * current_h * phi_new[j])
                
                K[i, i+6] += (1./Eta_vec[i])*(2.*Eta_vec[i]*psi_new[i]*phidot[i]
                                              + Eta_vec[i]*phi_new[i]*psidot[i]
                                              + Eta_vec[i]*phi_new[i]*psi_new[i]/dt) \
                             - (c_val/Eta_vec[i])*((Interaction[i] + A[i,i]*current_h*phi_new[i]*psi_new[i])
                                                   + psi_new[i]*(A[i,i]*current_h*phi_new[i]))
                K[i, 11] = 1./Eta_vec[i]
            else:
                # Inactive row i: Q[i] = phi[i]. dQ[i]/dphi[i] = 1.
                K[i, i] = 1.0

        K[5, 5] = phi0_p + 1./dt
        K[5, 11] = 1.0

        for i in range(5):
            k = i+6
            # Hill scaling for row i
            current_h = 1.0
            if i == 4:
                current_h = h_val_pg

            if active_mask[i] == 1:
                for j in range(5):
                    K[k, j] = -(c_val/Eta_vec[i])*(A[i,j]*current_h*psi_new[j]*phi_new[i]
                                                   + Interaction[i]*(1.0 if i==j else 0.0))
                K[k, i] += (psi_new[i]*phidot[i] + psi_new[i]*phi_new[i]/dt + 2.*phi_new[i]*psidot[i]) \
                           - (c_val/Eta_vec[i])*(A[i,i]*current_h*psi_new[i]*phi_new[i]
                                                 + Interaction[i] + phi_new[i]*A[i,i]*current_h*psi_new[i])
                for j in range(5):
                    K[k, j+6] = -(c_val/Eta_vec[i])*phi_new[i]*A[i,j]*current_h*phi_new[j]
                
                K[k, i+6] += psi_p[i] + (b_diag[i]*alpha_val/Eta_vec[i]) \
                             + (phi_new[i]*phidot[i] + phi_new[i]**2/dt) \
                             - (c_val/Eta_vec[i])*phi_new[i]*A[i,i]*current_h*phi_new[i]
            else:
                # Inactive row i+6: Q[i+6] = psi[i]. dQ/dpsi[i] = 1.
                K[k, k] = 1.0

        K[11, 0:5] = 1.0

        # --- Hill derivative contributions for i=4 (P.g.) ---
        if K_hill > 1e-9 and active_mask[4] == 1:
            c_eta4 = c_val / Eta_vec[4]
            # phi equation row 4: extra dI[4]/dphi3 and dI[4]/dpsi3 from Hill gate
            K[4, 3] += -c_eta4 * psi_new[4] * I_raw_4 * dh_dphi3
            K[4, 9] += -c_eta4 * psi_new[4] * I_raw_4 * dh_dpsi3  # col 9 = psi3
            # psi equation row 10: extra dI[4]/dphi3 and dI[4]/dpsi3 from Hill gate
            K[10, 3] += -c_eta4 * phi_new[4] * I_raw_4 * dh_dphi3
            K[10, 9] += -c_eta4 * phi_new[4] * I_raw_4 * dh_dpsi3
        # return K # Modified to modify in-place


# =============================================================================
# 2. JIT NEWTON + TIME INTEGRATION (5 Species)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _newton_step_jit(g_prev, dt, Kp1, Eta_vec, Eta_phi_vec, c_val, alpha_val, K_hill, n_hill, A, b_diag,
                         eps_tol, max_newton_iter, active_mask,
                         # Buffers
                         g_new, K_buf, Q_buf):
        
        # Initialize g_new from g_prev
        g_new[:] = g_prev[:]
        
        # Force lock initially
        for i in range(5):
            if active_mask[i] == 0:
                g_new[i] = 0.0
                g_new[i+6] = 0.0

        for _ in range(max_newton_iter):
            # No copies, access directly or use temp vars
            # But functions expect slices/arrays. 
            # We can pass slices of g_new, but Numba might copy slices.
            # However, for 1D arrays, basic slicing is a view in NumPy, but Numba support varies.
            # Safest is to extract to temp arrays if we want to be sure, or just index.
            # The original code copied: phi_new = g_new[0:5].copy()
            # Let's keep the copies for safety against aliasing if functions modify them, 
            # but wait, _compute_Q and _compute_K modify them (stabilization).
            # So we MUST pass mutable arrays.
            # Let's use g_new directly slices?
            # If we pass g_new[0:5], it's a view.
            
            # CAUTION: The original code did `phi_new = g_new[0:5].copy()` then `g_new[0:5] = phi_new`
            # This implies the stabilization modifies the local copy, then we commit it.
            # If we pass view, it modifies g_new directly. That should be fine and even better.
            
            phi_new = g_new[0:5] # View
            phi0_new = g_new[5]  # Scalar
            psi_new = g_new[6:11] # View
            gamma_new = g_new[11] # Scalar

            phi_old = g_prev[0:5]
            phi0_old = g_prev[5]
            psi_old = g_prev[6:11]

            _compute_Q_vector_numba(Q_buf, phi_new, phi0_new, psi_new, gamma_new,
                                    phi_old, phi0_old, psi_old,
                                    dt, Kp1, Eta_vec, Eta_phi_vec,
                                    c_val, alpha_val, K_hill, n_hill, A, b_diag, active_mask)
            
            # g_new is already updated by stabilization in _compute_Q (since phi_new is a view)
            # wait, scalar phi0_new is not a view. It's a value.
            # And gamma_new is a value.
            # We need to update g_new[5] and g_new[11] if they changed.
            # But _compute_Q modifies its arguments? 
            # In Python integers/floats are immutable. 
            # In Numba, if passed as scalar argument, they are passed by value (copy).
            # So `phi0_new` in _compute_Q is a local variable. Modifying it there DOES NOT affect caller.
            # The arrays `phi_new` are passed by reference (view).
            
            # Correct approach: _compute_Q should NOT modify scalars.
            # But wait, the original code did:
            # phi0_new = ... (stabilized)
            # g_new[5] = phi0_new
            # So we need to capture the stabilized values.
            # Since scalars can't be modified in-place via args, we need to handle them.
            # Or pass 1-element arrays? No.
            # We can replicate the stabilization logic here or accept that _compute_Q stabilization 
            # for scalars only affects Q calculation, not the state?
            # Original code:
            # 1. Stabilization modifies `phi_new` (array), `phi0_new` (scalar local).
            # 2. Q calculated using stabilized values.
            # 3. `g_new[0:5] = phi_new` (commits array changes).
            # 4. `g_new[5] = phi0_new` (commits scalar changes).
            
            # So YES, we need to update g_new[5] from the stabilized value.
            # Refactoring: Move stabilization out of compute_Q?
            # Or make compute_Q return stabilized scalars?
            # Or pass g_new and indices?
            
            # Simplest: Keep original structure but minimize allocations.
            # Allocating small 5-element arrays is cheap. The 12x12 matrix and solve are the heavy parts.
            # Let's optimize K and Q buffers.
            
            # To handle scalar updates correctly without allocation:
            # Just redo stabilization here? It's cheap.
            
            # --- Stabilization Inline ---
            eps = 1e-12
            for i in range(5):
                if active_mask[i] == 1:
                    if g_new[i] < eps: g_new[i] = eps
                    elif g_new[i] > 1.0-eps: g_new[i] = 1.0-eps
                    
                    if g_new[i+6] < eps: g_new[i+6] = eps
                    elif g_new[i+6] > 1.0-eps: g_new[i+6] = 1.0-eps
                else:
                    g_new[i] = 0.0
                    g_new[i+6] = 0.0
            
            if g_new[5] < eps: g_new[5] = eps
            elif g_new[5] > 1.0-eps: g_new[5] = 1.0-eps
            # ---------------------------
            
            # Now call compute_Q with stabilized g_new
            # Note: _compute_Q also has stabilization logic. It's redundant but harmless if values are already stabilized.
            # However, we must ensure _compute_Q uses the values from g_new.
            
            phi_new = g_new[0:5]
            phi0_new = g_new[5]
            psi_new = g_new[6:11]
            gamma_new = g_new[11]
            
            _compute_Q_vector_numba(Q_buf, phi_new, phi0_new, psi_new, gamma_new,
                                    phi_old, phi0_old, psi_old,
                                    dt, Kp1, Eta_vec, Eta_phi_vec,
                                    c_val, alpha_val, K_hill, n_hill, A, b_diag, active_mask)

            # NaN guard
            nan_found = False
            for i in range(12):
                if np.isnan(Q_buf[i]):
                    nan_found = True
                    break
            if nan_found:
                break

            _compute_jacobian_numba(K_buf, phi_new, phi0_new, psi_new, gamma_new,
                                    phi_old, psi_old,
                                    dt, Kp1, Eta_vec, Eta_phi_vec,
                                    c_val, alpha_val, K_hill, n_hill, A, b_diag, active_mask)

            # solve K * delta = -Q
            try:
                delta = np.linalg.solve(K_buf, -Q_buf)
            except Exception:
                # Regularization
                for i in range(12):
                    K_buf[i, i] += 1e-10
                delta = np.linalg.solve(K_buf, -Q_buf)

            norm_Q = 0.0
            for i in range(12):
                v = abs(Q_buf[i])
                if v > norm_Q:
                    norm_Q = v

            step = 1.0
            improved = False
            
            # g_trial buffer? We can allocate one small buffer for trial.
            # Or just let it allocate (it's 12 floats).
            # g_trial = g_new + step * delta  <- This allocates new array.
            
            while step > 1e-4:
                # g_trial computation without allocation:
                # We need a buffer for g_trial. Let's assume we can allocate one 12-float array.
                # It's better to pass it in.
                # For now, let's just optimize the outer loop.
                g_trial = g_new + step * delta
                
                # Force lock on trial
                for i in range(5):
                    if active_mask[i] == 0:
                        g_trial[i] = 0.0
                        g_trial[i+6] = 0.0
                
                # --- Stabilization for trial state ---
                eps = 1e-12
                for i in range(5):
                    if active_mask[i] == 1:
                        if g_trial[i] < eps: g_trial[i] = eps
                        elif g_trial[i] > 1.0-eps: g_trial[i] = 1.0-eps
                        
                        if g_trial[i+6] < eps: g_trial[i+6] = eps
                        elif g_trial[i+6] > 1.0-eps: g_trial[i+6] = 1.0-eps
                
                if g_trial[5] < eps: g_trial[5] = eps
                elif g_trial[5] > 1.0-eps: g_trial[5] = 1.0-eps
                # -------------------------------------

                phi_t = g_trial[0:5]
                phi0_t = g_trial[5]
                psi_t = g_trial[6:11]
                gamma_t = g_trial[11]

                # We need another Q buffer for trial? Or reuse Q_buf?
                # We need to compare with norm_Q (from Q_buf).
                # So we can overwrite Q_buf IF we don't need old Q_buf values.
                # We only need norm_Q.
                # So we can reuse Q_buf.
                
                _compute_Q_vector_numba(Q_buf, phi_t, phi0_t, psi_t, gamma_t,
                                        phi_old, phi0_old, psi_old,
                                        dt, Kp1, Eta_vec, Eta_phi_vec,
                                        c_val, alpha_val, K_hill, n_hill, A, b_diag, active_mask)

                # NaN check
                nan2 = False
                for i in range(12):
                    if np.isnan(Q_buf[i]):
                        nan2 = True
                        break
                if not nan2:
                    norm_trial = 0.0
                    for i in range(12):
                        v = abs(Q_buf[i])
                        if v > norm_trial:
                            norm_trial = v
                    if norm_trial < norm_Q:
                        g_new[:] = g_trial[:] # Copy back
                        norm_Q = norm_trial
                        improved = True
                        break

                step *= 0.5

            if not improved:
                # Line search failed. Do not take the full step delta.
                # Break to accept the current best g_new (from previous iter)
                break

            if norm_Q < eps_tol:
                break

        return g_new


    @njit(cache=True, fastmath=True)
    def _run_deterministic_jit(theta, dt, maxtimestep, eps_base, Kp1,
                               Eta_vec, Eta_phi_vec, c_val, alpha_val,
                               K_hill, n_hill,
                               phi_init, active_species_mask):
        # build initial state (12,)
        g_prev = np.zeros(12)
        # phi1..phi5
        for i in range(5):
            g_prev[i] = phi_init if active_species_mask[i] == 1 else 0.0
        # phi0
        g_prev[5] = 1.0 - np.sum(g_prev[0:5])
        # psi1..psi5
        for i in range(5):
            g_prev[6+i] = 0.999 if active_species_mask[i] == 1 else 0.0
        # gamma
        g_prev[11] = 0.0

        # theta -> A, b_diag
        A = np.zeros((5, 5))
        b_diag = np.zeros(5)

        # M1 block
        A[0, 0] = theta[0]
        A[0, 1] = theta[1]; A[1, 0] = theta[1]
        A[1, 1] = theta[2]
        b_diag[0] = theta[3]
        b_diag[1] = theta[4]

        # M2 block
        A[2, 2] = theta[5]
        A[2, 3] = theta[6]; A[3, 2] = theta[6]
        A[3, 3] = theta[7]
        b_diag[2] = theta[8]
        b_diag[3] = theta[9]

        # M3 cross
        A[0, 2] = theta[10]; A[2, 0] = theta[10]
        A[0, 3] = theta[11]; A[3, 0] = theta[11]
        A[1, 2] = theta[12]; A[2, 1] = theta[12]
        A[1, 3] = theta[13]; A[3, 1] = theta[13]

        # M4 (S5 self)
        A[4, 4] = theta[14]
        b_diag[4] = theta[15]

        # M5 (S5 cross)
        # a15, a25, a35, a45
        A[0, 4] = theta[16]; A[4, 0] = theta[16]
        A[1, 4] = theta[17]; A[4, 1] = theta[17]
        A[2, 4] = theta[18]; A[4, 2] = theta[18]
        A[3, 4] = theta[19]; A[4, 3] = theta[19]

        t_arr = np.empty(maxtimestep + 1, dtype=np.float64)
        g_arr = np.empty((maxtimestep + 1, 12), dtype=np.float64)
        t_arr[0] = 0.0
        g_arr[0] = g_prev
        
        # Buffers for Newton step
        g_new_buf = np.zeros(12)
        K_buf = np.zeros((12, 12))
        Q_buf = np.zeros(12)

        for step in range(maxtimestep):
            tt = (step + 1) * dt
            tol_t = eps_base * (1.0 + 10.0 * tt)

            # Call updated newton step
            # Note: _newton_step_jit now returns g_new (which is g_new_buf updated)
            # We pass g_new_buf as scratch space
            _newton_step_jit(
                g_prev, dt, Kp1, Eta_vec, Eta_phi_vec,
                c_val, alpha_val, K_hill, n_hill, A, b_diag,
                tol_t, 50, active_species_mask,
                g_new_buf, K_buf, Q_buf
            )

            # g_new_buf now holds the result
            g_prev[:] = g_new_buf[:]
            t_arr[step + 1] = tt
            g_arr[step + 1] = g_prev

        return t_arr, g_arr


    @njit(cache=True, fastmath=True)
    def _run_deterministic_jit_array(theta, dt, maxtimestep, eps_base, Kp1,
                                     Eta_vec, Eta_phi_vec, c_val, alpha_val,
                                     K_hill, n_hill,
                                     phi_init_array, active_species_mask):
        """
        JIT-compiled deterministic solver with per-species initial conditions.

        phi_init_array: (5,) array of initial phi values for each species
        """
        # build initial state (12,)
        g_prev = np.zeros(12)
        # phi1..phi5 from array
        for i in range(5):
            g_prev[i] = phi_init_array[i] if active_species_mask[i] == 1 else 0.0
        # phi0
        g_prev[5] = 1.0 - np.sum(g_prev[0:5])
        # psi1..psi5
        for i in range(5):
            g_prev[6+i] = 0.999 if active_species_mask[i] == 1 else 0.0
        # gamma
        g_prev[11] = 0.0

        # theta -> A, b_diag
        A = np.zeros((5, 5))
        b_diag = np.zeros(5)

        # M1 block
        A[0, 0] = theta[0]
        A[0, 1] = theta[1]; A[1, 0] = theta[1]
        A[1, 1] = theta[2]
        b_diag[0] = theta[3]
        b_diag[1] = theta[4]

        # M2 block
        A[2, 2] = theta[5]
        A[2, 3] = theta[6]; A[3, 2] = theta[6]
        A[3, 3] = theta[7]
        b_diag[2] = theta[8]
        b_diag[3] = theta[9]

        # M3 cross
        A[0, 2] = theta[10]; A[2, 0] = theta[10]
        A[0, 3] = theta[11]; A[3, 0] = theta[11]
        A[1, 2] = theta[12]; A[2, 1] = theta[12]
        A[1, 3] = theta[13]; A[3, 1] = theta[13]

        # M4 (S5 self)
        A[4, 4] = theta[14]
        b_diag[4] = theta[15]

        # M5 (S5 cross)
        # a15, a25, a35, a45
        A[0, 4] = theta[16]; A[4, 0] = theta[16]
        A[1, 4] = theta[17]; A[4, 1] = theta[17]
        A[2, 4] = theta[18]; A[4, 2] = theta[18]
        A[3, 4] = theta[19]; A[4, 3] = theta[19]

        t_arr = np.empty(maxtimestep + 1, dtype=np.float64)
        g_arr = np.empty((maxtimestep + 1, 12), dtype=np.float64)
        t_arr[0] = 0.0
        g_arr[0] = g_prev

        # Buffers for Newton step
        g_new_buf = np.zeros(12)
        K_buf = np.zeros((12, 12))
        Q_buf = np.zeros(12)

        for step in range(maxtimestep):
            tt = (step + 1) * dt
            tol_t = eps_base * (1.0 + 10.0 * tt)

            _newton_step_jit(
                g_prev, dt, Kp1, Eta_vec, Eta_phi_vec,
                c_val, alpha_val, K_hill, n_hill, A, b_diag,
                tol_t, 50, active_species_mask,
                g_new_buf, K_buf, Q_buf
            )

            g_prev[:] = g_new_buf[:]
            t_arr[step + 1] = tt
            g_arr[step + 1] = g_prev

        return t_arr, g_arr


def _compute_Q_vector_numpy(phi_new, phi0_new, psi_new, gamma_new,
                            phi_old, phi0_old, psi_old,
                            dt, Kp1, Eta_vec, Eta_phi_vec,
                            c_val, alpha_val, K_hill, n_hill, A, b_diag, active_mask):
    """
    Pure NumPy version of residual calculation for complex-step support.
    """
    # No strict locking/clipping for complex step (perturbation)
    # But usually we need it? 
    # For complex step, we assume we are near a valid state.
    # We should NOT clip complex values because it destroys the imaginary part.
    # However, if we are at boundary, it might be an issue.
    # Usually complex step is done around a valid real point.
    
    Q = np.zeros(12, dtype=phi_new.dtype)
    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt

    Interaction = A @ (phi_new * psi_new)

    # --- HILL FUNCTION GATING FOR P. GINGIVALIS (Index 4) ---
    if K_hill > 1e-9 and active_mask[4] == 1:
        fn_conc = phi_new[3] * psi_new[3]
        if fn_conc < 0.0:
            fn_conc = 0.0
        num = fn_conc**n_hill
        den = K_hill**n_hill + num
        if den < 1e-12:
            h_val = 0.0
        else:
            h_val = num / den
        Interaction[4] *= h_val
    # --------------------------------------------------------

    # Q[0..4] for phi_i
    for i in range(5):
        if active_mask[i] == 1:
            term1 = (Kp1 * (2.0 - 4.0 * phi_new[i])) / ((phi_new[i] - 1.0)**3 * phi_new[i]**3)
            term2 = (1.0 / Eta_vec[i]) * (gamma_new
                                          + (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) * phidot[i]
                                          + Eta_vec[i] * phi_new[i] * psi_new[i] * psidot[i])
            term3 = (c_val / Eta_vec[i]) * psi_new[i] * Interaction[i]
            Q[i] = term1 + term2 - term3
        else:
            Q[i] = phi_new[i]

    # Q[5] for phi0
    Q[5] = gamma_new + (Kp1 * (2.0 - 4.0 * phi0_new)) / ((phi0_new - 1.0)**3 * phi0_new**3) + phi0dot

    # Q[6..10] for psi_i
    for i in range(5):
        if active_mask[i] == 1:
            term1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0)**2 * psi_new[i]**3) \
                    - (2.0 * Kp1) / ((psi_new[i] - 1.0)**3 * psi_new[i]**2)
            term2 = (b_diag[i] * alpha_val / Eta_vec[i]) * psi_new[i]
            term3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i]**2 * psidot[i]
            term4 = (c_val / Eta_vec[i]) * phi_new[i] * Interaction[i]
            Q[6+i] = term1 + term2 + term3 - term4
        else:
            Q[6+i] = psi_new[i]

    # constraint
    Q[11] = np.sum(phi_new) + phi0_new - 1.0
    return Q


def theta_to_matrices_numpy(theta):
    """
    Convert parameter vector theta (20,) to Interaction matrix A (5x5) and decay vector b_diag (5,).
    Supports both float and complex inputs (for complex-step differentiation).
    """
    dtype = theta.dtype
    A = np.zeros((5, 5), dtype=dtype)
    b_diag = np.zeros(5, dtype=dtype)

    # M1 block
    A[0, 0] = theta[0]
    A[0, 1] = theta[1]; A[1, 0] = theta[1]
    A[1, 1] = theta[2]
    b_diag[0] = theta[3]
    b_diag[1] = theta[4]

    # M2 block
    A[2, 2] = theta[5]
    A[2, 3] = theta[6]; A[3, 2] = theta[6]
    A[3, 3] = theta[7]
    b_diag[2] = theta[8]
    b_diag[3] = theta[9]

    # M3 cross
    A[0, 2] = theta[10]; A[2, 0] = theta[10]
    A[0, 3] = theta[11]; A[3, 0] = theta[11]
    A[1, 2] = theta[12]; A[2, 1] = theta[12]
    A[1, 3] = theta[13]; A[3, 1] = theta[13]

    # M4 (S5 self)
    A[4, 4] = theta[14]
    b_diag[4] = theta[15]

    # M5 (S5 cross)
    A[0, 4] = theta[16]; A[4, 0] = theta[16]
    A[1, 4] = theta[17]; A[4, 1] = theta[17]
    A[2, 4] = theta[18]; A[4, 2] = theta[18]
    A[3, 4] = theta[19]; A[4, 3] = theta[19]
    
    return A, b_diag


# =============================================================================
# 3. SOLVER CLASS (5 Species)
# =============================================================================

class BiofilmNewtonSolver5S:
    THETA_NAMES = [
        "a11","a12","a22","b1","b2",
        "a33","a34","a44","b3","b4",
        "a13","a14","a23","a24",
        "a55","b5",
        "a15","a25","a35","a45"
    ]

    def __init__(
        self,
        dt: float = 1e-5,
        maxtimestep: int = 500,
        eps: float = 1e-6,
        Kp1: float = 1e-4,
        eta_vec=None,
        eta_phi_vec=None,
        c_const: float = 100.0,
        alpha_const: float = 100.0,
        alpha_schedule: dict | None = None,
        phi_init: float | np.ndarray = 0.2,
        active_species=None,
        use_numba: bool = True,
        max_newton_iter: int = 50,
        K_hill: float = 0.0,
        n_hill: float = 2.0,
    ):
        self.dt = float(dt)
        self.maxtimestep = int(maxtimestep)
        self.eps = float(eps)
        self.Kp1 = float(Kp1)
        self.c_const = float(c_const)
        self.alpha_const = float(alpha_const)
        self.alpha_schedule = alpha_schedule
        self.max_newton_iter = int(max_newton_iter)
        self.K_hill = float(K_hill)
        self.n_hill = float(n_hill)

        # Handle phi_init: can be scalar or array(5,)
        if isinstance(phi_init, np.ndarray):
            if phi_init.shape != (5,):
                raise ValueError(f"phi_init array must have shape (5,), got {phi_init.shape}")
            self.phi_init = phi_init.astype(np.float64)
            self.phi_init_is_array = True
        else:
            self.phi_init = float(phi_init)
            self.phi_init_is_array = False

        if active_species is None:
            active_species = [0, 1, 2, 3, 4]
        self.active_species = list(active_species)
        
        self.active_mask = np.zeros(5, dtype=np.int64)
        for i in self.active_species:
            if 0 <= i < 5:
                self.active_mask[i] = 1

        if eta_vec is None:
            eta_vec = np.ones(5, dtype=float)
        if eta_phi_vec is None:
            eta_phi_vec = np.ones(5, dtype=float)

        self.Eta_vec = np.asarray(eta_vec, dtype=float)
        self.Eta_phi_vec = np.asarray(eta_phi_vec, dtype=float)

        self.use_numba = bool(use_numba and HAS_NUMBA)

    def theta_to_matrices(self, theta):
        return theta_to_matrices_numpy(theta)

    def compute_Q_vector(self, g_new, g_old, t, dt, A, b_diag):
        return _compute_Q_vector_numpy(
            g_new[0:5], g_new[5], g_new[6:11], g_new[11],
            g_old[0:5], g_old[5], g_old[6:11],
            dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
            self.c_const, self.alpha_const, self.K_hill, self.n_hill, A, b_diag, self.active_mask
        )

    def compute_Jacobian_matrix(self, g_new, g_old, t, dt, A, b_diag):
        if self.use_numba and not np.iscomplexobj(g_new):
            K = np.zeros((12, 12))
            _compute_jacobian_numba(
                K,
                g_new[0:5], g_new[5], g_new[6:11], g_new[11],
                g_old[0:5], g_old[6:11],
                dt, self.Kp1, self.Eta_vec, self.Eta_phi_vec,
                self.c_const, self.alpha_const, self.K_hill, self.n_hill, A, b_diag, self.active_mask
            )
            return K
        else:
            # Fallback or complex case (not implemented for now)
            raise NotImplementedError("Only JIT Jacobian for real inputs is supported.")

    def solve(self, theta):
        if self.phi_init_is_array:
            return _run_deterministic_jit_array(
                theta, self.dt, self.maxtimestep, self.eps, self.Kp1,
                self.Eta_vec, self.Eta_phi_vec, self.c_const, self.alpha_const,
                self.K_hill, self.n_hill,
                self.phi_init, self.active_mask
            )
        else:
            return _run_deterministic_jit(
                theta, self.dt, self.maxtimestep, self.eps, self.Kp1,
                self.Eta_vec, self.Eta_phi_vec, self.c_const, self.alpha_const,
                self.K_hill, self.n_hill,
                self.phi_init, self.active_mask
            )

    def run_deterministic(self, theta):
        """Alias for solve to match LogLikelihoodEvaluator expectations."""
        return self.solve(theta)
