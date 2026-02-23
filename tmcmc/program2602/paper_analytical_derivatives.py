"""
paper_analytical_derivatives.py - Analytical Derivatives Matching Paper Model

This module provides analytical derivatives ∂G/∂θ that EXACTLY match the
authoritative paper model in improved1207_paper_jit.py.

CRITICAL: This is derived from _compute_Q_vector_numpy, treating it as the
exact definition of G. All derivatives are computed with g_new FIXED.

State definition (authoritative):
    g(10,) = [phi1, phi2, phi3, phi4, phi0, psi1, psi2, psi3, psi4, gamma]

Parameter mapping (theta to A, b_diag):
    theta[0:14] = [a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24]
    
    A[0,0]=theta[0], A[0,1]=theta[1], A[1,0]=theta[1], A[1,1]=theta[2]
    A[2,2]=theta[5], A[2,3]=theta[6], A[3,2]=theta[6], A[3,3]=theta[7]
    A[0,2]=theta[10], A[2,0]=theta[10]
    A[0,3]=theta[11], A[3,0]=theta[11]
    A[1,2]=theta[12], A[2,1]=theta[12]
    A[1,3]=theta[13], A[3,1]=theta[13]
    
    b_diag[0]=theta[3], b_diag[1]=theta[4], b_diag[2]=theta[8], b_diag[3]=theta[9]

Derivation:
    From _compute_Q_vector_numpy:
    - Q[0:4] (phi_i): depends on A through Interaction = A @ (phi_new * psi_new)
    - Q[4] (phi0): no dependence on A or b_diag
    - Q[5:9] (psi_i): depends on b_diag[i] and A through Interaction
    - Q[9] (constraint): no dependence on A or b_diag
"""

import numpy as np

# Try to import numba, but don't fail if unavailable
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def compute_dG_dtheta_numpy(
    g_new: np.ndarray,
    g_old: np.ndarray,
    t: float,
    dt: float,
    theta: np.ndarray,
    c_val: float,
    alpha_val: float,
    A: np.ndarray,
    b_diag: np.ndarray,
    Eta_vec: np.ndarray,
    Eta_phi_vec: np.ndarray,
    active_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute analytical ∂G/∂θ for all active parameters.
    
    This function computes derivatives by differentiating the exact Q vector
    from _compute_Q_vector_numpy with respect to theta parameters.
    
    Note on dependencies:
    - Only A and b_diag depend on theta parameters.
    - Eta_vec, Eta_phi_vec, Kp1, c_val, alpha_val, dt are fixed (not functions of theta).
    - Eta_phi_vec appears only in time-derivative terms (phidot, psidot) and does not
      depend on theta, so ∂G/∂θ does not include Eta_phi_vec contributions.
    - Q[4] (phi0) and Q[9] (constraint) do not depend on A or b_diag, so their
      derivatives w.r.t. theta are zero (not computed explicitly).
    
    Parameters
    ----------
    g_new : ndarray (10,)
        Current state [phi1..phi4, phi0, psi1..psi4, gamma]
    g_old : ndarray (10,)
        Previous state (unused in derivatives, kept for interface)
    t : float
        Current time (unused, kept for interface)
    dt : float
        Time step (unused, kept for interface)
    theta : ndarray (14,)
        Full parameter vector
    c_val : float
        Nutrient concentration
    alpha_val : float
        Antibiotic concentration
    A : ndarray (4, 4)
        Interaction matrix (derived from theta)
    b_diag : ndarray (4,)
        Growth rates (derived from theta)
    Eta_vec : ndarray (4,)
        Viscosity parameters (fixed)
    Eta_phi_vec : ndarray (4,)
        Phi-dependent viscosity (fixed)
    active_indices : ndarray
        Indices of active parameters (0-13)
        
    Returns
    -------
    dG_dtheta : ndarray (10, n_active)
        Derivative matrix [∂G/∂θ_k] for each active parameter
        Shape: (10, n_active) where n_active = len(active_indices)
    """
    # Extract state components
    phi_new = g_new[0:4]  # phi1..phi4
    phi0_new = g_new[4]
    psi_new = g_new[5:9]  # psi1..psi4
    gamma_new = g_new[9]
    
    # Precompute phi_psi = phi_new * psi_new for efficiency
    # Note: Interaction = A @ phi_psi appears in Q[0:4] and Q[5:9]
    # We compute derivatives of Interaction w.r.t. A elements explicitly
    phi_psi = phi_new * psi_new  # (4,)
    
    n_active = active_indices.shape[0]
    dG_dtheta = np.zeros((10, n_active), dtype=np.float64)
    
    # Loop over active parameters
    for k in range(n_active):
        idx = active_indices[k]
        
        # Map theta index to (A element or b_diag element)
        # Based on theta_to_matrices in improved1207_paper_jit.py
        
        if idx == 0:  # a11 = A[0,0]
            # Q[0] = ... - (c_val/Eta_vec[0]) * psi_new[0] * Interaction[0]
            # where Interaction[0] = sum_j A[0,j] * phi_new[j] * psi_new[j]
            # ∂Interaction[0]/∂A[0,0] = phi_new[0] * psi_new[0] = phi_psi[0]
            dInteraction_dA00 = phi_psi[0]
            dG_dtheta[0, k] = -(c_val / Eta_vec[0]) * psi_new[0] * dInteraction_dA00
            # Q[5] = ... - (c_val/Eta_vec[0]) * phi_new[0] * Interaction[0]
            dG_dtheta[5, k] = -(c_val / Eta_vec[0]) * phi_new[0] * dInteraction_dA00
            
        elif idx == 1:  # a12 = A[0,1] = A[1,0] (symmetric)
            # A[0,1] appears in Interaction[0] and Interaction[1] (symmetric matrix)
            # ∂Interaction[0]/∂A[0,1] = phi_new[1] * psi_new[1] = phi_psi[1]
            # ∂Interaction[1]/∂A[1,0] = phi_new[0] * psi_new[0] = phi_psi[0]
            dInteraction0_dA01 = phi_psi[1]
            dInteraction1_dA10 = phi_psi[0]
            dG_dtheta[0, k] = -(c_val / Eta_vec[0]) * psi_new[0] * dInteraction0_dA01
            dG_dtheta[1, k] = -(c_val / Eta_vec[1]) * psi_new[1] * dInteraction1_dA10
            dG_dtheta[5, k] = -(c_val / Eta_vec[0]) * phi_new[0] * dInteraction0_dA01
            dG_dtheta[6, k] = -(c_val / Eta_vec[1]) * phi_new[1] * dInteraction1_dA10
            
        elif idx == 2:  # a22 = A[1,1]
            # ∂Interaction[1]/∂A[1,1] = phi_psi[1]
            dInteraction1_dA11 = phi_psi[1]
            dG_dtheta[1, k] = -(c_val / Eta_vec[1]) * psi_new[1] * dInteraction1_dA11
            dG_dtheta[6, k] = -(c_val / Eta_vec[1]) * phi_new[1] * dInteraction1_dA11
            
        elif idx == 3:  # b1 = b_diag[0]
            # Q[5] = ... + (b_diag[0] * alpha_val / Eta_vec[0]) * psi_new[0]
            # ∂Q[5]/∂b_diag[0] = (alpha_val / Eta_vec[0]) * psi_new[0]
            # ⚠️ Note: If alpha_val == 0, this derivative is always 0 (b_i is non-identifiable)
            dG_dtheta[5, k] = (alpha_val / Eta_vec[0]) * psi_new[0]
            
        elif idx == 4:  # b2 = b_diag[1]
            # Q[6] = ... + (b_diag[1] * alpha_val / Eta_vec[1]) * psi_new[1]
            # ⚠️ Note: If alpha_val == 0, this derivative is always 0 (b_i is non-identifiable)
            dG_dtheta[6, k] = (alpha_val / Eta_vec[1]) * psi_new[1]
            
        elif idx == 5:  # a33 = A[2,2]
            # ∂Interaction[2]/∂A[2,2] = phi_psi[2]
            dInteraction2_dA22 = phi_psi[2]
            dG_dtheta[2, k] = -(c_val / Eta_vec[2]) * psi_new[2] * dInteraction2_dA22
            dG_dtheta[7, k] = -(c_val / Eta_vec[2]) * phi_new[2] * dInteraction2_dA22
            
        elif idx == 6:  # a34 = A[2,3] = A[3,2] (symmetric)
            # ∂Interaction[2]/∂A[2,3] = phi_psi[3]
            # ∂Interaction[3]/∂A[3,2] = phi_psi[2]
            dInteraction2_dA23 = phi_psi[3]
            dInteraction3_dA32 = phi_psi[2]
            dG_dtheta[2, k] = -(c_val / Eta_vec[2]) * psi_new[2] * dInteraction2_dA23
            dG_dtheta[3, k] = -(c_val / Eta_vec[3]) * psi_new[3] * dInteraction3_dA32
            dG_dtheta[7, k] = -(c_val / Eta_vec[2]) * phi_new[2] * dInteraction2_dA23
            dG_dtheta[8, k] = -(c_val / Eta_vec[3]) * phi_new[3] * dInteraction3_dA32
            
        elif idx == 7:  # a44 = A[3,3]
            # ∂Interaction[3]/∂A[3,3] = phi_psi[3]
            dInteraction3_dA33 = phi_psi[3]
            dG_dtheta[3, k] = -(c_val / Eta_vec[3]) * psi_new[3] * dInteraction3_dA33
            dG_dtheta[8, k] = -(c_val / Eta_vec[3]) * phi_new[3] * dInteraction3_dA33
            
        elif idx == 8:  # b3 = b_diag[2]
            # Q[7] = ... + (b_diag[2] * alpha_val / Eta_vec[2]) * psi_new[2]
            # ⚠️ Note: If alpha_val == 0, this derivative is always 0 (b_i is non-identifiable)
            dG_dtheta[7, k] = (alpha_val / Eta_vec[2]) * psi_new[2]
            
        elif idx == 9:  # b4 = b_diag[3]
            # Q[8] = ... + (b_diag[3] * alpha_val / Eta_vec[3]) * psi_new[3]
            # ⚠️ Note: If alpha_val == 0, this derivative is always 0 (b_i is non-identifiable)
            dG_dtheta[8, k] = (alpha_val / Eta_vec[3]) * psi_new[3]
            
        elif idx == 10:  # a13 = A[0,2] = A[2,0] (symmetric)
            # ∂Interaction[0]/∂A[0,2] = phi_psi[2]
            # ∂Interaction[2]/∂A[2,0] = phi_psi[0]
            dInteraction0_dA02 = phi_psi[2]
            dInteraction2_dA20 = phi_psi[0]
            dG_dtheta[0, k] = -(c_val / Eta_vec[0]) * psi_new[0] * dInteraction0_dA02
            dG_dtheta[2, k] = -(c_val / Eta_vec[2]) * psi_new[2] * dInteraction2_dA20
            dG_dtheta[5, k] = -(c_val / Eta_vec[0]) * phi_new[0] * dInteraction0_dA02
            dG_dtheta[7, k] = -(c_val / Eta_vec[2]) * phi_new[2] * dInteraction2_dA20
            
        elif idx == 11:  # a14 = A[0,3] = A[3,0] (symmetric)
            # ∂Interaction[0]/∂A[0,3] = phi_psi[3]
            # ∂Interaction[3]/∂A[3,0] = phi_psi[0]
            dInteraction0_dA03 = phi_psi[3]
            dInteraction3_dA30 = phi_psi[0]
            dG_dtheta[0, k] = -(c_val / Eta_vec[0]) * psi_new[0] * dInteraction0_dA03
            dG_dtheta[3, k] = -(c_val / Eta_vec[3]) * psi_new[3] * dInteraction3_dA30
            dG_dtheta[5, k] = -(c_val / Eta_vec[0]) * phi_new[0] * dInteraction0_dA03
            dG_dtheta[8, k] = -(c_val / Eta_vec[3]) * phi_new[3] * dInteraction3_dA30
            
        elif idx == 12:  # a23 = A[1,2] = A[2,1] (symmetric)
            # ∂Interaction[1]/∂A[1,2] = phi_psi[2]
            # ∂Interaction[2]/∂A[2,1] = phi_psi[1]
            dInteraction1_dA12 = phi_psi[2]
            dInteraction2_dA21 = phi_psi[1]
            dG_dtheta[1, k] = -(c_val / Eta_vec[1]) * psi_new[1] * dInteraction1_dA12
            dG_dtheta[2, k] = -(c_val / Eta_vec[2]) * psi_new[2] * dInteraction2_dA21
            dG_dtheta[6, k] = -(c_val / Eta_vec[1]) * phi_new[1] * dInteraction1_dA12
            dG_dtheta[7, k] = -(c_val / Eta_vec[2]) * phi_new[2] * dInteraction2_dA21
            
        elif idx == 13:  # a24 = A[1,3] = A[3,1] (symmetric)
            # ∂Interaction[1]/∂A[1,3] = phi_psi[3]
            # ∂Interaction[3]/∂A[3,1] = phi_psi[1]
            dInteraction1_dA13 = phi_psi[3]
            dInteraction3_dA31 = phi_psi[1]
            dG_dtheta[1, k] = -(c_val / Eta_vec[1]) * psi_new[1] * dInteraction1_dA13
            dG_dtheta[3, k] = -(c_val / Eta_vec[3]) * psi_new[3] * dInteraction3_dA31
            dG_dtheta[6, k] = -(c_val / Eta_vec[1]) * phi_new[1] * dInteraction1_dA13
            dG_dtheta[8, k] = -(c_val / Eta_vec[3]) * phi_new[3] * dInteraction3_dA31
    
    return dG_dtheta


# Numba version (identical logic, JIT-compiled)
if HAS_NUMBA:
    @njit(nogil=True, fastmath=False, cache=True)
    def compute_dG_dtheta_numba(
        g_new: np.ndarray,
        g_old: np.ndarray,
        t: float,
        dt: float,
        theta: np.ndarray,
        c_val: float,
        alpha_val: float,
        A: np.ndarray,
        b_diag: np.ndarray,
        Eta_vec: np.ndarray,
        Eta_phi_vec: np.ndarray,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        """
        JIT-compiled version of compute_dG_dtheta_numpy.
        Identical logic, just compiled for speed.
        """
        phi_new = g_new[0:4]
        phi0_new = g_new[4]
        psi_new = g_new[5:9]
        gamma_new = g_new[9]
        
        phi_psi = phi_new * psi_new
        
        n_active = active_indices.shape[0]
        dG_dtheta = np.zeros((10, n_active), dtype=np.float64)
        
        for k in range(n_active):
            idx = active_indices[k]
            
            if idx == 0:  # a11
                dInteraction0_dA00 = phi_psi[0]
                dG_dtheta[0, k] = -(c_val / Eta_vec[0]) * psi_new[0] * dInteraction0_dA00
                dG_dtheta[5, k] = -(c_val / Eta_vec[0]) * phi_new[0] * dInteraction0_dA00
            elif idx == 1:  # a12
                dInteraction0_dA01 = phi_psi[1]
                dInteraction1_dA10 = phi_psi[0]
                dG_dtheta[0, k] = -(c_val / Eta_vec[0]) * psi_new[0] * dInteraction0_dA01
                dG_dtheta[1, k] = -(c_val / Eta_vec[1]) * psi_new[1] * dInteraction1_dA10
                dG_dtheta[5, k] = -(c_val / Eta_vec[0]) * phi_new[0] * dInteraction0_dA01
                dG_dtheta[6, k] = -(c_val / Eta_vec[1]) * phi_new[1] * dInteraction1_dA10
            elif idx == 2:  # a22
                dInteraction1_dA11 = phi_psi[1]
                dG_dtheta[1, k] = -(c_val / Eta_vec[1]) * psi_new[1] * dInteraction1_dA11
                dG_dtheta[6, k] = -(c_val / Eta_vec[1]) * phi_new[1] * dInteraction1_dA11
            elif idx == 3:  # b1
                dG_dtheta[5, k] = (alpha_val / Eta_vec[0]) * psi_new[0]
            elif idx == 4:  # b2
                dG_dtheta[6, k] = (alpha_val / Eta_vec[1]) * psi_new[1]
            elif idx == 5:  # a33
                dInteraction2_dA22 = phi_psi[2]
                dG_dtheta[2, k] = -(c_val / Eta_vec[2]) * psi_new[2] * dInteraction2_dA22
                dG_dtheta[7, k] = -(c_val / Eta_vec[2]) * phi_new[2] * dInteraction2_dA22
            elif idx == 6:  # a34
                dInteraction2_dA23 = phi_psi[3]
                dInteraction3_dA32 = phi_psi[2]
                dG_dtheta[2, k] = -(c_val / Eta_vec[2]) * psi_new[2] * dInteraction2_dA23
                dG_dtheta[3, k] = -(c_val / Eta_vec[3]) * psi_new[3] * dInteraction3_dA32
                dG_dtheta[7, k] = -(c_val / Eta_vec[2]) * phi_new[2] * dInteraction2_dA23
                dG_dtheta[8, k] = -(c_val / Eta_vec[3]) * phi_new[3] * dInteraction3_dA32
            elif idx == 7:  # a44
                dInteraction3_dA33 = phi_psi[3]
                dG_dtheta[3, k] = -(c_val / Eta_vec[3]) * psi_new[3] * dInteraction3_dA33
                dG_dtheta[8, k] = -(c_val / Eta_vec[3]) * phi_new[3] * dInteraction3_dA33
            elif idx == 8:  # b3
                dG_dtheta[7, k] = (alpha_val / Eta_vec[2]) * psi_new[2]
            elif idx == 9:  # b4
                dG_dtheta[8, k] = (alpha_val / Eta_vec[3]) * psi_new[3]
            elif idx == 10:  # a13
                dInteraction0_dA02 = phi_psi[2]
                dInteraction2_dA20 = phi_psi[0]
                dG_dtheta[0, k] = -(c_val / Eta_vec[0]) * psi_new[0] * dInteraction0_dA02
                dG_dtheta[2, k] = -(c_val / Eta_vec[2]) * psi_new[2] * dInteraction2_dA20
                dG_dtheta[5, k] = -(c_val / Eta_vec[0]) * phi_new[0] * dInteraction0_dA02
                dG_dtheta[7, k] = -(c_val / Eta_vec[2]) * phi_new[2] * dInteraction2_dA20
            elif idx == 11:  # a14
                dInteraction0_dA03 = phi_psi[3]
                dInteraction3_dA30 = phi_psi[0]
                dG_dtheta[0, k] = -(c_val / Eta_vec[0]) * psi_new[0] * dInteraction0_dA03
                dG_dtheta[3, k] = -(c_val / Eta_vec[3]) * psi_new[3] * dInteraction3_dA30
                dG_dtheta[5, k] = -(c_val / Eta_vec[0]) * phi_new[0] * dInteraction0_dA03
                dG_dtheta[8, k] = -(c_val / Eta_vec[3]) * phi_new[3] * dInteraction3_dA30
            elif idx == 12:  # a23
                dInteraction1_dA12 = phi_psi[2]
                dInteraction2_dA21 = phi_psi[1]
                dG_dtheta[1, k] = -(c_val / Eta_vec[1]) * psi_new[1] * dInteraction1_dA12
                dG_dtheta[2, k] = -(c_val / Eta_vec[2]) * psi_new[2] * dInteraction2_dA21
                dG_dtheta[6, k] = -(c_val / Eta_vec[1]) * phi_new[1] * dInteraction1_dA12
                dG_dtheta[7, k] = -(c_val / Eta_vec[2]) * phi_new[2] * dInteraction2_dA21
            elif idx == 13:  # a24
                dInteraction1_dA13 = phi_psi[3]
                dInteraction3_dA31 = phi_psi[1]
                dG_dtheta[1, k] = -(c_val / Eta_vec[1]) * psi_new[1] * dInteraction1_dA13
                dG_dtheta[3, k] = -(c_val / Eta_vec[3]) * psi_new[3] * dInteraction3_dA31
                dG_dtheta[6, k] = -(c_val / Eta_vec[1]) * phi_new[1] * dInteraction1_dA13
                dG_dtheta[8, k] = -(c_val / Eta_vec[3]) * phi_new[3] * dInteraction3_dA31
        
        return dG_dtheta
else:
    # Fallback: use numpy version
    compute_dG_dtheta_numba = compute_dG_dtheta_numpy


def verify_against_complex_step(
    g_new: np.ndarray,
    g_old: np.ndarray,
    t: float,
    dt: float,
    theta: np.ndarray,
    c_val: float,
    alpha_val: float,
    A: np.ndarray,
    b_diag: np.ndarray,
    Kp1: float,
    Eta_vec: np.ndarray,
    Eta_phi_vec: np.ndarray,
    active_indices: np.ndarray,
    eps: float = 1e-30,
) -> tuple:
    """
    Verify analytical derivatives against complex-step differentiation.
    
    Uses eps=1e-30 for complex-step (safe for NumPy complex-step differentiation).
    
    Returns
    -------
    dG_analytical : ndarray (10, n_active)
        Analytical derivatives
    dG_complex : ndarray (10, n_active)
        Complex-step reference
    max_error : float
        Maximum relative error
    """
    from improved1207_paper_jit import _compute_Q_vector_numpy
    
    # Compute analytical
    dG_analytical = compute_dG_dtheta_numpy(
        g_new, g_old, t, dt, theta, c_val, alpha_val,
        A, b_diag, Eta_vec, Eta_phi_vec, active_indices
    )
    
    # Compute complex-step reference
    n_active = active_indices.shape[0]
    dG_complex = np.zeros((10, n_active), dtype=np.float64)
    
    phi_new = g_new[0:4]
    phi0_new = g_new[4]
    psi_new = g_new[5:9]
    gamma_new = g_new[9]
    phi_old = g_old[0:4]
    phi0_old = g_old[4]
    psi_old = g_old[5:9]
    
    for k in range(n_active):
        idx = active_indices[k]
        theta_plus = theta.astype(np.complex128)
        theta_plus[idx] += 1j * eps
        
        # Reconstruct A and b_diag from theta_plus (matching theta_to_matrices)
        A_plus = np.zeros((4, 4), dtype=np.complex128)
        b_diag_plus = np.zeros(4, dtype=np.complex128)
        
        # M1 block
        A_plus[0, 0] = theta_plus[0]
        A_plus[0, 1] = theta_plus[1]; A_plus[1, 0] = theta_plus[1]
        A_plus[1, 1] = theta_plus[2]
        b_diag_plus[0] = theta_plus[3]
        b_diag_plus[1] = theta_plus[4]
        
        # M2 block
        A_plus[2, 2] = theta_plus[5]
        A_plus[2, 3] = theta_plus[6]; A_plus[3, 2] = theta_plus[6]
        A_plus[3, 3] = theta_plus[7]
        b_diag_plus[2] = theta_plus[8]
        b_diag_plus[3] = theta_plus[9]
        
        # M3 cross-interaction block
        A_plus[0, 2] = theta_plus[10]; A_plus[2, 0] = theta_plus[10]
        A_plus[0, 3] = theta_plus[11]; A_plus[3, 0] = theta_plus[11]
        A_plus[1, 2] = theta_plus[12]; A_plus[2, 1] = theta_plus[12]
        A_plus[1, 3] = theta_plus[13]; A_plus[3, 1] = theta_plus[13]
        
        Q_plus = _compute_Q_vector_numpy(
            phi_new, phi0_new, psi_new, gamma_new,
            phi_old, phi0_old, psi_old,
            dt, Kp1, Eta_vec, Eta_phi_vec,
            c_val, alpha_val, A_plus, b_diag_plus
        )
        
        dG_complex[:, k] = np.imag(Q_plus) / eps
    
    # Compare
    abs_error = np.abs(dG_analytical - dG_complex)
    rel_error = abs_error / (np.abs(dG_complex) + 1e-16)
    max_error = np.max(rel_error)
    
    return dG_analytical, dG_complex, max_error

