"""
Improved TMCMC Function for Sharp Likelihood Peaks
===================================================

This module provides an improved version of the TMCMC algorithm with three
critical fixes for handling sharp likelihood peaks (like Case II):

1. Lower ESS target ratio (0.5 instead of 0.8)
2. Minimum beta increment to prevent stalling
3. Likelihood scaling option for sharp peaks

Author: Based on Fritsch et al. (2025) with stability improvements
"""

import numpy as np
from typing import List
from dataclasses import dataclass, field


@dataclass
class TMCMCResult:
    samples: List[np.ndarray]
    log_weights: List[np.ndarray]
    beta_schedule: List[float]
    logL_trace: List[np.ndarray] = field(default_factory=list)
    acceptance_rates: List[float] = field(default_factory=list)
    ess_trace: List[float] = field(default_factory=list)
    converged: bool = False


def tmcmc(log_likelihood, log_prior, theta_init_samples, n_stages=15,
          target_ess_ratio=0.5,      # ✅ Fix 1: Changed from 0.8
          min_delta_beta=0.01,        # ✅ Fix 2: NEW - minimum beta increment
          logL_scale=1.0,             # ✅ Fix 3: NEW - likelihood scaling
          adapt_cov=True, random_state=None,
          show_progress=True, model_name="") -> TMCMCResult:
    """
    Transitional Markov Chain Monte Carlo with stability improvements.
    
    This implementation includes three critical fixes for sharp likelihood peaks:
    
    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function
    log_prior : callable
        Log-prior function
    theta_init_samples : ndarray
        Initial samples (N x d)
    n_stages : int, default=15
        Maximum number of TMCMC stages
    target_ess_ratio : float, default=0.5
        Target ESS as fraction of N (0.5 is more aggressive than 0.8)
    min_delta_beta : float, default=0.01
        Minimum beta increment per stage (prevents stalling)
    logL_scale : float, default=1.0
        Scale factor for likelihood (<1.0 for sharp peaks, e.g., 0.2 for Case II)
    adapt_cov : bool, default=True
        Whether to adapt proposal covariance
    random_state : int, optional
        Random seed
    show_progress : bool, default=True
        Show progress bars and status messages
    model_name : str, default=""
        Model identifier for logging
    
    Returns
    -------
    TMCMCResult
        Results including samples, weights, beta schedule, and convergence info
    
    Examples
    --------
    >>> # For sharp likelihood peaks (Case II):
    >>> res = tmcmc(logL, log_prior, init_samples,
    ...             target_ess_ratio=0.5,
    ...             min_delta_beta=0.01,
    ...             logL_scale=0.2)  # Scale down sharp peak
    
    >>> # For normal cases:
    >>> res = tmcmc(logL, log_prior, init_samples,
    ...             target_ess_ratio=0.5,
    ...             min_delta_beta=0.01,
    ...             logL_scale=1.0)  # No scaling
    """
    
    rng = np.random.default_rng(random_state)
    theta_curr = np.array(theta_init_samples, dtype=float)
    N, d = theta_curr.shape

    beta_list = [0.0]
    samples_list = [theta_curr.copy()]
    logw_list = [np.zeros(N)]
    logL_trace = []
    acceptance_rates = []
    ess_trace = []
    converged = False

    # =========================================================================
    # Initial likelihood evaluation
    # =========================================================================
    if show_progress:
        print(f"    [{model_name}] Evaluating initial likelihoods ({N} samples)...")
        try:
            from tqdm import tqdm
            pbar = tqdm(total=N, desc="Init logL", ncols=80)
            use_tqdm = True
        except ImportError:
            use_tqdm = False
    
    logp_prior = np.zeros(N)
    logL = np.zeros(N)
    
    for i in range(N):
        logp_prior[i] = log_prior(theta_curr[i])
        logL[i] = log_likelihood(theta_curr[i])
        if show_progress and use_tqdm and (i+1) % max(1, N//20) == 0:
            pbar.update(N//20)
    
    if show_progress and use_tqdm:
        pbar.close()
    
    # Handle non-finite values
    logp_prior[~np.isfinite(logp_prior)] = -1e20
    logL[~np.isfinite(logL)] = -1e20
    logL_trace.append(logL.copy())
    
    # Print initial statistics
    print(f"    [{model_name}] Initial logL: "
          f"min={logL.min():.1f}, max={logL.max():.1f}, "
          f"mean={logL.mean():.1f}, std={logL.std():.1f}")
    
    if logL_scale != 1.0:
        print(f"    [{model_name}] Using logL scaling factor: {logL_scale:.2f}")

    beta = 0.0

    # =========================================================================
    # TMCMC Stages
    # =========================================================================
    for stage in range(1, n_stages + 1):
        
        # =====================================================================
        # Fix 3: Scale likelihood for ESS calculation (sharp peak handling)
        # =====================================================================
        logL_eff = logL * logL_scale
        
        def ess_for_delta(delta_beta):
            """Compute ESS for a given delta_beta using scaled likelihood"""
            x = delta_beta * (logL_eff - np.max(logL_eff))
            w_unnorm = np.exp(x)
            s = np.sum(w_unnorm)
            if s <= 0 or not np.isfinite(s):
                return 0.0
            w = w_unnorm / s
            return 1.0 / np.sum(w**2) if np.isfinite(w).all() else 0.0

        # Binary search for optimal delta_beta
        delta_low, delta_high = 0.0, 1.0 - beta
        for _ in range(25):
            mid = 0.5 * (delta_low + delta_high)
            if ess_for_delta(mid) >= target_ess_ratio * N:
                delta_low = mid
            else:
                delta_high = mid

        # =====================================================================
        # Fix 2: Force minimum beta increment (prevent stalling)
        # =====================================================================
        delta_beta = max(delta_low, min_delta_beta)
        beta_next = min(beta + delta_beta, 1.0)

        # Compute weights with UNSCALED likelihood (important!)
        x = delta_beta * (logL - np.max(logL))
        w_unnorm = np.exp(x)
        s = np.sum(w_unnorm)
        w = w_unnorm / s if (s > 0 and np.isfinite(s)) else np.ones(N) / N
        if not np.isfinite(w).all():
            w = np.ones(N) / N

        ess = 1.0 / np.sum(w**2)
        ess_trace.append(ess)
        
        # Print stage info with delta_beta
        status = "🎯 CONVERGED!" if beta_next >= 1.0 else ""
        print(f"    [{model_name}] Stage {stage}: "
              f"β={beta_next:.4f}, Δβ={delta_beta:.4f}, "
              f"ESS={ess:.1f}/{N} ({100*ess/N:.1f}%) {status}")
        
        beta_list.append(beta_next)

        # =====================================================================
        # Resampling
        # =====================================================================
        idx = rng.choice(N, size=N, p=w)
        theta_resampled = theta_curr[idx]
        
        # Adaptive covariance
        if adapt_cov and stage > 1:
            cov = np.cov(theta_resampled.T) + 1e-6 * np.eye(d)
        else:
            cov = 0.01 * np.eye(d)

        # =====================================================================
        # Metropolis-Hastings moves
        # =====================================================================
        theta_new = theta_resampled.copy()
        n_accepted = 0

        if show_progress and use_tqdm:
            pbar = tqdm(total=N, desc=f"[{model_name}] MH Stage {stage}", ncols=80)
        
        for n in range(N):
            th_old = theta_resampled[n]
            lp_old = log_prior(th_old)
            ll_old = log_likelihood(th_old)
            
            if not np.isfinite(lp_old) or not np.isfinite(ll_old):
                if show_progress and use_tqdm and (n+1) % max(1, N//10) == 0:
                    pbar.update(N//10)
                continue
            
            logpost_old = lp_old + beta_next * ll_old

            # Propose new sample
            prop = rng.multivariate_normal(th_old, cov)
            lp_prop = log_prior(prop)
            ll_prop = log_likelihood(prop)
            
            if not np.isfinite(lp_prop) or not np.isfinite(ll_prop):
                if show_progress and use_tqdm and (n+1) % max(1, N//10) == 0:
                    pbar.update(N//10)
                continue
            
            logpost_prop = lp_prop + beta_next * ll_prop

            # Accept/reject
            if rng.uniform() < np.exp(logpost_prop - logpost_old):
                theta_new[n] = prop
                n_accepted += 1
            
            if show_progress and use_tqdm and (n+1) % max(1, N//10) == 0:
                pbar.update(N//10)
        
        if show_progress and use_tqdm:
            pbar.close()

        acceptance_rate = n_accepted / N
        acceptance_rates.append(acceptance_rate)
        print(f"    [{model_name}] Acceptance rate: {100*acceptance_rate:.1f}%")
        
        # Update current samples
        theta_curr = theta_new.copy()
        
        # Recompute likelihoods
        logp_prior = np.array([log_prior(th) for th in theta_curr])
        logL = np.array([log_likelihood(th) for th in theta_curr])
        logp_prior[~np.isfinite(logp_prior)] = -1e20
        logL[~np.isfinite(logL)] = -1e20
        logL_trace.append(logL.copy())

        # Update beta
        beta = beta_next
        samples_list.append(theta_curr.copy())
        logw_list.append(np.log(w + 1e-300))

        # Check convergence
        if beta >= 1.0:
            print(f"    [{model_name}] ✓ Converged at stage {stage}")
            converged = True
            break

    if not converged:
        print(f"    [{model_name}] ⚠ WARNING: Did not converge within {n_stages} stages "
              f"(β={beta:.4f})")

    return TMCMCResult(
        samples=samples_list,
        log_weights=logw_list,
        beta_schedule=beta_list,
        logL_trace=logL_trace,
        acceptance_rates=acceptance_rates,
        ess_trace=ess_trace,
        converged=converged
    )


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("EXAMPLE USAGE")
    print("="*70)
    
    print("""
    # For sharp likelihood peaks (Case II M1):
    res_M1 = tmcmc(
        logL_M1, log_prior_M1, init_M1,
        n_stages=15,
        target_ess_ratio=0.5,      # More aggressive than 0.8
        min_delta_beta=0.01,        # Force progress
        logL_scale=0.2,             # Scale down sharp peak (IMPORTANT!)
        random_state=1234,
        model_name="M1"
    )
    
    # For moderate peaks (Case II M2):
    res_M2 = tmcmc(
        logL_M2, log_prior_M2, init_M2,
        n_stages=15,
        target_ess_ratio=0.5,
        min_delta_beta=0.01,
        logL_scale=0.5,             # Moderate scaling
        random_state=5678,
        model_name="M2"
    )
    
    # For normal cases (Case II M3):
    res_M3 = tmcmc(
        logL_M3, log_prior_M3, init_M3,
        n_stages=15,
        target_ess_ratio=0.5,
        min_delta_beta=0.01,
        logL_scale=1.0,             # No scaling needed
        random_state=9012,
        model_name="M3"
    )
    """)
