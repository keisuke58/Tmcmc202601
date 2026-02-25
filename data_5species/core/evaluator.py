"""
Log-likelihood evaluator using TSM-ROM with linearization management.

Extracted from case2_tmcmc_linearization.py for better modularity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
from pathlib import Path

# Add project root to sys.path to allow tmcmc imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tmcmc.utils import TimingStats, timed, LikelihoodHealthCounter
except ImportError:
    try:
        from utils import TimingStats, timed, LikelihoodHealthCounter
    except ImportError:
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        utils_dir = current_dir.parent / "utils"
        if str(utils_dir.parent) not in sys.path:
            sys.path.insert(0, str(utils_dir.parent))
        from utils import TimingStats, timed, LikelihoodHealthCounter
try:
    from tmcmc.debug import DebugLogger
except ImportError:
    try:
        from debug import DebugLogger
    except ImportError:
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        debug_dir = current_dir.parent / "debug"
        if str(debug_dir.parent) not in sys.path:
            sys.path.insert(0, str(debug_dir.parent))
        from debug import DebugLogger
try:
    from tmcmc.visualization.helpers import compute_phibar
except ImportError:
    try:
        from visualization.helpers import compute_phibar
    except ImportError:
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        vis_dir = current_dir.parent / "visualization"
        if str(vis_dir.parent) not in sys.path:
            sys.path.insert(0, str(vis_dir.parent))
        from visualization.helpers import compute_phibar

# Try to import from tmcmc.config, fallback to local config if not found
try:
    from tmcmc.config import DebugConfig, DebugLevel
except ImportError:
    try:
        from config import DebugConfig, DebugLevel
    except ImportError:
        # Fallback if neither works (e.g. running from different directory structure)
        import sys
        from pathlib import Path
        # Try to find config.py in program2602 relative to this file
        current_dir = Path(__file__).parent
        program_dir = current_dir.parent / "program2602"
        if str(program_dir) not in sys.path:
            sys.path.insert(0, str(program_dir))
        from config import DebugConfig, DebugLevel

logger = logging.getLogger(__name__)

# Import solver and TSM classes (these are external dependencies)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from improved1207_paper_jit import BiofilmNewtonSolver, HAS_NUMBA
from improved_5species_jit import BiofilmNewtonSolver5S
from demo_analytical_tsm_with_linearization_jit import BiofilmTSM_Analytical
from tmcmc_5species_tsm import BiofilmTSM5S


def build_species_sigma(
    sigma_global: float,
    n_species: int = 5,
    vd_species_idx: int = 2,
    vd_factor: float = 2.0,
) -> np.ndarray:
    """
    Build per-species observation noise vector.

    V. dispar (species 2) gets higher sigma to acknowledge that the
    Hamilton ODE cannot model nutrient-depletion-driven decline.

    Parameters
    ----------
    sigma_global : float
        Global observation noise (applied to all species).
    n_species : int
        Number of species.
    vd_species_idx : int
        Index of V. dispar (default 2).
    vd_factor : float
        Multiplicative factor for V. dispar sigma (default 2.0).

    Returns
    -------
    np.ndarray
        Per-species sigma, shape (n_species,).
    """
    sigma_arr = np.full(n_species, sigma_global, dtype=np.float64)
    if vd_species_idx < n_species:
        sigma_arr[vd_species_idx] *= vd_factor
    return sigma_arr


def build_likelihood_weights(
    n_obs: int,
    n_species: int,
    pg_species_idx: int = 4,
    n_late: int = 2,
    lambda_pg: float = 5.0,
    lambda_late: float = 3.0,
) -> np.ndarray:
    """
    Build a weight matrix for the weighted log-likelihood.

    The matrix has shape (n_obs, n_species).  Entries default to 1.0.
    Species ``pg_species_idx`` (P. gingivalis) receives an extra factor
    ``lambda_pg``, and the last ``n_late`` observation times receive an
    extra factor ``lambda_late``.  The two factors multiply, so P. gingivalis
    at the final timepoints gets weight ``lambda_pg * lambda_late``.

    Parameters
    ----------
    n_obs : int
        Number of observation times.
    n_species : int
        Number of species.
    pg_species_idx : int
        Column index for P. gingivalis (default 4).
    n_late : int
        How many of the *last* observation times to up-weight (default 2,
        i.e. days 15 and 21 out of {1,3,6,10,15,21}).
    lambda_pg : float
        Multiplicative weight for P. gingivalis across all times (default 5).
    lambda_late : float
        Multiplicative weight for the last ``n_late`` observations across
        all species (default 3).

    Returns
    -------
    np.ndarray
        Weight matrix, shape (n_obs, n_species).
    """
    W = np.ones((n_obs, n_species), dtype=np.float64)
    if pg_species_idx < n_species:
        W[:, pg_species_idx] *= lambda_pg
    if n_late > 0:
        W[-n_late:, :] *= lambda_late
    return W


def log_likelihood_sparse(
    mu: np.ndarray,
    sig: np.ndarray,
    data: np.ndarray,
    sigma_obs,
    rho: float = 0.0,
    health: Optional[Dict[str, int]] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute (optionally weighted) log-likelihood for sparse observations.

    When ``weights`` is provided (shape ``(n_obs, n_species)``), the
    diagonal-case contribution of each ``(time, species)`` pair is
    multiplied by ``weights[t, s]``.  This is equivalent to a power
    likelihood:  L_w = prod_{t,s} N(y|mu,v)^{w_{t,s}}.

    Supports:
    - Diagonal covariance (rho=0.0)
    - Equicorrelated covariance (rho != 0.0) where R_ij = rho (i!=j) and 1 (i=j)
      Cov_t = D_t * R * D_t, where D_t = diag(sqrt(var_total))

    Parameters
    ----------
    mu : np.ndarray
        Predicted mean at observation times: shape (n_obs, n_species)
    sig : np.ndarray
        Predicted variance at observation times: shape (n_obs, n_species)
    data : np.ndarray
        Observed data: shape (n_obs, n_species)
    sigma_obs : float or np.ndarray
        Observation noise standard deviation.  Scalar applies to all species;
        array of shape ``(n_species,)`` gives per-species noise.
    rho : float
        Observation correlation coefficient (default: 0.0)
    health : Dict[str, int], optional
        Health counter dictionary for tracking issues
    weights : np.ndarray, optional
        Per-(time, species) multiplicative weights, shape (n_obs, n_species).
        None means uniform weight = 1.

    Returns
    -------
    float
        Log-likelihood value
    """
    n_obs, n_species = data.shape
    logL = 0.0

    # Vectorize sigma_obs: scalar → per-species array
    sigma_obs_arr = np.atleast_1d(np.asarray(sigma_obs, dtype=np.float64))
    if sigma_obs_arr.size == 1:
        sigma_obs_arr = np.full(n_species, sigma_obs_arr[0])

    # Pre-compute R inverse and determinant if rho is used
    use_correlation = (abs(rho) > 1e-9) and (n_species > 1)

    if use_correlation:
        R = np.eye(n_species) + rho * (np.ones((n_species, n_species)) - np.eye(n_species))
        try:
            L_R = np.linalg.cholesky(R)
            log_det_R = 2.0 * np.sum(np.log(np.diag(L_R)))
        except np.linalg.LinAlgError:
            if health is not None:
                health["rho_error"] = 1
            return -1e20

    for i in range(n_obs):
        # 1. Variance vector and total covariance diagonal
        var_total_vec = np.zeros(n_species)
        for j in range(n_species):
            var_raw = sig[i, j] + sigma_obs_arr[j]**2

            # Health checks
            if health is not None:
                if not np.isfinite(var_raw):
                    health["n_var_raw_nonfinite"] = int(health.get("n_var_raw_nonfinite", 0)) + 1
                elif var_raw < 0.0:
                    health["n_var_raw_negative"] = int(health.get("n_var_raw_negative", 0)) + 1
                if (not np.isfinite(var_raw)) or (var_raw <= 1e-20):
                    health["n_var_total_clipped"] = int(health.get("n_var_total_clipped", 0)) + 1

            if not np.isfinite(var_raw) or var_raw <= 1e-20:
                var_total_vec[j] = 1e-20
            else:
                var_total_vec[j] = float(var_raw)

        residual = data[i, :] - mu[i, :]

        if not use_correlation:
            # Diagonal case — apply per-(time, species) weights
            for j in range(n_species):
                v = var_total_vec[j]
                w = 1.0 if weights is None else float(weights[i, j])
                logL -= w * 0.5 * np.log(2 * np.pi * v)
                logL -= w * 0.5 * (residual[j]**2) / v
        else:
            # Correlated case
            # Sigma = D R D
            # log|Sigma| = log|D|^2 + log|R| = sum(log(v_j)) + log|R|
            # z = D^-1 residual
            # Q = z^T R^-1 z

            # std_devs = sqrt(var)
            std_vec = np.sqrt(var_total_vec)

            # log|Sigma|
            # sum(log(var)) = 2 * sum(log(std))
            log_det_Sigma = np.sum(np.log(var_total_vec)) + log_det_R

            # z = residual / std
            z = residual / std_vec

            # Q = z^T R^-1 z
            # R y = z => y = R^-1 z.  Q = z^T y.
            # Solve L_R L_R^T y = z
            # Forward: L_R w = z
            # Backward: L_R^T y = w
            try:
                w_solve = np.linalg.solve(L_R, z)
                quad_form = np.dot(w_solve, w_solve) # w^T w = z^T (L_R^-T L_R^-1) z = z^T R^-1 z
            except Exception:
                 if health is not None:
                    health["solve_error"] = 1
                 return -1e20

            # For the correlated case, apply the mean weight for this timepoint
            w_t = 1.0
            if weights is not None:
                w_t = float(np.mean(weights[i, :]))
            logL -= w_t * 0.5 * (n_species * np.log(2 * np.pi) + log_det_Sigma + quad_form)

    return logL


class LogLikelihoodEvaluator:
    """
    Log-likelihood evaluator using TSM-ROM with linearization management.
    
    KEY FEATURE: Supports update_linearization_point() for 2-phase MCMC
    """
    
    def __init__(
        self,
        solver_kwargs: Dict[str, Any],
        active_species: List[int],
        active_indices: List[int],
        theta_base: np.ndarray,
        data: np.ndarray,
        idx_sparse: np.ndarray,
        sigma_obs: float,
        cov_rel: float,
        rho: float = 0.0,
        theta_linearization: Optional[np.ndarray] = None,
        use_analytical: bool = True,
        paper_mode: bool = False,
        debug_logger: Optional[DebugLogger] = None,
        use_absolute_volume: bool = False,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize likelihood evaluator with linearization support.
        
        Parameters
        ----------
        theta_linearization : ndarray (14,), optional
            Initial linearization point for TSM.
            If None, uses theta_base as linearization point.
        use_analytical : bool
            If True, use analytical derivatives (faster).
        paper_mode : bool, default=False
            If True and use_analytical=True, use paper_analytical_derivatives
            (exact match with improved1207_paper_jit.py, verified with complex-step).
            If False, use complex-step differentiation (slower but more robust for debugging).
        debug_logger : DebugLogger, optional
            Debug logger for controlling error output (ERROR/OFF mode: silent).
        use_absolute_volume : bool, default=False
            If True, use Absolute Volume (phi * gamma) for likelihood instead of default (phi * psi).
        weights : np.ndarray, optional
            Per-(time, species) likelihood weights, shape (n_obs, n_species).
            Built by ``build_likelihood_weights()``.  None = uniform.
        """
        self.active_species = list(active_species)
        self.active_indices = list(active_indices)
        self.theta_base = theta_base.copy()
        self.data = data
        self.idx_sparse = idx_sparse
        self.sigma_obs = sigma_obs
        self.cov_rel = cov_rel
        self.rho = rho
        self.n_species = len(active_species)
        self.solver_kwargs = solver_kwargs.copy()
        self.debug_logger = debug_logger or DebugLogger(DebugConfig(level=DebugLevel.OFF))
        self.use_absolute_volume = use_absolute_volume
        self.weights = weights  # (n_obs, n_species) or None

        # Tracking
        self.call_count = 0  # Number of ROM (TSM) evaluations
        self.fom_call_count = 0  # Number of FOM evaluations (for ROM error computation)
        self.timing = TimingStats()  # Wall-time breakdown for metrics.json
        self.health = LikelihoodHealthCounter()  # Likelihood/TSM health counters
        self.theta_history = []
        self.logL_history = []
        
        # Create solver
        # Check if we need 5-species solver
        self.is_5species = (len(self.active_species) == 5) or (max(self.active_species) >= 4)
        
        if self.is_5species:
            self.solver = BiofilmNewtonSolver5S(
                **solver_kwargs,
                active_species=self.active_species,
                use_numba=HAS_NUMBA,
            )
            # Use BiofilmTSM5S for 5-species model
            if theta_linearization is None:
                theta_linearization = theta_base.copy()
            
            self.tsm = BiofilmTSM5S(
                self.solver,
                active_theta_indices=self.active_indices,
                cov_rel=self.cov_rel,
                theta_linearization=theta_linearization,
            )
        else:
            self.solver = BiofilmNewtonSolver(
                **solver_kwargs,
                active_species=self.active_species,
                use_numba=HAS_NUMBA,
            )
            
            # Use BiofilmTSM_Analytical with linearization management
            if theta_linearization is None:
                theta_linearization = theta_base.copy()
            
            self.tsm = BiofilmTSM_Analytical(
                self.solver,
                active_theta_indices=self.active_indices,
                cov_rel=self.cov_rel,
                use_complex_step=True,
                use_analytical=use_analytical,
                theta_linearization=theta_linearization,
                paper_mode=paper_mode,
            )
        
        self._theta_linearization = theta_linearization.copy()
        self._linearization_enabled = False  # Start with linearization disabled (non-linear exploration)
        logger.info("TSM initialized (linearization disabled initially for exploration)")
    
    def update_linearization_point(self, theta_new_full: np.ndarray):
        """
        Update TSM linearization point.
        
        CRITICAL for 2-phase MCMC accuracy!
        
        Parameters
        ----------
        theta_new_full : ndarray (14,)
            New linearization point (typically MAP from Phase 1)
        """
        self.tsm.update_linearization_point(theta_new_full)
        self._theta_linearization = theta_new_full.copy()
        
        # Reset tracking for new phase
        self.theta_history = []
        self.logL_history = []
        self.call_count = 0
    
    def enable_linearization(self, enable: bool = True):
        """
        Enable or disable linearization dynamically.
        
        This allows switching between full TSM (non-linear) and linearized TSM
        during MCMC execution. Typically:
        - Initial exploration (small β): linearization disabled (full TSM)
        - Later stages (large β): linearization enabled (fast, accurate near MAP)
        
        Parameters
        ----------
        enable : bool
            If True, enable linearization. If False, use full TSM.
        """
        self.tsm.enable_linearization(enable)
        self._linearization_enabled = enable
    
    def get_linearization_point(self) -> np.ndarray:
        """Get current linearization point."""
        return self._theta_linearization.copy()
    
    def compute_ROM_error(self, theta_full: np.ndarray) -> float:
        """
        Compute ROM error based on observable φ̄ (living bacteria volume fraction).
        
        Paper-ready definition:
            ε_ROM = || φ̄_ROM(t_obs) − φ̄_FOM(t_obs) ||₂ / || φ̄_FOM(t_obs) ||₂
        
        where φ̄_i = φ_i * ψ_i (observable quantity used in likelihood).
        
        This is the error in the observable space, which directly relates to
        the likelihood approximation quality.
        
        Parameters
        ----------
        theta_full : ndarray (14,)
            Full parameter vector
            
        Returns
        -------
        rel_error : float
            Relative ROM error in observable space (φ̄)
        """
        try:
            with timed(self.timing, "rom_error.compute"):
                # ROM solution
                with timed(self.timing, "tsm.solve_tsm"):
                    t_arr_rom, x0_rom, sig2_rom = self.tsm.solve_tsm(theta_full)
                
                # FOM solution
                self.fom_call_count += 1  # Track FOM evaluations
                with timed(self.timing, "fom.run_deterministic"):
                    t_arr_fom, x0_fom = self.solver.run_deterministic(theta_full)
            
            # Compute φ̄ (observable) at observation times for comparison
            # φ̄_i = φ_i * ψ_i (living bacteria volume fraction)
            phibar_rom = compute_phibar(x0_rom, self.active_species)
            phibar_fom = compute_phibar(x0_fom, self.active_species)
            
            # Extract values at observation indices (sparse observations)
            phibar_rom_obs = phibar_rom[self.idx_sparse]
            phibar_fom_obs = phibar_fom[self.idx_sparse]
            
            # Relative error: || φ̄_ROM(t_obs) − φ̄_FOM(t_obs) ||₂ / || φ̄_FOM(t_obs) ||₂
            error_norm = np.linalg.norm(phibar_rom_obs - phibar_fom_obs)
            fom_norm = np.linalg.norm(phibar_fom_obs)

            # CRITICAL SAFETY:
            # If ||φ̄_FOM|| is (near) zero, the usual *relative* error becomes ill-posed.
            # Returning 0.0 here is dangerous: it can incorrectly signal "perfect ROM"
            # and enable/stop linearization updates.
            #
            # Policy:
            # - If both ROM and FOM are essentially zero at observation points → error 0.0 (they match).
            # - Otherwise → return +inf (treat as unreliable / catastrophic), and log diagnostics.
            eps = 1e-10
            if (not np.isfinite(fom_norm)) or (not np.isfinite(error_norm)):
                return np.inf

            if fom_norm < eps:
                if error_norm < eps:
                    return 0.0
                # Diagnostics (keep it cheap; no heavy formatting in tight loops)
                if hasattr(self, "debug_logger") and self.debug_logger:
                    if self.debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                        try:
                            self.debug_logger.log_warning(
                                "ROM error is ill-posed because ||φ̄_FOM(obs)|| is near-zero "
                                f"(||φ̄_FOM||={fom_norm:.3e}, ||Δ||={error_norm:.3e}). "
                                "Returning +inf to avoid false '0.0' ROM error."
                            )
                        except Exception:
                            pass
                else:
                    logger.warning(
                        "ROM error ill-posed: ||φ̄_FOM(obs)|| near-zero (||φ̄_FOM||=%.3e, ||Δ||=%.3e). Returning +inf.",
                        float(fom_norm),
                        float(error_norm),
                    )
                return np.inf

            rel_error = error_norm / fom_norm
            return float(rel_error)
        except Exception as e:
            # ERROR/OFF mode: silent
            # MINIMAL/VERBOSE: log warning
            if hasattr(self, "debug_logger") and self.debug_logger:
                if self.debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                    self.debug_logger.log_warning(f"ROM error computation failed: {e}")
            else:
                logger.warning("ROM error computation failed: %s", e)
            return np.inf  # Return large error if computation fails
    
    def __call__(self, theta_sub: np.ndarray) -> float:
        """Evaluate log-likelihood for given parameter subset."""
        self.call_count += 1
        self.health.n_calls += 1
        
        # Construct full parameter vector
        full_theta = self.theta_base.copy()
        for i, idx in enumerate(self.active_indices):
            full_theta[idx] = theta_sub[i]
        
        # Solve TSM
        try:
            with timed(self.timing, "tsm.solve_tsm"):
                t_arr, x0, sig2 = self.tsm.solve_tsm(full_theta)
        except Exception as e:
            self.health.n_tsm_fail += 1
            # ERROR/OFF mode: silent
            # MINIMAL/VERBOSE: log warning
            if hasattr(self, "debug_logger") and self.debug_logger:
                if self.debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                    self.debug_logger.log_warning(f"TSM failed: {e}")
            else:
                logger.warning("TSM failed: %s", e)
            return -1e20

        # Basic sanity: solver outputs must be finite (counts for later triage)
        n_bad = 0
        try:
            n_bad += int(np.size(t_arr) - np.isfinite(t_arr).sum())
            n_bad += int(np.size(x0) - np.isfinite(x0).sum())
            n_bad += int(np.size(sig2) - np.isfinite(sig2).sum())
        except Exception:
            n_bad += 1
        if n_bad > 0:
            self.health.n_output_nonfinite += int(n_bad)
            return -1e20
        
        # Compute predicted mean and variance at observation times
        mu = np.zeros((len(self.idx_sparse), self.n_species))
        sig = np.zeros((len(self.idx_sparse), self.n_species))
        
        # State vector: [phi_0..phi_{N-1}, phi0, psi_0..psi_{N-1}, gamma]
        # For N total species, psi_offset = N + 1
        n_state = x0.shape[1]
        n_total_species = (n_state - 2) // 2
        psi_offset = n_total_species + 1
        
        for i, sp in enumerate(self.active_species):
            # CRITICAL FIX: Check bounds explicitly instead of silent clipping
            # Silent clipping can hide bugs (e.g., idx_sparse calculation errors)
            if np.any(self.idx_sparse < 0) or np.any(self.idx_sparse >= sig2.shape[0]):
                invalid_min = np.min(self.idx_sparse[self.idx_sparse < 0]) if np.any(self.idx_sparse < 0) else None
                invalid_max = np.max(self.idx_sparse[self.idx_sparse >= sig2.shape[0]]) if np.any(self.idx_sparse >= sig2.shape[0]) else None
                raise IndexError(
                    f"Invalid idx_sparse: min={invalid_min}, max={invalid_max}, "
                    f"valid range=[0, {sig2.shape[0]-1}]. "
                    f"idx_sparse shape={self.idx_sparse.shape}, sig2 shape={sig2.shape}"
                )
            idx = self.idx_sparse

            phi = x0[idx, sp]
            sig2_phi = sig2[idx, sp]

            if self.use_absolute_volume:
                # Use phi * gamma
                gamma_idx = n_state - 1
                gamma = x0[idx, gamma_idx]
                sig2_gamma = sig2[idx, gamma_idx]

                mu[:, i] = phi * gamma
                # Var(phi*gamma) = phi^2 Var(gamma) + gamma^2 Var(phi) + 2 phi gamma Cov(phi, gamma)
                var_phibar = phi**2 * sig2_gamma + gamma**2 * sig2_phi
            else:
                # Use phi * psi (Default)
                psi = x0[idx, psi_offset + sp]
                sig2_psi = sig2[idx, psi_offset + sp]

                mu[:, i] = phi * psi
                # Var(phi*psi) = phi^2 Var(psi) + psi^2 Var(phi) + 2 phi psi Cov(phi,psi)
                var_phibar = phi**2 * sig2_psi + psi**2 * sig2_phi

            # Cov(phi,psi) or Cov(phi,gamma) can be computed from sensitivities x1:
            # Cov(x_a, x_b) = Σ_k (∂x_a/∂θ_k)(∂x_b/∂θ_k) Var(θ_k), assuming independent θ_k.
            x1 = getattr(self.tsm, "_last_x1", None)
            var_act = getattr(self.tsm, "_last_var_act", None)
            if x1 is not None and var_act is not None:
                try:
                    x1_phi = x1[idx, sp, :]  # (n_obs, n_active)
                    
                    if self.use_absolute_volume:
                         gamma_idx = n_state - 1
                         x1_other = x1[idx, gamma_idx, :]
                         other_val = gamma
                    else:
                         x1_other = x1[idx, psi_offset + sp, :]
                         other_val = x0[idx, psi_offset + sp] # Re-fetch psi just in case, though we have it above

                    cov_phi_other = np.sum(x1_phi * x1_other * var_act[None, :], axis=1)
                    var_phibar = var_phibar + 2.0 * phi * other_val * cov_phi_other
                except Exception:
                    # Fall back to diagonal approximation if shapes mismatch
                    pass

            sig[:, i] = var_phibar
        
        # Sanity: likelihood inputs must be finite
        n_bad2 = int(np.size(mu) - np.isfinite(mu).sum()) + int(np.size(sig) - np.isfinite(sig).sum())
        if n_bad2 > 0:
            self.health.n_output_nonfinite += int(n_bad2)
            return -1e20

        # Evaluate log-likelihood + increment per-entry variance health counters
        var_health: Dict[str, int] = {}
        logL = log_likelihood_sparse(
            mu, sig, self.data, self.sigma_obs,
            rho=self.rho, health=var_health, weights=self.weights,
        )
        self.health.n_var_raw_negative += int(var_health.get("n_var_raw_negative", 0))
        self.health.n_var_raw_nonfinite += int(var_health.get("n_var_raw_nonfinite", 0))
        self.health.n_var_total_clipped += int(var_health.get("n_var_total_clipped", 0))
        
        # Track evaluation
        self.theta_history.append(theta_sub.copy())
        self.logL_history.append(logL)
        
        return logL

    def get_health(self) -> Dict[str, int]:
        """Get health counters as dictionary."""
        return self.health.to_dict()
    
    def get_MAP(self) -> Tuple[np.ndarray, float]:
        """Get MAP estimate from evaluation history."""
        if len(self.logL_history) == 0:
            raise ValueError("No evaluations yet")

        idx_max = np.argmax(self.logL_history)
        theta_MAP = self.theta_history[idx_max]
        logL_MAP = self.logL_history[idx_max]

        return theta_MAP, logL_MAP


class DeepONetEvaluator:
    """
    Log-likelihood evaluator using DeepONet surrogate (drop-in replacement for
    LogLikelihoodEvaluator).

    Achieves ~80× per-sample speedup (~29× E2E TMCMC) by replacing Newton ODE solver + TSM-ROM with a
    pre-trained DeepONet operator network.

    Interface is identical to LogLikelihoodEvaluator:
        evaluator(theta_sub) -> float  (log-likelihood)
    """

    def __init__(
        self,
        surrogate,  # DeepONetSurrogate instance
        active_species: List[int],
        active_indices: List[int],
        theta_base: np.ndarray,
        data: np.ndarray,
        idx_sparse: np.ndarray,
        sigma_obs: float,
        rho: float = 0.0,
        weights: Optional[np.ndarray] = None,
    ):
        self.surrogate = surrogate
        self.active_species = list(active_species)
        self.active_indices = list(active_indices)
        self.theta_base = theta_base.copy()
        self.data = data
        self.idx_sparse = idx_sparse
        self.sigma_obs = sigma_obs
        self.rho = rho
        self.n_species = len(active_species)
        self.weights = weights

        # Tracking (same interface as LogLikelihoodEvaluator)
        self.call_count = 0
        self.fom_call_count = 0
        self.timing = TimingStats()
        self.health = LikelihoodHealthCounter()
        self.theta_history = []
        self.logL_history = []

        # Stubs for compatibility with TMCMC linearization interface
        self._theta_linearization = theta_base.copy()
        self._linearization_enabled = False

    # --- Linearization stubs (DeepONet doesn't need them) ---

    def update_linearization_point(self, theta_new_full: np.ndarray):
        self._theta_linearization = theta_new_full.copy()

    def enable_linearization(self, enable: bool = True):
        self._linearization_enabled = enable

    def get_linearization_point(self) -> np.ndarray:
        return self._theta_linearization.copy()

    def compute_ROM_error(self, theta_full: np.ndarray) -> float:
        return 0.0  # DeepONet has no ROM linearization error

    # --- Main interface ---

    def __call__(self, theta_sub: np.ndarray) -> float:
        """Evaluate log-likelihood using DeepONet surrogate."""
        self.call_count += 1
        self.health.n_calls += 1

        # Construct full 20D parameter vector
        full_theta = self.theta_base.copy()
        for i, idx in enumerate(self.active_indices):
            full_theta[idx] = theta_sub[i]

        # DeepONet prediction
        try:
            phi_full = self.surrogate.predict(full_theta)  # (n_time, 5)
        except Exception:
            self.health.n_tsm_fail += 1
            return -1e20

        if not np.all(np.isfinite(phi_full)):
            self.health.n_output_nonfinite += 1
            return -1e20

        # Extract at observation times and active species
        phi_obs = phi_full[self.idx_sparse]  # (n_obs, 5)
        mu = phi_obs[:, self.active_species]  # (n_obs, n_active_species)

        # DeepONet is deterministic → variance = 0 (only observation noise)
        sig = np.zeros_like(mu)

        # Evaluate log-likelihood
        var_health: Dict[str, int] = {}
        logL = log_likelihood_sparse(
            mu, sig, self.data, self.sigma_obs,
            rho=self.rho, health=var_health, weights=self.weights,
        )
        self.health.n_var_raw_negative += int(var_health.get("n_var_raw_negative", 0))
        self.health.n_var_raw_nonfinite += int(var_health.get("n_var_raw_nonfinite", 0))
        self.health.n_var_total_clipped += int(var_health.get("n_var_total_clipped", 0))

        self.theta_history.append(theta_sub.copy())
        self.logL_history.append(logL)

        return logL

    def get_health(self) -> Dict[str, int]:
        return self.health.to_dict()

    def get_MAP(self) -> Tuple[np.ndarray, float]:
        if len(self.logL_history) == 0:
            raise ValueError("No evaluations yet")
        idx_max = np.argmax(self.logL_history)
        return self.theta_history[idx_max], self.logL_history[idx_max]
