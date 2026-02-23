"""
TMCMC (Transitional MCMC) algorithm implementation.

Extracted from case2_tmcmc_linearization.py for better modularity.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

import sys
from pathlib import Path

# Add project root to sys.path to allow tmcmc imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tmcmc.utils import validate_tmcmc_inputs, TimingStats
except ImportError:
    # Try local utils relative to core
    utils_dir = current_dir.parent / "utils"
    if str(utils_dir.parent) not in sys.path:
        sys.path.insert(0, str(utils_dir.parent))
    from utils import validate_tmcmc_inputs, TimingStats

# Backward compatibility alias
_validate_tmcmc_inputs = validate_tmcmc_inputs

try:
    from tmcmc.debug import DebugLogger, SLACK_ENABLED, notify_slack, _slack_notifier
except ImportError:
    try:
        from debug import DebugLogger, SLACK_ENABLED, notify_slack, _slack_notifier
    except ImportError:
        debug_dir = current_dir.parent / "debug"
        if str(debug_dir.parent) not in sys.path:
            sys.path.insert(0, str(debug_dir.parent))
        from debug import DebugLogger, SLACK_ENABLED, notify_slack, _slack_notifier

try:
    from tmcmc.visualization.helpers import compute_phibar
except ImportError:
    try:
        from visualization.helpers import compute_phibar
    except ImportError:
        vis_dir = current_dir.parent / "visualization"
        if str(vis_dir.parent) not in sys.path:
            sys.path.insert(0, str(vis_dir.parent))
        from visualization.helpers import compute_phibar

# Try to import from tmcmc.config, fallback to local config if not found
try:
    from tmcmc.config import (
        DebugConfig,
        DebugLevel,
        TMCMC_DEFAULTS,
        PROPOSAL_DEFAULTS,
        CONVERGENCE_DEFAULTS,
        ROM_ERROR_DEFAULTS,
        LINEARIZATION_DEFAULTS,
        MAX_THETA0_STEP_NORM,
        MAX_LINEARIZATION_SUBUPDATES_PER_EVENT,
    )
except ImportError:
    try:
        from config import (
            DebugConfig,
            DebugLevel,
            TMCMC_DEFAULTS,
            PROPOSAL_DEFAULTS,
            CONVERGENCE_DEFAULTS,
            ROM_ERROR_DEFAULTS,
            LINEARIZATION_DEFAULTS,
            MAX_THETA0_STEP_NORM,
            MAX_LINEARIZATION_SUBUPDATES_PER_EVENT,
        )
    except ImportError:
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        program_dir = current_dir.parent / "program2602"
        if str(program_dir) not in sys.path:
            sys.path.insert(0, str(program_dir))
        from config import (
            DebugConfig,
            DebugLevel,
            TMCMC_DEFAULTS,
            PROPOSAL_DEFAULTS,
            CONVERGENCE_DEFAULTS,
            ROM_ERROR_DEFAULTS,
            LINEARIZATION_DEFAULTS,
            MAX_THETA0_STEP_NORM,
            MAX_LINEARIZATION_SUBUPDATES_PER_EVENT,
        )

# Import external dependencies
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcmc_diagnostics import MCMCDiagnostics

# NOTE: run_adaptive_MCMC will be imported in run_multi_chain_MCMC function
# when core.mcmc module is created. For now, it's imported locally in the function.

logger = logging.getLogger(__name__)

# Import constants from config
DEFAULT_N_PARTICLES = TMCMC_DEFAULTS.n_particles
DEFAULT_N_STAGES = TMCMC_DEFAULTS.n_stages
DEFAULT_TARGET_ESS_RATIO = TMCMC_DEFAULTS.target_ess_ratio


def compute_MAP_with_uncertainty(
    samples: np.ndarray, logL: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute MAP estimate and posterior statistics."""
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    idx_map = np.argmax(logL)
    MAP = samples[idx_map]
    ci_lower = np.percentile(samples, 2.5, axis=0)
    ci_upper = np.percentile(samples, 97.5, axis=0)
    
    return {
        "mean": mean,
        "std": std,
        "MAP": MAP,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
DEFAULT_MIN_DELTA_BETA = TMCMC_DEFAULTS.min_delta_beta
DEFAULT_UPDATE_LINEARIZATION_INTERVAL = TMCMC_DEFAULTS.update_linearization_interval
DEFAULT_N_MUTATION_STEPS = TMCMC_DEFAULTS.n_mutation_steps
DEFAULT_LINEARIZATION_THRESHOLD = TMCMC_DEFAULTS.linearization_threshold
MAX_LINEARIZATION_UPDATES = TMCMC_DEFAULTS.max_linearization_updates
MAX_DELTA_BETA = TMCMC_DEFAULTS.max_delta_beta
MUTATION_SCALE_FACTOR = TMCMC_DEFAULTS.mutation_scale_factor

OPTIMAL_SCALE_FACTOR = PROPOSAL_DEFAULTS.optimal_scale_factor
COVARIANCE_NUGGET_BASE = PROPOSAL_DEFAULTS.covariance_nugget_base
COVARIANCE_NUGGET_SCALE = PROPOSAL_DEFAULTS.covariance_nugget_scale

BETA_CONVERGENCE_THRESHOLD = CONVERGENCE_DEFAULTS.beta_convergence_threshold
THETA_CONVERGENCE_THRESHOLD = CONVERGENCE_DEFAULTS.theta_convergence_threshold

ROM_ERROR_THRESHOLD = ROM_ERROR_DEFAULTS.threshold
ROM_ERROR_FALLBACK = ROM_ERROR_DEFAULTS.fallback


@dataclass
class TMCMCResult:
    """Result from Transitional MCMC."""
    samples: np.ndarray
    logL_values: np.ndarray
    theta_MAP: np.ndarray
    beta_schedule: List[float]
    converged: bool
    theta0_history: Optional[List[np.ndarray]] = None  # Linearization point update history
    n_linearization_updates: int = 0  # Number of linearization updates performed
    final_MAP: Optional[np.ndarray] = None  # Final MAP from this chain (for global sharing)
    rom_error_pre_history: Optional[List[float]] = None  # ROM error history (pre-update, debug)
    rom_error_history: Optional[List[float]] = None  # ROM error history at each update
    acc_rate_history: Optional[List[float]] = None  # Acceptance rate history per stage
    n_rom_evaluations: int = 0  # Number of ROM (TSM) evaluations (for cost analysis)
    n_fom_evaluations: int = 0  # Number of FOM evaluations (for ROM error computation)
    wall_time_s: float = 0.0  # Wall time for this TMCMC chain
    timing_breakdown_s: Optional[Dict[str, float]] = None  # e.g., {"tsm_s":..., "fom_s":..., "tmcmc_overhead_s":...}
    likelihood_health: Optional[Dict[str, int]] = None  # Likelihood/TSM health counters
    stage_summary: Optional[List[Dict[str, Any]]] = None  # Per-stage summary rows (for CSV export)


def reflect_into_bounds(x: float, low: float, high: float) -> float:
    """
    Reflect a value into bounds [low, high] using reflection (folding).
    
    Parameters
    ----------
    x : float
        Value to reflect
    low : float
        Lower bound
    high : float
        Upper bound
        
    Returns
    -------
    float
        Reflected value within [low, high]
    """
    width = high - low
    if width <= 0:
        return np.clip(x, low, high)
    y = x
    # Fold by reflection (works even if far outside)
    y = (y - low) % (2 * width)
    y = 2 * width - y if y > width else y
    return low + y


def _evaluate_single_particle(args: Tuple[Callable, np.ndarray]) -> float:
    """
    Evaluate log-likelihood for a single particle (for parallel processing).
    
    Parameters
    ----------
    args : Tuple[Callable, np.ndarray]
        Tuple of (log_likelihood function, theta particle)
        
    Returns
    -------
    float
        Log-likelihood value
    """
    log_likelihood, theta = args
    return log_likelihood(theta)


def evaluate_particles_parallel(
    log_likelihood: Callable,
    theta: np.ndarray,
    n_jobs: Optional[int] = None,
    use_threads: bool = False,
) -> np.ndarray:
    """
    Evaluate log-likelihood for multiple particles in parallel.
    
    Parameters
    ----------
    log_likelihood : Callable
        Log-likelihood function
    theta : np.ndarray, shape (n_particles, n_params)
        Parameter particles to evaluate
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all available CPUs.
        If 1, uses sequential evaluation.
    use_threads : bool
        If True, use ThreadPoolExecutor (for GIL-friendly code).
        If False, use ProcessPoolExecutor (for CPU-bound code).
        
    Returns
    -------
    np.ndarray, shape (n_particles,)
        Log-likelihood values for each particle
    """
    n_particles = theta.shape[0]
    
    # Sequential evaluation if n_jobs=1 or None and small number of particles
    if n_jobs == 1 or (n_jobs is None and n_particles < 100):
        return np.array([log_likelihood(t) for t in theta])
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)  # Leave one core free
    n_jobs = min(n_jobs, n_particles)  # Don't use more workers than particles
    
    # Prepare arguments
    args_list = [(log_likelihood, theta[i]) for i in range(n_particles)]
    
    # Choose executor type
    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    # Parallel evaluation
    logL = np.zeros(n_particles)
    try:
        with ExecutorClass(max_workers=n_jobs) as executor:
            future_to_idx = {executor.submit(_evaluate_single_particle, args): i 
                            for i, args in enumerate(args_list)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    logL[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Particle {idx} evaluation failed: {e}")
                    logL[idx] = -1e20
    except Exception as e:
        logger.error(f"Parallel evaluation failed, falling back to sequential: {e}")
        # Fallback to sequential
        return np.array([log_likelihood(t) for t in theta])
    
    return logL


def choose_subset_size(beta_next: float) -> int:
    """
    Choose subset size for ROM error evaluation based on β value.
    
    βが大きい（分布が狭い）ほど、サブセットサイズを小さくしてFOM評価を削減。
    
    Parameters
    ----------
    beta_next : float
        Next β value (0 to 1)
        
    Returns
    -------
    int
        Subset size for ROM error evaluation
    """
    if beta_next < 0.6:
        return 20
    elif beta_next < 0.85:
        return 10
    else:
        return 5


def should_do_fom_check(
    beta_next: float,
    stage: int,
    update_interval: int,
    delta_theta0: Optional[float],
    last_rom_error: Optional[float],
    delta_tol: float = 5e-4,
    rom_tol: float = 0.01,
) -> bool:
    """
    Determine if FOM check should be performed.
    
    「βが大きい（分布が狭い）」「線形化点がほぼ動かない」「ROM誤差も十分小さい」
    なら、FOMチェックをスキップして計算コストを削減。
    
    Parameters
    ----------
    beta_next : float
        Next β value
    stage : int
        Current stage
    update_interval : int
        Linearization update interval
    delta_theta0 : Optional[float]
        Last linearization point change ||Δθ₀||
    last_rom_error : Optional[float]
        Last ROM error value
    delta_tol : float
        Tolerance for linearization point change
    rom_tol : float
        Tolerance for ROM error
        
    Returns
    -------
    bool
        True if FOM check should be performed
    """
    # Must be update interval and β > 0.5
    if not (beta_next > 0.5 and (stage % update_interval == 0)):
        return False
    
    # Skip condition: if error is unknown, perform FOM check (safe side)
    if last_rom_error is None or delta_theta0 is None:
        return True  # First perform FOM check (safe side)
    
    # Skip if linearization point hasn't moved much
    if delta_theta0 < delta_tol:
        return False
    # Skip if ROM error is already small (hysteresis for stability)
    if last_rom_error < rom_tol:
        return False
    return True


def run_TMCMC(
    log_likelihood: callable,
    prior_bounds: List[Tuple[float, float]],
    n_particles: int = DEFAULT_N_PARTICLES,
    n_stages: int = DEFAULT_N_STAGES,
    target_ess_ratio: float = DEFAULT_TARGET_ESS_RATIO,
    min_delta_beta: float = DEFAULT_MIN_DELTA_BETA,
    max_delta_beta: float = MAX_DELTA_BETA,
    logL_scale: float = 1.0,  # Deprecated, kept for compatibility
    seed: Optional[int] = None,
    model_name: str = "",
    evaluator: Optional[Any] = None,  # ★ LogLikelihoodEvaluator instance (for linearization update)
    theta_base_full: Optional[np.ndarray] = None,  # ★ Full 14-dim theta base
    active_indices: Optional[List[int]] = None,  # ★ Active parameter indices
    update_linearization_interval: int = DEFAULT_UPDATE_LINEARIZATION_INTERVAL,  # ★ Update linearization every N stages
    n_mutation_steps: int = DEFAULT_N_MUTATION_STEPS,  # ★ Number of MCMC steps per particle (K-step mutation)
    use_observation_based_update: bool = True,  # ★ Use observation-based linearization update (ROM error weighted)
    linearization_threshold: float = DEFAULT_LINEARIZATION_THRESHOLD,
    linearization_enable_rom_threshold: float = 0.05,
    debug_logger: Optional[DebugLogger] = None,  # ★ Debug logger (for controlled output)
    force_beta_one: bool = False,  # ★ If True, force β=1.0 at final stage (paper runs)
    n_jobs: Optional[int] = None,  # ★ Number of parallel jobs for particle evaluation (None = auto, 1 = sequential)
    use_threads: bool = False,  # ★ Use threads instead of processes (for GIL-friendly code)
) -> TMCMCResult:
    """
    Transitional MCMC (TMCMC) with β tempering + Linearization Update.
    
    ★ 論文通りにβ（tempering）を入れることで、精度・安定性が向上！
    ★ さらに、各stageで線形化点を更新することで、TSM-ROMの精度が向上！
    
    TMCMCはβ=0（事前分布）からβ=1（事後分布）へ段階的に遷移することで、
    多峰性や鋭いピークがある場合でも安定した探索が可能。
    
    線形化点更新機能：
    - 各stageの後にMAPを計算
    - 一定間隔（update_linearization_interval）で線形化点を更新
    - TSM-ROMの近似精度が段階的に向上
    
    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function
    prior_bounds : List[Tuple[float, float]]
        (lower, upper) bounds for each parameter
    n_particles : int
        Number of particles (samples per stage)
    n_stages : int
        Maximum number of TMCMC stages
    target_ess_ratio : float
        Target ESS as fraction of n_particles (0.5 = 50% ESS)
    min_delta_beta : float
        Minimum β increment per stage (progress floor). Used as a safety lower bound.
    max_delta_beta : float
        Maximum β increment per stage (jump cap). Critical to avoid large β jumps that can
        cause weight collapse and acceptance≈0.
    logL_scale : float
        [DEPRECATED] Scale factor for likelihood. Currently ignored for consistency with TMCMC theory.
        All likelihood calculations (ESS, resampling, mutation) now use unscaled logL.
    seed : int, optional
        Random seed
    model_name : str
        Model identifier for logging
    evaluator : LogLikelihoodEvaluator, optional
        Evaluator instance with update_linearization_point() method
    theta_base_full : ndarray (14,), optional
        Full 14-dimensional parameter base (for constructing full theta)
    active_indices : List[int], optional
        Active parameter indices (for constructing full theta)
    update_linearization_interval : int
        Update linearization point every N stages (default: 3)
        
    Returns
    -------
    TMCMCResult
        Samples, log-likelihood values, MAP, beta schedule, and convergence status
    """
    # ★ INPUT VALIDATION
    _validate_tmcmc_inputs(
        log_likelihood=log_likelihood,
        prior_bounds=prior_bounds,
        n_particles=n_particles,
        n_stages=n_stages,
        target_ess_ratio=target_ess_ratio,
        evaluator=evaluator,
        theta_base_full=theta_base_full,
        active_indices=active_indices,
    )
    
    # ★ CRITICAL FIX: Use default_rng consistently (remove np.random.seed)
    # np.random.seed is deprecated and causes non-reproducibility issues
    # default_rng is the recommended approach for NumPy >= 1.17
    rng = np.random.default_rng(seed)
    tmcmc_wall_start = time.perf_counter()
    
    n_params = len(prior_bounds)
    
    def log_prior(theta: np.ndarray) -> float:
        for i, (low, high) in enumerate(prior_bounds):
            if not (low <= theta[i] <= high):
                return -np.inf
        return 0.0
    
    # Initialize particles from prior
    theta = np.zeros((n_particles, n_params))
    for i in range(n_particles):
        for j, (low, high) in enumerate(prior_bounds):
            theta[i, j] = rng.uniform(low, high)
    
    # Evaluate initial log-likelihood (parallelized)
    # ★ DIAGNOSTIC: Log parallelization settings
    if n_jobs is None:
        effective_n_jobs = max(1, mp.cpu_count() - 1) if n_particles >= 100 else 1
    else:
        effective_n_jobs = min(n_jobs, n_particles) if n_jobs > 1 else n_jobs
    logger.info(
        f"[{model_name}] Parallelization: n_jobs={n_jobs} (effective={effective_n_jobs}), "
        f"use_threads={use_threads}, n_particles={n_particles}"
    )
    logL = evaluate_particles_parallel(log_likelihood, theta, n_jobs=n_jobs, use_threads=use_threads)
    beta = 0.0
    beta_schedule = [beta]
    
    # ★ Track linearization point updates
    theta0_history = []
    n_linearization_updates = 0
    
    # ★ Track diagnostic histories
    # ROM error at each linearization update:
    # - rom_error_pre_history: computed BEFORE θ0 update (debugging)
    # - rom_error_history: computed AFTER θ0 update (this is what we gate on / report)
    rom_error_pre_history = []
    rom_error_history = []
    acc_rate_history = []  # Acceptance rate per stage
    theta_MAP_posterior_history = []  # ★ Track posterior MAP at each stage (for final MAP selection)
    stage_summary: List[Dict[str, Any]] = []  # ★ Per-stage summary rows (exportable)
    
    # ★ 優先度S: Track last ROM error and delta_theta0 for skip conditions
    last_rom_error: Optional[float] = None
    last_delta_theta0: Optional[float] = None
    
    # ★ Track evaluation counts (for cost analysis)
    initial_rom_count = 0
    initial_fom_count = 0
    if evaluator is not None:
        initial_rom_count = evaluator.call_count
        initial_fom_count = evaluator.fom_call_count
        theta0_initial = evaluator.get_linearization_point()
        if theta0_initial is not None:
            theta0_history.append(theta0_initial.copy())
    
    # Initialize debug logger if not provided
    if debug_logger is None:
        debug_logger = DebugLogger(DebugConfig(level=DebugLevel.OFF))
    
    # ★ Set Slack thread for debug logger (will be set after thread creation)
    # This allows DebugLogger to add messages to the thread
    
    # ★ ERROR-CHECK: Check initial numerical errors
    debug_logger.check_numerical_errors(logL, theta, context="Initialization")
    
    # ★ Force initial log output (always show start of TMCMC)
    debug_logger.log_info(f"Initial LogL: min={logL.min():.1f}, max={logL.max():.1f}", force=True)
    if model_name:
        debug_logger.log_info(f"Model: {model_name}", force=True)
    debug_logger.log_info(f"Starting TMCMC with {n_particles} particles, {n_stages} stages...", force=True)
    
    # ★ Slack notification: TMCMC start with thread support (if model_name provided)
    slack_thread_ts = None
    if SLACK_ENABLED and model_name:
        title = (
            f"🔄 {model_name} TMCMC Started\n"
            f"   Particles: {n_particles}\n"
            f"   Stages: {n_stages}\n"
            f"   Initial LogL: [{logL.min():.1f}, {logL.max():.1f}]"
        )
        slack_thread_ts = _slack_notifier.start_thread(title)
        # If thread not available, fallback to regular notification
        if slack_thread_ts is None:
            notify_slack(title, raise_on_error=False)
        else:
            # Set thread for debug logger so it can add messages to the thread
            debug_logger.set_slack_thread(slack_thread_ts)
    
    for stage in range(1, n_stages + 1):
        # ★ Force stage start log (always show progress)
        debug_logger.log_info(f"Stage {stage}/{n_stages} starting...", force=True)
        # Per-stage counters/flags for later CSV export
        rom_error_pre_stage: Optional[float] = None
        rom_error_post_stage: Optional[float] = None
        delta_theta0_stage: Optional[float] = None
        # ★ Slack notification: 削除（詳細すぎるため、重要な情報のみ送信）
        # 1. Calculate Beta using ESS-based adaptive schedule
        # ★ CRITICAL FIX: logL_scale を撤廃（TMCMC理論との一貫性のため）
        # logL_scale は ESS計算・resampling・mutation で不整合を引き起こす
        # ESS計算で「スケール済み尤度」を見て、実際の重み更新で「未スケール尤度」を使うと、
        # beta が異常に速く 1.0 に到達し、posterior 探索が成立しない
        logL_eff = logL  # ★ logL_scale を撤廃（一貫性のため）
        delta_low, delta_high = 0.0, 1.0 - beta
        
        # Binary search for optimal delta_beta
        ess_at_delta_low = None  # ★ PRIORITY B: ESS値を記録（診断用）
        for _ in range(25):
            mid = 0.5 * (delta_low + delta_high)
            x = mid * (logL_eff - np.max(logL_eff))  # Shift for stability
            w = np.exp(x)
            sum_w = np.sum(w)
            if sum_w <= 0:
                ess = 0
            else:
                w_norm = w / sum_w
                ess = 1.0 / np.sum(w_norm**2)
            
            if ess >= target_ess_ratio * n_particles:
                delta_low = mid
                ess_at_delta_low = ess  # 最終的なESS値を記録
            else:
                delta_high = mid
        
        # ★ 高速化＋安全オプション（全モード共通）:
        # - 下限:  ESS が許す範囲でも、進行幅が小さくなりすぎないように min_delta_beta を保証
        # - 上限:  一気に β=1.0 近くまで飛ばないように MAX_DELTA_BETA でクリップ
        # - さらに、1.0 を超えないように (1.0 - beta) でもクリップ
        delta_beta_raw = max(delta_low, min_delta_beta)
        delta_beta = min(delta_beta_raw, float(max_delta_beta), 1.0 - beta)

        beta_next = min(beta + delta_beta, 1.0)

        # Paper-oriented safety: if the user set too few stages, still hit β=1.0 at the end.
        # This is mainly for stable reporting/plots; note that a large final jump can increase degeneracy.
        if force_beta_one and stage == n_stages and beta_next < 1.0:
            debug_logger.log_warning(
                f"Forcing final β to 1.0 at stage {stage}/{n_stages} (β was {beta_next:.4f}). "
                "Consider increasing n_stages for a smoother tempering schedule."
            )
            beta_next = 1.0
            delta_beta = 1.0 - beta
        
        # ★ ERROR-CHECK: Check beta progression
        debug_logger.check_beta_progression(beta_next, delta_beta, stage, context=f"Stage {stage}")
        
        # 2. Resample with weights
        log_w_unnorm = (beta_next - beta) * logL
        log_w_unnorm -= np.max(log_w_unnorm)  # Shift to prevent overflow
        w = np.exp(log_w_unnorm)
        w_sum = np.sum(w)
        
        if w_sum <= 0 or not np.isfinite(w_sum):
            debug_logger.log_warning("Weight sum issue. Using uniform.")
            w = np.ones(n_particles) / n_particles
        else:
            w /= w_sum

        # Diagnostics: actual ESS from the weights we will *actually* resample with.
        # (This can differ slightly from the binary-search ESS due to min/max delta clipping.)
        ess_weights = None
        try:
            if np.all(np.isfinite(w)) and float(np.sum(w)) > 0:
                ess_weights = float(1.0 / np.sum(w**2))
        except Exception:
            ess_weights = None
        
        # ★ PRIORITY B: βスケジュールの診断ログ（各stageで出力）
        # ESS計算の結果と実際のbeta進行を記録
        # 重みの尖り具合も記録（max(log_w) - min(log_w)）
        log_w_range = np.max(log_w_unnorm) - np.min(log_w_unnorm) if len(log_w_unnorm) > 0 else 0.0
        ess_at_delta_low_str = f"{ess_at_delta_low:.1f}" if ess_at_delta_low is not None else "N/A"
        ess_weights_str = f"{ess_weights:.1f}" if ess_weights is not None else "N/A"
        w_max = float(np.max(w)) if len(w) > 0 else float("nan")
        w_min = float(np.min(w)) if len(w) > 0 else float("nan")
        # ★ Force beta schedule log (always show progress)
        beta_msg = (
            f"      [TMCMC] Stage {stage}: β={beta:.4f} → {beta_next:.4f} (Δ={delta_beta:.6f}), "
            f"ESS={ess_at_delta_low_str}/{target_ess_ratio*n_particles:.1f} (actual={ess_weights_str}), "
            f"logL range=[{logL.min():.2f}, {logL.max():.2f}], "
            f"log_w range={log_w_range:.2f}, w[min,max]=[{w_min:.2e},{w_max:.2e}]"
        )
        logger.info("%s", beta_msg)
        # ★ Slack notification: 削除（詳細すぎるため、重要な情報のみ送信）
        
        # Resample particles
        # ★ Store particles and weights BEFORE resampling for weighted barycenter computation
        theta_before_resample = theta.copy()
        logL_before_resample = logL.copy()
        weights_before_resample = w.copy()  # ★ Store weights for barycenter
        
        # ★ CRITICAL FIX: Compute tempered posterior for MAP calculation
        # TMCMC stage k posterior: π_k(θ) ∝ p(θ) * p(D|θ)^β_k
        # So log_posterior = log_prior + beta * logL
        log_prior_before_resample = np.array([log_prior(t) for t in theta_before_resample])
        log_posterior_before_resample = log_prior_before_resample + beta_next * logL_before_resample
        
        idx = rng.choice(n_particles, size=n_particles, p=w)
        # Diagnostics: particle degeneracy after resampling (how many unique ancestors survived)
        try:
            n_unique_idx = int(np.unique(idx).size)
            unique_ratio = float(n_unique_idx) / float(n_particles)
            if debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                debug_logger.log_info(
                    f"Resample degeneracy: unique={n_unique_idx}/{n_particles} (unique_ratio={unique_ratio:.3f})"
                )
        except Exception:
            pass
        theta = theta[idx]
        logL = logL[idx]
        # Keep a copy of the post-resample population for potential recovery retries.
        theta_after_resample = theta.copy()
        logL_after_resample = logL.copy()
        
        # 3. Mutate (K-step MCMC with tempered posterior)
        # ★ 改善: 1-step → K-step mutation (resampling後の粒子相関を減らす)
        # 理由: resampling後は粒子が強く相関しているため、1-stepではESSが見かけ倒しになる
        # ★ 改善: Tempered covariance scaling (Del Moral et al., Ching & Chen)
        # Early stages (small β) need larger proposal variance for exploration
        cov_base = np.cov(theta.T)
        
        # ★ CRITICAL FIX: Handle 1D case (n_params == 1)
        # np.cov() returns scalar or 1D array for 1D input, but np.trace() requires 2D+
        # Ensure cov_base is always 2D for consistent handling
        if n_params == 1:
            # For 1D: cov_base is scalar, convert to 2D array
            cov_base = np.array([[cov_base]]) if np.isscalar(cov_base) else np.array([[cov_base.item()]])
        else:
            # For multi-D: ensure it's 2D (should already be, but be safe)
            if cov_base.ndim == 0:
                cov_base = np.array([[cov_base]])
            elif cov_base.ndim == 1:
                cov_base = np.diag(cov_base)
        
        # Optimal scaling: 2.38^2 / n_params (Gelman et al., 1996)
        # Tempered scaling: scale inversely with β (larger variance when β is small)
        optimal_scale = OPTIMAL_SCALE_FACTOR / n_params
        tempered_scale = optimal_scale / max(beta_next, 0.1)  # Avoid division by zero
        
        # ★ Adaptive scaling based on previous acceptance rate
        # - Low acceptance typically means steps are too large → reduce scale.
        # - Very high acceptance can mean steps are too small → slightly increase scale.
        adaptive_scale_factor = MUTATION_SCALE_FACTOR
        if len(acc_rate_history) > 0:
            prev_acc_rate = float(acc_rate_history[-1])
            if prev_acc_rate < 0.05:
                # Reduce scale factor when acceptance rate is very low
                # (cap at 0.1x to avoid freezing completely)
                shrink = max(0.1, prev_acc_rate / 0.05)
                adaptive_scale_factor = MUTATION_SCALE_FACTOR * shrink
                debug_logger.log_info(
                    f"⚠️  Low acceptance rate ({prev_acc_rate:.3f}), reducing proposal scale: {adaptive_scale_factor:.2f}x"
                )
            elif prev_acc_rate > 0.6:
                # Slightly increase step size if acceptance is extremely high
                grow = min(2.0, prev_acc_rate / 0.6)
                adaptive_scale_factor = MUTATION_SCALE_FACTOR * grow
                debug_logger.log_info(
                    f"ℹ️  High acceptance rate ({prev_acc_rate:.3f}), increasing proposal scale: {adaptive_scale_factor:.2f}x"
                )
        
        # ★ Global knob: MUTATION_SCALE_FACTOR controls overall jump size (and thus acceptance)
        cov = cov_base * (adaptive_scale_factor * tempered_scale)
        
        # ★ 優先度A: 共分散の正則化をスケール依存に（ロバスト性↑、歪み↓）
        # 固定 1e-6 はスケールによって大きすぎることがある
        # traceベースの正則化で、共分散の大きさに比例させる
        # ★ CRITICAL FIX: np.trace() requires 2D array, which we've ensured above
        scale = np.trace(cov_base) / n_params
        nugget = COVARIANCE_NUGGET_BASE + COVARIANCE_NUGGET_SCALE * scale
        cov += nugget * np.eye(n_params)

        # Diagnostics: proposal covariance scale/conditioning (helps explain extreme acceptance rates)
        try:
            if debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                cov_trace = float(np.trace(cov))
                covbase_trace = float(np.trace(cov_base))
                cond = None
                try:
                    cond = float(np.linalg.cond(cov))
                except Exception:
                    cond = None
                cond_str = f"{cond:.2e}" if cond is not None and np.isfinite(cond) else "N/A"
                debug_logger.log_info(
                    "Proposal cov stats: "
                    f"trace(cov_base)={covbase_trace:.2e}, trace(cov)={cov_trace:.2e}, "
                    f"scale={scale:.2e}, nugget={nugget:.2e}, "
                    f"tempered_scale={float(tempered_scale):.2e}, adapt_scale={float(adaptive_scale_factor):.2f}x, "
                    f"cond(cov)={cond_str}"
                )
        except Exception:
            pass
        
        # ★ ERROR-CHECK: Check covariance matrix validity
        debug_logger.check_covariance_matrix(cov, context=f"Stage {stage}, mutation covariance")
        
        def _mutate_population(cov_matrix: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
            """
            K-step mutation for the whole population. Returns (theta, logL, acc_rate, n_accepted, n_total).
            
            ★ PARALLELIZATION: Batch ALL proposals across all particles and evaluate in parallel.
            This maximizes parallelization efficiency by evaluating all proposals at once.
            """
            _theta = theta_after_resample.copy()
            _logL = logL_after_resample.copy()
            _acc = 0
            _total = 0
            n_steps = int(max(1, steps))
            
            mutation_start_time = time.perf_counter()
            
            # ★ PARALLELIZATION: Generate all proposals first, then evaluate in one batch
            # Structure: proposals[i][step] = proposal for particle i at step step
            all_proposals = []  # List of (particle_idx, step_idx, proposal, current_state)
            
            for i in range(n_particles):
                theta_current = _theta[i].copy()
                logL_current = _logL[i]
                for step in range(n_steps):
                    prop = rng.multivariate_normal(theta_current, cov_matrix)
                    for j, (low, high) in enumerate(prior_bounds):
                        prop[j] = reflect_into_bounds(prop[j], low, high)
                    lp_p = log_prior(prop)
                    if np.isfinite(lp_p):
                        all_proposals.append((i, step, prop, (theta_current.copy(), logL_current)))
            
            # ★ PARALLELIZATION: Evaluate all proposals in one batch
            if len(all_proposals) > 0:
                proposals_array = np.array([prop for _, _, prop, _ in all_proposals])
                ll_proposals = evaluate_particles_parallel(log_likelihood, proposals_array, n_jobs=n_jobs, use_threads=use_threads)
            else:
                ll_proposals = np.array([])
            
            # Process proposals sequentially per particle (MCMC acceptance logic)
            # Group proposals by particle and step to process in correct order
            proposals_by_particle = {i: [] for i in range(n_particles)}
            for prop_idx, (i, step, prop, (theta_init, logL_init)) in enumerate(all_proposals):
                proposals_by_particle[i].append((step, prop, ll_proposals[prop_idx], prop_idx))
            
            # Process each particle sequentially (state updates must be sequential)
            for i in range(n_particles):
                theta_current = _theta[i].copy()
                logL_current = _logL[i]
                
                # Sort proposals by step to process in order
                particle_proposals = sorted(proposals_by_particle[i], key=lambda x: x[0])
                
                for step, prop, ll_p, prop_idx in particle_proposals:
                    _total += 1
                    if not np.isfinite(ll_p):
                        continue
                    
                    log_ratio = (log_prior(prop) + beta_next * ll_p) - (log_prior(theta_current) + beta_next * logL_current)
                    if np.log(rng.random()) < log_ratio:
                        theta_current = prop.copy()
                        logL_current = ll_p
                        _acc += 1
                
                _theta[i] = theta_current
                _logL[i] = logL_current
            
            mutation_time = time.perf_counter() - mutation_start_time
            _acc_rate = _acc / _total if _total > 0 else 0.0
            
            # ★ DIAGNOSTIC: Log mutation performance (only for first stage to avoid spam)
            if stage == 1 and debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                effective_n_jobs = n_jobs if n_jobs is not None and n_jobs > 1 else (max(1, mp.cpu_count() - 1) if n_particles >= 100 else 1)
                logger.info(
                    f"[{model_name}] Mutation Stage {stage}: {len(all_proposals)} proposals evaluated in parallel "
                    f"(n_jobs={effective_n_jobs}), {_acc} accepted (rate={_acc_rate:.4f}), time={mutation_time:.2f}s"
                )
            
            return _theta, _logL, _acc_rate, int(_acc), int(_total)

        # First mutation attempt
        theta, logL, acc_rate, acc, total_proposals = _mutate_population(cov, n_mutation_steps)

        # Diagnostics: population diversity after mutation (rounded unique rows to tolerate tiny FP noise)
        try:
            if debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                theta_rounded = np.round(theta, 6)
                n_unique_theta = int(np.unique(theta_rounded, axis=0).shape[0])
                uniq_ratio = float(n_unique_theta) / float(n_particles)
                debug_logger.log_info(
                    f"Post-mutation diversity: unique≈{n_unique_theta}/{n_particles} (unique_ratio≈{uniq_ratio:.3f})"
                )
        except Exception:
            pass

        # Recovery when mutation gets stuck (avoid "continue with degenerate samples")
        # - Retry with smaller proposal scale
        # - If still stuck, add tiny jitter and retry once more
        if acc_rate < debug_logger.config.min_acceptance_rate:
            debug_logger.log_warning(
                f"Stage {stage}: acceptance rate {acc_rate:.4f} < {debug_logger.config.min_acceptance_rate:.4f}. "
                "Attempting recovery (shrink proposal covariance)."
            )
            theta, logL, acc_rate, acc, total_proposals = _mutate_population(cov * 0.3, max(1, n_mutation_steps // 2))

        if acc_rate < debug_logger.config.min_acceptance_rate:
            # Jitter around resampled population to break exact duplicates, then retry with small covariance.
            debug_logger.log_warning(
                f"Stage {stage}: still stuck after shrink (acc_rate={acc_rate:.4f}). "
                "Applying small jitter and retrying."
            )
            theta_after_resample = theta_after_resample.copy()
            jitter = rng.normal(loc=0.0, scale=1e-3, size=theta_after_resample.shape)
            theta_after_resample += jitter
            # Reflect jittered points back into bounds
            for i in range(n_particles):
                for j, (low, high) in enumerate(prior_bounds):
                    theta_after_resample[i, j] = reflect_into_bounds(theta_after_resample[i, j], low, high)
            # ★ PARALLELIZATION: Use parallel evaluation for jittered points
            logL_after_resample = evaluate_particles_parallel(log_likelihood, theta_after_resample, n_jobs=n_jobs, use_threads=use_threads)
            theta, logL, acc_rate, acc, total_proposals = _mutate_population(cov * 0.1, 1)

        # If still stuck, fail fast (do not proceed with a degenerate posterior)
        if acc_rate < debug_logger.config.min_acceptance_rate:
            raise RuntimeError(
                f"TMCMC mutation stuck: acc_rate={acc_rate:.4f} < {debug_logger.config.min_acceptance_rate:.4f} "
                f"after recovery attempts. Stage={stage}, beta_next={beta_next:.4f}."
            )
        
        # 4. ★ Update linearization point (if evaluator provided)
        # ⚠️ 重要: 線形化点更新後は必ずlogLを再計算する必要がある
        # 
        # ★ 理論的保証（論文での説明用）:
        # The proposed algorithm can be interpreted as a fixed-point iteration on the
        # linearization point under a progressively sharpened posterior (β: 0 → 1).
        # As β increases, the posterior becomes sharper, and the weighted barycenter
        # converges to the true posterior mean, providing a natural stopping criterion
        # for linearization point updates.
        # 
        # ★ LINEARIZATION POINT UPDATE TIMING RULE:
        # The linearization point is selected from particles AFTER mutation (theta, logL).
        # This ensures the linearization point reflects the current stage's posterior exploration
        # and is most consistent across stages.
        # 
        # Note: Observation-based ROM error computation uses theta_before_resample (correct),
        # but the final linearization point selection uses mutation result (theta, logL).
        if evaluator is not None and theta_base_full is not None and active_indices is not None:
            should_update = False
            
            # Phase A / stability:
            # Do NOT enable linearization immediately just because β crossed a threshold.
            # Enable only after MAP-based ROM error check at an update event.
            
            # Check if it's time to update (interval-based + β threshold)
            # ★ 改善: βが小さい段階（priorに近い）では線形化点更新を避ける
            # 理由: posteriorが十分sharpになってから更新することで、ROMの精度が向上
            # ★ 修正: stageベースに戻す（update_attempt_count のバグを回避）
            # update_attempt_count は should_update=True のときしか増えないため、
            # 一度 1 になると interval の倍数に戻らず更新が止まる問題があった
            if (beta_next > 0.5 and (stage % update_linearization_interval == 0)) or stage == n_stages:
                should_update = True
            elif beta_next <= 0.5:
                should_update = False  # Skip update if β is too small
            
            # ★ CRITICAL FIX: Initialize MAP variables
            # Separate posterior MAP (statistical) from linearization MAP (numerical)
            # 
            # theta_MAP_posterior: Statistical MAP estimate of the posterior distribution at stage k
            #   - Used for reporting, convergence diagnostics, and final results
            #   - Must reflect observation information if observation-based update is used
            #   - Should NOT be overwritten after observation-based correction
            #   - Definition: argmax_θ [log p(D|θ)^β_k * p(θ)] with observation correction
            # 
            # theta_MAP_linearize: Numerical anchor point for TSM-ROM linearization
            #   - Used for linearization point update in ROM
            #   - May differ from posterior MAP for numerical stability
            #   - Can be recomputed if needed for ROM accuracy
            #   - Default: same as posterior MAP, but can be overridden
            theta_MAP_posterior = None
            theta_MAP_linearize = None
            idx_MAP_posterior = None
            theta_MAP_posterior_computed = False
            theta_MAP_posterior_obs_corrected = None  # For assertion: verify observation-corrected MAP is not overwritten
            
            # Check convergence condition (stop if MAP hasn't moved much)
            # ★ CRITICAL FIX: Initialize should_do_fom to ensure it's defined in all code paths
            should_do_fom = True  # Default: do FOM check unless conditions suggest skipping
            
            if should_update:
                # ★ 新アイデア: 観測量ベースの線形化点更新（論文に強い）
                # 線形化点を「パラメータ空間」ではなく「観測量φ̄（データ）に基づいて」更新
                # 
                # 数式: θ₀_new = Σ_i [w_i / (1 + ε_obs(θ_i))] * θ_i
                # where:
                #   w_i: TMCMCの重み
                #   ε_obs(θ_i): 観測点φ̄におけるROM-FOM誤差
                #
                # 効果:
                # - データに効かないROM誤差は無視
                # - 観測に重要な方向だけに線形化点が引き寄せられる
                # - posterior精度が大幅向上、多峰性でも安定
                #
                # 論文での説明:
                # "The linearization point is updated to minimize approximation error
                #  at observation points that contribute to the likelihood."
                
                if use_observation_based_update:
                    # ★ 優先度S: FOMチェックのスキップ条件
                    # 「βが大きい（分布が狭い）」「線形化点がほぼ動かない」「ROM誤差も十分小さい」
                    # なら、FOMチェックをスキップして計算コストを削減
                    should_do_fom = should_do_fom_check(
                        beta_next=beta_next,
                        stage=stage,
                        update_interval=update_linearization_interval,
                        delta_theta0=last_delta_theta0,
                        last_rom_error=last_rom_error,
                    )
                    
                    if not should_do_fom:
                        # Skip FOM evaluation, MAP is already computed above
                        # ★ PRIORITY D: None フォーマット例外を確実に潰す
                        dtheta_str = "None" if last_delta_theta0 is None else f"{last_delta_theta0:.6f}"
                        rom_str = "None" if last_rom_error is None else f"{last_rom_error:.6f}"
                        debug_logger.log_info(f"Skipping FOM check (β={beta_next:.3f}, ||Δθ₀||={dtheta_str}, ε_ROM={rom_str})")
                        # MAP is already computed above (idx_MAP_stage, theta_MAP_stage, theta_full_MAP)
                        # ★ 2) スキップした場合のrom_error_historyの整合性
                        # スキップ時はnp.nanをappend（後でプロットでnan無視できる）
                        # Note: This will be appended later in the ROM error check section
                    else:
                        # ★ 優先度S: 動的サブセットサイズ（βに応じて縮める）
                        # βが大きい（分布が狭い）ほど、サブセットサイズを小さくしてFOM評価を削減
                        subset_size_base = choose_subset_size(beta_next)
                        subset_size = min(subset_size_base, n_particles)
                        
                        # ★ 優先度B: 重み付き・層化サンプリング（k-meansより軽い強化案）
                        # 重み上位＋ランダム（外れ値も拾う）
                        # ★ 3) subsetの層化サンプリングが「重みゼロ/NaN」でも壊れないか
                        weights_safe = weights_before_resample.copy()
                        # Check for NaN/Inf in weights
                        if not np.all(np.isfinite(weights_safe)):
                            # Fallback: use uniform weights if NaN/Inf detected
                            weights_safe = np.ones(n_particles) / n_particles
                            debug_logger.log_warning("Weights contain NaN/Inf, using uniform weights for subset selection")
                        
                        m = subset_size // 2
                        # Top particles by weight
                        top_idx = np.argsort(weights_safe)[-min(5*subset_size, n_particles):]
                        # ★ 上位候補集合が小さすぎるときはreplaceを許可 or 全体ランダムへ
                        if len(top_idx) < m:
                            # Not enough top candidates, use random sampling
                            subset_top = rng.choice(n_particles, size=m, replace=False)
                        else:
                            subset_top = rng.choice(top_idx, size=min(m, len(top_idx)), replace=False)
                        # Random particles (catch outliers)
                        subset_rand = rng.choice(n_particles, size=subset_size - len(subset_top), replace=False)
                        subset_idx = np.unique(np.concatenate([subset_top, subset_rand]))
                        subset_size = len(subset_idx)  # Actual size after deduplication
                        
                        debug_logger.log_observation_based_update(subset_size, n_particles)
                        
                        # ★ 修正: NaNで初期化（未計算の粒子を明確に区別）
                        # rom_errors == 0 は危険（本当に誤差0の粒子と区別できない）
                        rom_errors = np.full(n_particles, np.nan)
                        
                        # Step 1: サブセットのみROM誤差を計算
                        for i in subset_idx:
                            # Construct full theta for particle i
                            theta_i_full = theta_base_full.copy()
                            for j, idx in enumerate(active_indices):
                                theta_i_full[idx] = theta_before_resample[i, j]
                            
                            # Compute ROM error at observation points
                            try:
                                rom_errors[i] = evaluator.compute_ROM_error(theta_i_full)
                            except Exception as e:
                                # If error computation fails, use large error (low weight)
                                rom_errors[i] = 1.0  # Large error → low weight
                                debug_logger.log_warning(f"ROM error computation failed for particle {i}: {e}")
                        
                        # Step 2: サブセットのROM誤差から平均誤差を推定
                        # サブセットに含まれない粒子は平均誤差を使用
                        # ★ CRITICAL FIX: Handle all-NaN case (fallback to large error)
                        mean_rom_error_subset = np.nanmean(rom_errors)
                        if np.isnan(mean_rom_error_subset):
                            # All ROM errors are NaN: use large fallback value
                            mean_rom_error_subset = ROM_ERROR_FALLBACK
                            debug_logger.log_warning(f"All ROM errors are NaN, using fallback value {ROM_ERROR_FALLBACK}")
                        
                        # ★ 修正: NaNの粒子に平均値を割り当て（意味が明確、reviewerに説明しやすい）
                        rom_errors = np.where(
                            np.isnan(rom_errors),
                            mean_rom_error_subset,
                            rom_errors
                        )
                        
                        # Step 3: 重みを修正（観測点でのROM誤差を考慮）
                        # w_i' = w_i / (1 + ε_obs(θ_i))
                        # 誤差が大きい粒子の重みを下げる
                        weights_obs_corrected = weights_before_resample.copy()
                        for i in range(n_particles):
                            weights_obs_corrected[i] = weights_before_resample[i] / (1.0 + rom_errors[i])
                        
                        # Normalize corrected weights
                        weights_sum = np.sum(weights_obs_corrected)
                        if weights_sum > 0:
                            weights_obs_corrected /= weights_sum
                        else:
                            # Fallback to original weights if all errors are too large
                            weights_obs_corrected = weights_before_resample.copy()
                            debug_logger.log_warning("All ROM errors too large, using original weights")
                        
                        # ★ CRITICAL FIX: Compute posterior MAP using observation-corrected weights
                        # This MAP reflects observation information and should NOT be overwritten
                        # 
                        # Theory: The observation correction weights particles by ROM error at observation points.
                        # The corrected weight is: w_obs_corrected = w_original / (1 + ε_obs(θ))
                        # This means particles with lower ROM error at observation points get higher weight.
                        # 
                        # For posterior MAP: we want argmax_θ [log p(D|θ)^β * p(θ)] with observation correction.
                        # TMCMC stage k posterior: π_k(θ) ∝ p(θ) * p(D|θ)^β_k
                        # So log_posterior = log_prior + beta * logL
                        # 
                        # ★ APPROACH: Use observation-corrected posterior score
                        # Score = log_posterior - log(1 + ε_obs)
                        #       = (log_prior + beta * logL) - log(1 + ε_obs)
                        # This gives higher score to particles with both high posterior AND low ROM error.
                        # 
                        # Compute observation-corrected posterior score for each particle
                        obs_corrected_scores = log_posterior_before_resample.copy()
                        log_penalty = np.log(1.0 + rom_errors)
                        obs_corrected_scores -= log_penalty
                        
                        # ★ SCALE VERIFICATION: Log distributions to detect scale issues
                        logL_min, logL_median, logL_max = np.min(logL_before_resample), np.median(logL_before_resample), np.max(logL_before_resample)
                        log_penalty_min, log_penalty_median, log_penalty_max = np.min(log_penalty), np.median(log_penalty), np.max(log_penalty)
                        log_post_min, log_post_median, log_post_max = np.min(log_posterior_before_resample), np.median(log_posterior_before_resample), np.max(log_posterior_before_resample)
                        obs_score_min, obs_score_median, obs_score_max = np.min(obs_corrected_scores), np.median(obs_corrected_scores), np.max(obs_corrected_scores)
                        
                        debug_logger.log_info(
                            f"Scale check: logL=[{logL_min:.2f}, {logL_median:.2f}, {logL_max:.2f}], "
                            f"log_penalty=[{log_penalty_min:.4f}, {log_penalty_median:.4f}, {log_penalty_max:.4f}], "
                            f"log_post=[{log_post_min:.2f}, {log_post_median:.2f}, {log_post_max:.2f}], "
                            f"obs_score=[{obs_score_min:.2f}, {obs_score_median:.2f}, {obs_score_max:.2f}]"
                        )
                        
                        # Check if penalty is too weak (penalty << logL scale)
                        penalty_ratio = np.max(log_penalty) / (np.max(logL_before_resample) - np.min(logL_before_resample) + 1e-10)
                        if penalty_ratio < 0.01:
                            debug_logger.log_warning(f"Observation penalty may be too weak (max_penalty/max_logL_range={penalty_ratio:.4f} < 0.01)")
                        elif penalty_ratio > 0.1:
                            debug_logger.log_warning(f"Observation penalty may be too strong (max_penalty/max_logL_range={penalty_ratio:.4f} > 0.1)")
                        
                        # Find particle with highest observation-corrected posterior score
                        idx_MAP_posterior = np.argmax(obs_corrected_scores)
                        theta_MAP_posterior = theta_before_resample[idx_MAP_posterior]
                        
                        # Also compute standard MAP (without observation correction) for comparison
                        idx_MAP_standard = np.argmax(log_posterior_before_resample)
                        theta_MAP_standard = theta_before_resample[idx_MAP_standard]
                        
                        # Log the difference between standard MAP and observation-corrected MAP
                        if idx_MAP_posterior != idx_MAP_standard:
                            delta_map = np.linalg.norm(theta_MAP_posterior - theta_MAP_standard)
                            debug_logger.log_info(f"Observation-corrected MAP differs from standard MAP: ||Δ||={delta_map:.6f}")
                        else:
                            debug_logger.log_info("Observation-corrected MAP matches standard MAP (no correction effect)")
                        
                        # Report statistics
                        mean_rom_error = np.mean(rom_errors)
                        max_rom_error = np.max(rom_errors)
                        min_rom_error = np.min(rom_errors)
                        debug_logger.log_info(f"ROM errors: mean={mean_rom_error:.6f}, min={min_rom_error:.6f}, max={max_rom_error:.6f}")
                        
                        # Compute weighted means for comparison (optional, not used for linearization)
                        theta_weighted_mean_original = np.zeros(n_params)
                        for i in range(n_particles):
                            theta_weighted_mean_original += weights_before_resample[i] * theta_before_resample[i]
                        
                        theta_weighted_mean_obs = np.zeros(n_params)
                        for i in range(n_particles):
                            theta_weighted_mean_obs += weights_obs_corrected[i] * theta_before_resample[i]
                        
                        # Report difference between MAP and weighted means for comparison
                        delta_map_weighted_mean = np.linalg.norm(theta_MAP_posterior - theta_weighted_mean_original)
                        delta_weighted_mean_shift = np.linalg.norm(theta_weighted_mean_obs - theta_weighted_mean_original)
                        debug_logger.log_info(f"Posterior MAP-WeightedMean distance: {delta_map_weighted_mean:.6f}, WeightedMean shift (obs-corrected): {delta_weighted_mean_shift:.6f}")
                        
                        # Mark that posterior MAP has been computed with observation correction
                        theta_MAP_posterior_computed = True
                        
                        # ★ ASSERT: Observation-corrected MAP should NOT be overwritten
                        # Store a copy for verification
                        theta_MAP_posterior_obs_corrected = theta_MAP_posterior.copy()
                else:
                    # No observation-based update: compute standard MAP
                    theta_MAP_posterior_computed = False
                    theta_MAP_posterior_obs_corrected = None
                
                # ★ CRITICAL FIX: Separate Maximum Likelihood (ML) from MAP (posterior)
                # 
                # theta_ML_stage: Maximum Likelihood particle (argmax logL)
                #   - Used for comparison and diagnostics
                #   - Does NOT include prior information
                #   - Selected from theta_before_resample (for observation-based update compatibility)
                # 
                # theta_MAP_posterior: Statistical MAP estimate of the posterior distribution at stage k
                #   - Definition: argmax_θ [log p(θ) + β_k * log p(D|θ)]
                #   - Must reflect observation information if observation-based update is used
                #   - Should NOT be overwritten after observation-based correction
                #   - Used for reporting, convergence diagnostics, and final results
                #   - Selected from theta_before_resample (for observation-based update compatibility)
                # 
                # theta_MAP_linearize: Numerical anchor point for TSM-ROM linearization
                #   - Used for linearization point update in ROM
                #   - ★ UPDATE TIMING RULE: Selected from particles AFTER mutation (theta, logL)
                #   - This ensures consistency with current stage's posterior exploration
                #   - May differ from posterior MAP for numerical stability
                #   - Default: same as posterior MAP from mutation result, but can be overridden
                
                if not theta_MAP_posterior_computed:
                    # Standard MAP computation (no observation correction)
                    # Use tempered posterior: log_prior + beta * logL
                    idx_MAP_posterior = np.argmax(log_posterior_before_resample)
                    theta_MAP_posterior = theta_before_resample[idx_MAP_posterior]
                
                # Also compute ML for comparison (maximum likelihood, no prior)
                idx_ML_stage = np.argmax(logL_before_resample)
                theta_ML_stage = theta_before_resample[idx_ML_stage]
                
                # ★ CRITICAL: Record posterior MAP for final MAP selection
                # This ensures observation-corrected MAP is preserved
                # ★ ASSERT: Posterior MAP should NOT be overwritten after observation-based update
                if theta_MAP_posterior_computed:
                    # Verify that observation-corrected MAP is being used
                    assert theta_MAP_posterior is not None, "Observation-corrected MAP should be computed"
                    # ★ ASSERT: Verify that observation-corrected MAP was not overwritten
                    if theta_MAP_posterior_obs_corrected is not None:
                        assert np.allclose(theta_MAP_posterior, theta_MAP_posterior_obs_corrected), \
                            "Observation-corrected MAP should NOT be overwritten after computation"
                    # Store the observation-corrected MAP
                    theta_MAP_posterior_history.append(theta_MAP_posterior.copy())
                else:
                    # Store standard MAP
                    theta_MAP_posterior_history.append(theta_MAP_posterior.copy())
                
                # ★ LINEARIZATION POINT SELECTION: Use mutation result (theta, logL)
                # Compute tempered posterior for mutation result
                log_prior_after_mutation = np.array([log_prior(t) for t in theta])
                log_posterior_after_mutation = log_prior_after_mutation + beta_next * logL
                
                # Select linearization MAP from mutation result
                idx_MAP_linearize = np.argmax(log_posterior_after_mutation)
                theta_MAP_linearize = theta[idx_MAP_linearize]
                
                # Store for backward compatibility (will be used for linearization)
                idx_MAP_stage = idx_MAP_linearize
                theta_MAP_stage = theta_MAP_linearize.copy()
                
                # Construct full 14-dim theta using linearization MAP (for ROM/linearization operations)
                # ★ NOTE: This is used for linearization point update, not for reporting
                theta_full_MAP = theta_base_full.copy()
                for i, idx in enumerate(active_indices):
                    theta_full_MAP[idx] = theta_MAP_linearize[i]

                rom_error_pre_from_enable_check: Optional[float] = None
                
                # (Stability gate) Try enabling linearization only after a MAP-based ROM error check.
                # This prevents enabling linearization in regions where ε_ROM is still large.
                if (beta_next >= float(linearization_threshold)) and (not evaluator._linearization_enabled):
                    enabled_ok = False
                    try:
                        evaluator.enable_linearization(True)
                        rom_err_try = evaluator.compute_ROM_error(theta_full_MAP)
                        if np.isfinite(rom_err_try) and (rom_err_try <= float(linearization_enable_rom_threshold)):
                            enabled_ok = True
                            rom_error_pre_from_enable_check = float(rom_err_try)
                            debug_logger.log_info(
                                f"✅ Linearization enabled at β={beta_next:.4f} (threshold={float(linearization_threshold):.3f}) with ε_ROM(MAP)={float(rom_err_try):.6f} <= {float(linearization_enable_rom_threshold):.6f}"
                            )
                        else:
                            debug_logger.log_warning(
                                f"Keeping linearization OFF (unstable): ε_ROM(MAP)={float(rom_err_try):.6f} > {float(linearization_enable_rom_threshold):.6f} (β={beta_next:.4f}, threshold={float(linearization_threshold):.3f})"
                            )
                    except Exception as e:
                        debug_logger.log_warning(f"Linearization enable check failed: {e}. Keeping linearization OFF.")
                    finally:
                        if not enabled_ok:
                            try:
                                evaluator.enable_linearization(False)
                            except Exception:
                                pass

                # Also compute weighted mean for comparison/reporting (optional)
                theta_weighted_mean = np.zeros(n_params)
                for i in range(n_particles):
                    theta_weighted_mean += weights_before_resample[i] * theta_before_resample[i]
                theta_full_weighted_mean = theta_base_full.copy()
                for i, idx in enumerate(active_indices):
                    theta_full_weighted_mean[idx] = theta_weighted_mean[i]
                
                # Get current linearization point
                theta0_old = evaluator.get_linearization_point()
                
                # ★ 改善: 線形化点更新判定にROM誤差を追加（論文で映えるstopping criterion）
                # Check 1: θ空間での変化（MAPベース）
                delta_theta0 = None
                if theta0_old is not None:
                    delta_theta0 = np.linalg.norm(theta_full_MAP - theta0_old)
                    delta_theta0_stage = float(delta_theta0)
                    if delta_theta0 < THETA_CONVERGENCE_THRESHOLD:
                        should_update = False
                        debug_logger.log_warning(f"Linearization point converged (||Δθ₀||={delta_theta0:.6f} < {THETA_CONVERGENCE_THRESHOLD})")
                
                # Check 2: ROM誤差（論文でreviewerが大好きなstopping criterion）
                # Error in observable space: || φ̄_ROM(t_obs) − φ̄_FOM(t_obs) ||₂ / || φ̄_FOM(t_obs) ||₂
                rom_error_pre = None
                if should_update:
                    # ★ 優先度S: FOMチェックのスキップ条件を考慮
                    if use_observation_based_update and not should_do_fom:
                        # Skip FOM evaluation if conditions are met (use last known error)
                        rom_error_pre = last_rom_error
                        # Keep history aligned even when skipping FOM.
                        rom_error_pre_history.append(np.nan)
                    else:
                        # Use MAP for ROM error check
                        if rom_error_pre_from_enable_check is not None:
                            rom_error_pre = float(rom_error_pre_from_enable_check)
                        else:
                            rom_error_pre = evaluator.compute_ROM_error(theta_full_MAP)
                        rom_error_pre_stage = None if rom_error_pre is None else float(rom_error_pre)
                        
                        if rom_error_pre is not None:
                            # ★ ERROR-CHECK: Check ROM error explosion
                            # Use previous stage's acceptance rate (if available) to skip check when acc_rate is very low
                            prev_acc_rate = acc_rate_history[-1] if len(acc_rate_history) > 0 else None
                            debug_logger.check_rom_error_explosion(
                                rom_error_pre,
                                context=f"Stage {stage}, linearization pre-update",
                                acc_rate=prev_acc_rate,
                            )

                            # Record pre-update ROM error (debugging only)
                            rom_error_pre_history.append(rom_error_pre)
                            
                            if rom_error_pre < ROM_ERROR_THRESHOLD:
                                should_update = False
                                debug_logger.log_warning(
                                    f"ROM error sufficiently small (ε_ROM={rom_error_pre:.6f} < {ROM_ERROR_THRESHOLD})"
                                )
                                debug_logger.log_info("   where ε_ROM = || φ̄_ROM(t_obs) − φ̄_FOM(t_obs) ||₂ / || φ̄_FOM(t_obs) ||₂")
                            else:
                                debug_logger.log_rom_error(stage, rom_error_pre, ROM_ERROR_THRESHOLD)
                
                # Update linearization point if needed (use MAP, not weighted mean)
                if should_update and n_linearization_updates < MAX_LINEARIZATION_UPDATES:
                    # Stabilize θ0 updates:
                    # - Cap per-update step size (MAX_THETA0_STEP_NORM)
                    # - Allow a few sub-updates per event (MAX_LINEARIZATION_SUBUPDATES_PER_EVENT)
                    theta0_curr = evaluator.get_linearization_point()
                    theta0_start = theta0_curr.copy()

                    for _sub in range(MAX_LINEARIZATION_SUBUPDATES_PER_EVENT):
                        if n_linearization_updates >= MAX_LINEARIZATION_UPDATES:
                            break

                        delta_vec = theta_full_MAP - theta0_curr
                        delta_norm = float(np.linalg.norm(delta_vec))
                        if not np.isfinite(delta_norm) or delta_norm <= 1e-12:
                            break

                        alpha = 1.0 if delta_norm <= MAX_THETA0_STEP_NORM else (MAX_THETA0_STEP_NORM / delta_norm)
                        theta0_next = theta0_curr + alpha * delta_vec

                        # Apply update
                        evaluator.update_linearization_point(theta0_next)
                        n_linearization_updates += 1
                        theta0_history.append(theta0_next.copy())

                        # Report both MAP and weighted mean for comparison
                        delta_weighted_mean_map = np.linalg.norm(theta_weighted_mean - theta_MAP_stage)
                        debug_logger.log_info(f"MAP-WeightedMean distance: {delta_weighted_mean_map:.6f}")

                        # Track last delta_theta0 for skip conditions (use actual step)
                        last_delta_theta0 = float(np.linalg.norm(theta0_next - theta0_curr))

                        # Log update (actual step size)
                        debug_logger.log_linearization_update(
                            stage=stage,
                            beta=beta_next,
                            update_num=n_linearization_updates,
                            theta0_old=theta0_curr,
                            theta0_new=theta0_next,
                            delta_norm=last_delta_theta0,
                        )

                        # Recompute logL for all particles with new linearization point
                        debug_logger.log_info("Recomputing logL with new linearization point...")
                        logL_prev = logL.copy()
                        logL_new = evaluate_particles_parallel(log_likelihood, theta, n_jobs=n_jobs, use_threads=use_threads)
                        logL = logL_new
                        debug_logger.log_info(f"✅ LogL recomputed: min={logL.min():.1f}, max={logL.max():.1f}")
                        # Guardrail: detect suspicious likelihood scale jumps after linearization updates.
                        # This often indicates that the evaluator regime (ROM/linearized ROM/FOM) or variance model changed
                        # dramatically, which can invalidate tempering assumptions.
                        try:
                            if np.all(np.isfinite(logL_prev)) and np.all(np.isfinite(logL)):
                                prev_range = float(np.max(logL_prev) - np.min(logL_prev))
                                new_range = float(np.max(logL) - np.min(logL))
                                prev_med = float(np.median(logL_prev))
                                new_med = float(np.median(logL))
                                med_shift = float(abs(new_med - prev_med))
                                lin_enabled = bool(getattr(evaluator, "_linearization_enabled", False))
                                # Heuristic: median shift far larger than prior range or absolute huge jump
                                if (prev_range > 0 and med_shift > 50.0 * prev_range) or (med_shift > 1e3):
                                    debug_logger.log_warning(
                                        "Suspicious logL scale jump after θ₀ update: "
                                        f"median {prev_med:.2f}→{new_med:.2f} (|Δ|={med_shift:.2e}), "
                                        f"range {prev_range:.2f}→{new_range:.2f}, "
                                        f"linearization_enabled={lin_enabled}."
                                    )
                        except Exception:
                            # Never fail TMCMC because of diagnostics
                            pass

                        # Post-update ROM error (this is what we report/gate on)
                        rom_error_post = None
                        if use_observation_based_update and not should_do_fom:
                            rom_error_history.append(np.nan)
                        else:
                            rom_error_post = evaluator.compute_ROM_error(theta_full_MAP)
                            if rom_error_post is not None:
                                rom_error_history.append(rom_error_post)
                                last_rom_error = rom_error_post
                                rom_error_post_stage = float(rom_error_post)
                                debug_logger.log_info(
                                    f"[TMCMC] ROM error (post-update): {rom_error_post:.6f} (threshold: {ROM_ERROR_THRESHOLD})"
                                )

                        theta0_curr = theta0_next

                        # Stop further sub-updates if ROM error is now sufficiently small.
                        if rom_error_post is not None and rom_error_post < ROM_ERROR_THRESHOLD:
                            break
                elif n_linearization_updates >= MAX_LINEARIZATION_UPDATES:
                    debug_logger.log_warning(f"Reached max linearization updates ({MAX_LINEARIZATION_UPDATES}), stopping updates")
        
        # ★ ERROR-CHECK: Check acceptance rate (post-recovery)
        debug_logger.check_acceptance_rate(acc_rate, context=f"Stage {stage}")
        
        # ★ Log beta progress and acceptance rate
        debug_logger.log_beta_progress(stage, beta_next, delta_beta)
        debug_logger.log_acceptance_rate(stage, acc_rate, acc, total_proposals)
        
        # ★ ERROR-CHECK: Check numerical errors after mutation
        debug_logger.check_numerical_errors(logL, theta, context=f"Stage {stage}, after mutation")
        
        # ★ Record acceptance rate history
        acc_rate_history.append(acc_rate)

        # Diagnostics: likelihood/TSM health counters (high-signal when accuracy stagnates)
        try:
            if evaluator is not None and debug_logger.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                h = evaluator.get_health()
                # Only log if something looks off (keeps noise down)
                key_stats = {
                    "n_calls": int(h.get("n_calls", 0)),
                    "n_tsm_fail": int(h.get("n_tsm_fail", 0)),
                    "n_output_nonfinite": int(h.get("n_output_nonfinite", 0)),
                    "n_var_raw_negative": int(h.get("n_var_raw_negative", 0)),
                    "n_var_raw_nonfinite": int(h.get("n_var_raw_nonfinite", 0)),
                    "n_var_total_clipped": int(h.get("n_var_total_clipped", 0)),
                }
                if any(v > 0 for k, v in key_stats.items() if k != "n_calls"):
                    debug_logger.log_warning(f"Likelihood health (cumulative): {key_stats}")
                else:
                    debug_logger.log_info(f"Likelihood health (cumulative): {key_stats}")
        except Exception:
            pass

        # Record stage summary for offline debugging (CSV export)
        stage_summary.append(
            {
                "stage": int(stage),
                "beta": float(beta),
                "beta_next": float(beta_next),
                "delta_beta": float(delta_beta),
                "ess": float(ess_at_delta_low) if ess_at_delta_low is not None else None,
                "ess_target": float(target_ess_ratio * n_particles),
                "acc_rate": float(acc_rate),
                "logL_min": float(np.min(logL)) if len(logL) > 0 else None,
                "logL_max": float(np.max(logL)) if len(logL) > 0 else None,
                "linearization_enabled": int(bool(getattr(evaluator, "_linearization_enabled", False))) if evaluator is not None else 0,
                "rom_error_pre": rom_error_pre_stage,
                "rom_error_post": rom_error_post_stage,
                "delta_theta0": delta_theta0_stage,
            }
        )
        
        beta = beta_next
        beta_schedule.append(beta)
        
        if beta >= BETA_CONVERGENCE_THRESHOLD:
            debug_logger.log_info("✓ Converged! β reached 1.0", force=True)
            # ★ Slack notification: Convergence (add to thread if available)
            if SLACK_ENABLED and model_name:
                conv_msg = f"✅ Converged! Stage: {stage}/{n_stages}, β = {beta:.4f}"
                if slack_thread_ts:
                    _slack_notifier.add_to_thread(slack_thread_ts, conv_msg)
                else:
                    notify_slack(
                        f"✅ {model_name} TMCMC Converged\n"
                        f"   Stage: {stage}/{n_stages}\n"
                        f"   β = {beta:.4f} (reached {BETA_CONVERGENCE_THRESHOLD})",
                        raise_on_error=False
                    )
            break
    
    # ★ CRITICAL FIX: Extract final MAP
    # Priority: Use posterior MAP from last stage if available (preserves observation correction)
    # Fallback: Use standard MAP from final logL (if no observation-based update was used)
    # 
    # ★ ASSERT: Verify consistency of MAP history
    # The number of MAP records should match the number of stages where should_update=True
    # (This is approximate since should_update depends on conditions)
    if len(theta_MAP_posterior_history) > 0:
        # Use posterior MAP from last stage (preserves observation information)
        # Note: This is the MAP from the last stage where should_update=True
        theta_MAP = theta_MAP_posterior_history[-1].copy()
        map_source = "posterior (observation-corrected)"
    else:
        # Fallback: standard MAP computation (no observation-based update was used, or evaluator was None)
        # This happens when:
        # - evaluator is None (no linearization update)
        # - use_observation_based_update=False
        # - should_update was never True (e.g., β never reached threshold)
        # Compute tempered posterior for final particles
        log_prior_final = np.array([log_prior(t) for t in theta])
        log_posterior_final = log_prior_final + beta * logL
        idx_MAP = np.argmax(log_posterior_final)
        theta_MAP = theta[idx_MAP]
        map_source = "standard (from final posterior)"
    
    debug_logger.log_info(f"✅ TMCMC complete! Final β={beta:.4f}", force=True)
    debug_logger.log_info(f"🎯 MAP ({map_source}): {theta_MAP}", force=True)
    
    # ★ Slack notification: TMCMC complete (add to thread if available)
    if SLACK_ENABLED and model_name:
        complete_msg = (
            f"✅ TMCMC Complete\n"
            f"   Final β: {beta:.4f}\n"
            f"   Converged: {beta >= BETA_CONVERGENCE_THRESHOLD}\n"
            f"   Stages: {len(beta_schedule)}\n"
            f"   MAP ({map_source}): {theta_MAP}"
        )
        if slack_thread_ts:
            _slack_notifier.add_to_thread(slack_thread_ts, complete_msg)
        else:
            notify_slack(
                f"✅ {model_name} TMCMC Complete\n"
                f"   Final β: {beta:.4f}\n"
                f"   Converged: {beta >= BETA_CONVERGENCE_THRESHOLD}\n"
                f"   Stages: {len(beta_schedule)}",
                raise_on_error=False
            )
        
        # ★ ASSERT: Final MAP should match the last recorded posterior MAP (only if history exists)
        if len(theta_MAP_posterior_history) > 0:
            assert np.allclose(theta_MAP, theta_MAP_posterior_history[-1]), \
                "Final MAP should match last recorded posterior MAP"
    
    # Compute final MAP for global sharing (if multiple chains)
    # Note: theta0_history now contains MAP values
    final_MAP = None
    if len(theta0_history) > 0:
        # Use the last updated MAP
        final_MAP = theta0_history[-1].copy()
    elif evaluator is not None:
        # If no updates, use current linearization point
        final_MAP = evaluator.get_linearization_point()
    
    # ★ Calculate evaluation counts
    n_rom_evaluations = 0
    n_fom_evaluations = 0
    if evaluator is not None:
        n_rom_evaluations = evaluator.call_count - initial_rom_count
        n_fom_evaluations = evaluator.fom_call_count - initial_fom_count

    wall_time_s = float(time.perf_counter() - tmcmc_wall_start)
    timing_breakdown_s: Optional[Dict[str, float]] = None
    if evaluator is not None and hasattr(evaluator, "timing") and isinstance(getattr(evaluator, "timing"), TimingStats):
        tsm_s = float(evaluator.timing.get_s("tsm.solve_tsm"))
        fom_s = float(evaluator.timing.get_s("fom.run_deterministic"))
        tmcmc_overhead_s = float(max(0.0, wall_time_s - tsm_s - fom_s))
        timing_breakdown_s = {
            "tmcmc_total_s": wall_time_s,
            "tsm_s": tsm_s,
            "fom_s": fom_s,
            "tmcmc_overhead_s": tmcmc_overhead_s,
        }

    likelihood_health: Optional[Dict[str, int]] = None
    if evaluator is not None and hasattr(evaluator, "get_health"):
        try:
            likelihood_health = evaluator.get_health()  # type: ignore[assignment]
        except Exception:
            likelihood_health = None
    
    return TMCMCResult(
        samples=theta,
        logL_values=logL,
        theta_MAP=theta_MAP,
        beta_schedule=beta_schedule,
        converged=(beta >= BETA_CONVERGENCE_THRESHOLD),
        theta0_history=theta0_history if theta0_history else None,
        n_linearization_updates=n_linearization_updates,
        final_MAP=final_MAP,  # ★ For global chain sharing
        rom_error_pre_history=rom_error_pre_history if rom_error_pre_history else None,
        rom_error_history=rom_error_history if rom_error_history else None,  # ★ ROM error history
        acc_rate_history=acc_rate_history if acc_rate_history else None,  # ★ Acceptance rate history
        n_rom_evaluations=n_rom_evaluations,  # ★ Number of ROM evaluations
        n_fom_evaluations=n_fom_evaluations,  # ★ Number of FOM evaluations
        wall_time_s=wall_time_s,
        timing_breakdown_s=timing_breakdown_s,
        likelihood_health=likelihood_health,
        stage_summary=stage_summary if stage_summary else None,
    )


def run_multi_chain_MCMC(
    model_tag: str,
    evaluator_factory: callable,
    prior_bounds: List[Tuple[float, float]],
    mcmc_config: Any,  # MCMCConfig (imported locally to avoid circular dependency)
    proposal_cov: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Run multiple MCMC chains sequentially with diagnostics."""
    # Import locally to avoid circular dependency
    from case2_tmcmc_linearization import run_adaptive_MCMC
    
    logger.info("[%s] Running %s MCMC chains...", model_tag, mcmc_config.n_chains)
    
    all_samples = []
    all_logL = []
    all_MAP = []
    all_acc = []
    
    for chain_idx in range(mcmc_config.n_chains):
        seed = mcmc_config.n_chains * 1000 + chain_idx
        logger.info("Chain %s/%s", chain_idx + 1, mcmc_config.n_chains)
        
        evaluator = evaluator_factory()
        
        samples, logL, MAP, acc = run_adaptive_MCMC(
            log_likelihood=evaluator,
            prior_bounds=prior_bounds,
            n_samples=mcmc_config.n_samples,
            initial_scale=mcmc_config.initial_scale,
            burn_in=mcmc_config.n_burn_in,
            target_accept=mcmc_config.target_accept,
            adapt_start=mcmc_config.adapt_start,
            adapt_interval=mcmc_config.adapt_interval,
            proposal_cov=proposal_cov,
            seed=seed,
        )
        
        all_samples.append(samples)
        all_logL.append(logL)
        all_MAP.append(MAP)
        all_acc.append(acc)
    
    diag = MCMCDiagnostics(all_samples, [f"θ{i}" for i in range(len(prior_bounds))])
    diag.compute_all()
    
    best_logL = -np.inf
    best_theta = None
    for c, logL in enumerate(all_logL):
        idx = np.argmax(logL)
        if logL[idx] > best_logL:
            best_logL = logL[idx]
            best_theta = all_samples[c][idx]
    
    diagnostics = {
        "Rhat": diag.Rhat,
        "ESS": diag.ESS,
        "acc_rate_mean": float(np.mean(all_acc)),
        "MAP_global": best_theta,
        "MAP_logL": best_logL,
    }
    
    logger.info("[%s] Summary:", model_tag)
    logger.info("Mean acceptance rate: %.1f%%", diagnostics["acc_rate_mean"] * 100.0)
    logger.info("R-hat: %s", diag.Rhat)
    logger.info("ESS: %s", diag.ESS)
    logger.info("Global MAP: %s", best_theta)
    
    return all_samples, all_logL, best_theta, diagnostics


# Helper function for stable hashing (used in run_multi_chain_TMCMC)
def _stable_hash_int(text: str) -> int:
    """Stable, cross-run integer hash (unlike Python's built-in hash())."""
    import zlib
    return int(zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF)


def run_multi_chain_TMCMC(
    model_tag: str,
    make_evaluator: callable,  # ★ Should accept theta_linearization argument
    prior_bounds: List[Tuple[float, float]],
    theta_base_full: np.ndarray,  # ★ Full 14-dim theta base
    active_indices: List[int],  # ★ Active parameter indices
    theta_linearization_init: Optional[np.ndarray] = None,  # ★ Initial linearization point
    n_particles: int = 2000,
    n_stages: int = 30,
    target_ess_ratio: float = 0.5,
    min_delta_beta: float = 0.05,
    max_delta_beta: float = 0.2,
    logL_scale: float = 1.0,
    n_chains: int = 1,
    update_linearization_interval: int = 3,  # ★ Update every N stages
    n_mutation_steps: int = 5,  # ★ Number of MCMC steps per particle (K-step mutation)
    use_observation_based_update: bool = True,  # ★ Use observation-based linearization update (ROM error weighted)
    linearization_threshold: float = DEFAULT_LINEARIZATION_THRESHOLD,
    linearization_enable_rom_threshold: float = 0.05,
    debug_config: Optional[DebugConfig] = None,  # ★ Debug configuration
    seed: Optional[int] = None,  # ★ Base seed for reproducibility across runs
    force_beta_one: bool = False,  # ★ If True, force β=1.0 at final stage (paper runs)
    n_jobs: Optional[int] = None,  # ★ Number of parallel jobs for particle evaluation (None = auto, 1 = sequential)
    use_threads: bool = False,  # ★ Use threads instead of processes (for GIL-friendly code)
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, List[bool], Dict]:
    """
    Run multiple TMCMC chains sequentially with diagnostics + Linearization Update.
    
    ★ TMCMC版のマルチチェーン実行関数（β tempering + 線形化点更新）
    
    ★ 重要な改善点：
    1. make_evaluator(theta_linearization) で線形化点を受け取る
    2. 各stageでMAPを計算し、線形化点を更新
    3. 「TMCMC × 線形化更新」という論文の核を実現
    
    Parameters
    ----------
    model_tag : str
        Model identifier
    make_evaluator : callable
        Function that returns LogLikelihoodEvaluator instance
    prior_bounds : List[Tuple[float, float]]
        Prior bounds for each parameter
    n_particles : int
        Number of particles per chain
    n_stages : int
        Maximum number of TMCMC stages
    target_ess_ratio : float
        Target ESS ratio (0.5 = 50% ESS)
    min_delta_beta : float
        Minimum β increment per stage (progress floor).
    max_delta_beta : float
        Maximum β increment per stage (jump cap).
    logL_scale : float
        [DEPRECATED] Likelihood scaling factor. Currently ignored for consistency with TMCMC theory.
        All likelihood calculations (ESS, resampling, mutation) now use unscaled logL.
    n_chains : int
        Number of chains to run
        
    Returns
    -------
    all_samples : List[np.ndarray]
        Samples from each chain
    all_logL : List[np.ndarray]
        Log-likelihood values from each chain
    global_MAP : np.ndarray
        Global MAP across all chains
    converged_flags : List[bool]
        Convergence status for each chain
    diagnostics : Dict
        MCMC diagnostics
    """
    # Import locally to avoid circular dependency
    from config import DebugConfig, DebugLevel
    
    logger.info("[%s] Running %s TMCMC chains (β tempering + linearization update)...", model_tag, n_chains)
    
    # Initialize debug logger
    if debug_config is None:
        debug_config = DebugConfig(level=DebugLevel.OFF)
    debug_logger = DebugLogger(debug_config)
    
    # Initialize linearization point
    if theta_linearization_init is None:
        theta_linearization_init = theta_base_full.copy()
    
    all_samples = []
    all_logL = []
    all_MAP = []
    converged_flags = []
    all_beta_schedules = []
    all_theta0_histories = []  # ★ Track linearization point update history
    total_linearization_updates = 0  # ★ Track total updates
    all_MAPs = []  # ★ Collect MAPs from each chain for global sharing
    all_results = []  # ★ Store all TMCMCResult objects for diagnostics
    
    # ★ Global MAP for chain sharing (improves accuracy by using best estimate from each chain)
    global_MAP = theta_linearization_init.copy()
    
    for chain_idx in range(n_chains):
        # ★ PRIORITY A: seed に model_tag を含める（M1/M2同一挙動の切り分け）
        # 以前: seed = n_chains * 1000 + chain_idx  → M1/M2で同じseedになる可能性
        # 修正: model_tag を含めてモデルごとに異なるseedを生成
        base = int(seed or 0)
        seed_base = _stable_hash_int(model_tag) % (2**31)  # stable hash across runs
        chain_seed = (base + seed_base + n_chains * 1000 + chain_idx) % (2**31)
        logger.info("Chain %s/%s", chain_idx + 1, n_chains)
        logger.debug("seed: %s (model_tag: %s, chain: %s)", chain_seed, model_tag, chain_idx)
        
        # ★ Slack notification: 削除（詳細すぎるため、重要な情報のみ送信）
        
        # ★ Create evaluator with linearization point
        # Chain 0: use initial point
        # Chain 1+: use global MAP from previous chains (if available)
        if chain_idx == 0:
            # First chain: use initial linearization point
            current_linearization = theta_linearization_init.copy()
        else:
            # Subsequent chains: use global MAP from previous chains
            # ★ 改善④: Global MAP sharing across chains
            # This improves accuracy by leveraging the best estimate from all chains
            current_linearization = global_MAP.copy()
            logger.info("[Chain %s] Using global MAP from previous chains", chain_idx + 1)
        
        evaluator = make_evaluator(theta_linearization=current_linearization)
        # ★ Pass debug_logger to evaluator for silent error handling in ERROR/OFF mode
        if hasattr(evaluator, 'debug_logger') or hasattr(evaluator, '__dict__'):
            evaluator.debug_logger = debug_logger
        logger.info("[Chain %s] Initial linearization point set", chain_idx + 1)
        
        result = run_TMCMC(
            log_likelihood=evaluator,
            prior_bounds=prior_bounds,
            n_particles=n_particles,
            n_stages=n_stages,
            target_ess_ratio=target_ess_ratio,
            min_delta_beta=min_delta_beta,
            max_delta_beta=max_delta_beta,
            logL_scale=logL_scale,
            seed=chain_seed,
            model_name=f"{model_tag}_chain{chain_idx+1}",
            n_jobs=n_jobs,
            use_threads=use_threads,
            evaluator=evaluator,  # ★ Pass evaluator for linearization update
            theta_base_full=theta_base_full,  # ★ Pass full theta base
            active_indices=active_indices,  # ★ Pass active indices
            update_linearization_interval=update_linearization_interval,  # ★ Update interval
            n_mutation_steps=n_mutation_steps,  # ★ K-step mutation
            use_observation_based_update=use_observation_based_update,  # ★ Observation-based update (ROM error weighted)
            linearization_threshold=linearization_threshold,
            linearization_enable_rom_threshold=linearization_enable_rom_threshold,
            debug_logger=debug_logger,  # ★ Pass debug logger
            force_beta_one=force_beta_one,
        )
        
        all_samples.append(result.samples)
        all_logL.append(result.logL_values)
        all_MAP.append(result.theta_MAP)
        converged_flags.append(result.converged)
        all_beta_schedules.append(result.beta_schedule)
        all_results.append(result)  # ★ Store result for diagnostics
        
        # ★ Track linearization point update history
        if result.theta0_history is not None:
            all_theta0_histories.append(result.theta0_history)
        total_linearization_updates += result.n_linearization_updates
        
        # ★ Collect MAP from this chain for global sharing
        if result.final_MAP is not None:
            all_MAPs.append(result.final_MAP.copy())
            
            # Update global MAP: use the MAP with highest log-likelihood across all chains
            # This provides the best estimate from all chains
            if len(all_MAPs) > 0:
                # Use the MAP from the chain with highest log-likelihood
                # (already computed in global_MAP calculation below)
                # For now, use the latest MAP (can be improved to select best)
                global_MAP = all_MAPs[-1].copy()
                logger.info("[Chain %s] Global MAP updated from %s chains", chain_idx + 1, len(all_MAPs))
    
    # Global MAP (highest log-likelihood across all chains)
    # Find the chain and sample index with the highest log-likelihood
    best_logL = -np.inf
    best_chain_idx = 0
    best_sample_idx = 0
    for chain_idx, logL_chain in enumerate(all_logL):
        sample_idx = np.argmax(logL_chain)
        if logL_chain[sample_idx] > best_logL:
            best_logL = logL_chain[sample_idx]
            best_chain_idx = chain_idx
            best_sample_idx = sample_idx
    global_MAP = all_samples[best_chain_idx][best_sample_idx]
    
    # Diagnostics
    # ⚠️ 重要: TMCMCはMarkov chainではないため、R-hat/ESSは理論的に正当化されない
    # TMCMC uses resampling, particle duplication, and tempered likelihood,
    # which violate the Markov chain assumptions required for R-hat/ESS.
    # We compute them only as reference indicators, NOT for convergence judgment.
    from mcmc_diagnostics import MCMCDiagnostics
    # ★ 修正: MCMCDiagnosticsはチェーンのリスト（List[np.ndarray]）を想定している
    # all_samples_flat（2次元配列）を渡すと、chains[0]が1次元配列になってIndexErrorが発生
    param_names = [f"θ{i}" for i in range(len(prior_bounds))]
    diag = MCMCDiagnostics(all_samples, param_names)  # ← List of chains を渡す
    diag.compute_all()
    
    diagnostics = {
        # ⚠️ Reference only: R-hat/ESS are NOT theoretically valid for TMCMC
        # See: Del Moral et al. (2006), Ching & Chen (2007) for SMC/TMCMC theory
        "Rhat_reference": diag.Rhat,  # ★ Reference indicator only (not for convergence)
        "ESS_reference": diag.ESS,     # ★ Reference indicator only (not for convergence)
        "converged_chains": sum(converged_flags),
        "total_chains": n_chains,
        "MAP_global": global_MAP,
        "beta_schedules": all_beta_schedules,
        "theta0_history": all_theta0_histories,  # ★ Linearization point update history
        "total_linearization_updates": total_linearization_updates,  # ★ Total number of updates
        "rom_error_pre_histories": [r.rom_error_pre_history for r in all_results if r.rom_error_pre_history is not None],  # ★ pre-update ROM errors
        "rom_error_histories": [r.rom_error_history for r in all_results if r.rom_error_history is not None],  # ★ ROM error histories
        "acc_rate_histories": [r.acc_rate_history for r in all_results if r.acc_rate_history is not None],  # ★ Acceptance rate histories
        "n_rom_evaluations": [r.n_rom_evaluations for r in all_results],  # ★ ROM evaluation counts per chain
        "n_fom_evaluations": [r.n_fom_evaluations for r in all_results],  # ★ FOM evaluation counts per chain
        "tmcmc_wall_time_s": [float(r.wall_time_s) for r in all_results],  # ★ Wall time per chain
        "timing_breakdown_s": [r.timing_breakdown_s for r in all_results],  # ★ Per-chain breakdown (tsm/fom/overhead)
        "likelihood_health_histories": [r.likelihood_health for r in all_results if r.likelihood_health is not None],
        "stage_summaries": [r.stage_summary for r in all_results if r.stage_summary is not None],
        "note": "R-hat/ESS are computed for reference only. TMCMC convergence is judged by β=1.0 and chain consistency.",
    }

    # Aggregate likelihood health across chains (for quick checks / metrics.json)
    health_total: Dict[str, int] = {}
    for h in diagnostics.get("likelihood_health_histories", []):
        if not isinstance(h, dict):
            continue
        for k, v in h.items():
            try:
                health_total[k] = int(health_total.get(k, 0) + int(v))
            except Exception:
                continue
    if health_total:
        diagnostics["likelihood_health_total"] = health_total
    
    logger.info("[%s] TMCMC Summary:", model_tag)
    logger.info("Converged chains: %s/%s", sum(converged_flags), n_chains)
    logger.info("Global MAP: %s", global_MAP)
    
    # ★ Slack notification: All chains complete
    if SLACK_ENABLED:
        converged_count = sum(converged_flags)
        notify_slack(
            f"✅ {model_tag} All {n_chains} chains completed\n"
            f"   Converged: {converged_count}/{n_chains}\n"
            f"   Total linearization updates: {total_linearization_updates}"
        )
    
    return all_samples, all_logL, global_MAP, converged_flags, diagnostics


