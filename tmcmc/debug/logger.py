"""
Debug logger with hook-based control and Slack notification support.

Extracted from case2_tmcmc_linearization.py for better modularity.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Import config from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DebugConfig, DebugLevel

# ==============================================================================
# Slack notification support
# ==============================================================================

# Try to import from relative path (stranger/d.py)
try:
    stranger_path = Path(__file__).parent.parent.parent / "stranger"
    if stranger_path.exists():
        sys.path.insert(0, str(stranger_path))
        from d import notify_slack, SlackNotifier  # type: ignore
        # Enabled only when credentials are provided via environment variables.
        # - Webhook: SLACK_WEBHOOK_URL
        # - Bot: SLACK_BOT_TOKEN (+ SLACK_CHANNEL, depending on stranger/d.py)
        SLACK_ENABLED = bool(os.getenv("SLACK_WEBHOOK_URL") or os.getenv("SLACK_BOT_TOKEN"))
        # Initialize global SlackNotifier for thread support
        # Falls back to webhook if SLACK_BOT_TOKEN/SLACK_CHANNEL not set
        _slack_notifier = SlackNotifier(raise_on_error=False)
    else:
        # Fallback: define a no-op function if path doesn't exist
        def notify_slack(message: str, **kwargs) -> bool:  # type: ignore
            return False
        class SlackNotifier:  # type: ignore
            def start_thread(self, title: str) -> None:
                return None
            def add_to_thread(self, thread_ts: None, message: str) -> bool:
                return False
        _slack_notifier = SlackNotifier()
        SLACK_ENABLED = False
except (ImportError, ModuleNotFoundError):
    # Fallback: define a no-op function if import fails
    def notify_slack(message: str, **kwargs) -> bool:  # type: ignore
        return False
    class SlackNotifier:  # type: ignore
        def start_thread(self, title: str) -> None:
            return None
        def add_to_thread(self, thread_ts: None, message: str) -> bool:
            return False
    _slack_notifier = SlackNotifier()
    SLACK_ENABLED = False


# ==============================================================================
# DebugLogger
# ==============================================================================


class DebugLogger:
    """
    Debug logger with hook-based control.
    
    Design principles:
    - No performance impact when debug is OFF
    - Configurable via DebugConfig
    - Hook-based for flexibility
    - ERROR mode: Silent error detection (no print, raise exceptions)
    
    Examples
    --------
    >>> from config import DebugConfig, DebugLevel
    >>> config = DebugConfig(level=DebugLevel.VERBOSE)
    >>> logger = DebugLogger(config)
    >>> logger.log_beta_progress(stage=1, beta=0.1, delta_beta=0.05)
    """
    
    def __init__(self, config: DebugConfig, slack_thread_ts: Optional[str] = None):
        """
        Initialize debug logger.
        
        Parameters
        ----------
        config : DebugConfig
            Debug configuration
        slack_thread_ts : str, optional
            Slack thread timestamp for organized notifications
        """
        self.config = config
        self.hooks: Dict[str, List[Callable]] = {}
        self.slack_thread_ts = slack_thread_ts  # Thread timestamp for Slack notifications
        self._log = logging.getLogger(__name__)
        self._events_jsonl_path: Optional[Path] = None

    def set_events_jsonl(self, path: Optional[Path]) -> None:
        """
        Persist debug events as JSON Lines (one JSON object per line).
        
        This is intentionally separate from stdout/stderr so logs remain human-readable,
        while structured data becomes easy to aggregate.
        
        Parameters
        ----------
        path : Path, optional
            Path to JSONL file for event logging
        """
        self._events_jsonl_path = path

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Best-effort conversion for numpy/scalars/arrays."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
        except Exception:
            pass
        return obj
    
    def set_slack_thread(self, thread_ts: Optional[str]) -> None:
        """Set Slack thread timestamp for organized notifications."""
        self.slack_thread_ts = thread_ts
    
    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for a specific debug event."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)
    
    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit debug event to registered hooks."""
        if event in self.hooks:
            for callback in self.hooks[event]:
                callback(*args, **kwargs)

        # Optional: write structured events to events.jsonl (append mode).
        if self._events_jsonl_path is not None:
            try:
                payload = {
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "event": event,
                    **{k: self._json_safe(v) for k, v in kwargs.items()},
                }
                self._events_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self._events_jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            except Exception:
                # Never let event serialization break the run.
                pass
    
    def log_beta_progress(self, stage: int, beta: float, delta_beta: float) -> None:
        """Log β schedule progression."""
        if self.config.show_beta_progress:
            self._emit("beta_progress", stage=stage, beta=beta, delta_beta=delta_beta)
            msg = f"      [TMCMC] Stage {stage}: β={beta:.4f} (+{delta_beta:.4f})"
            self._log.info("%s", msg)
    
    def log_linearization_update(
        self, 
        stage: int, 
        beta: float, 
        update_num: int,
        theta0_old: Optional[np.ndarray],
        theta0_new: np.ndarray,
        delta_norm: float,
    ) -> None:
        """Log linearization point update."""
        if self.config.show_linearization_updates:
            self._emit(
                "linearization_update",
                stage=stage,
                beta=beta,
                update_num=update_num,
                theta0_old=theta0_old,
                theta0_new=theta0_new,
                delta_norm=delta_norm,
            )
            self._log.info(
                "      [TMCMC] Updated linearization point (stage %s, β=%.4f, update #%s)",
                stage,
                beta,
                update_num,
            )
            self._log.info("      [TMCMC] ||Δθ₀|| = %.6f", delta_norm)
    
    def log_rom_error(self, stage: int, rom_error: float, threshold: float) -> None:
        """Log ROM error."""
        if self.config.show_rom_errors:
            self._emit("rom_error", stage=stage, rom_error=rom_error, threshold=threshold)
            self._log.info("      [TMCMC] ROM error: %.6f (threshold: %s)", rom_error, threshold)
    
    def log_acceptance_rate(self, stage: int, acc_rate: float, n_accepted: int, n_total: int) -> None:
        """Log acceptance rate."""
        if self.config.show_acceptance_rates:
            self._emit("acceptance_rate", stage=stage, acc_rate=acc_rate, n_accepted=n_accepted, n_total=n_total)
            self._log.info(
                "      [TMCMC] Stage %s: Acc=%.2f (%s/%s proposals)",
                stage,
                acc_rate,
                n_accepted,
                n_total,
            )
            # Slack notification: Acceptance rate (only if low, to avoid spam)
            if SLACK_ENABLED and acc_rate < 0.1:
                acc_msg = f"⚠️ Low acceptance rate: {acc_rate:.2f} ({n_accepted}/{n_total}), Stage: {stage}"
                if self.slack_thread_ts:
                    _slack_notifier.add_to_thread(self.slack_thread_ts, acc_msg)
                else:
                    notify_slack(f"⚠️ [TMCMC] {acc_msg}", raise_on_error=False)
    
    def log_evaluation_counts(self, n_rom: int, n_fom: int) -> None:
        """Log evaluation counts."""
        if self.config.show_evaluation_counts:
            self._emit("evaluation_counts", n_rom=n_rom, n_fom=n_fom)
            self._log.info("      [TMCMC] Evaluations: ROM=%s, FOM=%s", n_rom, n_fom)
    
    def log_observation_based_update(self, subset_size: int, n_particles: int) -> None:
        """Log observation-based update start."""
        if self.config.show_linearization_updates:
            self._log.info(
                "      [TMCMC] Computing ROM errors for %s/%s particles (observation-based update)...",
                subset_size,
                n_particles,
            )
    
    def log_warning(self, message: str) -> None:
        """Log warning (only in MINIMAL/VERBOSE, silent in OFF/ERROR)."""
        # ERROR mode: silent (no print, only raise exceptions)
        # OFF mode: completely silent
        if self.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
            self._log.warning("      [TMCMC] %s", message)
            # Slack notification: All warnings (add to thread if available)
            if SLACK_ENABLED:
                if self.slack_thread_ts:
                    _slack_notifier.add_to_thread(self.slack_thread_ts, f"⚠️ {message}")
                else:
                    notify_slack(f"⚠️ [TMCMC] {message}", raise_on_error=False)
    
    def log_info(self, message: str, force: bool = False) -> None:
        """Log info message (only if debug enabled or forced)."""
        # ERROR mode: no output (silent)
        if force or (self.config.level != DebugLevel.OFF and self.config.level != DebugLevel.ERROR):
            self._log.info("      [TMCMC] %s", message)
    
    # ERROR-CHECK MODE methods (silent, raise exceptions)
    
    def check_numerical_errors(self, logL: np.ndarray, theta: np.ndarray, context: str = "") -> None:
        """Check for numerical errors (NaN/Inf)."""
        if not self.config.check_numerical_errors:
            return
        
        # Check logL
        if not np.all(np.isfinite(logL)):
            n_invalid = np.sum(~np.isfinite(logL))
            raise RuntimeError(
                f"Non-finite log-likelihood detected: {n_invalid}/{len(logL)} values "
                f"are NaN/Inf. Context: {context}"
            )
        
        # Check theta
        if not np.all(np.isfinite(theta)):
            n_invalid = np.sum(~np.isfinite(theta))
            raise RuntimeError(
                f"Non-finite parameters detected: {n_invalid}/{theta.size} values "
                f"are NaN/Inf. Context: {context}"
            )
        
        # Check if logL is stuck at -inf
        if np.all(logL == -np.inf):
            raise RuntimeError(
                f"All log-likelihood values are -inf. Model may be broken. Context: {context}"
            )
    
    def check_rom_error_explosion(self, rom_error: float, context: str = "", acc_rate: Optional[float] = None) -> None:
        """Check if ROM error exceeds hard limit."""
        if not self.config.check_rom_error_explosion:
            return
        
        # If acceptance rate is extremely low, ROM error check is unreliable
        # When acc_rate ≈ 0, particles are not moving, so ROM error may be artificially high
        # Skip error check in this case and just warn
        if acc_rate is not None and acc_rate < 0.01:
            if self.config.level in (DebugLevel.MINIMAL, DebugLevel.VERBOSE):
                warnings.warn(
                    f"ROM error check skipped: acc_rate={acc_rate:.4f} < 0.01. "
                    f"ROM error={rom_error:.3e} may be unreliable. Context: {context}",
                    RuntimeWarning,
                    stacklevel=2
                )
            return
        
        if rom_error > self.config.rom_error_hard_limit:
            # Make it a warning instead of error to allow continuation
            # ROM error explosion often happens when acceptance rate is very low
            if self.config.level == DebugLevel.ERROR:
                # ERROR mode: still raise, but with more context
                raise RuntimeError(
                    f"ROM error exploded: {rom_error:.3e} > {self.config.rom_error_hard_limit:.3e}. "
                    f"Model is likely broken. Context: {context}. "
                    f"Consider checking acceptance rate (may be too low)."
                )
            else:
                # Other modes: warn but continue
                warnings.warn(
                    f"ROM error very high: {rom_error:.3e} > {self.config.rom_error_hard_limit:.3e}. "
                    f"Context: {context}. Continuing anyway...",
                    RuntimeWarning,
                    stacklevel=2
                )
    
    def check_tmcmc_structure(self, weights: np.ndarray, ess: float, context: str = "") -> None:
        """Check TMCMC structure errors (zero weights, ESS=0, etc.)."""
        if not self.config.check_tmcmc_structure:
            return
        
        # Check if all weights are zero
        if np.all(weights == 0):
            raise RuntimeError(
                f"All TMCMC weights collapsed to zero. Resampling impossible. Context: {context}"
            )
        
        # Check ESS
        if ess <= 0:
            raise RuntimeError(
                f"ESS is zero or negative: {ess:.3e}. TMCMC cannot proceed. Context: {context}"
            )
    
    def check_acceptance_rate(self, acc_rate: float, context: str = "") -> None:
        """Check if acceptance rate is extremely low."""
        if not self.config.check_acceptance_rate:
            return
        
        if acc_rate < self.config.min_acceptance_rate:
            # ERROR mode: raise exception (silent error detection)
            # Other modes: warn
            if self.config.level == DebugLevel.ERROR:
                raise RuntimeError(
                    f"Acceptance rate too low: {acc_rate:.4f} < {self.config.min_acceptance_rate:.4f}. "
                    f"TMCMC may be stuck. Context: {context}"
                )
            else:
                warnings.warn(
                    f"Acceptance rate extremely low: {acc_rate:.4f} < {self.config.min_acceptance_rate:.4f}. "
                    f"TMCMC may be stuck. Context: {context}",
                    RuntimeWarning,
                    stacklevel=2
                )
    
    def check_covariance_matrix(self, cov: np.ndarray, context: str = "") -> None:
        """Check if covariance matrix is valid (positive definite)."""
        if not self.config.check_numerical_errors:
            return
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(cov)):
            raise RuntimeError(
                f"Covariance matrix contains NaN/Inf. Context: {context}"
            )
        
        # Check positive definiteness (eigenvalues > 0)
        # Use eigvalsh for symmetric matrices (more stable) and tolerance for floating error
        try:
            eigenvals = np.linalg.eigvalsh(cov)  # More stable for symmetric matrices
            # Tolerance for floating point errors (especially important for FAST_SANITY with small n_particles)
            if np.min(eigenvals) <= -1e-12:
                min_eigenval = np.min(eigenvals)
                raise RuntimeError(
                    f"Covariance matrix is not positive definite. "
                    f"Minimum eigenvalue: {min_eigenval:.3e}. Context: {context}"
                )
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"Failed to compute covariance matrix eigenvalues: {e}. Context: {context}"
            )
    
    def check_beta_progression(self, beta: float, delta_beta: float, stage: int, context: str = "") -> None:
        """Check if β is progressing (not stuck)."""
        if not self.config.check_tmcmc_structure:
            return
        
        # Check if beta is valid
        if not np.isfinite(beta) or not np.isfinite(delta_beta):
            raise RuntimeError(
                f"Beta progression contains NaN/Inf: β={beta:.4f}, Δβ={delta_beta:.4f}. "
                f"Stage: {stage}. Context: {context}"
            )
