"""
Shared configuration + logging helpers for tmcmc scripts.

Goals:
- Centralize defaults so they can be reused across scripts/tests.
- Avoid `print` in all tmcmc code paths (use logging instead).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Paths / run layout
# -----------------------------------------------------------------------------

RUNS_ROOT_DEFAULT = Path("tmcmc") / "_runs"


# -----------------------------------------------------------------------------
# TMCMC / ROM / convergence defaults
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TMCMCDefaults:
    n_particles: int = 2000
    n_stages: int = 30
    target_ess_ratio: float = 0.5
    min_delta_beta: float = 0.02
    update_linearization_interval: int = 3
    n_mutation_steps: int = 5
    linearization_threshold: float = 0.75
    max_linearization_updates: int = 5

    # Beta step control
    max_delta_beta: float = 0.05  # Reduced from 0.2 for better stability and convergence

    # Mutation proposal scaling
    mutation_scale_factor: float = 2.0


@dataclass(frozen=True)
class ROMErrorDefaults:
    threshold: float = 0.01
    fallback: float = 1.0


@dataclass(frozen=True)
class ConvergenceDefaults:
    beta_convergence_threshold: float = 1.0
    theta_convergence_threshold: float = 1e-3


@dataclass(frozen=True)
class ProposalDefaults:
    optimal_scale_factor: float = 2.38**2
    covariance_nugget_base: float = 1e-8
    covariance_nugget_scale: float = 1e-6


@dataclass(frozen=True)
class LinearizationDefaults:
    """Linearization update stabilization constants."""

    # Cap a single θ0 update step to avoid large jumps that can freeze mutation/acceptance.
    max_theta0_step_norm: float = 0.75
    # Allow multiple small sub-updates in a single update event (bounded by max_linearization_updates).
    max_linearization_subupdates_per_event: int = 3


TMCMC_DEFAULTS = TMCMCDefaults()
ROM_ERROR_DEFAULTS = ROMErrorDefaults()
CONVERGENCE_DEFAULTS = ConvergenceDefaults()
PROPOSAL_DEFAULTS = ProposalDefaults()
LINEARIZATION_DEFAULTS = LinearizationDefaults()

# Convenience constants for backward compatibility
MAX_THETA0_STEP_NORM = LINEARIZATION_DEFAULTS.max_theta0_step_norm
MAX_LINEARIZATION_SUBUPDATES_PER_EVENT = (
    LINEARIZATION_DEFAULTS.max_linearization_subupdates_per_event
)


# -----------------------------------------------------------------------------
# Debug configuration (verbosity gates)
# -----------------------------------------------------------------------------


class DebugLevel(Enum):
    """Debug output levels."""

    OFF = 0
    ERROR = 1
    MINIMAL = 2
    VERBOSE = 3


@dataclass
class DebugConfig:
    """Debug configuration for controlling diagnostic output."""

    level: DebugLevel = DebugLevel.ERROR
    show_beta_progress: bool = False
    show_linearization_updates: bool = False
    show_rom_errors: bool = False
    show_acceptance_rates: bool = False
    show_evaluation_counts: bool = False

    # ERROR-level checks (silent detection; handled by caller)
    check_numerical_errors: bool = True
    check_rom_error_explosion: bool = True
    check_tmcmc_structure: bool = True
    check_acceptance_rate: bool = True
    rom_error_hard_limit: float = 0.2
    min_acceptance_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.level == DebugLevel.OFF:
            self.show_beta_progress = False
            self.show_linearization_updates = False
            self.show_rom_errors = False
            self.show_acceptance_rates = False
            self.show_evaluation_counts = False
            self.check_numerical_errors = False
            self.check_rom_error_explosion = False
            self.check_tmcmc_structure = False
            self.check_acceptance_rate = False
            return

        if self.level == DebugLevel.ERROR:
            self.show_beta_progress = False
            self.show_linearization_updates = False
            self.show_rom_errors = False
            self.show_acceptance_rates = False
            self.show_evaluation_counts = False
            self.check_numerical_errors = True
            self.check_rom_error_explosion = True
            self.check_tmcmc_structure = True
            self.check_acceptance_rate = True
        elif self.level == DebugLevel.MINIMAL:
            self.show_beta_progress = True
            self.show_linearization_updates = True
            self.show_rom_errors = False
            self.show_acceptance_rates = False
            self.show_evaluation_counts = False
        elif self.level == DebugLevel.VERBOSE:
            self.show_beta_progress = True
            self.show_linearization_updates = True
            self.show_rom_errors = True
            self.show_acceptance_rates = True
            self.show_evaluation_counts = True


# -----------------------------------------------------------------------------
# Case2 model configuration
# -----------------------------------------------------------------------------

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "M1": {
        "dt": 1e-5,
        "maxtimestep": 7500,
        "c_const": 25.0,
        "alpha_const": 0.0,
        "phi_init": 0.02,
        "active_species": [0, 1],
        "active_indices": list(range(5)),  # θ indices [a11,a12,a22,b1,b2]
        "param_names": ["a11", "a12", "a22", "b1", "b2"],
    },
    "M2": {
        "dt": 1e-5,
        "maxtimestep": 7500,
        "c_const": 25.0,
        "alpha_const": 0.0,
        "phi_init": 0.02,
        "active_species": [2, 3],
        "active_indices": list(range(5, 10)),  # θ indices [a33,a34,a44,b3,b4]
        "param_names": ["a33", "a34", "a44", "b3", "b4"],
    },
    "M3": {
        "dt": 1e-4,
        "maxtimestep": 750,
        "c_const": 25.0,
        "alpha_const": 0.0,
        "phi_init": 0.02,
        "active_species": [0, 1, 2, 3],
        "active_indices": list(range(10, 14)),  # θ indices [a13,a14,a23,a24]
        "param_names": ["a13", "a14", "a23", "a24"],
    },
    # Validation setup from paper Case II (time-dependent antibiotics).
    # Table 3 / Fig. 15: alpha = 50 * I[t > 500] (with dt=1e-4).
    "M3_val": {
        "dt": 1e-4,
        "maxtimestep": 1500,
        "c_const": 25.0,
        "alpha_const": 0.0,  # baseline before switch
        # "ど真ん中": switch at the middle of the simulated time horizon
        "alpha_schedule": {"switch_frac": 0.5, "alpha_before": 0.0, "alpha_after": 50.0},
        "phi_init": 0.02,
        "active_species": [0, 1, 2, 3],
        "active_indices": list(range(10, 14)),  # keep same uncertainty block as M3
        "param_names": ["a13", "a14", "a23", "a24"],
    },
    "M4": {
        "dt": 1e-4,
        "maxtimestep": 750,
        "c_const": 25.0,
        "alpha_const": 0.0,
        "phi_init": 0.02,
        "active_species": [0, 1, 2, 3, 4],
        "active_indices": [14, 15],  # θ indices [a55, b5]
        "param_names": ["a55", "b5"],
    },
    "M5": {
        "dt": 1e-4,
        "maxtimestep": 750,
        "c_const": 25.0,
        "alpha_const": 0.0,
        "phi_init": 0.02,
        "active_species": [0, 1, 2, 3, 4],
        "active_indices": [16, 17, 18, 19],  # θ indices [a15, a25, a35, a45]
        "param_names": ["a15", "a25", "a35", "a45"],
    },
    "experiment": {"n_data": 20, "sigma_obs": 0.001, "cov_rel": 0.005, "aleatory_samples": 500},
}

PRIOR_BOUNDS_DEFAULT = (0.0, 3.0)


# -----------------------------------------------------------------------------
# Report thresholds (make_report.py)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportThresholds:
    fail_ess_min: int = 100
    fail_accept_rate_mean: float = 0.05
    fail_rom_error_max: float = 0.5
    warn_ess_min: int = 300
    warn_rom_error_max: float = 0.3
    models: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", ["M1", "M2", "M3"])


REPORT_THRESHOLDS = ReportThresholds()


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------


def setup_logging(
    level: str | int = "INFO",
    log_path: str | Path | None = None,
    *,
    force: bool = False,
) -> None:
    """
    Configure root logging in a safe, idempotent way.

    - Logs go to stderr by default.
    - If log_path is provided, also logs to that file (append mode).
    - If handlers are already present:
        - updates the log level
        - adds a FileHandler if log_path is provided and not already configured
    - If force=True, replaces existing handlers.
    """
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    else:
        lvl = int(level)

    root = logging.getLogger()
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def _has_file_handler(path: Path) -> bool:
        want = str(path.resolve())
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if str(Path(h.baseFilename).resolve()) == want:  # type: ignore[attr-defined]
                        return True
                except Exception:
                    continue
        return False

    if force and root.handlers:
        for h in list(root.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            root.removeHandler(h)

    # Ensure there is at least a console handler
    if not root.handlers:
        root.setLevel(lvl)
        sh = logging.StreamHandler()
        sh.setLevel(lvl)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    else:
        root.setLevel(lvl)
        for h in root.handlers:
            h.setLevel(lvl)

    # Optional file handler
    if log_path is not None:
        p = Path(log_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not _has_file_handler(p):
            fh = logging.FileHandler(p, mode="a", encoding="utf-8")
            fh.setLevel(lvl)
            fh.setFormatter(fmt)
            root.addHandler(fh)
