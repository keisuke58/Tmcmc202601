"""
Joint 4-condition log-likelihood evaluator for TMCMC.

Runs 4 ODE solvers (CS, CH, DS, DH) simultaneously with shared parameters.
Parameters free in multiple conditions are forced to the same value.

Joint θ = 15 A_ij (union of all conditions' free indices = DH's full set).
For each condition, only its unlocked subset contributes to logL.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Parameter index → name mapping (for logging)
PARAM_NAMES = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",  # M1
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",  # M2
    "a13",
    "a14",
    "a23",
    "a24",  # M3
    "a55",
    "b5",  # M4
    "a15",
    "a25",
    "a35",
    "a45",  # M5
]

# All 15 A_ij indices (excluding b_i: 3,4,8,9,15)
ALL_AIJ_INDICES = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]

# Condition-specific locks (A_ij only, with α*=0 b_i always locked)
CONDITION_AIJ_LOCKS = {
    "Commensal_Static": {6, 7, 11, 13, 14, 16, 17, 18, 19},  # 6 free
    "Commensal_HOBIC": {6, 13, 16, 17, 18},  # 10 free (unlocked a23)
    "Dysbiotic_Static": set(),  # 15 free (all unlocked)
    "Dysbiotic_HOBIC": set(),  # 15 free
}


def get_condition_free_aij(condition_key: str) -> List[int]:
    """Get free A_ij indices for a condition."""
    locks = CONDITION_AIJ_LOCKS.get(condition_key, set())
    return [idx for idx in ALL_AIJ_INDICES if idx not in locks]


class JointLogLikelihoodEvaluator:
    """
    Joint 4-condition log-likelihood evaluator.

    logL(θ_joint) = Σ_c logL_c(θ_c)

    where θ_c is the condition-specific subset of θ_joint.

    Parameters
    ----------
    evaluators : dict
        {condition_key: LogLikelihoodEvaluator} for each condition.
    joint_active_indices : list
        All 15 A_ij indices (the joint parameter vector).
    condition_weights : dict, optional
        {condition_key: weight}. Default: uniform (1.0 each).
    """

    def __init__(
        self,
        evaluators: Dict[str, Any],
        joint_active_indices: List[int] = None,
        condition_weights: Optional[Dict[str, float]] = None,
        parallel: bool = True,
    ):
        self.evaluators = evaluators
        self._parallel = parallel
        self.joint_active_indices = joint_active_indices or ALL_AIJ_INDICES
        self.n_joint = len(self.joint_active_indices)
        self.condition_weights = condition_weights or {k: 1.0 for k in evaluators}

        # Build mapping: for each condition, which positions in θ_joint
        # correspond to its active_indices
        self._condition_maps = {}
        for cond_key, evaluator in evaluators.items():
            cond_active = evaluator.active_indices
            mapping = []  # list of (joint_pos, cond_pos)
            for cond_pos, idx in enumerate(cond_active):
                if idx in self.joint_active_indices:
                    joint_pos = self.joint_active_indices.index(idx)
                    mapping.append((joint_pos, cond_pos))
            self._condition_maps[cond_key] = {
                "mapping": mapping,
                "n_cond": len(cond_active),
            }

        # Log the sharing structure
        for cond_key in evaluators:
            cond_free = get_condition_free_aij(cond_key)
            names = [PARAM_NAMES[i] for i in cond_free]
            logger.info(f"Joint: {cond_key} uses {len(cond_free)} params: {names}")
        logger.info(
            f"Joint: total unique params = {self.n_joint}, "
            f"conditions = {list(evaluators.keys())}"
        )

        # TMCMC compatibility attributes
        self.fom_call_count = 0
        self.rom_call_count = 0
        self._linearization_enabled = False

        # Tracking
        self.call_count = 0
        self.theta_history = []
        self.logL_history = []
        self._logL_per_condition = []  # for diagnostics

    def __call__(self, theta_joint_sub: np.ndarray) -> float:
        """
        Evaluate joint log-likelihood.

        Parameters
        ----------
        theta_joint_sub : ndarray, shape (n_joint,)
            Joint parameter vector (15 A_ij values).

        Returns
        -------
        float
            Sum of weighted log-likelihoods across all conditions.
        """
        self.call_count += 1
        self.fom_call_count += 1
        logL_total = 0.0
        logL_components = {}

        # Build condition-specific theta_sub for each condition
        cond_thetas = {}
        for cond_key, evaluator in self.evaluators.items():
            info = self._condition_maps[cond_key]
            theta_cond = np.zeros(info["n_cond"])
            for joint_pos, cond_pos in info["mapping"]:
                theta_cond[cond_pos] = theta_joint_sub[joint_pos]
            cond_thetas[cond_key] = theta_cond

        # Evaluate all 4 conditions sequentially
        # (ThreadPoolExecutor overhead > ODE solve time; Numba JIT is fast enough)
        for cond_key in self.evaluators:
            logL_c = self.evaluators[cond_key](cond_thetas[cond_key])
            w = self.condition_weights.get(cond_key, 1.0)
            logL_total += w * logL_c
            logL_components[cond_key] = logL_c

        self.theta_history.append(theta_joint_sub.copy())
        self.logL_history.append(logL_total)
        self._logL_per_condition.append(logL_components)

        return logL_total

    def get_MAP(self) -> Tuple[np.ndarray, float]:
        """Get MAP estimate from evaluation history."""
        if not self.logL_history:
            raise ValueError("No evaluations yet")
        idx_max = np.argmax(self.logL_history)
        return self.theta_history[idx_max], self.logL_history[idx_max]

    def get_MAP_per_condition(self) -> Dict[str, float]:
        """Get per-condition logL at the MAP point."""
        if not self._logL_per_condition:
            return {}
        idx_max = np.argmax(self.logL_history)
        return self._logL_per_condition[idx_max]

    def _joint_to_cond_full(self, theta_joint_full: np.ndarray) -> np.ndarray:
        """Expand joint 15-dim theta to 20-dim full theta for condition evaluators."""
        if len(theta_joint_full) >= 20:
            return theta_joint_full  # already full dim
        theta_20 = np.zeros(20)
        for j, aij_idx in enumerate(self.joint_active_indices):
            if j < len(theta_joint_full):
                theta_20[aij_idx] = theta_joint_full[j]
        return theta_20

    # Stubs for TMCMC compatibility
    def update_linearization_point(self, theta_new_full: np.ndarray):
        theta_20 = self._joint_to_cond_full(theta_new_full)
        for evaluator in self.evaluators.values():
            if hasattr(evaluator, "update_linearization_point"):
                evaluator.update_linearization_point(theta_20)

    def enable_linearization(self, enable: bool = True):
        self._linearization_enabled = enable
        for evaluator in self.evaluators.values():
            if hasattr(evaluator, "enable_linearization"):
                evaluator.enable_linearization(enable)

    def get_linearization_point(self) -> np.ndarray:
        # Return in joint dim (15)
        for evaluator in self.evaluators.values():
            if hasattr(evaluator, "get_linearization_point"):
                theta_20 = evaluator.get_linearization_point()
                # Extract joint indices
                return np.array([theta_20[i] for i in self.joint_active_indices])
        return np.zeros(self.n_joint)

    def compute_ROM_error(self, theta_full: np.ndarray) -> float:
        theta_20 = self._joint_to_cond_full(theta_full)
        errors = []
        for evaluator in self.evaluators.values():
            if hasattr(evaluator, "compute_ROM_error"):
                errors.append(evaluator.compute_ROM_error(theta_20))
        return float(np.mean(errors)) if errors else 0.0

    def get_health(self) -> Dict[str, Any]:
        health = {"joint_calls": self.call_count}
        for cond_key, evaluator in self.evaluators.items():
            if hasattr(evaluator, "get_health"):
                health[cond_key] = evaluator.get_health()
        return health
