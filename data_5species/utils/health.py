"""
Health monitoring utilities for tmcmc package.

Tracks health counters for likelihood/TSM evaluation to diagnose issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class LikelihoodHealthCounter:
    """
    Lightweight health counters for likelihood/TSM evaluation.

    Stored in diagnostics + metrics.json so failures can be triaged quickly.

    Attributes
    ----------
    n_calls : int
        Total number of likelihood evaluations
    n_tsm_fail : int
        Number of TSM solution failures
    n_output_nonfinite : int
        Count of non-finite entries seen in (t_arr/x0/sig2/mu/sig)
    n_var_raw_negative : int
        Number of negative variance values encountered
    n_var_raw_nonfinite : int
        Number of non-finite variance values encountered
    n_var_total_clipped : int
        Number of variance entries clipped to 1e-20

    Examples
    --------
    >>> health = LikelihoodHealthCounter()
    >>> health.n_calls += 1
    >>> health_dict = health.to_dict()
    """

    n_calls: int = 0
    n_tsm_fail: int = 0
    n_output_nonfinite: int = 0  # count of non-finite entries seen in (t_arr/x0/sig2/mu/sig)

    # Variance / likelihood stability
    n_var_raw_negative: int = 0
    n_var_raw_nonfinite: int = 0
    n_var_total_clipped: int = 0  # number of entries clipped to 1e-20

    def to_dict(self) -> Dict[str, int]:
        """
        Convert health counters to dictionary.

        Returns
        -------
        Dict[str, int]
            Dictionary of all health counters
        """
        return {
            "n_calls": int(self.n_calls),
            "n_tsm_fail": int(self.n_tsm_fail),
            "n_output_nonfinite": int(self.n_output_nonfinite),
            "n_var_raw_negative": int(self.n_var_raw_negative),
            "n_var_raw_nonfinite": int(self.n_var_raw_nonfinite),
            "n_var_total_clipped": int(self.n_var_total_clipped),
        }

    def add_from_dict(self, d: Dict[str, int]) -> None:
        """
        Add counts from another dictionary.

        Parameters
        ----------
        d : Dict[str, int]
            Dictionary of health counters to add
        """
        self.n_calls += int(d.get("n_calls", 0))
        self.n_tsm_fail += int(d.get("n_tsm_fail", 0))
        self.n_output_nonfinite += int(d.get("n_output_nonfinite", 0))
        self.n_var_raw_negative += int(d.get("n_var_raw_negative", 0))
        self.n_var_raw_nonfinite += int(d.get("n_var_raw_nonfinite", 0))
        self.n_var_total_clipped += int(d.get("n_var_total_clipped", 0))
