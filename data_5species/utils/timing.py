"""
Timing utilities for tmcmc package.

Provides timing statistics and context managers for performance measurement.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class TimingStats:
    """
    Lightweight timing aggregator (seconds + call counts) for metrics.json.
    
    Tracks wall time spent in different code sections, useful for performance
    analysis and optimization.
    
    Examples
    --------
    >>> stats = TimingStats()
    >>> with timed(stats, "computation"):
    ...     # do some work
    ...     pass
    >>> print(stats.get_s("computation"))
    """
    
    seconds: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, name: str, dt_s: float) -> None:
        """
        Add a timing measurement.
        
        Parameters
        ----------
        name : str
            Operation name
        dt_s : float
            Elapsed time in seconds
        """
        if not name:
            return
        if not np.isfinite(dt_s):
            return
        self.seconds[name] += float(dt_s)
        self.counts[name] += 1

    def get_s(self, name: str) -> float:
        """
        Get total time spent in an operation.
        
        Parameters
        ----------
        name : str
            Operation name
            
        Returns
        -------
        float
            Total time in seconds
        """
        return float(self.seconds.get(name, 0.0))

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of current timing statistics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with "seconds" and "counts" keys
        """
        # Convert defaultdicts to plain dicts
        return {
            "seconds": {k: float(v) for k, v in self.seconds.items()},
            "counts": {k: int(v) for k, v in self.counts.items()},
        }


@contextmanager
def timed(stats: Optional[TimingStats], name: str):
    """
    Context manager to accumulate wall time into TimingStats.
    
    Parameters
    ----------
    stats : TimingStats, optional
        Timing statistics object. If None, timing is not tracked.
    name : str
        Operation name for this timing measurement
        
    Examples
    --------
    >>> stats = TimingStats()
    >>> with timed(stats, "my_operation"):
    ...     # code to time
    ...     result = expensive_computation()
    >>> print(f"Time: {stats.get_s('my_operation')}s")
    """
    if stats is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add(name, time.perf_counter() - t0)
