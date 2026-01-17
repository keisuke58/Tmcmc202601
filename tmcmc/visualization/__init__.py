"""
Visualization utilities for tmcmc package.

This package contains plotting functionality extracted from case2_tmcmc_linearization.py:
- plot_manager: PlotManager class for managing plot generation and file tracking
- helpers: Helper functions for visualization (compute_phibar, compute_fit_metrics, etc.)
"""

from .plot_manager import (
    PlotManager,
)

from .helpers import (
    compute_phibar,
    compute_fit_metrics,
    export_tmcmc_diagnostics_tables,
)

__all__ = [
    "PlotManager",
    "compute_phibar",
    "compute_fit_metrics",
    "export_tmcmc_diagnostics_tables",
]
