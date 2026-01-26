"""
Main entry point and CLI processing for tmcmc package.

This package contains the main() function and related CLI/orchestration code
extracted from case2_tmcmc_linearization.py:
- case2_main: Main function and CLI processing
"""

from .case2_main import (
    main,
    parse_args,
    select_sparse_data_indices,
    generate_synthetic_data,
    _self_check_tsm_once,
    _default_output_root_for_mode,
    _stable_hash_int,
    MCMCConfig,
    ExperimentConfig,
    compute_MAP_with_uncertainty,
)

__all__ = [
    "main",
    "parse_args",
    "select_sparse_data_indices",
    "generate_synthetic_data",
    "_self_check_tsm_once",
    "_default_output_root_for_mode",
    "_stable_hash_int",
    "MCMCConfig",
    "ExperimentConfig",
    "compute_MAP_with_uncertainty",
]
