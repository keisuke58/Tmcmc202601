"""
Utility modules for tmcmc package.

This package contains reusable utilities extracted from case2_tmcmc_linearization.py:
- io: File I/O operations (save_json, write_csv, etc.)
- timing: Timing statistics and context managers
- health: Health counters for likelihood evaluation
- validation: Input validation functions
"""

from .io import (
    code_crc32,
    save_npy,
    save_likelihood_meta,
    save_json,
    write_csv,
    to_jsonable,
)

from .timing import (
    TimingStats,
    timed,
)

from .health import (
    LikelihoodHealthCounter,
)

from .validation import (
    validate_tmcmc_inputs,
)

__all__ = [
    # I/O
    "code_crc32",
    "save_npy",
    "save_likelihood_meta",
    "save_json",
    "write_csv",
    "to_jsonable",
    # Timing
    "TimingStats",
    "timed",
    # Health
    "LikelihoodHealthCounter",
    # Validation
    "validate_tmcmc_inputs",
]
