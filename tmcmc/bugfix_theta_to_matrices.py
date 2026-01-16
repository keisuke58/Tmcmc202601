"""
bugfix_theta_to_matrices.py - Patch for complex-step differentiation compatibility

Compatible with improved1207_paper_jit.py (the authoritative reference).

The authoritative file already has correct theta_to_matrices with complex dtype support.
This module provides a no-op patch for code that expects patch_biofilm_solver() to exist.

Usage:
    from bugfix_theta_to_matrices import patch_biofilm_solver
    patch_biofilm_solver()  # Safe to call - verifies implementation is correct
"""

import numpy as np
import logging

from config import setup_logging

logger = logging.getLogger(__name__)


def patch_biofilm_solver(verbose: bool = False):
    """
    Verify that BiofilmNewtonSolver.theta_to_matrices supports complex dtype.
    
    The authoritative improved1207_paper_jit.py already has this implemented
    correctly (lines 489-516). This function verifies correctness and does nothing
    else since no patch is needed.
    
    Parameters
    ----------
    verbose : bool, default=False
        If True, print verification messages. If False, silent (for production use).
    
    Returns
    -------
    BiofilmNewtonSolver : class
        The (verified) BiofilmNewtonSolver class
    """
    try:
        from improved1207_paper_jit import BiofilmNewtonSolver
    except ImportError:
        raise ImportError(
            "Could not import BiofilmNewtonSolver from improved1207_paper_jit. "
            "Ensure improved1207_paper_jit.py is in the Python path."
        )
    
    # Verify the implementation handles complex dtype correctly
    # Create a dummy solver instance to test the method
    solver = BiofilmNewtonSolver(
        dt=1e-5, maxtimestep=10, c_const=100.0, alpha_const=100.0,
        phi_init=0.2, active_species=[0, 1], use_numba=False
    )
    
    theta_test = np.array([
        0.8, 2.0, 1.0, 0.1, 0.2,
        1.5, 1.0, 2.0, 0.3, 0.4,
        2.0, 1.0, 2.0, 1.0
    ], dtype=np.complex128)
    theta_test[0] += 1j * 1e-30
    
    A, b_diag = solver.theta_to_matrices(theta_test)
    
    if not np.iscomplexobj(A):
        raise RuntimeError(
            "BiofilmNewtonSolver.theta_to_matrices does not preserve complex dtype! "
            "This indicates the authoritative file has been modified incorrectly."
        )
    
    if np.abs(np.imag(A[0, 0])) < 1e-40:
        raise RuntimeError(
            "Complex imaginary part not propagated correctly in theta_to_matrices."
        )
    
    if verbose:
        logger.info("BiofilmNewtonSolver.theta_to_matrices verified correct (complex-step ready)")
    
    return BiofilmNewtonSolver


# Alias for backward compatibility
def apply_complex_dtype_patch():
    """Alias for patch_biofilm_solver()."""
    return patch_biofilm_solver()


if __name__ == "__main__":
    setup_logging("INFO")
    logger.info("%s", "=" * 70)
    logger.info("Verifying theta_to_matrices complex-step compatibility")
    logger.info("%s", "=" * 70)
    
    try:
        BiofilmNewtonSolver = patch_biofilm_solver()
        logger.info("All checks passed!")
        logger.info("The authoritative improved1207_paper_jit.py is correct.")
    except Exception as e:
        logger.error("Verification failed: %s", e)
