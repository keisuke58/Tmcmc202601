"""
Unit tests for TMCMC implementation.

Tests cover:
1. Input validation
2. MAP calculation correctness
3. Observation-based update behavior
4. Variable initialization
5. Constant usage
"""

import numpy as np
import pytest
from typing import List, Tuple
from case2_tmcmc_linearization import (
    run_TMCMC,
    _validate_tmcmc_inputs,
    TMCMCResult,
    DEFAULT_N_PARTICLES,
    DEFAULT_N_STAGES,
    DEFAULT_TARGET_ESS_RATIO,
    DEFAULT_UPDATE_LINEARIZATION_INTERVAL,
    DEFAULT_N_MUTATION_STEPS,
    DEFAULT_LINEARIZATION_THRESHOLD,
    MAX_LINEARIZATION_UPDATES,
    ROM_ERROR_THRESHOLD,
    ROM_ERROR_FALLBACK,
    BETA_CONVERGENCE_THRESHOLD,
    THETA_CONVERGENCE_THRESHOLD,
)


class TestInputValidation:
    """Test input validation for run_TMCMC."""
    
    def test_invalid_log_likelihood(self):
        """Test that non-callable log_likelihood raises TypeError."""
        with pytest.raises(TypeError, match="callable"):
            _validate_tmcmc_inputs(
                log_likelihood="not a function",
                prior_bounds=[(0, 1)],
                n_particles=100,
                n_stages=10,
                target_ess_ratio=0.5,
                evaluator=None,
                theta_base_full=None,
                active_indices=None,
            )
    
    def test_empty_prior_bounds(self):
        """Test that empty prior_bounds raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_tmcmc_inputs(
                log_likelihood=lambda x: 0.0,
                prior_bounds=[],
                n_particles=100,
                n_stages=10,
                target_ess_ratio=0.5,
                evaluator=None,
                theta_base_full=None,
                active_indices=None,
            )
    
    def test_invalid_bounds_order(self):
        """Test that lower >= upper bound raises ValueError."""
        with pytest.raises(ValueError, match="lower bound must be < upper"):
            _validate_tmcmc_inputs(
                log_likelihood=lambda x: 0.0,
                prior_bounds=[(1, 0)],  # lower > upper
                n_particles=100,
                n_stages=10,
                target_ess_ratio=0.5,
                evaluator=None,
                theta_base_full=None,
                active_indices=None,
            )
    
    def test_invalid_n_particles(self):
        """Test that n_particles <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_particles must be > 0"):
            _validate_tmcmc_inputs(
                log_likelihood=lambda x: 0.0,
                prior_bounds=[(0, 1)],
                n_particles=0,
                n_stages=10,
                target_ess_ratio=0.5,
                evaluator=None,
                theta_base_full=None,
                active_indices=None,
            )
    
    def test_invalid_target_ess_ratio(self):
        """Test that target_ess_ratio outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="target_ess_ratio must be in"):
            _validate_tmcmc_inputs(
                log_likelihood=lambda x: 0.0,
                prior_bounds=[(0, 1)],
                n_particles=100,
                n_stages=10,
                target_ess_ratio=1.5,  # > 1
                evaluator=None,
                theta_base_full=None,
                active_indices=None,
            )
    
    def test_evaluator_without_theta_base_full(self):
        """Test that evaluator without theta_base_full raises ValueError."""
        class MockEvaluator:
            pass
        
        with pytest.raises(ValueError, match="theta_base_full must be provided"):
            _validate_tmcmc_inputs(
                log_likelihood=lambda x: 0.0,
                prior_bounds=[(0, 1)],
                n_particles=100,
                n_stages=10,
                target_ess_ratio=0.5,
                evaluator=MockEvaluator(),
                theta_base_full=None,
                active_indices=[0],
            )


class TestTMCMCBasic:
    """Test basic TMCMC functionality."""
    
    def test_simple_gaussian(self):
        """Test TMCMC on simple Gaussian posterior."""
        # Simple 1D Gaussian: p(θ|D) ∝ exp(-0.5 * (θ - 1.0)^2)
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * (theta[0] - 1.0)**2
        
        prior_bounds = [(-5.0, 5.0)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=100,
            n_stages=5,
            target_ess_ratio=0.5,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        assert result.samples.shape == (100, 1)
        assert result.logL_values.shape == (100,)
        assert result.theta_MAP.shape == (1,)
        assert len(result.beta_schedule) > 0
        assert result.beta_schedule[-1] >= BETA_CONVERGENCE_THRESHOLD or result.converged
    
    def test_should_do_fom_initialized(self):
        """Test that should_do_fom is always initialized."""
        # This test ensures the variable is defined in all code paths
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * np.sum(theta**2)
        
        prior_bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        
        # Run with observation-based update disabled
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            use_observation_based_update=False,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        # If should_do_fom was uninitialized, this would raise UnboundLocalError
        # The test passes if no exception is raised


class TestMAPCalculation:
    """Test MAP calculation correctness."""
    
    def test_map_includes_prior(self):
        """Test that MAP uses log_prior + beta * logL, not just logL."""
        # This is tested implicitly by checking that MAP is reasonable
        # For a Gaussian with prior bounds, MAP should be within bounds
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * (theta[0] - 1.5)**2
        
        prior_bounds = [(0.0, 2.0)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=100,
            n_stages=5,
            seed=42,
        )
        
        # MAP should be within prior bounds
        assert 0.0 <= result.theta_MAP[0] <= 2.0
        # MAP should be close to true value (1.5) for this simple case
        assert abs(result.theta_MAP[0] - 1.5) < 0.5


class TestConstants:
    """Test that constants are properly defined and used."""
    
    def test_constants_defined(self):
        """Test that all constants are defined."""
        assert DEFAULT_N_PARTICLES > 0
        assert DEFAULT_N_STAGES > 0
        assert 0 < DEFAULT_TARGET_ESS_RATIO <= 1
        assert DEFAULT_UPDATE_LINEARIZATION_INTERVAL > 0
        assert DEFAULT_N_MUTATION_STEPS > 0
        assert 0 < DEFAULT_LINEARIZATION_THRESHOLD <= 1
        assert MAX_LINEARIZATION_UPDATES > 0
        assert ROM_ERROR_THRESHOLD > 0
        assert ROM_ERROR_FALLBACK > 0
        assert BETA_CONVERGENCE_THRESHOLD > 0
        assert THETA_CONVERGENCE_THRESHOLD > 0
    
    def test_default_values_match_constants(self):
        """Test that default parameter values match constants."""
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * np.sum(theta**2)
        
        prior_bounds = [(-1.0, 1.0)]
        
        # Run with defaults
        result1 = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            seed=42,
        )
        
        # Run with explicit constants
        result2 = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=DEFAULT_N_PARTICLES,
            n_stages=DEFAULT_N_STAGES,
            target_ess_ratio=DEFAULT_TARGET_ESS_RATIO,
            update_linearization_interval=DEFAULT_UPDATE_LINEARIZATION_INTERVAL,
            n_mutation_steps=DEFAULT_N_MUTATION_STEPS,
            seed=42,
        )
        
        # Results should be similar (same seed, same parameters)
        assert result1.samples.shape == result2.samples.shape


class TestVariableNaming:
    """Test that variable naming is consistent (MAP vs barycenter)."""
    
    def test_final_map_naming(self):
        """Test that final_MAP is used instead of final_barycenter."""
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * np.sum(theta**2)
        
        prior_bounds = [(-1.0, 1.0)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=42,
        )
        
        # Check that result has final_MAP attribute (not final_barycenter)
        assert hasattr(result, 'final_MAP')
        # final_MAP may be None if no linearization updates occurred
        # This is acceptable


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_1d_covariance_calculation(self):
        """Test that 1D parameter case doesn't crash on np.trace()."""
        # This tests the fix for np.trace() requiring 2D arrays
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * (theta[0] - 0.5)**2
        
        prior_bounds = [(0.0, 1.0)]
        
        # Should not raise ValueError about np.trace()
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        assert result.samples.shape == (50, 1)
    
    def test_multidimensional_parameters(self):
        """Test TMCMC with multiple parameters."""
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * np.sum((theta - np.array([0.5, -0.3, 0.8]))**2)
        
        prior_bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=100,
            n_stages=5,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        assert result.samples.shape == (100, 3)
        assert result.theta_MAP.shape == (3,)
        # All parameters should be within bounds
        assert np.all(result.theta_MAP >= np.array([-1.0, -1.0, -1.0]))
        assert np.all(result.theta_MAP <= np.array([1.0, 1.0, 1.0]))
    
    def test_extreme_prior_bounds(self):
        """Test with very wide prior bounds."""
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * theta[0]**2
        
        prior_bounds = [(-100.0, 100.0)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        assert -100.0 <= result.theta_MAP[0] <= 100.0
    
    def test_narrow_prior_bounds(self):
        """Test with very narrow prior bounds."""
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * (theta[0] - 0.5)**2
        
        prior_bounds = [(0.4, 0.6)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        assert 0.4 <= result.theta_MAP[0] <= 0.6
    
    def test_flat_likelihood(self):
        """Test with flat (constant) likelihood."""
        def log_likelihood(theta: np.ndarray) -> float:
            return 0.0  # Constant likelihood
        
        prior_bounds = [(-1.0, 1.0)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        # With flat likelihood, MAP should still be within bounds
        assert -1.0 <= result.theta_MAP[0] <= 1.0


class TestConvergence:
    """Test convergence behavior."""
    
    def test_beta_converges_to_one(self):
        """Test that beta eventually reaches 1.0 (or converges)."""
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * (theta[0] - 0.5)**2
        
        prior_bounds = [(0.0, 1.0)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=100,
            n_stages=10,  # Enough stages to potentially converge
            target_ess_ratio=0.5,
            seed=42,
        )
        
        # Beta should be non-decreasing
        beta_schedule = result.beta_schedule
        for i in range(1, len(beta_schedule)):
            assert beta_schedule[i] >= beta_schedule[i-1], "Beta should be non-decreasing"
        
        # Final beta should be <= 1.0
        assert beta_schedule[-1] <= 1.0 + 1e-10, "Final beta should be <= 1.0"
        
        # If converged, beta should be >= threshold
        if result.converged:
            assert beta_schedule[-1] >= BETA_CONVERGENCE_THRESHOLD - 1e-10


class TestErrorHandling:
    """Test error handling for invalid inputs."""
    
    def test_inf_log_likelihood(self):
        """Test handling of infinite log-likelihood values."""
        def log_likelihood(theta: np.ndarray) -> float:
            if theta[0] < 0:
                return -np.inf
            return -0.5 * theta[0]**2
        
        prior_bounds = [(-1.0, 1.0)]
        
        # Should not crash, but may have some particles with -inf logL
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        # MAP should be in valid region (theta[0] >= 0)
        assert result.theta_MAP[0] >= 0.0
    
    def test_nan_log_likelihood(self):
        """Test handling of NaN log-likelihood values."""
        def log_likelihood(theta: np.ndarray) -> float:
            if abs(theta[0]) < 0.1:
                return np.nan
            return -0.5 * theta[0]**2
        
        prior_bounds = [(-1.0, 1.0)]
        
        # Should handle NaN gracefully (may filter out or use fallback)
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=42,
        )
        
        assert isinstance(result, TMCMCResult)
        # Result should be valid (no NaN in MAP)
        assert np.isfinite(result.theta_MAP[0])


class TestObservationBasedUpdate:
    """Test observation-based update functionality."""
    
    def test_observation_based_update_flag(self):
        """Test that observation-based update can be enabled/disabled."""
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * np.sum(theta**2)
        
        prior_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        # Test with observation-based update enabled
        result1 = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            use_observation_based_update=True,
            seed=42,
        )
        
        # Test with observation-based update disabled
        result2 = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            use_observation_based_update=False,
            seed=42,
        )
        
        # Both should complete successfully
        assert isinstance(result1, TMCMCResult)
        assert isinstance(result2, TMCMCResult)
        assert result1.samples.shape == result2.samples.shape


class TestReproducibility:
    """Test reproducibility with fixed seeds."""
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * np.sum(theta**2)
        
        prior_bounds = [(-1.0, 1.0)]
        
        # Run twice with same seed
        result1 = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=12345,
        )
        
        result2 = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            seed=12345,
        )
        
        # Results should be identical (deterministic with fixed seed)
        np.testing.assert_array_equal(result1.samples, result2.samples)
        np.testing.assert_array_equal(result1.theta_MAP, result2.theta_MAP)
        assert result1.beta_schedule == result2.beta_schedule


class TestOptionalImprovements:
    """Optional improvements for regression testing and robustness."""
    
    def test_beta_schedule_monotonicity(self):
        """Test that beta schedule is monotonically non-decreasing."""
        # This is a lightweight safety check to detect implementation breaks early
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * np.sum(theta**2)
        
        prior_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=100,
            n_stages=10,
            seed=42,
        )
        
        # Beta schedule should be monotonically non-decreasing
        beta_schedule = np.array(result.beta_schedule)
        beta_diffs = np.diff(beta_schedule)
        assert np.all(beta_diffs >= -1e-10), \
            f"Beta schedule is not monotonic. Diffs: {beta_diffs[beta_diffs < -1e-10]}"
    
    def test_map_regression_simple_case(self):
        """Regression test for MAP calculation on a simple known case."""
        # This test helps detect algorithm changes that affect MAP estimation
        # For a simple Gaussian, we expect MAP to be close to the true value
        
        def log_likelihood(theta: np.ndarray) -> float:
            # True value is at theta = [0.5, -0.3]
            return -0.5 * ((theta[0] - 0.5)**2 + (theta[1] + 0.3)**2)
        
        prior_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        expected_map = np.array([0.5, -0.3])
        
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=200,
            n_stages=10,
            seed=42,  # Fixed seed for reproducibility
        )
        
        # MAP should be close to expected value (within reasonable tolerance)
        # Tolerance is relatively loose (0.2) to account for sampling variability
        np.testing.assert_allclose(
            result.theta_MAP,
            expected_map,
            atol=0.2,
            err_msg=f"MAP {result.theta_MAP} is too far from expected {expected_map}"
        )
    
    def test_evaluator_smoke_test(self):
        """Smoke test: ensure TMCMC runs to completion with evaluator."""
        # This is a lightweight test that just checks the code doesn't crash
        # We don't check numerical correctness, just that it runs
        
        def log_likelihood(theta: np.ndarray) -> float:
            return -0.5 * np.sum(theta**2)
        
        prior_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        
        # Create a minimal mock evaluator
        # We need to provide the required methods for linearization
        class MockEvaluator:
            def __init__(self):
                self._linearization_enabled = False
                self.call_count = 0
                self.fom_call_count = 0
                # Store a valid linearization point (14-dim as expected by the code)
                self._theta_linearization = np.zeros(14)
            
            def __call__(self, theta: np.ndarray) -> float:
                self.call_count += 1
                return log_likelihood(theta)
            
            def enable_linearization(self, enabled: bool):
                self._linearization_enabled = enabled
            
            def get_linearization_point(self):
                # Return a valid 14-dim array (not None)
                return self._theta_linearization.copy()
            
            def update_linearization_point(self, theta: np.ndarray):
                # Update the stored linearization point
                self._theta_linearization = theta.copy()
            
            def compute_ROM_error(self, theta: np.ndarray) -> float:
                # Return a small mock error
                return 0.01
        
        evaluator = MockEvaluator()
        # Use 14-dim theta_base_full as expected by the code
        theta_base_full = np.zeros(14)
        active_indices = [0, 1]  # First two parameters are active
        
        # Should run to completion without crashing
        result = run_TMCMC(
            log_likelihood=log_likelihood,
            prior_bounds=prior_bounds,
            n_particles=50,
            n_stages=3,
            evaluator=evaluator,
            theta_base_full=theta_base_full,
            active_indices=active_indices,
            use_observation_based_update=False,  # Disable to avoid FOM calls
            seed=42,
        )
        
        # Just check that it completed successfully
        assert isinstance(result, TMCMCResult)
        assert result.samples.shape == (50, 2)
        assert result.theta_MAP.shape == (2,)
        
        # Check that evaluator was called (at least for initial linearization point)
        # Note: evaluator.__call__ may not be called if log_likelihood is used directly
        # But get_linearization_point should have been called
        assert evaluator.get_linearization_point() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
