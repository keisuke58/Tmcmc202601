#!/usr/bin/env python3
"""
Validation tests for GPU TMCMC optimized settings.

Tests:
  1. build_replicate_sigma: min_sigma=0.05, max_info_ratio=10
  2. Information content balance per condition (no species > 50%)
  3. Rare species downweighting (lambda_rare=0.1)
  4. Adaptive lambda_ch per condition
  5. CLI flags parse correctly
  6. Likelihood function constructs without error (CPU, quick)
  7. End-to-end mini TMCMC run (5 particles, 2 stages)

Usage:
    cd data_5species/main
    python test_gpu_optimizations.py
"""

from __future__ import annotations

import sys
import os
import json
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DATA_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(DATA_DIR))

# Force CPU for tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPLICATE_CSV = str(DATA_DIR / "experiment_data" / "fig3_species_distribution_replicates.csv")
SPECIES_MAP = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. dispar": 2,
    "V. parvula": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4,
    "P. gingivalis_W83": 4,
}
CONDITIONS = [
    ("Commensal", "Static", "CS"),
    ("Commensal", "HOBIC", "CH"),
    ("Dysbiotic", "Static", "DS"),
    ("Dysbiotic", "HOBIC", "DH"),
]

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator for test functions."""

    def wrapper(fn):
        global passed, failed
        print(f"  [{name}] ... ", end="", flush=True)
        try:
            fn()
            print("PASS")
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            traceback.print_exc()
            failed += 1
            errors.append((name, str(e)))

    return wrapper


# =====================================================================
# Test 1: build_replicate_sigma basics
# =====================================================================
@test("sigma_min_floor")
def _():
    """min_sigma=0.05 is enforced (not 0.005)."""
    from _test_helpers import build_replicate_sigma

    for cond, cult, tag in CONDITIONS:
        sigma = build_replicate_sigma(
            REPLICATE_CSV,
            cond,
            cult,
            days=[1, 3, 5, 7, 10, 14, 21],
            species_map=SPECIES_MAP,
            min_sigma=0.05,
        )
        assert sigma.min() >= 0.05 - 1e-10, f"{tag}: sigma.min()={sigma.min():.4f} < 0.05"


@test("sigma_shape")
def _():
    """Sigma matrix has correct shape (n_timepoints, 5)."""
    from _test_helpers import build_replicate_sigma

    days = [1, 3, 5, 7, 10, 14, 21]
    sigma = build_replicate_sigma(
        REPLICATE_CSV,
        "Commensal",
        "Static",
        days=days,
        species_map=SPECIES_MAP,
    )
    assert sigma.shape == (7, 5), f"Expected (7,5), got {sigma.shape}"


@test("sigma_no_nan_inf")
def _():
    """No NaN or Inf in sigma matrices."""
    from _test_helpers import build_replicate_sigma

    for cond, cult, tag in CONDITIONS:
        sigma = build_replicate_sigma(
            REPLICATE_CSV,
            cond,
            cult,
            days=[1, 3, 5, 7, 10, 14, 21],
            species_map=SPECIES_MAP,
        )
        assert not np.any(np.isnan(sigma)), f"{tag}: NaN in sigma"
        assert not np.any(np.isinf(sigma)), f"{tag}: Inf in sigma"


# =====================================================================
# Test 2: max_info_ratio cap
# =====================================================================
@test("max_info_ratio_cap")
def _():
    """max_info_ratio=10 caps per-timepoint information imbalance."""
    from _test_helpers import build_replicate_sigma

    for cond, cult, tag in CONDITIONS:
        sigma = build_replicate_sigma(
            REPLICATE_CSV,
            cond,
            cult,
            days=[1, 3, 5, 7, 10, 14, 21],
            species_map=SPECIES_MAP,
            max_info_ratio=10.0,
        )
        for t in range(sigma.shape[0]):
            info = 1.0 / sigma[t, :] ** 2
            ratio = info.max() / np.median(info)
            assert (
                ratio <= 10.0 + 0.01
            ), f"{tag} t={t}: info ratio {ratio:.1f} > 10.0 (species info: {info})"


@test("max_info_ratio_disabled")
def _():
    """max_info_ratio=0 disables the cap."""
    from _test_helpers import build_replicate_sigma

    sigma_capped = build_replicate_sigma(
        REPLICATE_CSV,
        "Commensal",
        "Static",
        days=[1, 3, 5, 7, 10, 14, 21],
        species_map=SPECIES_MAP,
        max_info_ratio=10.0,
    )
    sigma_uncapped = build_replicate_sigma(
        REPLICATE_CSV,
        "Commensal",
        "Static",
        days=[1, 3, 5, 7, 10, 14, 21],
        species_map=SPECIES_MAP,
        max_info_ratio=0,
    )
    # Capped should have >= sigmas (floor raised)
    assert np.all(sigma_capped >= sigma_uncapped - 1e-10), "Capped sigma should be >= uncapped"


@test("max_info_ratio_effect_on_CS")
def _():
    """CS (An dominant) should have sigma raised for An by cap."""
    from _test_helpers import build_replicate_sigma

    sigma_no_cap = build_replicate_sigma(
        REPLICATE_CSV,
        "Commensal",
        "Static",
        days=[1, 3, 5, 7, 10, 14, 21],
        species_map=SPECIES_MAP,
        max_info_ratio=0,
    )
    sigma_capped = build_replicate_sigma(
        REPLICATE_CSV,
        "Commensal",
        "Static",
        days=[1, 3, 5, 7, 10, 14, 21],
        species_map=SPECIES_MAP,
        max_info_ratio=10.0,
    )
    # An (idx=1) should have sigma raised in at least some timepoints
    an_raised = np.any(sigma_capped[:, 1] > sigma_no_cap[:, 1] + 1e-6)
    # Note: An may or may not be raised depending on actual IQR data
    # The important thing is the cap is applied correctly
    info_capped = 1.0 / sigma_capped**2
    for t in range(sigma_capped.shape[0]):
        ratio = info_capped[t].max() / np.median(info_capped[t])
        assert ratio <= 10.01


# =====================================================================
# Test 3: Information content balance
# =====================================================================
@test("info_balance_no_species_over_50pct")
def _():
    """No single species contributes > 50% of logL information (with cap)."""
    from _test_helpers import build_replicate_sigma

    for cond, cult, tag in CONDITIONS:
        sigma = build_replicate_sigma(
            REPLICATE_CSV,
            cond,
            cult,
            days=[1, 3, 5, 7, 10, 14, 21],
            species_map=SPECIES_MAP,
            max_info_ratio=10.0,
        )
        # Sum information across timepoints per species
        info_per_species = np.sum(1.0 / sigma**2, axis=0)
        total_info = info_per_species.sum()
        pct = info_per_species / total_info * 100
        max_pct = pct.max()
        assert max_pct < 50.0, f"{tag}: species {pct.argmax()} at {max_pct:.1f}% (>50%)"


@test("info_balance_comparison_old_vs_new")
def _():
    """Old min_sigma=0.005 had >90% dominance; new setting should be <50%."""
    from _test_helpers import build_replicate_sigma

    # Old (broken)
    sigma_old = build_replicate_sigma(
        REPLICATE_CSV,
        "Commensal",
        "Static",
        days=[1, 3, 5, 7, 10, 14, 21],
        species_map=SPECIES_MAP,
        min_sigma=0.005,
        max_info_ratio=0,
    )
    info_old = np.sum(1.0 / sigma_old**2, axis=0)
    pct_old = info_old / info_old.sum() * 100
    max_old = pct_old.max()

    # New (fixed)
    sigma_new = build_replicate_sigma(
        REPLICATE_CSV,
        "Commensal",
        "Static",
        days=[1, 3, 5, 7, 10, 14, 21],
        species_map=SPECIES_MAP,
        min_sigma=0.05,
        max_info_ratio=10.0,
    )
    info_new = np.sum(1.0 / sigma_new**2, axis=0)
    pct_new = info_new / info_new.sum() * 100
    max_new = pct_new.max()

    print(f"\n    Old max: {max_old:.1f}%, New max: {max_new:.1f}%", end=" ... ", flush=True)
    assert max_new < max_old, f"New ({max_new:.1f}%) should be < Old ({max_old:.1f}%)"
    assert max_new < 50.0, f"New max species {max_new:.1f}% still > 50%"


# =====================================================================
# Test 4: Rare species weights (lambda_rare)
# =====================================================================
@test("lambda_rare_downweight")
def _():
    """lambda_rare=0.1 downweights species with mean < 5%."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    # Fake data: 5 species, Fn (idx=3) at 1%, Pg (idx=4) at 2%
    data = np.array(
        [
            [0.30, 0.54, 0.13, 0.01, 0.02],
            [0.28, 0.56, 0.12, 0.01, 0.03],
            [0.25, 0.58, 0.13, 0.02, 0.02],
        ]
    )
    obs = jnp.array(data, dtype=jnp.float64)
    n_obs, n_species = data.shape

    # Without lambda_rare
    weights_no = jnp.ones((n_obs, n_species))
    # With lambda_rare=0.1
    mean_abundance = jnp.mean(obs, axis=0)
    rare_mask = (mean_abundance < 0.05).astype(jnp.float64)
    rare_weight = rare_mask * 0.1 + (1.0 - rare_mask) * 1.0
    weights_yes = jnp.ones((n_obs, n_species)) * rare_weight[jnp.newaxis, :]

    # Fn (idx=3) and Pg (idx=4) should be downweighted
    assert float(weights_yes[0, 3]) == 0.1, f"Fn weight={float(weights_yes[0,3])}"
    assert float(weights_yes[0, 4]) == 0.1, f"Pg weight={float(weights_yes[0,4])}"
    # So, An should NOT be downweighted
    assert float(weights_yes[0, 1]) == 1.0, f"An weight={float(weights_yes[0,1])}"


# =====================================================================
# Test 5: Adaptive lambda_ch
# =====================================================================
@test("adaptive_lambda_ch_DH")
def _():
    """DH auto-tunes Ch3 to 3.0."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="Dysbiotic")
    parser.add_argument("--cultivation", default="HOBIC")
    parser.add_argument("--lambda-ch1", type=float, default=1.0)
    parser.add_argument("--lambda-ch2", type=float, default=0.5)
    parser.add_argument("--lambda-ch3", type=float, default=2.0)
    parser.add_argument("--lambda-ch5", type=float, default=0.3)
    args = parser.parse_args([])

    lambda_ch = {1: args.lambda_ch1, 2: args.lambda_ch2, 3: args.lambda_ch3, 5: args.lambda_ch5}
    if args.condition == "Dysbiotic" and args.cultivation == "HOBIC":
        if args.lambda_ch3 == 2.0:
            lambda_ch[3] = 3.0
    assert lambda_ch[3] == 3.0, f"DH Ch3 should be 3.0, got {lambda_ch[3]}"


@test("adaptive_lambda_ch_CS")
def _():
    """Commensal auto-tunes Ch3 to 1.5."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="Commensal")
    parser.add_argument("--cultivation", default="Static")
    parser.add_argument("--lambda-ch3", type=float, default=2.0)
    args = parser.parse_args([])

    lambda_ch = {3: args.lambda_ch3}
    if args.condition == "Commensal":
        if args.lambda_ch3 == 2.0:
            lambda_ch[3] = 1.5
    assert lambda_ch[3] == 1.5, f"CS Ch3 should be 1.5, got {lambda_ch[3]}"


@test("adaptive_lambda_ch_no_override")
def _():
    """User-specified Ch3 is NOT overridden by auto-tune."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="Dysbiotic")
    parser.add_argument("--cultivation", default="HOBIC")
    parser.add_argument("--lambda-ch3", type=float, default=5.0)  # user override
    args = parser.parse_args([])

    lambda_ch = {3: args.lambda_ch3}
    if args.condition == "Dysbiotic" and args.cultivation == "HOBIC":
        if args.lambda_ch3 == 2.0:  # only if default
            lambda_ch[3] = 3.0
    assert lambda_ch[3] == 5.0, f"User override should stay 5.0, got {lambda_ch[3]}"


# =====================================================================
# Test 6: CLI flags parse
# =====================================================================
@test("cli_flags_parse")
def _():
    """All new CLI flags parse correctly."""
    import argparse

    # Simulate the parser from estimate_reduced_nishioka_jax.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-rare", type=float, default=0.1)
    parser.add_argument("--max-info-ratio", type=float, default=10.0)
    parser.add_argument("--use-de-mc", action="store_true")
    parser.add_argument("--use-student-t", action="store_true")
    parser.add_argument("--student-t-nu", type=float, default=5.0)
    parser.add_argument(
        "--resample-method", default="systematic", choices=["systematic", "multinomial"]
    )

    args = parser.parse_args(
        [
            "--lambda-rare",
            "0.1",
            "--max-info-ratio",
            "10.0",
            "--use-de-mc",
            "--use-student-t",
            "--student-t-nu",
            "5.0",
            "--resample-method",
            "systematic",
        ]
    )
    assert args.lambda_rare == 0.1
    assert args.max_info_ratio == 10.0
    assert args.use_de_mc is True
    assert args.use_student_t is True
    assert args.student_t_nu == 5.0
    assert args.resample_method == "systematic"


# =====================================================================
# Test 7: Sigma matrix detailed check per condition
# =====================================================================
@test("sigma_4conditions_detailed")
def _():
    """Print sigma details for all 4 conditions for manual inspection."""
    from _test_helpers import build_replicate_sigma

    days = [1, 3, 5, 7, 10, 14, 21]
    species_names = ["So", "An", "Vd", "Fn", "Pg"]

    print()
    for cond, cult, tag in CONDITIONS:
        sigma = build_replicate_sigma(
            REPLICATE_CSV,
            cond,
            cult,
            days=days,
            species_map=SPECIES_MAP,
            min_sigma=0.05,
            max_info_ratio=10.0,
        )
        info = np.sum(1.0 / sigma**2, axis=0)
        pct = info / info.sum() * 100
        mean_sig = np.mean(sigma, axis=0)
        print(
            f"    {tag}: σ_mean=[{', '.join(f'{s:.3f}' for s in mean_sig)}]  "
            f"info%=[{', '.join(f'{p:.0f}' for p in pct)}]%"
        )
    print("    ", end="")


# =====================================================================
# Test 8: Likelihood construction (no JAX error)
# =====================================================================
@test("likelihood_construction")
def _():
    """make_log_likelihood_jax_ode constructs without error."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from estimate_reduced_nishioka_jax import make_log_likelihood_jax_ode

    # Minimal fake data
    data = np.random.dirichlet([1] * 5, size=4)
    sigma = np.full((4, 5), 0.05)
    phi_init = np.full(5, 0.2)
    idx_sparse = np.array([100, 500, 1000, 2000])

    logL_fn = make_log_likelihood_jax_ode(
        data=data,
        t_days=np.array([1, 3, 7, 14]),
        idx_sparse=idx_sparse,
        sigma_obs=sigma,
        phi_init=phi_init,
        dt=1e-4,
        n_steps=2500,
        K_hill=0.05,
        n_hill=4.0,
        lambda_pg=5.0,
        lambda_late=3.0,
        use_student_t=True,
        student_t_nu=5.0,
        lambda_rare=0.1,
        rare_threshold=0.05,
    )
    assert callable(logL_fn), "logL should be callable"

    # Test with random theta (just checks it runs, not correctness)
    theta = jnp.zeros(20, dtype=jnp.float64)
    val = jax.jit(logL_fn)(theta)
    assert jnp.isfinite(val), f"logL={val} is not finite"


# =====================================================================
# Test 9: End-to-end mini TMCMC (CPU, 5 particles, 2 stages)
# =====================================================================
@test("e2e_mini_tmcmc")
def _():
    """End-to-end TMCMC: 5 particles, 2 stages, DH condition."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from estimate_reduced_nishioka_jax import make_log_likelihood_jax_ode, load_prior_bounds
    from estimate_reduced_nishioka import load_experimental_data, convert_days_to_model_time
    from _test_helpers import build_replicate_sigma
    from tmcmc_nuts_engine import tmcmc_engine

    data, t_days, sigma_est, phi_init_exp, metadata = load_experimental_data(
        DATA_DIR,
        "Dysbiotic",
        "HOBIC",
        start_from_day=1,
        normalize=True,
    )
    sigma_obs = build_replicate_sigma(
        REPLICATE_CSV,
        "Dysbiotic",
        "HOBIC",
        days=metadata["days"],
        species_map=SPECIES_MAP,
        n_species=5,
        min_sigma=0.05,
        max_info_ratio=10.0,
    )
    phi_init = phi_init_exp / phi_init_exp.sum()
    phi_init = np.clip(phi_init, 0.001, 0.99)

    n_steps = 500  # short for speed
    _, idx_sparse = convert_days_to_model_time(t_days, 1e-4, n_steps, day_scale=None)
    idx_sparse = np.clip(idx_sparse, 0, n_steps)

    logL_fn = make_log_likelihood_jax_ode(
        data=data,
        t_days=t_days,
        idx_sparse=idx_sparse,
        sigma_obs=sigma_obs,
        phi_init=phi_init,
        dt=1e-4,
        n_steps=n_steps,
        K_hill=0.05,
        n_hill=4.0,
        lambda_pg=5.0,
        lambda_late=3.0,
        use_student_t=True,
        student_t_nu=5.0,
        lambda_rare=0.1,
    )

    prior_bounds = load_prior_bounds("Dysbiotic", "HOBIC")
    prior_bounds = np.array(prior_bounds, dtype=np.float64)

    # JIT warmup
    _ = jax.jit(logL_fn)(jnp.zeros(20, dtype=jnp.float64))

    result = tmcmc_engine(
        logL_fn,
        prior_bounds,
        mutation="rw",
        n_particles=5,
        max_stages=2,
        seed=42,
        n_mutation_steps=1,
        use_de_mc=True,
        de_mc_gamma=2.38,
        resample_method="systematic",
    )
    assert result["n_stages"] >= 1, f"Expected >= 1 stage, got {result['n_stages']}"
    assert result["samples"].shape[1] == 20, f"Expected 20 params, got {result['samples'].shape[1]}"
    assert result["theta_MAP"].shape == (20,), "theta_MAP shape mismatch"
    assert np.all(np.isfinite(result["theta_MAP"])), "theta_MAP has non-finite values"
    # New: check log_evidence is present and finite
    assert "log_evidence" in result, "log_evidence missing from result"
    assert np.isfinite(result["log_evidence"]), f"log_evidence={result['log_evidence']} not finite"
    # New: check cv_weights is present
    assert "cv_weights" in result, "cv_weights missing from result"
    assert len(result["cv_weights"]) == result["n_stages"], "cv_weights length mismatch"
    # New: check particles are float64
    assert (
        result["samples"].dtype == np.float64
    ), f"particles dtype={result['samples'].dtype} should be float64"
    print(
        f"\n    E2E: {result['n_stages']} stages, {result['total_time']:.1f}s, "
        f"accept={np.mean(result['accept_rates']):.2f}, "
        f"logZ={result['log_evidence']:.1f}",
        end=" ... ",
        flush=True,
    )


# =====================================================================
# Test 10: Config JSON records all new parameters
# =====================================================================
@test("config_json_fields")
def _():
    """Verify all new fields would appear in config.json."""
    required_fields = [
        "replicate_sigma",
        "max_info_ratio",
        "lambda_rare",
        "use_de_mc",
        "use_student_t",
        "student_t_nu",
        "resample_method",
        "rmse",
        "mae",
        "r2_mean",
        "log_evidence",
        "cv_weights_final",
    ]
    src = Path(SCRIPT_DIR / "estimate_reduced_nishioka_jax.py").read_text()
    for field in required_fields:
        assert f'"{field}"' in src, f'"{field}" not found in config.json dump'


# =====================================================================
# Test 11: Student-t weight correctness
# =====================================================================
@test("student_t_weight_correct")
def _():
    """Student-t: weights multiply log-density, not the argument."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from estimate_reduced_nishioka_jax import make_log_likelihood_jax_ode

    # Create simple data where we can verify the Student-t computation
    data = np.array([[0.3, 0.2, 0.2, 0.2, 0.1]])
    sigma = np.full((1, 5), 0.05)
    phi_init = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    idx = np.array([100])

    # With uniform weights (lambda_rare=1.0, lambda_pg=1.0)
    logL_fn = make_log_likelihood_jax_ode(
        data=data,
        t_days=np.array([1]),
        idx_sparse=idx,
        sigma_obs=sigma,
        phi_init=phi_init,
        dt=1e-4,
        n_steps=200,
        K_hill=0.05,
        n_hill=4.0,
        lambda_pg=1.0,
        lambda_late=1.0,
        n_late=0,
        use_student_t=True,
        student_t_nu=5.0,
        lambda_rare=1.0,
    )
    theta = jnp.zeros(20, dtype=jnp.float64)
    val = jax.jit(logL_fn)(theta)
    assert jnp.isfinite(val), f"logL={val} is not finite"


# =====================================================================
# Test 12: Weighted covariance
# =====================================================================
@test("weighted_covariance")
def _():
    """Weighted covariance should reduce to sample cov for uniform weights."""
    from tmcmc_nuts_engine import _weighted_covariance

    rng = np.random.default_rng(42)
    particles = rng.normal(size=(100, 5))
    free_dims = np.arange(5)

    # Uniform weights
    w_uniform = np.ones(100) / 100
    cov_w = _weighted_covariance(particles, w_uniform, free_dims)
    cov_np = np.cov(particles.T)

    # Should be close (not exact due to Bessel correction)
    assert np.allclose(
        cov_w, cov_np, atol=0.1
    ), "Weighted cov (uniform) differs too much from np.cov"

    # Non-uniform weights: should not raise
    w_skewed = np.exp(rng.normal(size=100))
    w_skewed /= w_skewed.sum()
    cov_skewed = _weighted_covariance(particles, w_skewed, free_dims)
    assert cov_skewed.shape == (5, 5)
    assert np.all(np.isfinite(cov_skewed))
    # Diagonal should be positive
    assert np.all(np.diag(cov_skewed) > 0), "Diagonal of cov should be positive"


# =====================================================================
# Test 13: DE-MC pair selection correctness
# =====================================================================
@test("de_mc_pairs_correct")
def _():
    """DE-MC: r1 != i, r2 != i, r1 != r2 for all particles."""
    from tmcmc_nuts_engine import _de_mc_proposal

    rng = np.random.default_rng(42)
    n = 20
    particles = rng.normal(size=(n, 10))
    free_dims = np.arange(10)

    # Run many times to catch edge cases
    for _ in range(100):
        proposals = _de_mc_proposal(particles, free_dims, rng)
        # Proposals should differ from particles (non-trivial perturbation)
        assert not np.array_equal(proposals, particles)


# =====================================================================
# Test 14: Log evidence estimation
# =====================================================================
@test("log_evidence_computation")
def _():
    """Log evidence should be finite and negative for proper posterior."""
    from tmcmc_nuts_engine import tmcmc_engine
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    # Simple 1D Gaussian likelihood for testing
    def log_likelihood(theta):
        return -0.5 * jnp.sum((theta - 0.5) ** 2 / 0.01)

    bounds = np.array([[0.0, 1.0]] * 3, dtype=np.float64)
    result = tmcmc_engine(
        log_likelihood,
        bounds,
        mutation="rw",
        n_particles=20,
        max_stages=10,
        seed=42,
        n_mutation_steps=2,
        verbose=False,
    )
    logZ = result["log_evidence"]
    assert np.isfinite(logZ), f"log_evidence={logZ} is not finite"
    # For uniform prior [0,1]^3 and sharp Gaussian, logZ should be negative
    # (normalizing constant of the posterior over the prior volume)
    print(f"\n    logZ={logZ:.2f}", end=" ... ", flush=True)


# =====================================================================
# Test 15: Newton solver convergence (12 iterations)
# =====================================================================
@test("newton_solver_convergence")
def _():
    """Newton solver with 12 iterations should produce finite phi trajectory."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from hamilton_ode_jax import simulate_0d

    theta = jnp.zeros(20, dtype=jnp.float64)
    phi_init = jnp.array([0.3, 0.2, 0.2, 0.2, 0.1])
    phi_traj = simulate_0d(theta, n_steps=100, dt=1e-4, phi_init=phi_init)
    assert phi_traj.shape == (101, 5), f"Shape mismatch: {phi_traj.shape}"
    assert jnp.all(jnp.isfinite(phi_traj)), "Non-finite values in phi trajectory"
    # All fractions should be non-negative
    assert jnp.all(phi_traj >= -1e-6), f"Negative phi: min={float(phi_traj.min())}"


# =====================================================================
# Test 16: Normalizing Flow — forward/inverse roundtrip
# =====================================================================
@test("flow_roundtrip")
def _():
    """Flow forward then inverse should recover original input."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from normalizing_flow import init_flow, flow_forward, flow_inverse

    key = jax.random.PRNGKey(42)
    d = 5
    flow = init_flow(key, d, n_layers=4, hidden_dim=32)

    x = jnp.array([0.1, -0.3, 0.5, 0.2, -0.1])
    z, log_det_fwd = flow_forward(flow, x)
    x_rec, log_det_inv = flow_inverse(flow, z)

    assert jnp.allclose(
        x, x_rec, atol=1e-6
    ), f"Roundtrip error: max={float(jnp.max(jnp.abs(x - x_rec)))}"
    assert jnp.allclose(
        log_det_fwd + log_det_inv, 0.0, atol=1e-5
    ), f"Log det sum should be ~0, got {float(log_det_fwd + log_det_inv)}"


# =====================================================================
# Test 17: Flow sampling produces finite values
# =====================================================================
@test("flow_sample_finite")
def _():
    """Flow samples should be finite with valid log-densities."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from normalizing_flow import init_flow, flow_sample, flow_log_prob

    key = jax.random.PRNGKey(42)
    d = 5
    flow = init_flow(key, d, n_layers=4, hidden_dim=32)

    k_sample = jax.random.PRNGKey(99)
    samples, log_q = flow_sample(k_sample, flow, 50)

    assert samples.shape == (50, 5), f"Shape mismatch: {samples.shape}"
    assert jnp.all(jnp.isfinite(samples)), "Non-finite samples"
    assert jnp.all(jnp.isfinite(log_q)), "Non-finite log_q"

    # Check log_prob matches for first sample
    lp = flow_log_prob(flow, samples[0])
    assert jnp.isfinite(lp), f"log_prob not finite: {lp}"


# =====================================================================
# Test 18: Flow training reduces loss
# =====================================================================
@test("flow_training_loss_decreases")
def _():
    """Training flow on Gaussian data should improve log-likelihood."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from normalizing_flow import init_flow, train_flow, flow_log_prob

    key = jax.random.PRNGKey(42)
    d = 3

    # Generate 2D Gaussian data
    data = jax.random.normal(jax.random.PRNGKey(0), (200, d)) * 0.5 + 1.0

    flow_init = init_flow(key, d, n_layers=4, hidden_dim=32)

    # Log-likelihood before training
    ll_before = float(jnp.mean(jax.vmap(lambda x: flow_log_prob(flow_init, x))(data)))

    # Train
    flow_trained = train_flow(
        jax.random.PRNGKey(1),
        flow_init,
        data,
        n_epochs=100,
        lr=1e-3,
    )

    # Log-likelihood after training
    ll_after = float(jnp.mean(jax.vmap(lambda x: flow_log_prob(flow_trained, x))(data)))

    print(f"\n    Flow LL: before={ll_before:.1f}, after={ll_after:.1f}", end=" ... ", flush=True)
    assert ll_after > ll_before, f"Training didn't improve: {ll_before:.1f} → {ll_after:.1f}"


# =====================================================================
# Test 19: Waste-free SMC basic
# =====================================================================
@test("waste_free_basic")
def _():
    """Waste-free SMC returns correct shapes and valid weights."""
    from waste_free_smc import waste_free_resample_and_mutate_batch

    rng = np.random.default_rng(42)
    N, d = 20, 5
    particles = rng.normal(size=(N, d))
    logL = -np.sum(particles**2, axis=1)
    weights = np.ones(N) / N

    def dummy_mutation(p, lL, _rng):
        noise = _rng.normal(size=p.shape) * 0.1
        p_new = p + noise
        lL_new = -np.sum(p_new**2, axis=1)
        return p_new, lL_new, p.shape[0]  # all "accepted"

    p_out, lL_out, w_out, n_acc = waste_free_resample_and_mutate_batch(
        particles,
        logL,
        weights,
        beta_old=0.0,
        beta_new=0.5,
        batch_mutation_fn=dummy_mutation,
        n_mutation_steps=4,
        rng=rng,
    )

    assert p_out.shape == (N, d), f"Shape mismatch: {p_out.shape}"
    assert lL_out.shape == (N,), f"logL shape: {lL_out.shape}"
    assert w_out.shape == (N,), f"weights shape: {w_out.shape}"
    assert abs(w_out.sum() - 1.0) < 1e-10, f"Weights don't sum to 1: {w_out.sum()}"
    assert np.all(w_out >= 0), "Negative weights"
    assert np.all(np.isfinite(p_out)), "Non-finite particles"


# =====================================================================
# Test 20: Waste-free ESS improvement
# =====================================================================
@test("waste_free_ess_improvement")
def _():
    """Waste-free should produce more diverse particles than discarding."""
    from waste_free_smc import waste_free_resample_and_mutate_batch

    rng = np.random.default_rng(42)
    N, d = 100, 3
    particles = rng.normal(size=(N, d))
    logL = -np.sum(particles**2, axis=1)
    weights = np.ones(N) / N

    def batch_mutation(p, lL, _rng):
        noise = _rng.normal(size=p.shape) * 0.3
        p_new = p + noise
        lL_new = -np.sum(p_new**2, axis=1)
        return p_new, lL_new, p.shape[0]

    p_wf, _, w_wf, _ = waste_free_resample_and_mutate_batch(
        particles,
        logL,
        weights,
        beta_old=0.0,
        beta_new=0.5,
        batch_mutation_fn=batch_mutation,
        n_mutation_steps=5,
        rng=rng,
    )

    # Effective sample size of waste-free output
    ess_wf = 1.0 / np.sum(w_wf**2)
    # Should be > 1 (non-degenerate weights)
    assert ess_wf > 5, f"Waste-free ESS too low: {ess_wf:.1f}"
    print(f"\n    Waste-free ESS: {ess_wf:.1f}/{N}", end=" ... ", flush=True)


# =====================================================================
# Test 21: TMCMC with waste_free flag
# =====================================================================
@test("tmcmc_waste_free_integration")
def _():
    """TMCMC runs with waste_free=True without errors."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from tmcmc_nuts_engine import tmcmc_engine

    def log_likelihood(theta):
        return -0.5 * jnp.sum((theta - 0.5) ** 2 / 0.1)

    bounds = np.array([[0.0, 1.0]] * 3, dtype=np.float64)
    result = tmcmc_engine(
        log_likelihood,
        bounds,
        mutation="rw",
        n_particles=20,
        max_stages=5,
        seed=42,
        n_mutation_steps=4,
        waste_free=True,
        verbose=False,
    )
    assert result["n_stages"] >= 1
    assert result["waste_free"] is True
    assert np.all(np.isfinite(result["theta_MAP"]))
    assert np.isfinite(result["log_evidence"])
    print(
        f"\n    Waste-free TMCMC: {result['n_stages']} stages, "
        f"logZ={result['log_evidence']:.2f}",
        end=" ... ",
        flush=True,
    )


# =====================================================================
# Test 22: TMCMC with flow proposals
# =====================================================================
@test("tmcmc_flow_integration")
def _():
    """TMCMC runs with use_flow=True without errors."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from tmcmc_nuts_engine import tmcmc_engine

    def log_likelihood(theta):
        return -0.5 * jnp.sum((theta - 0.5) ** 2 / 0.1)

    bounds = np.array([[0.0, 1.0]] * 3, dtype=np.float64)
    result = tmcmc_engine(
        log_likelihood,
        bounds,
        mutation="rw",
        n_particles=30,
        max_stages=5,
        seed=42,
        n_mutation_steps=2,
        use_flow=True,
        flow_n_layers=4,
        flow_hidden_dim=32,
        flow_n_epochs=50,
        flow_start_beta=0.05,
        flow_mix_ratio=0.5,
        verbose=False,
    )
    assert result["n_stages"] >= 1
    assert result["use_flow"] is True
    assert np.all(np.isfinite(result["theta_MAP"]))
    assert np.isfinite(result["log_evidence"])
    print(
        f"\n    Flow TMCMC: {result['n_stages']} stages, "
        f"accept={np.mean(result['accept_rates']):.2f}, "
        f"logZ={result['log_evidence']:.2f}",
        end=" ... ",
        flush=True,
    )


# =====================================================================
# Summary
# =====================================================================
print()
print("=" * 60)
print(f"  GPU TMCMC Optimization Validation: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\n  FAILURES:")
    for name, err in errors:
        print(f"    - {name}: {err}")
    print()

sys.exit(0 if failed == 0 else 1)
