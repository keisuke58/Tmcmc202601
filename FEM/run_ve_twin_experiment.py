#!/usr/bin/env python3
"""
Viscoelastic Twin Experiment — Augmented Bayesian Inference
============================================================

Validates the VE-augmented TMCMC pipeline using synthetic mechanical
data, following Fritsch et al. (2025, arXiv:2512.15145) and
Nooranidoost et al. (2023).

Protocol:
  Phase 1: Fix ODE params (DI known), estimate VE params only (2D TMCMC)
           → proves identifiability from mechanical data
  Phase 2: Prediction uncertainty bounds for 4 conditions
           → quantifies VE error range
  Phase 3: Fisher information → optimal experimental design

Usage:
  python FEM/run_ve_twin_experiment.py
  python FEM/run_ve_twin_experiment.py --n-particles 2000 --seed 42

Output:
  FEM/figures/paper_final/fig29_ve_twin_experiment.png
  FEM/figures/paper_final/fig30_ve_prediction_bounds.png
  FEM/ve_twin_results.json

References:
  Fritsch et al. (2025) arXiv:2512.15145 — Bayesian updating + TSM
  Nooranidoost et al. (2023) — Bayesian VE estimation (P. aeruginosa)
  Shaw et al. (2004) PRL — universal 18-min relaxation time
  Towler et al. (2003) — rheometer creep, E0/Einf = 2-5
  Simo (1987) — exponential integrator algorithm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

# ═══════════════════════════════════════════════════════════════════════
# A. Constants & DI values from existing TMCMC runs
# ═══════════════════════════════════════════════════════════════════════

# Material model constants (from material_models.py)
E_MAX = 1000.0  # Pa, healthy biofilm
E_MIN = 10.0  # Pa, degraded biofilm
EXPONENT = 2.0  # power law
EPS_0 = 0.01  # step eigenstrain for stress relaxation

# VE constants
TAU_MAX = 60.0  # s, commensal
TAU_MIN = 2.0  # s, dysbiotic
RATIO_MIN = 2.0  # E0/Einf for commensal (r=0)
RATIO_MAX = 5.0  # E0/Einf for dysbiotic (r=1)
TAU_EXP = 1.5  # power-law exponent for tau(DI)

# DI values from MAP θ (4 conditions, from 3-model comparison)
# E(DI) = E_min + (E_max - E_min) * (1 - DI)^n, solved for DI
CONDITIONS = {
    "Commensal Static": {"abbrev": "CS", "E_map": 942.0, "color": "#2196F3"},
    "Commensal HOBIC": {"abbrev": "CH", "E_map": 995.0, "color": "#4CAF50"},
    "DH Baseline": {"abbrev": "DH", "E_map": 279.0, "color": "#FF9800"},
    "Dysbiotic Static": {"abbrev": "DS", "E_map": 32.0, "color": "#F44336"},
}

# Compute DI from E for each condition
for info in CONDITIONS.values():
    E = info["E_map"]
    # E = E_min + (E_max - E_min) * (1-DI)^n
    ratio = (E - E_MIN) / (E_MAX - E_MIN)
    info["DI"] = 1.0 - ratio ** (1.0 / EXPONENT)


# ═══════════════════════════════════════════════════════════════════════
# B. SLS Material Model
# ═══════════════════════════════════════════════════════════════════════


def compute_ve_params(di, log_tau=None, e0_einf_ratio=None):
    """
    DI + optional VE params → full SLS parameter set.

    If log_tau and e0_einf_ratio are provided, they override the
    default DI-dependent tau and ratio.
    """
    r = np.clip(di, 0.0, 1.0)

    E_inf = E_MAX * (1.0 - r) ** EXPONENT + E_MIN * r

    if e0_einf_ratio is not None:
        ratio = e0_einf_ratio
    else:
        ratio = RATIO_MIN + (RATIO_MAX - RATIO_MIN) * r

    E_0 = E_inf * ratio
    E_1 = E_0 - E_inf

    if log_tau is not None:
        tau = 10.0**log_tau
    else:
        tau = TAU_MAX * (1.0 - r) ** TAU_EXP + TAU_MIN * r

    eta = E_1 * tau

    return {"E_inf": E_inf, "E_0": E_0, "E_1": E_1, "tau": tau, "eta": eta}


def sls_stress_relaxation(E_inf, E_1, tau, eps_0, t):
    """σ(t) = [E_inf + E_1·exp(-t/τ)] · ε₀"""
    return (E_inf + E_1 * np.exp(-t / tau)) * eps_0


def sls_creep_displacement(E_inf, E_1, tau, sigma_0, L, t):
    """
    Creep displacement u(t) = σ₀ · J(t) · L for SLS.

    J(t) = 1/E_inf - E_1/(E_inf·E_0) · exp(-t/τ_retard)
    τ_retard = E_1·τ / E_inf
    """
    E_0 = E_inf + E_1
    tau_retard = E_1 * tau / E_inf
    J = 1.0 / E_inf - E_1 / (E_inf * E_0) * np.exp(-t / tau_retard)
    return sigma_0 * J * L


# ═══════════════════════════════════════════════════════════════════════
# C. TMCMC Implementation (Ching & Chen 2007)
# ═══════════════════════════════════════════════════════════════════════


def _find_delta_beta(logL, target_cov=1.0, max_delta=1.0):
    """Find Δβ s.t. CoV of importance weights ≈ target_cov."""
    lo, hi = 1e-8, max_delta
    for _ in range(60):
        mid = (lo + hi) / 2.0
        log_w = mid * logL
        log_w = log_w - log_w.max()
        w = np.exp(log_w)
        mean_w = w.mean()
        if mean_w < 1e-30:
            hi = mid
            continue
        cov = w.std() / mean_w
        if cov > target_cov:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def run_tmcmc(
    log_likelihood_fn,
    bounds,
    n_particles=1000,
    n_mutations=5,
    cov_scale=0.04,
    seed=42,
    verbose=True,
):
    """
    Standard TMCMC (Ching & Chen 2007).

    Parameters
    ----------
    log_likelihood_fn : callable(theta) -> float
    bounds : list of (lo, hi) — parameter bounds
    n_particles : int
    n_mutations : int — MCMC steps per stage
    cov_scale : float — proposal covariance scaling (β² = 0.04)
    seed : int
    verbose : bool

    Returns
    -------
    dict with samples, logL, MAP, beta_schedule, acceptance_rate
    """
    rng = np.random.default_rng(seed)
    n_dim = len(bounds)
    bounds_arr = np.array(bounds)

    # Initialize from prior (uniform)
    samples = np.column_stack([rng.uniform(lo, hi, n_particles) for lo, hi in bounds])

    # Evaluate log-likelihoods
    logL = np.array([log_likelihood_fn(s) for s in samples])
    logL = np.where(np.isfinite(logL), logL, -1e30)

    beta = 0.0
    beta_schedule = [0.0]
    n_accepted = 0
    n_proposed = 0
    stage = 0

    while beta < 1.0 - 1e-10:
        # Find delta beta
        delta_beta = _find_delta_beta(logL - logL.max(), target_cov=1.0, max_delta=1.0 - beta)
        beta_new = min(beta + delta_beta, 1.0)
        delta = beta_new - beta

        # Importance weights
        log_w = delta * (logL - logL.max())
        w = np.exp(log_w)
        w /= w.sum()

        # ESS
        ess = 1.0 / np.sum(w**2)

        # Weighted covariance
        mean = np.average(samples, weights=w, axis=0)
        diff = samples - mean
        cov = np.zeros((n_dim, n_dim))
        for k in range(n_particles):
            cov += w[k] * np.outer(diff[k], diff[k])
        cov_proposal = cov_scale * cov

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(cov_proposal)
        if eigvals.min() < 1e-15:
            cov_proposal += np.eye(n_dim) * 1e-10

        # Resample
        indices = rng.choice(n_particles, size=n_particles, p=w)
        samples = samples[indices].copy()
        logL = logL[indices].copy()

        # MCMC perturbation
        for i in range(n_particles):
            for _ in range(n_mutations):
                proposal = rng.multivariate_normal(samples[i], cov_proposal)
                n_proposed += 1

                # Check bounds
                in_bounds = np.all((proposal >= bounds_arr[:, 0]) & (proposal <= bounds_arr[:, 1]))
                if not in_bounds:
                    continue

                logL_prop = log_likelihood_fn(proposal)
                if not np.isfinite(logL_prop):
                    continue

                # MH acceptance (at current beta)
                log_alpha = beta_new * (logL_prop - logL[i])
                if np.log(rng.random()) < log_alpha:
                    samples[i] = proposal
                    logL[i] = logL_prop
                    n_accepted += 1

        beta = beta_new
        beta_schedule.append(beta)
        stage += 1

        if verbose:
            acc_rate = n_accepted / max(n_proposed, 1)
            print(
                f"  Stage {stage:2d}: β={beta:.4f}, ESS={ess:.0f}, "
                f"acc={acc_rate:.3f}, logL_max={logL.max():.2f}"
            )

    i_map = np.argmax(logL)
    return {
        "samples": samples,
        "logL": logL,
        "MAP": samples[i_map],
        "logL_MAP": logL[i_map],
        "beta_schedule": beta_schedule,
        "acceptance_rate": n_accepted / max(n_proposed, 1),
        "n_stages": stage,
    }


# ═══════════════════════════════════════════════════════════════════════
# D. Phase 1: VE Identifiability Proof (2D TMCMC)
# ═══════════════════════════════════════════════════════════════════════


def run_phase1(n_particles=1000, seed=42, sigma_noise_frac=0.05):
    """
    Estimate θ[20:21] from synthetic stress relaxation data
    with DI fixed at DH baseline value.

    Shows: VE params are identifiable from mechanical data.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: VE Parameter Identifiability (2D TMCMC)")
    print("=" * 60)

    # Ground truth
    DI_true = CONDITIONS["DH Baseline"]["DI"]
    theta_true = np.array([1.2, 3.5])  # [log10(tau), E0/Einf ratio]
    # → tau = 10^1.2 ≈ 15.8 s, E0/Einf = 3.5

    print("\nGround truth:")
    print(f"  DI = {DI_true:.4f}")
    print(f"  log10(τ) = {theta_true[0]:.2f} → τ = {10**theta_true[0]:.1f} s")
    print(f"  E₀/E∞ = {theta_true[1]:.2f}")

    # Compute VE params at ground truth
    ve_true = compute_ve_params(DI_true, theta_true[0], theta_true[1])
    print(f"  E∞ = {ve_true['E_inf']:.1f} Pa")
    print(f"  E₀ = {ve_true['E_0']:.1f} Pa")
    print(f"  E₁ = {ve_true['E_1']:.1f} Pa")
    print(f"  τ  = {ve_true['tau']:.1f} s")
    print(f"  η  = {ve_true['eta']:.0f} Pa·s")

    # Generate synthetic stress relaxation data
    t_obs = np.array([0.5, 1, 2, 5, 10, 20, 30, 50, 80, 120, 180, 240, 300, 400, 500, 600])
    sigma_true = sls_stress_relaxation(
        ve_true["E_inf"], ve_true["E_1"], ve_true["tau"], EPS_0, t_obs
    )

    # Add Gaussian noise
    rng = np.random.default_rng(seed + 100)
    sigma_noise = sigma_noise_frac * sigma_true.max()
    sigma_obs = sigma_true + rng.normal(0, sigma_noise, len(t_obs))

    print(
        f"\nSynthetic data: {len(t_obs)} time points, "
        f"σ_noise = {sigma_noise:.3f} Pa ({sigma_noise_frac*100:.0f}% of σ_max)"
    )

    # ── TMCMC without mechanical data (prior only) ──
    print("\n--- Run A: Prior only (no mechanical data) ---")

    def logL_prior_only(theta):
        """Only literature-informed Gaussian prior (like current pipeline)."""
        mu = np.array([1.0, 3.0])
        sig = np.array([0.5, 1.0])
        return -0.5 * np.sum(((theta - mu) / sig) ** 2)

    bounds = [(0.0, 2.0), (1.5, 6.0)]
    result_prior = run_tmcmc(
        logL_prior_only, bounds, n_particles=n_particles, n_mutations=3, seed=seed, verbose=False
    )
    print(f"  MAP: log₁₀(τ)={result_prior['MAP'][0]:.3f}, " f"E₀/E∞={result_prior['MAP'][1]:.3f}")
    print(
        f"  Stages: {result_prior['n_stages']}, "
        f"Acceptance: {result_prior['acceptance_rate']:.3f}"
    )

    # ── TMCMC with mechanical data ──
    print("\n--- Run B: Augmented with mechanical data ---")

    def logL_mechanical(theta):
        """Likelihood from stress relaxation data."""
        log_tau, ratio = theta
        ve = compute_ve_params(DI_true, log_tau, ratio)
        sigma_pred = sls_stress_relaxation(ve["E_inf"], ve["E_1"], ve["tau"], EPS_0, t_obs)
        return -0.5 * np.sum(((sigma_obs - sigma_pred) / sigma_noise) ** 2)

    result_mech = run_tmcmc(
        logL_mechanical, bounds, n_particles=n_particles, n_mutations=5, seed=seed
    )
    print(f"\n  MAP: log₁₀(τ)={result_mech['MAP'][0]:.3f} " f"(true: {theta_true[0]:.3f})")
    print(f"  MAP: E₀/E∞={result_mech['MAP'][1]:.3f} " f"(true: {theta_true[1]:.3f})")
    print(
        f"  Stages: {result_mech['n_stages']}, " f"Acceptance: {result_mech['acceptance_rate']:.3f}"
    )

    # ── TMCMC with both (prior + mechanical) ──
    print("\n--- Run C: Prior + Mechanical data ---")

    def logL_augmented(theta):
        """Combined prior penalty + mechanical likelihood."""
        return logL_prior_only(theta) + logL_mechanical(theta)

    result_aug = run_tmcmc(
        logL_augmented, bounds, n_particles=n_particles, n_mutations=5, seed=seed + 1
    )
    print(f"\n  MAP: log₁₀(τ)={result_aug['MAP'][0]:.3f} " f"(true: {theta_true[0]:.3f})")
    print(f"  MAP: E₀/E∞={result_aug['MAP'][1]:.3f} " f"(true: {theta_true[1]:.3f})")

    # Recovery statistics
    samples = result_mech["samples"]
    for i, (name, true_val) in enumerate([("log₁₀(τ)", theta_true[0]), ("E₀/E∞", theta_true[1])]):
        med = np.median(samples[:, i])
        q05, q95 = np.percentile(samples[:, i], [5, 95])
        bias = med - true_val
        print(
            f"\n  {name}: median={med:.3f}, 90% CI=[{q05:.3f}, {q95:.3f}], "
            f"bias={bias:.4f}, truth={true_val:.3f}"
        )
        in_ci = q05 <= true_val <= q95
        print(f"    Truth in 90% CI: {'YES' if in_ci else 'NO'}")

    return {
        "theta_true": theta_true.tolist(),
        "DI_true": DI_true,
        "t_obs": t_obs.tolist(),
        "sigma_obs": sigma_obs.tolist(),
        "sigma_true": sigma_true.tolist(),
        "sigma_noise": sigma_noise,
        "result_prior": {
            "samples": result_prior["samples"].tolist(),
            "MAP": result_prior["MAP"].tolist(),
        },
        "result_mech": {
            "samples": result_mech["samples"].tolist(),
            "MAP": result_mech["MAP"].tolist(),
            "logL_MAP": float(result_mech["logL_MAP"]),
            "acceptance_rate": result_mech["acceptance_rate"],
            "n_stages": result_mech["n_stages"],
            "beta_schedule": result_mech["beta_schedule"],
        },
        "result_aug": {
            "samples": result_aug["samples"].tolist(),
            "MAP": result_aug["MAP"].tolist(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# E. Phase 2: Prediction Uncertainty Bounds
# ═══════════════════════════════════════════════════════════════════════


def run_phase2(n_mc=5000, seed=42):
    """
    Monte Carlo propagation of VE parameter uncertainty through SLS.

    Samples (τ, E₀/E∞) from literature ranges and computes
    σ(t) and u(t) uncertainty bands for 4 conditions.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Prediction Uncertainty Bounds")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    t_eval = np.linspace(0.1, 600, 200)

    # Literature-informed parameter distributions
    # Shaw 2004: τ ∈ [1, 100] s → log10(τ) ∈ [0, 2]
    # Towler 2003: E₀/E∞ ∈ [2, 5]
    log_tau_samples = rng.uniform(0.0, 2.0, n_mc)
    ratio_samples = rng.uniform(2.0, 5.0, n_mc)

    results = {}

    for cond_name, info in CONDITIONS.items():
        DI = info["DI"]
        abbrev = info["abbrev"]

        # Stress relaxation ensemble
        sigma_ensemble = np.zeros((n_mc, len(t_eval)))
        # Creep displacement ensemble
        u_ensemble = np.zeros((n_mc, len(t_eval)))

        for k in range(n_mc):
            ve = compute_ve_params(DI, log_tau_samples[k], ratio_samples[k])
            sigma_ensemble[k] = sls_stress_relaxation(
                ve["E_inf"], ve["E_1"], ve["tau"], EPS_0, t_eval
            )
            u_ensemble[k] = (
                sls_creep_displacement(
                    ve["E_inf"],
                    ve["E_1"],
                    ve["tau"],
                    sigma_0=100.0,
                    L=0.2e-3,
                    t=t_eval,  # 100 Pa GCF, 0.2mm thickness
                )
                * 1e6
            )  # convert to μm

        # Elastic reference (no VE, just E_inf)
        ve_default = compute_ve_params(DI)
        sigma_elastic = ve_default["E_inf"] * EPS_0 * np.ones_like(t_eval)
        u_elastic = (100.0 / ve_default["E_inf"] * 0.2e-3) * 1e6

        # Statistics
        sigma_med = np.median(sigma_ensemble, axis=0)
        sigma_q05 = np.percentile(sigma_ensemble, 2.5, axis=0)
        sigma_q95 = np.percentile(sigma_ensemble, 97.5, axis=0)

        u_med = np.median(u_ensemble, axis=0)
        u_q05 = np.percentile(u_ensemble, 2.5, axis=0)
        u_q95 = np.percentile(u_ensemble, 97.5, axis=0)

        # Max VE error (relative to elastic)
        max_over = sigma_q95[0] / sigma_elastic[0] - 1  # at t=0
        max_under = 1 - sigma_q05[-1] / sigma_elastic[-1]  # at t=∞

        print(f"\n  {cond_name} ({abbrev}): DI={DI:.3f}, E∞={ve_default['E_inf']:.1f} Pa")
        print(
            f"    σ(t=0): [{sigma_q05[0]:.2f}, {sigma_q95[0]:.2f}] Pa "
            f"(elastic: {sigma_elastic[0]:.2f} Pa)"
        )
        print(f"    σ(t=600s): [{sigma_q05[-1]:.2f}, {sigma_q95[-1]:.2f}] Pa")
        print(f"    Max VE overshoot at t=0: +{max_over*100:.0f}%")
        print(f"    Max VE relaxation at t→∞: −{max_under*100:.0f}%")

        results[abbrev] = {
            "DI": DI,
            "E_inf": ve_default["E_inf"],
            "sigma_med": sigma_med.tolist(),
            "sigma_q05": sigma_q05.tolist(),
            "sigma_q95": sigma_q95.tolist(),
            "u_med": u_med.tolist(),
            "u_q05": u_q05.tolist(),
            "u_q95": u_q95.tolist(),
            "sigma_elastic": float(sigma_elastic[0]),
            "u_elastic": float(u_elastic),
        }

    results["t_eval"] = t_eval.tolist()
    return results


# ═══════════════════════════════════════════════════════════════════════
# F. Phase 3: Fisher Information & Optimal Experimental Design
# ═══════════════════════════════════════════════════════════════════════


def run_phase3(DI=None, sigma_noise=0.1):
    """
    Compute Fisher information matrix for SLS parameters.

    Determines which measurement times maximize parameter information.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Fisher Information & OED")
    print("=" * 60)

    if DI is None:
        DI = CONDITIONS["DH Baseline"]["DI"]

    # Evaluate at default VE params
    ve = compute_ve_params(DI)
    E_inf, E_1, tau = ve["E_inf"], ve["E_1"], ve["tau"]

    # Dense time grid
    t = np.linspace(0.1, 600, 1000)

    # Analytical Jacobian of σ(t) w.r.t. [log10(τ), E₀/E∞]
    # σ(t) = [E_inf + E_1·exp(-t/τ)] · ε₀
    #
    # θ₁ = log10(τ)  → τ = 10^θ₁
    #   ∂σ/∂θ₁ = E_1 · (t/τ) · exp(-t/τ) · ln(10) · ε₀
    #
    # θ₂ = E₀/E∞  → E_1 = E_inf · (θ₂ - 1)
    #   ∂σ/∂θ₂ = E_inf · exp(-t/τ) · ε₀

    dsigma_dtheta1 = E_1 * (t / tau) * np.exp(-t / tau) * np.log(10) * EPS_0
    dsigma_dtheta2 = E_inf * np.exp(-t / tau) * EPS_0

    # Fisher information per time point: F(t) = J(t)^T J(t) / σ_noise²
    # Total Fisher: F = Σ_k F(t_k)

    # Find optimal measurement times (D-optimal: maximize det(F))
    # For 2 params, we need at least 2 time points

    # Information density per time point
    info_density_1 = dsigma_dtheta1**2 / sigma_noise**2  # info about τ
    info_density_2 = dsigma_dtheta2**2 / sigma_noise**2  # info about ratio

    # Optimal: maximize det(F) = F11*F22 - F12²
    # This is achieved by spreading measurements at different times

    # Find peak information times
    t_opt_tau = t[np.argmax(info_density_1)]
    t_opt_ratio = t[np.argmax(info_density_2)]

    print(f"\n  DI = {DI:.4f}, E∞ = {E_inf:.1f} Pa, τ = {tau:.1f} s")
    print(f"\n  Peak info for τ (relaxation time): t* = {t_opt_tau:.1f} s")
    print(f"  Peak info for E₀/E∞ (ratio):       t* = {t_opt_ratio:.1f} s")

    # Compute Fisher matrix for different measurement protocols
    protocols = {
        "Short only (t < 10s)": t[t < 10],
        "Long only (t > 100s)": t[t > 100],
        "Uniform (16 pts)": np.linspace(0.5, 600, 16),
        "Log-spaced (16 pts)": np.geomspace(0.5, 600, 16),
        "D-optimal (2 pts)": np.array([t_opt_tau, max(t_opt_ratio, 0.5)]),
    }

    print(f"\n  {'Protocol':<28s}  {'det(F)':>10s}  {'σ(τ)':>8s}  {'σ(ratio)':>8s}")
    print("  " + "-" * 60)

    fisher_results = {}
    for name, t_k in protocols.items():
        J1 = E_1 * (t_k / tau) * np.exp(-t_k / tau) * np.log(10) * EPS_0
        J2 = E_inf * np.exp(-t_k / tau) * EPS_0

        F = (
            np.array(
                [
                    [np.sum(J1 * J1), np.sum(J1 * J2)],
                    [np.sum(J1 * J2), np.sum(J2 * J2)],
                ]
            )
            / sigma_noise**2
        )

        det_F = np.linalg.det(F)
        if det_F > 0:
            F_inv = np.linalg.inv(F)
            sigma_tau = np.sqrt(F_inv[0, 0])
            sigma_ratio = np.sqrt(F_inv[1, 1])
        else:
            sigma_tau = sigma_ratio = float("inf")

        print(f"  {name:<28s}  {det_F:10.2e}  {sigma_tau:8.4f}  {sigma_ratio:8.4f}")

        fisher_results[name] = {
            "det_F": float(det_F),
            "sigma_tau": float(sigma_tau),
            "sigma_ratio": float(sigma_ratio),
            "n_points": len(t_k),
        }

    return {
        "DI": DI,
        "t_opt_tau": float(t_opt_tau),
        "t_opt_ratio": float(t_opt_ratio),
        "info_density_tau": info_density_1.tolist(),
        "info_density_ratio": info_density_2.tolist(),
        "t_dense": t.tolist(),
        "protocols": fisher_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# G. Figure Generation
# ═══════════════════════════════════════════════════════════════════════


def generate_fig29(phase1_results, outdir):
    """
    Fig 29: Twin Experiment — VE Parameter Identifiability.

    4 panels:
      (a) Synthetic σ(t) data + posterior predictive
      (b) Prior vs posterior: log₁₀(τ)
      (c) Prior vs posterior: E₀/E∞
      (d) 2D posterior scatter with truth
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    theta_true = np.array(phase1_results["theta_true"])
    t_obs = np.array(phase1_results["t_obs"])
    sigma_obs = np.array(phase1_results["sigma_obs"])
    sigma_true = np.array(phase1_results["sigma_true"])
    samples_prior = np.array(phase1_results["result_prior"]["samples"])
    samples_mech = np.array(phase1_results["result_mech"]["samples"])
    samples_aug = np.array(phase1_results["result_aug"]["samples"])
    DI_true = phase1_results["DI_true"]

    # ── (a) Stress relaxation data + posterior predictive ──
    ax = axes[0, 0]
    t_fine = np.linspace(0.1, 700, 300)

    # Posterior predictive (random subset of posterior samples)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(samples_mech), min(200, len(samples_mech)), replace=False)
    for k in idx:
        ve = compute_ve_params(DI_true, samples_mech[k, 0], samples_mech[k, 1])
        sigma_pred = sls_stress_relaxation(ve["E_inf"], ve["E_1"], ve["tau"], EPS_0, t_fine)
        ax.plot(t_fine, sigma_pred, color="#2196F3", alpha=0.03, lw=0.5)

    # True curve
    ve_true = compute_ve_params(DI_true, theta_true[0], theta_true[1])
    sigma_true_fine = sls_stress_relaxation(
        ve_true["E_inf"], ve_true["E_1"], ve_true["tau"], EPS_0, t_fine
    )
    ax.plot(t_fine, sigma_true_fine, "k-", lw=1.5, label="True $\\sigma(t)$")

    # Elastic reference
    E_inf = ve_true["E_inf"]
    ax.axhline(
        E_inf * EPS_0,
        color="gray",
        ls="--",
        lw=1,
        alpha=0.5,
        label=f"Elastic $E_\\infty$·ε₀ = {E_inf*EPS_0:.2f} Pa",
    )

    # Data points
    ax.scatter(t_obs, sigma_obs, c="red", s=25, zorder=5, label=f"Synthetic data (n={len(t_obs)})")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Stress σ(t) [Pa]")
    ax.set_title("(a) Stress relaxation: data + posterior predictive")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 700)

    # ── (b) Posterior: log₁₀(τ) ──
    ax = axes[0, 1]
    bins = np.linspace(0.0, 2.0, 50)

    ax.hist(
        samples_prior[:, 0], bins=bins, density=True, alpha=0.3, color="gray", label="Prior only"
    )
    ax.hist(
        samples_mech[:, 0],
        bins=bins,
        density=True,
        alpha=0.6,
        color="#2196F3",
        label="With mech. data",
    )

    ax.axvline(theta_true[0], color="red", ls="--", lw=2, label=f"Truth = {theta_true[0]:.2f}")
    ax.axvline(
        samples_mech[:, 0].mean(),
        color="#2196F3",
        ls=":",
        lw=1.5,
        label=f"Mean = {samples_mech[:, 0].mean():.3f}",
    )

    ax.set_xlabel("log₁₀(τ / 1s)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Posterior: relaxation time τ")
    ax.legend(fontsize=7)

    # ── (c) Posterior: E₀/E∞ ──
    ax = axes[1, 0]
    bins = np.linspace(1.5, 6.0, 50)

    ax.hist(
        samples_prior[:, 1], bins=bins, density=True, alpha=0.3, color="gray", label="Prior only"
    )
    ax.hist(
        samples_mech[:, 1],
        bins=bins,
        density=True,
        alpha=0.6,
        color="#FF9800",
        label="With mech. data",
    )

    ax.axvline(theta_true[1], color="red", ls="--", lw=2, label=f"Truth = {theta_true[1]:.2f}")
    ax.axvline(
        samples_mech[:, 1].mean(),
        color="#FF9800",
        ls=":",
        lw=1.5,
        label=f"Mean = {samples_mech[:, 1].mean():.3f}",
    )

    ax.set_xlabel("E₀ / E∞")
    ax.set_ylabel("Density")
    ax.set_title("(c) Posterior: modulus ratio E₀/E∞")
    ax.legend(fontsize=7)

    # ── (d) 2D scatter: prior vs posterior ──
    ax = axes[1, 1]

    ax.scatter(
        samples_prior[:, 0], samples_prior[:, 1], c="gray", s=3, alpha=0.2, label="Prior only"
    )
    ax.scatter(
        samples_mech[:, 0], samples_mech[:, 1], c="#2196F3", s=3, alpha=0.3, label="With mech. data"
    )
    ax.scatter(
        samples_aug[:, 0], samples_aug[:, 1], c="#4CAF50", s=3, alpha=0.3, label="Prior + mech."
    )

    ax.plot(theta_true[0], theta_true[1], "r*", ms=15, zorder=10, label="Truth")

    ax.set_xlabel("log₁₀(τ / 1s)")
    ax.set_ylabel("E₀ / E∞")
    ax.set_title("(d) 2D posterior: identifiability proof")
    ax.legend(fontsize=7, markerscale=3)

    fig.suptitle(
        "Fig. 29: VE Twin Experiment — Synthetic Data Recovery\n"
        f"DI = {DI_true:.3f} (DH baseline), "
        f"N = {len(samples_mech)} particles, "
        f"σ_noise = {phase1_results['sigma_noise']:.3f} Pa",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = os.path.join(outdir, "fig29_ve_twin_experiment.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {outpath}")
    return outpath


def generate_fig30(phase2_results, phase3_results, outdir):
    """
    Fig 30: Prediction Uncertainty Bounds + Fisher Information.

    4 panels:
      (a) σ(t) with 95% CI for 4 conditions
      (b) Creep u(t) with 95% CI for 4 conditions
      (c) Elastic vs VE error range (bar chart)
      (d) Fisher information density
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    t_eval = np.array(phase2_results["t_eval"])

    # ── (a) Stress relaxation uncertainty ──
    ax = axes[0, 0]
    for cond_name, info in CONDITIONS.items():
        abbrev = info["abbrev"]
        color = info["color"]
        r = phase2_results[abbrev]

        sigma_med = np.array(r["sigma_med"])
        sigma_q05 = np.array(r["sigma_q05"])
        sigma_q95 = np.array(r["sigma_q95"])

        ax.fill_between(t_eval, sigma_q05, sigma_q95, color=color, alpha=0.2)
        ax.plot(t_eval, sigma_med, color=color, lw=1.5, label=f"{abbrev} (E∞={r['E_inf']:.0f} Pa)")

        # Elastic reference
        ax.axhline(r["sigma_elastic"], color=color, ls=":", lw=0.8, alpha=0.3)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("σ(t) [Pa]")
    ax.set_title("(a) Stress relaxation ε₀=0.01, 95% CI")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 600)

    # ── (b) Creep displacement uncertainty ──
    ax = axes[0, 1]
    for cond_name, info in CONDITIONS.items():
        abbrev = info["abbrev"]
        color = info["color"]
        r = phase2_results[abbrev]

        u_med = np.array(r["u_med"])
        u_q05 = np.array(r["u_q05"])
        u_q95 = np.array(r["u_q95"])

        ax.fill_between(t_eval, u_q05, u_q95, color=color, alpha=0.2)
        ax.plot(t_eval, u_med, color=color, lw=1.5, label=abbrev)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [μm]")
    ax.set_title("(b) Creep under 100 Pa GCF, L=0.2mm, 95% CI")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 600)
    ax.set_yscale("log")

    # ── (c) Elastic vs VE error range ──
    ax = axes[1, 0]
    abbrevs = [info["abbrev"] for info in CONDITIONS.values()]
    colors = [info["color"] for info in CONDITIONS.values()]

    # At t=0 (instantaneous) and t=600s (near-equilibrium)
    x = np.arange(len(abbrevs))
    width = 0.35

    sigma_elastic_vals = []
    sigma_ve_t0_lo = []
    sigma_ve_t0_hi = []
    sigma_ve_t600_lo = []
    sigma_ve_t600_hi = []

    for abbrev in abbrevs:
        r = phase2_results[abbrev]
        sigma_elastic_vals.append(r["sigma_elastic"])
        sigma_ve_t0_lo.append(np.array(r["sigma_q05"])[0])
        sigma_ve_t0_hi.append(np.array(r["sigma_q95"])[0])
        sigma_ve_t600_lo.append(np.array(r["sigma_q05"])[-1])
        sigma_ve_t600_hi.append(np.array(r["sigma_q95"])[-1])

    # Normalize to elastic
    for i in range(len(abbrevs)):
        e_ref = sigma_elastic_vals[i]
        if e_ref > 0:
            lo_t0 = (sigma_ve_t0_lo[i] / e_ref - 1) * 100
            hi_t0 = (sigma_ve_t0_hi[i] / e_ref - 1) * 100
            lo_t600 = (sigma_ve_t600_lo[i] / e_ref - 1) * 100
            hi_t600 = (sigma_ve_t600_hi[i] / e_ref - 1) * 100

            ax.bar(
                x[i] - width / 2,
                hi_t0 - lo_t0,
                width,
                bottom=lo_t0,
                color=colors[i],
                alpha=0.5,
                label="t=0" if i == 0 else "",
            )
            ax.bar(
                x[i] + width / 2,
                hi_t600 - lo_t600,
                width,
                bottom=lo_t600,
                color=colors[i],
                alpha=0.9,
                label="t=600s" if i == 0 else "",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(abbrevs)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Deviation from elastic [%]")
    ax.set_title("(c) VE deviation from elastic reference")
    ax.legend(fontsize=7)

    # ── (d) Fisher information density ──
    ax = axes[1, 1]
    t_dense = np.array(phase3_results["t_dense"])
    info_tau = np.array(phase3_results["info_density_tau"])
    info_ratio = np.array(phase3_results["info_density_ratio"])

    # Normalize to peak
    info_tau_norm = info_tau / (info_tau.max() + 1e-30)
    info_ratio_norm = info_ratio / (info_ratio.max() + 1e-30)

    ax.plot(
        t_dense,
        info_tau_norm,
        color="#2196F3",
        lw=1.5,
        label=f"τ (peak at t*={phase3_results['t_opt_tau']:.1f} s)",
    )
    ax.plot(
        t_dense,
        info_ratio_norm,
        color="#FF9800",
        lw=1.5,
        label=f"E₀/E∞ (peak at t*={phase3_results['t_opt_ratio']:.1f} s)",
    )

    ax.axvline(phase3_results["t_opt_tau"], color="#2196F3", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(phase3_results["t_opt_ratio"], color="#FF9800", ls="--", lw=0.8, alpha=0.5)

    ax.set_xlabel("Measurement time [s]")
    ax.set_ylabel("Normalized Fisher information")
    ax.set_title("(d) Optimal measurement times (OED)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 600)

    fig.suptitle(
        "Fig. 30: VE Prediction Bounds & Optimal Experimental Design\n"
        "Literature ranges: τ ∈ [1, 100] s (Shaw 2004), "
        "E₀/E∞ ∈ [2, 5] (Towler 2003)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = os.path.join(outdir, "fig30_ve_prediction_bounds.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {outpath}")
    return outpath


# ═══════════════════════════════════════════════════════════════════════
# H. Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Viscoelastic Twin Experiment")
    parser.add_argument(
        "--n-particles", type=int, default=1000, help="TMCMC particles (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-frac",
        type=float,
        default=0.05,
        help="Noise fraction for synthetic data (default: 0.05)",
    )
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    args = parser.parse_args()

    # Output directory
    outdir = os.path.join(os.path.dirname(__file__), "figures", "paper_final")
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    all_results = {}

    # ── Phase 1: VE Identifiability ──
    if not args.skip_phase1:
        phase1 = run_phase1(
            n_particles=args.n_particles,
            seed=args.seed,
            sigma_noise_frac=args.noise_frac,
        )
        all_results["phase1"] = phase1
        fig29_path = generate_fig29(phase1, outdir)
        all_results["fig29"] = fig29_path

    # ── Phase 2: Prediction Bounds ──
    if not args.skip_phase2:
        phase2 = run_phase2(n_mc=5000, seed=args.seed)
        all_results["phase2"] = phase2

    # ── Phase 3: Fisher Information ──
    if not args.skip_phase3:
        phase3 = run_phase3(sigma_noise=0.1)
        all_results["phase3"] = phase3

    # ── Fig 30 (needs Phase 2 + 3) ──
    if not args.skip_phase2 and not args.skip_phase3:
        fig30_path = generate_fig30(phase2, phase3, outdir)
        all_results["fig30"] = fig30_path

    # ── Save results JSON ──
    json_path = os.path.join(os.path.dirname(__file__), "ve_twin_results.json")

    # Convert numpy to lists for JSON serialization
    def _sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    # Save only metadata (not full sample arrays) to keep JSON small
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_particles": args.n_particles,
        "seed": args.seed,
        "noise_frac": args.noise_frac,
        "elapsed_s": time.time() - t0,
        "conditions": {k: {"DI": v["DI"], "E_map": v["E_map"]} for k, v in CONDITIONS.items()},
    }

    if "phase1" in all_results:
        p1 = all_results["phase1"]
        meta["phase1"] = {
            "theta_true": p1["theta_true"],
            "DI_true": p1["DI_true"],
            "MAP_mech": p1["result_mech"]["MAP"],
            "acceptance_rate": p1["result_mech"]["acceptance_rate"],
            "n_stages": p1["result_mech"]["n_stages"],
            "bias_log_tau": float(
                np.median(np.array(p1["result_mech"]["samples"])[:, 0]) - p1["theta_true"][0]
            ),
            "bias_ratio": float(
                np.median(np.array(p1["result_mech"]["samples"])[:, 1]) - p1["theta_true"][1]
            ),
        }

    if "phase3" in all_results:
        meta["phase3"] = {
            "t_opt_tau": all_results["phase3"]["t_opt_tau"],
            "t_opt_ratio": all_results["phase3"]["t_opt_ratio"],
            "protocols": all_results["phase3"]["protocols"],
        }

    with open(json_path, "w") as f:
        json.dump(_sanitize(meta), f, indent=2)
    print(f"\n  Results JSON: {json_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TOTAL ELAPSED: {elapsed:.1f} s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
