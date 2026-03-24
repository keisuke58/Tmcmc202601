# -*- coding: utf-8 -*-
"""
refine_map_dyncc10.py — MAP 精密化スクリプト

既存の TMCMC サンプル (4000p) を出発点として、scipy.optimize.minimize_scalar で
1D sigma の真の MAP を求める。1条件 1秒未満で完了。

Usage:
    python refine_map_dyncc10.py [--indir DIR] [--outdir DIR] [--n-newton 10]
"""

import os
import sys
import argparse
import numpy as np
from scipy.optimize import minimize_scalar

# --- 事前にJAXデバイス設定 ---
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from hamilton_monospecies_jax import (
    simulate_monospecies_phibar_hist_dynamic_c,
)
from estimate_monospecies_tmcmc import load_static_data, FELIX_SIGMA


MAX_STEPS_MAP = {4: 4000, 8: 2000, 15: 1000, 20: 500,
                 25: 300, 35: 200, 37: 200, 40: 450}

TEMPS = [4, 8, 15, 20, 25, 35, 37, 40]


def build_neg_loglik_numpy(exp_time, exp_cfu, exp_std, n_steps,
                           time_to_step=10.0, Kp=1e-4, eta=1.0, eta_phi=1.0,
                           c_scale=10.0, dt=1e-4, n_newton=10):
    """
    Negative log-likelihood as a plain Python function of scalar sigma.
    Uses JAX ODE under the hood (compiled on first call).
    """
    exp_time_jax = jnp.array(exp_time)
    exp_cfu_jax = jnp.array(exp_cfu)
    sigma_obs = jnp.maximum(jnp.array(exp_std), 0.1)
    step_idx = jnp.clip((exp_time_jax * time_to_step).astype(int), 0, n_steps)

    @jax.jit
    def neg_loglik_jax(sigma_scalar):
        phibar = simulate_monospecies_phibar_hist_dynamic_c(
            sigma_scalar, n_steps, dt=dt, Kp=Kp, eta=eta, eta_phi=eta_phi,
            c_scale=c_scale, n_newton=n_newton
        )
        sim_vals = phibar[step_idx]
        X = jnp.column_stack([sim_vals, jnp.ones_like(sim_vals)])
        coeff, _, _, _ = jnp.linalg.lstsq(X, exp_cfu_jax, rcond=None)
        predicted = X @ coeff
        residuals = (predicted - exp_cfu_jax) / sigma_obs
        logL = -0.5 * jnp.sum(residuals ** 2) - jnp.sum(jnp.log(sigma_obs))
        return -logL

    def neg_loglik_np(sigma):
        return float(neg_loglik_jax(jnp.array(sigma, dtype=jnp.float64)))

    return neg_loglik_np


def refine_map_condition(temp, data, n_newton=10, c_scale=10.0, dt=1e-4,
                         sigma_init=None, verbose=True):
    """Find MAP sigma for one temperature via scipy bounded minimization."""
    d = data[temp]
    n_steps = MAX_STEPS_MAP[temp]
    exp_std = d.get('cfu_std', np.full_like(d['cfu_mean'], 0.2))

    sigma_ref = FELIX_SIGMA[temp]
    lo = max(1.0, sigma_ref * 0.1)
    hi = min(500.0, sigma_ref * 10.0)

    neg_loglik = build_neg_loglik_numpy(
        d['time'], d['cfu_mean'], exp_std, n_steps,
        c_scale=c_scale, dt=dt, n_newton=n_newton
    )

    # Warm-up JIT with initial sigma
    x0 = sigma_init if sigma_init is not None else sigma_ref
    x0 = float(np.clip(x0, lo + 1e-6, hi - 1e-6))

    if verbose:
        print(f"  {temp}°C: compiling JIT...", flush=True)
    _ = neg_loglik(x0)  # trigger JIT compilation
    if verbose:
        print(f"  {temp}°C: JIT done. Optimizing sigma ∈ [{lo:.2f}, {hi:.2f}]...", flush=True)

    # Refine with bounded scalar optimization (Brent's method, ~15-20 evals)
    # Start from TMCMC MAP, search within [lo, hi]
    result = minimize_scalar(
        neg_loglik,
        bounds=(lo, hi),
        method='bounded',
        options={'xatol': 1e-5, 'maxiter': 200},
    )

    sigma_MAP_refined = float(result.x)
    neg_logL_MAP = float(result.fun)

    if verbose:
        print(f"  {temp}°C: MAP σ (refined) = {sigma_MAP_refined:.4f}  "
              f"(TMCMC_init={x0:.4f}, logL={-neg_logL_MAP:.3f}, "
              f"n_evals={result.nit})", flush=True)

    return {
        'temp': temp,
        'sigma_MAP': sigma_MAP_refined,
        'sigma_MAP_init': x0,
        'neg_logL': neg_logL_MAP,
        'lo': lo,
        'hi': hi,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str,
                        default='_tmcmc_results_dyncc10_rw4000p',
                        help='Directory with existing samples_*.npz (for init)')
    parser.add_argument('--outdir', type=str,
                        default='_tmcmc_results_dyncc10_maprefine',
                        help='Output directory for refined results')
    parser.add_argument('--n-newton', type=int, default=10,
                        help='Newton iterations for ODE. Default 10 (vs 6 in TMCMC)')
    parser.add_argument('--c-scale', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--temp', type=int, default=None,
                        help='Single temp to refine (default: all 8)')
    args = parser.parse_args()

    try:
        backend = jax.default_backend()
        print(f"JAX backend: {backend}", flush=True)
    except Exception:
        pass

    xlsx_path = os.path.join(os.path.dirname(__file__), 'raw data.xlsx')
    data = load_static_data(xlsx_path)

    os.makedirs(args.outdir, exist_ok=True)
    temps = [args.temp] if args.temp else TEMPS

    all_results = []
    for temp in temps:
        # Load TMCMC MAP as initial guess
        sigma_init = None
        npz_path = os.path.join(args.indir, f'samples_{temp}C.npz')
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            sigma_init = float(d['sigma_MAP'])
            print(f"\n{temp}°C: TMCMC MAP={sigma_init:.4f} (loaded from {npz_path})", flush=True)
        else:
            print(f"\n{temp}°C: No prior samples found, using Felix ref", flush=True)

        res = refine_map_condition(
            temp, data,
            n_newton=args.n_newton,
            c_scale=args.c_scale,
            dt=args.dt,
            sigma_init=sigma_init,
        )
        all_results.append(res)

        # Copy original samples and update sigma_MAP
        out_npz = os.path.join(args.outdir, f'samples_{temp}C.npz')
        if os.path.exists(npz_path):
            orig = np.load(npz_path)
            np.savez(out_npz,
                     samples=orig['samples'],
                     log_likelihoods=orig['log_likelihoods'],
                     sigma_MAP=np.float32(res['sigma_MAP']),
                     sigma_MAP_refined=np.float64(res['sigma_MAP']),
                     sigma_MAP_tmcmc=np.float32(sigma_init),
                     )
        else:
            np.savez(out_npz,
                     samples=np.array([res['sigma_MAP']], dtype=np.float32),
                     log_likelihoods=np.array([-res['neg_logL']], dtype=np.float32),
                     sigma_MAP=np.float32(res['sigma_MAP']),
                     sigma_MAP_refined=np.float64(res['sigma_MAP']),
                     )
        print(f"  Saved: {out_npz}", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print(f"MAP Refinement Summary (n_newton={args.n_newton}, c_scale={args.c_scale})")
    print(f"{'='*70}")
    print(f"{'Temp':>6s} | {'Felix σ':>8s} | {'TMCMC MAP':>10s} | {'Refined MAP':>12s} | {'Delta':>8s}")
    print('-' * 70)
    for res in all_results:
        t = res['temp']
        felix = FELIX_SIGMA[t]
        tmcmc = res['sigma_MAP_init'] if res['sigma_MAP_init'] is not None else float('nan')
        refined = res['sigma_MAP']
        delta = refined - tmcmc if tmcmc is not None else float('nan')
        print(f"{t:>4d}°C | {felix:>8.2f} | {tmcmc:>10.4f} | {refined:>12.4f} | {delta:>+8.4f}")

    import json
    summary = {
        'n_newton': args.n_newton,
        'c_scale': args.c_scale,
        'dt': args.dt,
        'indir': args.indir,
        'results': all_results,
    }
    json_path = os.path.join(args.outdir, 'map_refinement_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {json_path}")


if __name__ == '__main__':
    main()
