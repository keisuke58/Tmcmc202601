# -*- coding: utf-8 -*-
"""
estimate_monospecies_2param.py — 2-parameter TMCMC: (sigma, phi_init)

sigma  : growth-attachment rate [prior: Felix ±1 decade]
phi_init : initial biofilm fraction [prior: 0.01–0.4]

phi_init is the biological initial inoculum density.
Estimating it jointly with sigma allows the model to adapt the trajectory
SHAPE (not just scale), which reduces RMSE significantly.

Usage:
    python estimate_monospecies_2param.py [--n_particles 4000] [--outdir DIR]
"""

import os
import sys
import argparse
import time
import json
import csv
import numpy as np

# --- Early GPU setup ---
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from hamilton_monospecies_jax import simulate_monospecies_phibar_hist_dynamic_c
from estimate_monospecies_tmcmc import load_static_data, FELIX_SIGMA

MAX_STEPS_MAP = {4: 4000, 8: 2000, 15: 1000, 20: 500,
                 25: 300, 35: 200, 37: 200, 40: 450}
TEMPS = [4, 8, 15, 20, 25, 35, 37, 40]


def make_log_likelihood_2param(exp_time, exp_cfu, exp_std, n_steps,
                                time_to_step=10.0, Kp=1e-4, eta=1.0, eta_phi=1.0,
                                c_scale=10.0, dt=1e-4, n_newton=6):
    """
    Log-likelihood for theta = [sigma, phi_init].

    Profile-out the affine map cfu = a*phibar + b analytically.
    """
    exp_time_jax = jnp.array(exp_time)
    exp_cfu_jax = jnp.array(exp_cfu)
    sigma_obs = jnp.maximum(jnp.array(exp_std), 0.1)
    step_idx = jnp.clip((exp_time_jax * time_to_step).astype(int), 0, n_steps)

    def log_likelihood(theta):
        sigma = jnp.float64(theta[0])
        phi_init = jnp.float64(theta[1])

        phibar = simulate_monospecies_phibar_hist_dynamic_c(
            sigma, n_steps, dt=dt, Kp=Kp, eta=eta, eta_phi=eta_phi,
            c_scale=c_scale, n_newton=n_newton, phi_init=phi_init,
        )
        sim_vals = phibar[step_idx]

        # Profile affine map analytically
        X = jnp.column_stack([sim_vals, jnp.ones_like(sim_vals)])
        coeff, _, _, _ = jnp.linalg.lstsq(X, exp_cfu_jax, rcond=None)
        predicted = X @ coeff

        residuals = (predicted - exp_cfu_jax) / sigma_obs
        logL = -0.5 * jnp.sum(residuals ** 2) - jnp.sum(jnp.log(sigma_obs))
        return logL

    return log_likelihood


def run_2param_condition(temp, data, n_particles=4000, mutation='rw', seed=42,
                          max_stages=30, verbose=True, c_scale=10.0,
                          dt=1e-4, n_newton=6, target_ess_ratio=0.5):
    from tmcmc_nuts_engine import tmcmc_engine

    n_steps = MAX_STEPS_MAP[temp]
    d = data[temp]
    exp_std = d.get('cfu_std', np.full_like(d['cfu_mean'], 0.2))

    log_lik = make_log_likelihood_2param(
        d['time'], d['cfu_mean'], exp_std, n_steps,
        c_scale=c_scale, dt=dt, n_newton=n_newton,
    )

    # Prior: sigma in [lo_s, hi_s], phi_init in [0.01, 0.4]
    sigma_ref = FELIX_SIGMA[temp]
    lo_s = max(1.0, sigma_ref * 0.1)
    hi_s = min(500.0, sigma_ref * 10.0)
    prior_bounds = np.array([[lo_s, hi_s], [0.01, 0.40]])

    if verbose:
        print(f"\n{'='*65}")
        print(f"  {temp}°C [2-param]: n_steps={n_steps}, n_obs={len(d['time'])}")
        print(f"  theta = [sigma ∈ [{lo_s:.2f},{hi_s:.2f}], phi_init ∈ [0.01,0.40]]")
        print(f"  n_particles={n_particles}, mutation={mutation}")
        print(f"{'='*65}", flush=True)

    result = tmcmc_engine(
        log_likelihood=log_lik,
        prior_bounds=prior_bounds,
        mutation=mutation,
        n_particles=n_particles,
        max_stages=max_stages,
        target_ess_ratio=target_ess_ratio,
        hmc_step_size=0.05,
        hmc_n_leapfrog=5,
        nuts_max_depth=4,
        warmup_stages=2,
        seed=seed,
        label=f"{temp}C-2param-{mutation}",
        verbose=verbose,
    )

    samples = result['samples']  # (n_particles, 2)
    theta_MAP = result['theta_MAP']
    sigma_MAP = float(theta_MAP[0])
    phi_init_MAP = float(theta_MAP[1])

    if verbose:
        print(f"  MAP: sigma={sigma_MAP:.4f}, phi_init={phi_init_MAP:.4f}")
        print(f"  Mean: sigma={np.mean(samples[:,0]):.4f}±{np.std(samples[:,0]):.4f}, "
              f"phi_init={np.mean(samples[:,1]):.4f}±{np.std(samples[:,1]):.4f}", flush=True)

    return {
        'temp': temp,
        'sigma_MAP': sigma_MAP,
        'phi_init_MAP': phi_init_MAP,
        'sigma_mean': float(np.mean(samples[:, 0])),
        'sigma_std': float(np.std(samples[:, 0])),
        'phi_init_mean': float(np.mean(samples[:, 1])),
        'phi_init_std': float(np.std(samples[:, 1])),
        'n_stages': result['n_stages'],
        'total_time': result['total_time'],
        'samples': samples,
        'log_likelihoods': result['log_likelihoods'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--mutation', type=str, default='rw', choices=['rw', 'nuts', 'hmc'])
    parser.add_argument('--n_particles', type=int, default=4000)
    parser.add_argument('--max_stages', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='_tmcmc_results_dyncc10_2param')
    parser.add_argument('--c-scale', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--n-newton', type=int, default=6)
    parser.add_argument('--target-ess-ratio', type=float, default=0.5)
    args = parser.parse_args()

    # Apply device setting
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif args.device == 'gpu':
        os.environ.setdefault('JAX_PLATFORMS', 'cuda')

    try:
        print(f"JAX backend: {jax.default_backend()} | {jax.devices()}", flush=True)
    except Exception:
        pass

    xlsx_path = os.path.join(os.path.dirname(__file__), 'raw data.xlsx')
    data = load_static_data(xlsx_path)
    os.makedirs(args.outdir, exist_ok=True)

    temps = [args.temp] if args.temp else TEMPS
    all_results = []
    t_total = time.time()

    for temp in temps:
        res = run_2param_condition(
            temp, data,
            n_particles=args.n_particles,
            mutation=args.mutation,
            seed=args.seed,
            max_stages=args.max_stages,
            c_scale=args.c_scale,
            dt=args.dt,
            n_newton=args.n_newton,
            target_ess_ratio=args.target_ess_ratio,
        )
        all_results.append(res)

        np.savez(
            os.path.join(args.outdir, f'samples_{temp}C.npz'),
            samples=res['samples'],
            log_likelihoods=res['log_likelihoods'],
            sigma_MAP=np.float32(res['sigma_MAP']),
            phi_init_MAP=np.float32(res['phi_init_MAP']),
        )

    elapsed = time.time() - t_total

    # Summary
    print(f"\n{'='*80}")
    print(f"2-PARAM TMCMC Summary (dyncc10, {args.mutation.upper()}, {args.n_particles}p)")
    print(f"{'='*80}")
    print(f"{'Temp':>6s} | {'Felix σ':>8s} | {'MAP σ':>8s} | {'MAP φ₀':>8s} | "
          f"{'σ_mean±std':>14s} | {'Stages':>6s} | {'Time':>6s}")
    print('-' * 80)

    felix_sigma = {4: 4.25, 8: 10.0, 15: 25.0, 20: 50.0,
                   25: 100.0, 35: 110.0, 37: 115.0, 40: 40.0}
    for res in all_results:
        t = res['temp']
        print(f"{t:>4d}°C | {felix_sigma[t]:>8.2f} | {res['sigma_MAP']:>8.2f} | "
              f"{res['phi_init_MAP']:>8.4f} | "
              f"{res['sigma_mean']:>6.2f}±{res['sigma_std']:>5.2f} | "
              f"{res['n_stages']:>6d} | {res['total_time']:>5.0f}s")

    print(f"\nTotal: {elapsed:.0f}s")

    # Save summary CSV
    csv_path = os.path.join(args.outdir, 'tmcmc_2param_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['temp', 'felix_sigma', 'sigma_MAP', 'phi_init_MAP',
                    'sigma_mean', 'sigma_std', 'phi_init_mean', 'phi_init_std',
                    'n_stages', 'time_s'])
        for res in all_results:
            t = res['temp']
            w.writerow([t, felix_sigma[t], f"{res['sigma_MAP']:.4f}",
                        f"{res['phi_init_MAP']:.4f}",
                        f"{res['sigma_mean']:.4f}", f"{res['sigma_std']:.4f}",
                        f"{res['phi_init_mean']:.4f}", f"{res['phi_init_std']:.4f}",
                        res['n_stages'], f"{res['total_time']:.1f}"])
    print(f"Saved: {csv_path}")

    # JSON summary
    json_path = os.path.join(args.outdir, 'tmcmc_2param_summary.json')
    with open(json_path, 'w') as f:
        json.dump({
            'n_particles': args.n_particles,
            'c_scale': args.c_scale,
            'n_newton': args.n_newton,
            'results': [{k: (v.tolist() if hasattr(v, 'tolist') else v)
                         for k, v in r.items() if k not in ('samples', 'log_likelihoods')}
                        for r in all_results],
        }, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
