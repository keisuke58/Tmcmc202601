# -*- coding: utf-8 -*-
"""
estimate_monospecies_4param.py — 4-parameter TMCMC: (a11, c_scale, phi_init, eta_phi)

theta = [a11, c_scale, phi_init, eta_phi]
  a11      : growth-attachment rate       [prior: Felix ±1 decade]
  c_scale  : effective nutrient scale     [prior: 1–100, uniform]
             c(t) = c_scale * (1 - phi*psi)
  phi_init : initial biofilm fraction     [prior: 0.001–0.4, uniform]
             controls trajectory startup shape
  eta_phi  : biofilm attachment inertia   [prior: 0.1–20.0, uniform]
             (eta_phi + eta*psi^2)*phi_dot term in Q0
             larger = slower attachment, smaller = steeper rise

affine map  cfu = a * phibar + b  is analytically profiled out.

Usage:
    python estimate_monospecies_4param.py [--n_particles 4000] [--outdir DIR]
    python estimate_monospecies_4param.py --temp 37
"""

import os
import sys
import argparse
import time
import json
import csv
import numpy as np

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


def make_log_likelihood_4param(exp_time, exp_cfu, exp_std, n_steps,
                                time_to_step=10.0, Kp=1e-4, eta=1.0,
                                dt=1e-4, n_newton=6):
    """
    Log-likelihood for theta = [a11, c_scale, phi_init, eta_phi].
    Affine map cfu = a*phibar + b is profiled out analytically.
    """
    exp_time_jax = jnp.array(exp_time)
    exp_cfu_jax  = jnp.array(exp_cfu)
    sigma_obs    = jnp.maximum(jnp.array(exp_std), 0.1)
    step_idx     = jnp.clip((exp_time_jax * time_to_step).astype(int), 0, n_steps)

    def log_likelihood(theta):
        a11      = jnp.float64(theta[0])
        c_scale  = jnp.float64(theta[1])
        phi_init = jnp.float64(theta[2])
        eta_phi  = jnp.float64(theta[3])

        phibar = simulate_monospecies_phibar_hist_dynamic_c(
            a11, n_steps, dt=dt, Kp=Kp, eta=eta, eta_phi=eta_phi,
            c_scale=c_scale, alpha=0.0, b=0.0,
            n_newton=n_newton, phi_init=phi_init,
        )
        sim_vals = phibar[step_idx]

        X = jnp.column_stack([sim_vals, jnp.ones_like(sim_vals)])
        coeff, _, _, _ = jnp.linalg.lstsq(X, exp_cfu_jax, rcond=None)
        predicted = X @ coeff

        residuals = (predicted - exp_cfu_jax) / sigma_obs
        logL = -0.5 * jnp.sum(residuals ** 2) - jnp.sum(jnp.log(sigma_obs))
        return logL

    return log_likelihood


def run_4param_condition(temp, data, n_particles=4000, mutation='rw', seed=42,
                          max_stages=30, verbose=True,
                          dt=1e-4, n_newton=6, target_ess_ratio=0.5,
                          c_scale_lo=1.0, c_scale_hi=100.0,
                          phi_init_lo=0.001, phi_init_hi=0.4,
                          eta_phi_lo=0.1, eta_phi_hi=20.0):
    from tmcmc_nuts_engine import tmcmc_engine

    n_steps = MAX_STEPS_MAP[temp]
    d = data[temp]
    exp_std = d.get('cfu_std', np.full_like(d['cfu_mean'], 0.2))

    log_lik = make_log_likelihood_4param(
        d['time'], d['cfu_mean'], exp_std, n_steps,
        dt=dt, n_newton=n_newton,
    )

    sigma_ref = FELIX_SIGMA[temp]
    lo_s = max(1.0, sigma_ref * 0.1)
    hi_s = min(500.0, sigma_ref * 10.0)
    prior_bounds = np.array([[lo_s, hi_s],
                              [c_scale_lo, c_scale_hi],
                              [phi_init_lo, phi_init_hi],
                              [eta_phi_lo, eta_phi_hi]])

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {temp}°C [a11+c_scale+phi_init+eta_phi]: n_steps={n_steps}, n_obs={len(d['time'])}")
        print(f"  theta = [a11 ∈ [{lo_s:.1f},{hi_s:.1f}], c_scale ∈ [{c_scale_lo},{c_scale_hi}], "
              f"phi_init ∈ [{phi_init_lo},{phi_init_hi}], eta_phi ∈ [{eta_phi_lo},{eta_phi_hi}]]")
        print(f"  n_particles={n_particles}, mutation={mutation}")
        print(f"{'='*70}", flush=True)

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
        label=f"{temp}C-4param-{mutation}",
        verbose=verbose,
    )

    samples      = result['samples']          # (N, 4)
    theta_MAP    = result['theta_MAP']
    a11_MAP      = float(theta_MAP[0])
    c_scale_MAP  = float(theta_MAP[1])
    phi_init_MAP = float(theta_MAP[2])
    eta_phi_MAP  = float(theta_MAP[3])

    # RMSE at MAP
    phibar_map = np.asarray(simulate_monospecies_phibar_hist_dynamic_c(
        jnp.float64(a11_MAP), n_steps, dt=dt,
        c_scale=jnp.float64(c_scale_MAP), n_newton=n_newton,
        phi_init=jnp.float64(phi_init_MAP),
        eta_phi=jnp.float64(eta_phi_MAP),
    ))
    step_idx = np.clip((d['time'] * 10.0).astype(int), 0, n_steps)
    sv = phibar_map[step_idx]
    X  = np.column_stack([sv, np.ones_like(sv)])
    coeff, _, _, _ = np.linalg.lstsq(X, d['cfu_mean'], rcond=None)
    rmse = float(np.sqrt(np.mean((X @ coeff - d['cfu_mean'])**2)))

    if verbose:
        print(f"  MAP: a11={a11_MAP:.4f}, c_scale={c_scale_MAP:.4f}, "
              f"phi_init={phi_init_MAP:.5f}, eta_phi={eta_phi_MAP:.4f}, RMSE={rmse:.4f}")
        print(f"  Mean±std: a11={np.mean(samples[:,0]):.3f}±{np.std(samples[:,0]):.3f}, "
              f"c_scale={np.mean(samples[:,1]):.2f}±{np.std(samples[:,1]):.2f}, "
              f"phi_init={np.mean(samples[:,2]):.4f}±{np.std(samples[:,2]):.4f}, "
              f"eta_phi={np.mean(samples[:,3]):.3f}±{np.std(samples[:,3]):.3f}", flush=True)

    return {
        'temp': temp,
        'a11_MAP': a11_MAP,
        'c_scale_MAP': c_scale_MAP,
        'phi_init_MAP': phi_init_MAP,
        'eta_phi_MAP': eta_phi_MAP,
        'rmse_MAP': rmse,
        'a11_mean':      float(np.mean(samples[:, 0])),
        'a11_std':       float(np.std(samples[:, 0])),
        'c_scale_mean':  float(np.mean(samples[:, 1])),
        'c_scale_std':   float(np.std(samples[:, 1])),
        'phi_init_mean': float(np.mean(samples[:, 2])),
        'phi_init_std':  float(np.std(samples[:, 2])),
        'eta_phi_mean':  float(np.mean(samples[:, 3])),
        'eta_phi_std':   float(np.std(samples[:, 3])),
        'n_stages':   result['n_stages'],
        'total_time': result['total_time'],
        'samples':    samples,
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
    parser.add_argument('--outdir', type=str, default='_tmcmc_results_4param')
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--n-newton', type=int, default=6)
    parser.add_argument('--target-ess-ratio', type=float, default=0.5)
    parser.add_argument('--c-scale-lo', type=float, default=1.0)
    parser.add_argument('--c-scale-hi', type=float, default=100.0)
    parser.add_argument('--phi-init-lo', type=float, default=0.001)
    parser.add_argument('--phi-init-hi', type=float, default=0.4)
    parser.add_argument('--eta-phi-lo', type=float, default=0.1)
    parser.add_argument('--eta-phi-hi', type=float, default=20.0)
    args = parser.parse_args()

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
        res = run_4param_condition(
            temp, data,
            n_particles=args.n_particles,
            mutation=args.mutation,
            seed=args.seed,
            max_stages=args.max_stages,
            dt=args.dt,
            n_newton=args.n_newton,
            target_ess_ratio=args.target_ess_ratio,
            c_scale_lo=args.c_scale_lo,
            c_scale_hi=args.c_scale_hi,
            phi_init_lo=args.phi_init_lo,
            phi_init_hi=args.phi_init_hi,
            eta_phi_lo=args.eta_phi_lo,
            eta_phi_hi=args.eta_phi_hi,
        )
        all_results.append(res)

        np.savez(
            os.path.join(args.outdir, f'samples_{temp}C.npz'),
            samples=res['samples'],
            log_likelihoods=res['log_likelihoods'],
            a11_MAP=np.float64(res['a11_MAP']),
            c_scale_MAP=np.float64(res['c_scale_MAP']),
            phi_init_MAP=np.float64(res['phi_init_MAP']),
            eta_phi_MAP=np.float64(res['eta_phi_MAP']),
            rmse_MAP=np.float64(res['rmse_MAP']),
        )
        print(f"  Saved: {args.outdir}/samples_{temp}C.npz", flush=True)

    elapsed = time.time() - t_total

    print(f"\n{'='*95}")
    print(f"a11 + c_scale + phi_init + eta_phi TMCMC Summary ({args.mutation.upper()}, {args.n_particles}p)")
    print(f"{'='*95}")
    print(f"{'Temp':>6s} | {'Felix a11':>9s} | {'MAP a11':>8s} | {'MAP c':>7s} | "
          f"{'MAP φ₀':>8s} | {'MAP ηφ':>7s} | {'RMSE':>7s} | {'Stages':>6s}")
    print('-' * 95)
    for res in all_results:
        t = res['temp']
        print(f"{t:>4d}°C | {FELIX_SIGMA[t]:>9.2f} | {res['a11_MAP']:>8.3f} | "
              f"{res['c_scale_MAP']:>7.3f} | {res['phi_init_MAP']:>8.5f} | "
              f"{res['eta_phi_MAP']:>7.3f} | {res['rmse_MAP']:>7.4f} | {res['n_stages']:>6d}")
    print(f"\nTotal: {elapsed:.0f}s")

    csv_path = os.path.join(args.outdir, 'tmcmc_4param_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['temp', 'felix_a11', 'a11_MAP', 'c_scale_MAP', 'phi_init_MAP', 'eta_phi_MAP',
                    'rmse_MAP', 'a11_mean', 'a11_std', 'c_scale_mean', 'c_scale_std',
                    'phi_init_mean', 'phi_init_std', 'eta_phi_mean', 'eta_phi_std',
                    'n_stages', 'time_s'])
        for res in all_results:
            t = res['temp']
            w.writerow([t, FELIX_SIGMA[t],
                        f"{res['a11_MAP']:.4f}", f"{res['c_scale_MAP']:.4f}",
                        f"{res['phi_init_MAP']:.5f}", f"{res['eta_phi_MAP']:.4f}",
                        f"{res['rmse_MAP']:.4f}",
                        f"{res['a11_mean']:.4f}", f"{res['a11_std']:.4f}",
                        f"{res['c_scale_mean']:.4f}", f"{res['c_scale_std']:.4f}",
                        f"{res['phi_init_mean']:.5f}", f"{res['phi_init_std']:.5f}",
                        f"{res['eta_phi_mean']:.4f}", f"{res['eta_phi_std']:.4f}",
                        res['n_stages'], f"{res['total_time']:.1f}"])
    print(f"Saved: {csv_path}")

    json_path = os.path.join(args.outdir, 'tmcmc_4param_summary.json')
    with open(json_path, 'w') as f:
        json.dump({
            'n_particles': args.n_particles,
            'n_newton': args.n_newton,
            'results': [{k: (v.tolist() if hasattr(v, 'tolist') else v)
                         for k, v in r.items() if k not in ('samples', 'log_likelihoods')}
                        for r in all_results],
        }, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
