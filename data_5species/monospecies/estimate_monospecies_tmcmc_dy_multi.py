# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np

def _parse_device_early():
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "auto"

_device_early = _parse_device_early()
if _device_early == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
elif _device_early == "gpu":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from hamilton_monospecies_jax import (
    simulate_monospecies_phibar_hist,
    simulate_monospecies_phibar_hist_dynamic_c_sigma_series,
)
from estimate_monospecies_tmcmc import load_dynamic_data_with_reps, FELIX_SIGMA

def make_log_likelihood_dy_multi(
    dy_time, dy_cfu, dy_std, dy_temp_time, dy_temp_C,
    sigma_knots_base, n_steps, time_to_step=10.0,
    Kp=1e-4, eta=1.0, eta_phi=1.0, dynamic_c=True, c_scale=10.0, dt=1e-4, n_newton=6
):
    dy_time_jax = jnp.array(dy_time, dtype=jnp.float64)
    dy_cfu_jax = jnp.array(dy_cfu, dtype=jnp.float64)
    dy_std_jax = jnp.array(dy_std, dtype=jnp.float64)
    sigma_obs = jnp.maximum(dy_std_jax, 0.1)

    t_max = float(max(np.max(dy_time), np.max(dy_temp_time)))
    t_grid = np.arange(n_steps + 1, dtype=float) / float(time_to_step)
    T_grid = np.interp(t_grid, np.asarray(dy_temp_time, dtype=float), np.asarray(dy_temp_C, dtype=float))

    temps = np.array(sorted(sigma_knots_base.keys()), dtype=float)
    
    # Pre-compute indices and weights for linear interpolation of T_grid into the 8 temperature knots
    idx = np.searchsorted(temps, T_grid[:-1]) - 1
    idx = np.clip(idx, 0, len(temps) - 2)
    t0 = temps[idx]
    t1 = temps[idx + 1]
    w = (T_grid[:-1] - t0) / (t1 - t0)
    w = np.clip(w, 0.0, 1.0)
    
    idx_jax = jnp.array(idx, dtype=jnp.int32)
    w_jax = jnp.array(w, dtype=jnp.float64)

    def log_likelihood(theta):
        # theta is an array of 8 multipliers (k_4, k_8, k_15, k_20, k_25, k_35, k_37, k_40)
        # We multiply the base sigma knots by these multipliers to get the new knots
        base_vals = jnp.array([sigma_knots_base[t] for t in temps], dtype=jnp.float64)
        new_knots = theta * base_vals
        
        # Interpolate sigma_series using the new knots
        v0 = new_knots[idx_jax]
        v1 = new_knots[idx_jax + 1]
        sigma_series = v0 + w_jax * (v1 - v0)
        sigma_series = jnp.clip(sigma_series, 1.0, 500.0)

        phibar = simulate_monospecies_phibar_hist_dynamic_c_sigma_series(
            sigma_series, n_steps, dt=dt, Kp=Kp, eta=eta, eta_phi=eta_phi,
            c_scale=c_scale, n_newton=n_newton
        )

        step_idx = jnp.clip((dy_time_jax * time_to_step).astype(int), 0, n_steps)
        sim_vals = phibar[step_idx]

        X = jnp.column_stack([sim_vals, jnp.ones_like(sim_vals)])
        coeff, _, _, _ = jnp.linalg.lstsq(X, dy_cfu_jax, rcond=None)
        predicted = X @ coeff

        residuals = (predicted - dy_cfu_jax) / sigma_obs
        logL = -0.5 * jnp.sum(residuals**2) - jnp.sum(jnp.log(sigma_obs))
        return logL

    return log_likelihood

def run_tmcmc_dy_multi(
    dy, sigma_knots_base, n_particles=400, mutation='nuts', seed=42, max_stages=50,
    dynamic_c=True, c_scale=10.0, dt=1e-4, n_newton=6, time_to_step=10.0, outdir=None
):
    from tmcmc_nuts_engine import tmcmc_engine

    t_max = float(max(np.max(dy['time']), np.max(dy['temp_time'])))
    n_steps = int(np.ceil(t_max * time_to_step))

    log_lik = make_log_likelihood_dy_multi(
        dy_time=dy['time'], dy_cfu=dy['cfu_mean'], dy_std=dy.get('cfu_std', np.full_like(dy['cfu_mean'], 0.2)),
        dy_temp_time=dy['temp_time'], dy_temp_C=dy['temp_C'],
        sigma_knots_base=sigma_knots_base, n_steps=n_steps, time_to_step=time_to_step,
        dynamic_c=dynamic_c, c_scale=c_scale, dt=dt, n_newton=n_newton,
    )

    # 8 parameters: multipliers for each of the 8 temperatures. Prior [0.1, 10.0]
    prior_bounds = np.full((8, 2), [0.1, 10.0], dtype=float)

    print(f"\n{'='*60}")
    print("  Dy TMCMC - 8-parameter scaling (k_4 ... k_40)")
    print(f"  n_steps={n_steps}, n_obs={len(dy['time'])}")
    print(f"  mutation={mutation}, n_particles={n_particles}")
    print(f"{'='*60}")

    result = tmcmc_engine(
        log_likelihood=log_lik, prior_bounds=prior_bounds, mutation=mutation,
        n_particles=n_particles, max_stages=max_stages, target_ess_ratio=0.5,
        hmc_step_size=0.05, hmc_n_leapfrog=5, nuts_max_depth=6, warmup_stages=2,
        seed=seed, label="Dy-Multi", verbose=True,
    )
    
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        temps = sorted(sigma_knots_base.keys())
        np.savez(os.path.join(outdir, 'samples_Dy_multi.npz'),
                 samples=result['samples'], theta_MAP=result['theta_MAP'],
                 log_likelihoods=result['log_likelihoods'], temps=temps,
                 sigma_knots_base=np.array([sigma_knots_base[t] for t in temps]))
        import json
        with open(os.path.join(outdir, 'tmcmc_summary_Dy_multi.json'), 'w') as f:
            json.dump({
                'theta_MAP': result['theta_MAP'].tolist(),
                'temps': temps,
                'dynamic_c': dynamic_c, 'c_scale': c_scale, 'dt': dt, 'n_newton': n_newton,
            }, f, indent=2)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--mutation', type=str, default='rw')
    parser.add_argument('--n_particles', type=int, default=200)
    parser.add_argument('--max_stages', type=int, default=50)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    xlsx = os.path.join(os.path.dirname(__file__), 'raw data.xlsx')
    dy = load_dynamic_data_with_reps(xlsx, sheet='Dy')
    
    # Start with Felix baseline as base knots
    sigma_knots_base = {float(t): float(FELIX_SIGMA[int(t)]) for t in [4, 8, 15, 20, 25, 35, 37, 40]}
    
    run_tmcmc_dy_multi(
        dy, sigma_knots_base, n_particles=args.n_particles, mutation=args.mutation,
        max_stages=args.max_stages, outdir=args.outdir
    )

if __name__ == '__main__':
    main()
