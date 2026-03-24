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

from hamilton_monospecies_jax import simulate_monospecies_phibar_hist_dynamic_c_sigma_series
from estimate_monospecies_tmcmc import load_dynamic_data_with_reps

def make_log_likelihood_dy_ratkowsky(
    dy_time, dy_cfu, dy_std, dy_temp_time, dy_temp_C,
    n_steps, time_to_step=10.0,
    Kp=1e-4, eta=1.0, eta_phi=1.0, c_scale=10.0, dt=1e-4, n_newton=6
):
    dy_time_jax = jnp.array(dy_time, dtype=jnp.float64)
    dy_cfu_jax = jnp.array(dy_cfu, dtype=jnp.float64)
    dy_std_jax = jnp.array(dy_std, dtype=jnp.float64)
    sigma_obs = jnp.maximum(dy_std_jax, 0.1)

    t_max = float(max(np.max(dy_time), np.max(dy_temp_time)))
    t_grid = np.arange(n_steps + 1, dtype=float) / float(time_to_step)
    T_grid = np.interp(t_grid, np.asarray(dy_temp_time, dtype=float), np.asarray(dy_temp_C, dtype=float))
    T_grid_jax = jnp.array(T_grid[:-1], dtype=jnp.float64)

    def log_likelihood(theta):
        # theta = [b, T_min, T_max, c, k_lag]
        b_rat = theta[0]      # Growth rate scalar
        T_min = theta[1]      # Minimum growth temperature
        T_max = theta[2]      # Maximum growth temperature
        c_rat = theta[3]      # High temperature decline coefficient
        k_lag = theta[4]      # Dynamic lag adaptation factor

        # Ratkowsky-type secondary model for sigma(T)
        # sqrt(sigma) = b * (T - T_min) * (1 - exp(c * (T - T_max)))
        
        T_diff_min = jnp.maximum(T_grid_jax - T_min, 0.0)
        T_diff_max = T_grid_jax - T_max
        
        # Prevent exponential overflow and negative roots
        exp_term = jnp.exp(jnp.clip(c_rat * T_diff_max, -50.0, 0.0))
        sqrt_sigma_base = b_rat * T_diff_min * (1.0 - exp_term)
        sigma_base = jnp.maximum(sqrt_sigma_base, 0.0)**2
        
        # Introduce a simple dynamic adaptation (lag) memory effect
        # If temperature changes rapidly, effective sigma is damped
        dT_dt = jnp.append(jnp.array([0.0]), jnp.diff(T_grid_jax)) * time_to_step
        lag_penalty = 1.0 / (1.0 + k_lag * jnp.abs(dT_dt))
        
        sigma_series = jnp.clip(sigma_base * lag_penalty, 0.1, 500.0)

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

def run_tmcmc_dy_ratkowsky(
    dy, n_particles=400, mutation='nuts', seed=42, max_stages=60,
    c_scale=10.0, dt=1e-4, n_newton=6, time_to_step=10.0, outdir=None
):
    from tmcmc_nuts_engine import tmcmc_engine

    t_max = float(max(np.max(dy['time']), np.max(dy['temp_time'])))
    n_steps = int(np.ceil(t_max * time_to_step))

    log_lik = make_log_likelihood_dy_ratkowsky(
        dy_time=dy['time'], dy_cfu=dy['cfu_mean'], dy_std=dy.get('cfu_std', np.full_like(dy['cfu_mean'], 0.2)),
        dy_temp_time=dy['temp_time'], dy_temp_C=dy['temp_C'],
        n_steps=n_steps, time_to_step=time_to_step,
        c_scale=c_scale, dt=dt, n_newton=n_newton,
    )

    # theta = [b, T_min, T_max, c, k_lag]
    prior_bounds = np.array([
        [0.01, 5.0],    # b_rat: scale
        [-5.0, 5.0],    # T_min: typical min growth temp for Listeria/biofilm is around 0-4 C
        [42.0, 50.0],   # T_max: max growth temp is around 45 C
        [0.01, 2.0],    # c_rat: decline steepness
        [0.0, 10.0],    # k_lag: penalty for rapid temp changes
    ], dtype=float)

    print(f"\n{'='*60}")
    print("  Dy TMCMC - Ratkowsky Secondary Model + Dynamic Lag")
    print(f"  n_steps={n_steps}, n_obs={len(dy['time'])}")
    print(f"  mutation={mutation}, n_particles={n_particles}")
    print(f"{'='*60}")

    result = tmcmc_engine(
        log_likelihood=log_lik, prior_bounds=prior_bounds, mutation=mutation,
        n_particles=n_particles, max_stages=max_stages, target_ess_ratio=0.5,
        hmc_step_size=0.01, hmc_n_leapfrog=5, nuts_max_depth=6, warmup_stages=2,
        seed=seed, label="Dy-Ratkowsky", verbose=True,
    )
    
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        np.savez(os.path.join(outdir, 'samples_Dy_ratkowsky.npz'),
                 samples=result['samples'], theta_MAP=result['theta_MAP'],
                 log_likelihoods=result['log_likelihoods'])
        import json
        with open(os.path.join(outdir, 'tmcmc_summary_Dy_ratkowsky.json'), 'w') as f:
            json.dump({
                'theta_MAP': result['theta_MAP'].tolist(),
                'param_names': ['b', 'T_min', 'T_max', 'c', 'k_lag'],
                'c_scale': c_scale, 'dt': dt, 'n_newton': n_newton,
            }, f, indent=2)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--mutation', type=str, default='rw')
    parser.add_argument('--n_particles', type=int, default=200)
    parser.add_argument('--max_stages', type=int, default=60)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    xlsx = os.path.join(os.path.dirname(__file__), 'raw data.xlsx')
    dy = load_dynamic_data_with_reps(xlsx, sheet='Dy')
    
    run_tmcmc_dy_ratkowsky(
        dy, n_particles=args.n_particles, mutation=args.mutation,
        max_stages=args.max_stages, outdir=args.outdir
    )

if __name__ == '__main__':
    main()
