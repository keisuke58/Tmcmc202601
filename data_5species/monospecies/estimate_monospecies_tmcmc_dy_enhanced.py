# -*- coding: utf-8 -*-
"""
estimate_monospecies_tmcmc_dy_enhanced.py — 拡張 a₁₁(T,t) モデルの TMCMC

Models (--model フラグで選択):
  rat5     : Ratkowsky baseline (5 params)
  ctm5     : CTM/Rosso 1995 (5 params)
  rat_mem  : Ratkowsky + thermal memory τ (6 params)
  rat_asym : Ratkowsky + asymmetric lag (6 params)
  rat_full : Ratkowsky + τ + asymmetric lag (7 params)
  ctm_mem  : CTM + thermal memory τ (6 params)

Usage:
    python estimate_monospecies_tmcmc_dy_enhanced.py \
        --model rat_full --n_particles 2000 --outdir _tmcmc_results_dy_rat_full
"""

import os, sys, argparse, json
import numpy as np

def _parse_device_early():
    for i, a in enumerate(sys.argv):
        if a == "--device" and i+1 < len(sys.argv):
            return sys.argv[i+1]
    return "auto"

_dev = _parse_device_early()
if _dev == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
elif _dev == "gpu":
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from hamilton_monospecies_jax import simulate_monospecies_phibar_hist_dynamic_c_sigma_series
from estimate_monospecies_tmcmc import load_dynamic_data_with_reps

TIME_STEP = 10.0   # steps per minute
DT        = 1e-4
N_NEWTON  = 6
C_SCALE   = 10.0


# ─── a₁₁(T) base functions (pure JAX) ────────────────────────────────────────

def _ratkowsky_base(T, b, T_min, T_max, c):
    T_dm = jnp.maximum(T - T_min, 0.0)
    exp  = jnp.exp(jnp.clip(c * (T - T_max), -50.0, 0.0))
    return jnp.maximum(b * T_dm * (1.0 - exp), 0.0) ** 2


def _ctm_base(T, mu_opt, T_min, T_opt, T_max):
    num   = (T - T_max) * (T - T_min) ** 2
    d1    = T_opt - T_min
    d2    = d1*(T - T_opt) - (T_opt - T_max)*(T_opt + T_min - 2*T)
    denom = d1 * d2
    cm    = jnp.where(
        (T > T_min) & (T < T_max) & (jnp.abs(denom) > 1e-12),
        num / denom, 0.0)
    return jnp.maximum(mu_opt * cm, 0.0)


def _sym_lag(dT, k_lag):
    """dT: precomputed temperature differences (shape: n_steps)."""
    return 1.0 / (1.0 + k_lag * jnp.abs(dT))


def _asym_lag(dT, k_cool, k_heat):
    """dT: precomputed temperature differences (shape: n_steps)."""
    k = jnp.where(dT < 0, k_cool, k_heat)
    return 1.0 / (1.0 + k * jnp.abs(dT))


def _thermal_memory_jax(T_arr, log_tau):
    """Exponential IIR filter via lax.scan (JAX-traceable, vmap compatible)."""
    tau   = jnp.exp(log_tau)
    alpha = jnp.exp(-1.0 / (TIME_STEP * jnp.maximum(tau, 1e-6)))

    def step(T_eff_prev, T_t):
        T_eff_new = alpha * T_eff_prev + (1.0 - alpha) * T_t
        return T_eff_new, T_eff_new

    _, T_eff_series = jax.lax.scan(step, T_arr[0], T_arr[1:])
    return jnp.concatenate([T_arr[:1], T_eff_series])


# ─── Per-model a₁₁-series builders ───────────────────────────────────────────
# Signature: (theta, T_jax, dT_jax)
# dT_jax = precomputed padded temperature differences (constant per experiment)

def build_rat5(theta, T_jax, dT_jax):
    b, T_min, T_max, c, k_lag = theta
    s = _ratkowsky_base(T_jax, b, T_min, T_max, c)
    return jnp.clip(s * _sym_lag(dT_jax, k_lag), 0.1, 500.0)


def build_ctm5(theta, T_jax, dT_jax):
    mu_opt, T_min, T_opt, T_max, k_lag = theta
    s = _ctm_base(T_jax, mu_opt, T_min, T_opt, T_max)
    return jnp.clip(s * _sym_lag(dT_jax, k_lag), 0.1, 500.0)


def build_rat_mem(theta, T_jax, dT_jax):
    b, T_min, T_max, c, k_lag, log_tau = theta
    T_eff = _thermal_memory_jax(T_jax, log_tau)
    s = _ratkowsky_base(T_eff, b, T_min, T_max, c)
    return jnp.clip(s * _sym_lag(dT_jax, k_lag), 0.1, 500.0)


def build_rat_asym(theta, T_jax, dT_jax):
    b, T_min, T_max, c, k_cool, k_heat = theta
    s = _ratkowsky_base(T_jax, b, T_min, T_max, c)
    return jnp.clip(s * _asym_lag(dT_jax, k_cool, k_heat), 0.1, 500.0)


def build_rat_full(theta, T_jax, dT_jax):
    b, T_min, T_max, c, k_cool, k_heat, log_tau = theta
    T_eff = _thermal_memory_jax(T_jax, log_tau)
    s = _ratkowsky_base(T_eff, b, T_min, T_max, c)
    return jnp.clip(s * _asym_lag(dT_jax, k_cool, k_heat), 0.1, 500.0)


def build_ctm_mem(theta, T_jax, dT_jax):
    mu_opt, T_min, T_opt, T_max, k_lag, log_tau = theta
    T_eff = _thermal_memory_jax(T_jax, log_tau)
    s = _ctm_base(T_eff, mu_opt, T_min, T_opt, T_max)
    return jnp.clip(s * _sym_lag(dT_jax, k_lag), 0.1, 500.0)


# ─── Model registry ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    'rat5': {
        'builder': build_rat5,
        'param_names': ['b', 'T_min', 'T_max', 'c', 'k_lag'],
        'bounds': np.array([
            [0.01, 5.0], [-5.0, 5.0], [40.0, 52.0], [0.01, 3.0], [0.0, 5.0]
        ]),
    },
    'ctm5': {
        'builder': build_ctm5,
        'param_names': ['mu_opt', 'T_min', 'T_opt', 'T_max', 'k_lag'],
        'bounds': np.array([
            [0.1, 200.0], [-5.0, 10.0], [20.0, 40.0], [40.0, 55.0], [0.0, 5.0]
        ]),
    },
    'rat_mem': {
        'builder': build_rat_mem,
        'param_names': ['b', 'T_min', 'T_max', 'c', 'k_lag', 'log_tau'],
        'bounds': np.array([
            [0.01, 5.0], [-5.0, 5.0], [40.0, 52.0], [0.01, 3.0],
            [0.0, 5.0], [-3.0, 4.0]
        ]),
    },
    'rat_asym': {
        'builder': build_rat_asym,
        'param_names': ['b', 'T_min', 'T_max', 'c', 'k_cool', 'k_heat'],
        'bounds': np.array([
            [0.01, 5.0], [-5.0, 5.0], [40.0, 52.0], [0.01, 3.0],
            [0.0, 10.0], [0.0, 10.0]
        ]),
    },
    'rat_full': {
        'builder': build_rat_full,
        'param_names': ['b', 'T_min', 'T_max', 'c', 'k_cool', 'k_heat', 'log_tau'],
        'bounds': np.array([
            [0.01, 5.0], [-5.0, 5.0], [40.0, 52.0], [0.01, 3.0],
            [0.0, 10.0], [0.0, 10.0], [-3.0, 4.0]
        ]),
    },
    'ctm_mem': {
        'builder': build_ctm_mem,
        'param_names': ['mu_opt', 'T_min', 'T_opt', 'T_max', 'k_lag', 'log_tau'],
        'bounds': np.array([
            [0.1, 200.0], [-5.0, 10.0], [20.0, 40.0], [40.0, 55.0],
            [0.0, 5.0], [-3.0, 4.0]
        ]),
    },
}


def make_log_likelihood(dy, model_key, n_steps, time_to_step=10.0,
                         dt=1e-4, n_newton=6, c_scale=10.0, sigma_obs=0.1):
    cfg     = MODEL_REGISTRY[model_key]
    builder = cfg['builder']

    t_grid = np.arange(n_steps + 1, dtype=float) / time_to_step
    T_grid = np.interp(t_grid, np.array(dy['temp_time']), np.array(dy['temp_C']))
    T_jax  = jnp.array(T_grid[:-1], dtype=jnp.float64)
    # Precomputed constants — never change across likelihood evaluations
    dT_jax     = jnp.pad(jnp.diff(T_jax) * time_to_step, (1, 0))
    dy_time_j  = jnp.array(dy['time'], dtype=jnp.float64)
    dy_cfu_j   = jnp.array(dy['cfu_mean'], dtype=jnp.float64)
    dy_std_j   = jnp.array(
        dy.get('cfu_std', np.full_like(dy['cfu_mean'], sigma_obs)),
        dtype=jnp.float64)
    sigma_obs_j = jnp.maximum(dy_std_j, sigma_obs)
    step_idx    = jnp.clip((dy_time_j * time_to_step).astype(int), 0, n_steps)

    def log_likelihood(theta):
        sigma_series = builder(jnp.array(theta, dtype=jnp.float64), T_jax, dT_jax)
        phibar = simulate_monospecies_phibar_hist_dynamic_c_sigma_series(
            sigma_series, n_steps, dt=dt, c_scale=c_scale, n_newton=n_newton)
        sv   = phibar[step_idx]
        X    = jnp.column_stack([sv, jnp.ones_like(sv)])
        coeff, _, _, _ = jnp.linalg.lstsq(X, dy_cfu_j, rcond=None)
        pred = X @ coeff
        res  = (pred - dy_cfu_j) / sigma_obs_j
        return -0.5 * jnp.sum(res**2) - jnp.sum(jnp.log(sigma_obs_j))

    return log_likelihood


def run_tmcmc(dy, model_key, n_particles=2000, mutation='rw',
              max_stages=60, seed=42, outdir=None,
              dt=1e-4, n_newton=6, c_scale=10.0):
    from tmcmc_nuts_engine import tmcmc_engine

    cfg     = MODEL_REGISTRY[model_key]
    t_max   = float(max(np.max(dy['time']), np.max(dy['temp_time'])))
    n_steps = int(np.ceil(t_max * TIME_STEP))

    log_lik = make_log_likelihood(dy, model_key, n_steps,
                                   dt=dt, n_newton=n_newton, c_scale=c_scale)

    print(f"\n{'='*60}")
    print(f"  Dy TMCMC — {model_key} ({len(cfg['param_names'])} params)")
    print(f"  params: {cfg['param_names']}")
    print(f"  n_particles={n_particles}, mutation={mutation}")
    print(f"{'='*60}")

    result = tmcmc_engine(
        log_likelihood=log_lik,
        prior_bounds=cfg['bounds'],
        mutation=mutation,
        n_particles=n_particles,
        max_stages=max_stages,
        target_ess_ratio=0.5,
        hmc_step_size=0.01, hmc_n_leapfrog=5,
        nuts_max_depth=6, warmup_stages=2,
        seed=seed, label=f"Dy-{model_key}", verbose=True,
    )

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        np.savez(os.path.join(outdir, f'samples_Dy_{model_key}.npz'),
                 samples=result['samples'],
                 theta_MAP=result['theta_MAP'],
                 log_likelihoods=result['log_likelihoods'])
        with open(os.path.join(outdir, f'tmcmc_summary_Dy_{model_key}.json'), 'w') as f:
            json.dump({
                'model': model_key,
                'theta_MAP': result['theta_MAP'].tolist(),
                'param_names': cfg['param_names'],
                'c_scale': c_scale, 'dt': dt, 'n_newton': n_newton,
            }, f, indent=2)
        print(f"Saved: {outdir}/samples_Dy_{model_key}.npz")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--device',      type=str, default='auto')
    parser.add_argument('--mutation',    type=str, default='rw')
    parser.add_argument('--n_particles', type=int, default=2000)
    parser.add_argument('--max_stages',  type=int, default=60)
    parser.add_argument('--outdir',      type=str, required=True)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    xlsx = os.path.join(os.path.dirname(__file__), 'raw data.xlsx')
    dy   = load_dynamic_data_with_reps(xlsx, sheet='Dy')

    run_tmcmc(dy, args.model,
              n_particles=args.n_particles,
              mutation=args.mutation,
              max_stages=args.max_stages,
              seed=args.seed,
              outdir=args.outdir)


if __name__ == '__main__':
    main()
