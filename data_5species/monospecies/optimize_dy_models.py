# -*- coding: utf-8 -*-
"""
optimize_dy_models.py — Dy データに対する拡張モデル群の MAP 最適化 + 比較

Models:
  rat5   : Ratkowsky (baseline, 5 params)
  ctm5   : Cardinal Temperature Model / Rosso 1995 (5 params)
  rat_mem : Ratkowsky + thermal memory τ_T (6 params)
  rat_asym: Ratkowsky + asymmetric lag k_cool/k_heat (6 params)
  rat_full: Ratkowsky + τ_T + asymmetric lag (7 params)
  ctm_mem : CTM + thermal memory τ_T (6 params)

Usage:
    python optimize_dy_models.py [--n_restarts 20]
"""

import os, sys, argparse
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from hamilton_monospecies_jax import simulate_monospecies_phibar_hist_dynamic_c_sigma_series
from estimate_monospecies_tmcmc import load_dynamic_data_with_reps

BASE       = os.path.dirname(__file__)
N_STEPS    = 1500
TIME_STEP  = 10.0   # steps per minute
MIN_TO_H   = 1/60.0
DT         = 1e-4
N_NEWTON   = 6
C_SCALE    = 10.0


# ─── Thermal memory (exponential IIR filter) ──────────────────────────────────
def apply_thermal_memory(T_arr, tau_min):
    """Exponential IIR: T_eff[t] = α·T_eff[t-1] + (1-α)·T[t], via lfilter."""
    alpha = np.exp(-1.0 / (TIME_STEP * max(tau_min, 1e-6)))
    b_f   = [1.0 - alpha]
    a_f   = [1.0, -alpha]
    zi    = np.array([T_arr[0]])   # initial condition: T_eff[0] = T[0]
    T_eff, _ = lfilter(b_f, a_f, T_arr, zi=zi)
    return T_eff


# ─── σ(T) builders ────────────────────────────────────────────────────────────
def ratkowsky_base(T, b, T_min, T_max, c):
    T_diff_min = np.maximum(T - T_min, 0.0)
    T_diff_max = T - T_max
    exp_term   = np.exp(np.clip(c * T_diff_max, -50.0, 0.0))
    return np.maximum(b * T_diff_min * (1.0 - exp_term), 0.0) ** 2


def ctm_base(T, mu_opt, T_min, T_opt, T_max):
    """Cardinal Temperature Model (Rosso 1995)"""
    num    = (T - T_max) * (T - T_min) ** 2
    denom1 = (T_opt - T_min)
    denom2 = ((T_opt - T_min) * (T - T_opt) - (T_opt - T_max) * (T_opt + T_min - 2*T))
    denom  = denom1 * denom2
    cm     = np.where(
        (T > T_min) & (T < T_max) & (np.abs(denom) > 1e-12),
        num / denom, 0.0
    )
    return np.maximum(mu_opt * cm, 0.0)


def symmetric_lag(T_arr, k_lag):
    dT = np.concatenate([[0.0], np.diff(T_arr)]) * TIME_STEP
    return 1.0 / (1.0 + k_lag * np.abs(dT))


def asymmetric_lag(T_arr, k_cool, k_heat):
    dT = np.concatenate([[0.0], np.diff(T_arr)]) * TIME_STEP
    k  = np.where(dT < 0, k_cool, k_heat)
    return 1.0 / (1.0 + k * np.abs(dT))


# ─── σ series builders per model ──────────────────────────────────────────────
def build_rat5(theta, T_arr):
    b, T_min, T_max, c, k_lag = theta
    s = ratkowsky_base(T_arr, b, T_min, T_max, c)
    return np.clip(s * symmetric_lag(T_arr, k_lag), 0.1, 500.0)


def build_ctm5(theta, T_arr):
    mu_opt, T_min, T_opt, T_max, k_lag = theta
    s = ctm_base(T_arr, mu_opt, T_min, T_opt, T_max)
    return np.clip(s * symmetric_lag(T_arr, k_lag), 0.1, 500.0)


def build_rat_mem(theta, T_arr):
    b, T_min, T_max, c, k_lag, log_tau = theta
    tau = np.exp(log_tau)
    T_eff = apply_thermal_memory(T_arr, tau)
    s = ratkowsky_base(T_eff, b, T_min, T_max, c)
    return np.clip(s * symmetric_lag(T_arr, k_lag), 0.1, 500.0)


def build_rat_asym(theta, T_arr):
    b, T_min, T_max, c, k_cool, k_heat = theta
    s = ratkowsky_base(T_arr, b, T_min, T_max, c)
    return np.clip(s * asymmetric_lag(T_arr, k_cool, k_heat), 0.1, 500.0)


def build_rat_full(theta, T_arr):
    b, T_min, T_max, c, k_cool, k_heat, log_tau = theta
    tau = np.exp(log_tau)
    T_eff = apply_thermal_memory(T_arr, tau)
    s = ratkowsky_base(T_eff, b, T_min, T_max, c)
    return np.clip(s * asymmetric_lag(T_arr, k_cool, k_heat), 0.1, 500.0)


def build_ctm_mem(theta, T_arr):
    mu_opt, T_min, T_opt, T_max, k_lag, log_tau = theta
    tau = np.exp(log_tau)
    T_eff = apply_thermal_memory(T_arr, tau)
    s = ctm_base(T_eff, mu_opt, T_min, T_opt, T_max)
    return np.clip(s * symmetric_lag(T_arr, k_lag), 0.1, 500.0)


# ─── Simulation & profiled fit ─────────────────────────────────────────────────
def simulate(sigma_series):
    return np.asarray(simulate_monospecies_phibar_hist_dynamic_c_sigma_series(
        jnp.array(sigma_series, dtype=jnp.float64),
        N_STEPS, dt=DT, c_scale=C_SCALE, n_newton=N_NEWTON,
    ))


def profiled_rmse(sigma_series, obs_time, obs_cfu):
    phibar   = simulate(sigma_series)
    step_idx = np.clip((np.array(obs_time) * TIME_STEP).astype(int), 0, N_STEPS)
    sv       = phibar[step_idx]
    X        = np.column_stack([sv, np.ones_like(sv)])
    coeff, _, _, _ = np.linalg.lstsq(X, np.array(obs_cfu), rcond=None)
    pred     = X @ coeff
    rmse     = float(np.sqrt(np.mean((pred - obs_cfu)**2)))
    logL     = -0.5 * np.sum(((pred - obs_cfu) / 0.1)**2)  # σ_obs=0.1 fixed
    return rmse, logL, coeff, phibar


def neg_logL_fn(theta, builder, T_arr, obs_time, obs_cfu):
    try:
        ss = builder(theta, T_arr)
        _, logL, _, _ = profiled_rmse(ss, obs_time, obs_cfu)
        return -logL
    except (FloatingPointError, ValueError, OverflowError) as e:
        print(f"  [warn] neg_logL_fn: {e} at theta={np.round(theta, 3)}", flush=True)
        return 1e10


# ─── Model registry ───────────────────────────────────────────────────────────
MODELS = {
    'rat5': {
        'builder': build_rat5,
        'n_params': 5,
        'bounds': [
            (0.01, 5.0),   # b
            (-5.0, 5.0),   # T_min
            (40.0, 52.0),  # T_max
            (0.01, 3.0),   # c
            (0.0, 5.0),    # k_lag
        ],
        'label': 'Ratkowsky (5p)',
    },
    'ctm5': {
        'builder': build_ctm5,
        'n_params': 5,
        'bounds': [
            (0.1, 200.0),  # mu_opt
            (-5.0, 10.0),  # T_min
            (20.0, 40.0),  # T_opt
            (40.0, 55.0),  # T_max
            (0.0, 5.0),    # k_lag
        ],
        'label': 'CTM-Rosso (5p)',
    },
    'rat_mem': {
        'builder': build_rat_mem,
        'n_params': 6,
        'bounds': [
            (0.01, 5.0),   # b
            (-5.0, 5.0),   # T_min
            (40.0, 52.0),  # T_max
            (0.01, 3.0),   # c
            (0.0, 5.0),    # k_lag
            (-3.0, 4.0),   # log_tau (tau=0.05min..54min)
        ],
        'label': 'Rat + thermal memory τ (6p)',
    },
    'rat_asym': {
        'builder': build_rat_asym,
        'n_params': 6,
        'bounds': [
            (0.01, 5.0),   # b
            (-5.0, 5.0),   # T_min
            (40.0, 52.0),  # T_max
            (0.01, 3.0),   # c
            (0.0, 10.0),   # k_cool
            (0.0, 10.0),   # k_heat
        ],
        'label': 'Rat + asym lag (6p)',
    },
    'rat_full': {
        'builder': build_rat_full,
        'n_params': 7,
        'bounds': [
            (0.01, 5.0),   # b
            (-5.0, 5.0),   # T_min
            (40.0, 52.0),  # T_max
            (0.01, 3.0),   # c
            (0.0, 10.0),   # k_cool
            (0.0, 10.0),   # k_heat
            (-3.0, 4.0),   # log_tau
        ],
        'label': 'Rat + τ + asym lag (7p)',
    },
    'ctm_mem': {
        'builder': build_ctm_mem,
        'n_params': 6,
        'bounds': [
            (0.1, 200.0),  # mu_opt
            (-5.0, 10.0),  # T_min
            (20.0, 40.0),  # T_opt
            (40.0, 55.0),  # T_max
            (0.0, 5.0),    # k_lag
            (-3.0, 4.0),   # log_tau
        ],
        'label': 'CTM + thermal memory τ (6p)',
    },
}


def optimize_model(key, cfg, T_arr, obs_time, obs_cfu, n_restarts, rng):
    """Multi-start L-BFGS-B: n_restarts random starting points, keep best."""
    bounds = cfg['bounds']
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    best_val, best_x = np.inf, None
    for i in range(n_restarts):
        x0 = rng.uniform(lo, hi)
        try:
            res = minimize(
                neg_logL_fn, x0,
                args=(cfg['builder'], T_arr, obs_time, obs_cfu),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x
        except Exception:
            pass
    return best_x, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_restarts', type=int, default=40)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--outpng',  type=str,
                        default='figures/dy_enhanced_comparison.png')
    args = parser.parse_args()

    xlsx_path = os.path.join(BASE, 'raw data.xlsx')
    dy = load_dynamic_data_with_reps(xlsx_path, sheet='Dy')

    t_grid = np.arange(N_STEPS + 1) / TIME_STEP
    T_grid = np.interp(t_grid, dy['temp_time'], dy['temp_C'])
    T_arr  = T_grid[:-1]           # shape (N_STEPS,)
    t_h    = dy['time'] * MIN_TO_H
    t_full = t_grid * MIN_TO_H

    obs_time = dy['time']
    obs_cfu  = dy['cfu_mean']
    N_obs    = len(obs_cfu)

    rng = np.random.default_rng(args.seed)

    # Warm up JAX JIT with a dummy call
    print("JIT warm-up ...", flush=True)
    dummy_ss = np.ones(N_STEPS) * 5.0
    _ = profiled_rmse(dummy_ss, obs_time, obs_cfu)
    print("JIT done.", flush=True)

    results = {}

    for key, cfg in MODELS.items():
        print(f"\n{'─'*55}", flush=True)
        print(f"Optimizing: {cfg['label']}  ({args.n_restarts} restarts)", flush=True)
        theta_map, neg_logL = optimize_model(key, cfg, T_arr, obs_time, obs_cfu,
                                              args.n_restarts, rng)
        ss   = cfg['builder'](theta_map, T_arr)
        rmse, logL, coeff, phibar = profiled_rmse(ss, obs_time, obs_cfu)
        k_p  = cfg['n_params']
        aic  = 2*k_p - 2*logL
        bic  = np.log(N_obs)*k_p - 2*logL
        print(f"  theta_MAP = {np.round(theta_map, 4)}", flush=True)
        print(f"  RMSE={rmse:.4f}  logL={logL:.2f}  AIC={aic:.1f}  BIC={bic:.1f}", flush=True)
        results[key] = {
            'theta_map': theta_map,
            'sigma_series': ss,
            'phibar': phibar,
            'coeff': coeff,
            'rmse': rmse, 'logL': logL, 'aic': aic, 'bic': bic,
        }

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Model':<32} | {'k':>3} | {'RMSE':>7} | {'logL':>7} | {'AIC':>7} | {'BIC':>7}")
    print('-'*70)
    for key, r in sorted(results.items(), key=lambda x: x[1]['aic']):
        print(f"{MODELS[key]['label']:<32} | {MODELS[key]['n_params']:>3} | "
              f"{r['rmse']:>7.4f} | {r['logL']:>7.2f} | {r['aic']:>7.1f} | {r['bic']:>7.1f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_models = len(MODELS)
    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, n_models, figure=fig, hspace=0.5, wspace=0.3)

    palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']

    for i, (key, r) in enumerate(results.items()):
        col = palette[i % len(palette)]
        mapped = r['coeff'][0] * r['phibar'] + r['coeff'][1]

        # Top: fit
        ax = fig.add_subplot(gs[0, i])
        ax.plot(t_full, mapped, '-', color=col, lw=1.6, zorder=3)
        ax.scatter(t_h, obs_cfu, color='#e05a2b', s=18, zorder=4,
                   edgecolors='white', linewidths=0.4)
        for ti, reps in dy['reps_by_time'].items():
            ax.plot([ti*MIN_TO_H]*len(reps), reps, '_',
                    color='#e05a2b', ms=5, alpha=0.5, zorder=5)
        ax.set_title(f'{r["label"]}\nRMSE={r["rmse"]:.3f}  AIC={r["aic"]:.1f}',
                     fontsize=8, pad=4)
        ax.set_xlabel('Time (h)', fontsize=8)
        if i == 0:
            ax.set_ylabel('log₁₀(CFU/mL)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top','right']].set_visible(False)

        # Bottom: σ(t) + T(t) twin axis
        ax2 = fig.add_subplot(gs[1, i])
        ax2.plot(t_full[:-1], r['sigma_series'], '-', color=col, lw=1.4)
        ax2.set_xlabel('Time (h)', fontsize=8)
        if i == 0:
            ax2.set_ylabel('σ = a₁₁(t)', fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.spines[['top','right']].set_visible(False)

        ax2T = ax2.twinx()
        ax2T.plot(t_full[:-1], T_arr, '--', color='gray', lw=0.8, alpha=0.5)
        ax2T.set_ylabel('T (°C)', fontsize=7, color='gray')
        ax2T.tick_params(labelsize=6, colors='gray')

    fig.suptitle('Dy — Enhanced σ(T) models comparison (MAP via differential_evolution)',
                 fontsize=10, y=1.01)
    outpng = os.path.join(BASE, args.outpng)
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    fig.savefig(outpng, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {outpng}")

    # Save best theta to npz for follow-up TMCMC
    best_key = min(results, key=lambda k: results[k]['aic'])
    out_npz = os.path.join(BASE, f'_dy_map_{best_key}.npz')
    np.savez(out_npz, theta_MAP=results[best_key]['theta_map'],
             aic=results[best_key]['aic'], bic=results[best_key]['bic'],
             rmse=results[best_key]['rmse'])
    print(f"Best model: {results[best_key]['label']} → saved MAP to {out_npz}")


if __name__ == '__main__':
    main()
