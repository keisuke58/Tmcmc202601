# -*- coding: utf-8 -*-
"""
postprocess_dy.py — 動的温度 (Dy) 実験に対する3モデル比較可視化

Models:
  1-var    : sigma(t) = k * interp(sigma_knots, T(t))          [1 param]
  multi    : sigma(t) = k_i * interp(sigma_knots_i, T(t))      [8 params]
  ratkowsky: sigma(T) = [b*(T-T_min)*(1-exp(c*(T-T_max)))]^2  [5 params]

Usage:
    python postprocess_dy.py
"""

import os, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from hamilton_monospecies_jax import simulate_monospecies_phibar_hist_dynamic_c_sigma_series
from estimate_monospecies_tmcmc import load_dynamic_data_with_reps, load_sigma_knots_from_dir

BASE = os.path.dirname(__file__)
N_STEPS    = 1500
TIME_STEP  = 10.0
MIN_TO_H   = 1.0 / 60.0
DT         = 1e-4
N_NEWTON   = 6
C_SCALE    = 10.0
TEMPS_8    = [4.0, 8.0, 15.0, 20.0, 25.0, 35.0, 37.0, 40.0]


def build_sigma_series_1var(k, sigma_knots, T_grid):
    temps  = np.array(sorted(sigma_knots.keys()), dtype=float)
    sigmas = np.array([sigma_knots[t] for t in temps], dtype=float)
    sigma_base = np.interp(T_grid[:-1], temps, sigmas, left=sigmas[0], right=sigmas[-1])
    return np.clip(k * sigma_base, 1.0, 500.0)


def build_sigma_series_multi(theta, sigma_knots, T_grid):
    temps    = np.array(sorted(sigma_knots.keys()), dtype=float)
    base_arr = np.array([sigma_knots[t] for t in temps], dtype=float)
    new_knots = np.array(theta) * base_arr

    idx = np.searchsorted(temps, T_grid[:-1]) - 1
    idx = np.clip(idx, 0, len(temps) - 2)
    t0 = temps[idx]; t1 = temps[idx + 1]
    w  = np.clip((T_grid[:-1] - t0) / (t1 - t0), 0.0, 1.0)
    sigma_series = new_knots[idx] + w * (new_knots[idx + 1] - new_knots[idx])
    return np.clip(sigma_series, 1.0, 500.0)


def build_sigma_series_ratkowsky(theta, T_grid):
    b_rat, T_min, T_max, c_rat, k_lag = theta
    T_g = T_grid[:-1]
    T_diff_min = np.maximum(T_g - T_min, 0.0)
    T_diff_max = T_g - T_max
    exp_term   = np.exp(np.clip(c_rat * T_diff_max, -50.0, 0.0))
    sqrt_sigma = np.maximum(b_rat * T_diff_min * (1.0 - exp_term), 0.0)
    sigma_base = sqrt_sigma ** 2
    dT_dt      = np.concatenate([[0.0], np.diff(T_g)]) * TIME_STEP
    lag_penalty = 1.0 / (1.0 + k_lag * np.abs(dT_dt))
    return np.clip(sigma_base * lag_penalty, 0.1, 500.0)


def simulate(sigma_series):
    return np.asarray(simulate_monospecies_phibar_hist_dynamic_c_sigma_series(
        jnp.array(sigma_series, dtype=jnp.float64),
        N_STEPS, dt=DT, c_scale=C_SCALE, n_newton=N_NEWTON,
    ))


def profiled_fit(phibar, obs_time, obs_cfu):
    step_idx = np.clip((np.array(obs_time) * TIME_STEP).astype(int), 0, N_STEPS)
    sv = phibar[step_idx]
    X  = np.column_stack([sv, np.ones_like(sv)])
    coeff, _, _, _ = np.linalg.lstsq(X, np.array(obs_cfu), rcond=None)
    return X @ coeff, coeff


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    xlsx_path = os.path.join(BASE, 'raw data.xlsx')
    dy = load_dynamic_data_with_reps(xlsx_path, sheet='Dy')
    t_grid = np.arange(N_STEPS + 1) / TIME_STEP
    T_grid = np.interp(t_grid, dy['temp_time'], dy['temp_C'])
    t_h = dy['time'] * MIN_TO_H
    t_full_h = t_grid * MIN_TO_H

    sigma_knots = load_sigma_knots_from_dir(
        os.path.join(BASE, '_tmcmc_results_dyncc10_rw4000p'))

    # ── Load MAP values ────────────────────────────────────────────────────────
    d1 = np.load(os.path.join(BASE, '_tmcmc_results_dy_1var_rw4000p',   'samples_Dy_k.npz'),       allow_pickle=True)
    dm = np.load(os.path.join(BASE, '_tmcmc_results_dy_multi_rw4000p',  'samples_Dy_multi.npz'),   allow_pickle=True)
    dr = np.load(os.path.join(BASE, '_tmcmc_results_dy_ratkowsky_rw4000p', 'samples_Dy_ratkowsky.npz'), allow_pickle=True)

    k_MAP      = float(d1['k_MAP'])
    multi_MAP  = np.array(dm['theta_MAP'])
    rat_MAP    = np.array(dr['theta_MAP'])

    # ── Simulate ────────────────────────────────────────────────────────────────
    ss_1var  = build_sigma_series_1var(k_MAP,     sigma_knots, T_grid)
    ss_multi = build_sigma_series_multi(multi_MAP, sigma_knots, T_grid)
    ss_rat   = build_sigma_series_ratkowsky(rat_MAP, T_grid)

    pb_1var  = simulate(ss_1var)
    pb_multi = simulate(ss_multi)
    pb_rat   = simulate(ss_rat)

    pred_1var,  c1 = profiled_fit(pb_1var,  dy['time'], dy['cfu_mean'])
    pred_multi, cm = profiled_fit(pb_multi, dy['time'], dy['cfu_mean'])
    pred_rat,   cr = profiled_fit(pb_rat,   dy['time'], dy['cfu_mean'])

    rmse_1var  = float(np.sqrt(np.mean((pred_1var  - dy['cfu_mean'])**2)))
    rmse_multi = float(np.sqrt(np.mean((pred_multi - dy['cfu_mean'])**2)))
    rmse_rat   = float(np.sqrt(np.mean((pred_rat   - dy['cfu_mean'])**2)))

    print(f"1-var   k={k_MAP:.4f}                                  RMSE={rmse_1var:.4f}")
    print(f"multi   k=[{', '.join(f'{v:.2f}' for v in multi_MAP)}]  RMSE={rmse_multi:.4f}")
    print(f"Ratkowsky b={rat_MAP[0]:.3f} T_min={rat_MAP[1]:.2f} T_max={rat_MAP[2]:.2f} "
          f"c={rat_MAP[3]:.3f} k_lag={rat_MAP[4]:.5f}  RMSE={rmse_rat:.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    colors = ['steelblue', '#2ca02c', '#d62728']
    labels_model = ['1-var (k=1)', '8-var (multi)', 'Ratkowsky']
    phibars = [pb_1var, pb_multi, pb_rat]
    coeffs  = [c1, cm, cr]
    rmses   = [rmse_1var, rmse_multi, rmse_rat]
    sigma_ss = [ss_1var, ss_multi, ss_rat]

    for i, (col, lab, pb, coeff, rmse, ss) in enumerate(
            zip(colors, labels_model, phibars, coeffs, rmses, sigma_ss)):
        ax = fig.add_subplot(gs[0, i])
        mapped = coeff[0] * pb + coeff[1]
        ax.plot(t_full_h, mapped, '-', color=col, lw=1.8, zorder=3, label='MAP fit')
        ax.scatter(t_h, dy['cfu_mean'], color='#e05a2b', s=20, zorder=4,
                   edgecolors='white', linewidths=0.4, label='Experiment')
        # Error bars from reps
        for ti, reps in dy['reps_by_time'].items():
            ax.plot([ti * MIN_TO_H]*len(reps), reps, '_', color='#e05a2b',
                    ms=6, alpha=0.5, zorder=5)
        ax.set_title(f'({list("abc")[i]}) {lab}  RMSE={rmse:.3f}', fontsize=10)
        ax.set_xlabel('Time (h)', fontsize=9)
        ax.set_ylabel('log₁₀(CFU/mL)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.7)

        # σ(t) profile
        ax2 = fig.add_subplot(gs[1, i])
        ax2.plot(t_full_h[:-1], ss, '-', color=col, lw=1.4)
        ax2.set_xlabel('Time (h)', fontsize=9)
        ax2.set_ylabel('σ (a₁₁)', fontsize=9)
        ax2.set_title(f'σ(t) profile — {lab}', fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.spines[['top', 'right']].set_visible(False)

    # Temperature overlay on σ plots
    ax_T = fig.axes[3].twinx()
    ax_T.plot(t_grid[:-1] * MIN_TO_H, T_grid[:-1], '--', color='gray',
              lw=1.0, alpha=0.6, label='T(t)')
    ax_T.set_ylabel('Temp (°C)', fontsize=8, color='gray')
    ax_T.tick_params(labelsize=7, colors='gray')

    fig.suptitle('Dy (動的温度) データへのフィット — 3モデル比較\n'
                 '1-var: σ(t)=k·σₛₜₐₜ(T)   |   multi: σ(t)=kᵢ·σₛₜₐₜ(T) (8knots)   |   '
                 'Ratkowsky: σ(T)=[b(T-Tₘᵢₙ)(1-exp(c(T-Tₘₐₓ)))]²',
                 fontsize=9, y=1.01)

    outpng = os.path.join(BASE, 'figures', 'dy_3model_comparison.png')
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    fig.savefig(outpng, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {outpng}")

    print(f"\n{'='*55}")
    print(f"{'Model':<15} | {'Params':>6} | {'RMSE':>7} | {'logL':<12}")
    print('-'*55)
    # log-likelihoods from samples
    lL1 = float(np.max(d1['log_likelihoods']))
    lLm = float(np.max(dm['log_likelihoods']))
    lLr = float(np.max(dr['log_likelihoods']))
    N = 24
    for name, k_p, rmse, lL in [
        ('1-var',    1, rmse_1var,  lL1),
        ('multi',    8, rmse_multi, lLm),
        ('ratkowsky',5, rmse_rat,   lLr),
    ]:
        aic = 2*k_p - 2*lL
        bic = np.log(N)*k_p - 2*lL
        print(f"{name:<15} | {k_p:>6} | {rmse:>7.4f} | logL={lL:.2f}  AIC={aic:.1f}  BIC={bic:.1f}")


if __name__ == '__main__':
    main()
