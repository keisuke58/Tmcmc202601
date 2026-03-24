# -*- coding: utf-8 -*-
"""
postprocess_dy_enhanced.py — 拡張 a₁₁(T) モデル比較可視化

各モデルの TMCMC 結果 (npz) があれば自動ロード、なければ hardcoded MAP を使用。
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
from estimate_monospecies_tmcmc import load_dynamic_data_with_reps
from estimate_monospecies_tmcmc_dy_enhanced import (
    MODEL_REGISTRY, TIME_STEP, DT, N_NEWTON, C_SCALE
)

BASE      = os.path.dirname(__file__)
MIN_TO_H  = 1.0 / 60.0

# ── Hardcoded MAP fallbacks (TMCMC results, vancouver01 2026-03-25) ─────────
FALLBACK_MAP = {
    'rat5':     np.array([0.327, -0.810, 47.49, 1.258, 2.21e-4]),
    'ctm5':     None,   # ファイルなければスキップ
    'rat_mem':  None,
    'rat_asym': None,
    'rat_full': None,
    'ctm_mem':  None,
}

# ── Result directories (relative to BASE) ───────────────────────────────────
RESULT_DIRS = {
    'rat5':     '_tmcmc_results_dy_ratkowsky_rw4000p',
    'ctm5':     '_tmcmc_results_dy_ctm5',
    'rat_mem':  '_tmcmc_results_dy_rat_mem',
    'rat_asym': '_tmcmc_results_dy_rat_asym',
    'rat_full': '_tmcmc_results_dy_rat_full',
    'ctm_mem':  '_tmcmc_results_dy_ctm_mem',
}

MODEL_LABELS = {
    'rat5':     'Ratkowsky (5p)  ✓ best',
    'ctm5':     'CTM/Rosso (5p)',
    'rat_mem':  'Rat + memory τ (6p)',
    'rat_asym': 'Rat + asym lag (6p)',
    'rat_full': 'Rat + τ + asym (7p)',
    'ctm_mem':  'CTM + memory τ (6p)',
}

# AIC/BIC results from TMCMC (2026-03-25)
MODEL_STATS = {
    'rat5':     dict(logL=45.46, k=5,  aic=-80.9, bic=-75.0),
    'rat_mem':  dict(logL=45.60, k=6,  aic=-79.2, bic=-72.1),
    'ctm5':     dict(logL=44.10, k=5,  aic=-78.2, bic=-72.3),
    'rat_asym': dict(logL=25.19, k=6,  aic=-38.4, bic=-31.3),
    'rat_full': dict(logL=30.26, k=7,  aic=-46.5, bic=-38.3),
    'ctm_mem':  dict(logL=None,  k=6,  aic=None,  bic=None),
}

COLORS = {
    'rat5':     '#1f77b4',
    'ctm5':     '#2ca02c',
    'rat_mem':  '#ff7f0e',
    'rat_asym': '#d62728',
    'rat_full': '#9467bd',
    'ctm_mem':  '#8c564b',
}

ORDER = ['rat5', 'ctm5', 'rat_mem', 'rat_asym', 'rat_full', 'ctm_mem']


def load_map(model_key):
    result_dir = os.path.join(BASE, RESULT_DIRS[model_key])
    # Try .npz with standard theta_MAP key
    for fname in [f'samples_Dy_{model_key}.npz', f'samples_Dy_ratkowsky.npz']:
        path = os.path.join(result_dir, fname)
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            if 'theta_MAP' in d:
                return np.array(d['theta_MAP'])
    return FALLBACK_MAP.get(model_key)


def build_sigma_series(model_key, theta, T_jax, n_steps):
    cfg     = MODEL_REGISTRY[model_key]
    dT_jax  = jnp.pad(jnp.diff(T_jax) * TIME_STEP, (1, 0))
    return np.asarray(cfg['builder'](jnp.array(theta, dtype=jnp.float64), T_jax, dT_jax))


def simulate(sigma_series, n_steps):
    return np.asarray(simulate_monospecies_phibar_hist_dynamic_c_sigma_series(
        jnp.array(sigma_series, dtype=jnp.float64),
        n_steps, dt=DT, c_scale=C_SCALE, n_newton=N_NEWTON,
    ))


def profiled_fit(phibar, obs_time, obs_cfu, n_steps):
    step_idx = np.clip((np.array(obs_time) * TIME_STEP).astype(int), 0, n_steps)
    sv = phibar[step_idx]
    X  = np.column_stack([sv, np.ones_like(sv)])
    coeff, _, _, _ = np.linalg.lstsq(X, np.array(obs_cfu), rcond=None)
    return X @ coeff, coeff


def main():
    xlsx_path = os.path.join(BASE, 'raw data.xlsx')
    dy = load_dynamic_data_with_reps(xlsx_path, sheet='Dy')

    t_max   = float(max(np.max(dy['time']), np.max(dy['temp_time'])))
    n_steps = int(np.ceil(t_max * TIME_STEP))
    t_grid  = np.arange(n_steps + 1, dtype=float) / TIME_STEP
    T_grid  = np.interp(t_grid, dy['temp_time'], dy['temp_C'])
    T_jax   = jnp.array(T_grid[:-1], dtype=jnp.float64)

    t_h      = np.array(dy['time']) * MIN_TO_H
    t_full_h = t_grid * MIN_TO_H

    # ── Load MAP and compute predictions ────────────────────────────────────
    results = {}
    for key in ORDER:
        theta = load_map(key)
        if theta is None:
            print(f"  skip {key}: no result file and no fallback MAP")
            continue
        ss    = build_sigma_series(key, theta, T_jax, n_steps)
        pb    = simulate(ss, n_steps)
        pred, coeff = profiled_fit(pb, dy['time'], dy['cfu_mean'], n_steps)
        rmse  = float(np.sqrt(np.mean((pred - np.array(dy['cfu_mean']))**2)))
        results[key] = dict(theta=theta, sigma=ss, phibar=pb,
                            pred=pred, coeff=coeff, rmse=rmse)
        st = MODEL_STATS[key]
        print(f"  {key:<12} RMSE={rmse:.4f}  logL={st['logL']}  "
              f"AIC={st['aic']}  BIC={st['bic']}")

    if not results:
        print("No results available.")
        return

    n_avail = len(results)
    fig = plt.figure(figsize=(6 * n_avail, 9))
    gs  = gridspec.GridSpec(2, n_avail, figure=fig, hspace=0.5, wspace=0.35)

    for col, key in enumerate(k for k in ORDER if k in results):
        r   = results[key]
        col_c = COLORS[key]
        st  = MODEL_STATS[key]
        label = MODEL_LABELS[key]

        # Top: fit vs data
        ax = fig.add_subplot(gs[0, col])
        mapped = r['coeff'][0] * r['phibar'] + r['coeff'][1]
        ax.plot(t_full_h, mapped, '-', color=col_c, lw=1.8, zorder=3)
        ax.scatter(t_h, dy['cfu_mean'], color='#e05a2b', s=22, zorder=4,
                   edgecolors='white', linewidths=0.4)
        for ti, reps in dy['reps_by_time'].items():
            ax.plot([ti * MIN_TO_H]*len(reps), reps, '_',
                    color='#e05a2b', ms=6, alpha=0.5, zorder=5)
        aic_str = f"AIC={st['aic']}" if st['aic'] is not None else "AIC=—"
        ax.set_title(f'{label}\nRMSE={r["rmse"]:.3f}  {aic_str}', fontsize=8.5)
        ax.set_xlabel('Time (h)', fontsize=8)
        ax.set_ylabel('log₁₀(CFU/mL)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

        # Bottom: a₁₁(t) profile
        ax2 = fig.add_subplot(gs[1, col])
        ax2.plot(t_full_h[:-1], r['sigma'], '-', color=col_c, lw=1.4)
        ax2.set_xlabel('Time (h)', fontsize=8)
        ax2.set_ylabel('a₁₁(T,t)', fontsize=8)
        ax2.set_title(f'a₁₁ profile — {key}', fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.spines[['top', 'right']].set_visible(False)

        # Temperature overlay on a₁₁ plot
        ax_T = ax2.twinx()
        ax_T.plot(t_grid[:-1] * MIN_TO_H, T_grid[:-1],
                  '--', color='gray', lw=0.8, alpha=0.55, label='T(t)')
        ax_T.set_ylabel('T (°C)', fontsize=7, color='gray')
        ax_T.tick_params(labelsize=6, colors='gray')

    fig.suptitle('Dy 動的温度 — 拡張 a₁₁(T,t) モデル比較  (TMCMC MAP, N=24)',
                 fontsize=10, y=1.01)

    outpng = os.path.join(BASE, 'figures', 'dy_enhanced_model_comparison.png')
    os.makedirs(os.path.dirname(outpng), exist_ok=True)
    fig.savefig(outpng, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {outpng}")


if __name__ == '__main__':
    main()
