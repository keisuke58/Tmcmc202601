# -*- coding: utf-8 -*-
"""
postprocess_3param.py — 8-panel visualization for (sigma, c_scale, phi_init) TMCMC results.

Shows MAP fit + 95% posterior predictive CI vs experimental data.
Also prints RMSE comparison: 1D → 2D(cscale) → 3D(cscale+phi_init).

Usage:
    python postprocess_3param.py \
        --indir _tmcmc_results_3param \
        --outpng figures/monospecies_8panel_3param.png \
        --n_pp_samples 200
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines
import matplotlib.patches

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from hamilton_monospecies_jax import simulate_monospecies_phibar_hist_dynamic_c
from estimate_monospecies_tmcmc import load_static_data, FELIX_SIGMA

TEMPS = [4, 8, 15, 20, 25, 35, 37, 40]
MAX_STEPS = {4: 4000, 8: 2000, 15: 1000, 20: 500,
             25: 300,  35: 200,  37: 200, 40: 450}
TIME_TO_STEP = 10.0
MIN_TO_H = 1.0 / 60.0   # display in hours


def simulate_phibar(sigma, c_scale, phi_init, n_steps, dt=1e-4, n_newton=6):
    return np.asarray(simulate_monospecies_phibar_hist_dynamic_c(
        jnp.float64(sigma), n_steps, dt=dt,
        c_scale=jnp.float64(c_scale), n_newton=n_newton,
        phi_init=jnp.float64(phi_init),
    ))


def profiled_predictions(phibar, obs_time, obs_mean, n_steps):
    step_idx = np.clip((np.array(obs_time) * TIME_TO_STEP).astype(int), 0, n_steps)
    sv = np.array(phibar)[step_idx]
    X = np.column_stack([sv, np.ones_like(sv)])
    coeff, _, _, _ = np.linalg.lstsq(X, np.array(obs_mean), rcond=None)
    return X @ coeff, coeff


def posterior_predictive_ci(samples, obs_time, obs_mean, n_steps,
                             dt=1e-4, n_newton=6, batch_size=50):
    """
    samples: (N, 3) — [sigma, c_scale, phi_init]
    Returns lo, hi (95% CI) at obs_time.
    """
    step_idx = np.clip((np.array(obs_time) * TIME_TO_STEP).astype(int), 0, n_steps)
    obs_mean_arr = np.array(obs_mean)
    all_preds = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        for sigma, c_scale, phi_init in batch:
            pb = simulate_phibar(float(sigma), float(c_scale), float(phi_init),
                                  n_steps, dt=dt, n_newton=n_newton)
            sv = pb[step_idx]
            X = np.column_stack([sv, np.ones_like(sv)])
            coeff, _, _, _ = np.linalg.lstsq(X, obs_mean_arr, rcond=None)
            all_preds.append(X @ coeff)

    preds = np.array(all_preds)
    return np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',  type=str, default='_tmcmc_results_3param')
    parser.add_argument('--outpng', type=str,
                        default='figures/monospecies_8panel_3param.png')
    parser.add_argument('--indir_1d', type=str,
                        default='_tmcmc_results_dyncc10_rw4000p',
                        help='1D run dir for RMSE comparison')
    parser.add_argument('--indir_2d', type=str,
                        default='_tmcmc_results_cscale',
                        help='2D (cscale) run dir for RMSE comparison')
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--n-newton', type=int, default=6)
    parser.add_argument('--n_pp_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    xlsx_path = os.path.join(os.path.dirname(__file__), 'raw data.xlsx')
    exp_data = load_static_data(xlsx_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.outpng)), exist_ok=True)

    rng = np.random.default_rng(args.seed)

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.32)

    summary_rows = []

    for idx_temp, temp in enumerate(TEMPS):
        ax = fig.add_subplot(gs[idx_temp // 4, idx_temp % 4])
        d = exp_data[temp]
        n_steps = MAX_STEPS[temp]

        npz_path = os.path.join(args.indir, f'samples_{temp}C.npz')
        npz = np.load(npz_path)
        samples      = npz['samples']           # (N, 3): [sigma, c_scale, phi_init]
        sigma_MAP    = float(npz['sigma_MAP'])
        c_scale_MAP  = float(npz['c_scale_MAP'])
        phi_init_MAP = float(npz['phi_init_MAP'])

        # MAP trajectory
        phibar_map = simulate_phibar(sigma_MAP, c_scale_MAP, phi_init_MAP, n_steps,
                                      dt=args.dt, n_newton=args.n_newton)
        pred_map_obs, coeff = profiled_predictions(phibar_map, d['time'], d['cfu_mean'], n_steps)
        rmse = float(np.sqrt(np.mean((pred_map_obs - d['cfu_mean'])**2)))

        t_full_h = np.arange(n_steps + 1) / TIME_TO_STEP * MIN_TO_H
        mapped_full = coeff[0] * phibar_map + coeff[1]
        ax.plot(t_full_h, mapped_full, '-', color='steelblue', lw=1.8, zorder=3)

        # Posterior predictive CI
        n_sel = min(args.n_pp_samples, len(samples))
        sel_idx = rng.choice(len(samples), size=n_sel, replace=False)
        lo, hi = posterior_predictive_ci(
            samples[sel_idx], d['time'], d['cfu_mean'], n_steps,
            dt=args.dt, n_newton=args.n_newton,
        )
        ax.fill_between(d['time'] * MIN_TO_H, lo, hi, alpha=0.25, color='steelblue')

        # Experimental data
        ax.scatter(d['time'] * MIN_TO_H, d['cfu_mean'], color='#e05a2b', s=18, zorder=4,
                   linewidths=0.4, edgecolors='white')

        title = (f'({list("abcdefgh")[idx_temp]}) {temp}°C  '
                 f'a₁₁={sigma_MAP:.2f}  c₀={c_scale_MAP:.1f}  φ₀={phi_init_MAP:.3f}  RMSE={rmse:.3f}')
        ax.set_title(title, fontsize=8, pad=5)
        ax.set_xlabel('Time (h)', fontsize=9)
        ax.set_ylabel('log₁₀(CFU/mL)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        if idx_temp == 0:
            ax.legend(fontsize=7, framealpha=0.7,
                      handles=[
                          matplotlib.lines.Line2D([], [], color='steelblue', lw=1.8, label='MAP fit'),
                          matplotlib.patches.Patch(color='steelblue', alpha=0.25, label='95% CI'),
                          matplotlib.lines.Line2D([], [], marker='o', color='w',
                                                  markerfacecolor='#e05a2b', ms=5, label='Experiment'),
                      ])

        summary_rows.append({
            'temp': temp,
            'sigma_MAP': sigma_MAP,
            'c_scale_MAP': c_scale_MAP,
            'phi_init_MAP': phi_init_MAP,
            'rmse': rmse,
        })
        print(f"  {temp}°C: σ={sigma_MAP:.3f}, c={c_scale_MAP:.3f}, φ₀={phi_init_MAP:.4f}, RMSE={rmse:.4f}")

    fig.suptitle(
        'Monospecies TMCMC — free params: $a_{11}$ (growth/attachment), '
        '$c_0$ [nutrient: $c(t)=c_0(1-\\phi\\psi)$], '
        '$\\phi_0$ (initial biofilm fraction)',
        fontsize=10, y=1.02)
    fig.savefig(args.outpng, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {args.outpng}")

    # 3-way RMSE comparison
    print(f"\n{'='*85}")
    print(f"{'Temp':>5s} | {'Felix σ':>8s} | {'1D RMSE':>8s} | {'2D RMSE':>8s} | "
          f"{'3D RMSE':>8s} | {'Δ(2D→3D)':>9s}")
    print('-' * 85)

    for row in summary_rows:
        t = row['temp']
        d = exp_data[t]
        n_steps = MAX_STEPS[t]

        rmse_1d = float('nan')
        if os.path.isdir(args.indir_1d):
            p1 = os.path.join(args.indir_1d, f'samples_{t}C.npz')
            if os.path.exists(p1):
                d1 = np.load(p1)
                s1 = float(d1['sigma_MAP'])
                pb1 = simulate_phibar(s1, 10.0, 0.1, n_steps, dt=args.dt, n_newton=args.n_newton)
                pr1, _ = profiled_predictions(pb1, d['time'], d['cfu_mean'], n_steps)
                rmse_1d = float(np.sqrt(np.mean((pr1 - d['cfu_mean'])**2)))

        rmse_2d = float('nan')
        if os.path.isdir(args.indir_2d):
            p2 = os.path.join(args.indir_2d, f'samples_{t}C.npz')
            if os.path.exists(p2):
                d2 = np.load(p2)
                s2 = float(d2['sigma_MAP'])
                c2 = float(d2['c_scale_MAP'])
                pb2 = simulate_phibar(s2, c2, 0.1, n_steps, dt=args.dt, n_newton=args.n_newton)
                pr2, _ = profiled_predictions(pb2, d['time'], d['cfu_mean'], n_steps)
                rmse_2d = float(np.sqrt(np.mean((pr2 - d['cfu_mean'])**2)))

        delta = row['rmse'] - rmse_2d
        print(f"{t:>4d}°C | {FELIX_SIGMA[t]:>8.2f} | {rmse_1d:>8.4f} | {rmse_2d:>8.4f} | "
              f"{row['rmse']:>8.4f} | {delta:>+9.4f}")


if __name__ == '__main__':
    main()
