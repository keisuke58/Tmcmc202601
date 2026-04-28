# -*- coding: utf-8 -*-
"""
postprocess_cscale.py — 8-panel visualization for (sigma, c_scale) TMCMC results.

Shows MAP fit + 95% posterior predictive CI vs experimental data.
Also prints comparison table: 1D (c_scale=10 fixed) vs 2D (c_scale free).

Usage:
    python postprocess_cscale.py \
        --indir _tmcmc_results_cscale \
        --outpng figures/monospecies_8panel_cscale.png \
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
PHI_INIT = 0.1


def simulate_phibar(sigma, c_scale, n_steps, dt=1e-4, n_newton=6):
    return np.asarray(simulate_monospecies_phibar_hist_dynamic_c(
        jnp.float64(sigma), n_steps, dt=dt,
        c_scale=jnp.float64(c_scale), n_newton=n_newton, phi_init=PHI_INIT,
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
    samples: (N, 2) — [sigma, c_scale]
    Returns lo, hi (95% CI) at obs_time.
    """
    step_idx = np.clip((np.array(obs_time) * TIME_TO_STEP).astype(int), 0, n_steps)
    obs_mean_arr = np.array(obs_mean)
    all_preds = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        for sigma, c_scale in batch:
            pb = simulate_phibar(float(sigma), float(c_scale), n_steps, dt=dt, n_newton=n_newton)
            sv = pb[step_idx]
            X = np.column_stack([sv, np.ones_like(sv)])
            coeff, _, _, _ = np.linalg.lstsq(X, obs_mean_arr, rcond=None)
            all_preds.append(X @ coeff)

    preds = np.array(all_preds)
    return np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir',  type=str, default='_tmcmc_results_cscale')
    parser.add_argument('--outpng', type=str,
                        default='figures/monospecies_8panel_cscale.png')
    parser.add_argument('--indir_1d', type=str,
                        default='_tmcmc_results_dyncc10_rw4000p',
                        help='1D run dir for RMSE comparison')
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
        samples    = npz['samples']          # (N, 2): [sigma, c_scale]
        sigma_MAP   = float(npz['sigma_MAP'])
        c_scale_MAP = float(npz['c_scale_MAP'])

        # MAP trajectory
        phibar_map = simulate_phibar(sigma_MAP, c_scale_MAP, n_steps,
                                      dt=args.dt, n_newton=args.n_newton)
        pred_map_obs, coeff = profiled_predictions(phibar_map, d['time'], d['cfu_mean'], n_steps)
        rmse = float(np.sqrt(np.mean((pred_map_obs - d['cfu_mean'])**2)))

        t_full = np.arange(n_steps + 1) / TIME_TO_STEP
        mapped_full = coeff[0] * phibar_map + coeff[1]
        ax.plot(t_full, mapped_full, '-', color='steelblue', lw=1.8, zorder=3)

        # Posterior predictive CI
        n_sel = min(args.n_pp_samples, len(samples))
        sel_idx = rng.choice(len(samples), size=n_sel, replace=False)
        lo, hi = posterior_predictive_ci(
            samples[sel_idx], d['time'], d['cfu_mean'], n_steps,
            dt=args.dt, n_newton=args.n_newton,
        )
        ax.fill_between(d['time'], lo, hi, alpha=0.25, color='steelblue')

        # Experimental data
        reps_by_time = {}
        for t_pt, v in zip(
            [float(t) for t in d['time']],
            d['cfu_mean'],  # use cfu_mean as proxy if reps not available
        ):
            reps_by_time.setdefault(t_pt, []).append(v)

        ax.scatter(d['time'], d['cfu_mean'], color='#e05a2b', s=18, zorder=4,
                   linewidths=0.4, edgecolors='white')

        title = (f'({list("abcdefgh")[idx_temp]}) {temp}°C  '
                 f'a₁₁={sigma_MAP:.2f} c={c_scale_MAP:.1f}  RMSE={rmse:.3f}')
        ax.set_title(title, fontsize=9, pad=5)
        ax.set_xlabel('Time (min)', fontsize=9)
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
            'rmse': rmse,
        })
        print(f"  {temp}°C: a₁₁={sigma_MAP:.3f}, c_scale={c_scale_MAP:.3f}, RMSE={rmse:.4f}")

    fig.suptitle('Monospecies TMCMC — (a₁₁, c_scale) free  [dyncc, RW]',
                 fontsize=12, y=1.01)
    fig.savefig(args.outpng, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {args.outpng}")

    # Compare with 1D (c_scale=10 fixed)
    if os.path.isdir(args.indir_1d):
        print(f"\n{'='*75}")
        print(f"{'Temp':>5s} | {'Felix σ':>8s} | {'1D σ (c=10)':>11s} | "
              f"{'2D σ':>8s} | {'2D c':>8s} | {'RMSE_1D':>8s} | {'RMSE_2D':>8s}")
        print('-' * 75)
        for row in summary_rows:
            t = row['temp']
            npz1d_path = os.path.join(args.indir_1d, f'samples_{t}C.npz')
            if os.path.exists(npz1d_path):
                d1 = np.load(npz1d_path)
                sigma1d = float(d1['sigma_MAP'])
                # Compute 1D RMSE
                d = exp_data[t]
                n_steps = MAX_STEPS[t]
                pb1d = simulate_phibar(sigma1d, 10.0, n_steps, dt=args.dt, n_newton=args.n_newton)
                pred1d, _ = profiled_predictions(pb1d, d['time'], d['cfu_mean'], n_steps)
                rmse1d = float(np.sqrt(np.mean((pred1d - d['cfu_mean'])**2)))
            else:
                sigma1d, rmse1d = float('nan'), float('nan')
            print(f"{t:>4d}°C | {FELIX_SIGMA[t]:>8.2f} | {sigma1d:>11.3f} | "
                  f"{row['sigma_MAP']:>8.3f} | {row['c_scale_MAP']:>8.3f} | "
                  f"{rmse1d:>8.4f} | {row['rmse']:>8.4f}")


if __name__ == '__main__':
    main()
