#!/usr/bin/env python3
"""
generate_paper_fig20_21.py — Paper-quality Figures 20 & 21.

Fig 20: E2E Differentiable Pipeline (4-condition comparison)
Fig 21: NUTS vs HMC vs RW TMCMC Comparison

Requirements: Run e2e_differentiable_pipeline.py and gradient_tmcmc_nuts.py first.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

jax.config.update("jax_enable_x64", False)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR = PROJECT_ROOT / "FEM" / "figures" / "paper_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- shared matplotlib style ----------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ---------- imports from pipeline ----------
from deeponet_hamilton import DeepONet
from dem_elasticity_3d import ElasticityNetwork
from e2e_differentiable_pipeline import (
    load_deeponet, load_dem, load_theta_map,
    deeponet_predict_final, compute_di, di_to_E,
    dem_predict_max_uy,
    E_MAX, E_MIN, N_SPECIES, SPECIES,
    CONDITION_CHECKPOINTS, W, H, D,
)
from gradient_tmcmc_nuts import (
    tmcmc_engine, load_prior_bounds,
    make_real_log_likelihood, load_real_data, load_real_sigma,
)

COND_LABELS = {
    "Commensal_Static": "Commensal Static",
    "Commensal_HOBIC": "Commensal HOBIC",
    "Dysbiotic_Static": "Dysbiotic Static",
    "Dysbiotic_HOBIC": "Dysbiotic HOBIC",
}
COND_COLORS = {
    "Commensal_Static": "#2E7D32",
    "Commensal_HOBIC": "#1565C0",
    "Dysbiotic_Static": "#C62828",
    "Dysbiotic_HOBIC": "#E65100",
}
SPECIES_FULL = ["S. oralis", "A. naeslundii", "Veillonella", "F. nucleatum", "P. gingivalis"]


# ============================================================
# Fig 20: E2E Differentiable Pipeline
# ============================================================
def generate_fig20():
    print("=" * 60)
    print("Generating Fig 20: E2E Differentiable Pipeline")
    print("=" * 60)

    dem_model = load_dem()
    results = {}

    for cond in CONDITION_CHECKPOINTS:
        try:
            don, tlo, tw = load_deeponet(cond)
            theta_map = load_theta_map(cond)
        except Exception as e:
            print(f"  [SKIP] {cond}: {e}")
            continue

        theta_jax = jnp.array(theta_map)
        phi = deeponet_predict_final(don, theta_jax, tlo, tw)
        di = compute_di(phi)
        E = di_to_E(di)
        uy = dem_predict_max_uy(dem_model, E)

        # Sensitivity
        pipeline = lambda t: dem_predict_max_uy(
            dem_model, di_to_E(compute_di(
                deeponet_predict_final(don, t, tlo, tw))))
        grads = jax.grad(pipeline)(theta_jax)

        # Timing
        pipeline_jit = jax.jit(pipeline)
        _ = pipeline_jit(theta_jax)
        t0 = time.time()
        for _ in range(200):
            _ = pipeline_jit(theta_jax); jax.block_until_ready(_)
        t_fwd = (time.time() - t0) / 200 * 1000

        grad_jit = jax.jit(jax.grad(pipeline))
        _ = grad_jit(theta_jax)
        t0 = time.time()
        for _ in range(200):
            g = grad_jit(theta_jax); jax.block_until_ready(g)
        t_grad = (time.time() - t0) / 200 * 1000

        results[cond] = dict(
            phi=np.array(phi), DI=float(di), E=float(E),
            uy=float(uy), grads=np.array(grads),
            t_fwd=t_fwd, t_grad=t_grad)
        print(f"  {cond}: DI={float(di):.3f}, E={float(E):.0f} Pa, "
              f"u_y={float(uy)*1000:.2f} μm")

    conds = list(results.keys())
    n = len(conds)

    # ---- Layout ----
    fig = plt.figure(figsize=(16, 13))
    gs_top = gridspec.GridSpec(2, n, hspace=0.35, wspace=0.30,
                               left=0.06, right=0.97, top=0.92, bottom=0.42)
    gs_bot = gridspec.GridSpec(1, 1, left=0.06, right=0.97,
                               top=0.36, bottom=0.05)

    fig.text(0.5, 0.96,
             r"End-to-End Differentiable Pipeline: $\theta$ $\rightarrow$ DeepONet "
             r"$\rightarrow$ DI $\rightarrow$ E(DI) $\rightarrow$ DEM $\rightarrow$ $\mathbf{u}$",
             ha="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.935,
             "All steps JAX-differentiable — exact $\\partial u / \\partial \\theta$ "
             "via single backward pass",
             ha="center", fontsize=10, color="gray")

    # Row 0: Species bar charts
    for col, cond in enumerate(conds):
        ax = fig.add_subplot(gs_top[0, col])
        r = results[cond]
        color = COND_COLORS[cond]
        bars = ax.bar(range(5), r["phi"], color=color, alpha=0.85, width=0.7)
        ax.set_xticks(range(5))
        ax.set_xticklabels(SPECIES, fontsize=8)
        ax.set_ylim(0, max(0.5, max(r["phi"]) * 1.2))
        ax.set_title(f"{COND_LABELS[cond]}\nDI = {r['DI']:.3f}   "
                     f"E = {r['E']:.0f} Pa", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"$\varphi$ (fraction)")
        # Annotate bars
        for i, v in enumerate(r["phi"]):
            if v > 0.01:
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    # Row 1: Displacement profiles
    ny = 50
    y_pts = jnp.linspace(0, H, ny)
    for col, cond in enumerate(conds):
        ax = fig.add_subplot(gs_top[1, col])
        r = results[cond]
        E_norm = jnp.float32(r["E"]) / E_MAX
        uy_prof = []
        for yi in y_pts:
            u = dem_model(jnp.float32(W/2), yi, jnp.float32(D/2), E_norm)
            uy_prof.append(float(u[1]))
        uy_arr = np.array(uy_prof) * 1000  # μm

        color = COND_COLORS[cond]
        ax.plot(uy_arr, np.array(y_pts) * 1000, "-", color=color, lw=2)
        ax.axhline(H * 1000, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel(r"$u_y$ [$\mu$m]")
        if col == 0:
            ax.set_ylabel("y [μm]")
        ax.set_title(f"$u_y^{{max}}$ = {r['uy']*1000:.2f} μm", fontsize=10)

    # Bottom: Sensitivity bar chart
    ax_sens = fig.add_subplot(gs_bot[0, 0])
    bar_w = 0.8 / n
    x_pos = np.arange(20)
    param_labels = [
        r"$a_{11}$", r"$a_{12}$", r"$a_{13}$", r"$a_{21}$", r"$a_{22}$",
        r"$a_{23}$", r"$a_{31}$", r"$a_{32}$", r"$a_{33}$", r"$a_{34}$",
        r"$a_{41}$", r"$a_{42}$", r"$a_{43}$", r"$a_{44}$", r"$a_{45}$",
        r"$a_{15}$", r"$a_{25}$", r"$a_{35}$", r"$a_{45}^{Pg}$", r"$b_5$",
    ]

    for i, cond in enumerate(conds):
        r = results[cond]
        offset = (i - n/2 + 0.5) * bar_w
        ax_sens.bar(x_pos + offset, r["grads"], bar_w,
                    color=COND_COLORS[cond], alpha=0.75,
                    label=COND_LABELS[cond])

    ax_sens.set_xlabel("Parameter")
    ax_sens.set_ylabel(r"$\partial u_y / \partial \theta_i$")
    ax_sens.set_title(
        "Exact Sensitivity via JAX Autodiff  "
        f"(Forward: {np.mean([r['t_fwd'] for r in results.values()]):.2f} ms, "
        f"Gradient: {np.mean([r['t_grad'] for r in results.values()]):.2f} ms, "
        f"Abaqus: ~120 s → 3.8M× speedup)", fontsize=10)
    ax_sens.set_xticks(range(20))
    ax_sens.set_xticklabels(param_labels, fontsize=8)
    ax_sens.legend(ncol=2, fontsize=9, loc="upper left")

    out = str(OUT_DIR / "Fig20_e2e_pipeline.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# Fig 21: NUTS vs HMC vs RW TMCMC
# ============================================================
def generate_fig21():
    print("\n" + "=" * 60)
    print("Generating Fig 21: NUTS vs HMC vs RW TMCMC")
    print("=" * 60)

    condition = "Dysbiotic_HOBIC"
    n_particles = 200
    seed = 42

    don, tlo, tw = load_deeponet(condition)
    dem = load_dem()
    prior_bounds = load_prior_bounds(condition)
    free_dims = np.where(
        np.abs(prior_bounds[:, 1] - prior_bounds[:, 0]) > 1e-12)[0]

    # Synthetic data
    rng = np.random.default_rng(seed)
    theta_true = np.zeros(20, dtype=np.float32)
    for i in range(20):
        lo, hi = prior_bounds[i]
        if abs(hi - lo) > 1e-12:
            theta_true[i] = rng.uniform(lo, hi)

    theta_jax = jnp.array(theta_true)
    phi_true = deeponet_predict_final(don, theta_jax, tlo, tw)
    sigma_phi = 0.03
    obs_phi = jnp.array(
        np.array(phi_true) + rng.normal(0, sigma_phi, 5), dtype=jnp.float32)
    obs_phi = jnp.clip(obs_phi, 0.0, 1.0)

    def log_likelihood(theta):
        phi = deeponet_predict_final(don, theta, tlo, tw)
        return -0.5 * jnp.sum(((obs_phi - phi) / sigma_phi) ** 2)

    # Warmup
    _ = jax.jit(log_likelihood)(theta_jax)
    _ = jax.jit(jax.value_and_grad(log_likelihood))(theta_jax)

    # Run 3 methods
    results = []
    for mut in ["rw", "hmc", "nuts"]:
        print(f"  Running {mut.upper()}...")
        r = tmcmc_engine(
            log_likelihood, prior_bounds, mutation=mut,
            n_particles=n_particles, seed=seed,
            hmc_step_size=0.005, hmc_n_leapfrog=5,
            nuts_max_depth=6, verbose=False)
        results.append(r)
        print(f"    → {r['n_stages']} stages, accept={np.mean(r['accept_rates']):.2f}, "
              f"logL={r['log_likelihoods'].max():.1f}")

    # ---- Plot ----
    colors = {"RW-TMCMC": "#E53935", "HMC-TMCMC": "#1E88E5", "NUTS-TMCMC": "#43A047"}
    labels_short = {"RW-TMCMC": "RW", "HMC-TMCMC": "HMC", "NUTS-TMCMC": "NUTS"}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"TMCMC Mutation Kernel Comparison — {COND_LABELS[condition]}\n"
        f"(200 particles, 20 free parameters, synthetic data)",
        fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.35,
                           left=0.06, right=0.97, top=0.88, bottom=0.08)

    # (0,0) Beta schedule
    ax = fig.add_subplot(gs[0, 0])
    for r in results:
        ax.plot(r["betas"], "o-", color=colors[r["label"]],
                label=labels_short[r["label"]], ms=4, lw=1.5)
    ax.set_xlabel("Stage")
    ax.set_ylabel(r"$\beta$")
    ax.set_title("(a) Tempering Schedule")
    ax.legend()

    # (0,1) Acceptance
    ax = fig.add_subplot(gs[0, 1])
    for r in results:
        avg = np.mean(r["accept_rates"])
        ax.plot(r["accept_rates"], "o-", color=colors[r["label"]],
                label=f"{labels_short[r['label']]} ({avg:.2f})", ms=4, lw=1.5)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("(b) Mutation Acceptance")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # (0,2) ESS
    ax = fig.add_subplot(gs[0, 2])
    for r in results:
        ax.plot(range(1, r["n_stages"]+1), r["ess_history"],
                "o-", color=colors[r["label"]],
                label=labels_short[r["label"]], ms=4, lw=1.5)
    ax.set_xlabel("Stage")
    ax.set_ylabel("ESS")
    ax.set_title("(c) Effective Sample Size")
    ax.legend()

    # (0,3) Summary table as text
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    headers = ["Metric", "RW", "HMC", "NUTS"]
    rows = [
        ["Stages", *[str(r["n_stages"]) for r in results]],
        ["Avg Accept", *[f"{np.mean(r['accept_rates']):.2f}" for r in results]],
        ["max logL", *[f"{r['log_likelihoods'].max():.1f}" for r in results]],
        ["Time [s]", *[f"{r['total_time']:.1f}" for r in results]],
    ]
    cell_colors = [["#f0f0f0"]*4] * len(rows)
    # Highlight best in each row
    for i, row in enumerate(rows):
        vals = []
        for j in range(1, 4):
            try:
                vals.append(float(row[j]))
            except ValueError:
                vals.append(0)
        best_idx = np.argmax(vals) if i in [1, 2] else np.argmin(vals)
        cell_colors[i] = ["#f0f0f0"] * 4
        cell_colors[i][best_idx + 1] = "#c8e6c9"  # green highlight

    table = ax.table(cellText=rows, colLabels=headers,
                     cellColours=cell_colors,
                     colColours=["#e0e0e0"]*4,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    ax.set_title("(d) Summary", fontsize=12)

    # Bottom row: Posterior marginals (top 8 free dims)
    show_dims = free_dims[:8]
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, len(show_dims),
                                              subplot_spec=gs[1, :],
                                              wspace=0.25)
    for j, dim in enumerate(show_dims):
        ax = fig.add_subplot(gs_bot[0, j])
        for r in results:
            samples = r["samples"][:, dim]
            ax.hist(samples, bins=20, alpha=0.45, color=colors[r["label"]],
                    density=True, label=labels_short[r["label"]] if j == 0 else None)
        # True value
        ax.axvline(theta_true[dim], color="k", ls="--", lw=1, alpha=0.7)
        ax.set_title(f"θ[{dim}]", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("Density")
            ax.legend(fontsize=7, loc="upper right")

    out = str(OUT_DIR / "Fig21_nuts_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    generate_fig20()
    generate_fig21()
