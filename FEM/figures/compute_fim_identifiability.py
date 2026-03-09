#!/usr/bin/env python3
"""
compute_fim_identifiability.py
===============================
Fisher Information Matrix (FIM) analysis at MAP θ for 4 conditions.

Computes:
  J = ∂ŷ/∂θ  (Jacobian of forward model at MAP)
  FIM = Jᵀ J / σ²
  Eigenvalues, condition number, Cramér-Rao lower bounds.

Usage:
    python compute_fim_identifiability.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data_5species", "main"))

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hamilton_ode_jax import simulate_0d

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MAP theta ──
THETA_MAP = {
    "CS": [
        1.536236640466116,
        1.3880556313842831,
        1.9197954201188194,
        2.150483767115132,
        2.2316741803232696,
        1.0296774349511466,
        1.0417656712645311,
        1.623797304470055,
        2.7544371454844288,
        0.5935035606458788,
        1.847419819245041,
        1.4271288753237532,
        2.6944938956805666,
        1.0717506611299759,
        2.6218711407667232,
        1.8829404991239027,
        0.7970839320495732,
        2.0153274210891707,
        1.3656251356446887,
        2.7882575113018295,
    ],
    "CH": [
        1.5056652065934646,
        1.7542905969328197,
        1.9285209675150043,
        0.9081033065025761,
        2.4734870100070907,
        0.874843634768725,
        1.910581807797522,
        0.9183158182889031,
        0.9975507837370292,
        2.034641584116253,
        2.2596470369326314,
        0.4820223351956451,
        2.2650288291853045,
        0.2003634980279898,
        0.8860325112425053,
        1.1985803901565544,
        0.49272572524761205,
        2.6236922504865094,
        2.08725104261472,
        1.6615603976240991,
    ],
    "DH": [
        0.5205432728346953,
        1.8207970826367483,
        0.7441415924024912,
        2.079550814896673,
        2.4150726454161093,
        0.02381842698048997,
        -0.4405399465360915,
        2.8055397562966213,
        4.988259496272629,
        0.8919944578434205,
        -0.4394405153313352,
        -0.25301273261222956,
        1.653078969987149,
        1.42047351260726,
        0.009783074923792157,
        0.1909563915343552,
        1.5646252065632602,
        1.2470380878419847,
        17.335474403914507,
        4.578676373959917,
    ],
    "DS": [
        1.5595547813502633,
        0.32468509954461666,
        0.9985804345222741,
        0.48475195198210874,
        2.680076305564069,
        2.1917777751746574,
        2.5024755440826474,
        1.94310035110333,
        2.352477015539854,
        2.099383363353033,
        1.4726548661623906,
        0.15867885117862723,
        0.9788605211302381,
        2.615939634439106,
        2.951557999442703,
        1.544449169406222,
        2.6992474076759474,
        2.8424497326133427,
        2.0301089216114456,
        2.164548871623156,
    ],
}

# Experimental initial conditions (Day 1 data)
PHI_INIT = {
    "CS": [0.1756, 0.0049, 0.8098, 0.0049, 0.0049],
    "CH": [0.7463, 0.0498, 0.0995, 0.0995, 0.0050],
    "DH": [0.0400, 0.0100, 0.9400, 0.0050, 0.0050],
    "DS": [0.1531, 0.0816, 0.6122, 0.0510, 0.1020],
}

# Observation noise (mean IQR/1.35)
SIGMA = {"CS": 0.1111, "CH": 0.1568, "DH": 0.2321, "DS": 0.2469}

# Free parameter indices per condition (condition-specific locking from paper)
FREE_INDICES = {
    "CS": [0, 1, 2, 3, 4, 5, 8, 10, 12, 16, 17],  # 11 free (9 locked)
    "CH": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 19],  # 13 free (7 locked)
    "DS": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 18, 19],  # 15 free (5 locked)
    "DH": list(range(20)),  # 20 free (0 locked)
}

PARAM_NAMES = [
    "a11",
    "a12",
    "a22",
    "b1",
    "b2",
    "a33",
    "a34",
    "a44",
    "b3",
    "b4",
    "a13",
    "a14",
    "a23",
    "a24",
    "a55",
    "b5",
    "a15",
    "a25",
    "a35",
    "a45",
]

# Time points: days → Hamilton time indices
N_STEPS = 2500
DT = 1e-4
T_MAX = N_STEPS * DT
DAYS = np.array([1, 3, 6, 10, 15, 21])
DAY_SCALE = T_MAX * 0.95 / DAYS.max()
IDX_OBS = np.round(DAYS * DAY_SCALE / DT).astype(int)
# Skip day 1 (= initial condition, trivially zero Jacobian)
IDX_FIT = IDX_OBS[1:]  # days 3, 6, 10, 15, 21 → 5 time points × 5 species = 25 obs


def forward_model(theta, phi_init):
    """θ(20) → ŷ(25) = phi at 5 observation time points (days 3-21), flattened."""
    phi_traj = simulate_0d(
        theta,
        n_steps=N_STEPS,
        dt=DT,
        phi_init=phi_init,
        K_hill=0.05,
        n_hill=2.0,
        c_const=25.0,
        alpha_const=100.0,
    )
    # Extract at observation indices (skip day 1 = IC)
    phi_obs = phi_traj[IDX_FIT, :]  # (5, 5)
    return phi_obs.flatten()  # (25,)


def compute_fim(cond):
    """Compute FIM for one condition — both full 20D and free-params-only."""
    theta = jnp.array(THETA_MAP[cond])
    phi_init = jnp.array(PHI_INIT[cond])
    sigma = SIGMA[cond]
    free_idx = FREE_INDICES[cond]
    n_free = len(free_idx)

    print(f"\n{'='*60}")
    print(f"Condition: {cond} (σ = {sigma:.4f}, {n_free} free params)")
    print(f"{'='*60}")

    # Full Jacobian: ∂ŷ/∂θ at MAP  (25 × 20)
    print("  Computing Jacobian via jax.jacfwd...")
    J_fn = jax.jacfwd(lambda th: forward_model(th, phi_init))
    J_full = np.array(J_fn(theta))

    # ── Full 20D FIM ──
    FIM_full = J_full.T @ J_full / (sigma**2)
    eigvals_full = np.sort(np.linalg.eigvalsh(FIM_full))[::-1]
    n_pos_full = int(np.sum(eigvals_full > eigvals_full[0] * 1e-12))
    print(f"\n  [Full 20D] Rank: {n_pos_full}/20")

    # ── Free-params-only FIM ──
    J_free = J_full[:, free_idx]  # (25, n_free)
    FIM_free = J_free.T @ J_free / (sigma**2)
    eigvals_free, eigvecs_free = np.linalg.eigh(FIM_free)
    eigvals_free_sorted = np.sort(eigvals_free)[::-1]
    n_pos_free = int(np.sum(eigvals_free > eigvals_free_sorted[0] * 1e-12))
    kappa_free = eigvals_free_sorted[0] / max(eigvals_free_sorted[max(n_pos_free - 1, 0)], 1e-30)

    free_names = [PARAM_NAMES[i] for i in free_idx]
    print(f"  [Free {n_free}D] Rank: {n_pos_free}/{n_free}, κ = {kappa_free:.2e}")
    print(
        f"  λ_max = {eigvals_free_sorted[0]:.4e}, λ_min = {eigvals_free_sorted[n_pos_free-1]:.4e}"
    )

    print(f"\n  Eigenvalues ({n_free}D, descending):")
    for i, ev in enumerate(eigvals_free_sorted):
        marker = " ← WEAK" if ev < eigvals_free_sorted[0] * 1e-8 else ""
        print(f"    λ_{i:2d} = {ev:12.4e}{marker}")

    # CRB for free params
    crb_free = None
    if n_pos_free == n_free:
        FIM_free_inv = np.linalg.inv(FIM_free)
        crb_free = np.sqrt(np.diag(FIM_free_inv))
        print("\n  Cramér-Rao lower bounds:")
        for i, (name, bound) in enumerate(zip(free_names, crb_free)):
            map_val = THETA_MAP[cond][free_idx[i]]
            pct = bound / max(abs(map_val), 1e-6) * 100
            marker = " ← POOR" if pct > 100 else ""
            print(f"    {name:5s}: σ_CR ≥ {bound:.4f} ({pct:6.1f}% of |MAP|){marker}")
    else:
        print(f"\n  Free FIM rank-deficient: {n_pos_free}/{n_free}")
        weak_mask = eigvals_free < eigvals_free_sorted[0] * 1e-12
        weak_dirs = eigvecs_free[:, weak_mask]
        n_weak = weak_dirs.shape[1]
        print(f"  {n_weak} unidentifiable direction(s):")
        for d in range(min(n_weak, 5)):
            top = np.argsort(np.abs(weak_dirs[:, d]))[::-1][:3]
            desc = ", ".join(f"{free_names[p]}({weak_dirs[p,d]:+.2f})" for p in top)
            print(f"    Dir {d}: {desc}")

    return {
        "J_full": J_full,
        "FIM_full": FIM_full,
        "eigvals_full": eigvals_full,
        "rank_full": n_pos_full,
        "J_free": J_free,
        "FIM_free": FIM_free,
        "eigvals_free": eigvals_free_sorted,
        "rank_free": n_pos_free,
        "n_free": n_free,
        "cond_number": kappa_free,
        "crb_free": crb_free,
        "free_idx": free_idx,
        "free_names": free_names,
    }


def plot_results(results):
    """Eigenvalue spectrum: full 20D vs free-params-only."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cond_colors = {"CS": "#2196F3", "CH": "#4CAF50", "DS": "#FF9800", "DH": "#F44336"}

    # ── Col 0: Full 20D eigenvalue spectrum ──
    ax = axes[0]
    for cond, data in results.items():
        ev = np.maximum(data["eigvals_full"], 1e-20)
        ax.semilogy(
            range(1, 21),
            ev,
            "o-",
            markersize=4,
            color=cond_colors[cond],
            label=f"{cond} (rank={data['rank_full']})",
        )
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Full 20D FIM — Data information only")
    ax.legend()
    ax.set_xlim(0.5, 20.5)
    ax.grid(True, alpha=0.3)

    # ── Col 1: Free-params-only eigenvalue spectrum ──
    ax = axes[1]
    for cond, data in results.items():
        n_free = data["n_free"]
        ev = np.maximum(data["eigvals_free"], 1e-20)
        ax.semilogy(
            range(1, n_free + 1),
            ev,
            "o-",
            markersize=5,
            color=cond_colors[cond],
            label=f"{cond} ({n_free}D, rank={data['rank_free']}, κ={data['cond_number']:.0e})",
        )
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Free-params FIM — After biological constraints")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Col 2: CRB bar chart for conditions with full-rank free FIM ──
    ax = axes[2]
    plotted = False
    for cond in ["CS", "CH", "DS"]:
        data = results[cond]
        if data["crb_free"] is not None:
            crb = data["crb_free"]
            n_free = data["n_free"]
            y_pos = np.arange(n_free)
            ax.barh(
                y_pos - 0.25 + 0.25 * list(results.keys()).index(cond),
                crb,
                height=0.25,
                color=cond_colors[cond],
                alpha=0.7,
                label=cond,
            )
            if not plotted:
                ax.set_yticks(range(n_free))
                ax.set_yticklabels(data["free_names"], fontsize=7)
                plotted = True
    ax.set_xlabel(r"$\sigma_{\mathrm{CR}}$ (Cramér-Rao lower bound)")
    ax.set_title("CRB — Minimum achievable std dev")
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "paper_final", "fim_identifiability.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


def print_latex_table(results):
    """Output LaTeX-ready summary table."""
    print("\n" + "=" * 60)
    print("LaTeX summary table")
    print("=" * 60)
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(r" & \textbf{CS} & \textbf{CH} & \textbf{DS} & \textbf{DH} \\")
    print(r"\midrule")

    # Full 20D rank
    vals = [str(results[c]["rank_full"]) for c in ["CS", "CH", "DS", "DH"]]
    print(f"Full FIM rank (of 20) & {' & '.join(vals)} \\\\")

    # Free params
    vals = [str(results[c]["n_free"]) for c in ["CS", "CH", "DS", "DH"]]
    print(f"$n_{{\\text{{free}}}}$ & {' & '.join(vals)} \\\\")

    # Free FIM rank
    vals = [str(results[c]["rank_free"]) for c in ["CS", "CH", "DS", "DH"]]
    print(f"Free FIM rank & {' & '.join(vals)} \\\\")

    # Condition number
    vals = []
    for c in ["CS", "CH", "DS", "DH"]:
        k = results[c]["cond_number"]
        vals.append(f"${k:.1e}$")
    print(f"$\\kappa$ & {' & '.join(vals)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    results = {}
    for cond in ["CS", "CH", "DS", "DH"]:
        results[cond] = compute_fim(cond)

    plot_results(results)
    print_latex_table(results)
