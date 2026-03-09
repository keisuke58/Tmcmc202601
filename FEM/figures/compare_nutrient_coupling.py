#!/usr/bin/env python3
"""
compare_nutrient_coupling.py
============================
0D Hamilton ODE: constant b vs Monod b(t) の比較。

4条件 (CS, CH, DH, DS) × 消費率スイープで栄養枯渇の効果を可視化。

Usage:
    python compare_nutrient_coupling.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data_5species", "main"))

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hamilton_ode_jax import simulate_0d, simulate_0d_nutrient

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MAP theta for 4 conditions ──
THETA_MAP = {
    "Commensal Static": [
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
    "Commensal HOBIC": [
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
    "Dysbiotic HOBIC": [
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
    "Dysbiotic Static": [
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

SPECIES_NAMES = ["S. oralis", "A. naeslundii", "Veillonella", "F. nucleatum", "P. gingivalis"]
SPECIES_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

N_STEPS = 2500
DT = 1e-4


def compute_di(phi):
    """Shannon entropy-based Dysbiosis Index."""
    p = phi / (phi.sum(axis=-1, keepdims=True) + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p), axis=-1)
    return 1.0 - H / np.log(5)


def run_comparison():
    """Run both models for all 4 conditions with consumption rate sweep."""
    results = {}

    # Sweep consumption multipliers: 1x (original) → 50x (strong depletion)
    g_base = jnp.array([1.0, 0.8, 0.3, 0.5, 0.3])
    g_multipliers = [10.0, 50.0, 200.0]

    # Supply rates: static = 0 (closed batch), HOBIC = moderate
    supply_map = {
        "Commensal Static": 0.0,
        "Commensal HOBIC": 0.5,
        "Dysbiotic HOBIC": 0.5,
        "Dysbiotic Static": 0.0,
    }

    for cond_name, theta_list in THETA_MAP.items():
        theta = jnp.array(theta_list)
        supply = supply_map[cond_name]
        print(f"Running {cond_name} (supply={supply:.2f})...")

        # Constant b (original)
        phi_const = np.array(simulate_0d(theta, n_steps=N_STEPS, dt=DT))

        # Nutrient-coupled b(t): sweep g_consumption multiplier
        results_nutrient = {}
        for mult in g_multipliers:
            g_cons = g_base * mult
            phi_nut, S_nut = simulate_0d_nutrient(
                theta,
                n_steps=N_STEPS,
                dt=DT,
                K_S=0.5,
                g_consumption=g_cons,
                supply_rate=supply,
                S_ext=1.0,
                S_init=1.0,
            )
            results_nutrient[mult] = (np.array(phi_nut), np.array(S_nut))

        results[cond_name] = {
            "phi_const": phi_const,
            "nutrient": results_nutrient,
            "supply": supply,
        }

    return results, g_multipliers


def plot_results(results, g_multipliers):
    """4 conditions × (phi trajectory + nutrient S(t) + DI comparison)."""
    conditions = list(results.keys())
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(
        r"0D Hamilton ODE: constant $b$ vs Monod $b_{\mathrm{eff}}(t) = b \cdot S/(K_S + S)$",
        fontsize=16,
        y=0.98,
    )

    t = np.arange(N_STEPS + 1) * DT
    mult_colors = {10.0: "#1976D2", 50.0: "#E65100", 200.0: "#C62828"}

    for row, cond in enumerate(conditions):
        data = results[cond]
        phi_const = data["phi_const"]

        # ── Col 0: phi trajectories ──
        ax_phi = axes[row, 0]
        # Constant b (dashed, thin)
        for s in range(5):
            ax_phi.plot(t, phi_const[:, s], "--", color=SPECIES_COLORS[s], alpha=0.4, linewidth=1)
        # Strongest depletion case (solid)
        best_mult = g_multipliers[-1]
        phi_nut, _ = data["nutrient"][best_mult]
        for s in range(5):
            ax_phi.plot(
                t,
                phi_nut[:, s],
                "-",
                color=SPECIES_COLORS[s],
                linewidth=1.5,
                label=SPECIES_NAMES[s],
            )
        ax_phi.set_ylabel(r"$\varphi_i$")
        ax_phi.set_title(f"{cond}\n(dashed=const $b$, solid=$b(t)$ g×{best_mult:.0f})")
        if row == 0:
            ax_phi.legend(fontsize=7, loc="upper right", ncol=2)
        if row == 3:
            ax_phi.set_xlabel("Hamilton time")

        # ── Col 1: Nutrient S(t) ──
        ax_s = axes[row, 1]
        for mult in g_multipliers:
            _, S_nut = data["nutrient"][mult]
            ax_s.plot(t, S_nut, linewidth=1.5, color=mult_colors[mult], label=f"$g$ × {mult:.0f}")
        ax_s.axhline(0.5, color="gray", ls=":", alpha=0.5, label="$K_S$")
        ax_s.set_ylabel("$S(t)$")
        ax_s.set_title(f"Nutrient (supply={data['supply']:.1f})")
        ax_s.set_ylim(-0.05, 1.1)
        ax_s.legend(fontsize=8)
        if row == 3:
            ax_s.set_xlabel("Hamilton time")

        # ── Col 2: DI comparison ──
        ax_di = axes[row, 2]
        di_const = compute_di(phi_const)
        ax_di.plot(t, di_const, "k--", linewidth=1.5, label="constant $b$")
        for mult in g_multipliers:
            phi_nut, _ = data["nutrient"][mult]
            di_nut = compute_di(phi_nut)
            ax_di.plot(
                t, di_nut, linewidth=1.5, color=mult_colors[mult], label=f"$b(t)$, $g$×{mult:.0f}"
            )
        ax_di.set_ylabel("DI")
        ax_di.set_title("Dysbiosis Index")
        ax_di.set_ylim(-0.05, 1.05)
        ax_di.legend(fontsize=8)
        if row == 3:
            ax_di.set_xlabel("Hamilton time")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(os.path.dirname(__file__), "paper_final", "nutrient_coupling_0d.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def print_summary(results, g_multipliers):
    """Print key metrics."""
    print("\n" + "=" * 70)
    print("Summary: constant b vs Monod b(t)")
    print("=" * 70)
    for cond, data in results.items():
        phi_const = data["phi_const"]
        di_const = compute_di(phi_const)[-1]
        print(f"\n{cond} (supply={data['supply']:.1f}):")
        print(f"  DI (constant b)  = {di_const:.4f}")
        for mult in g_multipliers:
            phi_nut, S_nut = data["nutrient"][mult]
            di_nut = compute_di(phi_nut)[-1]
            S_final = S_nut[-1]
            b_ratio = S_final / (0.5 + S_final)  # Monod at end
            print(
                f"  g×{mult:5.0f}: DI={di_nut:.4f} (ΔDI={di_nut-di_const:+.4f}), "
                f"S_end={S_final:.4f}, b_eff/b={b_ratio:.3f}"
            )
        # Show final phi for strongest depletion
        phi_nut, _ = data["nutrient"][g_multipliers[-1]]
        print(f"  Final φ (const):  [{', '.join(f'{v:.4f}' for v in phi_const[-1])}]")
        print(
            f"  Final φ (g×{g_multipliers[-1]:.0f}): [{', '.join(f'{v:.4f}' for v in phi_nut[-1])}]"
        )


if __name__ == "__main__":
    results, g_multipliers = run_comparison()
    plot_results(results, g_multipliers)
    print_summary(results, g_multipliers)
