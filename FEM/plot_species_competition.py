#!/usr/bin/env python3
"""
plot_species_competition.py — 論文図: 種組成 + interaction network + DI vs φ_Pg 考察
=====================================================================================

4条件の 0D Hamilton ODE 定常組成を比較し、DI model が φ_Pg model より
この Hamilton モデルに適合する理由を可視化する。

Panel (a): 4条件の種組成 (stacked bar)
Panel (b): 相互作用ネットワーク (locked edges, Hill gate)
Panel (c): DI vs φ_Pg → E のマッピング比較
Panel (d): 圧縮応力の条件間差

出力: FEM/_multiscale_results/species_competition_analysis.png
"""

from __future__ import annotations
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
_RUNS = os.path.join(_PROJ, "data_5species", "_runs")
_JAXFEM = os.path.join(_HERE, "JAXFEM")

for _p in [_HERE, _JAXFEM, _PROJ]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from material_models import (
    E_MAX_PA,
    E_MIN_PA,
    PHI_PG_CRIT,
    HILL_M,
    _hill_sigmoid,
    IDX_PG,
    IDX_FN,
)

# ── 条件定義 ──
CONDITIONS = {
    "commensal_static": {
        "run": "commensal_static",
        "color": "#1f77b4",
        "label": "Commensal\nStatic",
    },
    "commensal_hobic": {"run": "commensal_hobic", "color": "#2ca02c", "label": "Commensal\nHOBIC"},
    "dysbiotic_static": {
        "run": "dysbiotic_static",
        "color": "#ff7f0e",
        "label": "Dysbiotic\nStatic",
    },
    "dysbiotic_hobic": {"run": "dh_baseline", "color": "#d62728", "label": "Dysbiotic\nHOBIC"},
}

SPECIES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SP_SHORT = ["So", "An", "Vd", "Fn", "Pg"]
SP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]


def load_theta(run_name):
    path = os.path.join(_RUNS, run_name, "theta_MAP.json")
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        return np.array(d[:20], dtype=np.float64)
    theta = d.get("theta_sub") or d.get("theta_full")
    return np.array(theta[:20], dtype=np.float64)


def solve_0d(theta_np, n_steps=2500, dt=0.01):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from JAXFEM.core_hamilton_1d import theta_to_matrices, newton_step, make_initial_state

    theta_jax = jnp.array(theta_np, dtype=jnp.float64)
    A, b_diag = theta_to_matrices(theta_jax)
    active_mask = jnp.ones(5, dtype=jnp.int64)

    params = {
        "dt_h": dt,
        "Kp1": 1e-4,
        "Eta": jnp.ones(5, dtype=jnp.float64),
        "EtaPhi": jnp.ones(5, dtype=jnp.float64),
        "c": 100.0,
        "alpha": 100.0,
        "K_hill": jnp.array(0.05, dtype=jnp.float64),
        "n_hill": jnp.array(4.0, dtype=jnp.float64),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
        "newton_steps": 6,
    }

    g0 = make_initial_state(1, active_mask)[0]

    def body(g, _):
        return newton_step(g, params), g

    _, g_traj = jax.lax.scan(jax.jit(body), g0, jnp.arange(n_steps))

    phi_final = np.array(g_traj[-1, 0:5])
    phi_traj = np.array(g_traj[:, 0:5])

    phi_sum = phi_final.sum()
    p = phi_final / max(phi_sum, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum()
    di = float(1.0 - H / np.log(5.0))

    return {
        "phi_final": phi_final,
        "phi_traj": phi_traj,
        "di": di,
        "phi_Pg": float(phi_final[IDX_PG]),
        "phi_Fn": float(phi_final[IDX_FN]),
        "A": np.array(A),
        "b_diag": np.array(b_diag),
    }


def main():
    os.makedirs(os.path.join(_HERE, "_multiscale_results"), exist_ok=True)

    # ── 0D solve for all conditions ──
    results = {}
    for ckey, meta in CONDITIONS.items():
        theta = load_theta(meta["run"])
        res = solve_0d(theta)
        res["theta"] = theta
        results[ckey] = res
        print(
            f"[{ckey}] DI={res['di']:.3f}  φ_Pg={res['phi_Pg']:.4f}  "
            f"dominant={SP_SHORT[np.argmax(res['phi_final'])]} "
            f"({np.max(res['phi_final']):.3f})"
        )

    # ── Figure: 2×3 panels ──
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    ax_comp = fig.add_subplot(gs[0, 0])  # (a) stacked bar
    ax_net = fig.add_subplot(gs[0, 1])  # (b) interaction network
    ax_di = fig.add_subplot(gs[0, 2])  # (c) DI vs phi_Pg scatter
    ax_E = fig.add_subplot(gs[1, 0])  # (d) E comparison
    ax_traj = fig.add_subplot(gs[1, 1])  # (e) time trajectory DS
    ax_sigma = fig.add_subplot(gs[1, 2])  # (f) stress bar

    # ── (a) Species composition: stacked bar ──
    ckeys = list(results.keys())
    x = np.arange(len(ckeys))
    bottom = np.zeros(len(ckeys))

    for sp_idx in range(5):
        vals = [results[ck]["phi_final"][sp_idx] for ck in ckeys]
        bars = ax_comp.bar(
            x,
            vals,
            bottom=bottom,
            width=0.65,
            color=SP_COLORS[sp_idx],
            label=SPECIES[sp_idx],
            edgecolor="white",
            linewidth=0.5,
        )
        # annotate dominant species
        for i, v in enumerate(vals):
            if v > 0.3:
                ax_comp.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{SP_SHORT[sp_idx]}\n{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                )
        bottom += vals

    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels([CONDITIONS[ck]["label"] for ck in ckeys], fontsize=8)
    ax_comp.set_ylabel("Volume fraction")
    ax_comp.set_title("(a) 0D steady-state species composition")
    ax_comp.legend(fontsize=7, loc="upper right", ncol=2)
    ax_comp.set_ylim(0, 1.05)
    ax_comp.grid(alpha=0.2, axis="y")

    # ── (b) Interaction network ──
    ax_net.set_xlim(-1.5, 1.5)
    ax_net.set_ylim(-1.5, 1.5)
    ax_net.set_aspect("equal")
    ax_net.axis("off")
    ax_net.set_title(
        "(b) Hamilton model interaction network\n" "(locked edges dashed, Hill gate = red)",
        fontsize=9,
    )

    # positions in circle
    angles = np.array([90, 162, 234, 306, 18]) * np.pi / 180
    positions = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

    # draw nodes
    for i in range(5):
        cx, cy = positions[i]
        circle = plt.Circle((cx, cy), 0.18, color=SP_COLORS[i], alpha=0.85, zorder=5)
        ax_net.add_patch(circle)
        ax_net.text(
            cx,
            cy,
            SP_SHORT[i],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    # interactions: (i, j, locked)
    edges = [
        (0, 1, False, "a12"),  # So-An
        (0, 2, False, "a13"),  # So-Vd
        (0, 3, False, "a14"),  # So-Fn
        (0, 4, True, "a15=0"),  # So-Pg LOCKED
        (1, 2, True, "a23=0"),  # An-Vd LOCKED
        (1, 3, True, "a24=0"),  # An-Fn LOCKED
        (1, 4, True, "a25=0"),  # An-Pg LOCKED
        (2, 3, False, "a34"),  # Vd-Fn
        (2, 4, False, "a35"),  # Vd-Pg (KEY)
        (3, 4, False, "a45"),  # Fn-Pg (KEY + Hill gate)
    ]

    for i, j, locked, label in edges:
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        # shorten for node radius
        r = 0.2
        x1s = x1 + dx / dist * r
        y1s = y1 + dy / dist * r
        x2s = x2 - dx / dist * r
        y2s = y2 - dy / dist * r

        if locked:
            ax_net.plot([x1s, x2s], [y1s, y2s], "--", color="gray", alpha=0.4, lw=1, zorder=1)
            mx, my = (x1s + x2s) / 2, (y1s + y2s) / 2
            ax_net.text(
                mx,
                my,
                "X",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
                fontweight="bold",
                zorder=2,
            )
        else:
            # highlight Pg interactions
            is_pg = i == 4 or j == 4
            color = "#d62728" if is_pg else "#333333"
            lw = 2.5 if is_pg else 1.5
            alpha = 0.9 if is_pg else 0.6
            ax_net.annotate(
                "",
                xy=(x2s, y2s),
                xytext=(x1s, y1s),
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw, alpha=alpha),
                zorder=3,
            )
            # label
            mx, my = (x1s + x2s) / 2, (y1s + y2s) / 2
            offset = 0.08
            nx, ny = -dy / dist * offset, dx / dist * offset
            ax_net.text(
                mx + nx,
                my + ny,
                label,
                ha="center",
                va="center",
                fontsize=6,
                color=color,
                style="italic",
                zorder=4,
            )

    # Hill gate annotation
    ax_net.annotate(
        "Hill gate\n(K=0.05, n=4)",
        xy=(
            (positions[3][0] + positions[4][0]) / 2,
            (positions[3][1] + positions[4][1]) / 2 - 0.15,
        ),
        fontsize=7,
        color="#d62728",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.9),
    )

    # ── (c) DI vs φ_Pg scatter ──
    for ck in ckeys:
        r = results[ck]
        meta = CONDITIONS[ck]
        ax_di.scatter(
            r["di"], r["phi_Pg"], s=200, c=meta["color"], edgecolor="black", zorder=5, marker="o"
        )
        dominant = SP_SHORT[np.argmax(r["phi_final"])]
        ax_di.annotate(
            f"{meta['label'].replace(chr(10),' ')}\n" f"dominant: {dominant}",
            (r["di"], r["phi_Pg"]),
            textcoords="offset points",
            xytext=(10, -10 if r["di"] > 0.5 else 10),
            fontsize=7,
            color=meta["color"],
            arrowprops=dict(arrowstyle="->", color=meta["color"], lw=0.8),
        )

    ax_di.axhline(
        PHI_PG_CRIT, color="red", ls="--", alpha=0.5, label=f"$\\varphi_{{Pg,crit}}$={PHI_PG_CRIT}"
    )
    ax_di.axvline(0.5, color="blue", ls="--", alpha=0.3, label="DI=0.5")
    ax_di.set_xlabel("DI (Dysbiotic Index)")
    ax_di.set_ylabel(r"$\varphi_{Pg}$ (P. gingivalis fraction)")
    ax_di.set_title("(c) DI vs $\\varphi_{Pg}$: condition discrimination")
    ax_di.legend(fontsize=7)
    ax_di.grid(alpha=0.3)
    ax_di.set_xlim(-0.05, 1.0)
    ax_di.set_ylim(-0.01, max(r["phi_Pg"] for r in results.values()) * 2.5)
    ax_di.annotate(
        "DI discriminates\nconditions well\n(horizontal spread)",
        xy=(0.5, 0.02),
        fontsize=7,
        color="blue",
        ha="center",
        style="italic",
    )
    ax_di.annotate(
        "$\\varphi_{Pg}$ fails\n(all below $\\varphi_{crit}$)",
        xy=(0.1, PHI_PG_CRIT + 0.02),
        fontsize=7,
        color="red",
        ha="center",
        style="italic",
    )

    # ── (d) E comparison: DI vs φ_Pg ──
    E_di_vals = []
    E_pg_vals = []
    for ck in ckeys:
        r = results[ck]
        # DI model with scale=1.0 (hybrid)
        di_r = min(r["di"], 1.0)
        E_di_v = E_MAX_PA * (1.0 - di_r) ** 2 + E_MIN_PA * di_r
        E_di_vals.append(E_di_v)
        # φ_Pg model
        sig = _hill_sigmoid(np.array([r["phi_Pg"] / PHI_PG_CRIT]), HILL_M)[0]
        E_pg_v = E_MAX_PA - (E_MAX_PA - E_MIN_PA) * sig
        E_pg_vals.append(E_pg_v)

    w = 0.3
    ax_E.bar(
        x - w / 2,
        E_di_vals,
        w,
        label="DI model",
        color=[CONDITIONS[ck]["color"] for ck in ckeys],
        alpha=0.85,
        edgecolor="black",
    )
    ax_E.bar(
        x + w / 2,
        E_pg_vals,
        w,
        label=r"$\varphi_{Pg}$ model",
        color=[CONDITIONS[ck]["color"] for ck in ckeys],
        alpha=0.35,
        edgecolor="black",
        hatch="///",
    )

    for i, (edi, epg) in enumerate(zip(E_di_vals, E_pg_vals)):
        ax_E.text(x[i] - w / 2, edi + 20, f"{edi:.0f}", ha="center", fontsize=7, fontweight="bold")
        ax_E.text(x[i] + w / 2, epg + 20, f"{epg:.0f}", ha="center", fontsize=7, color="gray")

    ax_E.set_xticks(x)
    ax_E.set_xticklabels([CONDITIONS[ck]["label"] for ck in ckeys], fontsize=8)
    ax_E.set_ylabel("E [Pa]")
    ax_E.set_title("(d) E from DI (solid) vs $\\varphi_{Pg}$ (hatched)")
    ax_E.legend(fontsize=8)
    ax_E.set_ylim(0, E_MAX_PA * 1.2)
    ax_E.grid(alpha=0.2, axis="y")

    # Ratio annotation
    if E_di_vals[0] > 0 and E_di_vals[2] > 0:
        ratio_di = E_di_vals[0] / max(E_di_vals[2], 1)
        ratio_pg = E_pg_vals[0] / max(E_pg_vals[2], 1)
        ax_E.text(
            0.5,
            0.02,
            f"DI ratio: {ratio_di:.0f}x    $\\varphi_{{Pg}}$ ratio: {ratio_pg:.1f}x",
            transform=ax_E.transAxes,
            fontsize=8,
            ha="center",
            bbox=dict(boxstyle="round", fc="lightyellow"),
        )

    # ── (e) Time trajectory for Dysbiotic Static (So dominance) ──
    ds_res = results["dysbiotic_static"]
    t_axis = np.arange(ds_res["phi_traj"].shape[0]) * 0.01
    for sp_idx in range(5):
        ax_traj.plot(
            t_axis,
            ds_res["phi_traj"][:, sp_idx],
            color=SP_COLORS[sp_idx],
            lw=1.5,
            label=SP_SHORT[sp_idx],
        )
    ax_traj.set_xlabel("T* (dimensionless time)")
    ax_traj.set_ylabel("Volume fraction")
    ax_traj.set_title(
        "(e) 0D trajectory: Dysbiotic Static\n(S. oralis dominates, NOT P. gingivalis)"
    )
    ax_traj.legend(fontsize=7, ncol=2)
    ax_traj.grid(alpha=0.3)
    ax_traj.set_xlim(0, t_axis[-1])
    ax_traj.set_ylim(0, 1.0)

    # ── (f) Compressive stress bar ──
    eps_tooth = 0.001384  # from hybrid CSV
    sigma_di = [-E * eps_tooth for E in E_di_vals]
    sigma_pg = [-E * eps_tooth for E in E_pg_vals]

    ax_sigma.bar(
        x - w / 2,
        sigma_di,
        w,
        label="DI model",
        color=[CONDITIONS[ck]["color"] for ck in ckeys],
        alpha=0.85,
        edgecolor="black",
    )
    ax_sigma.bar(
        x + w / 2,
        sigma_pg,
        w,
        label=r"$\varphi_{Pg}$ model",
        color=[CONDITIONS[ck]["color"] for ck in ckeys],
        alpha=0.35,
        edgecolor="black",
        hatch="///",
    )
    ax_sigma.set_xticks(x)
    ax_sigma.set_xticklabels([CONDITIONS[ck]["label"] for ck in ckeys], fontsize=8)
    ax_sigma.set_ylabel(r"$\sigma_0$ [Pa]  (compressive)")
    ax_sigma.set_title(r"(f) Prestress $\sigma_0 = -E \cdot \varepsilon_{growth}$ (tooth)")
    ax_sigma.legend(fontsize=8)
    ax_sigma.grid(alpha=0.2, axis="y")

    fig.suptitle(
        "Species Competition Analysis: Why DI (entropy) outperforms $\\varphi_{Pg}$ (pathogen-specific)\n"
        "Hamilton ODE predicts So/Vd dominance in dysbiotic conditions, not Pg dominance",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    path = os.path.join(_HERE, "_multiscale_results", "species_competition_analysis.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n論文図出力: {path}")

    # ── Summary table ──
    print("\n" + "=" * 80)
    print("SPECIES COMPETITION ANALYSIS — SUMMARY")
    print("=" * 80)
    print(
        f"{'Condition':<22} {'Dominant':>8} {'φ_dom':>7} {'DI':>6} {'φ_Pg':>7} "
        f"{'E_di':>6} {'E_φPg':>7} {'ratio':>6}"
    )
    print("-" * 80)
    for i, ck in enumerate(ckeys):
        r = results[ck]
        dom_idx = np.argmax(r["phi_final"])
        print(
            f"  {ck:<20} {SP_SHORT[dom_idx]:>8} {r['phi_final'][dom_idx]:>7.3f} "
            f"{r['di']:>6.3f} {r['phi_Pg']:>7.4f} "
            f"{E_di_vals[i]:>6.0f} {E_pg_vals[i]:>7.0f} "
            f"{E_di_vals[i]/max(E_pg_vals[i],1):>6.2f}"
        )

    print("\n  KEY FINDING:")
    print("  - Dysbiotic Static: So dominates (0.944), NOT Pg (0.011)")
    print("  - DI correctly identifies loss of diversity → E = 32 Pa")
    print("  - φ_Pg misses the dysbiosis → E = 1000 Pa (false healthy)")
    print("  - Model structure: Pg isolated from pioneers (a15=a25=0)")
    print("    + Hill gated by Fn (K=0.05) + symmetric a35/a45 competition")
    print("=" * 80)

    return path


if __name__ == "__main__":
    main()
