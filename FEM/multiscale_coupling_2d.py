#!/usr/bin/env python3
"""
multiscale_coupling_2d.py — 2D Hamilton+栄養PDE 連成パイプライン
================================================================

1D 拡散均質化の限界を克服する 2D バイオフィルム・マルチスケールモデル。

Key advantage over 1D
---------------------
  - 1D: diffusion homogenizes species → DI ≈ 0 everywhere
  - 2D: egg-shaped biofilm geometry (Klempt 2024) creates 2D nutrient gradients
        → spatially varying species composition → DI(x,y) varies

Physics
-------
  [Hamilton PDE — φᵢ(x,y), ψᵢ, γ]  5-species biofilm kinetics
      with nutrient coupling: c_local modulates interaction strength
  [Nutrient PDE — c(x,y,t)]
      ∂c/∂t = D_c Δc − Σ g_i φ_i c/(k+c)
      BC: c = 1 on domain boundary (Dirichlet)
  [Biofilm mask — egg-shape φ₀(x,y)]
      Species confined to biofilm interior
      Nutrients diffuse freely outside biofilm

Flow
----
  TMCMC MAP θ (4 conditions × 20 params)
      ↓
  2D Hamilton + nutrient PDE (egg-shaped biofilm, Nx×Ny grid)
      ↓
  φᵢ(x,y,T), c(x,y,T), DI(x,y), E(x,y), α_Monod(x,y)
      ↓
  Comparison figures: 2D fields, condition comparison, 1D vs 2D

Usage
-----
  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \\
      Tmcmc202601/FEM/multiscale_coupling_2d.py

Output
------
  FEM/_multiscale_2d_results/
      ├── 2d_fields_{condition}.npz          (raw simulation data)
      ├── 2d_nutrient_comparison.png         (4-condition c field)
      ├── 2d_di_comparison.png               (4-condition DI field — KEY)
      ├── 2d_species_composition.png         (φ_Pg, φ_total spatial)
      ├── 1d_vs_2d_comparison.png            (DI distribution comparison)
      └── 2d_summary.json                    (numerical summary)

Environment: klempt_fem conda env (Python 3.11, JAX 0.9.0.1)
"""

from __future__ import annotations
import json
import os
import sys
import time

import subprocess
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ── パス設定 ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
_RUNS_DIR = os.path.join(_PROJ, "data_5species", "_runs")

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from JAXFEM.core_hamilton_2d_nutrient import (
    Config2D,
    egg_shape_mask,
    run_simulation_coupled,
    compute_di_field,
    compute_alpha_monod,
    _make_reaction_step_c,
    _make_nutrient_step_stable,
)
from material_models import E_MAX_PA, E_MIN_PA, DI_SCALE

OUT_DIR = os.path.join(_HERE, "_multiscale_2d_results")

# ── 条件マッピング ───────────────────────────────────────────────────────────
CONDITIONS = {
    "commensal_static": {
        "run": "commensal_static",
        "color": "#1f77b4",
        "label": "Commensal Static",
        "linestyle": "-",
    },
    "commensal_hobic": {
        "run": "commensal_hobic",
        "color": "#2ca02c",
        "label": "Commensal HOBIC",
        "linestyle": "--",
    },
    "dysbiotic_static": {
        "run": "dysbiotic_static",
        "color": "#ff7f0e",
        "label": "Dysbiotic Static",
        "linestyle": "-.",
    },
    "dysbiotic_hobic": {
        "run": "dh_baseline",
        "color": "#d62728",
        "label": "Dysbiotic HOBIC",
        "linestyle": ":",
    },
}

SPECIES_NAMES = ["S. oralis", "A. naeslundii", "V. dispar", "F. nucleatum", "P. gingivalis"]
SPECIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

# ── 物理パラメータ ───────────────────────────────────────────────────────────
G_EFF = 50.0  # effective nutrient consumption (→ Thiele ~ 4)
D_C = 1.0  # nutrient diffusion coefficient
K_MONOD = 1.0  # Monod half-saturation
K_ALPHA = 0.05  # growth-eigenstrain coupling
L_BIOFILM = 0.2  # biofilm thickness [mm] for scale conversion

# ── シミュレーション設定 ─────────────────────────────────────────────────────
# T* = N_MACRO × N_REACT × DT_H  (species dynamics need T*≥10 to develop)
NX = 12
NY = 12
N_MACRO = 1000  # → T* = 1000 × 10 × 1e-3 = 10.0
N_REACT = 10  # reaction sub-steps per macro
DT_H = 1e-3  # Hamilton time step
N_SUB_C = 20  # nutrient PDE sub-steps (CFL stability)
SAVE_EVERY = 100  # save snapshot every N macro steps

# Hamilton coupling scale: nutrient PDE gives c ∈ [0,1], but Hamilton model
# was calibrated with c = 100.0.  Scale c_pde → c_hamilton = c_pde * C_SCALE.
C_HAMILTON_SCALE = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────


def load_theta(condition_key: str) -> np.ndarray:
    """TMCMC ランディレクトリから theta_MAP を読み込む。"""
    run_name = CONDITIONS[condition_key]["run"]
    path = os.path.join(_RUNS_DIR, run_name, "theta_MAP.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"theta_MAP.json not found: {path}")
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        return np.array(d[:20], dtype=np.float64)
    theta = d.get("theta_sub") or d.get("theta_full")
    return np.array(theta[:20], dtype=np.float64)


def compute_E_from_DI(di_field):
    """DI field → E field [Pa] using the standard DI model."""
    r = np.clip(di_field / DI_SCALE, 0.0, 1.0)
    return E_MAX_PA * (1.0 - r) ** 2.0 + E_MIN_PA * r


# ─────────────────────────────────────────────────────────────────────────────
# 0D Hamilton JAX ソルバー (条件別 DI・組成 reference)
# ─────────────────────────────────────────────────────────────────────────────


def solve_0d_reference(theta_np: np.ndarray) -> dict:
    """TMCMC MAP θ から 0D Hamilton ODE 参照解を計算する。"""
    from JAXFEM.core_hamilton_1d import theta_to_matrices, newton_step, make_initial_state

    theta_jax = jnp.array(theta_np, dtype=jnp.float64)
    A, b_diag = theta_to_matrices(theta_jax)
    active_mask = jnp.ones(5, dtype=jnp.int64)

    params = {
        "dt_h": 0.01,
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

    # Use JIT for Newton step, but Python loop (not lax.scan)
    # to avoid LLVM compilation issues
    _newton_jit = jax.jit(newton_step)
    g = g0
    for _ in range(2500):
        g = _newton_jit(g, params)
    phi_final = np.array(g[0:5])

    phi_sum = phi_final.sum()
    p = phi_final / max(phi_sum, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum()
    di_0d = float(1.0 - H / np.log(5.0))

    return {
        "phi_final": phi_final,
        "di_0d": di_0d,
        "phi_total": float(phi_sum),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2D シミュレーション実行
# ─────────────────────────────────────────────────────────────────────────────


def run_2d(theta: np.ndarray, condition_label: str, reaction_fn=None, nutrient_fn=None) -> dict:
    """1条件の 2D Hamilton + nutrient PDE シミュレーションを実行する。"""
    cfg = Config2D(
        Nx=NX,
        Ny=NY,
        Lx=1.0,
        Ly=1.0,
        dt_h=DT_H,
        n_react_sub=N_REACT,
        n_macro=N_MACRO,
        save_every=SAVE_EVERY,
        D_c=D_C,
        k_monod=K_MONOD,
        # Species-weighted consumption (scaled by G_EFF)
        g_consumption=np.array([G_EFF * 1.0, G_EFF * 1.0, G_EFF * 0.8, G_EFF * 0.5, G_EFF * 0.3]),
        c_boundary=1.0,
        K_hill=0.05,
        n_hill=4.0,
        newton_iters=6,
    )

    # Egg-shaped biofilm mask (Klempt 2024)
    mask = egg_shape_mask(
        NX, NY, cfg.Lx, cfg.Ly, ax=0.35, ay=0.25, cx=0.5, cy=0.5, skew=0.3, eps=0.1
    )

    t0 = time.time()
    print(f"\n  [{condition_label}] 2D coupled simulation starting...")
    print(f"  Grid: {NX}x{NY}, n_macro={N_MACRO}, T*={N_MACRO*N_REACT*DT_H:.2f}")
    print(f"  Biofilm mask: egg-shape, area fraction = {mask.mean():.3f}")

    result = run_simulation_coupled(
        theta,
        cfg,
        biofilm_mask=mask,
        n_sub_c=N_SUB_C,
        reaction_fn=reaction_fn,
        nutrient_fn=nutrient_fn,
        c_hamilton_scale=C_HAMILTON_SCALE,
    )

    elapsed = time.time() - t0
    print(f"  [{condition_label}] Completed in {elapsed:.1f} s")

    # Compute derived fields
    phi_snaps = result["phi_snaps"]  # (n_snap, 5, Nx, Ny)
    c_snaps = result["c_snaps"]  # (n_snap, Nx, Ny)
    t_snaps = result["t_snaps"]  # (n_snap,)

    di_snaps = compute_di_field(phi_snaps)  # (n_snap, Nx, Ny)
    alpha_monod = compute_alpha_monod(
        phi_snaps,
        c_snaps,
        t_snaps,
        k_alpha=K_ALPHA,
        k_monod=K_MONOD,
    )  # (Nx, Ny)

    # Final time fields
    phi_final = phi_snaps[-1]  # (5, Nx, Ny)
    c_final = c_snaps[-1]  # (Nx, Ny)
    di_final = di_snaps[-1]  # (Nx, Ny)
    E_final = compute_E_from_DI(di_final)  # (Nx, Ny)
    phi_total = phi_final.sum(axis=0)  # (Nx, Ny)
    eps_growth = alpha_monod / 3.0  # (Nx, Ny)

    # Statistics inside biofilm
    bf = mask > 0.5
    di_bf = di_final[bf]
    E_bf = E_final[bf]
    c_bf = c_final[bf]

    return {
        "phi_snaps": phi_snaps,
        "c_snaps": c_snaps,
        "t_snaps": t_snaps,
        "di_snaps": di_snaps,
        "alpha_monod": alpha_monod,
        "phi_final": phi_final,
        "c_final": c_final,
        "di_final": di_final,
        "E_final": E_final,
        "phi_total": phi_total,
        "eps_growth": eps_growth,
        "biofilm_mask": mask,
        "cfg": cfg,
        "elapsed_s": elapsed,
        # Summary stats (inside biofilm only)
        "di_mean": float(di_bf.mean()),
        "di_min": float(di_bf.min()),
        "di_max": float(di_bf.max()),
        "di_std": float(di_bf.std()),
        "E_mean": float(E_bf.mean()),
        "c_min_bf": float(c_bf.min()),
        "c_mean_bf": float(c_bf.mean()),
        "phi_Pg_mean": float(phi_final[4][bf].mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 図1: 2D 場の比較 (nutrient + DI + E + alpha)
# ─────────────────────────────────────────────────────────────────────────────


def plot_2d_fields(results: dict, out_dir: str):
    """4条件 × 4場 の 2D コンター図。"""
    ckeys = list(results.keys())
    n_cond = len(ckeys)

    x = np.linspace(0, 1, NX)
    y = np.linspace(0, 1, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(n_cond, 4, figsize=(18, 4 * n_cond + 1))
    if n_cond == 1:
        axes = axes[np.newaxis, :]

    titles = ["(i) Nutrient c(x,y)", "(ii) DI(x,y)", "(iii) E [Pa]", "(iv) α_Monod"]

    for row, ckey in enumerate(ckeys):
        res = results[ckey]
        mask = res["biofilm_mask"]

        fields = [res["c_final"], res["di_final"], res["E_final"], res["alpha_monod"]]
        cmaps = ["viridis", "RdYlBu_r", "RdYlBu", "hot"]
        vlims = [(0, 1), (0, 1), (E_MIN_PA, E_MAX_PA), None]

        for col, (fld, cmap, vlim) in enumerate(zip(fields, cmaps, vlims)):
            ax = axes[row, col]
            kw = {}
            if vlim is not None:
                kw["vmin"], kw["vmax"] = vlim
            im = ax.contourf(X, Y, fld, levels=20, cmap=cmap, **kw)
            ax.contour(X, Y, mask, levels=[0.5], colors="white", linewidths=1.5)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_aspect("equal")

            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(CONDITIONS[ckey]["label"], fontsize=10, fontweight="bold")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])

    fig.suptitle(
        "2D Coupled Hamilton+Nutrient: Spatial Fields\n"
        f"Grid {NX}×{NY}, T*={N_MACRO*N_REACT*DT_H:.1f}, "
        f"Biofilm=egg-shape (Klempt 2024)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(out_dir, "2d_fields_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 図2: DI 空間分布 — KEY figure (1D では不可能)
# ─────────────────────────────────────────────────────────────────────────────


def plot_di_comparison(results: dict, out_dir: str):
    """DI(x,y) の 2D 分布 + ヒストグラム (1D vs 2D の違いを強調)。"""
    ckeys = list(results.keys())
    n_cond = len(ckeys)

    x = np.linspace(0, 1, NX)
    y = np.linspace(0, 1, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(2, n_cond, figsize=(5 * n_cond, 9))
    if n_cond == 1:
        axes = axes[:, np.newaxis]

    for col, ckey in enumerate(ckeys):
        res = results[ckey]
        mask = res["biofilm_mask"]
        di = res["di_final"]

        # Top row: DI(x,y) contour
        ax = axes[0, col]
        im = ax.contourf(X, Y, di, levels=20, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.contour(X, Y, mask, levels=[0.5], colors="white", linewidths=1.5)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{CONDITIONS[ckey]['label']}\nDI mean={res['di_mean']:.3f}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

        # Bottom row: DI histogram (inside biofilm)
        ax = axes[1, col]
        di_bf = di[mask > 0.5]
        ax.hist(
            di_bf,
            bins=30,
            color=CONDITIONS[ckey]["color"],
            alpha=0.7,
            edgecolor="black",
            density=True,
        )
        ax.axvline(di_bf.mean(), color="red", ls="--", lw=2, label=f"mean={di_bf.mean():.3f}")
        ax.axvline(0.0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("DI")
        ax.set_ylabel("Density")
        ax.set_title(
            f"DI distribution\nstd={di_bf.std():.4f}, range=[{di_bf.min():.3f}, {di_bf.max():.3f}]",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle(
        "Dysbiotic Index DI(x,y) — Spatial heterogeneity in 2D\n"
        "KEY: In 1D, DI ≈ 0 everywhere (homogenized). In 2D, DI varies spatially!",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(out_dir, "2d_di_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 図3: 種組成 φ_i(x,y) の空間パターン
# ─────────────────────────────────────────────────────────────────────────────


def plot_species_spatial(results: dict, out_dir: str):
    """φ_i(x,y) の空間分布 (5種 × 4条件)。"""
    ckeys = list(results.keys())
    n_cond = len(ckeys)

    x = np.linspace(0, 1, NX)
    y = np.linspace(0, 1, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(n_cond, 5, figsize=(22, 4 * n_cond + 1))
    if n_cond == 1:
        axes = axes[np.newaxis, :]

    for row, ckey in enumerate(ckeys):
        res = results[ckey]
        mask = res["biofilm_mask"]
        phi = res["phi_final"]  # (5, Nx, Ny)

        for sp in range(5):
            ax = axes[row, sp]
            phi_sp = phi[sp]
            vmax = max(phi_sp[mask > 0.5].max(), 0.01)
            im = ax.contourf(X, Y, phi_sp, levels=20, cmap="YlOrRd", vmin=0, vmax=vmax)
            ax.contour(X, Y, mask, levels=[0.5], colors="white", linewidths=1)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_aspect("equal")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])

            if row == 0:
                ax.set_title(
                    SPECIES_NAMES[sp], fontsize=10, color=SPECIES_COLORS[sp], fontweight="bold"
                )
            if sp == 0:
                ax.set_ylabel(CONDITIONS[ckey]["label"], fontsize=10, fontweight="bold")

    fig.suptitle(
        "Species composition φᵢ(x,y) — Spatial patterns in 2D biofilm\n"
        "Interior (nutrient-depleted) vs. surface (nutrient-rich) → species segregation",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(out_dir, "2d_species_composition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 図4: 1D vs 2D 比較 (DI分布, E分布)
# ─────────────────────────────────────────────────────────────────────────────


def plot_1d_vs_2d(results_2d: dict, out_dir: str):
    """1D vs 2D 比較: DI と E の分布。"""
    ckeys = list(results_2d.keys())
    n_cond = len(ckeys)

    # Load 1D results
    _1d_dir = os.path.join(_HERE, "_multiscale_results")
    results_1d = {}
    for ckey in ckeys:
        csv_path = os.path.join(_1d_dir, f"macro_eigenstrain_{ckey}.csv")
        if os.path.isfile(csv_path):
            with open(csv_path) as f:
                lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
            cols = lines[0].split(",")
            data = np.array(
                [[float(v) for v in l.split(",")] for l in lines[1:] if l.strip()],
                dtype=np.float64,
            )
            col_map = {col: i for i, col in enumerate(cols)}
            results_1d[ckey] = {
                "DI": data[:, col_map["DI"]],
                "E_Pa": data[:, col_map["E_Pa"]],
                "c": data[:, col_map["c"]],
            }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) DI: box plot 1D vs 2D
    ax = axes[0, 0]
    positions = np.arange(n_cond)
    width = 0.3
    bp_data_1d = []
    bp_data_2d = []
    bp_labels = []
    for i, ckey in enumerate(ckeys):
        mask = results_2d[ckey]["biofilm_mask"]
        di_2d_bf = results_2d[ckey]["di_final"][mask > 0.5]
        bp_data_2d.append(di_2d_bf)

        if ckey in results_1d:
            bp_data_1d.append(results_1d[ckey]["DI"])
        else:
            bp_data_1d.append(np.array([0.0]))
        bp_labels.append(CONDITIONS[ckey]["label"].split()[0])

    bp1 = ax.boxplot(
        bp_data_1d,
        positions=positions - width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    bp2 = ax.boxplot(
        bp_data_2d,
        positions=positions + width / 2,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp1["boxes"]:
        patch.set_facecolor("#aec7e8")
    for patch in bp2["boxes"]:
        patch.set_facecolor("#ffbb78")
    ax.set_xticks(positions)
    ax.set_xticklabels(bp_labels)
    ax.set_ylabel("DI")
    ax.set_title("(a) DI distribution: 1D (blue) vs 2D (orange)")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["1D", "2D"], fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # (b) DI histogram overlay (all conditions)
    ax = axes[0, 1]
    for i, ckey in enumerate(ckeys):
        mask = results_2d[ckey]["biofilm_mask"]
        di_bf = results_2d[ckey]["di_final"][mask > 0.5]
        ax.hist(
            di_bf,
            bins=25,
            alpha=0.4,
            color=CONDITIONS[ckey]["color"],
            label=f"2D {CONDITIONS[ckey]['label']}",
            density=True,
        )
        if ckey in results_1d:
            di_1d = results_1d[ckey]["DI"]
            ax.axvline(di_1d.mean(), color=CONDITIONS[ckey]["color"], ls="--", lw=2, alpha=0.8)
    ax.set_xlabel("DI")
    ax.set_ylabel("Density")
    ax.set_title("(b) DI histograms: 2D (filled) vs 1D mean (dashed)")
    ax.legend(fontsize=7)

    # (c) E: mean comparison
    ax = axes[1, 0]
    E_1d = [results_1d[ck]["E_Pa"].mean() if ck in results_1d else 0 for ck in ckeys]
    E_2d = [results_2d[ck]["E_mean"] for ck in ckeys]
    x_pos = np.arange(n_cond)
    ax.bar(x_pos - 0.15, E_1d, 0.3, label="1D", color="#aec7e8", edgecolor="black")
    ax.bar(x_pos + 0.15, E_2d, 0.3, label="2D", color="#ffbb78", edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bp_labels, rotation=15, ha="right")
    ax.set_ylabel("E [Pa]")
    ax.set_title("(c) Mean E [Pa]: 1D vs 2D")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (d) Summary text
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "2D vs 1D Comparison Summary",
        "=" * 35,
        "",
        f"Grid: {NX}×{NY} (2D) vs 30 nodes (1D)",
        f"T*: {N_MACRO*N_REACT*DT_H:.1f} (2D) vs 20.0 (1D)",
        "Biofilm: egg-shape (2D) vs bar [0,1] (1D)",
        "",
        "KEY FINDING:",
        "  1D: DI ≈ 0 everywhere (diffusion homogenizes)",
        "  2D: DI varies spatially (preserved by geometry)",
        "",
    ]
    for ckey in ckeys:
        res2d = results_2d[ckey]
        label = CONDITIONS[ckey]["label"]
        di_1d_str = "N/A"
        if ckey in results_1d:
            di_1d_mean = results_1d[ckey]["DI"].mean()
            di_1d_std = results_1d[ckey]["DI"].std()
            di_1d_str = f"{di_1d_mean:.4f}±{di_1d_std:.4f}"
        lines.append(f"  {label}:")
        lines.append(f"    1D DI: {di_1d_str}")
        lines.append(f"    2D DI: {res2d['di_mean']:.4f}±{res2d['di_std']:.4f}")
        lines.append(f"           range [{res2d['di_min']:.3f}, {res2d['di_max']:.3f}]")

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        family="monospace",
        va="top",
    )

    fig.suptitle(
        "1D vs 2D Multiscale Comparison\n"
        "1D diffusion homogenizes species → DI≈0. "
        "2D egg-shape preserves spatial heterogeneity.",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(out_dir, "1d_vs_2d_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# サマリーテーブル
# ─────────────────────────────────────────────────────────────────────────────


def print_summary(results: dict, ref_0d: dict):
    """サマリーテーブルを表示する。"""
    print()
    print("=" * 85)
    print("  2D Coupled Hamilton+Nutrient — Summary")
    print(f"  Grid: {NX}×{NY},  T*={N_MACRO*N_REACT*DT_H:.1f},  " f"D_c={D_C},  g_eff={G_EFF}")
    print("=" * 85)
    hdr = (
        f"  {'Condition':<25} {'DI_0D':>7} {'DI_2D':>7} {'DI_std':>7} "
        f"{'c_min':>7} {'E_mean':>8} {'φ_Pg':>7}"
    )
    print(hdr)
    print("  " + "-" * 80)
    for ckey, res in results.items():
        label = CONDITIONS[ckey]["label"]
        di_0d = ref_0d[ckey]["di_0d"] if ckey in ref_0d else 0.0
        print(
            f"  {label:<25} "
            f"{di_0d:7.4f} "
            f"{res['di_mean']:7.4f} "
            f"{res['di_std']:7.4f} "
            f"{res['c_min_bf']:7.4f} "
            f"{res['E_mean']:8.1f} "
            f"{res['phi_Pg_mean']:7.4f}"
        )
    print()
    print("  KEY: 1D model gives DI ≈ 0 (homogenized)")
    print("       2D model gives spatially varying DI (preserved by geometry)")
    print("       Nutrient depletion in biofilm interior → species segregation")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────


def run_single_condition(ckey: str):
    """1条件をサブプロセス内で実行し、NPZ + ref_0d JSON を保存する。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    info = CONDITIONS[ckey]
    print(f"\n── {info['label']} ({info['run']}) ──")

    # 1. Load TMCMC MAP theta
    theta = load_theta(ckey)
    print(f"  θ[18]=a35={theta[18]:.4f}  θ[19]=a45={theta[19]:.4f}")

    # 2. 0D reference
    print("  [0D] Computing reference composition...", flush=True)
    r0d = solve_0d_reference(theta)
    print(f"  [0D] DI_0D={r0d['di_0d']:.4f}  φ_Pg={r0d['phi_final'][4]:.4f}")

    # Save 0D reference
    ref_path = os.path.join(OUT_DIR, f"ref_0d_{ckey}.json")
    with open(ref_path, "w") as f:
        json.dump(
            {
                "phi_final": r0d["phi_final"].tolist(),
                "di_0d": r0d["di_0d"],
                "phi_total": r0d["phi_total"],
            },
            f,
        )

    # 3. 2D coupled simulation (JIT functions compiled fresh in this process)
    _reaction_fn = _make_reaction_step_c(N_REACT, 6)
    _nutrient_fn = _make_nutrient_step_stable(N_SUB_C)
    res = run_2d(theta, info["label"], reaction_fn=_reaction_fn, nutrient_fn=_nutrient_fn)

    # 4. Save raw data
    npz_path = os.path.join(OUT_DIR, f"2d_fields_{ckey}.npz")
    np.savez_compressed(
        npz_path,
        phi_final=res["phi_final"],
        c_final=res["c_final"],
        di_final=res["di_final"],
        E_final=res["E_final"],
        alpha_monod=res["alpha_monod"],
        biofilm_mask=res["biofilm_mask"],
    )
    print(f"  Data saved: {npz_path}")

    # 5. Save summary stats as JSON (for parent process)
    stats_path = os.path.join(OUT_DIR, f"stats_{ckey}.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "di_mean": res["di_mean"],
                "di_std": res["di_std"],
                "di_min": res["di_min"],
                "di_max": res["di_max"],
                "E_mean": res["E_mean"],
                "c_min_bf": res["c_min_bf"],
                "c_mean_bf": res["c_mean_bf"],
                "phi_Pg_mean": res["phi_Pg_mean"],
                "elapsed_s": res["elapsed_s"],
            },
            f,
            indent=2,
        )
    print(f"  Stats saved: {stats_path}")


def load_condition_results(ckey: str) -> dict | None:
    """サブプロセスが保存した NPZ + JSON を読み込む。"""
    npz_path = os.path.join(OUT_DIR, f"2d_fields_{ckey}.npz")
    stats_path = os.path.join(OUT_DIR, f"stats_{ckey}.json")
    ref_path = os.path.join(OUT_DIR, f"ref_0d_{ckey}.json")

    if not os.path.isfile(npz_path) or not os.path.isfile(stats_path):
        return None

    data = np.load(npz_path)
    with open(stats_path) as f:
        stats = json.load(f)
    ref_0d = None
    if os.path.isfile(ref_path):
        with open(ref_path) as f:
            ref_0d = json.load(f)

    return {
        "phi_final": data["phi_final"],
        "c_final": data["c_final"],
        "di_final": data["di_final"],
        "E_final": data["E_final"],
        "alpha_monod": data["alpha_monod"],
        "biofilm_mask": data["biofilm_mask"],
        **stats,
        "ref_0d": ref_0d,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── サブプロセスモード: --single <ckey> ──
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        ckey = sys.argv[2]
        if ckey not in CONDITIONS:
            print(f"ERROR: Unknown condition '{ckey}'. " f"Available: {list(CONDITIONS.keys())}")
            sys.exit(1)
        run_single_condition(ckey)
        return

    # ── メインモード: 各条件をサブプロセスで順次実行 ──
    print()
    print("=" * 85)
    print("  multiscale_coupling_2d.py")
    print("  2D Hamilton + Nutrient PDE — Overcoming 1D Diffusion Homogenization")
    print(f"  Output: {OUT_DIR}")
    print("  Mode: subprocess isolation (fresh memory per condition)")
    print("=" * 85)

    python_exe = sys.executable
    this_script = os.path.abspath(__file__)
    completed = []

    for ckey in CONDITIONS:
        info = CONDITIONS[ckey]
        print(f"\n{'='*65}")
        print(f"  Launching subprocess: {info['label']} ({ckey})")
        print(f"{'='*65}")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["TF_NUM_INTRAOP_THREADS"] = "1"
        env["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false " "--xla_force_host_platform_device_count=1"
        )
        ret = subprocess.run(
            [python_exe, this_script, "--single", ckey],
            timeout=1800,  # 30 min max per condition
            env=env,
        )
        if ret.returncode != 0:
            print(f"  WARNING: {ckey} failed (exit code {ret.returncode})")
        else:
            completed.append(ckey)
            print(f"  OK: {ckey}")

    # ── 結果を読み込んで図を生成 ──
    results = {}
    ref_0d = {}
    for ckey in completed:
        res = load_condition_results(ckey)
        if res is None:
            print(f"  WARNING: Could not load results for {ckey}")
            continue
        r0d = res.pop("ref_0d", None)
        results[ckey] = res
        if r0d is not None:
            ref_0d[ckey] = r0d

    if not results:
        print("ERROR: No results. Check TMCMC run directories.")
        sys.exit(1)

    # Summary
    print_summary(results, ref_0d)

    # Figures
    print("  Generating comparison figures...")
    plot_2d_fields(results, OUT_DIR)
    plot_di_comparison(results, OUT_DIR)
    plot_species_spatial(results, OUT_DIR)
    plot_1d_vs_2d(results, OUT_DIR)

    # Save summary JSON
    summary = {}
    for ckey, res in results.items():
        summary[ckey] = {
            "di_mean": res["di_mean"],
            "di_std": res["di_std"],
            "di_min": res["di_min"],
            "di_max": res["di_max"],
            "E_mean": res["E_mean"],
            "c_min_bf": res["c_min_bf"],
            "phi_Pg_mean": res["phi_Pg_mean"],
            "elapsed_s": res["elapsed_s"],
            "di_0d": ref_0d[ckey]["di_0d"] if ckey in ref_0d else None,
        }
    json_path = os.path.join(OUT_DIR, "2d_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {json_path}")

    print()
    print("=" * 85)
    print(f"  Complete! {len(results)}/{len(CONDITIONS)} conditions")
    print(f"  Output: {OUT_DIR}")
    print("=" * 85)


# ─────────────────────────────────────────────────────────────────────────────
# Viscoelastic 2D analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_2d_viscoelastic_analysis(
    theta: np.ndarray,
    condition_name: str,
    t_mech: np.ndarray = None,
    nu: float = 0.30,
    alpha_coeff: float = 0.05,
) -> dict:
    """
    Full viscoelastic pipeline: Hamilton ODE → DI(x,y) → VE params → FEM time integration.

    Two-stage approach:
      1. Growth phase (days): Hamilton ODE → steady-state φ, DI
      2. Mechanical event (seconds): step eigenstrain → SLS relaxation

    Parameters
    ----------
    theta : (20,) — Hamilton ODE parameters
    condition_name : str — condition key (e.g. 'commensal_static')
    t_mech : array — mechanical time points [s] (default: [0,1,5,10,30,60,120,300])
    nu : float — Poisson's ratio

    Returns
    -------
    dict with DI_final, VE params, FEM time history
    """
    from material_models import compute_viscoelastic_params_di

    if t_mech is None:
        t_mech = np.array([0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0])

    # Stage 1: Growth phase → steady-state fields
    print(f"  [{condition_name}] Stage 1: growth phase (Hamilton 2D)...")
    result_2d = run_2d(theta, condition_name)
    di_final = result_2d["di_final"]  # (NX, NY)
    mask = result_2d["biofilm_mask"]

    # Compute VE params from DI
    ve = compute_viscoelastic_params_di(di_final, di_scale=DI_SCALE)
    E_inf_field = ve["E_inf"]
    E_1_field = ve["E_1"]
    tau_field = ve["tau"]

    # Eigenstrain from growth
    eps_growth = result_2d["eps_growth"]

    # Stage 2: Mechanical event → viscoelastic FEM
    print(f"  [{condition_name}] Stage 2: VE FEM ({len(t_mech)} time steps)...")
    from JAXFEM.solve_stress_2d import solve_2d_fem_viscoelastic_sls

    fem_result = solve_2d_fem_viscoelastic_sls(
        E_inf_field,
        E_1_field,
        tau_field,
        nu,
        eps_growth,
        NX,
        NY,
        t_mech,
        Lx=1.0,
        Ly=1.0,
        bc_type="bottom_fixed",
    )

    # Summary stats inside biofilm
    bf = mask > 0.5
    di_bf = di_final[bf]

    return {
        "condition": condition_name,
        "di_final": di_final,
        "E_inf_field": E_inf_field,
        "E_1_field": E_1_field,
        "tau_field": tau_field,
        "eps_growth": eps_growth,
        "biofilm_mask": mask,
        "t_mech": t_mech,
        "fem_result": fem_result,
        "di_mean": float(di_bf.mean()),
        "di_std": float(di_bf.std()),
        "sigma_vm_t0": fem_result["sigma_vm_history"][0],
        "sigma_vm_final": fem_result["sigma_vm_history"][-1],
        "u_max_history": np.array(
            [
                np.sqrt(
                    fem_result["u_history"][ti, :, 0] ** 2 + fem_result["u_history"][ti, :, 1] ** 2
                ).max()
                for ti in range(len(t_mech))
            ]
        ),
    }


if __name__ == "__main__":
    main()
