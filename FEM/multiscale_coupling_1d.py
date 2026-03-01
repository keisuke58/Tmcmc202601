#!/usr/bin/env python3
"""
multiscale_coupling_1d.py — ミクロ↔マクロ 二スケール連成パイプライン
====================================================================

ミクロスケール (μm):  Hamilton PDE (5菌種 + 栄養輸送 Klempt 2024)
  → φᵢ(x,T_final),  c(x,T_final),  α(x,T) = k_α ∫φ_total dt

マクロスケール (mm):  Abaqus/FEM 入力 CSV を生成
  → 深さ方向の固有ひずみ・DI・E(DI) プロファイル

スケール変換
-----------
  x ∈ [0,1] (正規化ミクロ座標)
      x = 0 : 歯面 (tooth surface, no nutrient flux)
      x = 1 : 唾液側 (saliva, c = 1)
  depth_mm = x * L_biofilm   (L_biofilm = 0.2 mm, Abaqus biofilm mode に合わせる)

物理モデル (Klempt et al. 2024, Eq. 8–11 の1D版)
--------------------------------------------------
  Hamilton PDE (φᵢ, ψᵢ, γ の進化):
      → 各ノードを Newton 法で解く

  栄養 PDE:
      ∂c/∂t = D_c ∂²c/∂x² − g_eff φ_total c / (k_monod + c)
      BC: c(x=1, t) = 1  (Dirichlet: 唾液)
          ∂c/∂x|_{x=0} = 0  (Neumann: 歯面)

  成長固有ひずみ:
      α̇(x,t) = k_α φ_total(x,t)
      α(x,T) = k_α ∫₀ᵀ φ_total(x,t) dt    [空間場として計算]
      ε_growth(x) = α(x,T) / 3              [等方体積固有ひずみ成分]

フロー
------
  TMCMC MAP θ (4条件 × 20パラメータ)
      ↓
  Hamilton 1D + 栄養PDE (N=30 ノード, n_macro=80 ステップ)
      ↓
  φᵢ(x,T), c(x,T), α(x,T), DI(x), E(DI(x))
      ↓
  macro_eigenstrain_{condition}.csv  [Abaqus 固有ひずみ入力]
  multiscale_comparison.png          [4条件比較図 4パネル]

使い方
------
  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \\
      Tmcmc202601/FEM/multiscale_coupling_1d.py

出力
----
  FEM/_multiscale_results/
      ├── macro_eigenstrain_commensal_static.csv
      ├── macro_eigenstrain_commensal_hobic.csv
      ├── macro_eigenstrain_dysbiotic_static.csv
      ├── macro_eigenstrain_dysbiotic_hobic.csv
      └── multiscale_comparison.png

環境: klempt_fem conda env (Python 3.11, JAX 0.9.0.1)
参考: Klempt et al. (2024) Biomech Model Mechanobiol 23:2091-2113
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ── パス設定 ─────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))  # FEM/
_FEMROOT = _HERE  # FEM/
_PROJ = os.path.dirname(_HERE)  # Tmcmc202601/
_RUNS_DIR = os.path.join(_PROJ, "data_5species", "_runs")

if _FEMROOT not in sys.path:
    sys.path.insert(0, _FEMROOT)

# JAXFEM パッケージから連成 PDE ソルバーをインポート
from JAXFEM.core_hamilton_1d_nutrient import simulate_hamilton_1d_nutrient
from JAXFEM.core_hamilton_1d import (
    theta_to_matrices,
    newton_step,
    make_initial_state,
)

OUT_DIR = os.path.join(_HERE, "_multiscale_results")

# ── TMCMC 最良ラン (2026-02-08, 1000 particles) の MAP パラメータ ──────────────
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

# ── 物理パラメータ ────────────────────────────────────────────────────────────
# Klempt (2024) ベースの無次元パラメータ
D_EFF = jnp.array([0.001, 0.001, 0.0008, 0.0005, 0.0002])  # 菌種別拡散係数
D_C = 1.0  # 栄養拡散係数 (非次元)
G_EFF = 50.0  # 栄養消費係数 → Thiele 数 ~ 4
K_MONOD = 1.0  # Monod 半飽和定数
K_ALPHA = 0.05  # 成長–固有ひずみ結合 [T*^{-1}]

# シミュレーション設定
# 目標: T_total = N_MACRO * N_REACT * DT_H = 1000 * 20 * 1e-3 = 20 T*
# (バイオフィルム成長が収束するまでの典型的な無次元時刻)
# CFL 安定性:
#   栄養PDE: dt_c = N_REACT * DT_H / N_SUB_C = 20e-3/60 = 3.3e-4 < dx²/(2Dc) = 5.9e-4 ✓
#   菌種拡散: dt_diff = N_REACT * DT_H = 0.02 < dx²/(2 D_max) = 0.59 ✓
N_NODES = 30  # 空間ノード数
N_MACRO = 1000  # マクロタイムステップ数  → T_total = 20 T*
N_REACT = 20  # 反応サブステップ数
N_SUB_C = 60  # 栄養 PDE サブステップ数 (CFL 安定: dt_c = 3.3e-4)
DT_H = 1e-3  # Hamilton タイムステップ (0D ODE の dt=0.01 より小さく安全)

# スケール変換
L_BIOFILM_MM = 0.2  # バイオフィルム厚さ [mm] (Abaqus biofilm mode と一致)

# マクロ材料パラメータ (material_models.py に集約、互換性のため残す)
E_MAX_PA = 1000.0  # E_max [Pa] (EPS-rich commensal)
E_MIN_PA = 10.0  # E_min [Pa] (Klempt 2024 cite)
DI_SCALE = 0.025778  # DI スケール係数 (aggregate_di_credible.py より)
N_POWER = 2.0  # 冪乗則指数

# φ_Pg / Virulence モデル (material_models.py)
from material_models import compute_E_phi_pg, compute_E_virulence

# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ関数
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


def compute_di(phi_species: np.ndarray) -> np.ndarray:
    """
    空間 Dysbiotic Index DI(x) を計算する。

    phi_species : (N, 5) float — 菌種別 volume fraction
    Returns     : (N,) float — DI ∈ [0,1]  (0=均一, 1=単一種独占)

    DI = 1 - H/H_max,  H = -Σ p_i ln p_i,  H_max = ln(5)
    """
    phi_sum = phi_species.sum(axis=1, keepdims=True)
    phi_sum = np.where(phi_sum > 0, phi_sum, 1.0)
    p = phi_species / phi_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum(axis=1)
    return 1.0 - H / np.log(5.0)


def compute_E_DI(di: np.ndarray) -> np.ndarray:
    """E(DI) 冪乗則: E = E_max*(1-r)^n + E_min*r, r = clamp(DI/s, 0, 1)."""
    r = np.clip(di / DI_SCALE, 0.0, 1.0)
    return E_MAX_PA * (1.0 - r) ** N_POWER + E_MIN_PA * r


def compute_E_EPS_synergy(phi_species: np.ndarray) -> np.ndarray:
    """EPS synergy model: φ → E (delegated to material_models)."""
    from material_models import compute_E_eps_synergy

    return compute_E_eps_synergy(np.atleast_2d(phi_species)).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# 0D Hamilton JAX ソルバー (条件別 DI・組成推定)
# ─────────────────────────────────────────────────────────────────────────────


def solve_0d_hamilton_jax(
    theta_np: np.ndarray,
    n_steps: int = 2500,
    dt: float = 0.01,
) -> dict:
    """
    TMCMC MAP θ から 0D Hamilton ODE を JAX で解き、条件別の定常組成を返す。

    0D = 単一空間ノード (拡散なし)。
    TMCMC フィッティング時の BiofilmNewtonSolver5S と同等の物理。

    Returns
    -------
    phi_final   : (5,)  定常状態の菌種 volume fraction
    alpha_0d    : float α = k_α ∫ φ_total dt  (0D スカラー)
    di_0d       : float Dysbiotic Index (0D スカラー)
    phi_traj    : (n_steps, 5)  軌跡
    t_axis      : (n_steps,)  時間軸 [T*]
    """
    theta_jax = jnp.array(theta_np, dtype=jnp.float64)
    A, b_diag = theta_to_matrices(theta_jax)
    active_mask = jnp.ones(5, dtype=jnp.int64)

    params_0d = {
        "dt_h": dt,
        "Kp1": 1e-4,
        "Eta": jnp.ones(5, dtype=jnp.float64),
        "EtaPhi": jnp.ones(5, dtype=jnp.float64),
        "c": 100.0,  # 固定結合定数 (TMCMC ODE と同じ)
        "alpha": 100.0,  # Lennard-Jones ポテンシャル高さ
        "K_hill": jnp.array(0.05, dtype=jnp.float64),
        "n_hill": jnp.array(4.0, dtype=jnp.float64),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
        "newton_steps": 6,
    }

    # 初期状態 (N=1 ノード)
    g0 = make_initial_state(1, active_mask)[0]  # (12,)

    def body(g, _):
        return newton_step(g, params_0d), g

    _, g_traj = jax.lax.scan(jax.jit(body), g0, jnp.arange(n_steps))
    # g_traj: (n_steps, 12)

    phi_traj = np.array(g_traj[:, 0:5])  # (n_steps, 5)
    phi_total = phi_traj.sum(axis=1)  # (n_steps,)
    t_axis = np.arange(n_steps, dtype=float) * dt
    alpha_0d = float(K_ALPHA * np.trapezoid(phi_total, t_axis))
    phi_final = phi_traj[-1]  # (5,)

    # 0D DI
    phi_sum = phi_final.sum()
    p = phi_final / max(phi_sum, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H_0d = -(p * log_p).sum()
    di_0d = float(1.0 - H_0d / np.log(5.0))

    return {
        "phi_final": phi_final,
        "phi_total_final": float(phi_sum),
        "alpha_0d": alpha_0d,
        "di_0d": di_0d,
        "phi_traj": phi_traj,
        "t_axis": t_axis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ミクロシミュレーション: 1条件の解析
# ─────────────────────────────────────────────────────────────────────────────


def run_micro_simulation(theta: np.ndarray, condition_label: str) -> dict:
    """
    Hamilton 1D + 栄養 PDE をミクロスケールで解き、マクロ連成用フィールドを返す。

    Returns dict with:
      x_norm   : (N,) 正規化座標 [0=歯面, 1=唾液]
      depth_mm : (N,) 物理深さ [mm]
      phi      : (N, 5) 最終時刻の菌種 volume fraction
      phi_total: (N,) 菌種合計
      c        : (N,) 最終時刻の栄養濃度
      alpha    : (N,) 成長固有ひずみ α(x,T)
      eps_growth: (N,) 等方体積固有ひずみ = α/3
      di       : (N,) Dysbiotic Index
      E_Pa     : (N,) E(DI) [Pa]
    """
    theta_jax = jnp.array(theta, dtype=jnp.float64)

    t0 = time.time()
    print(
        f"  [{condition_label}] Hamilton 1D + 栄養PDE を実行中 "
        f"(N={N_NODES}, n_macro={N_MACRO}) ...",
        flush=True,
    )

    G_all, c_all = simulate_hamilton_1d_nutrient(
        theta=theta_jax,
        D_eff=D_EFF,
        D_c=D_C,
        g_eff=G_EFF,
        k_monod=K_MONOD,
        k_alpha=K_ALPHA,
        n_macro=N_MACRO,
        n_react_sub=N_REACT,
        n_sub_c=N_SUB_C,
        N=N_NODES,
        L=1.0,
        dt_h=DT_H,
    )
    # G_all shape: (n_macro+1, N, 12)  — 12 = [φ₁..φ₅, φ₀, ψ₁..ψ₅, γ]
    # c_all shape: (n_macro+1, N)

    elapsed = time.time() - t0
    logger.info("  [{condition_label}] 完了 ({elapsed:.1f} s)", flush=True)

    # ── α(x,T) を時間積分で計算 ──────────────────────────────────────────────
    dt_macro = DT_H * N_REACT  # マクロステップの時間幅 [T*]
    phi_total_traj = np.array(G_all[:, :, 0:5].sum(axis=2))  # (n_macro+1, N)
    c_traj = np.array(c_all)  # (n_macro+1, N)
    t_axis = np.arange(N_MACRO + 1, dtype=float) * dt_macro

    # (A) 単純積分 α(x,T) = k_α ∫ φ_total dt  [拡散のみ考慮]
    alpha_field = K_ALPHA * np.trapezoid(phi_total_traj, t_axis, axis=0)  # (N,)

    # (B) 栄養制限型 α_Monod(x,T) = k_α ∫ φ_total · c/(k+c) dt  [KEY: 空間非一様]
    #   near tooth (x=0): c≈0.004 → c/(k+c)≈0.004  → alpha_monod ≈ 0
    #   near saliva(x=1): c=1.000 → c/(k+c)≈0.500  → alpha_monod ≈ alpha/2
    monod_factor = c_traj / (K_MONOD + c_traj)  # (n_macro+1, N)
    alpha_monod = K_ALPHA * np.trapezoid(phi_total_traj * monod_factor, t_axis, axis=0)  # (N,)

    # ── 最終時刻の場 ─────────────────────────────────────────────────────────
    phi_final = np.array(G_all[-1, :, 0:5])  # (N, 5)
    phi_total_f = phi_final.sum(axis=1)  # (N,)
    c_final = np.array(c_all[-1, :])  # (N,)

    # ── マクロ量計算 ─────────────────────────────────────────────────────────
    di = compute_di(phi_final)
    E_Pa = compute_E_DI(di)
    eps_gr = alpha_monod / 3.0  # 栄養制限型を主固有ひずみとして使用

    # φ_Pg / Virulence ベース E(x) (メカニズムベース)
    E_phi_pg = compute_E_phi_pg(phi_final)
    E_vir = compute_E_virulence(phi_final)

    # ── スケール変換: 正規化 x → 物理深さ [mm] ───────────────────────────────
    x_norm = np.linspace(0.0, 1.0, N_NODES)
    depth_mm = x_norm * L_BIOFILM_MM

    return {
        "x_norm": x_norm,
        "depth_mm": depth_mm,
        "phi": phi_final,
        "phi_total": phi_total_f,
        "c": c_final,
        "alpha": alpha_field,  # (A) 単純積分 α(x)
        "alpha_monod": alpha_monod,  # (B) 栄養制限型 α_Monod(x) [空間非一様]
        "eps_growth": eps_gr,  # = alpha_monod / 3
        "di": di,
        "E_Pa": E_Pa,  # DI-based (legacy)
        "E_phi_pg": E_phi_pg,  # φ_Pg-based (mechanism)
        "E_virulence": E_vir,  # Pg+Fn weighted (mechanism)
        "phi_Pg": phi_final[:, 4],  # P. gingivalis fraction
        "t_end_Tstar": float(t_axis[-1]),
        "elapsed_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV エクスポート (Abaqus 固有ひずみ入力)
# ─────────────────────────────────────────────────────────────────────────────


def export_macro_csv(result: dict, condition_key: str) -> str:
    """
    マクロ FEM 入力 CSV を書き出す。

    各行は深さ方向の 1 ノードに対応し、以下のカラムを持つ:
      depth_mm, depth_norm,
      phi1..phi5 (菌種別 φ),
      phi_total, c, DI, alpha, eps_growth, E_Pa

    biofilm_conformal_tet.py の --growth-eigenstrain フラグへの入力や、
    将来の深さ依存 USDFLD として利用できる。
    """
    fname = f"macro_eigenstrain_{condition_key}.csv"
    path = os.path.join(OUT_DIR, fname)

    header = (
        "# Two-scale coupling: micro Hamilton 1D → macro Abaqus eigenstrain\n"
        f"# condition: {condition_key}\n"
        f"# L_biofilm: {L_BIOFILM_MM} mm  (x=0: tooth, x=1: saliva)\n"
        f"# k_alpha: {K_ALPHA}  (growth-eigenstrain coupling)\n"
        f"# t_end_Tstar: {result['t_end_Tstar']:.4f}\n"
        "# alpha: simple integral k_alpha*int(phi_total dt)\n"
        "# alpha_monod: nutrient-limited k_alpha*int(phi_total*c/(k+c) dt) [KEY]\n"
        "depth_mm,depth_norm,"
        "phi_So,phi_An,phi_Vd,phi_Fn,phi_Pg,"
        "phi_total,c,DI,alpha,alpha_monod,eps_growth,E_Pa,E_phi_pg,E_virulence\n"
    )

    rows = []
    for k in range(N_NODES):
        row = (
            f"{result['depth_mm'][k]:.8e},"
            f"{result['x_norm'][k]:.8e},"
            + ",".join(f"{result['phi'][k, i]:.8e}" for i in range(5))
            + f",{result['phi_total'][k]:.8e}"
            f",{result['c'][k]:.8e}"
            f",{result['di'][k]:.8e}"
            f",{result['alpha'][k]:.8e}"
            f",{result['alpha_monod'][k]:.8e}"
            f",{result['eps_growth'][k]:.8e}"
            f",{result['E_Pa'][k]:.8e}"
            f",{result['E_phi_pg'][k]:.8e}"
            f",{result['E_virulence'][k]:.8e}"
        )
        rows.append(row)

    with open(path, "w") as f:
        f.write(header)
        f.write("\n".join(rows) + "\n")

    size_kb = os.path.getsize(path) / 1024
    logger.info("  CSV 出力: %s  (%.1f KB, %d nodes)", path, size_kb, N_NODES)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4条件比較図 (4パネル)
# ─────────────────────────────────────────────────────────────────────────────


def plot_multiscale_comparison(results: dict[str, dict]) -> str:
    """
    4パネル比較図を生成する。

    Panel 1 (TL): φ_total(depth) — 菌叢量の深さプロファイル
    Panel 2 (TR): c(depth) — 栄養濃度の深さプロファイル
    Panel 3 (BL): α(depth) — 成長固有ひずみ場 [ミクロ→マクロ連成の核心]
    Panel 4 (BR): DI(depth) + E(DI) 右軸 — Dysbiotic Index と局所剛性
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    ax_phi, ax_c, ax_alpha, ax_di = axes

    for ckey, res in results.items():
        info = CONDITIONS[ckey]
        x = res["depth_mm"]
        kw = dict(color=info["color"], ls=info["linestyle"], lw=1.8, label=info["label"])

        # Panel 1: φ_total
        ax_phi.plot(x, res["phi_total"], **kw)

        # Panel 2: c
        ax_c.plot(x, res["c"], **kw)

        # Panel 3: α_Monod (key panel: spatially non-uniform due to c(x))
        ax_alpha.plot(x, res["alpha_monod"], **kw)

        # Panel 4: DI (left) + E_Pa (right, shared)
        ax_di.plot(x, res["di"], **kw)

    # ── Panel 1: φ_total ──────────────────────────────────────────────────────
    ax_phi.set_xlabel("Depth from tooth surface [mm]")
    ax_phi.set_ylabel(r"$\phi_\mathrm{total}$ (total biofilm fraction)")
    ax_phi.set_title("(a) Biofilm volume fraction profile")
    ax_phi.set_xlim([0, L_BIOFILM_MM])
    ax_phi.set_ylim(bottom=0)
    ax_phi.legend(fontsize=8, loc="upper right")
    ax_phi.grid(alpha=0.3)
    ax_phi.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_phi.text(
        0.005, ax_phi.get_ylim()[1] * 0.95, "tooth", fontsize=7, color="gray", ha="left", va="top"
    )
    ax_phi.text(
        L_BIOFILM_MM * 0.98,
        ax_phi.get_ylim()[1] * 0.95,
        "saliva",
        fontsize=7,
        color="gray",
        ha="right",
        va="top",
    )

    # ── Panel 2: c (nutrient) ─────────────────────────────────────────────────
    ax_c.set_xlabel("Depth from tooth surface [mm]")
    ax_c.set_ylabel(r"$c$ (nutrient concentration, normalized)")
    ax_c.set_title("(b) Steady-state nutrient profile (Klempt 2024 RD-PDE)")
    ax_c.set_xlim([0, L_BIOFILM_MM])
    ax_c.set_ylim([0, 1.05])
    ax_c.legend(fontsize=8, loc="lower right")
    ax_c.grid(alpha=0.3)
    ax_c.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, label="BC: c=1 (saliva)")

    # ── Panel 3: α_Monod (key panel: spatially non-uniform) ──────────────────
    ax_alpha.set_xlabel("Depth from tooth surface [mm]")
    ax_alpha.set_ylabel(
        r"$\alpha_\mathrm{Monod}(x,T) = k_\alpha \int_0^T \phi_\mathrm{tot}" r"\frac{c}{k+c}\,dt$"
    )
    ax_alpha.set_title(
        r"(c) Nutrient-limited growth eigenstrain $\alpha_\mathrm{Monod}(x)$"
        "\n[Micro→Macro bridge: highest near saliva, suppressed near tooth]"
    )
    ax_alpha.set_xlim([0, L_BIOFILM_MM])
    ax_alpha.set_ylim(bottom=0)
    ax_alpha.legend(fontsize=8, loc="lower right")
    ax_alpha.grid(alpha=0.3)
    ax_alpha.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_alpha.text(
        0.005,
        ax_alpha.get_ylim()[1] * 0.5,
        "nutrient\ndeprived",
        fontsize=6,
        color="gray",
        ha="left",
        va="center",
        style="italic",
    )
    # ε_growth right axis
    ax_alpha2 = ax_alpha.twinx()
    for ckey, res in results.items():
        ax_alpha2.plot(
            res["depth_mm"],
            res["eps_growth"],
            color=CONDITIONS[ckey]["color"],
            ls=CONDITIONS[ckey]["linestyle"],
            lw=0.8,
            alpha=0.5,
        )
    ax_alpha2.set_ylabel(
        r"$\varepsilon_\mathrm{growth} = \alpha_\mathrm{Monod}/3$", color="gray", fontsize=8
    )
    ax_alpha2.tick_params(axis="y", labelcolor="gray")
    ax_alpha2.set_ylim(bottom=0)

    # ── Panel 4: φ_Pg + E comparison ────────────────────────────────────────────
    ax_di.set_xlabel("Depth from tooth surface [mm]")
    ax_di.set_ylabel(r"$\varphi_\mathrm{Pg}$")
    ax_di.set_title(r"(d) E: $\varphi_\mathrm{Pg}$-based (solid) vs DI-based (dotted)")
    ax_di.set_xlim([0, L_BIOFILM_MM])
    ax_di.legend(fontsize=8, loc="upper right")
    ax_di.grid(alpha=0.3)
    for ckey, res in results.items():
        ax_di.plot(
            res["depth_mm"],
            res["phi_Pg"],
            color=CONDITIONS[ckey]["color"],
            ls=CONDITIONS[ckey]["linestyle"],
            lw=1.2,
            alpha=0.7,
        )
    ax_di.set_ylim(bottom=0)
    ax_e = ax_di.twinx()
    for ckey, res in results.items():
        c = CONDITIONS[ckey]["color"]
        ax_e.plot(res["depth_mm"], res["E_phi_pg"], color=c, ls="-", lw=1.5, alpha=0.6)
        ax_e.plot(res["depth_mm"], res["E_Pa"], color=c, ls=":", lw=0.8, alpha=0.4)
    ax_e.set_ylabel("E [Pa]  (solid=φ_Pg, dotted=DI)", color="gray", fontsize=8)
    ax_e.tick_params(axis="y", labelcolor="gray")
    ax_e.set_ylim([0, E_MAX_PA * 1.05])

    plt.suptitle(
        "Two-scale Coupling: Micro (Hamilton PDE, μm) → Macro (Abaqus FEM, mm)\n"
        f"k_α={K_ALPHA}, g_eff={G_EFF}, N={N_NODES}, n_macro={N_MACRO}  |  "
        f"L_biofilm={L_BIOFILM_MM} mm",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "multiscale_comparison.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("  図 出力: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# サマリーテーブル
# ─────────────────────────────────────────────────────────────────────────────


def print_summary_table(results: dict[str, dict]) -> None:
    """4条件のマクロ量サマリーを表示する。"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("  二スケール連成 サマリー (ミクロ → マクロ)")
    logger.info(
        "  k_α = %s,  L_biofilm = %s mm,  N = %d,  n_macro = %d",
        K_ALPHA,
        L_BIOFILM_MM,
        N_NODES,
        N_MACRO,
    )
    logger.info("=" * 80)
    hdr = (
        f"  {'条件':<22}  {'α_Monod[tooth]':>14}  {'α_Monod[sal]':>12}  "
        f"{'ratio':>6}  {'c_min':>7}  {'E_mean[Pa]':>11}"
    )
    logger.info("%s", hdr)
    logger.info("  " + "-" * 80)
    for ckey, res in results.items():
        label = CONDITIONS[ckey]["label"]
        am = res["alpha_monod"]
        tooth_val = am[0]
        sal_val = am[-1]
        ratio = sal_val / max(tooth_val, 1e-12)
        logger.info(
            "  %-22s  %14.4f  %12.4f  %6.1fx  %7.4f  %11.1f",
            label,
            tooth_val,
            sal_val,
            ratio,
            res["c"].min(),
            res["E_Pa"].mean(),
        )
    logger.info("")
    logger.info("  KEY: α_Monod(x,T) = k_α ∫ φ_total · c/(k+c) dt")
    logger.info("       x=0(歯面)で c≈0.004 → α_Monod ≈ 0  (栄養枯渇で成長なし)")
    logger.info("       x=1(唾液)で c=1.000 → α_Monod ≈ α/2 (栄養豊富で成長あり)")
    logger.info("       → 空間非一様固有ひずみ場 ε_growth = α_Monod/3 として Abaqus に入力")
    logger.info("")

    ckeys = list(results.keys())
    if len(ckeys) >= 2:
        ref = results[ckeys[0]]
        logger.info("  α_mean 差分 (基準: %s):", CONDITIONS[ckeys[0]]["label"])
        for ck in ckeys[1:]:
            diff = results[ck]["alpha"].mean() - ref["alpha"].mean()
            pct = diff / (ref["alpha"].mean() + 1e-12) * 100
            logger.info("    %-22s: Δα = %+.4f  (%+.1f%%)", CONDITIONS[ck]["label"], diff, pct)
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Abaqus コマンド候補を表示
# ─────────────────────────────────────────────────────────────────────────────


def print_abaqus_next_steps(csv_paths: dict[str, str]) -> None:
    """
    生成した CSV を使って Abaqus biofilm mode + eigenstrain を実行する
    コマンド例を表示する。
    """
    logger.info("=" * 80)
    logger.info("  次のステップ: マクロ FEM (Abaqus) への固有ひずみ入力")
    logger.info("=" * 80)
    logger.info("")
    logger.info("  各条件の代表的 α_mean を --growth-eigenstrain に渡す:")
    logger.info("")
    stl = "external_tooth_models/.../P1_Tooth_23.stl"
    for ckey, csv_path in csv_paths.items():
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=6)
        alpha_mean = data[:, 10].mean()
        logger.info("  # %s", CONDITIONS[ckey]["label"])
        logger.info("  python3 biofilm_conformal_tet.py \\")
        logger.info("      --stl %s \\", stl)
        logger.info("      --di-csv _di_credible/%s/p50_field.csv \\", ckey)
        logger.info("      --out p23_%s_multiscale.inp \\", ckey)
        logger.info("      --mode biofilm \\")
        logger.info("      --growth-eigenstrain %.4f", alpha_mean)
        logger.info("")
    logger.info("  詳細 CSV (深さ方向プロファイル) は将来の USDFLD/FORTRAN 実装で直接利用可。")
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("")
    logger.info("=" * 80)
    logger.info("  multiscale_coupling_1d.py")
    logger.info("  ミクロ↔マクロ 二スケール連成パイプライン")
    logger.info("  出力先: %s", OUT_DIR)
    logger.info("=" * 80)

    results = {}
    res_0d = {}
    csv_paths = {}

    for ckey in CONDITIONS:
        info = CONDITIONS[ckey]
        logger.info("")
        logger.info("── %s (%s) ──", info["label"], info["run"])

        try:
            theta = load_theta(ckey)
        except FileNotFoundError as e:
            logger.info("  SKIP: %s", e)
            continue
        logger.info(
            "  θ[18]=a35=%.4f  θ[19]=a45=%.4f  (Vd→Pg, Fn→Pg 促進)",
            theta[18],
            theta[19],
        )

        logger.info("  [0D] JAX Hamilton ODE (n=2500, dt=0.01, T*=25) ...")
        r0d = solve_0d_hamilton_jax(theta, n_steps=2500, dt=0.01)
        res_0d[ckey] = r0d
        phi0 = r0d["phi_final"]
        logger.info(
            "  [0D] α_0D=%.4f  DI_0D=%.4f  φ_Pg=%.4f  φ_tot=%.4f",
            r0d["alpha_0d"],
            r0d["di_0d"],
            phi0[4],
            r0d["phi_total_final"],
        )

        res = run_micro_simulation(theta, info["label"])
        results[ckey] = res

        csv_path = export_macro_csv(res, ckey)
        csv_paths[ckey] = csv_path

        ratio_1d = res["alpha_monod"][-1] / max(res["alpha_monod"][0], 1e-9)
        logger.info(
            "  [1D] α_Monod: tooth=%.4f  saliva=%.4f  ratio=%.1fx",
            res["alpha_monod"][0],
            res["alpha_monod"][-1],
            ratio_1d,
        )

    if not results:
        logger.error("シミュレーション結果なし。TMCMC ランディレクトリを確認してください。")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 80)
    logger.info("  [0D Hamilton ODE] 条件別 定常組成 (T*=25, k_α=0.05)")
    logger.info("=" * 80)
    hdr0 = (
        f"  {'条件':<22}  {'DI_0D':>7}  {'α_0D':>7}  "
        f"{'φ_So':>6}  {'φ_Vd':>6}  {'φ_Pg':>6}  {'φ_tot':>6}"
    )
    logger.info("%s", hdr0)
    logger.info("  " + "-" * 68)
    for ck, r0 in res_0d.items():
        φ = r0["phi_final"]
        logger.info(
            "  %-22s  %7.4f  %7.4f  %6.4f  %6.4f  %6.4f  %6.4f",
            CONDITIONS[ck]["label"],
            r0["di_0d"],
            r0["alpha_0d"],
            φ[0],
            φ[2],
            φ[4],
            r0["phi_total_final"],
        )
    logger.info("")
    logger.info("  [1D Hamilton PDE] 空間プロファイル (栄養制限型 α_Monod)")
    logger.info("  DI≈0 in 1D: diffusion homogenizes species → condition diff. from 0D")

    print_summary_table(results)

    logger.info("  4条件比較図を生成中 ...")
    plot_multiscale_comparison(results)

    print_abaqus_next_steps(csv_paths)

    logger.info("=" * 80)
    logger.info("  完了! 出力: %s", OUT_DIR)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
