#!/usr/bin/env python3
"""
generate_hybrid_macro_csv.py — P2: Hybrid macro CSV (0D composition × 1D spatial profile)
==========================================================================================

3つの E(x) モデルを比較:
  1. DI model:        E(DI_0D)     — entropy-based, species-blind
  2. φ_Pg model:      E(φ_Pg_0D)  — mechanism-based, Pg-specific
  3. Virulence model:  E(V_0D)    — Pg + Fn weighted

Hybrid アプローチ:
  φ_i(x) ← 1D Hamilton PDE の空間プロファイル  [空間構造を保持]
  DI, φ_Pg, V ← 0D Hamilton ODE の定常値       [条件差を保持]

出力
----
  FEM/_multiscale_results/
    macro_eigenstrain_{condition}_hybrid.csv  — 3 E モデル付き Hybrid CSV
    hybrid_3model_comparison.png             — DI vs φ_Pg vs Virulence 比較図

使い方
------
  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
      Tmcmc202601/FEM/generate_hybrid_macro_csv.py
"""

from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── パス設定 ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
_RUNS = os.path.join(_PROJ, "data_5species", "_runs")
_JAXFEM = os.path.join(_HERE, "JAXFEM")

for _p in [_HERE, _JAXFEM, _PROJ]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

IN_DIR = os.path.join(_HERE, "_multiscale_results")
OUT_DIR = IN_DIR

# ── material_models から 3 モデルをインポート ──────────────────────────────────
from material_models import (
    compute_E_di,
    E_MAX_PA,
    E_MIN_PA,
    DI_SCALE,
    PHI_PG_CRIT,
    HILL_M,
    W_PG,
    W_FN,
    V_CRIT,
    IDX_PG,
    IDX_FN,
)

# ── 条件→ランディレクトリ のマッピング ────────────────────────────────────────
CONDITIONS = {
    "commensal_static": {
        "run": "commensal_static",
        "color": "#1f77b4",
        "label": "Commensal Static",
    },
    "commensal_hobic": {
        "run": "commensal_hobic",
        "color": "#2ca02c",
        "label": "Commensal HOBIC",
    },
    "dysbiotic_static": {
        "run": "dysbiotic_static",
        "color": "#ff7f0e",
        "label": "Dysbiotic Static",
    },
    "dysbiotic_hobic": {
        "run": "dh_baseline",
        "color": "#d62728",
        "label": "Dysbiotic HOBIC",
    },
}

# ── マクロ材料パラメータ ──────────────────────────────────────────────────────
K_ALPHA = 0.05  # 成長–固有ひずみ結合
L_BIO_MM = 0.2  # バイオフィルム厚さ [mm]


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────


def compute_E_DI_scalar(di_scalar: float, n_nodes: int) -> np.ndarray:
    """DI スカラーから E_Pa 配列 (N,) を計算する (DI model)."""
    di_arr = np.full(n_nodes, di_scalar)
    return compute_E_di(di_arr, di_scale=1.0)


def compute_E_phi_pg_scalar(phi_5: np.ndarray, n_nodes: int) -> np.ndarray:
    """0D φ_Pg スカラーから E_Pa 配列 (N,) を計算する (φ_Pg model)."""
    phi_pg = float(phi_5[IDX_PG])
    # Hill sigmoid
    x = phi_pg / PHI_PG_CRIT
    xm = x**HILL_M
    sig = xm / (1.0 + xm)
    E = E_MAX_PA - (E_MAX_PA - E_MIN_PA) * sig
    return np.full(n_nodes, E)


def compute_E_virulence_scalar(phi_5: np.ndarray, n_nodes: int) -> np.ndarray:
    """0D Virulence スカラーから E_Pa 配列 (N,) を計算する (Virulence model)."""
    v = W_PG * float(phi_5[IDX_PG]) + W_FN * float(phi_5[IDX_FN])
    x = v / V_CRIT
    xm = x**HILL_M
    sig = xm / (1.0 + xm)
    E = E_MAX_PA - (E_MAX_PA - E_MIN_PA) * sig
    return np.full(n_nodes, E)


def load_theta(run_name: str) -> np.ndarray:
    """TMCMC ラン から theta_MAP を読み込む。"""
    path = os.path.join(_RUNS, run_name, "theta_MAP.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"theta_MAP.json not found: {path}")
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        return np.array(d[:20], dtype=np.float64)
    theta = d.get("theta_sub") or d.get("theta_full")
    return np.array(theta[:20], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 0D JAX Hamilton ODE (条件別 DI・φ_Pg)
# ─────────────────────────────────────────────────────────────────────────────


def solve_0d_composition(theta_np: np.ndarray, n_steps: int = 2500, dt: float = 0.01) -> dict:
    """
    0D JAX Hamilton ODE → DI_0D, φ_Pg_0D, V_0D, 全 E モデル。

    Returns
    -------
    dict:
      di_0d      : float
      phi_final  : (5,)
      phi_Pg_0d  : float
      V_0d       : float — Virulence index
      E_di       : float — E from DI model
      E_phi_pg   : float — E from φ_Pg model
      E_vir      : float — E from Virulence model
    """
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

    g0 = make_initial_state(1, active_mask)[0]  # (12,)

    def body(g, _):
        return newton_step(g, params), g

    _, g_traj = jax.lax.scan(jax.jit(body), g0, jnp.arange(n_steps))
    phi_traj = np.array(g_traj[:, 0:5])  # (n_steps, 5)
    phi_final = phi_traj[-1]  # (5,)

    # DI = 1 - H/H_max
    phi_sum = phi_final.sum()
    p = phi_final / max(phi_sum, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum()
    di_0d = float(1.0 - H / np.log(5.0))

    # φ_Pg, Virulence
    phi_pg_0d = float(phi_final[IDX_PG])
    v_0d = W_PG * phi_pg_0d + W_FN * float(phi_final[IDX_FN])

    # 各 E モデル (スカラー)
    E_di_val = float(compute_E_DI_scalar(di_0d, 1)[0])
    E_pg_val = float(compute_E_phi_pg_scalar(phi_final, 1)[0])
    E_vir_val = float(compute_E_virulence_scalar(phi_final, 1)[0])

    # EPS synergy model
    from material_models import compute_E_eps_synergy

    E_eps_val = float(compute_E_eps_synergy(phi_final.reshape(1, -1))[0])

    return {
        "di_0d": di_0d,
        "phi_final": phi_final,
        "phi_Pg_0d": phi_pg_0d,
        "V_0d": v_0d,
        "E_di": E_di_val,
        "E_phi_pg": E_pg_val,
        "E_vir": E_vir_val,
        "E_eps_synergy": E_eps_val,
        "phi_traj": phi_traj,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 既存 CSV の読み込み
# ─────────────────────────────────────────────────────────────────────────────


def _read_commented_csv(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    cols = lines[0].split(",")
    data = np.array(
        [[float(v) for v in l.split(",")] for l in lines[1:] if l.strip()],
        dtype=np.float64,
    )
    return {col: data[:, i] for i, col in enumerate(cols)}


def load_original_csv(condition_key: str) -> dict | None:
    """既存の macro_eigenstrain_*.csv を読み込む。"""
    path = os.path.join(IN_DIR, f"macro_eigenstrain_{condition_key}.csv")
    if not os.path.isfile(path):
        print(f"  [{condition_key}] 警告: 元 CSV が見つかりません: {path}")
        return None
    d = _read_commented_csv(path)
    return {
        "depth_mm": d["depth_mm"],
        "depth_norm": d["depth_norm"],
        "phi_So": d["phi_So"],
        "phi_An": d["phi_An"],
        "phi_Vd": d["phi_Vd"],
        "phi_Fn": d["phi_Fn"],
        "phi_Pg": d["phi_Pg"],
        "phi_total": d["phi_total"],
        "c": d["c"],
        "alpha": d["alpha"],
        "alpha_monod": d["alpha_monod"],
        "eps_growth": d["eps_growth"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid CSV 書き出し (3 E モデル付き)
# ─────────────────────────────────────────────────────────────────────────────


def export_hybrid_csv(orig: dict, res_0d: dict, condition_key: str) -> str:
    """
    0D composition × 1D spatial profile の Hybrid CSV (3 E モデル)。

    DI, φ_Pg, Virulence = 0D 値 (条件別定数) で E を計算。
    alpha_monod(x) は 1D PDE の空間勾配をそのまま保持。
    """
    N = len(orig["depth_mm"])
    di_0d = res_0d["di_0d"]
    phi_pg = res_0d["phi_Pg_0d"]
    v_0d = res_0d["V_0d"]
    E_di = res_0d["E_di"]
    E_pg = res_0d["E_phi_pg"]
    E_vir = res_0d["E_vir"]

    # Compute VE parameters from DI
    from material_models import compute_viscoelastic_params_di

    ve = compute_viscoelastic_params_di(np.array([di_0d]), di_scale=1.0)
    tau_val = float(ve["tau"][0])
    E_0_val = float(ve["E_0"][0])
    eta_val = float(ve["eta"][0])

    header = (
        "# Hybrid macro CSV: 0D composition × 1D spatial alpha_monod\n"
        f"# condition: {condition_key}\n"
        f"# DI_0D:    {di_0d:.6f}  → E_di    = {E_di:.1f} Pa\n"
        f"# phi_Pg_0D: {phi_pg:.6f}  → E_phi_pg = {E_pg:.1f} Pa  "
        f"(phi_crit={PHI_PG_CRIT}, m={HILL_M})\n"
        f"# V_0D:     {v_0d:.6f}  → E_vir   = {E_vir:.1f} Pa  "
        f"(V_crit={V_CRIT}, w_Pg={W_PG}, w_Fn={W_FN})\n"
        f"# VE: tau={tau_val:.2f}s, E_0={E_0_val:.1f}Pa, eta={eta_val:.1f}Pa·s\n"
        f"# alpha_monod(x): from 1D Hamilton + nutrient PDE  [spatial]\n"
        f"# eps_growth = alpha_monod / 3  [isotropic eigenstrain]\n"
        "depth_mm,depth_norm,"
        "phi_So,phi_An,phi_Vd,phi_Fn,phi_Pg,"
        "phi_total,c,DI,alpha,alpha_monod,eps_growth,"
        "E_di,E_phi_pg,E_virulence,"
        "tau_s,E_0_Pa,eta_Pas\n"
    )

    rows = []
    for k in range(N):
        row = (
            f"{orig['depth_mm'][k]:.8e},"
            f"{orig['depth_norm'][k]:.8e},"
            f"{orig['phi_So'][k]:.8e},"
            f"{orig['phi_An'][k]:.8e},"
            f"{orig['phi_Vd'][k]:.8e},"
            f"{orig['phi_Fn'][k]:.8e},"
            f"{orig['phi_Pg'][k]:.8e},"
            f"{orig['phi_total'][k]:.8e},"
            f"{orig['c'][k]:.8e},"
            f"{di_0d:.8e},"
            f"{orig['alpha'][k]:.8e},"
            f"{orig['alpha_monod'][k]:.8e},"
            f"{orig['eps_growth'][k]:.8e},"
            f"{E_di:.8e},"
            f"{E_pg:.8e},"
            f"{E_vir:.8e},"
            f"{tau_val:.8e},"
            f"{E_0_val:.8e},"
            f"{eta_val:.8e}"
        )
        rows.append(row)

    fname = f"macro_eigenstrain_{condition_key}_hybrid.csv"
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write(header + "\n".join(rows) + "\n")

    size_kb = os.path.getsize(path) / 1024
    print(f"  [{condition_key}] Hybrid CSV: {path}  ({size_kb:.1f} KB)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3 モデル比較図
# ─────────────────────────────────────────────────────────────────────────────


def plot_3model_comparison(summary: list[dict]) -> str:
    """
    DI vs φ_Pg vs Virulence の 3 モデル比較図 (論文 Fig. 11 候補)。

    Panel (a): 3 モデルの E(x) 応答曲線
    Panel (b): 条件別 E の棒グラフ (3 モデル並列)
    Panel (c): 条件別 σ_compressive at tooth end
    Panel (d): 0D 組成 (φ_Pg, φ_Fn) の条件別比較
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_curve, ax_E, ax_sigma, ax_phi = axes.ravel()

    n = len(summary)
    x = np.arange(n)
    labels = [s["label"] for s in summary]
    colors = [s["color"] for s in summary]

    E_di = [s["E_di"] for s in summary]
    E_pg = [s["E_phi_pg"] for s in summary]
    E_vir = [s["E_vir"] for s in summary]

    # --- (a) Model response curves ---
    from material_models import _hill_sigmoid

    phi_pg_arr = np.linspace(0, 0.8, 200)
    sig_pg = _hill_sigmoid(phi_pg_arr / PHI_PG_CRIT, HILL_M)
    E_curve = E_MAX_PA - (E_MAX_PA - E_MIN_PA) * sig_pg
    ax_curve.plot(phi_pg_arr, E_curve, "b-", lw=2, label=r"$E(\varphi_{Pg})$ model")
    ax_curve.axvline(
        PHI_PG_CRIT,
        color="red",
        ls="--",
        lw=1,
        alpha=0.7,
        label=f"$\\varphi_{{crit}}={PHI_PG_CRIT}$",
    )
    for s in summary:
        ax_curve.plot(
            s["phi_Pg_0d"],
            s["E_phi_pg"],
            "o",
            color=s["color"],
            ms=10,
            zorder=5,
            markeredgecolor="black",
        )
        ax_curve.annotate(
            s["label"].split()[0],
            (s["phi_Pg_0d"], s["E_phi_pg"]),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=7,
            color=s["color"],
        )
    ax_curve.set_xlabel(r"$\varphi_{Pg}$ (P. gingivalis volume fraction)")
    ax_curve.set_ylabel("E [Pa]")
    ax_curve.set_title(r"(a) $\varphi_{Pg}$ model: E($\varphi_{Pg}$) with Hill sigmoid")
    ax_curve.legend(fontsize=8)
    ax_curve.grid(alpha=0.3)
    ax_curve.set_ylim(bottom=0)

    # --- (b) Grouped bar: 3 models × 4 conditions ---
    w = 0.25
    bars_di = ax_E.bar(
        x - w,
        E_di,
        w,
        label="DI model",
        color=[c for c in colors],
        alpha=0.4,
        edgecolor="black",
        hatch="///",
    )
    bars_pg = ax_E.bar(
        x,
        E_pg,
        w,
        label=r"$\varphi_{Pg}$ model",
        color=[c for c in colors],
        alpha=0.85,
        edgecolor="black",
    )
    bars_vir = ax_E.bar(
        x + w,
        E_vir,
        w,
        label="Virulence model",
        color=[c for c in colors],
        alpha=0.6,
        edgecolor="black",
        hatch="\\\\\\",
    )

    ax_E.set_xticks(x)
    ax_E.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_E.set_ylabel("E [Pa]")
    ax_E.set_title("(b) Elastic modulus by condition (3 models)")
    ax_E.legend(fontsize=8)
    ax_E.set_ylim(0, E_MAX_PA * 1.15)
    ax_E.axhline(E_MAX_PA, ls=":", color="blue", alpha=0.3)
    ax_E.axhline(E_MIN_PA, ls=":", color="red", alpha=0.3)
    ax_E.grid(alpha=0.3, axis="y")

    for bars, vals in [(bars_di, E_di), (bars_pg, E_pg), (bars_vir, E_vir)]:
        for bar, val in zip(bars, vals):
            ax_E.text(
                bar.get_x() + bar.get_width() / 2,
                val + 15,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # --- (c) Compressive stress at tooth end: σ₀ = -E * eps_growth[0] ---
    eps_tooth = [s["eps_growth_tooth"] for s in summary]
    sigma_di = [-E_di[i] * eps_tooth[i] for i in range(n)]
    sigma_pg = [-E_pg[i] * eps_tooth[i] for i in range(n)]
    sigma_vir = [-E_vir[i] * eps_tooth[i] for i in range(n)]

    ax_sigma.bar(
        x - w,
        sigma_di,
        w,
        label="DI",
        color=[c for c in colors],
        alpha=0.4,
        edgecolor="black",
        hatch="///",
    )
    ax_sigma.bar(
        x,
        sigma_pg,
        w,
        label=r"$\varphi_{Pg}$",
        color=[c for c in colors],
        alpha=0.85,
        edgecolor="black",
    )
    ax_sigma.bar(
        x + w,
        sigma_vir,
        w,
        label="Virulence",
        color=[c for c in colors],
        alpha=0.6,
        edgecolor="black",
        hatch="\\\\\\",
    )
    ax_sigma.set_xticks(x)
    ax_sigma.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_sigma.set_ylabel(r"$\sigma_0$ [Pa]  (compressive, < 0)")
    ax_sigma.set_title(
        r"(c) Compressive prestress $\sigma_0 = -E \cdot \varepsilon_{growth}$ (tooth end)"
    )
    ax_sigma.legend(fontsize=8)
    ax_sigma.grid(alpha=0.3, axis="y")

    # --- (d) 0D composition: φ_Pg and φ_Fn ---
    phi_pg_vals = [s["phi_Pg_0d"] for s in summary]
    phi_fn_vals = [s["phi_Fn_0d"] for s in summary]
    phi_tot_vals = [s["phi_total_0d"] for s in summary]

    ax_phi.bar(
        x - w / 2,
        phi_pg_vals,
        w,
        label=r"$\varphi_{Pg}$ (P. gingivalis)",
        color="#d62728",
        alpha=0.85,
        edgecolor="black",
    )
    ax_phi.bar(
        x + w / 2,
        phi_fn_vals,
        w,
        label=r"$\varphi_{Fn}$ (F. nucleatum)",
        color="#9467bd",
        alpha=0.85,
        edgecolor="black",
    )
    ax_phi.axhline(
        PHI_PG_CRIT,
        color="red",
        ls="--",
        lw=1,
        alpha=0.7,
        label=f"$\\varphi_{{Pg,crit}}={PHI_PG_CRIT}$",
    )
    ax_phi.set_xticks(x)
    ax_phi.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_phi.set_ylabel("Volume fraction")
    ax_phi.set_title("(d) 0D steady-state pathogen fractions")
    ax_phi.legend(fontsize=8)
    ax_phi.grid(alpha=0.3, axis="y")
    ax_phi.set_ylim(0, max(phi_pg_vals + phi_fn_vals) * 1.2 + 0.05)

    for i, (pg, fn) in enumerate(zip(phi_pg_vals, phi_fn_vals)):
        ax_phi.text(x[i] - w / 2, pg + 0.01, f"{pg:.3f}", ha="center", va="bottom", fontsize=7)
        ax_phi.text(x[i] + w / 2, fn + 0.01, f"{fn:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "Material Model Comparison: DI (entropy) vs $\\varphi_{Pg}$ (mechanism) vs Virulence (Pg+Fn)\n"
        "0D Hamilton ODE → condition-specific composition → E mapping",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "hybrid_3model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  3 モデル比較図: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("  generate_hybrid_macro_csv.py — 3-Model Hybrid CSV 生成")
    print("=" * 70)
    print(
        f"  Models: DI (s={DI_SCALE:.4f}), "
        f"phi_Pg (crit={PHI_PG_CRIT}, m={HILL_M}), "
        f"Virulence (V_crit={V_CRIT})"
    )
    print()

    summary = []

    for ckey, meta in CONDITIONS.items():
        print(f"[{ckey}]  ({meta['label']})")

        # 1. theta_MAP ロード
        try:
            theta = load_theta(meta["run"])
        except FileNotFoundError as e:
            print(f"  エラー: {e}")
            continue

        # 2. 0D Hamilton ODE → DI_0D, φ_Pg_0D, V_0D, 3E
        t0 = time.time()
        print("  0D Hamilton ODE (n=2500, dt=0.01, T*=25) ...", flush=True)
        res0d = solve_0d_composition(theta)
        elapsed = time.time() - t0

        phi = res0d["phi_final"]
        print(f"  DI_0D  = {res0d['di_0d']:.4f}   → E_di    = {res0d['E_di']:.1f} Pa")
        print(f"  φ_Pg   = {res0d['phi_Pg_0d']:.4f}   → E_phi_pg = {res0d['E_phi_pg']:.1f} Pa")
        print(f"  V      = {res0d['V_0d']:.4f}   → E_vir   = {res0d['E_vir']:.1f} Pa")
        print(
            f"  φ: So={phi[0]:.4f} An={phi[1]:.4f} Vd={phi[2]:.4f} "
            f"Fn={phi[3]:.4f} Pg={phi[4]:.4f}  ({elapsed:.1f}s)"
        )

        # 3. 既存 1D CSV ロード
        orig = load_original_csv(ckey)
        if orig is None:
            print("  スキップ: 元 CSV なし。先に multiscale_coupling_1d.py を実行。")
            continue

        # 4. Hybrid CSV 出力 (3 E モデル)
        export_hybrid_csv(orig, res0d, ckey)

        summary.append(
            {
                "condition_key": ckey,
                "label": meta["label"],
                "color": meta["color"],
                "di_0d": res0d["di_0d"],
                "phi_Pg_0d": res0d["phi_Pg_0d"],
                "phi_Fn_0d": float(phi[IDX_FN]),
                "phi_total_0d": float(phi.sum()),
                "V_0d": res0d["V_0d"],
                "E_di": res0d["E_di"],
                "E_phi_pg": res0d["E_phi_pg"],
                "E_vir": res0d["E_vir"],
                "eps_growth_tooth": float(orig["eps_growth"][0]),
            }
        )
        print()

    if not summary:
        print("エラー: 処理できた条件がありません。")
        return

    # 5. 3 モデル比較図
    plot_3model_comparison(summary)

    # 6. サマリ表示
    print()
    print("=" * 78)
    print("3-MODEL HYBRID CSV サマリ")
    print(
        f"{'Condition':<25} {'DI_0D':>7} {'φ_Pg':>7} {'V':>7} "
        f"{'E_di':>7} {'E_φPg':>7} {'E_vir':>7}"
    )
    print("-" * 78)
    for s in summary:
        print(
            f"  {s['condition_key']:<23} "
            f"{s['di_0d']:>7.4f} "
            f"{s['phi_Pg_0d']:>7.4f} "
            f"{s['V_0d']:>7.4f} "
            f"{s['E_di']:>7.1f} "
            f"{s['E_phi_pg']:>7.1f} "
            f"{s['E_vir']:>7.1f}"
        )

    # commensal vs dysbiotic の E 比率
    if len(summary) >= 3:
        comm_avg_di = np.mean([s["E_di"] for s in summary if "commensal" in s["condition_key"]])
        dysb_avg_di = np.mean([s["E_di"] for s in summary if "dysbiotic" in s["condition_key"]])
        comm_avg_pg = np.mean([s["E_phi_pg"] for s in summary if "commensal" in s["condition_key"]])
        dysb_avg_pg = np.mean([s["E_phi_pg"] for s in summary if "dysbiotic" in s["condition_key"]])
        comm_avg_vir = np.mean([s["E_vir"] for s in summary if "commensal" in s["condition_key"]])
        dysb_avg_vir = np.mean([s["E_vir"] for s in summary if "dysbiotic" in s["condition_key"]])
        print()
        print("  E_commensal / E_dysbiotic:")
        print(
            f"    DI model:        {comm_avg_di:.0f} / {dysb_avg_di:.0f} = {comm_avg_di/max(dysb_avg_di,1e-6):.1f}x"
        )
        print(
            f"    φ_Pg model:      {comm_avg_pg:.0f} / {dysb_avg_pg:.0f} = {comm_avg_pg/max(dysb_avg_pg,1e-6):.1f}x"
        )
        print(
            f"    Virulence model: {comm_avg_vir:.0f} / {dysb_avg_vir:.0f} = {comm_avg_vir/max(dysb_avg_vir,1e-6):.1f}x"
        )

    print("=" * 78)
    print()
    print("次のステップ: generate_abaqus_eigenstrain.py を再実行")
    print("  → 3 E モデルの Abaqus inp を条件別に生成")
    print("=" * 78)


if __name__ == "__main__":
    main()
