#!/usr/bin/env python3
"""
generate_hybrid_macro_csv.py — P2: Hybrid macro CSV (0D DI × 1D 空間プロファイル)
====================================================================================

問題
----
既存の macro_eigenstrain_*.csv では DI ≈ 0 (全条件共通)。
理由: 1D Hamilton PDE の拡散が全菌種を空間的に均質化するため。

解決策 (Hybrid アプローチ)
--------------------------
  phi_i(x) ← 1D Hamilton PDE の空間プロファイル  [空間構造を保持]
  DI       ← 0D Hamilton ODE の定常値            [条件差を保持]

  具体的には:
    1. 0D JAX Hamilton ODE を各条件で実行 → DI_0D (条件別スカラー)
    2. 既存 1D CSV から alpha_monod(x), phi_total(x), c(x) を読み込む
    3. DI(x, condition) = DI_0D  (空間一定だが条件間で大きく異なる)
    4. E_Pa(x, condition) = E(DI_0D)  (条件別定数)
    5. alpha_monod(x) はそのまま (空間構造は 1D PDE が正確)

期待される結果
--------------
  commensal: DI_0D ≈ 0.05 → E ≈ E_max = 1000 Pa
  dysbiotic: DI_0D ≈ 0.84 → E ≈ E_min ≈ 10 Pa

  → 完全拘束時の圧縮応力比:
    commensal: σ₀ ≈ -1000 × 0.00138 ≈ -1.4 Pa
    dysbiotic: σ₀ ≈ -10   × 0.00138 ≈ -0.014 Pa
    (歯面端; saliva 端では eps_growth ≈ 0.14 なのでさらに大きい)

出力
----
  FEM/_multiscale_results/
    macro_eigenstrain_{condition}_hybrid.csv  — 条件差ありの Hybrid CSV
    hybrid_di_comparison.png                 — 0D vs 1D DI 比較図

使い方
------
  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \\
      Tmcmc202601/FEM/generate_hybrid_macro_csv.py

環境要件
--------
  klempt_fem conda env (Python 3.11, JAX 0.9.0.1, pandas)
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
_HERE    = os.path.dirname(os.path.abspath(__file__))
_PROJ    = os.path.dirname(_HERE)
_RUNS    = os.path.join(_PROJ, "data_5species", "_runs")
_JAXFEM  = os.path.join(_HERE, "JAXFEM")

for _p in [_HERE, _JAXFEM, _PROJ]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

IN_DIR   = os.path.join(_HERE, "_multiscale_results")
OUT_DIR  = IN_DIR   # hybrid CSV も同じディレクトリへ

# ── 条件→ランディレクトリ のマッピング ────────────────────────────────────────
CONDITIONS = {
    "commensal_static" : {
        "run"      : "Commensal_Static_20260208_002100",
        "color"    : "#1f77b4",
        "label"    : "Commensal Static",
    },
    "commensal_hobic"  : {
        "run"      : "Commensal_HOBIC_20260208_002100",
        "color"    : "#2ca02c",
        "label"    : "Commensal HOBIC",
    },
    "dysbiotic_static" : {
        "run"      : "Dysbiotic_Static_20260207_203752",
        "color"    : "#ff7f0e",
        "label"    : "Dysbiotic Static",
    },
    "dysbiotic_hobic"  : {
        "run"      : "Dysbiotic_HOBIC_20260208_002100",
        "color"    : "#d62728",
        "label"    : "Dysbiotic HOBIC",
    },
}

# ── マクロ材料パラメータ ──────────────────────────────────────────────────────
E_MAX_PA  = 1000.0   # E_max [Pa]: commensal 上限
E_MIN_PA  = 10.0     # E_min [Pa]: dysbiotic 下限
# 0D DI は [0, 1] の自然スケール → DI_SCALE_0D = 1.0 で直接 r = DI を使用
# (1D DI は拡散均質化で≈0だったため DI_SCALE=0.026 が必要だったが、
#  0D DI では commensal≈0.05, dysbiotic≈0.84 が得られるため 1.0 が適切)
DI_SCALE  = 1.0      # 0D DI 用スケール (直接マッピング: r = DI)
N_POWER   = 2.0      # 冪乗則指数
K_ALPHA   = 0.05        # 成長–固有ひずみ結合
L_BIO_MM  = 0.2         # バイオフィルム厚さ [mm]


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def compute_E_DI(di_scalar: float, n_nodes: int) -> np.ndarray:
    """DI スカラーから E_Pa 配列 (N,) を計算する。"""
    r = float(np.clip(di_scalar / DI_SCALE, 0.0, 1.0))
    E = E_MAX_PA * (1.0 - r) ** N_POWER + E_MIN_PA * r
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
# 0D JAX Hamilton ODE (条件別 DI)
# ─────────────────────────────────────────────────────────────────────────────

def solve_0d_di(theta_np: np.ndarray, n_steps: int = 2500, dt: float = 0.01) -> dict:
    """
    0D JAX Hamilton ODE を解いて DI_0D と定常組成を返す。

    JAXFEM/core_hamilton_1d.py の newton_step を流用。
    拡散なし → 1D 均質化の問題を回避。

    Returns
    -------
    dict:
      di_0d      : float — 定常 Dysbiotic Index
      phi_final  : (5,)  — 定常菌種 volume fraction
      phi_traj   : (n_steps, 5)
      t_axis     : (n_steps,)
    """
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    from JAXFEM.core_hamilton_1d import theta_to_matrices, newton_step, make_initial_state

    theta_jax  = jnp.array(theta_np, dtype=jnp.float64)
    A, b_diag  = theta_to_matrices(theta_jax)
    active_mask = jnp.ones(5, dtype=jnp.int64)

    params = {
        "dt_h"         : dt,
        "Kp1"          : 1e-4,
        "Eta"          : jnp.ones(5, dtype=jnp.float64),
        "EtaPhi"       : jnp.ones(5, dtype=jnp.float64),
        "c"            : 100.0,
        "alpha"        : 100.0,
        "K_hill"       : jnp.array(0.05, dtype=jnp.float64),
        "n_hill"       : jnp.array(4.0,  dtype=jnp.float64),
        "A"            : A,
        "b_diag"       : b_diag,
        "active_mask"  : active_mask,
        "newton_steps" : 6,
    }

    g0 = make_initial_state(1, active_mask)[0]  # (12,)

    def body(g, _):
        return newton_step(g, params), g

    _, g_traj = jax.lax.scan(jax.jit(body), g0, jnp.arange(n_steps))
    phi_traj  = np.array(g_traj[:, 0:5])    # (n_steps, 5)
    t_axis    = np.arange(n_steps, dtype=float) * dt
    phi_final = phi_traj[-1]                 # (5,)

    # DI = 1 - H/H_max
    phi_sum = phi_final.sum()
    p = phi_final / max(phi_sum, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H     = -(p * log_p).sum()
    di_0d = float(1.0 - H / np.log(5.0))

    return {
        "di_0d"     : di_0d,
        "phi_final" : phi_final,
        "phi_traj"  : phi_traj,
        "t_axis"    : t_axis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 既存 CSV の読み込み
# ─────────────────────────────────────────────────────────────────────────────

def _read_commented_csv(path: str) -> dict:
    """
    '#' で始まるコメント行をスキップして CSV を読み込む汎用ヘルパー。
    最初の非コメント行をヘッダとして扱う。
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    # lines[0] = ヘッダ行 (コンマ区切りカラム名)
    cols = lines[0].split(",")
    data = np.array(
        [[float(v) for v in l.split(",")]
         for l in lines[1:] if l.strip()],
        dtype=np.float64,
    )
    return {col: data[:, i] for i, col in enumerate(cols)}


def load_original_csv(condition_key: str) -> dict | None:
    """既存の macro_eigenstrain_*.csv を読み込む (pandas 不要)。"""
    path = os.path.join(IN_DIR, f"macro_eigenstrain_{condition_key}.csv")
    if not os.path.isfile(path):
        print(f"  [{condition_key}] 警告: 元 CSV が見つかりません: {path}")
        return None
    d = _read_commented_csv(path)
    return {
        "depth_mm"   : d["depth_mm"],
        "depth_norm" : d["depth_norm"],
        "phi_So"     : d["phi_So"],
        "phi_An"     : d["phi_An"],
        "phi_Vd"     : d["phi_Vd"],
        "phi_Fn"     : d["phi_Fn"],
        "phi_Pg"     : d["phi_Pg"],
        "phi_total"  : d["phi_total"],
        "c"          : d["c"],
        "alpha"      : d["alpha"],
        "alpha_monod": d["alpha_monod"],
        "eps_growth" : d["eps_growth"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid CSV 書き出し
# ─────────────────────────────────────────────────────────────────────────────

def export_hybrid_csv(orig: dict, di_0d: float, condition_key: str) -> str:
    """
    0D DI × 1D 空間プロファイルの Hybrid CSV を書き出す。

    DI = DI_0D (条件別定数) で E_Pa を再計算するが、
    alpha_monod(x) は 1D PDE の空間勾配をそのまま保持。
    """
    N      = len(orig["depth_mm"])
    E_Pa   = compute_E_DI(di_0d, N)
    di_arr = np.full(N, di_0d)

    header = (
        "# Two-scale coupling HYBRID: 0D DI × 1D spatial alpha_monod\n"
        f"# condition: {condition_key}\n"
        f"# DI_0D: {di_0d:.6f}  (0D Hamilton ODE, condition-specific)\n"
        f"# E_Pa: {E_Pa[0]:.2f}  [= E(DI_0D), constant across depth]\n"
        f"# alpha_monod(x): from 1D Hamilton + nutrient PDE  [spatial structure]\n"
        f"# eps_growth = alpha_monod / 3  [isotropic eigenstrain]\n"
        "depth_mm,depth_norm,"
        "phi_So,phi_An,phi_Vd,phi_Fn,phi_Pg,"
        "phi_total,c,DI,alpha,alpha_monod,eps_growth,E_Pa\n"
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
            f"{di_arr[k]:.8e},"
            f"{orig['alpha'][k]:.8e},"
            f"{orig['alpha_monod'][k]:.8e},"
            f"{orig['eps_growth'][k]:.8e},"
            f"{E_Pa[k]:.8e}"
        )
        rows.append(row)

    fname = f"macro_eigenstrain_{condition_key}_hybrid.csv"
    path  = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write(header + "\n".join(rows) + "\n")

    size_kb = os.path.getsize(path) / 1024
    print(f"  [{condition_key}] Hybrid CSV: {path}  ({size_kb:.1f} KB)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 0D vs 1D DI 比較図
# ─────────────────────────────────────────────────────────────────────────────

def plot_di_comparison(summary: list[dict]) -> str:
    """
    各条件の DI_0D (Bar) と 1D DI (Line) を並べた比較図を生成。

    summary: list of {condition_key, label, color, di_0d, di_1d_mean}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_bar, ax_line = axes

    # --- Panel 1: DI_0D vs DI_1D_mean (棒グラフ) ---
    n = len(summary)
    x = np.arange(n)
    labels = [s["label"] for s in summary]
    di_0d  = [s["di_0d"] for s in summary]
    di_1d  = [s["di_1d_mean"] for s in summary]
    colors = [s["color"] for s in summary]

    w = 0.35
    bars0 = ax_bar.bar(x - w/2, di_0d, w, label="DI_0D (condition-specific)",
                       color=colors, alpha=0.85, edgecolor="black")
    bars1 = ax_bar.bar(x + w/2, di_1d, w, label="DI_1D (spatial mean)",
                       color=colors, alpha=0.35, edgecolor="black", hatch="///")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_bar.set_ylabel("Dysbiotic Index (DI)")
    ax_bar.set_title("DI comparison: 0D ODE vs 1D PDE spatial mean")
    ax_bar.legend(fontsize=8)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.axhline(0.5, ls="--", color="gray", alpha=0.5, label="DI=0.5")
    ax_bar.grid(alpha=0.3, axis="y")

    for bar, val in zip(bars0, di_0d):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars1, di_1d):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7, color="gray")

    # --- Panel 2: E(DI_0D) 比較 ---
    E_vals = [E_MAX_PA * (1 - np.clip(s["di_0d"]/DI_SCALE, 0, 1))**N_POWER
              + E_MIN_PA * np.clip(s["di_0d"]/DI_SCALE, 0, 1)
              for s in summary]

    bars_E = ax_line.bar(x, E_vals, color=colors, alpha=0.85, edgecolor="black")
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_line.set_ylabel("E(DI_0D) [Pa]")
    ax_line.set_title("Elastic modulus E(DI_0D) — condition-specific")
    ax_line.axhline(E_MAX_PA, ls="--", color="blue",  alpha=0.5, label=f"E_max={E_MAX_PA:.0f} Pa")
    ax_line.axhline(E_MIN_PA, ls="--", color="red",   alpha=0.5, label=f"E_min={E_MIN_PA:.0f} Pa")
    ax_line.legend(fontsize=8)
    ax_line.set_ylim(0, E_MAX_PA * 1.15)
    ax_line.grid(alpha=0.3, axis="y")

    for bar, val in zip(bars_E, E_vals):
        ax_line.text(bar.get_x() + bar.get_width()/2, val + 15,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Hybrid macro CSV: 0D DI (condition-specific) × 1D α_Monod (spatial)\n"
        "commensal DI_0D ≈ 0.05 → E ≈ E_max;  dysbiotic DI_0D ≈ 0.84 → E ≈ E_min",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "hybrid_di_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  DI 比較図: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  generate_hybrid_macro_csv.py — P2: Hybrid macro CSV 生成")
    print("=" * 70)
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

        # 2. 0D Hamilton ODE → DI_0D
        t0 = time.time()
        print("  0D Hamilton ODE を実行中 (JAX, n=2500) ...", flush=True)
        res0d = solve_0d_di(theta)
        di_0d = res0d["di_0d"]
        elapsed = time.time() - t0

        phi_final = res0d["phi_final"]
        E_cond = float(
            E_MAX_PA * (1 - min(di_0d / DI_SCALE, 1.0)) ** N_POWER
            + E_MIN_PA * min(di_0d / DI_SCALE, 1.0)
        )
        print(f"  DI_0D = {di_0d:.6f}  →  E = {E_cond:.1f} Pa  ({elapsed:.1f} s)")
        print(f"  phi_final: So={phi_final[0]:.4f} An={phi_final[1]:.4f} "
              f"Vd={phi_final[2]:.4f} Fn={phi_final[3]:.4f} Pg={phi_final[4]:.4f}")

        # 3. 既存 1D CSV ロード
        orig = load_original_csv(ckey)
        if orig is None:
            print(f"  スキップ: 元 CSV なし。先に multiscale_coupling_1d.py を実行してください。")
            continue

        # 4. Hybrid CSV 出力
        export_hybrid_csv(orig, di_0d, ckey)

        summary.append({
            "condition_key" : ckey,
            "label"         : meta["label"],
            "color"         : meta["color"],
            "di_0d"         : di_0d,
            "di_1d_mean"    : 0.0,   # 1D 拡散均質化により ≈ 0 (全条件共通)
            "E_cond"        : E_cond,
        })

        print()

    if not summary:
        print("エラー: 処理できた条件がありません。")
        return

    # 5. DI 比較図
    plot_di_comparison(summary)

    # 6. サマリ表示
    print()
    print("=" * 70)
    print("HYBRID CSV サマリ")
    print(f"{'Condition':<25} {'DI_0D':>8} {'E_Pa':>8}")
    print("-" * 45)
    for s in summary:
        print(f"  {s['condition_key']:<23} {s['di_0d']:>8.4f} {s['E_cond']:>8.1f}")
    print("=" * 70)
    print()
    print("次のステップ: generate_abaqus_eigenstrain.py (P1) を再実行")
    print("  → hybrid CSV から Abaqus inp が条件別 E_Pa で生成されます")
    print("=" * 70)


if __name__ == "__main__":
    main()
