#!/usr/bin/env python3
"""
JAXFEM/extract_alpha_field_1d.py
=================================
Option D Step 1 — Hamilton 1D + 栄養場 c(x,t) を連成させ、
空間分布した α(x)（成長誘起固有ひずみ場）を抽出して CSV/図として出力する。

物理モデル
----------
  [Hamilton PDE]  φᵢ(x,t) の反応+拡散 (局所 c(x) を使用)
  [栄養 PDE]      ∂c/∂t = D_c Δc - g_eff φ_total c/(k+c)
                  BC: c(x=L)=1.0 (歯肉溝液), ∂c/∂x|_{x=0}=0 (歯面)
  [成長膨張]      α(x) = k_α ∫ φ_total(x,t) dt
                  ε_growth(x) = α(x) / 3

出力
----
  alpha_field_1d.csv  : x_norm, x_mm, c_final, phi_total_final, alpha_x, eps_growth_x
  alpha_field_1d.png  : 4 パネル診断プロット
    panel 1: φᵢ(x) species profiles at t_final
    panel 2: c(x,t) 栄養場の時間発展
    panel 3: φ_total(x,t) の時間発展
    panel 4: α(x) / ε_growth(x) フィールド

使い方
------
  # FEM ディレクトリから直接実行:
  python3 JAXFEM/extract_alpha_field_1d.py

  # TMCMC MAP theta を渡す場合:
  python3 JAXFEM/extract_alpha_field_1d.py --theta-json /path/to/theta_MAP.json

  # 栄養パラメータを変更:
  python3 JAXFEM/extract_alpha_field_1d.py --D-c 1.0 --g-eff 50.0 --k-monod 1.0

  # 出力先・解像度を変更:
  python3 JAXFEM/extract_alpha_field_1d.py --N 60 --n-macro 120 --out-dir _results/
"""

from __future__ import print_function, division
import os
import sys
import json
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FEM_DIR = os.path.dirname(_HERE)
if _FEM_DIR not in sys.path:
    sys.path.insert(0, _FEM_DIR)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from JAXFEM.core_hamilton_1d import THETA_DEMO
from JAXFEM.core_hamilton_1d_nutrient import simulate_hamilton_1d_nutrient

SPECIES_NAMES = [
    "S. oralis",
    "A. naeslundii",
    "V. dispar",
    "F. nucleatum",
    "P. gingivalis",
]
SPECIES_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# ---------------------------------------------------------------------------
# シミュレーション実行 & α(x) 抽出
# ---------------------------------------------------------------------------

def run_and_extract(
    theta,
    D_eff,
    D_c=1.0,
    g_eff=50.0,
    k_monod=1.0,
    k_alpha=0.05,
    N=30,
    L=1.0,
    n_macro=60,
    n_react_sub=20,
    n_sub_c=10,
    dt_h=1e-5,
    biofilm_thickness_mm=0.2,
):
    """
    Hamilton 1D + 栄養場シミュレーションを実行し、α(x) フィールドを返す。

    Returns
    -------
    dict with keys:
        x_norm, x_mm         : (N,)        座標
        phi_all              : (T+1, N, 5) 各種 φᵢ(x,t)
        phi_total            : (T+1, N)    φ_total(x,t)
        c_all                : (T+1, N)    栄養場 c(x,t)
        alpha_x, eps_x       : (N,)        α(x), ε_growth(x)
        t_arr                : (T+1,)      時刻
        k_alpha, D_c, g_eff  : float       パラメータ記録
    """
    print("=" * 58)
    print("  Hamilton 1D + Nutrient c(x,t) simulation")
    print(f"  N={N}  n_macro={n_macro}  n_react_sub={n_react_sub}  n_sub_c={n_sub_c}")
    print(f"  D_c={D_c}  g_eff={g_eff}  k_monod={k_monod}  k_alpha={k_alpha}")
    # Thiele 数の目安: Φ = sqrt(g_eff * L² / (D_c * k_monod))
    thiele = (g_eff / (D_c * k_monod)) ** 0.5 * L
    t_end_est = dt_h * n_react_sub * n_macro
    print(f"  dt_h={dt_h:.1e}  T*_end≈{t_end_est:.4f}")
    print(f"  Thiele number Φ ≈ {thiele:.2f}  ({'diffusion-limited' if thiele > 3 else 'reaction-limited'})")
    print("=" * 58)

    G_all, c_all = simulate_hamilton_1d_nutrient(
        theta=theta,
        D_eff=D_eff,
        D_c=D_c,
        g_eff=g_eff,
        k_monod=k_monod,
        k_alpha=k_alpha,
        n_macro=n_macro,
        n_react_sub=n_react_sub,
        n_sub_c=n_sub_c,
        N=N,
        L=L,
        dt_h=dt_h,
    )

    G_np = np.array(G_all)
    c_np = np.array(c_all)

    x_norm = np.linspace(0.0, L, N)
    x_mm = x_norm * biofilm_thickness_mm
    dt_macro = dt_h * n_react_sub
    t_arr = np.arange(n_macro + 1) * dt_macro

    phi_all = G_np[:, :, 0:5]       # (T+1, N, 5)
    phi_total = phi_all.sum(axis=2)  # (T+1, N)

    # Monod 重み: 栄養制限を成長膨張に反映
    #   monod(x,t) = c(x,t) / (k_monod + c(x,t))
    # → 歯面側 (c≈0) は成長せず α≈0、外側 (c=1) は最大成長
    monod = c_np / (k_monod + c_np + 1e-12)           # (T+1, N) ∈ [0, 0.5]
    phi_weighted = phi_total * monod                   # Monod 重み付き積分対象

    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    # α_unweighted: 従来定義（参考値）
    alpha_x_raw = k_alpha * _trapz(phi_total, t_arr, axis=0)
    # α_monod: Monod 重み付き（物理的に正しい）
    alpha_x = k_alpha * _trapz(phi_weighted, t_arr, axis=0)  # (N,)
    eps_x = alpha_x / 3.0

    # --- サマリー ---
    c_f = c_np[-1]
    phi_f = phi_total[-1]
    print(f"\n  t_end = {t_arr[-1]:.5f} [Hamilton T*]")
    print(f"  c(x) at t_final  : mean={c_f.mean():.4f}  "
          f"x=0端: {c_f[0]:.4f}  x=L端: {c_f[-1]:.4f}")
    print(f"  φ_total at t_final: mean={phi_f.mean():.4f}  "
          f"max={phi_f.max():.4f}  min={phi_f.min():.4f}")
    print(f"  α(x)              : mean={alpha_x.mean():.5f}  "
          f"max={alpha_x.max():.5f}  min={alpha_x.min():.5f}")
    print(f"  ε_growth(x)=α/3   : mean={eps_x.mean():.5f}  "
          f"max={eps_x.max():.5f}  min={eps_x.min():.5f}")
    ratio = alpha_x.max() / (alpha_x.mean() + 1e-12)
    ratio_raw = alpha_x_raw.max() / (alpha_x_raw.mean() + 1e-12)
    print(f"  α_raw(x) 空間変動  : max/mean = {ratio_raw:.4f}  (Monod なし)")
    print(f"  α(x)  空間変動     : max/mean = {ratio:.4f}  (Monod 重み付き)")
    print(f"  α(x)  x=0端       : {alpha_x[0]:.5f}  (歯面 — 栄養乏しい)")
    print(f"  α(x)  x=L端       : {alpha_x[-1]:.5f}  (外側 — 栄養豊富)")

    print("\n  φᵢ(x) 空間平均 at t_final:")
    for name, val in zip(SPECIES_NAMES, phi_all[-1].mean(axis=0)):
        print(f"    {name:20s}: {val:.4f}")

    return dict(
        x_norm=x_norm,
        x_mm=x_mm,
        phi_all=phi_all,
        phi_total=phi_total,
        c_all=c_np,
        alpha_x=alpha_x,
        eps_x=eps_x,
        t_arr=t_arr,
        k_alpha=k_alpha,
        D_c=D_c,
        g_eff=g_eff,
        k_monod=k_monod,
    )


# ---------------------------------------------------------------------------
# 出力: CSV
# ---------------------------------------------------------------------------

def save_csv(out_path, res):
    data = np.column_stack([
        res["x_norm"],
        res["x_mm"],
        res["c_all"][-1],
        res["phi_total"][-1],
        res["alpha_x"],
        res["eps_x"],
    ])
    header = "x_norm,x_mm,c_final,phi_total_final,alpha_x_monod,eps_growth_x_monod"
    np.savetxt(out_path, data, delimiter=",", header=header, comments="")
    print(f"\n  CSV saved: {out_path}")


# ---------------------------------------------------------------------------
# 出力: 4 パネル診断プロット
# ---------------------------------------------------------------------------

def make_plot(out_path, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib 未インストール、プロットをスキップ)")
        return

    x = res["x_norm"]
    phi_all = res["phi_all"]
    phi_total = res["phi_total"]
    c_all = res["c_all"]
    alpha_x = res["alpha_x"]
    eps_x = res["eps_x"]
    t_arr = res["t_arr"]
    T = len(t_arr) - 1

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    snap_idx = np.linspace(0, T, 5, dtype=int)
    cmap = plt.cm.plasma

    # --- Panel 1: φᵢ(x) at t_final ---
    ax = axes[0]
    for i, (name, col) in enumerate(zip(SPECIES_NAMES, SPECIES_COLORS)):
        ax.plot(x, phi_all[-1, :, i], color=col, lw=1.8, label=name)
    ax.set_xlabel("x (0=tooth, 1=fluid)")
    ax.set_ylabel(r"$\varphi_i$")
    ax.set_title("Species φᵢ(x)\nt = t_final")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    # --- Panel 2: c(x,t) スナップショット ---
    ax = axes[1]
    for k, idx in enumerate(snap_idx):
        col = cmap(k / max(len(snap_idx) - 1, 1))
        ax.plot(x, c_all[idx], color=col, lw=1.4,
                label=f"t={t_arr[idx]:.4f}")
    ax.set_xlabel("x (0=tooth, 1=fluid)")
    ax.set_ylabel("c (nutrient)")
    ax.set_title("Nutrient c(x,t)\n(BC: c(1)=1, dc/dx|₀=0)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

    # --- Panel 3: φ_total(x,t) スナップショット ---
    ax = axes[2]
    for k, idx in enumerate(snap_idx):
        col = cmap(k / max(len(snap_idx) - 1, 1))
        ax.plot(x, phi_total[idx], color=col, lw=1.4,
                label=f"t={t_arr[idx]:.4f}")
    ax.set_xlabel("x (0=tooth, 1=fluid)")
    ax.set_ylabel(r"$\varphi_{total}$")
    ax.set_title("Total biofilm φ_total(x,t)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    # --- Panel 4: α(x) と ε(x) ---
    ax = axes[3]
    ax.fill_between(x, alpha_x, alpha=0.15, color="tab:red")
    ax.plot(x, alpha_x, color="tab:red", lw=2.2,
            label=rf"$\alpha(x)$ [mean={alpha_x.mean():.4f}]")
    ax2 = ax.twinx()
    ax2.plot(x, eps_x * 100, color="tab:orange", lw=1.6, ls="--",
             label=f"eps_growth [%] [mean={eps_x.mean()*100:.3f}%]")
    ax.set_xlabel("x (0=tooth, 1=fluid)")
    ax.set_ylabel(r"$\alpha(x)$", color="tab:red")
    ax2.set_ylabel(r"$\varepsilon_{growth}$ [%]", color="tab:orange")
    ax.set_title(
        r"Growth eigenstrain $\alpha(x)$"
        + f"\nk_alpha={res['k_alpha']}, g_eff={res['g_eff']}"
    )
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7, loc="best")

    fig.suptitle(
        "Hamilton 1D + Nutrient c(x,t) → Growth eigenstrain α(x)  [Option D Step 1]",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hamilton 1D + nutrient → α(x) eigenstrain field (Option D Step 1)"
    )
    p.add_argument("--theta-json", type=str, default=None,
                   help="TMCMC theta_MAP.json (省略時は THETA_DEMO)")
    p.add_argument("--D-c",    type=float, default=1.0,
                   help="栄養拡散係数 D_c (default: 1.0)")
    p.add_argument("--g-eff",  type=float, default=50.0,
                   help="栄養消費係数 g_eff (default: 50.0, Thiele~7)")
    p.add_argument("--k-monod", type=float, default=1.0,
                   help="Monod 半飽和定数 (default: 1.0)")
    p.add_argument("--k-alpha", type=float, default=0.05,
                   help="成長-固有ひずみ結合 k_α (default: 0.05)")
    p.add_argument("--N",        type=int,   default=30,
                   help="空間ノード数 (default: 30)")
    p.add_argument("--n-macro",  type=int,   default=60,
                   help="マクロタイムステップ数 (default: 60)")
    p.add_argument("--n-react-sub", type=int, default=20,
                   help="Hamilton 反応サブステップ数 (default: 20)")
    p.add_argument("--n-sub-c",  type=int,   default=10,
                   help="栄養 PDE サブステップ数 (default: 10)")
    p.add_argument("--dt-h",     type=float, default=1e-5,
                   help="Hamilton タイムステップ (default: 1e-5)")
    p.add_argument("--biofilm-thickness-mm", type=float, default=0.2,
                   help="バイオフィルム厚み [mm] (default: 0.2)")
    p.add_argument("--out-dir",  type=str,   default=".",
                   help="出力ディレクトリ (default: .)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.theta_json:
        with open(args.theta_json) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            theta_list = raw
        elif isinstance(raw, dict):
            theta_list = raw.get(
                "theta_full", raw.get("theta_sub", raw.get("theta"))
            )
            if theta_list is None:
                raise ValueError("theta_MAP.json に theta キーが見つかりません")
        theta = jnp.array(theta_list, dtype=jnp.float64)
        print(f"Loaded theta from: {args.theta_json}")
    else:
        theta = THETA_DEMO
        print("Using THETA_DEMO (built-in)")

    print(f"theta[:5] = {np.array(theta[:5]).round(4)}")

    D_eff = jnp.array([0.001, 0.001, 0.0008, 0.0005, 0.0002])

    res = run_and_extract(
        theta=theta,
        D_eff=D_eff,
        D_c=args.D_c,
        g_eff=args.g_eff,
        k_monod=args.k_monod,
        k_alpha=args.k_alpha,
        N=args.N,
        n_macro=args.n_macro,
        n_react_sub=args.n_react_sub,
        n_sub_c=args.n_sub_c,
        dt_h=args.dt_h,
        biofilm_thickness_mm=args.biofilm_thickness_mm,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "alpha_field_1d.csv")
    png_path = os.path.join(args.out_dir, "alpha_field_1d.png")

    save_csv(csv_path, res)
    make_plot(png_path, res)

    print("\n" + "=" * 58)
    print("  Next steps:")
    print("  Step 2: alpha_field_1d.csv → Abaqus 材料点マッピング")
    print("          biofilm_conformal_tet.py --growth-eigenstrain-field")
    print("  Step 3: core_hamilton_2d.py へ拡張 (歯周ポケット断面)")
    print("=" * 58)


if __name__ == "__main__":
    main()
