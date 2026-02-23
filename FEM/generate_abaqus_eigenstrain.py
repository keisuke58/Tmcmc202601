#!/usr/bin/env python3
"""
generate_abaqus_eigenstrain.py — P1: Abaqus 深さ方向固有ひずみ入力生成
=======================================================================

macro_eigenstrain_{condition}_hybrid.csv (なければ原版) を読み込み、
Abaqus キーワード形式 .inp ファイルを生成する。

物理モデル (熱膨張アナロジー)
-----------------------------
  固有ひずみ = alpha_monod(depth) / 3   [等方体積ひずみ成分]
  実装方法: *EXPANSION TYPE=ISO, alpha_th = 1.0
    → *TEMPERATURE: node_id, ΔT = eps_growth(depth)
    → Abaqus が ε_th = alpha_th * ΔT = eps_growth を計算

Abaqus モデル構成
-----------------
  要素タイプ : T3D2  (2節点線形 3D トラス, 最小限の 1D 実証)
  方向       : z 軸 = バイオフィルム深さ  (node 1 = 歯面, node N = 唾液側)
  要素数     : N_NODES - 1 = 29 要素
  断面積     : 1.0 mm²  (仮想; 応力 σ = F/A で規格化)
  BC         : 歯面端 (node 1) を全自由度固定 → 完全拘束
               → 発生応力 σ₀ = -E * eps_growth  (compressive prestress)
  材料定数   : E と ν は CSV の E_Pa カラムの平均値を使用, ν = 0.45

出力
----
  FEM/_abaqus_input/
    biofilm_1d_bar_{condition}.inp       — Abaqus キーワード入力
    eigenstrain_field_{condition}.csv    — (node_id, depth_mm, ΔT=eps_growth)
    compare_conditions.png              — 4 条件 eps_growth(depth) 比較図
    sigma_max_summary.txt               — 最大圧縮応力サマリ

使い方
------
  # hybrid CSV 生成後 (P2 実行後) に実行:
  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \\
      Tmcmc202601/FEM/generate_abaqus_eigenstrain.py

  # Abaqus で実行 (Abaqus がインストールされている場合):
  abaqus job=biofilm_1d_bar_commensal_static input=biofilm_1d_bar_commensal_static.inp

参考
----
  Klempt et al. (2024) Biomech Model Mechanobiol 23:2091-2113
  Abaqus Analysis User's Guide, Sec. 6.1.1 (Thermal expansion)
"""

from __future__ import annotations
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ディレクトリ設定 ──────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
IN_DIR    = os.path.join(_HERE, "_multiscale_results")
OUT_DIR   = os.path.join(_HERE, "_abaqus_input")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 条件メタデータ (色・ラベル) ───────────────────────────────────────────────
CONDITIONS_META = {
    "commensal_static" : {"color": "#1f77b4", "label": "Commensal Static"},
    "commensal_hobic"  : {"color": "#2ca02c", "label": "Commensal HOBIC"},
    "dysbiotic_static" : {"color": "#ff7f0e", "label": "Dysbiotic Static"},
    "dysbiotic_hobic"  : {"color": "#d62728", "label": "Dysbiotic HOBIC"},
}

NU = 0.45          # ポアソン比 (バイオフィルム典型値)
ALPHA_TH = 1.0     # 熱膨張係数 (正規化: ΔT = eps_growth)


# ─────────────────────────────────────────────────────────────────────────────
# CSV 読み込み
# ─────────────────────────────────────────────────────────────────────────────

def _read_commented_csv(path: str) -> dict:
    """'#' コメント行をスキップして CSV をロードするヘルパー。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    cols = lines[0].split(",")
    data = np.array(
        [[float(v) for v in l.split(",")]
         for l in lines[1:] if l.strip()],
        dtype=np.float64,
    )
    return {col: data[:, i] for i, col in enumerate(cols)}


def load_csv(condition_key: str) -> dict | None:
    """hybrid 版を優先して CSV を読み込む (pandas 不要)。"""
    for suffix in ["_hybrid", ""]:
        path = os.path.join(IN_DIR, f"macro_eigenstrain_{condition_key}{suffix}.csv")
        if os.path.isfile(path):
            d = _read_commented_csv(path)
            tag = "(hybrid)" if suffix == "_hybrid" else "(original)"
            print(f"  [{condition_key}] CSV ロード {tag}: {path}")
            return {
                "depth_mm"   : d["depth_mm"],
                "depth_norm" : d["depth_norm"],
                "alpha_monod": d["alpha_monod"],
                "eps_growth" : d["eps_growth"],
                "DI"         : d["DI"],
                "E_Pa"       : d["E_Pa"],
                "phi_total"  : d["phi_total"],
                "c"          : d["c"],
                "path"       : path,
                "suffix"     : suffix,
            }
    print(f"  [{condition_key}] 警告: CSV が見つかりません")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Abaqus .inp 生成 (T3D2 バー要素)
# ─────────────────────────────────────────────────────────────────────────────

def generate_abaqus_inp(data: dict, condition_key: str, label: str) -> str:
    """
    深さ方向 1D バーモデルの Abaqus inp ファイルを生成する。

    T3D2 (2 節点 3D トラス) を z 軸方向に積み上げた構成。
    固有ひずみ = 熱膨張アナロジー: ΔT_node = eps_growth(depth)

    Parameters
    ----------
    data : dict — CSV から読み込んだフィールドデータ
    condition_key : str — 条件キー (ファイル名に使用)
    label : str — 条件ラベル (コメントに使用)

    Returns
    -------
    path : str — 出力 .inp ファイルのパス
    """
    depth_mm  = data["depth_mm"]
    eps_gr    = data["eps_growth"]
    E_Pa      = data["E_Pa"]
    N         = len(depth_mm)

    E_mean    = float(E_Pa.mean())
    E_tooth   = float(E_Pa[0])
    E_saliva  = float(E_Pa[-1])
    eps_max   = float(eps_gr.max())
    eps_min   = float(eps_gr.min())

    # 最大圧縮応力 (完全拘束時): σ₀ = -E * eps_growth
    sigma_max_compressive = -E_tooth * eps_gr[0]      # 歯面 (最大圧縮)
    sigma_min_compressive = -E_saliva * eps_gr[-1]    # 唾液側 (最大伸長)

    lines = []

    # ── ヘッダ ──────────────────────────────────────────────────────────────
    lines += [
        f"**",
        f"** {'='*60}",
        f"** ABAQUS INPUT FILE — BIOFILM 1D EIGENSTRAIN BAR",
        f"** Condition: {condition_key}  ({label})",
        f"** Two-scale coupling: micro Hamilton 1D → macro Abaqus",
        f"** Source CSV: {os.path.basename(data['path'])}",
        f"** Generated by: generate_abaqus_eigenstrain.py",
        f"**",
        f"** Physical model (thermal expansion analogy):",
        f"**   eps_growth(x) = alpha_monod(x) / 3  [isotropic eigenstrain]",
        f"**   alpha_th = {ALPHA_TH:.1f}  →  ΔT_node = eps_growth(depth)",
        f"**",
        f"** Key values:",
        f"**   E_mean = {E_mean:.1f} Pa,  nu = {NU:.2f}",
        f"**   eps_growth: [{eps_min:.6f}, {eps_max:.6f}]",
        f"**   sigma_compressive (tooth end): {sigma_max_compressive:.4f} Pa",
        f"**   sigma_compressive (saliva end): {sigma_min_compressive:.4f} Pa",
        f"** {'='*60}",
        f"**",
        f"*HEADING",
        f"Biofilm 1D eigenstrain bar  ({condition_key})",
        f"** SI units: Pa, mm, N",
        f"**",
    ]

    # ── 節点 (z 軸方向) ──────────────────────────────────────────────────────
    lines += ["**", "** ─── NODES ───", "*NODE"]
    for i, d in enumerate(depth_mm):
        # (x=0, y=0, z=depth_mm)
        lines.append(f"{i+1:5d},  0.000000,  0.000000,  {d:.8f}")

    # ── 要素 (T3D2) ──────────────────────────────────────────────────────────
    lines += [
        "**",
        "** ─── ELEMENTS (T3D2: 2-node linear 3D truss) ───",
        f"*ELEMENT, TYPE=T3D2, ELSET=BIOFILM_BAR",
    ]
    for i in range(N - 1):
        lines.append(f"{i+1:5d},  {i+1},  {i+2}")

    # ── 節点セット ────────────────────────────────────────────────────────────
    lines += [
        "**",
        "** ─── NODE SETS ───",
        "*NSET, NSET=TOOTH_SURFACE",
        "1",
        f"*NSET, NSET=SALIVA_SURFACE",
        f"{N}",
        f"*NSET, NSET=ALL_NODES, GENERATE",
        f"1, {N}, 1",
    ]

    # ── 材料定義 ──────────────────────────────────────────────────────────────
    mat_name = f"BIOFILM_{condition_key.upper()}"
    lines += [
        "**",
        f"** ─── MATERIAL: {mat_name} ───",
        f"*MATERIAL, NAME={mat_name}",
        "**",
        "** Elastic: E_mean [Pa], nu = 0.45",
        "*ELASTIC",
        f"{E_mean:.6f},  {NU:.4f}",
        "**",
        "** Thermal expansion analogy: alpha_th = 1.0",
        "** → strain = 1.0 * ΔT = eps_growth  (isotropic, ZERO=0.0)",
        "*EXPANSION, TYPE=ISO, ZERO=0.0",
        f"{ALPHA_TH:.4f},  0.0",
        "**",
    ]

    # ── 断面 (T3D2 は *SOLID SECTION 非対応 → *TRUSS SECTION) ────────────────
    lines += [
        "** ─── SECTION ───",
        f"*SOLID SECTION, ELSET=BIOFILM_BAR, MATERIAL={mat_name}",
        "1.0",   # 断面積 [mm²]
        "**",
    ]

    # ── 初期条件 (参照温度 = 0) ───────────────────────────────────────────────
    lines += [
        "** ─── INITIAL CONDITIONS ───",
        "** 参照温度 T_ref = 0  (ΔT = T_applied - T_ref = T_applied)",
        "*INITIAL CONDITIONS, TYPE=TEMPERATURE",
    ]
    for i in range(N):
        lines.append(f"{i+1:5d},  0.0")

    # ── 境界条件: 歯面端を全 DOF 固定 ────────────────────────────────────────
    lines += [
        "**",
        "** ─── BOUNDARY CONDITIONS ───",
        "** 歯面端 (node 1, z=0) を全自由度固定",
        "** → 固有ひずみが完全に阻害 → 最大圧縮応力を生成",
        "*BOUNDARY",
        "TOOTH_SURFACE, 1, 3, 0.0",
        "**",
    ]

    # ── ステップ: 固有ひずみ適用 ──────────────────────────────────────────────
    lines += [
        "** ─── STEP: Apply eigenstrain via temperature field ───",
        "*STEP, NLGEOM=NO, NAME=EIGENSTRAIN_STEP",
        "** 線形静解析  (固有ひずみは初期応力として)",
        "*STATIC",
        "**",
        "** 温度場: ΔT(node) = eps_growth(depth)",
        "** → 熱膨張ひずみ ε_th = 1.0 × ΔT = eps_growth",
        "** → 拘束端では応力 σ = -E × eps_growth  (圧縮)",
        "*TEMPERATURE",
    ]
    for i, eps in enumerate(eps_gr):
        lines.append(f"{i+1:5d},  {eps:.10f}")

    # ── 出力要求 ─────────────────────────────────────────────────────────────
    lines += [
        "**",
        "** ─── OUTPUT REQUESTS ───",
        "*OUTPUT, FIELD, FREQUENCY=1",
        "*NODE OUTPUT",
        "U, NT",
        "*ELEMENT OUTPUT",
        "S, E, EE, IE",
        "**",
        "*OUTPUT, HISTORY, FREQUENCY=1",
        "*NODE HISTORY, NSET=TOOTH_SURFACE",
        "U3",
        "*NODE HISTORY, NSET=SALIVA_SURFACE",
        "U3",
        "**",
        "*END STEP",
        "**",
        "** ─── END OF INPUT ───",
    ]

    # ── ファイル書き出し ───────────────────────────────────────────────────────
    fname = f"biofilm_1d_bar_{condition_key}.inp"
    path  = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [{condition_key}] Abaqus inp 出力: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 固有ひずみフィールド CSV (節点 ID + 深さ + ΔT)
# ─────────────────────────────────────────────────────────────────────────────

def export_field_csv(data: dict, condition_key: str) -> str:
    """
    (node_id, depth_mm, eps_growth, E_Pa, sigma_compressive) の CSV を書き出す。

    Abaqus 外部データ入力や後処理に使用。
    """
    depth_mm = data["depth_mm"]
    eps_gr   = data["eps_growth"]
    E_Pa     = data["E_Pa"]
    N        = len(depth_mm)

    sigma_compressive = -E_Pa * eps_gr   # 完全拘束時の圧縮応力 [Pa]

    header = (
        f"# Abaqus eigenstrain field — {condition_key}\n"
        f"# node_id: 1=tooth (z=0), {N}=saliva (z={depth_mm[-1]:.4f} mm)\n"
        f"# eps_growth = alpha_monod/3  [isotropic eigenstrain]\n"
        f"# sigma_compressive = -E_Pa * eps_growth  [fully constrained]\n"
        "node_id,depth_mm,eps_growth,DeltaT,E_Pa,sigma_compressive_Pa\n"
    )

    rows = []
    for i in range(N):
        rows.append(
            f"{i+1},{depth_mm[i]:.8f},{eps_gr[i]:.10f},"
            f"{eps_gr[i]:.10f},{E_Pa[i]:.4f},{sigma_compressive[i]:.6f}"
        )

    fname = f"eigenstrain_field_{condition_key}.csv"
    path  = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write(header + "\n".join(rows) + "\n")

    print(f"  [{condition_key}] 固有ひずみフィールド CSV: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 4 条件比較図
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(all_data: dict) -> str:
    """
    4 条件の eps_growth(depth) + E_Pa(depth) + σ(depth) を比較する 3 パネル図。
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax_eps, ax_E, ax_sigma = axes

    for ckey, data in all_data.items():
        meta  = CONDITIONS_META[ckey]
        col   = meta["color"]
        lbl   = meta["label"]
        depth = data["depth_mm"]
        eps   = data["eps_growth"]
        E     = data["E_Pa"]
        sigma = -E * eps   # 圧縮応力 (完全拘束時)

        suffix_tag = " [hybrid]" if data["suffix"] == "_hybrid" else ""

        ax_eps.plot(depth, eps,   color=col, lw=2, label=lbl + suffix_tag)
        ax_E.plot(  depth, E,     color=col, lw=2, label=lbl + suffix_tag)
        ax_sigma.plot(depth, sigma, color=col, lw=2, label=lbl + suffix_tag)

    # 体裁
    ax_eps.set_xlabel("Depth from tooth [mm]")
    ax_eps.set_ylabel(r"$\varepsilon_{growth}$ = $\alpha_{Monod}$ / 3")
    ax_eps.set_title("Isotropic eigenstrain field")
    ax_eps.legend(fontsize=8)
    ax_eps.grid(alpha=0.3)

    ax_E.set_xlabel("Depth from tooth [mm]")
    ax_E.set_ylabel("E [Pa]")
    ax_E.set_title("Local elastic modulus E(DI)")
    ax_E.legend(fontsize=8)
    ax_E.grid(alpha=0.3)

    ax_sigma.set_xlabel("Depth from tooth [mm]")
    ax_sigma.set_ylabel(r"$\sigma_0$ [Pa]  (compressive, < 0)")
    ax_sigma.set_title(r"Compressive prestress $\sigma_0 = -E \cdot \varepsilon_{growth}$")
    ax_sigma.legend(fontsize=8)
    ax_sigma.grid(alpha=0.3)

    fig.suptitle(
        "Two-scale coupling: micro Hamilton 1D → macro Abaqus\n"
        "Eigenstrain, Modulus, and Prestress by depth (biofilm layer 0→0.2 mm)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "compare_conditions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  比較図: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# サマリテキスト
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(all_data: dict) -> str:
    """最大圧縮応力サマリを txt に書き出す。"""
    lines = [
        "=" * 70,
        "BIOFILM EIGENSTRAIN SUMMARY  (Two-scale coupling result)",
        "=" * 70,
        f"{'Condition':<25} {'E_mean[Pa]':>10} {'eps_tooth':>10} {'eps_saliva':>10} {'σ0_tooth[Pa]':>13}",
        "-" * 70,
    ]
    for ckey, data in all_data.items():
        E_mean   = data["E_Pa"].mean()
        eps_t    = data["eps_growth"][0]
        eps_s    = data["eps_growth"][-1]
        sigma_t  = -data["E_Pa"][0] * eps_t
        lines.append(
            f"{ckey:<25} {E_mean:>10.1f} {eps_t:>10.6f} {eps_s:>10.4f} {sigma_t:>13.4f}"
        )
    lines += [
        "-" * 70,
        "",
        "Note: sigma_compressive = -E * eps_growth  (fully constrained bar)",
        "      eps_growth = alpha_monod / 3  (nutrient-limited isotropic eigenstrain)",
        "      For hybrid CSV: E uses condition-specific DI_0D from 0D Hamilton ODE.",
        "=" * 70,
    ]

    path = os.path.join(OUT_DIR, "sigma_max_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  サマリ: {path}")
    print()
    print("\n".join(lines))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  generate_abaqus_eigenstrain.py — P1: Abaqus 固有ひずみ入力生成")
    print("=" * 70)
    print(f"  入力ディレクトリ: {IN_DIR}")
    print(f"  出力ディレクトリ: {OUT_DIR}")
    print()

    all_data = {}
    for ckey, meta in CONDITIONS_META.items():
        print(f"[{ckey}]")
        data = load_csv(ckey)
        if data is None:
            continue

        generate_abaqus_inp(data, ckey, meta["label"])
        export_field_csv(data, ckey)
        all_data[ckey] = data
        print()

    if not all_data:
        print("エラー: CSV ファイルが見つかりませんでした。")
        print("先に generate_hybrid_macro_csv.py (P2) または multiscale_coupling_1d.py を実行してください。")
        return

    plot_comparison(all_data)
    write_summary(all_data)

    print()
    print("=" * 70)
    print("完了。出力ファイル:")
    for ckey in all_data:
        print(f"  _abaqus_input/biofilm_1d_bar_{ckey}.inp")
        print(f"  _abaqus_input/eigenstrain_field_{ckey}.csv")
    print("  _abaqus_input/compare_conditions.png")
    print("  _abaqus_input/sigma_max_summary.txt")
    print()
    print("次のステップ:")
    print("  1. generate_hybrid_macro_csv.py (P2) を実行して hybrid CSV を生成")
    print("  2. 本スクリプトを再実行 → hybrid 版の E_Pa で σ₀ が条件別に分化")
    print("  3. Abaqus で .inp を実行: abaqus job=biofilm_1d_bar_{condition}")
    print("=" * 70)


if __name__ == "__main__":
    main()
