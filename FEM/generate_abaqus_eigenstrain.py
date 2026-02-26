#!/usr/bin/env python3
"""
generate_abaqus_eigenstrain.py — P1: Abaqus 深さ方向固有ひずみ入力生成
=======================================================================

macro_eigenstrain_{condition}_hybrid.csv を読み込み、
3つの E モデル (DI, φ_Pg, Virulence) ごとに Abaqus .inp を生成する。

物理モデル (熱膨張アナロジー)
-----------------------------
  固有ひずみ = alpha_monod(depth) / 3   [等方体積ひずみ成分]
  *EXPANSION TYPE=ISO, alpha_th = 1.0
  *TEMPERATURE: node_id, ΔT = eps_growth(depth)

3つの E モデル
--------------
  1. DI model:        E(DI_0D)     — entropy-based
  2. φ_Pg model:      E(φ_Pg_0D)  — Pg-specific Hill sigmoid
  3. Virulence model:  E(V_0D)    — Pg + Fn weighted

出力
----
  FEM/_abaqus_input/
    biofilm_1d_bar_{condition}_{model}.inp      — 条件×モデル Abaqus inp
    eigenstrain_field_{condition}.csv            — (node_id, depth_mm, ΔT)
    compare_conditions_3model.png               — 3モデル比較図
    sigma_max_summary_3model.txt                — サマリ

使い方
------
  ~/.pyenv/versions/miniconda3-latest/envs/klempt_fem/bin/python \
      Tmcmc202601/FEM/generate_abaqus_eigenstrain.py
"""

from __future__ import annotations
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ディレクトリ設定 ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(_HERE, "_multiscale_results")
OUT_DIR = os.path.join(_HERE, "_abaqus_input")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 条件メタデータ ───────────────────────────────────────────────────────────
CONDITIONS_META = {
    "commensal_static": {"color": "#1f77b4", "label": "Commensal Static"},
    "commensal_hobic": {"color": "#2ca02c", "label": "Commensal HOBIC"},
    "dysbiotic_static": {"color": "#ff7f0e", "label": "Dysbiotic Static"},
    "dysbiotic_hobic": {"color": "#d62728", "label": "Dysbiotic HOBIC"},
}

# E モデル定義
E_MODELS = {
    "di": {"col": "E_di", "label": "DI model", "desc": "entropy-based Dysbiotic Index"},
    "phi_pg": {
        "col": "E_phi_pg",
        "label": r"$\varphi_{Pg}$ model",
        "desc": "Pg-specific Hill sigmoid",
    },
    "virulence": {
        "col": "E_virulence",
        "label": "Virulence model",
        "desc": "Pg+Fn weighted Hill sigmoid",
    },
}

NU = 0.45  # ポアソン比
ALPHA_TH = 1.0  # 熱膨張係数


# ─────────────────────────────────────────────────────────────────────────────
# CSV 読み込み
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


def load_csv(condition_key: str) -> dict | None:
    """hybrid 版を優先して CSV を読み込む (3 E モデル対応)。"""
    for suffix in ["_hybrid", ""]:
        path = os.path.join(IN_DIR, f"macro_eigenstrain_{condition_key}{suffix}.csv")
        if os.path.isfile(path):
            d = _read_commented_csv(path)
            tag = "(hybrid)" if suffix == "_hybrid" else "(original)"
            print(f"  [{condition_key}] CSV ロード {tag}: {os.path.basename(path)}")

            result = {
                "depth_mm": d["depth_mm"],
                "depth_norm": d["depth_norm"],
                "alpha_monod": d["alpha_monod"],
                "eps_growth": d["eps_growth"],
                "phi_total": d["phi_total"],
                "c": d["c"],
                "path": path,
                "suffix": suffix,
            }

            # 3 E モデルのカラムを読み込む (存在すれば)
            for model_key, model_info in E_MODELS.items():
                col = model_info["col"]
                if col in d:
                    result[col] = d[col]
                else:
                    # 後方互換: E_Pa のみの古い CSV
                    if col == "E_di" and "E_Pa" in d:
                        result[col] = d["E_Pa"]
                    elif col == "E_di" and "DI" in d:
                        result[col] = d.get("E_Pa", np.full(len(d["depth_mm"]), 500.0))

            # DI カラム (存在すれば)
            if "DI" in d:
                result["DI"] = d["DI"]

            return result

    print(f"  [{condition_key}] 警告: CSV が見つかりません")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Abaqus .inp 生成
# ─────────────────────────────────────────────────────────────────────────────


def generate_abaqus_inp(
    data: dict,
    condition_key: str,
    label: str,
    model_key: str,
    E_Pa_arr: np.ndarray,
) -> str:
    """
    T3D2 バーモデルの Abaqus inp を生成。

    Parameters
    ----------
    data : dict — CSV フィールドデータ
    condition_key : str — 条件キー
    label : str — 条件ラベル
    model_key : str — E モデル ("di", "phi_pg", "virulence")
    E_Pa_arr : (N,) — E 値配列

    Returns
    -------
    path : str — 出力 .inp ファイルのパス
    """
    depth_mm = data["depth_mm"]
    eps_gr = data["eps_growth"]
    N = len(depth_mm)

    E_mean = float(E_Pa_arr.mean())
    eps_max = float(eps_gr.max())
    eps_min = float(eps_gr.min())

    sigma_tooth = -E_Pa_arr[0] * eps_gr[0]
    sigma_saliva = -E_Pa_arr[-1] * eps_gr[-1]

    model_desc = E_MODELS[model_key]["desc"]

    lines = []

    # ── ヘッダ ──
    lines += [
        "**",
        f"** {'='*60}",
        "** BIOFILM 1D EIGENSTRAIN BAR",
        f"** Condition: {condition_key}  ({label})",
        f"** E model: {model_key}  ({model_desc})",
        f"** E_mean = {E_mean:.1f} Pa,  nu = {NU:.2f}",
        f"** eps_growth: [{eps_min:.6f}, {eps_max:.6f}]",
        f"** sigma_tooth: {sigma_tooth:.4f} Pa",
        f"** sigma_saliva: {sigma_saliva:.4f} Pa",
        f"** {'='*60}",
        "*HEADING",
        f"Biofilm 1D eigenstrain ({condition_key}, E_model={model_key})",
        "**",
    ]

    # ── 節点 (z 軸方向) ──
    lines += ["*NODE"]
    for i, d in enumerate(depth_mm):
        lines.append(f"{i+1:5d},  0.000000,  0.000000,  {d:.8f}")

    # ── 要素 (T3D2) ──
    lines += ["*ELEMENT, TYPE=T3D2, ELSET=BIOFILM_BAR"]
    for i in range(N - 1):
        lines.append(f"{i+1:5d},  {i+1},  {i+2}")

    # ── 節点セット ──
    lines += [
        "*NSET, NSET=TOOTH_SURFACE",
        "1",
        "*NSET, NSET=SALIVA_SURFACE",
        f"{N}",
        "*NSET, NSET=ALL_NODES, GENERATE",
        f"1, {N}, 1",
    ]

    # ── 材料定義 ──
    mat_name = f"BIOFILM_{condition_key.upper()}_{model_key.upper()}"
    lines += [
        f"*MATERIAL, NAME={mat_name}",
        "*ELASTIC",
        f"{E_mean:.6f},  {NU:.4f}",
        "*EXPANSION, TYPE=ISO, ZERO=0.0",
        f"{ALPHA_TH:.4f},  0.0",
    ]

    # ── 断面 ──
    lines += [
        f"*SOLID SECTION, ELSET=BIOFILM_BAR, MATERIAL={mat_name}",
        "1.0",
    ]

    # ── 初期温度 ──
    lines += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE"]
    for i in range(N):
        lines.append(f"{i+1:5d},  0.0")

    # ── 境界条件 ──
    lines += [
        "*BOUNDARY",
        "TOOTH_SURFACE, 1, 3, 0.0",
    ]

    # ── ステップ ──
    lines += [
        "*STEP, NLGEOM=NO, NAME=EIGENSTRAIN_STEP",
        "*STATIC",
        "*TEMPERATURE",
    ]
    for i, eps in enumerate(eps_gr):
        lines.append(f"{i+1:5d},  {eps:.10f}")

    # ── 出力要求 ──
    lines += [
        "*OUTPUT, FIELD, FREQUENCY=1",
        "*NODE OUTPUT",
        "U, NT",
        "*ELEMENT OUTPUT",
        "S, E, EE, IE",
        "*OUTPUT, HISTORY, FREQUENCY=1",
        "*NODE HISTORY, NSET=TOOTH_SURFACE",
        "U3",
        "*NODE HISTORY, NSET=SALIVA_SURFACE",
        "U3",
        "*END STEP",
    ]

    fname = f"biofilm_1d_bar_{condition_key}_{model_key}.inp"
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"    [{model_key}] inp: {fname}  (E={E_mean:.0f} Pa)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 固有ひずみフィールド CSV
# ─────────────────────────────────────────────────────────────────────────────


def export_field_csv(data: dict, condition_key: str) -> str:
    """3 E モデルの応力を含む CSV を書き出す。"""
    depth_mm = data["depth_mm"]
    eps_gr = data["eps_growth"]
    N = len(depth_mm)

    header_parts = [
        f"# Abaqus eigenstrain field — {condition_key}",
        "# 3 E models: DI, phi_Pg, Virulence",
        "node_id,depth_mm,eps_growth,DeltaT",
    ]

    E_cols = {}
    for mk, mi in E_MODELS.items():
        col = mi["col"]
        if col in data:
            E_cols[mk] = data[col]
            header_parts[-1] += f",E_{mk},sigma_{mk}"

    header = "\n".join(header_parts[:2]) + "\n" + header_parts[2] + "\n"

    rows = []
    for i in range(N):
        row = f"{i+1},{depth_mm[i]:.8f},{eps_gr[i]:.10f},{eps_gr[i]:.10f}"
        for mk in E_cols:
            E_val = E_cols[mk][i]
            sigma = -E_val * eps_gr[i]
            row += f",{E_val:.4f},{sigma:.6f}"
        rows.append(row)

    fname = f"eigenstrain_field_{condition_key}.csv"
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write(header + "\n".join(rows) + "\n")

    print(f"  [{condition_key}] CSV: {fname}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3モデル比較図
# ─────────────────────────────────────────────────────────────────────────────


def plot_comparison_3model(all_data: dict) -> str:
    """
    3モデル × 4条件 の応力プロファイル比較図 (3×2 panel)。

    Row 1: eps_growth(depth), E(depth) for DI, E(depth) for phi_Pg
    Row 2: σ_DI(depth), σ_phi_pg(depth), σ_virulence(depth)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    model_keys = ["di", "phi_pg", "virulence"]
    model_labels = ["DI model", r"$\varphi_{Pg}$ model", "Virulence model"]

    for ckey, data in all_data.items():
        meta = CONDITIONS_META[ckey]
        col = meta["color"]
        lbl = meta["label"]
        depth = data["depth_mm"]
        eps = data["eps_growth"]
        suffix_tag = " [H]" if data["suffix"] == "_hybrid" else ""

        # Row 1, Col 1: eps_growth
        axes[0, 0].plot(depth, eps, color=col, lw=2, label=lbl + suffix_tag)

        # Row 1, Col 2-3: E for each model
        for j, mk in enumerate(model_keys[:2]):
            ecol = E_MODELS[mk]["col"]
            if ecol in data:
                axes[0, j + 1].plot(depth, data[ecol], color=col, lw=2, label=lbl)

        # Row 2: σ for each model
        for j, mk in enumerate(model_keys):
            ecol = E_MODELS[mk]["col"]
            if ecol in data:
                sigma = -data[ecol] * eps
                axes[1, j].plot(depth, sigma, color=col, lw=2, label=lbl)

    # 体裁
    axes[0, 0].set_ylabel(r"$\varepsilon_{growth}$")
    axes[0, 0].set_title("(a) Eigenstrain field")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(alpha=0.3)

    for j, (mk, ml) in enumerate(zip(model_keys[:2], model_labels[:2])):
        axes[0, j + 1].set_ylabel("E [Pa]")
        axes[0, j + 1].set_title(f"({'bc'[j]}) E: {ml}")
        axes[0, j + 1].legend(fontsize=7)
        axes[0, j + 1].grid(alpha=0.3)
        axes[0, j + 1].set_ylim(bottom=0)

    for j, (mk, ml) in enumerate(zip(model_keys, model_labels)):
        axes[1, j].set_xlabel("Depth [mm]")
        axes[1, j].set_ylabel(r"$\sigma_0$ [Pa]")
        axes[1, j].set_title(f"({'def'[j]}) Stress: {ml}")
        axes[1, j].legend(fontsize=7)
        axes[1, j].grid(alpha=0.3)

    for ax in axes.ravel():
        ax.set_xlabel("Depth from tooth [mm]")

    fig.suptitle(
        "3-Model Comparison: Eigenstrain, Modulus, and Prestress\n"
        "DI (entropy-based) vs $\\varphi_{Pg}$ (mechanism) vs Virulence (Pg+Fn)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "compare_conditions_3model.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  3モデル比較図: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# サマリテキスト
# ─────────────────────────────────────────────────────────────────────────────


def write_summary_3model(all_data: dict) -> str:
    """3モデルの最大圧縮応力サマリ。"""
    lines = [
        "=" * 90,
        "BIOFILM EIGENSTRAIN SUMMARY — 3 E MODELS",
        "=" * 90,
        f"{'Condition':<25} {'eps_tooth':>10} {'eps_sal':>8} "
        + "".join(f" {'E_'+mk:>8} {'σ_'+mk:>9}" for mk in ["di", "phi_pg", "vir"]),
        "-" * 90,
    ]
    for ckey, data in all_data.items():
        eps_t = data["eps_growth"][0]
        eps_s = data["eps_growth"][-1]
        parts = f"{ckey:<25} {eps_t:>10.6f} {eps_s:>8.4f}"
        for mk_full, mk_short in [("di", "di"), ("phi_pg", "phi_pg"), ("virulence", "vir")]:
            ecol = E_MODELS[mk_full]["col"]
            if ecol in data:
                E_mean = data[ecol].mean()
                sigma_t = -data[ecol][0] * eps_t
                parts += f" {E_mean:>8.1f} {sigma_t:>9.4f}"
            else:
                parts += f" {'N/A':>8} {'N/A':>9}"
        lines.append(parts)

    lines += [
        "-" * 90,
        "",
        "σ_tooth = -E * eps_growth[0]  (fully constrained bar, tooth end)",
        "DI model:   E(DI_0D),  species-blind entropy",
        "φ_Pg model: E(φ_Pg_0D), Pg-specific Hill sigmoid (mechanism-based)",
        "Virulence:  E(V_0D),   Pg + 0.3*Fn weighted Hill sigmoid",
        "=" * 90,
    ]

    path = os.path.join(OUT_DIR, "sigma_max_summary_3model.txt")
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
    print("  generate_abaqus_eigenstrain.py — 3-Model Abaqus INP 生成")
    print("=" * 70)
    print(f"  入力: {IN_DIR}")
    print(f"  出力: {OUT_DIR}")
    print()

    all_data = {}
    inp_count = 0

    for ckey, meta in CONDITIONS_META.items():
        print(f"[{ckey}]")
        data = load_csv(ckey)
        if data is None:
            continue

        # 各 E モデルで INP 生成
        for model_key, model_info in E_MODELS.items():
            col = model_info["col"]
            if col in data:
                E_Pa = data[col]
                generate_abaqus_inp(data, ckey, meta["label"], model_key, E_Pa)
                inp_count += 1
            else:
                print(f"    [{model_key}] スキップ (E カラムなし)")

        # 固有ひずみフィールド CSV (共通: eps_growth は E モデルに依存しない)
        export_field_csv(data, ckey)
        all_data[ckey] = data
        print()

    if not all_data:
        print("エラー: CSV が見つかりません。")
        return

    plot_comparison_3model(all_data)
    write_summary_3model(all_data)

    print()
    print("=" * 70)
    print(f"完了。{inp_count} 個の Abaqus inp ファイルを生成。")
    print()
    for ckey in all_data:
        for mk in E_MODELS:
            fname = f"biofilm_1d_bar_{ckey}_{mk}.inp"
            if os.path.isfile(os.path.join(OUT_DIR, fname)):
                print(f"  {fname}")
    print()
    print("比較図: _abaqus_input/compare_conditions_3model.png")
    print("サマリ: _abaqus_input/sigma_max_summary_3model.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
