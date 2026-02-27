#!/usr/bin/env python3
"""
run_full_viscoelastic_analysis.py — 4条件 × 材料モデル比較 (弾性 vs 粘弾性)

出力:
  1. 材料パラメータ比較表 (E, MR, Prony, SLS, UMAT)
  2. 2D FEM 弾性 vs 粘弾性の応力・変位比較
  3. 応力緩和の時間履歴 (全条件)
  4. 図の自動生成

Usage:
    python run_full_viscoelastic_analysis.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "JAXFEM"))

from material_models import (
    compute_E_di,
    compute_di,
    compute_mooney_rivlin_params,
    compute_prony_params_di,
    compute_viscosity_di,
    compute_relaxation_modulus,
    compute_viscoelastic_params_di,
    sls_stress_relaxation,
    DI_SCALE,
    E_MAX_PA,
    E_MIN_PA,
)
from solve_stress_2d import solve_2d_fem, solve_2d_fem_viscoelastic

# ── 4 条件の MAP theta ──────────────────────────────────────────
_RUNS = _HERE.parent / "data_5species" / "_runs"
CONDITIONS = ["commensal_static", "commensal_hobic", "dh_baseline", "dysbiotic_static"]
COND_LABELS = {
    "commensal_static": "CS (Commensal Static)",
    "commensal_hobic": "CH (Commensal HOBIC)",
    "dh_baseline": "DH (Dahl–Hedberg)",
    "dysbiotic_static": "DS (Dysbiotic Static)",
}


def load_theta(cond):
    """Load MAP theta for a condition."""
    tp = _RUNS / cond / "theta_MAP.json"
    if tp.exists():
        with open(tp) as f:
            d = json.load(f)
        if "theta_full" in d:
            return np.array(d["theta_full"], dtype=np.float64)
        elif "theta_sub" in d:
            return np.array(d["theta_sub"], dtype=np.float64)
    return None


def compute_0d_di(theta, K_hill=0.05, n_hill=4.0, dt=1e-5, maxtimestep=60000):
    """Run 0D Hamilton ODE and return final DI."""
    from improved_5species_jit import BiofilmNewtonSolver5S

    solver = BiofilmNewtonSolver5S(
        dt=dt,
        maxtimestep=maxtimestep,
        K_hill=K_hill,
        n_hill=n_hill,
    )
    t_arr, g_arr = solver.solve(theta)
    phi_final = g_arr[-1, :5].copy()
    phi_final = np.clip(phi_final, 0, None)
    phi_sum = phi_final.sum()
    if phi_sum > 0:
        p = phi_final / phi_sum
    else:
        p = np.ones(5) / 5.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum()
    di = 1.0 - H / np.log(5.0)
    return di, phi_final


def main():
    outdir = _HERE / "_viscoelastic_results"
    outdir.mkdir(exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    print("=" * 80)
    print("  VISCOELASTIC MATERIAL MODEL COMPARISON")
    print("  4 conditions × Linear / Neo-Hookean / Mooney-Rivlin / SLS viscoelastic")
    print("=" * 80)

    # ── Part 1: 0D DI + 材料パラメータ ──────────────────────────────
    print("\n" + "─" * 80)
    print("  Part 1: 0D Hamilton ODE → DI → 材料パラメータ")
    print("─" * 80)

    sys.path.insert(0, str(_HERE.parent / "tmcmc" / "program2602"))

    results = {}
    for cond in CONDITIONS:
        theta = load_theta(cond)
        if theta is None:
            print(f"  SKIP: {cond} (no theta_MAP.json)")
            continue
        di, phi_final = compute_0d_di(theta)
        results[cond] = {"theta": theta, "di": di, "phi": phi_final}

    if not results:
        print("ERROR: No MAP theta found. Aborting.")
        return

    # 材料パラメータ計算
    nu = 0.30
    print(f"\n  {'Condition':<25} {'DI':>8} {'E [Pa]':>10} {'DI/s':>8}")
    print("  " + "-" * 55)
    for cond in CONDITIONS:
        if cond not in results:
            continue
        r = results[cond]
        di = r["di"]
        E = compute_E_di(np.array([di]), di_scale=DI_SCALE).item()
        r["E"] = E
        r["di_ratio"] = di / DI_SCALE
        print(f"  {COND_LABELS.get(cond, cond):<25} {di:8.5f} {E:10.1f} {r['di_ratio']:8.3f}")

    # ── Mooney-Rivlin パラメータ ──
    print(f"\n  Mooney-Rivlin (c01_ratio=0.15, ν={nu}):")
    print(
        f"  {'Condition':<25} {'C10 [Pa]':>10} {'C01 [Pa]':>10} {'D1':>12} {'μ [Pa]':>10} {'K [Pa]':>10}"
    )
    print("  " + "-" * 80)
    for cond in CONDITIONS:
        if cond not in results:
            continue
        E = results[cond]["E"]
        mr = compute_mooney_rivlin_params(np.array([E]), nu=nu)
        results[cond]["mr"] = {k: v.item() for k, v in mr.items()}
        p = results[cond]["mr"]
        print(
            f"  {COND_LABELS.get(cond, cond):<25} {p['C10']:10.3f} {p['C01']:10.3f} "
            f"{p['D1']:12.6f} {p['mu']:10.3f} {p['K']:10.1f}"
        )

    # ── Prony series パラメータ ──
    print("\n  Prony series (1-term, DI-dependent):")
    print(f"  {'Condition':<25} {'g1':>8} {'τ1 [s]':>10} {'G∞/G0':>8} {'η [Pa·s]':>10}")
    print("  " + "-" * 70)
    for cond in CONDITIONS:
        if cond not in results:
            continue
        di = results[cond]["di"]
        pp = compute_prony_params_di(np.array([di]))
        eta = compute_viscosity_di(np.array([di]))
        results[cond]["prony"] = {
            "g1": pp["g1"].item(),
            "tau1": pp["tau1"].item(),
            "eta": eta.item(),
        }
        p = results[cond]["prony"]
        print(
            f"  {COND_LABELS.get(cond, cond):<25} {p['g1']:8.3f} {p['tau1']:10.2f} "
            f"{1-p['g1']:8.3f} {p['eta']:10.1f}"
        )

    # ── SLS (Standard Linear Solid) パラメータ ──
    print("\n  SLS viscoelastic (Zener model, DI-dependent):")
    print(
        f"  {'Condition':<25} {'E∞ [Pa]':>10} {'E0 [Pa]':>10} {'E1 [Pa]':>10} {'τ [s]':>8} {'η [Pa·s]':>10} {'E0/E∞':>6}"
    )
    print("  " + "-" * 85)
    for cond in CONDITIONS:
        if cond not in results:
            continue
        di = results[cond]["di"]
        vp = compute_viscoelastic_params_di(np.array([di]))
        results[cond]["sls"] = {k: v.item() for k, v in vp.items()}
        p = results[cond]["sls"]
        ratio = p["E_0"] / p["E_inf"] if p["E_inf"] > 0 else float("inf")
        print(
            f"  {COND_LABELS.get(cond, cond):<25} {p['E_inf']:10.1f} {p['E_0']:10.1f} "
            f"{p['E_1']:10.1f} {p['tau']:8.2f} {p['eta']:10.1f} {ratio:6.2f}"
        )

    # ── Part 2: 応力緩和の解析解 ──────────────────────────────────
    print("\n" + "─" * 80)
    print("  Part 2: SLS 応力緩和 σ(t) = [E∞ + E1·exp(-t/τ)]·ε0")
    print("─" * 80)

    eps_0 = 0.01  # 1% step strain
    t_arr = np.linspace(0, 200, 500)

    print(f"\n  Step strain ε0 = {eps_0}")
    print(f"  {'Condition':<25} {'σ(0) [Pa]':>12} {'σ(∞) [Pa]':>12} {'σ∞/σ0':>8} {'t_50% [s]':>10}")
    print("  " + "-" * 70)

    relaxation_data = {}
    for cond in CONDITIONS:
        if cond not in results:
            continue
        p = results[cond]["sls"]
        sigma = sls_stress_relaxation(p["E_inf"], p["E_1"], p["tau"], eps_0, t_arr)
        s0 = sigma[0]
        s_inf = sigma[-1]
        # half-life: time to reach (s0+s_inf)/2
        s_half = (s0 + s_inf) / 2.0
        idx_half = np.searchsorted(-sigma, -s_half)
        t_half = t_arr[min(idx_half, len(t_arr) - 1)]
        ratio = s_inf / s0 if s0 > 0 else 0
        print(
            f"  {COND_LABELS.get(cond, cond):<25} {s0:12.4f} {s_inf:12.4f} {ratio:8.3f} {t_half:10.2f}"
        )
        relaxation_data[cond] = sigma

    # ── Part 3: 2D FEM 比較 (弾性 vs 粘弾性) ──────────────────────
    print("\n" + "─" * 80)
    print("  Part 3: 2D FEM 弾性 vs 粘弾性 (20×20 grid)")
    print("─" * 80)

    Nx, Ny = 20, 20
    fem_results = {}

    for cond in CONDITIONS:
        if cond not in results:
            continue
        r = results[cond]
        di = r["di"]
        E_val = r["E"]
        prony = r["prony"]

        # Uniform field (0D-derived)
        E_field = np.full((Nx, Ny), E_val)
        eps_growth = np.full((Nx, Ny), eps_0)

        print(f"\n  --- {COND_LABELS.get(cond, cond)} ---")
        print(f"      E={E_val:.1f} Pa, g1={prony['g1']:.3f}, τ1={prony['tau1']:.1f}s")

        # (a) 弾性
        t0 = time.perf_counter()
        res_elastic = solve_2d_fem(E_field, nu, eps_growth, Nx, Ny)
        t_el = time.perf_counter() - t0

        # (b) 粘弾性 (Prony)
        t0 = time.perf_counter()
        t_total = 5.0 * prony["tau1"]
        dt_v = max(prony["tau1"] / 20.0, 0.1)
        res_visco = solve_2d_fem_viscoelastic(
            E_field,
            nu,
            eps_growth,
            Nx,
            Ny,
            g1=prony["g1"],
            tau1=prony["tau1"],
            t_total=t_total,
            dt=dt_v,
        )
        t_vi = time.perf_counter() - t0

        svm_el = res_elastic["sigma_vm"]
        svm_vi = res_visco["sigma_vm"]
        u_el = np.sqrt(res_elastic["u"][:, 0] ** 2 + res_elastic["u"][:, 1] ** 2).max()
        u_vi = np.sqrt(res_visco["u"][:, 0] ** 2 + res_visco["u"][:, 1] ** 2).max()

        fem_results[cond] = {
            "elastic": res_elastic,
            "visco": res_visco,
        }

        print(
            f"      Elastic:      σ_vm mean={svm_el.mean():.4f} Pa, max={svm_el.max():.4f}, |u|_max={u_el:.2e}, t={t_el:.3f}s"
        )
        print(
            f"      Viscoelastic: σ_vm mean={svm_vi.mean():.4f} Pa, max={svm_vi.max():.4f}, |u|_max={u_vi:.2e}, t={t_vi:.3f}s"
        )
        stress_ratio = svm_vi.mean() / svm_el.mean() if svm_el.mean() > 0 else 0
        print(
            f"      Relaxation:   σ_visco/σ_elastic = {stress_ratio:.3f} ({(1-stress_ratio)*100:.1f}% reduction)"
        )

    # ── Part 4: 条件間比較サマリー ──────────────────────────────────
    print("\n" + "─" * 80)
    print("  Part 4: 全条件比較サマリー")
    print("─" * 80)

    print(
        f"\n  {'Condition':<15} {'DI':>7} {'E [Pa]':>8} {'σ_el':>8} {'σ_vi':>8} {'Relax%':>7} {'|u|_el':>10} {'|u|_vi':>10}"
    )
    print("  " + "-" * 80)

    summary_rows = []
    for cond in CONDITIONS:
        if cond not in fem_results:
            continue
        r = results[cond]
        fe = fem_results[cond]
        svm_el = fe["elastic"]["sigma_vm"].mean()
        svm_vi = fe["visco"]["sigma_vm"].mean()
        u_el = np.sqrt(fe["elastic"]["u"][:, 0] ** 2 + fe["elastic"]["u"][:, 1] ** 2).max()
        u_vi = np.sqrt(fe["visco"]["u"][:, 0] ** 2 + fe["visco"]["u"][:, 1] ** 2).max()
        relax_pct = (1 - svm_vi / svm_el) * 100 if svm_el > 0 else 0

        label = cond.split("_")[0][:2].upper()
        print(
            f"  {label:<15} {r['di']:7.5f} {r['E']:8.1f} {svm_el:8.4f} {svm_vi:8.4f} "
            f"{relax_pct:6.1f}% {u_el:10.2e} {u_vi:10.2e}"
        )
        summary_rows.append(
            {
                "condition": cond,
                "DI": r["di"],
                "E_Pa": r["E"],
                "sigma_vm_elastic": svm_el,
                "sigma_vm_visco": svm_vi,
                "relaxation_pct": relax_pct,
                "u_max_elastic": u_el,
                "u_max_visco": u_vi,
            }
        )

    # CS vs DS ratio
    if "commensal_static" in results and "dysbiotic_static" in results:
        E_cs = results["commensal_static"]["E"]
        E_ds = results["dysbiotic_static"]["E"]
        print(f"\n  E_CS/E_DS = {E_cs/E_ds:.1f}× stiffness ratio")

        if "commensal_static" in fem_results and "dysbiotic_static" in fem_results:
            u_cs = np.sqrt(
                fem_results["commensal_static"]["elastic"]["u"][:, 0] ** 2
                + fem_results["commensal_static"]["elastic"]["u"][:, 1] ** 2
            ).max()
            u_ds = np.sqrt(
                fem_results["dysbiotic_static"]["elastic"]["u"][:, 0] ** 2
                + fem_results["dysbiotic_static"]["elastic"]["u"][:, 1] ** 2
            ).max()
            print(f"  u_DS/u_CS = {u_ds/u_cs:.1f}× displacement ratio (elastic)")

    # ── Part 5: Neo-Hookean vs Mooney-Rivlin vs Linear 比較 ────────
    print("\n" + "─" * 80)
    print("  Part 5: 構成則比較 (小ひずみ極限での等価性)")
    print("─" * 80)

    print("\n  Linear elastic → Hyperelastic パラメータ変換:")
    print(
        f"  {'Condition':<15} {'E [Pa]':>8} {'μ=E/2(1+ν)':>12} {'K=E/3(1-2ν)':>12} {'C10(NH)':>10} {'C10(MR)':>10} {'C01(MR)':>10}"
    )
    print("  " + "-" * 85)
    for cond in CONDITIONS:
        if cond not in results:
            continue
        r = results[cond]
        E = r["E"]
        mu = E / (2 * (1 + nu))
        K_bulk = E / (3 * (1 - 2 * nu))
        # Neo-Hookean: C10 = μ/2
        c10_nh = mu / 2.0
        # Mooney-Rivlin: C10 + C01 = μ/2, C01/C10 = 0.15
        c10_mr = r["mr"]["C10"]
        c01_mr = r["mr"]["C01"]
        label = cond.split("_")[0][:2].upper()
        print(
            f"  {label:<15} {E:8.1f} {mu:12.3f} {K_bulk:12.1f} {c10_nh:10.3f} {c10_mr:10.3f} {c01_mr:10.3f}"
        )

    print("\n  小ひずみでは Linear ≈ NH ≈ MR (同じ σ = Eε)")
    print("  大ひずみ (ε > 5%) で差が出る:")
    print("    - MR は I2 項 (C01) で剛性硬化が弱い → 柔らかい応答")
    print("    - NH は I1 のみ → MR より硬い")
    print("    - バイオフィルムでは ε~O(0.1) → MR の寄与あり")

    # ── Part 6: UMAT 用パラメータ (F = Fe·Fv·Fg) ──────────────────
    print("\n" + "─" * 80)
    print("  Part 6: UMAT (乗法的分解 F = Fe·Fv·Fg) パラメータ")
    print("─" * 80)

    print("\n  NPROPS=5: C10, C01, D1, η, mat_type")
    print(f"  {'Condition':<15} {'C10':>8} {'C01':>8} {'D1':>12} {'η [Pa·s]':>10} {'type':>5}")
    print("  " + "-" * 60)
    for cond in CONDITIONS:
        if cond not in results:
            continue
        r = results[cond]
        mr = r["mr"]
        eta = r["prony"]["eta"]
        label = cond.split("_")[0][:2].upper()
        print(
            f"  {label:<15} {mr['C10']:8.3f} {mr['C01']:8.3f} {mr['D1']:12.6f} {eta:10.1f} {'MR':>5}"
        )

    # ── Part 7: 応力緩和の時間履歴プロット ──────────────────────────
    print("\n" + "─" * 80)
    print("  Part 7: 図の生成")
    print("─" * 80)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure A: SLS 応力緩和 (4条件)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "commensal_static": "tab:blue",
        "commensal_hobic": "tab:green",
        "dh_baseline": "tab:orange",
        "dysbiotic_static": "tab:red",
    }

    # (a) 応力緩和曲線
    ax = axes[0, 0]
    for cond in CONDITIONS:
        if cond not in relaxation_data:
            continue
        sigma = relaxation_data[cond]
        label = cond.split("_")[0][:2].upper()
        ax.plot(t_arr, sigma, lw=2, color=colors[cond], label=label)
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("σ(t) [Pa]", fontsize=12)
    ax.set_title("(a) SLS Stress Relaxation (ε₀ = 1%)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # (b) 材料パラメータ barplot
    ax = axes[0, 1]
    conds_present = [c for c in CONDITIONS if c in results]
    x_pos = np.arange(len(conds_present))
    labels = [c.split("_")[0][:2].upper() for c in conds_present]
    E_vals = [results[c]["E"] for c in conds_present]
    E_inf_vals = [results[c]["sls"]["E_inf"] for c in conds_present]
    E_0_vals = [results[c]["sls"]["E_0"] for c in conds_present]

    w = 0.25
    bars1 = ax.bar(x_pos - w, E_vals, w, label="E (elastic)", color="steelblue")
    bars2 = ax.bar(x_pos, E_inf_vals, w, label="E∞ (long-term)", color="coral")
    bars3 = ax.bar(x_pos + w, E_0_vals, w, label="E₀ (instantaneous)", color="goldenrod")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Modulus [Pa]", fontsize=12)
    ax.set_title("(b) Elastic vs Viscoelastic Moduli", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # (c) 2D FEM σ_vm 比較 barplot
    ax = axes[1, 0]
    s_el = [fem_results[c]["elastic"]["sigma_vm"].mean() for c in conds_present if c in fem_results]
    s_vi = [fem_results[c]["visco"]["sigma_vm"].mean() for c in conds_present if c in fem_results]
    x2 = np.arange(len(s_el))
    ax.bar(x2 - 0.15, s_el, 0.3, label="Elastic σ_vm", color="steelblue")
    ax.bar(x2 + 0.15, s_vi, 0.3, label="Viscoelastic σ_vm", color="coral")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels[: len(s_el)], fontsize=11)
    ax.set_ylabel("Mean σ_vm [Pa]", fontsize=12)
    ax.set_title("(c) 2D FEM: Elastic vs Viscoelastic Stress", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # (d) 粘弾性時間履歴 (2D FEM mean σ_vm)
    ax = axes[1, 1]
    for cond in CONDITIONS:
        if cond not in fem_results:
            continue
        rv = fem_results[cond]["visco"]
        label = cond.split("_")[0][:2].upper()
        # Normalize by elastic
        s_el_mean = fem_results[cond]["elastic"]["sigma_vm"].mean()
        ax.plot(
            rv["snap_times"],
            rv["snap_sigma_vm_mean"] / s_el_mean,
            lw=2,
            color=colors[cond],
            label=label,
        )
    ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5, label="Elastic")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("σ_vm(t) / σ_elastic", fontsize=12)
    ax.set_title("(d) 2D FEM Stress Relaxation History", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Viscoelastic Material Model Comparison: 4 Biofilm Conditions",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    figpath = figdir / "viscoelastic_comparison_4conditions.png"
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure A: {figpath}")

    # Figure B: Mooney-Rivlin パラメータ vs DI
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    di_range = np.linspace(0, DI_SCALE, 200)
    E_range = compute_E_di(di_range, di_scale=DI_SCALE)

    # (a) E vs DI
    ax = axes[0]
    ax.plot(di_range, E_range, "k-", lw=2)
    for cond in CONDITIONS:
        if cond not in results:
            continue
        label = cond.split("_")[0][:2].upper()
        ax.plot(
            results[cond]["di"],
            results[cond]["E"],
            "o",
            color=colors[cond],
            ms=10,
            label=label,
            zorder=5,
        )
    ax.set_xlabel("DI", fontsize=12)
    ax.set_ylabel("E [Pa]", fontsize=12)
    ax.set_title("(a) E(DI) with MAP conditions", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # (b) Prony g1, tau1 vs DI
    ax = axes[1]
    pp_range = compute_prony_params_di(di_range)
    ax2 = ax.twinx()
    (l1,) = ax.plot(di_range, pp_range["g1"], "b-", lw=2, label="g₁")
    (l2,) = ax2.plot(di_range, pp_range["tau1"], "r--", lw=2, label="τ₁ [s]")
    for cond in CONDITIONS:
        if cond not in results:
            continue
        di = results[cond]["di"]
        ax.plot(di, results[cond]["prony"]["g1"], "bo", ms=8)
        ax2.plot(di, results[cond]["prony"]["tau1"], "r^", ms=8)
    ax.set_xlabel("DI", fontsize=12)
    ax.set_ylabel("g₁ (relaxation ratio)", fontsize=12, color="blue")
    ax2.set_ylabel("τ₁ [s]", fontsize=12, color="red")
    ax.set_title("(b) Prony Parameters vs DI", fontsize=13)
    ax.legend(handles=[l1, l2], fontsize=10)
    ax.grid(alpha=0.3)

    # (c) SLS E_0/E_inf ratio vs DI
    ax = axes[2]
    vp_range = compute_viscoelastic_params_di(di_range)
    ratio_range = vp_range["E_0"] / (vp_range["E_inf"] + 1e-12)
    ax.plot(di_range, ratio_range, "k-", lw=2, label="E₀/E∞")
    ax3 = ax.twinx()
    ax3.plot(di_range, vp_range["tau"], "g--", lw=2, label="τ [s]")
    for cond in CONDITIONS:
        if cond not in results:
            continue
        di = results[cond]["di"]
        p = results[cond]["sls"]
        ratio = p["E_0"] / p["E_inf"] if p["E_inf"] > 0 else 0
        ax.plot(di, ratio, "ko", ms=8)
        ax3.plot(di, p["tau"], "g^", ms=8)
    ax.set_xlabel("DI", fontsize=12)
    ax.set_ylabel("E₀/E∞ ratio", fontsize=12)
    ax3.set_ylabel("τ [s]", fontsize=12, color="green")
    ax.set_title("(c) SLS Stiffness Ratio & τ vs DI", fontsize=13)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Material Parameters as Functions of Dysbiosis Index", fontsize=14, weight="bold", y=1.02
    )
    fig.tight_layout()
    figpath = figdir / "material_params_vs_DI.png"
    fig.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure B: {figpath}")

    # Figure C: 2D FEM fields (elastic vs visco) for extreme conditions
    for cond in ["commensal_static", "dysbiotic_static"]:
        if cond not in fem_results:
            continue
        fe = fem_results[cond]
        n_ex, n_ey = Nx - 1, Ny - 1
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        label = COND_LABELS.get(cond, cond)

        # Top: elastic
        svm_el = fe["elastic"]["sigma_vm"].reshape(n_ex, n_ey)
        u_el = fe["elastic"]["u_grid"]
        u_mag_el = np.sqrt(u_el[..., 0] ** 2 + u_el[..., 1] ** 2)

        ax = axes[0, 0]
        im = ax.imshow(svm_el.T, origin="lower", cmap="jet", aspect="equal")
        plt.colorbar(im, ax=ax, label="σ_vm [Pa]")
        ax.set_title(f"Elastic σ_vm (max={svm_el.max():.4f})")

        ax = axes[0, 1]
        im = ax.imshow(u_mag_el.T, origin="lower", cmap="plasma", aspect="equal")
        plt.colorbar(im, ax=ax, label="|u|")
        ax.set_title(f"Elastic |u| (max={u_mag_el.max():.2e})")

        ax = axes[0, 2]
        sxx_el = fe["elastic"]["sigma_xx"].reshape(n_ex, n_ey)
        im = ax.imshow(sxx_el.T, origin="lower", cmap="RdBu_r", aspect="equal")
        plt.colorbar(im, ax=ax, label="σ_xx [Pa]")
        ax.set_title("Elastic σ_xx")

        # Bottom: visco
        svm_vi = fe["visco"]["sigma_vm"].reshape(n_ex, n_ey)
        u_vi = fe["visco"]["u_grid"]
        u_mag_vi = np.sqrt(u_vi[..., 0] ** 2 + u_vi[..., 1] ** 2)

        ax = axes[1, 0]
        im = ax.imshow(
            svm_vi.T, origin="lower", cmap="jet", aspect="equal", vmin=0, vmax=svm_el.max()
        )
        plt.colorbar(im, ax=ax, label="σ_vm [Pa]")
        ax.set_title(f"Visco σ_vm (max={svm_vi.max():.4f})")

        ax = axes[1, 1]
        im = ax.imshow(u_mag_vi.T, origin="lower", cmap="plasma", aspect="equal")
        plt.colorbar(im, ax=ax, label="|u|")
        ax.set_title(f"Visco |u| (max={u_mag_vi.max():.2e})")

        ax = axes[1, 2]
        # Stress relaxation time history
        rv = fe["visco"]
        ax.plot(rv["snap_times"], rv["snap_sigma_vm_mean"], "r-", lw=2, label="Visco")
        ax.axhline(fe["elastic"]["sigma_vm"].mean(), color="b", ls="--", lw=1.5, label="Elastic")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mean σ_vm [Pa]")
        ax.set_title("Stress Relaxation History")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.suptitle(f"{label}: Elastic (top) vs Viscoelastic (bottom)", fontsize=14, weight="bold")
        fig.tight_layout()
        tag = cond.split("_")[0][:2].upper()
        figpath = figdir / f"fem_2d_elastic_vs_visco_{tag}.png"
        fig.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure C ({tag}): {figpath}")

    # ── Save summary JSON ──
    summary = {
        "conditions": {},
        "parameters": {"nu": nu, "eps_0": eps_0, "grid": f"{Nx}x{Ny}"},
    }
    for cond in CONDITIONS:
        if cond not in results:
            continue
        r = results[cond]
        fe_el = fem_results[cond]["elastic"]["sigma_vm"]
        fe_vi = fem_results[cond]["visco"]["sigma_vm"]
        summary["conditions"][cond] = {
            "DI": r["di"],
            "E_Pa": r["E"],
            "mooney_rivlin": r["mr"],
            "prony": r["prony"],
            "sls": r["sls"],
            "fem_elastic_sigma_vm_mean": float(fe_el.mean()),
            "fem_elastic_sigma_vm_max": float(fe_el.max()),
            "fem_visco_sigma_vm_mean": float(fe_vi.mean()),
            "fem_visco_sigma_vm_max": float(fe_vi.max()),
            "relaxation_pct": float((1 - fe_vi.mean() / fe_el.mean()) * 100),
        }
    with open(outdir / "viscoelastic_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON: {outdir / 'viscoelastic_comparison_summary.json'}")

    print("\n" + "=" * 80)
    print("  DONE. All results in:", outdir)
    print("=" * 80)


if __name__ == "__main__":
    main()
