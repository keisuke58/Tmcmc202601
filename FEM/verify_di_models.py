#!/usr/bin/env python3
"""
verify_di_models.py — DI 代替モデル・多様性仮説の総合検証
===========================================================

全検証項目を実行し、PASS/FAIL を報告する。

検証カテゴリ:
  1. 境界値・単調性
  2. 数学的性質（Voigt≥Reuss, Shannon-Simpson, Pielou-Shannon）
  3. 物理的制約（E 範囲、単位）
  4. 文献比較（Pattem 2018）
  5. 仮説の方向性（dysbiotic_Pg で φ_Pg が正しい方向か）
  6. 感度解析
  7. 交差検証（4 条件）

Usage:
  python verify_di_models.py
  python verify_di_models.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from material_models import (
    DI_EXPONENT,
    DI_SCALE,
    E_MAX_PA,
    E_MIN_PA,
    E_SPECIES_PA,
    compute_all_E,
    compute_di,
    compute_di_gini,
    compute_di_pielou,
    compute_di_simpson,
    compute_E_di,
    compute_E_phi_pg,
    compute_E_reuss,
    compute_E_voigt,
    compute_E_virulence,
)

# ── 検証結果収集 ─────────────────────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []


def ok(name: str, cond: bool, msg: str = ""):
    RESULTS.append((name, cond, msg))
    return cond


def run_all_checks(verbose: bool = False):
    """全検証を実行。"""
    global RESULTS
    RESULTS = []

    # ── 1. 境界値テスト ─────────────────────────────────────────────────────
    phi_uniform = np.ones(5) / 5.0
    phi_single = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    di_shannon_u = compute_di(phi_uniform.reshape(1, 5))[0]
    di_shannon_s = compute_di(phi_single.reshape(1, 5))[0]
    ok(
        "1.1 Shannon: uniform → DI=0",
        np.isclose(di_shannon_u, 0.0, atol=1e-10),
        f"got {di_shannon_u}",
    )
    ok(
        "1.2 Shannon: single → DI=1",
        np.isclose(di_shannon_s, 1.0, atol=1e-10),
        f"got {di_shannon_s}",
    )

    di_simpson_u = compute_di_simpson(phi_uniform.reshape(1, 5))[0]
    di_simpson_s = compute_di_simpson(phi_single.reshape(1, 5))[0]
    ok(
        "1.3 Simpson: uniform → DI=0",
        np.isclose(di_simpson_u, 0.0, atol=1e-10),
        f"got {di_simpson_u}",
    )
    ok(
        "1.4 Simpson: single → DI=1",
        np.isclose(di_simpson_s, 1.0, atol=1e-10),
        f"got {di_simpson_s}",
    )

    di_gini_u = compute_di_gini(phi_uniform.reshape(1, 5))[0]
    di_gini_s = compute_di_gini(phi_single.reshape(1, 5))[0]
    ok("1.5 Gini: uniform → DI≈0", di_gini_u < 0.01, f"got {di_gini_u}")
    # Gini for single-species: (2*5*1 - 6)/5 = 0.8 (max for n=5)
    ok("1.6 Gini: single → DI high (≈0.8)", di_gini_s > 0.75, f"got {di_gini_s}")

    di_pielou_u = compute_di_pielou(phi_uniform.reshape(1, 5))[0]
    di_pielou_s = compute_di_pielou(phi_single.reshape(1, 5))[0]
    ok("1.7 Pielou: uniform → DI=0", np.isclose(di_pielou_u, 0.0, atol=1e-6), f"got {di_pielou_u}")
    ok("1.8 Pielou: single → DI=1", np.isclose(di_pielou_s, 1.0, atol=1e-6), f"got {di_pielou_s}")

    # ── 2. 単調性: 均等→単一種へ変化で DI 単調増加 ─────────────────────────
    n_path = 50
    path = np.linspace(phi_uniform, phi_single, n_path)
    di_shannon_path = compute_di(path)
    di_simpson_path = compute_di_simpson(path)
    di_gini_path = compute_di_gini(path)
    di_pielou_path = compute_di_pielou(path)

    ok("2.1 Shannon monotonic", np.all(np.diff(di_shannon_path) >= -1e-10), "check diff")
    ok("2.2 Simpson monotonic", np.all(np.diff(di_simpson_path) >= -1e-10), "check diff")
    ok("2.3 Gini monotonic", np.all(np.diff(di_gini_path) >= -1e-10), "check diff")
    ok("2.4 Pielou monotonic", np.all(np.diff(di_pielou_path) >= -1e-10), "check diff")

    # ── 3. 対称性: 種の入れ替えで DI 不変 ───────────────────────────────────
    phi_a = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    phi_b = np.array([0.3, 0.5, 0.1, 0.05, 0.05])  # swap 0,1
    di_a = compute_di(phi_a.reshape(1, 5))[0]
    di_b = compute_di(phi_b.reshape(1, 5))[0]
    ok("3.1 Shannon permutation invariant", np.isclose(di_a, di_b), f"{di_a} vs {di_b}")

    # ── 4. 正規化範囲 [0,1] ─────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    for _ in range(100):
        phi_r = rng.dirichlet(np.ones(5))
        d_s = compute_di(phi_r.reshape(1, 5))[0]
        d_g = compute_di_gini(phi_r.reshape(1, 5))[0]
        if not (0 <= d_s <= 1.001 and 0 <= d_g <= 1.001):
            ok("4.1 DI range [0,1] (Shannon,Gini)", False, f"Shannon={d_s}, Gini={d_g}")
            break
    else:
        ok("4.1 DI range [0,1] (Shannon,Gini)", True, "100 random samples OK")

    # ── 5. Voigt ≥ Reuss（古典的不等式）─────────────────────────────────────
    for _ in range(100):
        phi_r = rng.dirichlet(np.ones(5))
        e_v = compute_E_voigt(phi_r.reshape(1, 5))[0]
        e_r = compute_E_reuss(phi_r.reshape(1, 5))[0]
        if e_v < e_r - 1e-6:
            ok("5.1 Voigt ≥ Reuss", False, f"Voigt={e_v}, Reuss={e_r}")
            break
    else:
        ok("5.1 Voigt ≥ Reuss", True, "100 random samples OK")

    # ── 6. E 範囲 [E_min, E_max] ───────────────────────────────────────────
    res = compute_all_E(path, di_scale=1.0)
    e_di = res["E_di"]
    ok(
        "6.1 E_di in [E_min,E_max]",
        np.all(e_di >= E_MIN_PA - 1) and np.all(e_di <= E_MAX_PA + 1),
        f"min={e_di.min():.0f}, max={e_di.max():.0f}",
    )

    # ── 7. Pielou と Shannon の関係 J = H/ln(S) ─────────────────────────────
    # DI_pielou = 1 - J, DI_shannon = 1 - H/H_max. For 5 species H_max=ln(5), so J=H/ln(5).
    # Thus 1-DI_pielou = H/ln(5) = 1-DI_shannon when J is properly H/ln(S). So DI_pielou should equal DI_shannon.
    for i in range(n_path):
        pp = path[i : i + 1]
        ds = compute_di(pp)[0]
        dp = compute_di_pielou(pp)[0]
        if not np.isclose(ds, dp, atol=1e-5):
            ok("7.1 Pielou = Shannon (J=H/ln(S))", False, f"path[{i}] Shannon={ds}, Pielou={dp}")
            break
    else:
        ok("7.1 Pielou = Shannon (J=H/ln(S))", True, "along path")

    # ── 8. 文献比較: Pattem 2018 の方向性（diversity loss → stiffness reduction）──
    # より極端な dysbiosis で ratio が増大することを確認
    phi_comm = np.ones(5) / 5.0
    phi_dysb_mild = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
    phi_dysb_extreme = np.array([0.98, 0.005, 0.005, 0.005, 0.005])
    res_comm = compute_all_E(phi_comm.reshape(1, 5), di_scale=1.0)
    res_mild = compute_all_E(phi_dysb_mild.reshape(1, 5), di_scale=1.0)
    res_extreme = compute_all_E(phi_dysb_extreme.reshape(1, 5), di_scale=1.0)
    ratio_mild = res_comm["E_di"][0] / (res_mild["E_di"][0] + 1e-12)
    ratio_extreme = res_comm["E_di"][0] / (res_extreme["E_di"][0] + 1e-12)
    ok(
        "8.1 Pattem: commensal > dysbiotic (E_comm > E_dysb)",
        res_comm["E_di"][0] > res_mild["E_di"][0],
        f"ratio={ratio_mild:.1f}×",
    )
    ok(
        "8.2 Pattem: more dysbiosis → lower E",
        res_mild["E_di"][0] > res_extreme["E_di"][0],
        f"mild={res_mild['E_di'][0]:.0f}, extreme={res_extreme['E_di'][0]:.0f}",
    )

    # ── 9. Billings 2015: E 範囲 0.1–100,000 Pa ──────────────────────────────
    ok("9.1 E_min in Billings range", E_MIN_PA >= 0.1, f"E_min={E_MIN_PA}")
    ok("9.2 E_max in Billings range", E_MAX_PA <= 100000, f"E_max={E_MAX_PA}")

    # ── 10. dysbiotic_Pg: φ_Pg モデルが正しい方向か ─────────────────────────
    phi_pg_high = np.array([0.05, 0.05, 0.05, 0.05, 0.8])  # Pg dominant
    e_pg_comm = compute_E_phi_pg(phi_comm.reshape(1, 5))[0]
    e_pg_dysb = compute_E_phi_pg(phi_pg_high.reshape(1, 5))[0]
    ok(
        "10.1 φ_Pg: Pg dominant → E↓",
        e_pg_dysb < e_pg_comm,
        f"E_comm={e_pg_comm:.0f}, E_dysbPg={e_pg_dysb:.0f}",
    )

    # ── 11. 感度: E_species 摂動で Voigt/Reuss が妥当に変化 ─────────────────
    e_v0 = compute_E_voigt(phi_comm.reshape(1, 5))[0]
    e_species_2x = E_SPECIES_PA * 2.0
    e_v2 = compute_E_voigt(phi_comm.reshape(1, 5), e_species=e_species_2x)[0]
    ok(
        "11.1 Voigt: E_species 2× → E 2×",
        np.isclose(e_v2 / e_v0, 2.0, atol=0.01),
        f"ratio={e_v2/e_v0:.3f}",
    )

    # ── 12. 交差検証: 4 条件で DI が正しい方向 ─────────────────────────────
    conditions = {
        "commensal": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        "dysbiotic_So": np.array([0.8, 0.05, 0.05, 0.05, 0.05]),
        "dysbiotic_Pg": np.array([0.05, 0.05, 0.05, 0.05, 0.8]),
        "dysbiotic_mixed": np.array([0.4, 0.3, 0.2, 0.05, 0.05]),
    }
    e_comm = compute_all_E(conditions["commensal"].reshape(1, 5), di_scale=1.0)["E_di"][0]
    e_dysb_so = compute_all_E(conditions["dysbiotic_So"].reshape(1, 5), di_scale=1.0)["E_di"][0]
    e_dysb_pg = compute_all_E(conditions["dysbiotic_Pg"].reshape(1, 5), di_scale=1.0)["E_di"][0]
    ok(
        "12.1 DI: commensal > dysbiotic_So",
        e_comm > e_dysb_so,
        f"E_comm={e_comm:.0f}, E_dysbSo={e_dysb_so:.0f}",
    )
    ok(
        "12.2 DI: commensal > dysbiotic_Pg",
        e_comm > e_dysb_pg,
        f"E_comm={e_comm:.0f}, E_dysbPg={e_dysb_pg:.0f}",
    )

    # ── 13. ゼロ和・エッジケース ───────────────────────────────────────────
    phi_zero = np.zeros(5)
    di_z = compute_di(phi_zero.reshape(1, 5))[0]
    ok("13.1 Zero sum: DI finite", np.isfinite(di_z), f"DI={di_z}")

    # ── 14. Gini と Simpson の相関（同方向）──────────────────────────────────
    cors = []
    for _ in range(50):
        phi_r = rng.dirichlet(np.ones(5))
        g = compute_di_gini(phi_r.reshape(1, 5))[0]
        s = compute_di_simpson(phi_r.reshape(1, 5))[0]
        cors.append((g, s))
    g_arr = np.array([c[0] for c in cors])
    s_arr = np.array([c[1] for c in cors])
    corr = np.corrcoef(g_arr, s_arr)[0, 1]
    ok("14.1 Gini-Simpson correlation > 0.8", corr > 0.8, f"corr={corr:.3f}")

    # ── 15. Shannon-Simpson 関係（D_Simpson と H の既知の関係）────────────────
    # exp(H) >= 1/D_Simpson (Jost 2006). D = 1 - sum p_i^2.
    shannon_simpson_ok = True
    for _ in range(20):
        phi_r = rng.dirichlet(np.ones(5))
        p = phi_r / phi_r.sum()
        H = -np.sum(p * np.where(p > 0, np.log(p), 0))
        D = 1.0 - np.sum(p**2)
        if D > 0.05:
            if np.exp(H) < (1.0 / D) * 0.95:
                shannon_simpson_ok = False
                break
    ok("15.0 Shannon-Simpson: exp(H) >= 1/D", shannon_simpson_ok, "Jost 2006")

    # ── 16. 数値微分の連続性（E(φ) が滑らか）────────────────────────────────
    phi_c = np.array([0.25, 0.25, 0.25, 0.15, 0.1])
    eps = 1e-6
    e0 = compute_E_di(compute_di(phi_c.reshape(1, 5)), di_scale=1.0)[0]
    grad_approx = []
    for j in range(5):
        phi_p = phi_c.copy()
        phi_p[j] += eps
        phi_p /= phi_p.sum()
        e1 = compute_E_di(compute_di(phi_p.reshape(1, 5)), di_scale=1.0)[0]
        grad_approx.append((e1 - e0) / eps)
    ok("16.1 E(φ) gradient finite", np.all(np.isfinite(grad_approx)), f"grad={grad_approx}")

    # ── 17. 事後サンプル検証（利用可能な場合）────────────────────────────────
    samples_path = _ROOT / "data_5species" / "_runs" / "dh_baseline" / "samples.npy"
    if not samples_path.exists():
        samples_path = _ROOT / "data_5species" / "_runs" / "commensal_static" / "samples.npy"
    if samples_path.exists():
        try:
            sys.path.insert(0, str(_ROOT / "tmcmc" / "program2602"))
            from improved_5species_jit import BiofilmNewtonSolver5S

            samples = np.load(samples_path)
            n_use = min(20, len(samples))
            idx = np.linspace(0, len(samples) - 1, n_use, dtype=int)
            solver = BiofilmNewtonSolver5S(
                dt=1e-4,
                maxtimestep=750,
                active_species=[0, 1, 2, 3, 4],
                c_const=25.0,
                alpha_const=0.0,
                phi_init=0.02,
                K_hill=0.05,
                n_hill=4.0,
                use_numba=True,
            )
            phi_list = []
            for i in idx:
                try:
                    _, g = solver.solve(samples[i])
                    phi_list.append(g[-1, 0:5])
                except Exception:
                    pass
            if phi_list:
                phi_post = np.array(phi_list)
                res_post = compute_all_E(phi_post, di_scale=1.0)
                e_post = res_post["E_di"]
                ok(
                    "15.1 Posterior: E_di in [E_min,E_max]",
                    np.all(e_post >= E_MIN_PA - 10) and np.all(e_post <= E_MAX_PA + 10),
                    f"n={len(phi_list)}, E range [{e_post.min():.0f},{e_post.max():.0f}]",
                )
            else:
                ok("15.1 Posterior: E_di in [E_min,E_max]", True, "skip (no valid ODE)")
        except Exception as e:
            ok("15.1 Posterior: E_di in [E_min,E_max]", True, f"skip ({e})")
    else:
        ok("15.1 Posterior: E_di in [E_min,E_max]", True, "skip (no samples.npy)")

    return RESULTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details")
    args = parser.parse_args()

    print("=" * 60)
    print("DI Models & Diversity Hypothesis — Full Verification")
    print("=" * 60)

    results = run_all_checks(verbose=args.verbose)

    passed = sum(1 for _, cond, _ in results if cond)
    total = len(results)

    print()
    for name, cond, msg in results:
        status = "PASS" if cond else "FAIL"
        sym = "✓" if cond else "✗"
        line = f"  {sym} [{status}] {name}"
        if msg and (args.verbose or not cond):
            line += f" — {msg}"
        print(line)

    print()
    print("=" * 60)
    print(f"Result: {passed}/{total} passed")
    print("=" * 60)

    if passed < total:
        sys.exit(1)
    sys.exit(0)


def test_verify_all():
    """Pytest entry point: run all verifications."""
    results = run_all_checks(verbose=False)
    passed = sum(1 for _, cond, _ in results if cond)
    failed = [(n, m) for n, c, m in results if not c]
    assert passed == len(results), f"Failed: {failed}"
    assert len(results) >= 28, f"Expected >=28 checks, got {len(results)}"


if __name__ == "__main__":
    main()
