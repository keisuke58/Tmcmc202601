#!/usr/bin/env python3
"""
test_log_evidence_fix.py — SMC の log-evidence 累積が log-sum-exp 恒等式を
満たすことを確認する。

背景
----
`tmcmc_nuts_engine.py` の evidence 累積は、数値安定化のため重みから
max を引いたあと、それを足し戻していなかった。各ステージで
delta_beta * max(logL) だけずれ、logL が負の領域では log_evidence が
過大に出て、恒等的に成り立つはずの ln Z <= max logL を破る。

論文 Table 3 では 8 セル中 6 セルでこの違反が出ている（Phase 2 は全条件）。

Usage
-----
  python3 tools/test_log_evidence_fix.py
"""

from itertools import pairwise

import numpy as np


def accumulate_buggy(logL, betas):
    """修正前: max を引いたまま足し戻さない。"""
    lz = 0.0
    for b0, b1 in pairwise(betas):
        lw = (b1 - b0) * logL
        w = np.exp(np.clip(lw - np.max(lw), -500, 500))
        lz += np.log(np.mean(w) + 1e-300)
    return lz


def accumulate_fixed(logL, betas):
    """修正後: log-sum-exp の補正項を足し戻す。"""
    lz = 0.0
    for b0, b1 in pairwise(betas):
        lw = (b1 - b0) * logL
        m = np.max(lw)
        w = np.exp(np.clip(lw - m, -500, 500))
        lz += m + np.log(np.mean(w) + 1e-300)
    return lz


def reference(logL, betas):
    """シフト無しの直接計算（小さい値域でのみ安全）。"""
    lz = 0.0
    for b0, b1 in pairwise(betas):
        lz += np.log(np.mean(np.exp((b1 - b0) * logL)))
    return lz


def main() -> int:
    rng = np.random.default_rng(0)
    ok = True

    print("【1】単段 beta: 0 -> 1。ln Z は ln(mean(L)) に厳密一致するはず")
    print(f"{'':>4} {'厳密解':>12} {'修正後':>12} {'修正前':>12} {'修正前の誤差':>14}")
    for scale in (1.0, 10.0, 100.0):
        logL = -rng.random(2000) * scale  # 負の対数尤度
        exact = np.log(np.mean(np.exp(logL)))
        fixed = accumulate_fixed(logL, [0.0, 1.0])
        buggy = accumulate_buggy(logL, [0.0, 1.0])
        ok &= abs(fixed - exact) < 1e-9
        print(f"{scale:>4.0f} {exact:>12.5f} {fixed:>12.5f} {buggy:>12.5f} {buggy - exact:>+14.5f}")

    print("\n【2】多段。修正前のずれは Σ Δβ·max(logL) に一致するはず")
    print(f"{'stages':>7} {'厳密解':>12} {'修正後':>12} {'修正前':>12} {'予測ずれ':>12}")
    for n_stages in (2, 5, 20):
        logL = -rng.random(2000) * 20.0
        betas = list(np.linspace(0.0, 1.0, n_stages + 1))
        exact = reference(logL, betas)
        fixed = accumulate_fixed(logL, betas)
        buggy = accumulate_buggy(logL, betas)
        predicted = sum(
            (b1 - b0) * np.max((b1 - b0) * logL) / (b1 - b0) for b0, b1 in pairwise(betas)
        )
        ok &= abs(fixed - exact) < 1e-9
        ok &= abs((buggy - exact) + predicted) < 1e-9
        print(f"{n_stages:>7} {exact:>12.5f} {fixed:>12.5f} {buggy:>12.5f} {-predicted:>+12.5f}")

    print("\n【3】ln Z <= max logL（恒等式）を満たすか")
    print(
        f"{'scale':>6} {'max logL':>12} {'修正後 ln Z':>13} {'修正前 ln Z':>13} {'修正前の違反':>14}"
    )
    for scale in (1.0, 20.0, 120.0):
        logL = -rng.random(3000) * scale
        betas = list(np.linspace(0.0, 1.0, 11))
        mx = float(np.max(logL))
        fixed = accumulate_fixed(logL, betas)
        buggy = accumulate_buggy(logL, betas)
        ok &= fixed <= mx + 1e-9
        viol = "はい" if buggy > mx else "いいえ"
        print(f"{scale:>6.0f} {mx:>12.5f} {fixed:>13.5f} {buggy:>13.5f} {viol:>14}")

    print("\n【4】論文 Table 3 の値域を再現する。ずれの大きさは max(logL) で決まる")
    print("    （上の【1】〜【3】は max(logL)≈0 なのでずれが小さく見えるだけ）")
    print(f"{'max logL':>10} {'厳密解':>12} {'修正前':>12} {'過大分':>10} {'修正前の違反':>14}")
    for target_max in (-6.3, -21.6, -112.3, -170.4):
        logL = -rng.random(3000) * 5.0
        logL = logL - logL.max() + target_max  # max(logL) を狙った値に
        betas = list(np.linspace(0.0, 1.0, 8))
        exact = accumulate_fixed(logL, betas)
        buggy = accumulate_buggy(logL, betas)
        viol = "はい" if buggy > target_max else "いいえ"
        print(
            f"{target_max:>10.1f} {exact:>12.2f} {buggy:>12.2f} "
            f"{buggy - exact:>+10.2f} {viol:>14}"
        )
        ok &= exact <= target_max + 1e-9

    print("\n【5】論文 Table 3 に補正を当てたらどうなるか")
    print("    補正量 ≈ max(logL)（Σ Δβ = 1 のため）")
    print(
        f"{'Cond/Phase':>12} {'max ln L':>10} {'報告 ln Ẑ':>11} "
        f"{'補正後 ≈':>10} {'≤ max lnL?':>11}"
    )
    table = [
        ("CS P1", -6.3, -2.6),
        ("CH P1", -3.6, -1.3),
        ("DS P1", -1.2, -3.8),
        ("DH P1", -6.6, -9.6),
        ("CS P2", -21.6, -3.3),
        ("CH P2", -112.3, -8.5),
        ("DS P2", -93.4, -2.8),
        ("DH P2", -170.4, -11.9),
    ]
    for name, mx, rep in table:
        corr = rep + mx
        print(
            f"{name:>12} {mx:>10.1f} {rep:>11.1f} {corr:>10.1f} "
            f"{'✓' if corr <= mx else '✗':>11}"
        )

    print("\n" + ("PASS — 修正後は厳密解と一致し ln Z <= max logL を満たす" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
