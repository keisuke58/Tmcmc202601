#!/usr/bin/env python3
"""
hill_gate_diagnostic.py — Hill ゲートが実際に効いているかを測る

`hill_gate_sensitivity.py` との違い:

  1. **K_hill = 0（ゲート OFF）をスイープに含める。** 既存ツールは
     K ∈ [0.02, 0.05, 0.10, 0.20] で、ゲート無しを一度も試していない。
  2. **ゲートの値 h(t) そのものを軌道に沿って記録する。**
     結論だけでなく「そもそも binding しているのか」を見る。
  3. **Fn 除去の反実仮想を両条件で走らせる。**
     論文が主張する予測がゲートから出ているのか、相互作用行列から
     出ているのかを切り分ける。
  4. 結論を決め打ちで印字しない。

ゲート本体（core_hamilton_1d.py:92-97 / improved_5species_jit.py:87-95）:

    Ia = A @ (phi * psi)
    fn = phi[3] * psi[3]                        # F. nucleatum
    h  = fn**n / (K**n + fn**n)
    Ia[4] *= h                                  # P. gingivalis の行だけ

⚠️ 既知の制約 — 物理パラメータが run の config と揃っていない
------------------------------------------------------------
本スクリプトは `core_hamilton_1d` の 0D デモ既定値
（dt_h=0.01, c=100, alpha=100, Eta=1）で回す。実際の run はたとえば
`_runs/Dysbiotic_HOBIC_K0.05_n4.0_1k30/config.json` が
dt=1e-4, c_const=25.0, alpha_const=0.0 であり、**まるで違う**。

dt を100倍にして回すと Fn が全体を占める退化状態に落ちるが、それは
この既定値の産物であってモデルの性質ではない。**結論を出す前に
run の config.json を読んで渡すように直すこと。** 現状は
「ゲートの binding を見る枠組み」であって、論文の再現ではない。

なお論文の production は JAX 経路（`colab_package/hamilton_ode_jax.py`
の `simulate_0d`）であり、`_runs/` の結果は numba 経路
（`tmcmc/program2602/improved_5species_jit.py`）。ソルバが2つある。

Usage
-----
  python3 tools/hill_gate_diagnostic.py
  python3 tools/hill_gate_diagnostic.py --n-steps 2500 --k-hill 0.0 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "FEM"))

FN = 3  # F. nucleatum
PG = 4  # P. gingivalis

# core_hamilton_1d の fallback と同じ dysbiotic 寄りの theta
THETA_FALLBACK = np.array(
    [
        1.34,
        -0.18,
        1.79,
        1.17,
        2.58,
        3.51,
        2.73,
        0.71,
        2.10,
        0.37,
        2.05,
        -0.15,
        3.56,
        0.16,
        0.12,
        0.32,
        1.49,
        2.10,
        2.41,
        2.50,
    ]
)


def _gate(fn_conc: np.ndarray, K: float, n: float) -> np.ndarray:
    """ゲート値。K=0 は OFF（恒等的に 1）。"""
    if K <= 1e-9:
        return np.ones_like(fn_conc)
    num = fn_conc**n
    return num / (K**n + num)


def run_trajectory(theta, K_hill, n_hill, n_steps, fn_active=True):
    """0D Hamilton ODE を走らせ、軌道とゲート値の履歴を返す。"""
    import jax
    import jax.numpy as jnp
    from JAXFEM.core_hamilton_1d import make_initial_state, newton_step, theta_to_matrices

    jax.config.update("jax_enable_x64", True)

    theta_arr = np.asarray(theta, dtype=np.float64)
    if len(theta_arr) < 20:
        theta_arr = np.pad(theta_arr, (0, 20 - len(theta_arr)), constant_values=0.5)
    A, b_diag = theta_to_matrices(jnp.array(theta_arr[:20], dtype=jnp.float64))

    active = np.ones(5, dtype=np.int64)
    if not fn_active:
        active[FN] = 0  # F. nucleatum を除く
    active_mask = jnp.array(active, dtype=jnp.int64)

    params = {
        "dt_h": 0.01,
        "Kp1": 1e-4,
        "Eta": jnp.ones(5, dtype=jnp.float64),
        "EtaPhi": jnp.ones(5, dtype=jnp.float64),
        "c": 100.0,
        "alpha": 100.0,
        "K_hill": jnp.array(K_hill, dtype=jnp.float64),
        "n_hill": jnp.array(n_hill, dtype=jnp.float64),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
        "newton_steps": 6,
    }

    g = make_initial_state(1, active_mask)[0]
    step = jax.jit(newton_step)

    fn_hist = np.empty(n_steps)
    pg_hist = np.empty(n_steps)
    for t in range(n_steps):
        g = step(g, params)
        gn = np.asarray(g)
        fn_hist[t] = gn[FN] * gn[6 + FN]  # phi_Fn * psi_Fn
        pg_hist[t] = gn[PG]

    phi_final = np.asarray(g[0:5])
    p = phi_final / max(phi_final.sum(), 1e-12)
    H = -np.sum(np.where(p > 0, p * np.log(p), 0.0))

    h_hist = _gate(fn_hist, K_hill, n_hill)
    return {
        "K_hill": K_hill,
        "n_hill": n_hill,
        "fn_active": fn_active,
        "di": float(1.0 - H / np.log(5.0)),
        "phi_pg_final": float(phi_final[PG]),
        "phi_final": phi_final.tolist(),
        # ゲート診断 — ここが本題
        "gate_min": float(h_hist.min()),
        "gate_median": float(np.median(h_hist)),
        "gate_max": float(h_hist.max()),
        "gate_frac_below_099": float((h_hist < 0.99).mean()),
        "gate_frac_below_090": float((h_hist < 0.90).mean()),
        "gate_frac_below_050": float((h_hist < 0.50).mean()),
        "fn_conc_median": float(np.median(fn_hist)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--theta",
        type=str,
        default=None,
        help="theta_MAP.json / theta_mean.json へのパス。省略時は THETA_DEMO",
    )
    ap.add_argument("--k-hill", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--n-hill", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--n-steps", type=int, default=2500)
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "tools" / "_hill_diagnostic")
    args = ap.parse_args()

    if args.theta:
        with open(args.theta) as f:
            d = json.load(f)
        theta = np.array(d.get("theta_full") or d.get("theta_sub"), dtype=np.float64)
        theta_src = args.theta
    else:
        try:
            from JAXFEM.core_hamilton_1d import THETA_DEMO

            theta = np.array(THETA_DEMO)
            theta_src = "THETA_DEMO"
        except ImportError:
            theta = THETA_FALLBACK
            theta_src = "fallback"

    results = []
    print(f"theta: {theta_src}   n_steps: {args.n_steps}\n")
    print(
        f"{'K':>6} {'n':>4} {'DI':>8} {'phi_Pg':>9} "
        f"{'h_med':>8} {'h_min':>8} {'<0.99':>7} {'<0.90':>7} {'<0.50':>7}"
    )
    print("-" * 74)
    for K in args.k_hill:
        for n in args.n_hill:
            if K <= 1e-9 and n != args.n_hill[0]:
                continue  # K=0 は n に依らないので1回だけ
            r = run_trajectory(theta, K, n, args.n_steps)
            results.append(r)
            print(
                f"{K:>6.3f} {n:>4.1f} {r['di']:>8.4f} {r['phi_pg_final']:>9.5f} "
                f"{r['gate_median']:>8.4f} {r['gate_min']:>8.4f} "
                f"{r['gate_frac_below_099']:>7.2%} {r['gate_frac_below_090']:>7.2%} "
                f"{r['gate_frac_below_050']:>7.2%}"
            )

    # --- Fn 除去の反実仮想: 予測がゲート由来か A 由来かの切り分け ---
    print("\nF. nucleatum 除去（active_mask[3]=0）")
    print(f"{'K':>6} {'n':>4} {'phi_Pg(Fn有)':>13} {'phi_Pg(Fn無)':>13} {'変化':>10}")
    print("-" * 52)
    counterfactual = []
    # スイープと同じ (K, n) の組で回す。n を取り違えると結論が変わる。
    cf_pairs = [(r["K_hill"], r["n_hill"]) for r in results]
    for K, n in cf_pairs:
        with_fn = run_trajectory(theta, K, n, args.n_steps, fn_active=True)
        without = run_trajectory(theta, K, n, args.n_steps, fn_active=False)
        a, b = with_fn["phi_pg_final"], without["phi_pg_final"]
        rel = (b - a) / a if abs(a) > 1e-12 else float("nan")
        counterfactual.append(
            {
                "K_hill": K,
                "n_hill": n,
                "phi_pg_with_fn": a,
                "phi_pg_without_fn": b,
                "rel_change": rel,
            }
        )
        print(f"{K:>6.3f} {n:>4.1f} {a:>13.5f} {b:>13.5f} {rel:>9.1%}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "diagnostic_results.json"
    with open(out, "w") as f:
        json.dump(
            {
                "theta_source": theta_src,
                "n_steps": args.n_steps,
                "sweep": results,
                "fn_removal": counterfactual,
            },
            f,
            indent=2,
        )
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
