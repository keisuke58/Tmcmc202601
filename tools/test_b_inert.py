#!/usr/bin/env python3
"""b (theta[3,4,8,9,15]) が alpha*=0 では動力学に入らないことを固定する。

背景
----
論文 Sec.2:

    the ... experiments do not involve antibiotic agents, so we set alpha* = 0,
    which eliminates the decay ... b_i psi_i from Eq. (8); the antibiotic
    sensitivity matrix B is therefore inactive and excluded

コード colab_package/hamilton_ode_jax.py:189 も同じ:

    t2 = (b_diag[i] * alpha / Eta[i]) * psi_new[i]

alpha=0 なら b がいくつでも t2 の当該項は 0。simulate_0d の alpha_const 既定は 0.0 で、
make_log_likelihood_jax_ode は alpha_const を渡さないので、**推定器では常に alpha=0**。

つまり b の 5 成分は尤度に一切入らない。それでも推定すると:

  - 事後は事前分布のまま（Drive の MAP の b 値は事前分布内をさまよった跡）
  - TMCMC は 20 次元中 5 次元を尤度勾配ゼロのパラメータに費やす。提案共分散が
    20 次元で推定され、受容率が薄まり、有効次元あたりの ESS が落ちる

対策として estimate_reduced_nishioka_jax.py は既定で prior_bounds[b] = [0, 0] とし、
tmcmc_nuts_engine.py:414 の free_mask がその次元を除外する（--estimate-b で従来動作）。

この検証が固定すること
--------------------
1. alpha=0 では b を変えても軌道が完全に一致する（bit-identical）
2. alpha>0 では b が効く（1 が alpha=0 固有であることの確認）
3. prior_bounds[b] = [0, 0] にすると engine の free_mask が 15 次元を返す
"""

import sys

import numpy as np

sys.path.insert(0, "/home/user/tmcmc202601/colab_package")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from hamilton_ode_jax import simulate_0d

B_DIMS = [3, 4, 8, 9, 15]

# 論文 DH MAP（Drive ultimate_10000p/dh_baseline）
TH = np.array(
    [
        1.6741284184996776,
        0.7561425032717847,
        0.011645004350580017,
        1.0683117596252159,
        2.2334857825268910,
        -0.2967534215182007,
        -0.7692892460758661,
        4.4502966735680890,
        5.1332970377985020,
        2.0641684469882007,
        -2.7275643812832593,
        0.3823650005847802,
        0.4578965835649136,
        0.4504108715059160,
        3.4338386488771517,
        0.2314680978386234,
        0.0071306103359640,
        -0.4937799401980541,
        -0.4043026052994370,
        5.6306251388832890,
    ]
)
PHI0 = jnp.array([0.04, 0.01, 0.94, 0.005, 0.005])  # DH Day-1 実測（正規化）


def traj(theta, alpha):
    return np.asarray(
        simulate_0d(
            jnp.array(theta),
            n_steps=500,
            dt=1e-4,
            phi_init=PHI0,
            K_hill=1e-8,
            n_hill=2.0,
            c_const=25.0,
            alpha_const=alpha,
        )
    )


def main():
    rng = np.random.default_rng(0)
    th_zero = TH.copy()
    th_zero[B_DIMS] = 0.0
    th_rand = TH.copy()
    th_rand[B_DIMS] = rng.uniform(0.0, 5.0, len(B_DIMS))

    fails = []

    # [1] alpha=0 では b が無影響
    base = traj(TH, 0.0)
    d_zero = np.abs(base - traj(th_zero, 0.0)).max()
    d_rand = np.abs(base - traj(th_rand, 0.0)).max()
    ok = d_zero == 0.0 and d_rand == 0.0
    print(f"[1] alpha=0 で b を 0 にした差        max|diff| = {d_zero:.3e}")
    print(
        f"    alpha=0 で b を乱数にした差      max|diff| = {d_rand:.3e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    print("    -> b は動力学に入らない。推定しても事後は事前分布のまま")
    if not ok:
        fails.append("1: alpha=0 でも b が軌道に影響している")

    # [2] alpha>0 なら b は効く（1 が alpha=0 固有であること）
    d_alpha = np.abs(traj(TH, 100.0) - traj(th_zero, 100.0)).max()
    ok = d_alpha > 1e-6
    print(
        f"\n[2] alpha=100 で b を 0 にした差      max|diff| = {d_alpha:.3e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    print("    -> b が無影響なのは alpha=0 のときだけ")
    if not ok:
        fails.append("2: alpha>0 でも b が効かない（b の実装が壊れている）")

    # [3] prior_bounds[b] = [0,0] で engine の free_mask が 15 次元になる
    pb = np.zeros((20, 2))
    pb[:, 0], pb[:, 1] = -3.0, 3.0
    for i in B_DIMS:
        pb[i] = [0.0, 0.0]
    # tmcmc_nuts_engine.py:414 と同じ式
    free_mask = np.abs(pb[:, 1] - pb[:, 0]) > 1e-12
    n_free = int(free_mask.sum())
    ok = n_free == 15 and not free_mask[B_DIMS].any()
    print(f"\n[3] free_mask が返す自由次元数        {n_free}  {'OK' if ok else 'FAIL'}")
    print(f"    b の次元が free か                {free_mask[B_DIMS].tolist()}")
    print("    -> prior_bounds を下限=上限にすれば engine が自動で除外する")
    if not ok:
        fails.append(f"3: 自由次元が 15 でない（{n_free}）")

    print("\n" + "=" * 62)
    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    print("3件すべて OK。b は 0 に固定して 15 次元で回すのが正しい。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
