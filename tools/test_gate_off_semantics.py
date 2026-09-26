#!/usr/bin/env python3
"""K_hill=0 が「ゲート OFF」であって「Pg の相互作用ゼロ」ではないことを固定する。

背景
----
colab_package/hamilton_ode_jax.py は以前こう書いていた:

    hill_mask = (K_hill > 1e-9) * (active_mask[4] == 1)
    factor = where(den > eps, num / den, 0.0) * hill_mask
    Ia = Ia.at[4].set(Ia[4] * factor)

hill_mask を factor に掛けているので、K_hill=0 では factor=0、つまり Ia[4]=0。
ゲートが切れるのではなく **Pg の相互作用が全部消える**。a45 に 0 が掛かるため、
「ゲート無しで a45 を推定し直す」という実験そのものが成立しなくなる。

numba 版 tmcmc/program2602/improved_5species_jit.py:89 は

    if K_hill > 0.0:
        ...
        Interaction[4] *= hill_gate

と if でガードしているので K_hill=0 は h=1 の真の OFF。**2実装の意味が違っていた。**
論文のパイプラインは JAX 側なので、`--K-hill 0` で回すと結果が壊れる。

この検証が固定すること
--------------------
1. K_hill=0 と K_hill=1e-8（h≈1）が同じ軌道を与える = 真のゲート OFF
2. K_hill=0 の軌道が「a45=0 にした軌道」と一致しない = Pg の相互作用は生きている
3. K_hill=0.05 は 0 と異なる = ゲートが実際に効いている
4. K_hill=0 で a45 を変えると軌道が変わる = a45 が identifiable な状態
"""

import sys

import numpy as np

sys.path.insert(0, "/home/user/tmcmc202601/colab_package")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from hamilton_ode_jax import simulate_0d

# 論文 DH MAP（Drive ultimate_10000p/dh_baseline）。theta[19] = a45 = +5.6306
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
KW = {"n_steps": 500, "dt": 1e-4, "phi_init": PHI0, "n_hill": 2.0, "c_const": 25.0}


def traj(theta, K_hill):
    return np.asarray(simulate_0d(jnp.array(theta), K_hill=K_hill, **KW))


def main():
    th_a45_zero = TH.copy()
    th_a45_zero[19] = 0.0
    th_a45_half = TH.copy()
    th_a45_half[19] = TH[19] / 2

    t_k0 = traj(TH, 0.0)
    t_tiny = traj(TH, 1e-8)
    t_gate = traj(TH, 0.05)
    t_a45_0 = traj(th_a45_zero, 0.0)
    t_a45_h = traj(th_a45_half, 0.0)

    fails = []

    d = np.abs(t_k0 - t_tiny).max()
    ok = d < 1e-10
    print(
        f"[1] K_hill=0 と K_hill=1e-8 が一致          max|diff| = {d:.3e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    print("    -> K_hill=0 は h=1 の真のゲート OFF")
    if not ok:
        fails.append("1: K_hill=0 が gate-off になっていない")

    d = np.abs(t_k0 - t_a45_0).max()
    ok = d > 1e-6
    print(
        f"\n[2] K_hill=0 と a45=0 が別物                max|diff| = {d:.3e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    print("    -> K_hill=0 でも Pg の相互作用は生きている")
    if not ok:
        fails.append("2: K_hill=0 が Pg の相互作用を消している（旧バグ）")

    d = np.abs(t_k0 - t_gate).max()
    ok = d > 1e-6
    print(
        f"\n[3] K_hill=0 と K_hill=0.05 が別物          max|diff| = {d:.3e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    print("    -> ゲートは有効化すると実際に効く")
    if not ok:
        fails.append("3: ゲートを入れても軌道が変わらない")

    d = np.abs(t_k0 - t_a45_h).max()
    ok = d > 1e-6
    print(
        f"\n[4] K_hill=0 で a45 を半分にすると変わる     max|diff| = {d:.3e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    print("    -> ゲート OFF でも a45 は identifiable")
    if not ok:
        fails.append("4: ゲート OFF で a45 が軌道に影響しない")

    print("\n" + "=" * 62)
    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    print("4件すべて OK。--K-hill 0 は真のゲート OFF として使える。")
    print("（それでも再実行では 1e-8 を明示する方が意図が読み取れる）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
