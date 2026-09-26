#!/usr/bin/env python3
"""論文の DH MAP（Drive ultimate_10000p, a45=5.63）をゲート OFF で前進計算する。

これが一番安い決定的な試験: 論文自身の MAP がゲート無しでも Pg サージを出すなら、
サージを作っているのは a45 であってゲートではない。出なければ逆。

⚠️ 重要 — JAX 実装では K_hill=0 はゲート OFF ではない
----------------------------------------------------
colab_package/hamilton_ode_jax.py:141-146

    hill_mask = (K_hill > 1e-9) * (active_mask[4] == 1)
    factor = where(den > eps, num / den, 0.0) * hill_mask
    Ia = Ia.at[4].set(Ia[4] * factor)

hill_mask を factor に**掛けている**ので、K_hill=0 では factor=0、つまり
Ia[4] = 0 になる。**Pg の相互作用が全部消える。** a45 に 0 が掛かるので
a45 の値は結果にまったく影響しない。

一方 numba 版 tmcmc/program2602/improved_5species_jit.py:89 は

    if K_hill > 0.0:
        ...
        Interaction[4] *= hill_gate

と if でガードしているので、K_hill=0 なら掛け算自体が起きず **h=1 の真の OFF**。

**2つの実装で K_hill=0 の意味が違う。** 論文のパイプラインは JAX 側。
真のゲート OFF は **K_hill = 1e-8**（実測の fn 0.031〜0.275 の全域で h ≈ 1）。

→ ゲート無しの再実行で `--K-hill 0` を渡すと、ゲート OFF ではなく
   「Pg が誰とも相互作用しない」モデルを推定してしまう。**必ず 1e-8 を使う。**

結果（2026-09-26, phi_init = Day-1 実測を正規化）:

    設定                      RMSE    Pg D15   Pg D21   D21/D15
    ゲート ON  K=0.05 n=4    0.069    0.052    0.1476     2.82
    ゲート ON  K=0.05 n=2    0.094    0.052    0.0461     0.89
    真のゲート OFF K=1e-8     0.095    0.046    0.0411     0.90
    K=0（Pg 相互作用ゼロ）    0.078    0.052    0.0321     0.61

    実測 Pg D21 = 0.1651、実測 D21/D15 = 7.76
    論文の DH Phase 2 RMSE = 0.087

真のゲート OFF でもサージは出ない（D21/D15 = 0.90）。ゲート ON n=4 だけが
立ち上がる（2.82）。**Fn の立ち上がりを Pg に伝えているのはゲートであって
a45 ではない。**

RMSE はこの差を十分に判別しない。**相互作用を完全に持たない Pg（K=0, RMSE 0.078）が、
物理的に正しい2つの変種（真の OFF 0.095、ON n=2 0.094）より RMSE で勝っている。**
Pg は分率が小さいため。論文の主要指標が論文の目玉の主張を検証できていない。

⚠️ 既知の制約 — 設定は推測である
------------------------------
ultimate_10000p の config.json が手元に無いため、以下は推測:

    dt=1e-4, n_steps=2500, c_const=25.0, alpha_const=0.0

phi_init も use_exp_init の有無が分からないので両方試している。
Day-1 実測（正規化）のほうが論文の RMSE（DH Phase 2 = 0.087）と同じ桁になるので
そちらを採用したが、確定ではない。

run の config.json（K_hill / n_hill を含む形）と run 時のログを回収したら、
必ず実際の設定で再実行して上の表を差し替えること。
n_hill が 2 か 4 かでサージが出るか出ないかが変わるので、ここは詰めないと結論できない。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/tmcmc202601/colab_package")
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from hamilton_ode_jax import simulate_0d

# Drive ultimate_10000p / dh_baseline / theta_MAP.json （論文 Table 5 を再現した MAP）
TH_DH = np.array(
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
assert abs(TH_DH[19] - 5.6306) < 1e-3, "a45 が論文の 5.63 と違う"

data = np.load("/home/user/tmcmc202601/_runs/Dysbiotic_HOBIC_K0.05_n4.0_1k30/data.npy")
data = data / data.sum(1, keepdims=True)  # 論文と同じ normalize=True
IDX = jnp.array(
    np.load("/home/user/tmcmc202601/_runs/Dysbiotic_HOBIC_K0.05_n4.0_1k30/idx_sparse.npy")
)
DAYS = [1, 3, 6, 10, 15, 21]

INITS = {
    "uniform 0.2 (use_exp_init なし)": jnp.full(5, 0.2),
    "Day-1 実測（正規化）": jnp.array(data[0]),
}

print("論文 DH MAP (a45 = +5.631) の前進計算。Pg / Fn の予測を実測と比べる。\n")
print(f"実測 Pg = {np.round(data[:,4],4)}")
print(f"実測 Fn = {np.round(data[:,3],4)}\n")

for iname, phi0 in INITS.items():
    print(f"=== phi_init: {iname}")
    for label, K, n in [
        ("ゲート ON  K=0.05 n=2      ", 0.05, 2.0),
        ("ゲート ON  K=0.05 n=4      ", 0.05, 4.0),
        ("★真のゲート OFF K=1e-8      ", 1e-8, 2.0),
        ("K=0 (Pg 相互作用ゼロ・OFFでない)", 0.0, 2.0),
    ]:
        pred = np.asarray(
            simulate_0d(
                jnp.array(TH_DH),
                n_steps=2500,
                dt=1e-4,
                phi_init=phi0,
                K_hill=K,
                n_hill=n,
                c_const=25.0,
            )[IDX, :]
        )
        norm = pred / np.maximum(pred.sum(1, keepdims=True), 1e-12)
        rmse = float(np.sqrt(((data - norm) ** 2).mean()))
        # 終端サージの指標は Day1 比ではなく Day15 -> Day21 比で見る（実測 7.76）
        surge = norm[-1, 4] / max(norm[-2, 4], 1e-9)
        print(f"  {label}  RMSE={rmse:.4f}  Pg D21/D15={surge:.2f}")
        print(f"      Pg 予測 = {np.round(norm[:,4],4)}")
        print(f"      Fn 予測 = {np.round(norm[:,3],4)}")
    print()
print("論文の DH Phase 2 RMSE = 0.087")
