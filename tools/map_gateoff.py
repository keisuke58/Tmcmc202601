#!/usr/bin/env python3
"""論文の DH MAP（Drive ultimate_10000p, a45=5.63）をゲート OFF で前進計算する。

これが一番安い決定的な試験: 論文自身の MAP がゲート無しでも Pg サージを出すなら、
サージを作っているのは a45 であってゲートではない。出なければ逆。

結果（2026-09-26, phi_init = Day-1 実測を正規化）:

    ゲート ON  K=0.05 n=4   RMSE 0.069   Pg Day21 0.1476   (4.6x)
    ゲート ON  K=0.05 n=2   RMSE 0.094   Pg Day21 0.0461   (1.4x)
    ゲート OFF K=0          RMSE 0.078   Pg Day21 0.0321   (1.0x)

    実測 Pg Day21 = 0.1651 / 論文の DH Phase 2 RMSE = 0.087

ゲートを外すとサージが消える。Fn は OFF でも Day21 に 0.204 まで上がるのに
Pg がついてこない。Fn の立ち上がりを Pg に伝えているのはゲートであって a45 ではない。

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
        ("ゲート ON  K=0.05 n=2", 0.05, 2.0),
        ("ゲート ON  K=0.05 n=4", 0.05, 4.0),
        ("ゲート OFF K=0      ", 0.0, 2.0),
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
        surge = norm[-1, 4] / max(norm[0, 4], 1e-9)
        print(f"  {label}  RMSE={rmse:.4f}")
        print(f"      Pg 予測 = {np.round(norm[:,4],4)}   Day21/Day1 = {surge:>6.1f}x")
        print(f"      Fn 予測 = {np.round(norm[:,3],4)}")
    print()
print("論文の DH Phase 2 RMSE = 0.087")
