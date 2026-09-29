#!/usr/bin/env python3
"""論文 MAP で Pg 関連の相互作用係数を振り、尤度がどれだけ動くかを測る。

動機
----
論文は終端での P. gingivalis 急増を a45（Fn-Pg）に帰属させ、
95% 信用区間を [+3.03, +5.95] と狭く報告している。本当に a45 が
尤度を動かしているのかを直接確認する。

方法
----
論文 MAP（Drive ultimate_10000p）を固定し、Pg に関わる5成分
（a15 So-Pg, a25 An-Pg, a35 Vei-Pg, a45 Fn-Pg, a55 Pg 自己）を
1つずつ値域の端まで動かして chi の最大変化を見る。

    chi = sqrt(mean(((obs - pred) / sigma_i)^2))

sigma_i は replicate 由来の種別 sigma（tools/build_condition_data.py）。

結果（2026-09-26）
-----------------
    条件  ゲート        a15      a25      a35      a45      a55
    CS    ON n=2      0.030    0.099    0.014    0.000    0.000
    CS    OFF         1.330    0.266    0.504    0.000    0.000
    CH    ON n=2      0.065    0.118    0.035    0.000    0.000
    CH    OFF         1.761    0.025    0.032    0.000    0.000
    DS    ON n=2      0.301    1.167    0.672    0.712    0.804
    DS    OFF         0.267    1.085    0.325    0.378    1.097
    DH    ON n=2      0.000    0.594    0.317    0.004    0.001
    DH    OFF         0.000    2.044    3.348    0.001    0.000

1. a45 は CS / CH / **DH** で平ら。DH でもゲートの有無に関係なく 0.004 以下で、
   MAP の 5.63 を -3 まで動かしても実質無変化。
   DS だけは効く（0.38-0.71）。DS は Pg が Day1 から 0.10 と大きいので当然。
   **問題は、論文の主張（a45^DH = 5.63）が載っているのが DH であること。**
2. Pg を動かしているのは a25 (An-Pg) と a35 (Vei-Pg)。DH では a45 の 2000 倍以上。
   しかも a25 は論文 Table 1 で「既知の経路なし」に分類されている辺
3. 尤度が平らなら周辺事後は事前分布に近づくので、DH について報告されている狭い
   95% CI [+3.03, +5.95] とは両立しない。さらにその CI は論文記載の
   cross-species 事前分布 [-3,3] の外
4. ゲートを外すと commensal で Pg 関連が強く効く（CS の a15 が 1.330、
   chi 自体が 0.66）。ゲート ON では Fn が全時点 0.005 のため h ~ 0.01 で
   Pg の相互作用の 99% が殺されていた。「commensal では pathogen 関連が
   近ゼロに集中する」という Abstract の主張に直結する

⚠️ 既知の制約 — 設定は推定値
---------------------------
ultimate_10000p の config.json に K_hill も dt も記録されていなかったため
（この欠落は estimate_reduced_nishioka_jax.py 側で修正済み）、論文の RMSE と
同じ桁が出る設定を使っている:

    dt=1e-4, n_steps=2500, c_const=25.0, alpha_const=0.0
    phi_init = Day-1 実測を正規化

run のログを回収したら実設定で再実行して上の表を差し替えること。
ただし 2. と 3. はゲート ON/OFF・commensal/dysbiotic のどの組み合わせでも
成立しており、定性的には設定に依らない。

補足 — 再探索した解では a45 が効く
--------------------------------
ゲート OFF で theta を探し直した解（RMSE 0.0584）では a45 を 0 にすると
終端の急増が消える（Day21/Day15 が 5.61 -> 1.07）。論文の MAP は
「a45 が何もしない盆地」にあり、再探索した解は別の盆地にある。
→ 再実行に意味があり、かつ Phase 2 の warm start を使い回せない理由。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "colab_package"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from build_condition_data import build, sigma_per_species
from hamilton_ode_jax import simulate_0d

N, DT, C = 2500, 1e-4, 25.0
IDX = jnp.array(np.load(ROOT / "_runs" / "Dysbiotic_HOBIC_K0.05_n4.0_1k30" / "idx_sparse.npy"))

PG_ENTRIES = [("a15", 16), ("a25", 17), ("a35", 18), ("a45", 19), ("a55", 14)]
CONDS = {
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "DH": ("Dysbiotic", "HOBIC"),
}

# Drive ultimate_10000p の MAP
MAP_PAPER = {
    "CS": [
        0.0493937266784579,
        1.1205172419069056,
        1.9718872944671450,
        0.8463892209691881,
        1.7383313516512064,
        -0.9909414989478150,
        -0.3639154133219180,
        0.1943734928747933,
        0.5333021792251369,
        -0.0674052146174146,
        2.5786161282915490,
        -0.2935238920901646,
        1.0344354104599200,
        -0.4311218191408299,
        -0.9452396592151930,
        -0.9693013413003873,
        -0.9605625623557498,
        0.8814116746080459,
        0.3044696063873287,
        -0.6808571003051518,
    ],
    "CH": [
        -0.4239504263350337,
        -0.5511365239227430,
        -0.2538664606791382,
        6.5706700499037110,
        1.1616727004468488,
        -0.5626322796110986,
        -0.8463658711979718,
        0.9004270874003831,
        0.0883541202147635,
        2.4936519284359730,
        0.4684310713745831,
        -0.9791522571776690,
        0.4458412589356100,
        -0.0917129998948555,
        0.5695158663404118,
        -0.8978953169323581,
        -0.0541189744870024,
        -0.4145836145421854,
        -0.6455821017102155,
        -0.0669857068459373,
    ],
    "DS": [
        0.2844232052037148,
        0.7240025395633488,
        1.2582929182080167,
        2.4076578813598966,
        0.0430934857594695,
        1.0000395052394193,
        0.7800134862997422,
        0.3892606083205742,
        0.3776370613193656,
        1.1413603262575438,
        -0.7436978597804416,
        0.4173525323996778,
        0.3581308269661704,
        2.2600206523866095,
        0.2972171166778930,
        1.2080636076426903,
        -0.4487996939476591,
        1.9595568605254927,
        1.0135060093790136,
        2.1987743944538920,
    ],
    "DH": [
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
    ],
}


def chi_fn(data, sigma, phi0, K, n):
    obs, sig = jnp.array(data), jnp.array(sigma)

    def chi(theta):
        pred = simulate_0d(
            jnp.array(theta), n_steps=N, dt=DT, phi_init=phi0, K_hill=K, n_hill=n, c_const=C
        )[IDX, :]
        pred = jnp.clip(pred, 1e-10, 1 - 1e-10)
        pred = pred / jnp.maximum(pred.sum(1, keepdims=True), 1e-12)
        return float(jnp.sqrt((((obs - pred) / sig) ** 2).mean()))

    return chi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", default="CS,CH,DS,DH")
    ap.add_argument("--lo", type=float, default=-3.0, help="振る下限（論文記載の事前分布）")
    ap.add_argument("--hi", type=float, default=3.0, help="振る上限")
    args = ap.parse_args()

    print(f"論文 MAP を固定し Pg 関連5成分を [{args.lo}, {args.hi}] に振ったときの chi の最大変化")
    print("（chi は replicate 由来の種別 sigma で規格化した残差）\n")
    hdr = "".join(f"{nm:>9}" for nm, _ in PG_ENTRIES)
    print(f"{'条件':<5}{'ゲート':<14}{'chi(MAP)':>10}{hdr}")
    print("-" * (29 + 9 * len(PG_ENTRIES)))

    for tag in args.conditions.split(","):
        tag = tag.strip()
        cond, cult = CONDS[tag]
        _, data = build(cond, cult)
        sigma = sigma_per_species(cond, cult)
        phi0 = jnp.array(data[0])
        th = np.array(MAP_PAPER[tag])
        for K, n, lbl in [(0.05, 2.0, "ON K=.05 n=2"), (1e-8, 2.0, "OFF")]:
            chi = chi_fn(data, sigma, phi0, K, n)
            base = chi(th)
            deltas = []
            for _, k in PG_ENTRIES:
                d = 0.0
                for v in (args.lo, args.hi):
                    alt = th.copy()
                    alt[k] = v
                    d = max(d, abs(chi(alt) - base))
                deltas.append(d)
            row = "".join(f"{d:>9.3f}" for d in deltas)
            print(f"{tag:<5}{lbl:<14}{base:>10.4f}{row}", flush=True)

    print(
        "\n-> a45 が平らで a25 (An-Pg) / a35 (Vei-Pg) が支配していれば、論文は主張を\n"
        "   別の辺に載せている。ゲート OFF で commensal の値が大きくなるなら、\n"
        "   ゲートが Pg の相互作用を muted にしていたということ。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
