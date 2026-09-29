#!/usr/bin/env python3
"""弱情報事前分布の配線を検証する（--prior-scale）。

なぜ入れたか
-----------
尤度に縮退方向がある。DH の a35 (Vei-Pg) は箱を [-4,8] から [-15,20] に広げても
新しい境界に張り付き、chi が 0.6233 -> 0.5104 と改善し続ける。有限の最適値を
持たないので、箱一様の事前分布では **事前分布の選択が結果を決めてしまう**。
論文 Sec.6.8 の r_i = sigma_post/Delta_prior も、事後が狭いのは箱が切っている
からで、データが決めているのではない（B2）。

そこで A の15成分に N(0, prior_scale^2) の弱情報事前分布を置く。箱は support
として残す。

TMCMC で気をつける点
------------------
stage m の目標は pi(theta) * L(theta)^beta なので、
  - MH 比には log pi の差を **beta を掛けずに** 足す（engine は元からそうなっている）
  - **stage 0 の集団は pi からのサンプルでなければならない**
後者が抜けていた。engine は箱一様で初期化していたので、非一様な事前分布を
渡すと stage 0 の目標分布がずれ、事後もエビデンスも狂う。
prior_sample_fn を追加して初期化もそこから引くようにした。

この検証が固定すること
--------------------
1. log_prior が N(0, sd^2) の対数密度（定数差を除いて）になっている
2. prior_sample_fn の引きが N(0, sd^2) を箱で truncate したものになっている
   （平均 ~ 0、標準偏差 ~ sd、全て箱の中、b の5列は 0）
3. engine が prior_sample_fn を初期化に使う（使わない場合は警告を出す）
4. prior_scale = 0 なら log_prior_fn / prior_sample_fn は None（従来動作）
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "colab_package"))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from tmcmc_nuts_engine import tmcmc_engine

B_DIMS = [3, 4, 8, 9, 15]
FREE = [i for i in range(20) if i not in B_DIMS]
SD = 3.0
LO, HI = -15.0, 20.0


def build(sd, lo, hi):
    """estimate_reduced_nishioka_jax.py が作るものと同じ2つを返す。"""
    free = np.array(FREE, dtype=np.int32)
    free_j = jnp.array(free)

    def log_prior(theta):
        return -0.5 * jnp.sum((theta[free_j] / sd) ** 2)

    def sample_prior(rng, n):
        out = np.zeros((n, 20), dtype=np.float64)
        for i in free:
            col = np.empty(n)
            filled = 0
            for _ in range(200):
                cand = rng.normal(0.0, sd, size=max(n - filled, 1) * 2)
                cand = cand[(cand >= lo) & (cand <= hi)]
                take = min(len(cand), n - filled)
                if take > 0:
                    col[filled : filled + take] = cand[:take]
                    filled += take
                if filled >= n:
                    break
            if filled < n:
                col[filled:] = rng.uniform(lo, hi, n - filled)
            out[:, i] = col
        return out

    return log_prior, sample_prior


def main():
    log_prior, sample_prior = build(SD, LO, HI)
    fails = []

    # [1] log_prior が正規分布の対数密度になっているか
    rng = np.random.default_rng(0)
    th_a = np.zeros(20)
    th_b = np.zeros(20)
    th_b[FREE[0]] = SD  # 1 成分だけ sd ずらす
    d = float(log_prior(jnp.array(th_a))) - float(log_prior(jnp.array(th_b)))
    ok = abs(d - 0.5) < 1e-10  # -0.5*(0) - (-0.5*(1)^2) = 0.5
    print(f"[1] log_prior(0) - log_prior(1sd) = {d:.10f}  期待 0.5  {'OK' if ok else 'FAIL'}")
    th_c = np.zeros(20)
    th_c[B_DIMS] = 5.0  # b を動かしても事前分布は変わらない
    ok2 = abs(float(log_prior(jnp.array(th_c))) - float(log_prior(jnp.array(th_a)))) < 1e-12
    print(f"    b を動かしても log_prior が不変              {'OK' if ok2 else 'FAIL'}")
    if not (ok and ok2):
        fails.append("1: log_prior が期待どおりでない")

    # [2] prior_sample_fn の引きが truncated N(0, sd^2) か
    X = sample_prior(rng, 20000)
    m, sd_hat = X[:, FREE].mean(), X[:, FREE].std()
    inside = bool(((X[:, FREE] >= LO) & (X[:, FREE] <= HI)).all())
    b_zero = bool((X[:, B_DIMS] == 0.0).all())
    ok = abs(m) < 0.05 and abs(sd_hat - SD) < 0.05 and inside and b_zero
    print(f"\n[2] 引き 20000 個: 平均 {m:+.4f}（期待 0）  標準偏差 {sd_hat:.4f}（期待 {SD}）")
    print(
        f"    全て箱 [{LO}, {HI}] の中: {inside}   b の5列が 0: {b_zero}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    if not ok:
        fails.append("2: prior_sample_fn の分布が期待どおりでない")

    # [3] engine が prior_sample_fn を初期化に使うか
    #     尤度は theta[0] だけに依存する安い関数にする
    def cheap_logL(theta):
        return -0.5 * ((theta[0] - 2.0) / 0.5) ** 2

    pb = np.zeros((20, 2))
    pb[:, 0], pb[:, 1] = LO, HI
    for i in B_DIMS:
        pb[i] = [0.0, 0.0]

    res = tmcmc_engine(
        cheap_logL,
        pb,
        mutation="rw",
        n_particles=128,
        max_stages=3,
        seed=1,
        verbose=False,
        log_prior_fn=log_prior,
        prior_sample_fn=sample_prior,
        n_mutation_steps=1,
    )
    S = res["samples"]
    b_fixed = bool((S[:, B_DIMS] == 0.0).all())
    in_box = bool(((S[:, FREE] >= LO - 1e-9) & (S[:, FREE] <= HI + 1e-9)).all())
    ok = b_fixed and in_box and S.shape == (128, 20)
    print(
        f"\n[3] engine 実行: samples {S.shape}  b の5列が 0: {b_fixed}  "
        f"箱の中: {in_box}  {'OK' if ok else 'FAIL'}"
    )
    print(
        f"    theta[0] の事後平均 = {S[:, 0].mean():+.3f}（尤度の最尤は +2.0、"
        f"事前分布 N(0,{SD}^2) が引き戻す）"
    )
    if not ok:
        fails.append("3: engine が prior_sample_fn を正しく扱っていない")

    # [4] 事前分布ありの方が theta[0] が 0 寄りになる（弱情報事前分布が効いている）
    res0 = tmcmc_engine(
        cheap_logL,
        pb,
        mutation="rw",
        n_particles=128,
        max_stages=3,
        seed=1,
        verbose=False,
        n_mutation_steps=1,
    )
    m_prior, m_flat = float(S[:, 0].mean()), float(res0["samples"][:, 0].mean())
    ok = m_prior < m_flat
    print(
        f"\n[4] theta[0] の事後平均: 事前分布あり {m_prior:+.3f} < "
        f"箱一様 {m_flat:+.3f}  {'OK' if ok else 'FAIL'}"
    )
    print("    -> 弱情報事前分布が縮約として効いている")
    if not ok:
        fails.append("4: 弱情報事前分布が効いていない")

    print("\n" + "=" * 66)
    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    print("4件すべて OK。--prior-scale の配線は正しい。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
