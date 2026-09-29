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
  - MH 比には log pi の差を **beta を掛けずに** 足す
  - **stage 0 の集団は pi からのサンプルでなければならない**

この検証の初版が見落としたこと（2026-09-30）
----------------------------------------
engine には **2つのコピー**がある:

    colab_package/tmcmc_nuts_engine.py          <- 初版のテストはこちらを import
    data_5species/main/tmcmc_nuts_engine.py     <- **本番の estimator が使うのはこちら**

`estimate_reduced_nishioka_jax.py` は自分と同じディレクトリの後者を掴む。
初版のテストは `colab_package` だけを sys.path に入れていたので **本番の経路を
一度も通しておらず**、「4件すべて OK」は本番について何も言っていなかった。
実際、本番側のコピーは

  - `prior_sample_fn` を受け取らず TypeError で即死し（589f682 で修正）、
  - `log_prior_fn` を **初期粒子の引き直しにしか使っておらず、MH 比にも
    HMC/NUTS の勾配にも入れていなかった**（=事前分布が効かないまま静かに走る）

という状態だった。後者は TypeError と違って落ちないので質が悪い。

そこで、このテストは **両方のコピーに対して同じ検証を回す**。

この検証が固定すること
--------------------
1. log_prior が N(0, sd^2) の対数密度（定数差を除いて）になっている
2. prior_sample_fn の引きが N(0, sd^2) を箱で truncate したものになっている
3. engine が prior_sample_fn を初期化に使う
4. 弱情報事前分布が縮約として効いている
5. **初期化を揃えても効く** = 事前分布が MH 比に入っている（初期化だけではない）
   -> rw と nuts の両方で確認する
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

B_DIMS = [3, 4, 8, 9, 15]
FREE = [i for i in range(20) if i not in B_DIMS]
SD = 3.0
LO, HI = -15.0, 20.0

# 本番が使うのは data_5species/main の方。両方回す。
ENGINES = {
    "data_5species/main (本番)": ROOT / "data_5species/main/tmcmc_nuts_engine.py",
    "colab_package": ROOT / "colab_package/tmcmc_nuts_engine.py",
}


def load_engine(path, name):
    """同名モジュールが衝突しないよう、パス指定で個別に読み込む。"""
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"_engine_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.tmcmc_engine
    finally:
        sys.path.pop(0)


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


def cheap_logL(theta):
    """theta[0] だけに依存する安い尤度。最尤は +8（事前分布と綱引きになる位置）。"""
    return -0.5 * ((theta[0] - 8.0) / 1.0) ** 2


def bounds():
    pb = np.zeros((20, 2))
    pb[:, 0], pb[:, 1] = LO, HI
    for i in B_DIMS:
        pb[i] = [0.0, 0.0]
    return pb


def check_static(log_prior, sample_prior, fails):
    """engine に依存しない部分（1, 2）。"""
    rng = np.random.default_rng(0)
    th_a, th_b = np.zeros(20), np.zeros(20)
    th_b[FREE[0]] = SD
    d = float(log_prior(jnp.array(th_a))) - float(log_prior(jnp.array(th_b)))
    ok = abs(d - 0.5) < 1e-10
    print(f"[1] log_prior(0) - log_prior(1sd) = {d:.10f}  期待 0.5  {'OK' if ok else 'FAIL'}")
    th_c = np.zeros(20)
    th_c[B_DIMS] = 5.0
    ok2 = abs(float(log_prior(jnp.array(th_c))) - float(log_prior(jnp.array(th_a)))) < 1e-12
    print(f"    b を動かしても log_prior が不変              {'OK' if ok2 else 'FAIL'}")
    if not (ok and ok2):
        fails.append("1: log_prior が期待どおりでない")

    X = sample_prior(rng, 20000)
    m, sd_hat = X[:, FREE].mean(), X[:, FREE].std()
    inside = bool(((X[:, FREE] >= LO) & (X[:, FREE] <= HI)).all())
    b_zero = bool((X[:, B_DIMS] == 0.0).all())
    ok = abs(m) < 0.05 and abs(sd_hat - SD) < 0.05 and inside and b_zero
    print(f"\n[2] 引き 20000 個: 平均 {m:+.4f}（期待 0）  標準偏差 {sd_hat:.4f}（期待 {SD}）")
    print(f"    全て箱の中: {inside}   b の5列が 0: {b_zero}  {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append("2: prior_sample_fn の分布が期待どおりでない")


def check_engine(tag, engine, log_prior, sample_prior, fails):
    pb = bounds()
    common = {"n_particles": 128, "max_stages": 3, "seed": 1,
              "verbose": False, "n_mutation_steps": 1}

    # [3] prior_sample_fn が初期化に使われる
    res = engine(cheap_logL, pb, mutation="rw", log_prior_fn=log_prior,
                 prior_sample_fn=sample_prior, **common)
    S = res["samples"]
    b_fixed = bool((S[:, B_DIMS] == 0.0).all())
    in_box = bool(((S[:, FREE] >= LO - 1e-9) & (S[:, FREE] <= HI + 1e-9)).all())
    ok = b_fixed and in_box and S.shape == (128, 20)
    print(f"\n[3] {tag}: samples {S.shape}  b 固定 {b_fixed}  箱の中 {in_box}  "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"3 [{tag}]: engine が prior_sample_fn を正しく扱っていない")

    # [4] 弱情報事前分布が縮約として効いている
    res0 = engine(cheap_logL, pb, mutation="rw", **common)
    m_prior, m_flat = float(S[:, 0].mean()), float(res0["samples"][:, 0].mean())
    ok = m_prior < m_flat
    print(f"[4] theta[0] の事後平均: 事前分布あり {m_prior:+.3f} < 箱一様 {m_flat:+.3f}  "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"4 [{tag}]: 弱情報事前分布が効いていない")

    # [5] 初期化を揃えても効くか = MH 比に入っているか
    #     prior_sample_fn を渡さず log_prior_fn だけ渡す。初期粒子の生成は
    #     箱一様（seed 共通）なので、差が出れば MH 比（NUTS なら勾配）に
    #     事前分布が入っている証拠になる。
    for mut in ("rw", "nuts"):
        a = engine(cheap_logL, pb, mutation=mut, log_prior_fn=log_prior, **common)
        b = engine(cheap_logL, pb, mutation=mut, **common)
        ma, mb = float(a["samples"][:, 0].mean()), float(b["samples"][:, 0].mean())
        ok = ma < mb - 1e-6
        print(f"[5] {mut:<4} 初期化を揃えた比較: 事前分布あり {ma:+.3f} < なし {mb:+.3f}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(
                f"5 [{tag}/{mut}]: 事前分布が MH 比に入っていない"
                "（初期粒子の生成にしか使われていない疑い）"
            )


def main():
    log_prior, sample_prior = build(SD, LO, HI)
    fails = []
    check_static(log_prior, sample_prior, fails)
    for tag, path in ENGINES.items():
        print(f"\n--- engine: {tag}")
        engine = load_engine(path, tag.split("/")[0])
        check_engine(tag, engine, log_prior, sample_prior, fails)

    print("\n" + "=" * 66)
    if fails:
        for f in fails:
            print("FAIL " + f)
        return 1
    print("すべて OK。--prior-scale の配線は両方の engine で正しい。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
