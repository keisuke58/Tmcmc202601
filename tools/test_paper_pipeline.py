#!/usr/bin/env python3
"""論文パイプライン（estimate_paper_jax.py）の検証。

なぜこのテストか
----------------
2026-09 に、GPU で回した事後分布がすべて無効だったことが分かった。原因は二つ:

1. 前進モデル・エンジン・ローダーが同名の別ファイルで、sys.path の順で暗黙に選ばれていた。
   推定と評価が別の前進モデルを使い、保存した logL と粒子が対応しなくなった。
2. 2026-03-30 に前進モデルだけが 15 次元になり、20 次元の θ を渡す estimator と食い違った。
   a35/a45（θ[18], θ[19]）は前進モデルに読まれず、事後は事前分布のまま漂っていた。

それまでのテスト（test_b_inert / test_gate_off_semantics / test_weak_prior）は
colab_package 側のコピーを import しており、本番の経路を一度も通っていなかった。
このテストは **estimate_paper_jax.py が実際に読み込むモジュール** を対象にする。

固定すること
------------
[1] estimate_paper_jax は固有名のモジュール（hamilton_ode_jax_paper / tmcmc_engine_paper）を使う
[2] θ の並び: A の 15 成分はすべて前進モデルを動かし、b の 5 成分は動かさない（ゲート ON/OFF とも）
[3] K_hill=0 は真のゲート OFF（Pg 行を 0 倍しない）で、ゲート ON とは別の軌道になる
[4] 論文の 4 条件の RMSE（0.119 / 0.104 / 0.033 / 0.087）を論文 MAP から再現する
[5] 感度チェックは本物の尤度を通し、θ[18] を読まない尤度は止める
[6] 2026-09 の事故そのもの（15 次元モデルに 20 次元 θ）を感度チェックが捕まえる
[7] 事前分布が MH 比に入る（初期化を揃えても縮約が効く）— rw と nuts
[8] beta=1 まで回した run では ln Z <= max logL
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "data_5species" / "main"
sys.argv = [sys.argv[0], "--device", "cpu"]  # estimate_paper_jax の早期デバイス判定用
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN.parent))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import estimate_paper_jax as EP

H, E = EP._H, EP._E
A_DIMS = [i for i in range(20) if i not in EP.B_DIMS]
PAPER_RMSE = {"CS": 0.1190, "CH": 0.1040, "DS": 0.0327, "DH": 0.0868}  # 原稿は 3 桁
CC = {
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "DH": ("Dysbiotic", "HOBIC"),
}
fails = []


def check(cond, msg):
    print(f"    {'OK  ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def paper_maps():
    src = (ROOT / "tools" / "table5_check.py").read_text()
    return {
        k: np.array(v) for k, v in eval(re.search(r"MAP = (\{.*?\n\})", src, re.S).group(1)).items()
    }


def data_for(tag):
    cond, cult = CC[tag]
    data, t_days, sig, phi_exp, _ = EP.load_experimental_data(
        EP.DATA_DIR, cond, cult, 1, normalize=True, use_exp_init=True
    )
    _, idx = EP.convert_days_to_model_time(t_days, 1e-4, 2500, day_scale=None)
    p0 = np.clip(phi_exp / phi_exp.sum(), 0.001, 0.99)  # 論文時点の estimator と同じ
    return data, t_days, sig, p0, np.clip(idx, 0, 2500)


def run_phi(theta, p0, K, n):
    return np.asarray(
        H.simulate_0d(
            jnp.array(theta), n_steps=2500, dt=1e-4, phi_init=jnp.array(p0), K_hill=K, n_hill=n
        )
    )


MAPS = paper_maps()
dh = data_for("DH")

print("[1] estimate_paper_jax が読み込むモジュール")
check(Path(H.__file__).name == "hamilton_ode_jax_paper.py", f"前進モデル = {Path(H.__file__).name}")
check(Path(E.__file__).name == "tmcmc_engine_paper.py", f"エンジン   = {Path(E.__file__).name}")
check(
    EP.simulate_0d is H.simulate_0d and EP.tmcmc_engine is E.tmcmc_engine,
    "estimator の関数は上の2つを指す",
)

print("\n[2] θ の並び（論文 DH MAP から各成分を +1 動かす）")
for K, n, lbl in ((0.05, 4.0, "ゲート ON"), (0.0, 4.0, "ゲート OFF")):
    base = run_phi(MAPS["DH"], dh[3], K, n)
    eff = {}
    for i in range(20):
        t = MAPS["DH"].copy()
        t[i] += 1.0
        eff[i] = np.abs(run_phi(t, dh[3], K, n) - base).max()
    dead_a = [EP.THETA_NAMES[i] for i in A_DIMS if eff[i] < 1e-8]
    live_b = [EP.THETA_NAMES[i] for i in EP.B_DIMS if eff[i] != 0.0]
    check(not dead_a, f"{lbl}: A の 15 成分がすべて効く（効かない成分 {dead_a}）")
    check(not live_b, f"{lbl}: b の 5 成分は効果ゼロ（効いた成分 {live_b}）")

print("\n[3] ゲート OFF の意味")
on, off = run_phi(MAPS["DH"], dh[3], 0.05, 4.0), run_phi(MAPS["DH"], dh[3], 0.0, 4.0)
check(
    np.abs(on - off).max() > 1e-3,
    f"K=0 とゲート ON は別の軌道（max|Δ| = {np.abs(on - off).max():.3f}）",
)

print("\n[4] 論文の RMSE を論文 MAP から再現（ゲート ON, n=4）")
for tag in ("CS", "CH", "DS", "DH"):
    data, t_days, sig, p0, idx = data_for(tag)
    g = np.asarray(
        H.simulate_0d_full(
            jnp.array(MAPS[tag]),
            n_steps=2500,
            dt=1e-4,
            phi_init=jnp.array(p0),
            K_hill=0.05,
            n_hill=4.0,
        )
    )
    p = np.clip(g[idx, :5], 1e-10, 1 - 1e-10)
    p = p / p.sum(1, keepdims=True)
    r = float(np.sqrt(np.mean((data - p) ** 2)))
    check(abs(r - PAPER_RMSE[tag]) < 5e-4, f"{tag}: RMSE {r:.4f}（論文 {PAPER_RMSE[tag]:.3f}）")

print("\n[5] 感度チェック")
data, t_days, sig, p0, idx = dh
ll_real = EP.make_log_likelihood_jax_ode(
    data=data,
    t_days=t_days,
    idx_sparse=idx,
    sigma_obs=sig,
    phi_init=p0,
    K_hill=0.0,
    n_hill=4.0,
)
pb = EP.load_prior_bounds("Dysbiotic", "HOBIC")
pb[EP.B_DIMS] = 0.0
try:
    EP.check_sensitivity(ll_real, pb)
    check(True, "論文の尤度（ゲート OFF・b 固定）は通る")
except RuntimeError as e:
    check(False, f"論文の尤度で止まった: {e}")


def ll_without_18(theta):
    return ll_real(theta.at[18].set(0.0))


try:
    EP.check_sensitivity(ll_without_18, pb)
    check(False, "θ[18] を読まない尤度が通ってしまった")
except RuntimeError as e:
    check("a35" in str(e), f"θ[18] を読まない尤度は止まる（{str(e)[:40]}…）")

print("\n[6] 2026-09 の事故の再現: 15 次元モデルに 20 次元 θ を渡す")
spec = importlib.util.spec_from_file_location("hm_15d", MAIN / "hamilton_ode_jax.py")
hm15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hm15)
obs, phi0j, idxj = jnp.array(data), jnp.array(p0), jnp.array(idx)


def ll_15d(theta):
    p = hm15.simulate_0d(theta, n_steps=2500, dt=1e-4, phi_init=phi0j, K_hill=1e-8, n_hill=2.0)[
        idxj
    ]
    p = jnp.clip(p, 1e-10, 1.0)
    p = p / p.sum(1, keepdims=True)
    return -0.5 * jnp.sum(((obs - p) / 0.1) ** 2)


try:
    EP.check_sensitivity(ll_15d, pb)
    check(False, "食い違ったモデルが感度チェックを通ってしまった")
except RuntimeError as e:
    dead = re.findall(r"'(\w+)'", str(e))
    check({"a35", "a45"} <= set(dead), f"食い違いを検出（尤度に効かない次元: {dead}）")

print("\n[7] 事前分布が MH 比に入る（初期化を揃えて比較）")
B = np.zeros((20, 2))
B[:, 0], B[:, 1] = -15.0, 20.0
B[EP.B_DIMS] = 0.0


def cheap(theta):
    return -0.5 * ((theta[0] - 8.0) / 1.0) ** 2


def lp(theta):
    return -0.5 * jnp.sum((theta[jnp.array(A_DIMS)] / 3.0) ** 2)


common = {"n_particles": 128, "max_stages": 3, "seed": 1, "verbose": False, "n_mutation_steps": 1}
for mut in ("rw", "nuts"):
    a = E.tmcmc_engine(cheap, B, mutation=mut, log_prior_fn=lp, **common)
    b = E.tmcmc_engine(cheap, B, mutation=mut, **common)
    ma, mb = float(a["samples"][:, 0].mean()), float(b["samples"][:, 0].mean())
    check(ma < mb - 1e-6, f"{mut}: 事前分布あり {ma:+.3f} < なし {mb:+.3f}")

print("\n[8] ln Z <= max logL（beta=1 まで回す）")
res = E.tmcmc_engine(
    cheap,
    B,
    mutation="rw",
    n_particles=256,
    max_stages=40,
    seed=3,
    verbose=False,
    n_mutation_steps=3,
)
bf = float(np.asarray(res["betas"])[-1])
check(bf > 1 - 1e-9, f"beta_final = {bf:.4f}")
check(
    res["log_evidence"] <= res["log_likelihoods"].max() + 1e-9,
    f"ln Z = {res['log_evidence']:.3f} <= max logL = {res['log_likelihoods'].max():.3f}",
)

print("\n" + "=" * 70)
if fails:
    for f in fails:
        print("FAIL", f)
    sys.exit(1)
print("すべて OK。論文パイプラインの前進モデル・エンジン・尤度は論文と一致し、ガードが効く。")
