#!/usr/bin/env python3
"""論文の予測「Fn を除くと Pg の終盤サージが消える」を、事後サンプルで検証する。

なぜ要るか
----------
論文の予測は、未記述の Hill ゲート h(Fn) が保証していた（Fn → 0 なら h → 0 で Pg の相互作用が消える）。
ゲートを外したモデルでは、この予測は推定された相互作用 A だけから出るかどうかが問われる。
そこで事後サンプルごとに 3 通りの前進計算をして、Pg の Day21/Day15 比を比べる:

  baseline : そのまま（run と同じ phi_init・K_hill・n_hill）
  no_Fn    : Fn を系から除く（active_mask[3]=0。共培養から F. nucleatum を抜く実験に対応）
             初期値を 0 にするだけでは Fn が増えて戻るので不可
  a45_0    : a45 = 0 にする（Fn–Pg の相互作用だけを切る。サージが a45 由来かを見る）

読み方:
- baseline でサージが出て、no_Fn と a45_0 で消える → 予測は推定された a45 から出ている（論文が強くなる）
- no_Fn では消えるが a45_0 では消えない → サージは Fn を介した別の経路（a35 など）から出ている
- どちらでも消えない → 「Fn を抜くとサージが消える」はゲートなしのモデルでは予測されない

使い方:
    python3 tools/knockout_fn.py data_5species/main/_runs/paper_gateoff --glob 'DH_*mut80*' \
        [--n-samples 500] [--json out.json]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "data_5species" / "main"
# estimate_paper_jax は import 時に sys.argv の --device を見る（cpu なら GPU を隠す）ので、
# 一時的に差し替える。GPU で回すときは KNOCKOUT_DEVICE=gpu
_argv = sys.argv
sys.argv = [_argv[0], "--device", os.environ.get("KNOCKOUT_DEVICE", "cpu")]
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN.parent))

import jax

jax.config.update("jax_enable_x64", True)
import estimate_paper_jax as EP
import jax.numpy as jnp

sys.argv = _argv
assert Path(EP._H.__file__).name == "hamilton_ode_jax_paper.py", EP._H.__file__

FN, PG, A45 = 3, 4, 19
_H = EP._H


def simulate_masked(theta, active_mask, n_steps, dt, phi_init, K_hill, n_hill):
    """hamilton_ode_jax_paper.simulate_0d と同じ計算で、active_mask だけを指定できる版。

    Fn を「居ない」ことにするには active_mask[3] = 0 が要る。初期値を 0 にするだけでは
    clip で 1e-10 になったあと増えて戻る（このモデルでは phi=0 が吸収状態ではない）。
    active_mask は毎ステップ Fn の phi と psi を 0 にし、ゲート ON のときは h(Fn)=0 になる。
    """
    A, b_diag = _H.theta_to_matrices(theta)
    g0 = _H.make_initial_state(phi_init, active_mask)
    params = {
        "dt_h": dt,
        "Kp1": 1e-4,
        "Eta": jnp.ones(5, dtype=jnp.float64),
        "EtaPhi": jnp.ones(5, dtype=jnp.float64),
        "c": 25.0,
        "alpha": 0.0,
        "K_hill": jnp.array(K_hill, dtype=jnp.float64),
        "n_hill": jnp.array(n_hill, dtype=jnp.float64),
        "A": A,
        "b_diag": b_diag,
        "active_mask": active_mask,
    }

    def body(g, _):
        g_next = _H.newton_step(g, params)
        return g_next, g_next

    _, g_traj = jax.lax.scan(body, g0, jnp.arange(n_steps))
    return jnp.concatenate([g0[0:5][jnp.newaxis, :], g_traj[:, 0:5]], axis=0)


def run_dir(d, n_samples, seed):
    with open(d / "run_record.json") as f:
        rec = json.load(f)
    a = rec["args"]
    S = np.load(d / "samples.npy")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(S), size=min(n_samples, len(S)), replace=False)
    th = jnp.array(S[idx])

    _, t_days, _, _, _ = EP.load_experimental_data(
        EP.DATA_DIR,
        a["condition"],
        a["cultivation"],
        a["start_from_day"],
        normalize=True,
        use_exp_init=a["use_exp_init"],
    )
    _, idx_t = EP.convert_days_to_model_time(t_days, a["dt"], a["n_steps"], day_scale=None)
    idx_t = np.clip(idx_t, 0, a["n_steps"])
    days = np.asarray(t_days).tolist()
    k15, k21 = int(idx_t[days.index(15)]), int(idx_t[days.index(21)])

    phi0 = jnp.array(rec["phi_init"], dtype=jnp.float64)
    mask_all = jnp.ones(5, dtype=jnp.int64)
    mask_noFn = mask_all.at[FN].set(0)

    # 自作の simulate_masked が simulate_0d と同じ計算をしているかを先に確かめる
    ref = EP.simulate_0d(
        th[0],
        n_steps=a["n_steps"],
        dt=a["dt"],
        phi_init=phi0,
        K_hill=a["K_hill"],
        n_hill=a["n_hill"],
    )
    mine = simulate_masked(th[0], mask_all, a["n_steps"], a["dt"], phi0, a["K_hill"], a["n_hill"])
    dmax = float(jnp.max(jnp.abs(ref - mine)))
    if dmax > 1e-12:
        raise RuntimeError(f"{d.name}: simulate_masked が simulate_0d と一致しない（{dmax:.2e}）")

    def sim(theta, mask):
        traj = simulate_masked(theta, mask, a["n_steps"], a["dt"], phi0, a["K_hill"], a["n_hill"])
        p = jnp.clip(traj[jnp.array([k15, k21])], 1e-10, 1.0)
        p = p / p.sum(axis=1, keepdims=True)  # 尤度・RMSE と同じ正規化
        return p[1, PG] / p[0, PG], p[1, PG], traj[k21, FN]

    f = jax.jit(jax.vmap(sim, in_axes=(0, None)))
    base = [np.asarray(x) for x in f(th, mask_all)]
    nofn = [np.asarray(x) for x in f(th, mask_noFn)]
    a45z = [np.asarray(x) for x in f(th.at[:, A45].set(0.0), mask_all)]

    def q(x):
        return [float(v) for v in np.percentile(x, [5, 50, 95])]

    return {
        "run": d.name,
        "K_hill": a["K_hill"],
        "n": len(idx),
        "ratio_base": q(base[0]),
        "ratio_noFn": q(nofn[0]),
        "ratio_a45_0": q(a45z[0]),
        "pgD21_base": q(base[1]),
        "pgD21_noFn": q(nofn[1]),
        # サージが「消えた」= Day21/Day15 比が 1.5 未満（実測 2.74、ゲートなしの論文 MAP で 0.90）
        "frac_surge_base": float(np.mean(base[0] >= 1.5)),
        "frac_surge_noFn": float(np.mean(nofn[0] >= 1.5)),
        "frac_surge_a45_0": float(np.mean(a45z[0] >= 1.5)),
        "fn_D21_noFn_max": float(np.max(nofn[2])),  # Fn が本当に居ないことの確認
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(MAIN / "_runs" / "paper_gateoff"))
    ap.add_argument("--glob", default="DH_*mut80*")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dirs = [d for d in sorted(Path(args.root).glob(args.glob)) if (d / "run_record.json").exists()]
    if not dirs:
        print(f"対象の run が無い: {args.root}/{args.glob}")
        return 1
    print(f"JAX devices: {jax.devices()}")
    rows = [run_dir(d, args.n_samples, args.seed) for d in dirs]

    def fmt(q):
        return f"{q[1]:5.2f} [{q[0]:5.2f},{q[2]:5.2f}]"

    print(
        f"{'run':<40}{'K':>5}  {'Pg D21/D15 そのまま':>20}  {'Fn を除く':>20}  {'a45=0':>20}"
        f"  {'サージ(≥1.5)の割合 そのまま/Fn除く/a45=0':>12}"
    )
    for r in rows:
        print(
            f"{r['run']:<40}{r['K_hill']:>5.2f}  {fmt(r['ratio_base']):>20}  {fmt(r['ratio_noFn']):>20}"
            f"  {fmt(r['ratio_a45_0']):>20}  {r['frac_surge_base']:5.0%} / {r['frac_surge_noFn']:5.0%}"
            f" / {r['frac_surge_a45_0']:5.0%}"
        )
    bad = [r["run"] for r in rows if r["fn_D21_noFn_max"] > 1e-4]
    if bad:
        print(f"\n警告: Fn を除いたのに Day21 で Fn が 1e-4 を超えて戻っている run: {bad}")
    print("\n比は事後サンプルの中央値 [5%, 95%]。実測の Pg D21/D15 は DH 2.74。")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
