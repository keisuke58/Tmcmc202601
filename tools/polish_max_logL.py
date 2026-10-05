#!/usr/bin/env python3
"""run の上位粒子から尤度を局所最大化し、「本当の max logL」を run 間で比べる。

なぜ要るか（判定 5 の読み方）
---------------------------
判定 5 は「事前分布なしの run の max logL >= 事前分布ありの run の max logL − 0.5」。
事前分布なしの run が主要なモードに届いていなければ、ここで落ちる。

ただし max logL は **事後からの 2000 粒子のうち最良のもの**で、尤度の最大値そのものではない。
平らな事前分布で箱 [−15, 20]^15 を覆うと、事後の質量は尤度がやや低い広い領域にも広がるので
（体積の効果）、同じモードに届いていても粒子の最良値は数 nats 下がりうる。

そこで各 run の上位粒子から L-BFGS-B で尤度を局所最大化し（上位 K 粒子をまとめて 1 回の最適化で）、磨いた値どうしを比べる:
- 事前分布なしの磨いた値 ≈ 事前分布ありの磨いた値 → 同じモードに届いている。判定 5 の差は体積の効果
- 事前分布なしの磨いた値 < 事前分布ありの磨いた値 → 事前分布なしは主要なモードに届いていない

尤度は run_record.json の実効値（sigma_obs・phi_init・psi_fixed・K_hill など）から組み直し、
**保存した logL.npy を再現できることを確かめてから**最大化する（再現できなければ止まる）。
対象は ψ 固定・単一チャネルの run（pilot / ident）だけ。

使い方:
    python3 tools/polish_max_logL.py data_5species/main/_runs/paper_gateoff --glob '*ident*mut80*'
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "data_5species" / "main"
# estimate_paper_jax は import 時に sys.argv の --device を見る（cpu なら GPU を隠す）ので、
# 一時的に差し替える。GPU で回すときは POLISH_DEVICE=gpu（polish_max_logL_job.sh が渡す）
_argv = sys.argv
sys.argv = [_argv[0], "--device", os.environ.get("POLISH_DEVICE", "cpu")]
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN.parent))

import jax

jax.config.update("jax_enable_x64", True)
import estimate_paper_jax as EP
import jax.numpy as jnp

sys.argv = _argv
assert Path(EP._H.__file__).name == "hamilton_ode_jax_paper.py", EP._H.__file__


def build_loglik(rec):
    a = rec["args"]
    if a.get("multichannel"):
        raise ValueError("多チャネルの run は対象外（pilot / ident だけ）")
    data, t_days, _sig, _phi, _ = EP.load_experimental_data(
        EP.DATA_DIR,
        a["condition"],
        a["cultivation"],
        a["start_from_day"],
        normalize=True,
        use_exp_init=a["use_exp_init"],
    )
    _, idx = EP.convert_days_to_model_time(t_days, a["dt"], a["n_steps"], day_scale=None)
    psi = rec.get("psi_fixed")
    return EP.make_log_likelihood_jax_ode(
        data=data,
        t_days=t_days,
        idx_sparse=np.clip(idx, 0, a["n_steps"]),
        sigma_obs=np.asarray(rec["sigma_obs"]),
        phi_init=np.asarray(rec["phi_init"]),
        dt=a["dt"],
        n_steps=a["n_steps"],
        K_hill=a["K_hill"],
        n_hill=a["n_hill"],
        lambda_pg=a["lambda_pg"],
        lambda_late=a["lambda_late"],
        use_student_t=a["use_student_t"],
        student_t_nu=a["student_t_nu"],
        lambda_rare=a["lambda_rare"],
        psi_fixed=None if psi is None else np.asarray(psi),
        lambda_ch={},  # ψ 固定・単一チャネル（estimator の mc_kwargs と同じ）
    )


def polish(d, top_k, maxiter):
    with open(d / "run_record.json") as f:
        rec = json.load(f)
    S, L = np.load(d / "samples.npy"), np.load(d / "logL.npy")
    ll = build_loglik(rec)
    ll_vmap = jax.jit(jax.vmap(ll))

    # 尤度を正しく組み直せたか: 上位 200 粒子で保存 logL を再現する
    order = np.argsort(-L)
    chk = order[:200]
    diff = float(np.max(np.abs(np.asarray(ll_vmap(jnp.array(S[chk]))) - L[chk])))
    if diff > 1e-6:
        raise RuntimeError(
            f"{d.name}: 保存 logL を再現できない（max|差| = {diff:.2e}）。比較しない"
        )

    free = np.array(rec["free_dims"])
    pb = np.array(rec["prior_bounds_final"], dtype=float)
    top = order[:top_k]
    base = np.array(S[top])  # (K, 20)。固定次元はこの値のまま
    K, nf = len(top), len(free)

    # K 粒子をまとめて 1 回の L-BFGS-B で磨く。目的関数は粒子ごとに分かれた和なので、
    # 各粒子の最大化と同じ解に向かう。1 回の評価は vmap 1 回（GPU なら 1 粒子とほぼ同じ時間）
    def neg_sum(x):
        th = base_j.at[:, free_j].set(x.reshape(K, nf))
        return -jnp.sum(jax.vmap(ll)(th))

    base_j, free_j = jnp.array(base), jnp.array(free)
    vg = jax.jit(jax.value_and_grad(neg_sum))

    def f(x):
        v, g = vg(jnp.array(x))
        return float(v), np.asarray(g, dtype=np.float64)

    bounds = [tuple(pb[i]) for i in free] * K
    x0 = base[:, free].ravel()
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    th = base.copy()
    th[:, free] = r.x.reshape(K, nf)
    vals = np.asarray(ll_vmap(jnp.array(th)))
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    k = int(np.argmax(vals))
    best, best_theta = float(vals[k]), th[k]
    return {
        "run": d.name,
        "sample_max": float(L.max()),
        "polished_max": float(best),
        "gain": float(best - L.max()),
        "recheck": diff,
        "n_iter": int(r.nit),
        "converged": bool(r.success),
        "theta": best_theta.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(MAIN / "_runs" / "paper_gateoff"))
    ap.add_argument("--glob", default="*ident*")
    ap.add_argument("--top-k", type=int, default=20, help="磨く上位粒子の数")
    ap.add_argument("--maxiter", type=int, default=300, help="L-BFGS-B の反復の上限")
    ap.add_argument("--json", default=None, help="結果（磨いた θ を含む）の書き出し先")
    args = ap.parse_args()

    dirs = [d for d in sorted(Path(args.root).glob(args.glob)) if (d / "run_record.json").exists()]
    if not dirs:
        print(f"対象の run が無い: {args.root}/{args.glob}")
        return 1
    print(f"JAX devices: {jax.devices()}")
    rows = [polish(d, args.top_k, args.maxiter) for d in dirs]

    print(f"{'run':<40}{'粒子の最良':>11}{'磨いた値':>11}{'差':>8}{'反復':>6}{'収束':>5}")
    for r in rows:
        print(
            f"{r['run']:<40}{r['sample_max']:>11.3f}{r['polished_max']:>11.3f}{r['gain']:>8.3f}"
            f"{r['n_iter']:>6}{'yes' if r['converged'] else 'no':>5}"
        )

    # 判定 5 を磨いた値で読み直す（事前分布以外が同じ群どうし）
    def group(name):
        return re.sub(r"_seed\d+$", "", name)

    def prior_free(key):
        return re.sub(r"_ident_prior[0-9.]+", "_ident_prior*", key)

    best, conv = {}, {}
    for r in rows:
        g = group(r["run"])
        best[g] = max(best.get(g, -np.inf), r["polished_max"])
        conv[g] = conv.get(g, True) and r["converged"]
    print()
    for g0, m0 in best.items():
        if "_ident_prior0" not in g0:
            continue
        for g1, m1 in best.items():
            if g1 != g0 and prior_free(g1) == prior_free(g0):
                if not (conv[g0] and conv[g1]):
                    print(
                        f"保留 5（磨いた値） {g0} {m0:.3f} / {g1} {m1:.3f}: L-BFGS-B が収束していない run が"
                        " ある。--maxiter を上げて回し直す（収束していない値では判定しない）"
                    )
                    continue
                ok = m0 >= m1 - 0.5
                print(
                    f"{'PASS' if ok else 'FAIL'} 5（磨いた値） {g0} {m0:.3f} >= {g1} {m1:.3f} − 0.5"
                    + ("" if ok else "  → 事前分布なしは主要なモードに届いていない")
                )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
