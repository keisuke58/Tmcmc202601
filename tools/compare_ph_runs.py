#!/usr/bin/env python3
"""pH を尤度に入れた run と入れない run を、pH の項を除いた尤度で比べる（2026-10-08s）。

なぜ要るか
---------
2026-10-08 に pH チャネル（ch5、重み 0.3）を推定から外した（`2026-10-08r.md`）。
pH あり の run と pH なし の run の `max_logL` はそのままでは比べられない（項の数が違う）。
ここでは **pH あり の run の粒子を、pH の項を落とした尤度で読み直し**、
組成（ch1）＋全量（ch2）＋生存率（ch3）の部分だけで max logL を比べる。

尤度は run_record.json の実効値（sigma_obs・phi_init・lambda_ch_effective など）から組み直し、
**保存した logL.npy を再現できることを確かめてから** pH を落とす（再現できなければその run は比べない）。
pH なし の run では λ5 がもともと 0 なので、落とす前後で値は変わらない（確認用に両方出す）。

RMSE と生存率の当てはまりは config.json の `multichannel_rmse` をそのまま読む（再計算しない）。

使い方:
    python3 tools/compare_ph_runs.py \
        DH_p2_nonarrow_a11 DH_p2_nonarrow_a45_noph CH_p2_mut150_wide CH_p2_mut150_wide_noph
    （群の名前を seed 抜きで渡す。--root で run の置き場所を変えられる）
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "data_5species" / "main"
# estimate_paper_jax は import 時に sys.argv の --device を見る（polish_max_logL.py と同じ作法）
_argv = sys.argv
sys.argv = [_argv[0], "--device", os.environ.get("COMPARE_DEVICE", "cpu")]
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN.parent))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import estimate_paper_jax as EP  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.argv = _argv
assert Path(EP._H.__file__).name == "hamilton_ode_jax_paper.py", EP._H.__file__

A45 = 19  # θ の並びでの a45


def build_loglik(rec, drop_pH):
    """run_record の実効値から多チャネル尤度を組み直す。drop_pH なら λ5 = 0 にする。"""
    a = rec["args"]
    data, t_days, _sig, _phi, _ = EP.load_experimental_data(
        EP.DATA_DIR,
        a["condition"],
        a["cultivation"],
        a["start_from_day"],
        normalize=True,
        use_exp_init=a["use_exp_init"],
    )
    _, idx = EP.convert_days_to_model_time(t_days, a["dt"], a["n_steps"], day_scale=None)
    lam = {int(k): float(v) for k, v in (rec.get("lambda_ch_effective") or {}).items()}
    mc_kwargs = {}
    if a.get("multichannel"):
        mc = EP.load_multichannel_data(
            data_dir=EP.DATA_DIR / "experiment_data",
            condition=a["condition"],
            cultivation=a["cultivation"],
            days_filter=list(t_days),
        )
        if drop_pH:
            lam[5] = 0.0
        mc_kwargs = {
            "data_total": mc.get("data_total"),
            "sigma_obs_total": mc.get("sigma_obs_total"),
            "data_viability": mc.get("data_viability"),
            "sigma_obs_viability": mc.get("sigma_obs_viability", 0.10),
            "lambda_ch": lam,
        }
        # pH の項は λ5 = 0 でも渡しておく（estimator と同じ形の尤度にするため）
        if mc.get("data_pH") is not None:
            _, idx_pH = EP.convert_days_to_model_time(
                mc["t_pH_days"], a["dt"], a["n_steps"], day_scale=None
            )
            mc_kwargs["data_pH"] = mc["data_pH"]
            mc_kwargs["idx_pH"] = np.clip(idx_pH, 0, a["n_steps"])
            mc_kwargs["sigma_obs_pH"] = mc.get("sigma_obs_pH", 0.15)
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
        **mc_kwargs,
    )


def row_for(d, top_k):
    with open(d / "run_record.json") as f:
        rec = json.load(f)
    with open(d / "config.json") as f:
        cfg = json.load(f)
    S, L = np.load(d / "samples.npy"), np.load(d / "logL.npy")
    order = np.argsort(-L)
    top = order[: min(top_k, len(order))]

    # 1) 保存 logL を再現できるか（pH の項を含めたまま）
    ll = jax.jit(jax.vmap(build_loglik(rec, drop_pH=False)))
    got = np.asarray(ll(jnp.array(S[top])))
    diff = float(np.max(np.abs(got - L[top])))
    if diff > 1e-6:
        raise RuntimeError(f"{d.name}: 保存 logL を再現できない（max|差| = {diff:.2e}）。比べない")

    # 2) pH の項を落として読み直す
    ll0 = jax.jit(jax.vmap(build_loglik(rec, drop_pH=True)))
    vals = np.asarray(ll0(jnp.array(S[top])))
    vals = np.where(np.isfinite(vals), vals, -np.inf)

    mcr = cfg.get("multichannel_rmse") or {}
    lam5 = float((rec.get("lambda_ch_effective") or {}).get("5", 0.0))
    q = np.percentile(S[:, A45], [5, 50, 95])
    return {
        "run": d.name,
        "lambda_pH": lam5,
        "n_part": cfg.get("n_particles"),
        "rmse_ch1": mcr.get("ch1_species", {}).get("rmse", cfg.get("rmse")),
        "rmse_ch3": mcr.get("ch3_viability", {}).get("rmse"),
        "r2_ch3": mcr.get("ch3_viability", {}).get("r2"),
        "rmse_ch5": mcr.get("ch5_pH", {}).get("rmse"),
        "max_logL": float(L.max()),
        "max_logL_noph": float(vals.max()),
        "a45_p05": float(q[0]),
        "a45_p50": float(q[1]),
        "a45_p95": float(q[2]),
        "recheck": diff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="seed 抜きの群の名前（例 DH_p2_nonarrow_a11）")
    ap.add_argument("--root", default=str(MAIN / "_runs" / "paper_gateoff"))
    ap.add_argument("--top-k", type=int, default=200, help="読み直す上位粒子の数")
    args = ap.parse_args()

    root = Path(args.root)
    print(
        f"{'run':44s} {'λ_pH':>5s} {'粒子':>5s} {'RMSE(組成)':>10s} "
        f"{'RMSE(生存率)':>12s} {'R²(生存率)':>10s} {'maxlogL':>9s} "
        f"{'maxlogL(pH 抜き)':>16s} {'a45 5/50/95%':>24s}"
    )
    missing = []
    for g in args.groups:
        dirs = sorted(root.glob(f"{g}_seed*"))
        dirs = [d for d in dirs if (d / "samples.npy").exists()]
        if not dirs:
            missing.append(g)
            print(f"{g:44s}   -- run が無い（未完了）")
            continue
        for d in dirs:
            r = row_for(d, args.top_k)
            print(
                f"{r['run']:44s} {r['lambda_pH']:5.1f} {r['n_part'] or 0:5d} "
                f"{r['rmse_ch1']:10.4f} "
                f"{(r['rmse_ch3'] if r['rmse_ch3'] is not None else float('nan')):12.4f} "
                f"{(r['r2_ch3'] if r['r2_ch3'] is not None else float('nan')):10.3f} "
                f"{r['max_logL']:9.2f} {r['max_logL_noph']:16.2f} "
                f"{r['a45_p50']:+8.2f} [{r['a45_p05']:+.2f}, {r['a45_p95']:+.2f}]"
            )
    if missing:
        print("\n未完了（run が無い）: " + ", ".join(missing))
    print(
        "\nmaxlogL(pH 抜き) は上位 "
        f"{args.top_k} 粒子を λ_pH=0 の尤度で読み直した最大値。"
        "λ_pH=0 の run では maxlogL と一致するのが正しい。"
    )


if __name__ == "__main__":
    main()
