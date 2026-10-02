#!/usr/bin/env python3
"""9/30 のゲート OFF run で logL.npy と samples.npy が対応していない件の検証。

背景（2026-10-02）:
  dh_gateoff_sigma6_5000p_seed7_cpualign_20260930 で、theta_MAP.json
  （= samples[argmax(logL.npy)]）を run 自身の尤度で再評価すると -33.33 になるが、
  run が記録した max logL は -6.511。300 粒子で記録 logL と再計算 logL の相関は
  r=0.17（noprior は -0.11）で、ほぼ無相関。並べ替えでも初期ドローでもなく、
  尤度設定 24 通りの総当たりでも -6.511 は再現しない。

やること:
  1. 各 run の samples.npy 全粒子で logL を再計算し、記録値と突き合わせる
  2. 真の最良粒子（再計算 logL が最大）の RMSE と Pg Day21/Day15 を出す
  3. 結果を JSON と CSV に書く

使い方:
  python tools/recheck_gateoff_logL.py [--glob '<_runs のパターン>'] [--out <接頭辞>]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / "data_5species" / "main"
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(REPO / "colab_package"))

import estimate_reduced_nishioka_jax as R
from estimate_reduced_nishioka import (
    convert_days_to_model_time,
    load_experimental_data,
)
from hamilton_ode_jax import simulate_0d

CODE2COND = {
    "dh": ("Dysbiotic", "HOBIC"),
    "ds": ("Dysbiotic", "Static"),
    "ch": ("Commensal", "HOBIC"),
    "cs": ("Commensal", "Static"),
}
DT, N_STEPS, C_CONST = 1e-4, 2500, 25.0
_cache = {}


def setup(code):
    """run と同じ尤度を組む。config.json に記録された設定に合わせてある。"""
    if code in _cache:
        return _cache[code]
    cond, cult = CODE2COND[code]
    data, t_days, sigma, phi_exp, _ = load_experimental_data(
        MAIN.parent, cond, cult, 1, normalize=True
    )
    phi0 = np.clip(phi_exp / phi_exp.sum(), 0.01, 0.99)
    _, idx = convert_days_to_model_time(t_days, DT, N_STEPS, day_scale=None)
    idx = np.clip(idx, 0, N_STEPS)
    ll = R.make_log_likelihood_jax_ode(
        data=data,
        t_days=t_days,
        idx_sparse=idx,
        sigma_obs=sigma,
        phi_init=phi0,
        dt=DT,
        n_steps=N_STEPS,
        K_hill=0.0,
        n_hill=2.0,
        lambda_pg=1.0,
        lambda_late=1.0,
        sign_prior=False,
        sign_lambda=0.1,
        lambda_bc=0.0,
    )
    _cache[code] = (data, t_days, phi0, idx, np.asarray(sigma), jax.jit(jax.vmap(ll)))
    return _cache[code]


def forward(theta, phi0, idx):
    traj = simulate_0d(
        jnp.array(theta),
        n_steps=N_STEPS,
        dt=DT,
        phi_init=jnp.array(phi0),
        K_hill=0.0,
        n_hill=2.0,
        c_const=C_CONST,
    )
    pred = np.asarray(traj)[idx, :]
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    return pred / pred.sum(axis=1, keepdims=True)


def metrics(theta, data, t_days, phi0, idx, sigma):
    pred = forward(theta, phi0, idx)
    res = data - pred
    k15, k21 = t_days.tolist().index(15), t_days.tolist().index(21)
    return {
        "rmse": float(np.sqrt(np.mean(res**2))),
        "chi": float(np.sqrt(np.mean((res / sigma) ** 2))),
        "pg_ratio": float(pred[k21, 4] / max(pred[k15, 4], 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*_gateoff_*_cpualign_20260930")
    ap.add_argument("--out", default=str(MAIN / "_runs" / "logL_recheck"))
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()

    runs = sorted((MAIN / "_runs").glob(args.glob))
    print(f"{len(runs)} run を検証", flush=True)
    hdr = (
        f"{'run':52}{'記録max':>9}{'再計算max':>10}{'MAPの再計算':>12}"
        f"{'相関r':>8}{'真RMSE':>8}{'MAP RMSE':>9}{'真chi':>7}{'真D21/15':>9}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    rows = []
    for d in runs:
        code = d.name.split("_")[0]
        data, t_days, phi0, idx, sigma, f = setup(code)
        S = np.load(d / "samples.npy")
        L = np.load(d / "logL.npy")
        new = np.concatenate(
            [np.asarray(f(jnp.array(S[i : i + args.chunk]))) for i in range(0, len(S), args.chunk)]
        )
        i_true, i_rec = int(np.argmax(new)), int(np.argmax(L))
        r = float(np.corrcoef(L, new)[0, 1])
        mt = metrics(S[i_true], data, t_days, phi0, idx, sigma)
        mm = metrics(S[i_rec], data, t_days, phi0, idx, sigma)
        row = {
            "run": d.name,
            "cond": code,
            "arm": d.name.split("_")[2],
            "seed": int(d.name.split("seed")[1].split("_")[0]),
            "recorded_max_logL": float(L.max()),
            "recomputed_max_logL": float(new.max()),
            "recomputed_logL_at_saved_MAP": float(new[i_rec]),
            "corr_recorded_recomputed": r,
            "idx_true_best": i_true,
            "idx_saved_MAP": i_rec,
            "rmse_true_best": mt["rmse"],
            "chi_true_best": mt["chi"],
            "pg_ratio_true_best": mt["pg_ratio"],
            "rmse_saved_MAP": mm["rmse"],
            "chi_saved_MAP": mm["chi"],
            "pg_ratio_saved_MAP": mm["pg_ratio"],
            "theta_true_best": {str(j): float(v) for j, v in enumerate(S[i_true])},
        }
        rows.append(row)
        print(
            f"{d.name:52}{L.max():9.3f}{new.max():10.3f}{new[i_rec]:12.3f}"
            f"{r:8.3f}{mt['rmse']:8.4f}{mm['rmse']:9.4f}{mt['chi']:7.2f}{mt['pg_ratio']:9.2f}",
            flush=True,
        )
        with open(args.out + ".json", "w") as fj:
            json.dump(rows, fj, indent=1)

    keys = [k for k in rows[0] if k != "theta_true_best"]
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n書き出し: {args.out}.json / {args.out}.csv", flush=True)


if __name__ == "__main__":
    main()
