#!/usr/bin/env python3
"""2 通路のパイロット v2（2026-10-13c）: 推定と同じ尤度・実際の箱で −logL を最小化する。

v1（two_channel_pilot.py）は重みなしの MSE の和を全次元 [-15, 20] で最小化していたので、
GPU の推定とは別の問題を解いていた。v2 は
  - 目的関数: estimate_paper_jax.make_log_likelihood_jax_ode(two_channel=True, ψ 自由)
    （λ_Pg 5・終盤 ×3・珍しい種 ×0.1・種ごとの σ、λ_ch = {1:1, 2:1, 3: 条件ごと, 5:0}）
  - 箱: load_prior_bounds（論文の箱）＋ 13b の p2 の OVERRIDE（既定は 13b の表。--override で変更可）
  - 初期値: 今の最終 MAP（v1 と同じ）
を使い、−logL を L-BFGS-B（箱つき）で最小化する。出力は v1 と同じ形（RMSE の内訳、ψ の中央値）に
`neg_logL` を足す。tools/two_channel_pilot_table.py でそのまま表にできる。

    python3 tools/two_channel_pilot_v2.py --tag CH [--restarts 2 --maxiter 250 --out x.json]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OVERRIDE_13B = {
    "CH": "1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5",
    "CS": "5:-15:20;0:-5:2;10:-1:20",
    "DS": "5:-15:20;6:-15:20;19:-15:20;14:-5:4;12:-5:3;16:-5:2.5",
    "DH": "12:-15:20;5:-15:20;6:-15:20;18:-25:20;13:-15:20;0:-5:5;11:-8:3;19:-5:20;1:-5:3;2:-10:5;7:-10:5;10:-8:3",
}
LAMBDA_CH3 = {"CS": 1.5, "CH": 1.5, "DS": 2.0, "DH": 3.0}  # estimate_paper_jax の自動調整と同じ（Commensal は 1.5、DH は 3.0。2026-10-13d: CS を 2.0 → 1.5 に訂正）

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, choices=["CS", "CH", "DS", "DH"])
ap.add_argument("--override", default=None, help="idx:lo:hi;... 既定は 13b の p2 の OVERRIDE")
ap.add_argument("--lambda-ch2", type=float, default=1.0)
# 重みの感度（2026-10-13）。既定は推定と同じ
ap.add_argument("--lambda-pg", type=float, default=5.0)
ap.add_argument("--lambda-late", type=float, default=3.0)
ap.add_argument("--lambda-rare", type=float, default=0.1)
ap.add_argument("--data-source", choices=["digitised", "raw"], default="raw")
ap.add_argument("--lambda-ch3", type=float, default=None, help="既定は条件ごとの自動調整（LAMBDA_CH3）")
ap.add_argument("--restarts", type=int, default=2)
ap.add_argument("--maxiter", type=int, default=250)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
ap.add_argument("--rescore", default=None, help="保存した JSON の theta_opt で内訳だけ出し直す（最適化しない）。--out に書く")
args = ap.parse_args()
sys.argv = [sys.argv[0], "--device", "cpu"]
sys.path.insert(0, str(ROOT / "data_5species" / "main"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import estimate_paper_jax as E  # noqa: E402
E.DATA_DIR = E.DATA_DIRS[args.data_source]

COND = {"CS": ("Commensal", "Static"), "CH": ("Commensal", "HOBIC"),
        "DS": ("Dysbiotic", "Static"), "DH": ("Dysbiotic", "HOBIC")}[args.tag]
MAPJ = ROOT / f"docs/revision/generated/final_theta_MAP/{args.tag}.json"
if MAPJ.exists():
    d = json.load(open(MAPJ))
else:  # DS・DH は段 (iii) の群で max logL の seed
    G = {"DS": "DS_p2_wide80_p18k_wide2_seed*", "DH": "DH_p2_nonarrow_w3_noph_p14k_seed*"}[args.tag]
    runs = sorted(glob.glob(str(ROOT / "docs/revision/generated/final_runs" / G)))
    best = max(runs, key=lambda r: json.load(open(Path(r) / "run_record.json"))["max_logL"])
    d = json.load(open(Path(best) / "theta_MAP.json"))
th0 = np.array([d[str(i)] for i in range(20)])

data, t_days, sig, phi0, _ = E.load_experimental_data(E.DATA_DIR, *COND, 1, normalize=True, use_exp_init=True)
_, idx = E.convert_days_to_model_time(t_days, 1e-4, 2500, day_scale=None)
idx = np.clip(idx, 0, 2500)
phi0 = np.clip(phi0 / phi0.sum(), 0.001, 0.99)
mc = E.load_multichannel_data(data_dir=E.DATA_DIR / "experiment_data", condition=COND[0],
                              cultivation=COND[1], days_filter=t_days.tolist())
lam = {1: 1.0, 2: args.lambda_ch2, 3: LAMBDA_CH3[args.tag] if args.lambda_ch3 is None else args.lambda_ch3, 5: 0.0}
logL = E.make_log_likelihood_jax_ode(
    data=data, t_days=t_days, idx_sparse=idx, sigma_obs=sig, phi_init=phi0, dt=1e-4, n_steps=2500,
    K_hill=0.0, n_hill=4.0, lambda_pg=args.lambda_pg, lambda_late=args.lambda_late, lambda_rare=args.lambda_rare,
    data_total=mc["data_total"], sigma_obs_total=mc["sigma_obs_total"],
    data_viability=mc["data_viability"], sigma_obs_viability=mc["sigma_obs_viability"],
    lambda_ch=lam, two_channel=True)

# 箱: 論文の箱 + OVERRIDE（estimate_paper_jax と同じ規則。固定次元は上書きしない）
bounds = np.array(E.load_prior_bounds(*COND), dtype=float)
ovr = OVERRIDE_13B[args.tag] if args.override is None else args.override
for spec in ovr.replace(";", ",").split(","):
    if not spec:
        continue
    i, lo, hi = spec.split(":")
    i = int(i)
    if abs(bounds[i, 1] - bounds[i, 0]) > 1e-12:
        bounds[i] = [float(lo), float(hi)]
free = [i for i in range(20) if i not in E.B_DIMS and abs(bounds[i, 1] - bounds[i, 0]) > 1e-12]
th0[free] = np.clip(th0[free], bounds[free, 0], bounds[free, 1])

# v1 と同じ重みなしの RMSE の内訳（比較用）
Yv = jnp.array(data)
Yt = jnp.array(mc["data_total"])
Yt = Yt / Yt.sum(1, keepdims=True)
L = jnp.array(mc["data_viability"])
# load_experimental_data(use_exp_init=True) は Day 1 を既に外している（観測は Day 3〜21 の 5 行）。[1:] は Day 3 を落としていた（2026-10-13 修正）
k = jnp.array(idx)
PSI_D = E.species_viability_from_data(COND[0], COND[1], t_days, mc["data_viability"])
_tot = np.asarray(mc["data_total"], dtype=float)
_tot = _tot / _tot.sum(1, keepdims=True)
PSI_M = jnp.array((_tot >= 0.02).astype(float))
PSI_D = jnp.array(PSI_D)


def parts(th):
    g = E.simulate_0d_full(th, n_steps=2500, dt=1e-4, phi_init=jnp.array(phi0), K_hill=0.0, n_hill=4.0)
    phi = jnp.clip(g[k, 0:5], 1e-10, 1)
    pb = jnp.clip(phi * g[k, 6:11], 1e-12, 1)
    n = lambda x: x / x.sum(1, keepdims=True)  # noqa: E731
    return {"viable_vs_phi": jnp.mean((Yv - n(phi)) ** 2),
            "viable_vs_phipsi": jnp.mean((Yv - n(pb)) ** 2),
            "total_vs_phi": jnp.mean((Yt - n(phi)) ** 2),
            "live": jnp.mean((L - pb.sum(1) / phi.sum(1)) ** 2),
            "psi_vs_data": jnp.sum(PSI_M * (g[k, 6:11] - PSI_D) ** 2) / jnp.sum(PSI_M)}


def report(name, th):
    p = {k_: float(np.sqrt(v)) for k_, v in parts(jnp.array(th)).items()}
    p["neg_logL"] = -float(logL(jnp.array(th)))
    print(f"{name}: -logL {p['neg_logL']:.3f}  RMSE viable~phi {p['viable_vs_phi']:.3f}  "
          f"viable~phipsi {p['viable_vs_phipsi']:.3f}  total~phi {p['total_vs_phi']:.3f}  "
          f"live {p['live']:.3f}  psi~data {p['psi_vs_data']:.3f}", flush=True)
    return p


print(f"{args.tag}: lambda_ch {lam}, override '{ovr}', free dims {free}", flush=True)
out = {"tag": args.tag, "map": report(f"{args.tag} MAP (now, under 2c logL)", th0)}
if args.rescore:
    _old = json.load(open(args.rescore))
    out = dict(_old, map=out["map"], two_channel_opt=report(f"{args.tag} rescored optimum", np.array(_old["theta_opt"])), rescored=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    sys.exit(0)
vg = jax.jit(jax.value_and_grad(lambda th: -logL(th)))


def obj(z):
    th = th0.copy()
    th[free] = z
    v, g = vg(jnp.array(th))
    v = float(v)
    return (1e12, np.zeros_like(z)) if not np.isfinite(v) else (v, np.nan_to_num(np.array(g)[free]))


rng = np.random.default_rng(args.seed)
best = None
width = bounds[free, 1] - bounds[free, 0]
for r in range(args.restarts):
    z0 = th0[free] + (0 if r == 0 else rng.normal(0, 0.1, len(free)) * width)
    z0 = np.clip(z0, bounds[free, 0], bounds[free, 1])
    res = minimize(obj, z0, jac=True, method="L-BFGS-B", bounds=[tuple(bounds[i]) for i in free],
                   options={"maxiter": args.maxiter})
    th = th0.copy()
    th[free] = res.x
    report(f"  restart {r} (nit {res.nit}, {res.message})", th)
    if best is None or res.fun < best.fun:
        best, bth = res, th
out["two_channel_opt"] = report(f"{args.tag} two-channel optimum (2c logL)", bth)
g = np.array(E.simulate_0d_full(jnp.array(bth), n_steps=2500, dt=1e-4, phi_init=jnp.array(phi0), K_hill=0.0, n_hill=4.0))
out["psi_opt_median_days"] = np.median(g[idx, 6:11], axis=0).round(3).tolist()
out["psi_data_median_days"] = np.median(np.array(PSI_D), axis=0).round(3).tolist()
out["theta_opt"] = bth.round(4).tolist()
_j15, _j21 = int(np.where(np.asarray(t_days) == 15)[0][0]), int(np.where(np.asarray(t_days) == 21)[0][0])
_fr = g[:, 0:5] / g[:, 0:5].sum(1, keepdims=True)
out["pg_ratio_21_15"] = float(_fr[idx[_j21], 4] / max(_fr[idx[_j15], 4], 1e-12))
# 箱の端（幅の 5% 以内）にある自由次元
edge = [i for i in free if min(bth[i] - bounds[i, 0], bounds[i, 1] - bth[i]) < 0.05 * (bounds[i, 1] - bounds[i, 0])]
out["at_edge"] = edge
out["bounds"] = bounds.round(4).tolist()
out["args"] = {"version": 2, "data_source": args.data_source, "lambda_ch": lam, "lambda_pg": args.lambda_pg, "lambda_late": args.lambda_late,
               "lambda_rare": args.lambda_rare, "override": ovr, "restarts": args.restarts,
               "maxiter": args.maxiter, "seed": args.seed}
print("psi (median over days):", out["psi_opt_median_days"], " data:", out["psi_data_median_days"],
      " at_edge:", edge, flush=True)
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
