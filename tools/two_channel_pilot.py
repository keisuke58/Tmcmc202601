#!/usr/bin/env python3
"""2 通路の組成（生きた菌 ↔ φψ、全菌 ↔ φ）で当てられるかの CPU パイロット（2026-10-13）。

いまの尤度: 生きた菌の組成（PMA）を φ/Σφ と比べ、群集の生きた割合を Σφψ/Σφ と比べる。
2 通路案:   生きた菌の組成を φψ/Σφψ、全菌の組成（Heine Fig. S1）を φ/Σφ、生きた割合はそのまま。

最終 MAP から始めて、2 通路の二乗誤差の和
    J = MSE(生きた菌, φψ/Σφψ) + MSE(全菌, φ/Σφ) + MSE(生きた割合, Σφψ/Σφ)
を勾配法（L-BFGS-B、箱 [lo, hi]）で最小化し、いまの MAP と並べて出す。事後分布ではなく、
「2 通路で同時に合わせられる θ があるか」の目安。fit_ceiling.py と同じ作法。

    python3 tools/two_channel_pilot.py --tag CH [--restarts 2 --maxiter 150]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, choices=["CS", "CH", "DS", "DH"])
ap.add_argument("--lo", type=float, default=-15.0)
ap.add_argument("--hi", type=float, default=20.0)
ap.add_argument("--restarts", type=int, default=2)
ap.add_argument("--maxiter", type=int, default=150)
ap.add_argument("--out", default=None)
args = ap.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT / "data_5species" / "main"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import estimate_paper_jax as E  # noqa: E402

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

data, t_days, _s, phi0, _ = E.load_experimental_data(E.DATA_DIR, *COND, 1, normalize=True, use_exp_init=True)
_, idx = E.convert_days_to_model_time(t_days, 1e-4, 2500, day_scale=None)
idx = np.clip(idx, 0, 2500)
phi0 = np.clip(phi0 / phi0.sum(), 0.001, 0.99)
mc = E.load_multichannel_data(data_dir=E.DATA_DIR / "experiment_data", condition=COND[0],
                              cultivation=COND[1], days_filter=t_days.tolist())
Yv = jnp.array(data)
Yt = jnp.array(mc["data_total"])
Yt = Yt / Yt.sum(1, keepdims=True)
L = jnp.array(mc["data_viability"])
k = jnp.array(idx[1:])  # Day 1 は初期値なので当てるのは 2 日目以降


def parts(th):
    g = E.simulate_0d_full(th, n_steps=2500, dt=1e-4, phi_init=jnp.array(phi0), K_hill=0.0, n_hill=4.0)
    phi = jnp.clip(g[k, 0:5], 1e-10, 1)
    pb = jnp.clip(phi * g[k, 6:11], 1e-12, 1)
    n = lambda x: x / x.sum(1, keepdims=True)  # noqa: E731
    return {"viable_vs_phi": jnp.mean((Yv[1:] - n(phi)) ** 2),
            "viable_vs_phipsi": jnp.mean((Yv[1:] - n(pb)) ** 2),
            "total_vs_phi": jnp.mean((Yt[1:] - n(phi)) ** 2),
            "live": jnp.mean((L[1:] - pb.sum(1) / phi.sum(1)) ** 2)}


def J(th):
    p = parts(th)
    return p["viable_vs_phipsi"] + p["total_vs_phi"] + p["live"]


def report(name, th):
    p = {k_: float(np.sqrt(v)) for k_, v in parts(jnp.array(th)).items()}
    print(f"{name}: RMSE  viable~phi {p['viable_vs_phi']:.3f}  viable~phipsi {p['viable_vs_phipsi']:.3f}"
          f"  total~phi {p['total_vs_phi']:.3f}  live {p['live']:.3f}", flush=True)
    return p


out = {"tag": args.tag, "map": report(f"{args.tag} MAP (now)", th0)}
free = [i for i in range(20) if i not in E.B_DIMS]
vg = jax.jit(jax.value_and_grad(J))


def obj(z):
    th = th0.copy()
    th[free] = z
    v, g = vg(jnp.array(th))
    v = float(v)
    return (1e6, np.zeros_like(z)) if not np.isfinite(v) else (v, np.nan_to_num(np.array(g)[free]))


rng = np.random.default_rng(0)
best = None
for r in range(args.restarts):
    z0 = np.clip(th0[free] + (0 if r == 0 else rng.normal(0, 2.0, len(free))), args.lo, args.hi)
    res = minimize(obj, z0, jac=True, method="L-BFGS-B", bounds=[(args.lo, args.hi)] * len(free),
                   options={"maxiter": args.maxiter})
    th = th0.copy()
    th[free] = res.x
    report(f"  restart {r} (nit {res.nit})", th)
    if best is None or res.fun < best.fun:
        best, bth = res, th
out["two_channel_opt"] = report(f"{args.tag} two-channel optimum", bth)
g = np.array(E.simulate_0d_full(jnp.array(bth), n_steps=2500, dt=1e-4, phi_init=jnp.array(phi0), K_hill=0.0, n_hill=4.0))
out["psi_opt_median_days"] = np.median(g[idx[1:], 6:11], axis=0).round(3).tolist()
out["theta_opt"] = bth.round(4).tolist()
print("psi (median over days):", out["psi_opt_median_days"], flush=True)
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
