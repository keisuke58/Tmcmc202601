#!/usr/bin/env python3
"""当てはまりの上限の診断（2026-10-11f）。

最終 MAP（final_theta_MAP/{TAG}.json）から勾配法で局所最適化して、
  (1) 尤度（run と同じ設定）を箱 [lo, hi] で最大化したときの logL と組成 RMSE
  (2) 組成の二乗誤差だけを最小化したときの RMSE（モデルの表現力の上限）
を出す。(1) で RMSE が下がれば箱が原因、(2) でしか下がらなければ生存率チャネルとの競合、
(2) でも下がらなければモデル（初期値・構造）が原因。

尤度は paper_gateoff_job.sh の p2/ult と同じ: --multichannel λch1=1, λch2=0, λch3=2→Commensal は 1.5,
λch5=0, λpg=5, λlate=3, λrare=0.1, K_hill=0, n_hill=4, --use-exp-init。
最初に logL(MAP) を出すので、run の max logL と一致することを確かめてから読むこと。

  python3 tools/fit_ceiling.py --tag CS [--lo -15 --hi 20 --restarts 3 --maxiter 200 --device gpu]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, choices=["CS", "CH", "DS", "DH"])
ap.add_argument("--lo", type=float, default=-15.0)
ap.add_argument("--hi", type=float, default=20.0)
ap.add_argument("--restarts", type=int, default=3)
ap.add_argument("--maxiter", type=int, default=200)
ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
ap.add_argument("--only", type=int, choices=[1, 2], default=None,
                help="run only (1) or (2), to split them into separate jobs")
args = ap.parse_args()
sys.argv = [sys.argv[0]]

sys.path.insert(0, str(ROOT / "data_5species" / "main"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", args.device)
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import estimate_paper_jax as E  # noqa: E402

COND = {"CS": ("Commensal", "Static"), "CH": ("Commensal", "HOBIC"),
        "DS": ("Dysbiotic", "Static"), "DH": ("Dysbiotic", "HOBIC")}[args.tag]
LAM3 = {"Commensal": 1.5, "Dysbiotic": 2.0}[COND[0]]
if COND == ("Dysbiotic", "HOBIC"):
    LAM3 = 3.0

d = json.load(open(ROOT / f"docs/revision/generated/final_theta_MAP/{args.tag}.json"))
th0 = np.array([d[str(i)] for i in range(20)])
D = E.DATA_DIR
data, t_days, sig, phi0, _ = E.load_experimental_data(D, *COND, 1, normalize=True, use_exp_init=True)
_, idx = E.convert_days_to_model_time(t_days, 1e-4, 2500, day_scale=None)
idx = np.clip(idx, 0, 2500)
phi0 = np.clip(phi0 / phi0.sum(), 0.001, 0.99)
mc = E.load_multichannel_data(data_dir=D / "experiment_data", condition=COND[0],
                              cultivation=COND[1], days_filter=t_days.tolist())
ll = E.make_log_likelihood_jax_ode(
    data=data, t_days=t_days, idx_sparse=idx, sigma_obs=sig, phi_init=phi0, dt=1e-4,
    n_steps=2500, K_hill=0.0, n_hill=4.0, lambda_pg=5.0, lambda_late=3.0,
    use_student_t=False, student_t_nu=5.0, lambda_rare=0.1, psi_fixed=None,
    data_total=mc.get("data_total"), sigma_obs_total=mc.get("sigma_obs_total"),
    data_viability=mc.get("data_viability"),
    sigma_obs_viability=mc.get("sigma_obs_viability", 0.10),
    lambda_ch={1: 1.0, 2: 0.0, 3: LAM3, 5: 0.0})
f = jax.jit(ll)
print(f"{args.tag}: logL(MAP) = {float(f(jnp.array(th0))):.3f}  ← run の max logL と一致するか確認", flush=True)


def comp(th):
    g = np.array(E.simulate_0d_full(jnp.array(th), n_steps=2500, dt=1e-4,
                                    phi_init=jnp.array(phi0), K_hill=0.0, n_hill=4.0))
    p = np.clip(g[idx, 0:5], 1e-10, 1)
    p /= p.sum(1, keepdims=True)
    viab = (g[idx, 0:5] * g[idx, 6:11]).sum(1) / g[idx, 0:5].sum(1)
    return np.sqrt(np.mean((data - p) ** 2)), np.sqrt(np.mean((data - p) ** 2, 0)), viab


free = [i for i in range(20) if i not in E.B_DIMS]
bnds = [(args.lo, args.hi)] * len(free)
r0, sp0, v0 = comp(th0)
print(f"MAP: RMSE {r0:.4f}  種ごと {np.round(sp0, 3)}  生存率 予測 {np.round(v0, 3)}"
      f"  実測 {np.round(mc.get('data_viability'), 3)}", flush=True)


def sse(x):
    g = E.simulate_0d_full(x, n_steps=2500, dt=1e-4, phi_init=jnp.array(phi0), K_hill=0.0, n_hill=4.0)
    p = jnp.clip(g[idx, 0:5], 1e-10, 1)
    p = p / p.sum(1, keepdims=True)
    return jnp.mean((jnp.array(data) - p) ** 2)


parts = [("(1) 尤度を最大化", lambda x: -ll(x)), ("(2) 組成だけ", sse)]
if args.only is not None:
    parts = [parts[args.only - 1]]
for name, fun in parts:
    vg = jax.jit(jax.value_and_grad(fun))

    def obj(z, vg=vg):
        th = th0.copy()
        th[free] = z
        v, g = vg(jnp.array(th))
        v = float(v)
        if not np.isfinite(v):
            return 1e6, np.zeros_like(z)
        return v, np.nan_to_num(np.array(g)[free])

    rng = np.random.default_rng(0)
    best = None
    for r in range(args.restarts):
        z0 = np.clip(th0[free] + (0 if r == 0 else rng.normal(0, 2.0, len(free))), args.lo, args.hi)
        res = minimize(obj, z0, jac=True, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": args.maxiter})
        th = th0.copy()
        th[free] = res.x
        print(f"  {name} restart {r}: logL {float(f(jnp.array(th))):.3f}  RMSE {comp(th)[0]:.4f}"
              f"  nit {res.nit}", flush=True)
        if best is None or res.fun < best.fun:
            best = res
    th = th0.copy()
    th[free] = best.x
    r, sp, v = comp(th)
    print(f"{name} best: logL {float(f(jnp.array(th))):.3f}  RMSE {r:.4f}  種ごと {np.round(sp, 3)}"
          f"  生存率 {np.round(v, 3)}", flush=True)
    print(f"  theta: {np.round(th, 3).tolist()}", flush=True)
