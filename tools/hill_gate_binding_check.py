"""
hill_gate_binding_check.py — A0 step 1: is the F.n Hill gate binding in the posterior region?

For posterior samples of each run, re-simulates the 20-parameter Hamilton ODE
(tmcmc/program2602/improved_5species_jit.py, the solver used by
estimate_reduced_nishioka.py) with the run's own Hill gate (K, n) and with the
gate switched off (K=0), and reports
  * h(phibar_Fn) at the observation times,
  * the change in phibar_Pg at the observation times when the gate is removed,
  * RMSE vs data.npy with and without the gate.

Requires the LFS objects of the run directories (git lfs pull).

Usage
-----
  python tools/hill_gate_binding_check.py [N_SUB]   # default 200 samples per run
"""

import json
import sys

import numpy as np
from pathlib import Path

R = str(Path(__file__).resolve().parent.parent) + "/"
sys.path.insert(0, R + "tmcmc/program2602")
from improved_5species_jit import BiofilmNewtonSolver5S

RUNS = {
    "DH 1k30 (2000)": R + "_runs/Dysbiotic_HOBIC_K0.05_n4.0_1k30",
    "DH sweep K.05n4": R + "_sweeps/K0.05_n4.0",
    "DH sweep baseline": R + "_sweeps/K0.05_n4.0_baseline",
    "DS posterior": R + "data_5species/_runs/dysbiotic_static_posterior",
    "CH posterior": R + "data_5species/_runs/commensal_hobic_posterior",
    "CS posterior": R + "data_5species/_runs/commensal_static_posterior",
}
NSUB = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(0)
out = {}
for name, d in RUNS.items():
    cfg = json.load(open(d + "/config.json"))
    S = np.load(d + "/samples.npy")
    data = np.load(d + "/data.npy")
    idx = np.load(d + "/idx_sparse.npy").astype(int)
    K, n = cfg["K_hill"], cfg["n_hill"]
    kw = dict(
        dt=cfg["dt"],
        maxtimestep=cfg["maxtimestep"],
        c_const=cfg["c_const"],
        alpha_const=cfg["alpha_const"],
        phi_init=cfg["phi_init"],
        Kp1=cfg["Kp1"],
    )
    on = BiofilmNewtonSolver5S(**kw, K_hill=K, n_hill=n)
    off = BiofilmNewtonSolver5S(**kw, K_hill=0.0, n_hill=n)
    sel = rng.choice(len(S), min(NSUB, len(S)), replace=False)
    try:
        MAP = json.load(open(d + "/theta_MAP.json"))
        MAP = np.array(
            MAP["theta_full"] if "theta_full" in MAP else [MAP[str(i)] for i in range(20)]
        )
    except Exception:
        MAP = None
    rows = []
    thetas = [("MAP", MAP)] if MAP is not None else []
    thetas += [(int(i), S[i]) for i in sel]
    for tag, th in thetas:
        _, g1 = on.run_deterministic(th)
        _, g0 = off.run_deterministic(th)
        pb1 = g1[:, 0:5] * g1[:, 6:11]
        pb0 = g0[:, 0:5] * g0[:, 6:11]
        fn = np.maximum(pb1[:, 3], 0)
        h = fn**n / (K**n + fn**n)
        rows.append(
            dict(
                tag=tag,
                h_obs=h[idx].tolist(),
                frac_h_lt_05=float(np.mean(h < 0.5)),
                frac_h_lt_09=float(np.mean(h < 0.9)),
                pg_on=pb1[idx, 4].tolist(),
                pg_off=pb0[idx, 4].tolist(),
                dmax_obs=float(np.max(np.abs(pb1[idx] - pb0[idx]))),
                dmax_pg_obs=float(np.max(np.abs(pb1[idx, 4] - pb0[idx, 4]))),
                rmse_on=float(np.sqrt(np.mean((pb1[idx] - data) ** 2))),
                rmse_off=float(np.sqrt(np.mean((pb0[idx] - data) ** 2))),
                finite=bool(np.all(np.isfinite(pb1)) and np.all(np.isfinite(pb0))),
            )
        )
    out[name] = dict(K=K, n=n, n_eval=len(rows), rows=rows)
    post = [r for r in rows if r["tag"] != "MAP"]
    H = np.array([r["h_obs"] for r in post])
    dm = np.array([r["dmax_pg_obs"] for r in post])
    dr = np.array([r["rmse_off"] - r["rmse_on"] for r in post])
    print(f"\n== {name}  (K={K}, n={n}, N={len(post)})")
    if MAP is not None:
        m = rows[0]
        print(
            "  MAP  h@obs =",
            np.round(m["h_obs"], 3),
            " max|dPg|@obs =",
            f"{m['dmax_pg_obs']:.3g}",
            " RMSE on/off =",
            f"{m['rmse_on']:.4f}/{m['rmse_off']:.4f}",
        )
    print("  posterior h@obs median =", np.round(np.median(H, 0), 3))
    print(
        "  posterior h@obs 5–95%  =",
        np.round(np.percentile(H, 5, 0), 3),
        "…",
        np.round(np.percentile(H, 95, 0), 3),
    )
    print(f"  share of samples with h<0.5 at ANY obs time: {np.mean((H < 0.5).any(1)):.2f}")
    print(f"  share with h<0.5 at LAST obs time: {np.mean(H[:, -1] < 0.5):.2f}")
    print(
        f"  max|Pg_on - Pg_off| @obs: median {np.median(dm):.3g}, 95% {np.percentile(dm, 95):.3g}"
    )
    print(
        f"  RMSE(off) - RMSE(on): median {np.median(dr):+.4f}, 5–95% [{np.percentile(dr, 5):+.4f}, {np.percentile(dr, 95):+.4f}]"
    )
OUT = Path(R) / "tools" / "_hill_binding"
OUT.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT / "hill_gate_binding.json", "w"))
