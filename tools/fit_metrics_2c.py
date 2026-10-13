#!/usr/bin/env python3
"""2 通路の当てはまりの内訳を run の MAP から出す（13g の対照表、2026-10-13）。

RMSE の定義は tools/two_channel_pilot_v2.py の parts() と同じ（重みなし、Day 1 = 初期値は除く）:
  viable_vs_phipsi  生きた菌の組成 ↔ φψ/Σφψ（2 通路の ch1）
  viable_vs_phi     生きた菌の組成 ↔ φ/Σφ（古い尤度の比べ方、参考）
  total_vs_phi      全菌の組成 ↔ φ/Σφ（ch2）
  live              群集の生きた割合 ↔ Σφψ/Σφ（ch3）
  psi_vs_data       ψ ↔ ψ^data（全菌の割合 2% 以上の種・日）
Pg の Day21/Day15 の比は生きた菌の組成（φψ/Σφψ）で、実測（生きた菌の組成）と並べる。

    python3 tools/fit_metrics_2c.py data_5species/main/_runs/paper_gateoff 'CH_p2_2c_seed*' 'CH_p2_2c_now_seed*'
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.argv, _argv = [sys.argv[0], "--device", "cpu"], sys.argv
sys.path.insert(0, str(ROOT / "data_5species" / "main"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp  # noqa: E402

import estimate_paper_jax as E  # noqa: E402

sys.argv = _argv
KEYS = ["viable_vs_phipsi", "viable_vs_phi", "total_vs_phi", "live", "psi_vs_data"]
_cache = {}


def data_for(cond, cult, dt, n_steps, ddir):
    key = (cond, cult, dt, n_steps, str(ddir))
    if key not in _cache:
        data, t_days, _sig, _phi0, _ = E.load_experimental_data(ddir, cond, cult, 1, normalize=True,
                                                                use_exp_init=True)
        _, idx = E.convert_days_to_model_time(t_days, dt, n_steps, day_scale=None)
        idx = np.clip(idx, 0, n_steps)
        mc = E.load_multichannel_data(data_dir=ddir / "experiment_data", condition=cond,
                                      cultivation=cult, days_filter=t_days.tolist())
        tot = np.asarray(mc["data_total"], dtype=float)
        tot = tot / tot.sum(1, keepdims=True)
        live = np.asarray(mc["data_viability"], dtype=float)
        psi_d = E.species_viability_from_data(cond, cult, t_days, live)
        _cache[key] = dict(data=np.asarray(data), t_days=list(np.asarray(t_days, dtype=float)), idx=idx,
                           tot=tot, live=live, psi_d=psi_d, psi_m=(tot >= 0.02).astype(float))
    return _cache[key]


def metrics(d):
    rec = json.load(open(d / "run_record.json"))
    a = rec["args"]
    th = json.load(open(d / "theta_MAP.json"))
    theta = jnp.array([th[str(i)] for i in range(20)], dtype=jnp.float64)
    D = data_for(a["condition"], a["cultivation"], a["dt"], a["n_steps"], E.data_dir_for(a))
    g = np.asarray(E.simulate_0d_full(theta, n_steps=a["n_steps"], dt=a["dt"],
                                      phi_init=jnp.array(rec["phi_init"], dtype=jnp.float64),
                                      K_hill=a["K_hill"], n_hill=a["n_hill"]))
    k = D["idx"][1:]
    phi = np.clip(g[k, 0:5], 1e-10, 1)
    psi = g[k, 6:11]
    pb = np.clip(phi * psi, 1e-12, 1)
    n = lambda x: x / x.sum(1, keepdims=True)  # noqa: E731
    Yv, Yt, L = D["data"][1:], D["tot"][1:], D["live"][1:]
    M = D["psi_m"][1:]
    out = {
        "viable_vs_phipsi": np.sqrt(np.mean((Yv - n(pb)) ** 2)),
        "viable_vs_phi": np.sqrt(np.mean((Yv - n(phi)) ** 2)),
        "total_vs_phi": np.sqrt(np.mean((Yt - n(phi)) ** 2)),
        "live": np.sqrt(np.mean((L - pb.sum(1) / phi.sum(1)) ** 2)),
        "psi_vs_data": np.sqrt(np.sum(M * (psi - D["psi_d"][1:]) ** 2) / np.sum(M)),
    }
    days = D["t_days"][1:]
    k15, k21 = days.index(15.0), days.index(21.0)
    out["pg_ratio"] = n(pb)[k21, 4] / max(n(pb)[k15, 4], 1e-12)
    out["obs_pg_ratio"] = Yv[k21, 4] / max(Yv[k15, 4], 1e-12)
    out["max_logL"] = rec["max_logL"]
    out["weights"] = {w: a.get(w) for w in ("lambda_pg", "lambda_late", "lambda_rare")}
    return {k_: (float(v) if isinstance(v, (float, np.floating)) else v) for k_, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--out", default=None, help="拡張子なし。.md と .json を書く")
    a = ap.parse_args()
    res = {}
    L = ["| run | 重み (Pg, late, rare) | max logL | 生きた菌↔φψ | (生きた菌↔φ) | 全菌↔φ | 生きた割合 | ψ↔ψ^data | Pg D21/D15（実測） |",
         "|---|---|---|---|---|---|---|---|---|"]
    for gl in a.globs:
        ds = sorted(Path(p) for p in glob.glob(str(Path(a.root) / gl)) if (Path(p) / "run_record.json").exists())
        if not ds:
            L.append(f"| `{gl}`: run なし | | | | | | | | |")
            continue
        rows = {d.name: metrics(d) for d in ds}
        res[gl] = rows
        for name, m in rows.items():
            w = m["weights"]
            L.append(f"| {name} | {w['lambda_pg']}, {w['lambda_late']}, {w['lambda_rare']} | {m['max_logL']:.2f} | "
                     f"{m['viable_vs_phipsi']:.4f} | {m['viable_vs_phi']:.4f} | {m['total_vs_phi']:.4f} | "
                     f"{m['live']:.4f} | {m['psi_vs_data']:.3f} | {m['pg_ratio']:.2f}（{m['obs_pg_ratio']:.2f}） |")
        mean = {k_: np.mean([m[k_] for m in rows.values()]) for k_ in KEYS + ["pg_ratio"]}
        L.append(f"| **`{gl}` の平均** | | | {mean['viable_vs_phipsi']:.4f} | {mean['viable_vs_phi']:.4f} | "
                 f"{mean['total_vs_phi']:.4f} | {mean['live']:.4f} | {mean['psi_vs_data']:.3f} | {mean['pg_ratio']:.2f} |")
    md = "\n".join(L) + "\n"
    print(md)
    if a.out:
        Path(a.out + ".md").write_text(md)
        json.dump(res, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
