#!/usr/bin/env python3
"""論文パイプライン（estimate_paper_jax.py）の run から RMSE・Pg 比・事後を回収する。

なぜ eval_gateoff_runs.py を使わないか
------------------------------------
eval_gateoff_runs.py は colab_package の hamilton_ode_jax を import し、n_hill=2・φ の clip 0.01・
K_hill=0 固定で予測する。論文パイプラインの run（hamilton_ode_jax_paper・n_hill=4・clip 0.001・
ゲート ON の対照あり）とは前進モデルが違う。推定と評価で別のモデルを使うのは、2026-09 に
保存 logL と粒子が対応しなくなった事故（paper_gateoff_pipeline.md §1 の ③）と同じ形。

ここでは estimator が実際に使ったモジュール（estimate_paper_jax が import するもの）と、
run_record.json に記録された実効値（K_hill, n_hill, dt, n_steps, phi_init）だけで予測する。
さらに estimator が config.json に書いた RMSE と照合し、ずれたら行に印を付ける。

使い方:
    python3 tools/eval_paper_runs.py data_5species/main/_runs/paper_gateoff --glob '*mut80*' \
        [--csv out.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "data_5species" / "main"
# estimate_paper_jax は import 時に sys.argv の --device を見るので、一時的に差し替える
_argv = sys.argv
sys.argv = [_argv[0], "--device", "cpu"]
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN.parent))

import jax

jax.config.update("jax_enable_x64", True)
import estimate_paper_jax as EP
import jax.numpy as jnp

sys.argv = _argv
assert Path(EP._H.__file__).name == "hamilton_ode_jax_paper.py", EP._H.__file__

WATCH = {"a33": 5, "a35": 18, "a45": 19}  # a33 は DS で箱の端に積んでいた成分
EDGE_FRAC = 0.05
_data_cache = {}


def load_data(cond, cult, dt, n_steps):
    key = (cond, cult, dt, n_steps)
    if key not in _data_cache:
        data, t_days, _sig, _phi, _ = EP.load_experimental_data(
            EP.DATA_DIR, cond, cult, 1, normalize=True, use_exp_init=True
        )
        _, idx = EP.convert_days_to_model_time(t_days, dt, n_steps, day_scale=None)
        _data_cache[key] = (data, np.asarray(t_days).tolist(), np.clip(idx, 0, n_steps))
    return _data_cache[key]


def evaluate(d):
    with open(d / "run_record.json") as f:
        rec = json.load(f)
    a = rec["args"]
    with open(d / "theta_MAP.json") as f:
        th = json.load(f)
    theta = np.array([th[str(i)] for i in range(20)], dtype=np.float64)
    data, t_days, idx = load_data(a["condition"], a["cultivation"], a["dt"], a["n_steps"])
    traj = EP.simulate_0d(
        jnp.array(theta),
        n_steps=a["n_steps"],
        dt=a["dt"],
        phi_init=jnp.array(rec["phi_init"], dtype=jnp.float64),
        K_hill=a["K_hill"],
        n_hill=a["n_hill"],
    )
    pred = np.clip(np.asarray(traj)[idx, :], 1e-10, 1 - 1e-10)
    pred = pred / pred.sum(axis=1, keepdims=True)  # estimator の RMSE と同じ正規化
    rmse = float(np.sqrt(np.mean((data - pred) ** 2)))
    k15, k21 = t_days.index(15), t_days.index(21)
    row = {
        "run": d.name,
        "seed": a["seed"],
        "K_hill": a["K_hill"],
        "n_mut": a["n_mutation_steps"],
        "max_logL": rec["max_logL"],
        "rmse": rmse,
        "pg_ratio": float(pred[k21, 4] / max(pred[k15, 4], 1e-12)),
        "obs_pg_ratio": float(data[k21, 4] / max(data[k15, 4], 1e-12)),
    }
    cfg = d / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            r0 = json.load(f).get("rmse")
        row["rmse_estimator"] = r0
        row["rmse_match"] = r0 is not None and abs(r0 - rmse) < 1e-6
    else:
        row["rmse_estimator"], row["rmse_match"] = None, None
    s = np.load(d / "samples.npy")
    pb = np.array(rec["prior_bounds_final"], dtype=float)
    for name, j in WATCH.items():
        x, (lo, hi) = s[:, j], pb[j]
        m = EDGE_FRAC * (hi - lo)
        row[f"{name}_map"] = float(theta[j])
        row[f"{name}_p05"], row[f"{name}_p50"], row[f"{name}_p95"] = (
            float(v) for v in np.percentile(x, [5, 50, 95])
        )
        row[f"{name}_edge"] = float(np.mean((x < lo + m) | (x > hi - m))) if hi > lo else 0.0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(MAIN / "_runs" / "paper_gateoff"))
    ap.add_argument("--glob", default="*")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    dirs = [
        d
        for d in sorted(Path(args.root).glob(args.glob))
        if (d / "run_record.json").exists() and (d / "theta_MAP.json").exists()
    ]
    if not dirs:
        print(f"評価できる run が無い: {args.root}/{args.glob}")
        return 1
    rows = [evaluate(d) for d in dirs]

    print(
        f"{'run':<40}{'K':>5}{'maxlogL':>9}{'RMSE':>8}{'照合':>4}{'D21/15':>8}{'実測':>6}"
        f"{'a33 [5,50,95]%':>24}{'端':>5}{'a45 [5,50,95]%':>24}{'端':>5}"
    )
    for r in rows:
        ok = {True: "OK", False: "NG", None: "-"}[r["rmse_match"]]
        print(
            f"{r['run']:<40}{r['K_hill']:>5.2f}{r['max_logL']:>9.2f}{r['rmse']:>8.4f}{ok:>4}"
            f"{r['pg_ratio']:>8.2f}{r['obs_pg_ratio']:>6.2f}"
            f"  [{r['a33_p05']:+6.2f},{r['a33_p50']:+6.2f},{r['a33_p95']:+6.2f}]{r['a33_edge']:>5.0%}"
            f"  [{r['a45_p05']:+6.2f},{r['a45_p50']:+6.2f},{r['a45_p95']:+6.2f}]{r['a45_edge']:>5.0%}"
        )
    bad = [r["run"] for r in rows if r["rmse_match"] is False]
    if bad:
        print(f"\n警告: estimator の RMSE と一致しない run（前進モデルか設定の取り違え）: {bad}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
