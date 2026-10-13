#!/usr/bin/env python3
"""予測の確認（leave-one-day-out、2026-10-13）: 1 日を外して推定した群が、その日をどれだけ予測できるか。

外した日の観測（生きた菌の組成・全菌の組成・群集の生きた割合）と、事後予測（N 粒子）を比べる:
- 事後予測の中央値の RMSE（組成は 5 種、生きた割合は 1 つ）
- 90% 予測区間に観測が入る割合（組成の 5 種）
同じ日を、その日も使って推定した群（--full）でも同じ数字を出して並べる（当てはめ vs 予測）。
組成の比べ方は run に合わせる（2 通路なら生きた菌 ↔ φψ/Σφψ、そうでなければ φ/Σφ）。

    python3 tools/holdout_eval.py <root> --holdout 'CH_p2_2c_ho10_seed*' --full 'CH_p2_2c_seed*' --day 10 \
        --out docs/revision/generated/holdout/CH_day10.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_figures as MPF  # noqa: E402


def evaluate(root, pattern, day, n, rng):
    g = MPF.group(root, pattern, pattern.split("_")[0])
    if g is None:
        raise SystemExit(f"run が無い: {root}/{pattern}")
    rec = g["best"]["rec"]
    a = rec["args"]
    MPF.EP.DATA_DIR = MPF.EP.data_dir_for(a)  # run が読んだデータ
    days = MPF.model_days(rec)
    k = int(np.argmin(np.abs(days - day)))
    S = g["pooled"]
    X = MPF.simulate_full(S[rng.choice(len(S), size=min(n, len(S)), replace=False)], rec)[:, k, :]
    phi, psi = np.clip(X[:, 0:5], 1e-10, 1), np.clip(X[:, 6:11], 1e-10, 1)
    two = bool(a.get("two_channel"))
    viab_pred = phi * psi if two else phi
    viab_pred = viab_pred / viab_pred.sum(1, keepdims=True)
    tot_pred = phi / phi.sum(1, keepdims=True)
    live_pred = (phi * psi).sum(1) / phi.sum(1)
    # 観測（全日のデータから、外した日を取る）
    data, t_days, _i = MPF.exp_data(a["condition"], a["cultivation"], {**a, "start_from_day": 1})
    j = int(np.where(np.asarray(t_days) == day)[0][0])
    mc = MPF.EP.load_multichannel_data(data_dir=MPF.EP.DATA_DIR / "experiment_data", condition=a["condition"],
                                       cultivation=a["cultivation"], days_filter=list(t_days))
    obs = {"viable": np.asarray(data[j]), "total": np.asarray(mc["data_total"])[j] / np.sum(mc["data_total"][j]),
           "live": float(np.asarray(mc["data_viability"])[j])}
    out = {"runs": [r["dir"].name for r in g["runs"]], "two_channel": two, "day": day}
    for name, P in (("viable", viab_pred), ("total", tot_pred)):
        med, lo, hi = np.median(P, 0), np.quantile(P, 0.05, 0), np.quantile(P, 0.95, 0)
        out[name] = {"rmse_median": float(np.sqrt(np.mean((med - obs[name]) ** 2))),
                     "coverage90": float(np.mean((obs[name] >= lo) & (obs[name] <= hi))),
                     "pred_median": med.round(4).tolist(), "obs": np.round(obs[name], 4).tolist()}
    out["live"] = {"abs_err_median": float(abs(np.median(live_pred) - obs["live"])),
                   "in90": bool(np.quantile(live_pred, 0.05) <= obs["live"] <= np.quantile(live_pred, 0.95)),
                   "pred_median": float(np.median(live_pred)), "obs": obs["live"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--full", default=None)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    res = {"holdout": evaluate(args.root, args.holdout, args.day, args.n, rng)}
    if args.full:
        res["full"] = evaluate(args.root, args.full, args.day, args.n, rng)
    for key, r in res.items():
        print(f"[{key}] day {args.day}: 生きた菌 RMSE {r['viable']['rmse_median']:.3f}（90% 区間に {r['viable']['coverage90']:.0%}）"
              f"  全菌 RMSE {r['total']['rmse_median']:.3f}（{r['total']['coverage90']:.0%}）"
              f"  生きた割合 誤差 {r['live']['abs_err_median']:.3f}（90% 区間に{'入る' if r['live']['in90'] else '入らない'}）")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
