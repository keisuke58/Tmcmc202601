#!/usr/bin/env python3
"""事後分布の粒子と点モデルの軌道を FEM 側（pde-fem-biofilm）へ書き出す（2026-10-19）。

各条件の最終段（CS・CH・DS は段 (iv)、DH は段 (iii)）の 3 seed の粒子をまとめ、200 個を一様に間引く
（TMCMC の最終段 β = 1 の粒子は重みが等しい）。各粒子で前進モデル（推定と同じ関数）を解き、共通の時刻で書き出す。
MAP（docs/revision/generated/final_theta_MAP/<tag>.json）の軌道も 1 本。生データ（反復ごとの値）は入れない。

    python3 tools/export_posterior_trajectories.py --out <pde-fem-biofilm>/data/posterior_final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_figures as MPF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "docs/revision/generated/final_runs"
MAPDIR = ROOT / "docs/revision/generated/final_theta_MAP"
# 最終段の run と MAP の seed（final_theta_MAP/README.md）
GROUPS = {"CS": ("CS_ult_2c_raw_seed*", 42), "CH": ("CH_ult_2c_raw_a33w_seed*", 7),
          "DS": ("DS_ult_2c_raw_seed*", 42), "DH": ("DH_p2_2c_raw_seed*", 42)}
STAGE = {"CS": "iv", "CH": "iv", "DS": "iv", "DH": "iii"}
ORDER = ["a11", "a12", "a22", "a33", "a34", "a44", "a13", "a14", "a23", "a24", "a55", "a15", "a25", "a35", "a45"]
GRID = np.arange(0.0, 21.0 + 1e-9, 0.25)


def on_grid(days, y):
    """y: (n, T, k) → (n, len(GRID), k)。"""
    return np.stack([np.stack([np.interp(GRID, days, y[i, :, j]) for j in range(y.shape[2])], axis=1)
                     for i in range(y.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tag, (pat, map_seed) in GROUPS.items():
        g = MPF.group(str(FINAL), pat, tag)
        runs = g["runs"]
        p0 = np.asarray(runs[0]["rec"]["phi_init"], float)
        for r in runs[1:]:  # 3 seed で初期値と時間の設定が同じことを確かめる
            assert np.allclose(np.asarray(r["rec"]["phi_init"], float), p0), tag
            for k in ("n_steps", "dt", "K_hill", "n_hill"):
                assert r["rec"]["args"][k] == runs[0]["rec"]["args"][k], (tag, k)
        rec = next(r["rec"] for r in runs if r["dir"].name.endswith(f"seed{map_seed}"))
        MPF.EP.DATA_DIR = MPF.EP.data_dir_for(rec["args"])
        names = list(rec["theta_names"])
        S = g["pooled"]
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(len(S), size=min(a.n, len(S)), replace=False)
        th = S[pick]
        m = json.load(open(MAPDIR / f"{tag}.json"))
        th_map = np.array([m[str(i)] for i in range(len(names))])[None, :]
        days = MPF.model_days(rec)
        X = MPF.simulate_full(th, rec)
        Xm = MPF.simulate_full(th_map, rec)

        def pack(X):
            phi, psi = X[:, :, 0:5], X[:, :, 6:11]
            extra = np.stack([X[:, :, 5], (phi * psi).sum(2), phi.sum(2)], axis=2)
            G = on_grid(days, np.concatenate([phi, psi, extra], axis=2))
            return G[:, :, 0:5], G[:, :, 5:10], G[:, :, 10], G[:, :, 11], G[:, :, 12]

        phi, psi, phi0, sl, sp = pack(X)
        mphi, mpsi, mphi0, msl, msp = pack(Xm)
        idx = [names.index(n) for n in ORDER]
        np.savez_compressed(
            out / f"{tag}_particles.npz",
            t_days=GRID, theta=th, theta_names=np.array(names), a15=th[:, idx], a15_names=np.array(ORDER),
            free_dims=np.array(rec["free_dims"]), phi=phi, psi=psi, phi0=phi0, sum_phipsi=sl, sum_phi=sp,
            map_theta=th_map[0], map_phi=mphi[0], map_psi=mpsi[0], map_phi0=mphi0[0], map_sum_phipsi=msl[0],
            map_sum_phi=msp[0], phi_init=p0, psi_init=np.asarray(rec.get("psi_init") or [], float),
            runs=np.array([r["dir"].name for r in runs]), stage=STAGE[tag])
        q = np.percentile(sl, [5, 50, 95], axis=0)
        summary[tag] = {"runs": [r["dir"].name for r in runs], "n_pooled": int(len(S)), "n_saved": int(len(th)),
                        "map_run_seed": map_seed, "sum_phipsi_day21_q05_q50_q95": [float(v) for v in q[:, -1]],
                        "map_sum_phipsi_day21": float(msl[0, -1])}
        print(tag, summary[tag])
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
