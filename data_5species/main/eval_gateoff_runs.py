#!/usr/bin/env python3
"""ゲート OFF の GPU run を RMSE と Pg サージで評価する。

docs/gate_off_map_reference.md の CPU 参照値と突き合わせるための集計。

正規化について（ここを間違えると数字が合わない）
------------------------------------------------
estimate_reduced_nishioka_jax.py:144-149 と同じ手順を踏む:

  1. 観測は load_experimental_data(normalize=True) で時点ごとに和 1 の分率
  2. 予測は phi_traj[idx] を clip したあと **時点ごとに和 1 へ正規化**
  3. RMSE は正規化後どうしの差で取る

予測を正規化せずに比べると、Pg のように分率が小さい種で系統的にずれる。

使い方:
    python eval_gateoff_runs.py [_runs のグロブ ...]
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

MAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN_DIR))
sys.path.insert(0, str(MAIN_DIR.parent.parent / "colab_package"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from estimate_reduced_nishioka import (
    convert_days_to_model_time,
    load_experimental_data,
)
from hamilton_ode_jax import simulate_0d

DATA_DIR = MAIN_DIR.parent  # estimate_reduced_nishioka_jax.py:34 と同じ

CODE2COND = {
    "dh": ("Dysbiotic", "HOBIC"),
    "ds": ("Dysbiotic", "Static"),
    "ch": ("Commensal", "HOBIC"),
    "cs": ("Commensal", "Static"),
}
DT, N_STEPS, C_CONST = 1e-4, 2500, 25.0
_cache = {}


def setup(code):
    if code in _cache:
        return _cache[code]
    cond, cult = CODE2COND[code]
    data, t_days, sigma_est, phi_init_exp, _ = load_experimental_data(
        DATA_DIR, cond, cult, 1, normalize=True
    )
    # run 側は --use-exp-init 付きなので Day1 実測を正規化したものを使う
    phi0 = phi_init_exp / phi_init_exp.sum()
    phi0 = np.clip(phi0, 0.01, 0.99)
    _, idx = convert_days_to_model_time(t_days, DT, N_STEPS, day_scale=None)
    idx = np.clip(idx, 0, N_STEPS)
    _cache[code] = (data, t_days, phi0, idx, sigma_est)
    return _cache[code]


def predict(theta, phi0, idx):
    traj = simulate_0d(
        jnp.array(theta),
        n_steps=N_STEPS,
        dt=DT,
        phi_init=jnp.array(phi0),
        K_hill=0.0,  # f25131e 以降 K=0 は真のゲート OFF
        n_hill=2.0,
        c_const=C_CONST,
    )
    pred = np.asarray(traj)[idx, :]
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    return pred / pred.sum(axis=1, keepdims=True)  # 時点ごとに和 1


BOX_LO, BOX_HI = -15.0, 20.0
EDGE_FRAC = 0.05  # 箱の端から幅の何割以内を「張り付き」とみなすか


def parse_name(name):
    """run 名から (code, arm, 粒子数, seed) を取る。

    旧: dh_gateoff_sigma6_5000p_seed42_cpualign_20260930 -> (dh, sigma6, 5000, 42)
    新: DH_pilot_gateon_mut80_seed42                      -> (dh, pilot/gateon, -1, 42)
        DH_ident_prior6_mut80_seed123                     -> (dh, ident/prior6, -1, 123)
    """
    parts = name.split("_")
    code = parts[0].lower()
    npart = next((int(x[:-1]) for x in parts if x.endswith("p") and x[:-1].isdigit()), -1)
    seed = next((int(x[4:]) for x in parts if x.startswith("seed") and x[4:].isdigit()), -1)
    if len(parts) > 1 and parts[1] in ("pilot", "p1", "p2", "ult", "ident"):
        # 新しい命名: STAGE と、それを修飾する部分（gateon / prior6 / mut80 …）を arm にする
        mods = [x for x in parts[2:] if not x.startswith("seed")]
        arm = "/".join([parts[1]] + mods) if mods else parts[1]
    else:
        arm = parts[2] if len(parts) > 2 else "?"
    return code, arm, npart, seed


def posterior_stats(samples, j, bounds=None):
    """theta[j] の事後統計。箱の端は run 自身の箱（無ければ [BOX_LO, BOX_HI]）基準。"""
    x = samples[:, j].astype(float)
    BOX_LO, BOX_HI = bounds if bounds else (globals()["BOX_LO"], globals()["BOX_HI"])
    m = EDGE_FRAC * (BOX_HI - BOX_LO)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {
        "mean": float(x.mean()),
        "sd": float(x.std()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "edge_pct": float(np.mean((x < BOX_LO + m) | (x > BOX_HI - m)) * 100),
        "p_pos": float(np.mean(x > 0)),
    }


def evaluate(d):
    name = Path(d).name
    code, arm, npart, seed = parse_name(name)
    data, t_days, phi0, idx, sigma = setup(code)
    with open(Path(d, "theta_MAP.json")) as f:
        th = json.load(f)
    theta = np.array([th[str(i)] for i in range(20)], dtype=np.float64)
    pred = predict(theta, phi0, idx)
    res = data - pred
    k15, k21 = t_days.tolist().index(15), t_days.tolist().index(21)
    row = {
        "run": name,
        "cond": code,
        "arm": arm,
        "n_particles": npart,
        "seed": seed,
        "rmse": float(np.sqrt(np.mean(res**2))),
        "chi": float(np.sqrt(np.mean((res / sigma) ** 2))),
        "pg_d15": float(pred[k15, 4]),
        "pg_d21": float(pred[k21, 4]),
        "pg_ratio": float(pred[k21, 4] / max(pred[k15, 4], 1e-12)),
        "obs_pg_ratio": float(data[k21, 4] / max(data[k15, 4], 1e-12)),
        "a35_map": float(theta[18]),
        "a45_map": float(theta[19]),
    }
    logl = Path(d, "logL.npy")
    if logl.exists():
        try:
            row["max_logL"] = float(np.load(logl).max())
        except ValueError:  # git-lfs のポインタなど
            row["max_logL"] = float("nan")
    smp = Path(d, "samples.npy")
    if smp.exists():
        try:
            S = np.load(smp)
            # run 自身の箱を config.json から読む（無ければ既定の [-15, 20]）
            bounds_all = None
            cfg = Path(d, "config.json")
            if cfg.exists():
                try:
                    with open(cfg) as f:
                        bounds_all = json.load(f).get("prior_bounds_final")
                except (json.JSONDecodeError, OSError):
                    pass
            for label, j in (("a35", 18), ("a45", 19)):
                bj = None
                if bounds_all and len(bounds_all) > j:
                    lo, hi = bounds_all[j]
                    if hi > lo:
                        bj = (float(lo), float(hi))
                for k, v in posterior_stats(S, j, bj).items():
                    row[f"{label}_{k}"] = v
        except ValueError:
            pass
    return row


COLUMNS = [
    "run",
    "cond",
    "arm",
    "n_particles",
    "seed",
    "max_logL",
    "rmse",
    "chi",
    "pg_d15",
    "pg_d21",
    "pg_ratio",
    "obs_pg_ratio",
    "a35_map",
    "a35_mean",
    "a35_sd",
    "a35_ci_lo",
    "a35_ci_hi",
    "a35_edge_pct",
    "a35_p_pos",
    "a45_map",
    "a45_mean",
    "a45_sd",
    "a45_ci_lo",
    "a45_ci_hi",
    "a45_edge_pct",
    "a45_p_pos",
]


def main(patterns, csv_path=None):
    dirs = sorted({d for p in patterns for d in glob.glob(p) if Path(d, "theta_MAP.json").exists()})
    if not dirs:
        print("評価できる run がありません（theta_MAP.json 待ち）")
        return
    # 論文パイプライン（estimate_paper_jax.py）の run はこの評価器で読まない。ここは
    # colab_package の前進モデル（n_hill=2・K_hill=0 固定）で予測するので、別のモデルになる
    # （ゲート ON の run で RMSE 0.126 → 0.272 にずれるのを確認済み）。tools/eval_paper_runs.py を使う
    paper = [d for d in dirs if Path(d, "run_record.json").exists()]
    if paper:
        print(
            f"論文パイプラインの run {len(paper)} 本は評価しない（tools/eval_paper_runs.py を使う）: "
            + ", ".join(Path(d).name for d in paper[:5])
            + (" …" if len(paper) > 5 else "")
        )
    dirs = [d for d in dirs if d not in paper]
    rows = [evaluate(d) for d in dirs if parse_name(Path(d).name)[0] in CODE2COND]

    hdr = (
        f"{'run':<44}{'RMSE':>8}{'chi':>7}{'D21/15':>8}"
        f"{'a35MAP':>8}{'a35sd':>7}{'a35端%':>8}{'a45MAP':>8}{'P(a45>0)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['run']:<44}{r['rmse']:8.4f}{r['chi']:7.2f}{r['pg_ratio']:8.2f}"
            f"{r['a35_map']:+8.2f}{r.get('a35_sd', float('nan')):7.2f}"
            f"{r.get('a35_edge_pct', float('nan')):7.1f}%"
            f"{r['a45_map']:+8.2f}{r.get('a45_p_pos', float('nan')):9.2f}"
        )
    for code in sorted({r["cond"] for r in rows}):
        data, t_days, _, _, _ = setup(code)
        k15, k21 = t_days.tolist().index(15), t_days.tolist().index(21)
        print(
            f"[実測 {code}] Pg D15={data[k15,4]:.4f} D21={data[k21,4]:.4f} "
            f"D21/15={data[k21,4]/data[k15,4]:.2f}"
        )

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {csv_path}  ({len(rows)} run)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("patterns", nargs="*", default=None, help="_runs のグロブ")
    ap.add_argument("--csv", default=None, help="CSV の書き出し先")
    a = ap.parse_args()
    main(a.patterns or ["_runs/*_gateoff_*cpualign*"], a.csv)
