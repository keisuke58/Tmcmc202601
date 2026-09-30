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

import glob
import json
import sys
from pathlib import Path

import numpy as np

MAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN_DIR))
sys.path.insert(0, str(MAIN_DIR.parent.parent / "colab_package"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from estimate_reduced_nishioka import (  # noqa: E402
    convert_days_to_model_time,
    load_experimental_data,
)
from hamilton_ode_jax import simulate_0d  # noqa: E402

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


def main(patterns):
    dirs = sorted({d for p in patterns for d in glob.glob(p) if Path(d, "theta_MAP.json").exists()})
    if not dirs:
        print("評価できる run がありません（theta_MAP.json 待ち）")
        return
    hdr = (
        f"{'run':<42}{'RMSE':>8}{'chi':>8}{'PgD15':>8}{'PgD21':>8}{'D21/15':>8}{'a35':>8}{'a45':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for d in dirs:
        name = Path(d).name
        code = name.split("_")[0]
        if code not in CODE2COND:
            continue
        data, t_days, phi0, idx, sigma = setup(code)
        th = json.load(open(Path(d, "theta_MAP.json")))
        theta = np.array([th[str(i)] for i in range(20)], dtype=np.float64)
        pred = predict(theta, phi0, idx)
        res = data - pred
        rmse = float(np.sqrt(np.mean(res**2)))
        chi = float(np.sqrt(np.mean((res / sigma) ** 2)))
        d15, d21 = pred[t_days.tolist().index(15), 4], pred[t_days.tolist().index(21), 4]
        print(
            f"{name:<42}{rmse:8.4f}{chi:8.3f}{d15:8.4f}{d21:8.4f}"
            f"{d21/max(d15,1e-12):8.2f}{theta[18]:+8.2f}{theta[19]:+8.2f}"
        )
    for code in sorted({Path(d).name.split("_")[0] for d in dirs} & set(CODE2COND)):
        data, t_days, _, _, _ = setup(code)
        d15, d21 = data[t_days.tolist().index(15), 4], data[t_days.tolist().index(21), 4]
        print(f"[実測 {code}] Pg D15={d15:.4f} D21={d21:.4f} D21/15={d21/d15:.2f}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["_runs/*_gateoff_*cpualign*"]
    main(args)
