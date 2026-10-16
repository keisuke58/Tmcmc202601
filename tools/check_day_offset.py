#!/usr/bin/env python3
"""1 日のずれ（Day 1 の組成を t=0=day 0 に置いている）の影響: 同じ MAP で観測日の対応を t=s·d と t=s·(d−1) で比べる。

2026-10-12 初版（読み取りデータ・1 通路）。2026-10-16: run 群を引数で渡し、run が読んだデータと尤度の対応
（2 通路なら生きた菌 ↔ φψ/Σφψ、全菌 ↔ φ/Σφ、tools/comp_pred.py）で比べる版に。再推定はしない。

    python3 tools/check_day_offset.py 'CS_ult_2c_raw_seed*' 'CH_ult_2c_raw_a33w_seed*' [--root docs/revision/generated/final_runs]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("globs", nargs="+", help="条件ごとの run 群（max logL が最大の seed の MAP を使う）")
ap.add_argument("--root", default=str(ROOT / "docs/revision/generated/final_runs"))
args = ap.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT / "data_5species/main"))
sys.path.insert(0, str(ROOT / "tools"))
import jax  # noqa: E402

jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp  # noqa: E402
import estimate_paper_jax as E  # noqa: E402
import comp_pred as CP  # noqa: E402

for pat in args.globs:
    runs = sorted(glob.glob(str(Path(args.root) / pat)))
    if not runs:
        print(f"{pat}: run なし")
        continue
    d = Path(max(runs, key=lambda r: float(np.load(Path(r) / "logL.npy").max())))
    rec = json.load(open(d / "run_record.json"))
    a = rec["args"]
    th = json.load(open(d / "theta_MAP.json"))
    th = jnp.array([th[str(i)] for i in range(20)])
    data, td, _, _, _ = E.load_experimental_data(E.data_dir_for(a), a["condition"], a["cultivation"], 1,
                                                 normalize=True, use_exp_init=True)
    keep = td > 1 + 1e-9
    two = CP.is_two_channel(rec)
    tot, _ = CP.observations(E, rec, td[keep]) if two else (None, None)
    g = np.asarray(E.simulate_0d_full(th, n_steps=a["n_steps"], dt=a["dt"], phi_init=jnp.array(rec["phi_init"]),
                                      psi_init=rec.get("psi_init"), K_hill=a["K_hill"], n_hill=a["n_hill"]))
    fr = CP.fractions(g, two)
    out = []
    for off in (0, 1):
        _, idx = E.convert_days_to_model_time(td - off, a["dt"], a["n_steps"], day_scale=None) if off == 0 else (None, None)
        if off == 0:
            idx0 = np.clip(idx, 0, a["n_steps"])
            step_per_day = idx0[-1] / td[-1]
        idx = np.clip(np.round((td - off) * step_per_day).astype(int), 0, a["n_steps"])[keep]
        r = [float(np.sqrt(np.mean((data[keep] - fr["viable"][idx]) ** 2)))]
        if tot is not None:
            r.append(float(np.sqrt(np.mean((tot - fr["total"][idx]) ** 2))))
        out.append(r)
    lab = "生きた菌" + ("・全菌" if tot is not None else "")
    print(f"{d.name}: RMSE（{lab}） t=s·d {out[0]}  t=s·(d−1) {out[1]}  差 {[round(y - x, 4) for x, y in zip(*out)]}")
