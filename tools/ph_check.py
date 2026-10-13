#!/usr/bin/env python3
"""pH の答え合わせ（推定に使っていないデータによる検証、2026-10-12）。

pH（Heine 2025 Fig. 4A、HOBIC の CH・DH）は尤度に入れていない（λ_ch5 = 0、2026-10-08 のユーザー判断）。
ここでは:
  1. 回帰式を**実測だけ**から一度決めて固定する: 実測の種の割合（Fig. 3A の中央値）→ 実測 pH。
     割合の和が 1 なので 5 種と切片は共線。Vei を基準にした 4 種（So・An・Fn・Pg）＋切片の線形式にする
     （data_5species/docs/fig_ph_independent_validation.py の fit_ph_regression と同じ形）。
  2. 較正済みモデルの事後標本で組成 φ_i/Σφ を前進計算し、固定した回帰式に通して pH の予測帯を作る。
  3. 予測帯と実測 pH の時系列を比べる（RMSE・R²・90% 帯の被覆率）。
     参考として、実測の組成を回帰式に通した値（回帰式そのものの当てはまり、上限）も出す。

  python3 tools/ph_check.py [--root docs/revision/generated/final_runs] [--n-post 200]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT / "docs/revision/generated/final_runs"))
ap.add_argument("--glob-CH", default="CH_ult_mut150_wide2_noph_sd4_seed*")
ap.add_argument("--glob-DH", default="DH_p2_nonarrow_w3_noph_p14k_seed*")
ap.add_argument("--n-post", type=int, default=200)
ap.add_argument("--start-day", type=float, default=0.0, help="モデル時刻 0 を何日目に置くか（0 = 接種、本文の対応。1 = Day 1 から始める）")
ap.add_argument("--data-dir", default=None, help="組成・pH を読む場所（既定は CH の run の data_source: EP.data_dir_for）")
ap.add_argument("--ph-csv", default=None, help="pH の CSV（time_days, pH_commensal, pH_dysbiotic）。既定は <data-dir>/experiment_data/fig4A_pH_timeseries.csv")
ap.add_argument("--out", default=str(ROOT / "docs/revision/EML_submission/figures/provisional_2026-10-11/ph_check"))
args = ap.parse_args()
sys.argv = [sys.argv[0]]

sys.path.insert(0, str(ROOT / "data_5species" / "main"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import estimate_paper_jax as E  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import comp_pred as CP  # noqa: E402  （2 通路の run では φψ/Σφψ を生きた菌の組成と比べる）

# データの場所: run が読んだデータ（2026-10-13、raw / digitised）。--data-dir で上書き
_r0 = sorted(Path(args.root).glob(args.glob_CH))
DDIR = Path(args.data_dir) if args.data_dir else E.data_dir_for(json.load(open(_r0[0] / "run_record.json"))["args"])
EXP = DDIR / "experiment_data"
PH_CSV = Path(args.ph_csv) if args.ph_csv else EXP / "fig4A_pH_timeseries.csv"
if not PH_CSV.exists():
    raise SystemExit(f"pH の CSV が無い: {PH_CSV}（生データの pH は tools/convert_raw_ph.py で作る。--ph-csv で指定も可）")
print(f"データ: {EXP}  pH: {PH_CSV}")
SPECIES = ["So", "An", "Vei", "Fn", "Pg"]
REG = [0, 1, 3, 4]  # So・An・Fn・Pg（Vei が基準）
EXP_DAYS = np.array([1.0, 3.0, 6.0, 10.0, 15.0, 21.0])
CMAP = {"S. oralis": "So", "A. naeslundii": "An", "V. dispar": "Vei", "V. parvula": "Vei",
        "F. nucleatum": "Fn", "P. gingivalis_20709": "Pg", "P. gingivalis_W83": "Pg"}

ph = pd.read_csv(PH_CSV)
t_ph = ph["time_days"].values
PH = {"CH": ph["pH_commensal"].values, "DH": ph["pH_dysbiotic"].values}
sp = pd.read_csv(EXP / "fig3_species_distribution_summary.csv")


def medians(cond):
    sub = sp[(sp.condition == cond) & (sp.cultivation == "HOBIC")]
    M = np.zeros((len(EXP_DAYS), 5))
    for k, s in enumerate(SPECIES):
        q = sub[sub.species.map(CMAP) == s].sort_values("day")
        M[:, k] = np.interp(EXP_DAYS, q["day"].values, q["median"].values / 100.0)
    return M / M.sum(1, keepdims=True)


COMP = {"CH": medians("Commensal"), "DH": medians("Dysbiotic")}

# 1. 回帰式（実測だけ）: pH 測定時刻に実測の組成を線形補間して当てる
mask = (t_ph >= 1.0) & (t_ph <= 21.0)


def design(C):
    return np.column_stack([np.ones(len(C))] + [C[:, k] for k in REG])


Xs, ys = [], []
for c in ["CH", "DH"]:
    Ct = np.column_stack([np.interp(t_ph[mask], EXP_DAYS, COMP[c][:, k]) for k in range(5)])
    Xs.append(design(Ct))
    ys.append(PH[c][mask])
X, y = np.vstack(Xs), np.concatenate(ys)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
fit_r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
print("回帰式（実測だけ）: pH = " + f"{beta[0]:+.3f}" +
      "".join(f" {b:+.3f}·{SPECIES[k]}" for b, k in zip(beta[1:], REG)) + f"   R² {fit_r2:.3f}")

# 2. 事後予測
out = {"beta": beta.tolist(), "regression_r2_on_data": float(fit_r2)}
bands = {}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, (tag, cond, glob) in zip(axes, [("CH", "Commensal", args.glob_CH), ("DH", "Dysbiotic", args.glob_DH)]):
    runs = sorted(Path(args.root).glob(glob))
    rec = json.load(open(runs[0] / "run_record.json"))
    a = rec["args"]
    S = np.concatenate([np.load(r / "samples.npy") for r in runs])
    pick = S[np.random.default_rng(0).choice(len(S), size=min(args.n_post, len(S)), replace=False)]
    sim = jax.jit(jax.vmap(lambda th: E.simulate_0d_full(th, n_steps=a["n_steps"], dt=a["dt"],
                                                         phi_init=jnp.array(rec["phi_init"]),
                                                         K_hill=a["K_hill"], n_hill=a["n_hill"])))
    # 回帰式は実測の生きた菌の組成 → pH なので、モデル側も生きた菌の組成の予測（尤度と同じ対応）
    P = CP.fractions(np.asarray(sim(jnp.array(pick))), CP.is_two_channel(rec))["viable"]
    data, t_days, _, _, _ = E.load_experimental_data(E.data_dir_for(a), cond, "HOBIC", 1, normalize=True, use_exp_init=True)
    _, idx = E.convert_days_to_model_time(t_days, a["dt"], a["n_steps"], day_scale=None)
    days_axis = args.start_day + np.arange(a["n_steps"] + 1) * (t_days[-1] / idx[-1])
    pH_post = beta[0] + P[..., REG] @ beta[1:]  # (n, steps)
    lo, med, hi = np.percentile(pH_post, [5, 50, 95], axis=0)
    tt, yy = t_ph[mask], PH[tag][mask]
    m_at = np.interp(tt, days_axis, med)
    lo_at, hi_at = np.interp(tt, days_axis, lo), np.interp(tt, days_axis, hi)
    rmse = float(np.sqrt(np.mean((yy - m_at) ** 2)))
    r2 = float(1 - np.sum((yy - m_at) ** 2) / np.sum((yy - yy.mean()) ** 2))
    cover = float(np.mean((yy >= lo_at) & (yy <= hi_at)))
    Ct = np.column_stack([np.interp(tt, EXP_DAYS, COMP[tag][:, k]) for k in range(5)])
    rmse_reg = float(np.sqrt(np.mean((yy - design(Ct) @ beta) ** 2)))
    out[tag] = {"runs": [r.name for r in runs], "n_points": int(mask.sum()), "rmse_posterior_median": rmse,
                "r2_posterior_median": r2, "coverage_90": cover, "rmse_regression_on_measured_composition": rmse_reg}
    print(f"{tag}: 事後中央値 RMSE {rmse:.3f}  R² {r2:.3f}  90% 帯の被覆 {cover:.2f}"
          f"  （実測組成を回帰式に通した RMSE {rmse_reg:.3f}、{mask.sum()} 点）")
    bands[tag] = dict(days=days_axis, lo=lo, med=med, hi=hi, t_meas=tt, ph_meas=yy)
    ax.fill_between(days_axis, lo, hi, color="0.75", lw=0, label="model, 90% band")
    ax.plot(days_axis, med, color="0.2", lw=1.2, label=f"model, median (RMSE {rmse:.2f})")
    ax.plot(tt, yy, "o", ms=3, color="C3", label="measured pH (not used in inference)")
    ax.set_title(tag)
    ax.set_xlabel("Day")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("pH")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(args.out + ".pdf")
fig.savefig(args.out + ".png", dpi=130)
json.dump(out, open(args.out + ".json", "w"), indent=2)
np.savez(args.out + "_bands.npz", **{f"{t}_{k}": v for t, d in bands.items() for k, v in d.items()})
