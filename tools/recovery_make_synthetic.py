#!/usr/bin/env python3
"""回復テスト用の合成データを作る（共著者レビュー B2、2026-10-12h）。

真値 θ_true = 最終 run 群のうち max logL の seed の MAP（tools/fisher_ident.py と同じ選び方）。
これを run と同じ前進モデル（n_steps・dt・phi_init・K_hill・n_hill）で解き、尤度と同じ手順で観測日に写す:
  組成（ch1・ch2）: φ を観測添字で取り出し clip(1e-10, 1-1e-10) → 行和で正規化（尤度の phi_pred と同じ）
  生存率（ch3）  : Σφψ / Σφ
  pH（ch5）      : 7.5 − 0.74 φ̄_So − 0.48 φ̄_Vd（λ_ch5 = 0 の run では尤度に効かないが形をそろえる）
ノイズ（--noise 倍）: 組成は種ごとの sigma_obs（run_record）で N(0, σ²) を足し、0 で切って行和 1 に正規化し直す
（実データの経路と同じく、観測は「正規化された割合」）。生存率は sigma_obs_viability、pH は 0.15。
ch2（total）は sigma_obs_total。乱数の種は --rng-seed（既定 0）で固定。

出力は JSON 1 つ（tools/recovery_run.py が estimator のローダーに差し込む）。データ CSV は作らない:
estimator は DATA_DIR/experiment_data の固定のファイル名から読み、σ も replicate CSV から作るので、
CSV を書き換えて同じ σ を再現するより、ローダーの戻り値の観測値だけを差し替える方が確実（σ・日・初期値は本物のまま）。

あわせて検算する:
  - 無ノイズ（--noise 0）なら、θ_true での log L はちょうど 0（全項が ≤ 0 なので最大）
  - ノイズありなら、θ_true での log L が「ノイズの分布から期待される値」（モンテカルロ 2000 回）の範囲に入る

  python3 tools/recovery_make_synthetic.py --tag CH [--noise 1] [--rng-seed 0] [--out <path>]
  python3 tools/recovery_make_synthetic.py --tag CS --noise 0
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_common as RC  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tag", required=True, choices=list(RC.GROUPS))
ap.add_argument("--root", default=str(RC.FINAL_ROOT), help="実データの最終 run 群の置き場所")
ap.add_argument("--glob", default=None, help="実データの最終 run 群（既定は recovery_common.GROUPS）")
ap.add_argument("--noise", type=float, default=1.0, help="ノイズの倍率（0 で無ノイズ）")
ap.add_argument("--rng-seed", type=int, default=0)
ap.add_argument("--out", default=None,
                help="出力 JSON（既定 data_5species/main/_runs/recovery/<TAG>_syn_n<noise>_r<seed>/synthetic.json）")
ap.add_argument("--n-mc", type=int, default=2000, help="期待値の検算に使うノイズの引き直し回数")
args = ap.parse_args()

E = RC.import_estimator("cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

tag = args.tag
cond, cult, glob_default = RC.GROUPS[tag]
glob = args.glob or glob_default
runs = RC.load_group(args.root, glob)
best = RC.best_run(runs)
rec = RC.read_record(best)
a = rec["args"]
assert (a["condition"], a["cultivation"]) == (cond, cult), (a["condition"], a["cultivation"])
if not a.get("multichannel"):
    raise SystemExit("最終段（多チャネル）の run を渡すこと")
if a.get("replicate_sigma") or abs(a.get("sigma_scale", 1.0) - 1.0) > 1e-12:
    raise SystemExit("replicate_sigma / sigma_scale≠1 の run は未対応")
theta = RC.read_map(best)
pb = np.array(rec["prior_bounds_final"], dtype=float)
lay = RC.observation_layout(E, rec)
if not np.allclose(np.asarray(lay["sigma_obs"]), np.asarray(rec["sigma_obs"])):
    raise SystemExit(f"sigma_obs が run_record と違う: {lay['sigma_obs']} vs {rec['sigma_obs']}")
mc_real = RC._REAL_LOADERS["mc"](data_dir=E.DATA_DIR / "experiment_data", condition=cond, cultivation=cult,
                                 days_filter=lay["t_days"].tolist())
sig = np.asarray(rec["sigma_obs"], dtype=float)
sig_v = float(mc_real.get("sigma_obs_viability", 0.10))
sig_tot = np.asarray(mc_real.get("sigma_obs_total") if mc_real.get("sigma_obs_total") is not None else 0.08)
lam = RC.lambda_effective(rec)

# ---- 真値での前進計算（尤度と同じ simulate_0d_full）
g = np.asarray(E.simulate_0d_full(jnp.array(theta), n_steps=a["n_steps"], dt=a["dt"],
                                  phi_init=jnp.array(rec["phi_init"]), K_hill=a["K_hill"], n_hill=a["n_hill"]))
rows, idx = lay["rows"], lay["idx_lik"]
phi = np.clip(g[idx, 0:5], 1e-10, 1 - 1e-10)
comp_clean = phi / np.maximum(phi.sum(1, keepdims=True), 1e-12)
phi_raw, psi_raw = g[idx, 0:5], g[idx, 6:11]
viab_clean = (phi_raw * psi_raw).sum(1) / np.maximum(phi_raw.sum(1), 1e-12)

rng = np.random.default_rng(args.rng_seed)


def noisy_comp(clean, s, r):
    x = clean + args.noise * s * r.standard_normal(clean.shape)
    x = np.clip(x, 0.0, None)
    return x / np.maximum(x.sum(1, keepdims=True), 1e-12)


comp = lay["data_real"].copy()  # 尤度に入らない行（ic_day より前）は実データのまま（初期値に使われる）
comp[rows] = noisy_comp(comp_clean, sig, rng) if args.noise > 0 else comp_clean
viab_all = np.asarray(mc_real["data_viability"], dtype=float).copy()
viab_all[rows] = np.clip(viab_clean + args.noise * sig_v * rng.standard_normal(len(rows)), 0.0, 1.0)
total = None
if mc_real.get("data_total") is not None:
    total = np.asarray(mc_real["data_total"], dtype=float).copy()
    total[rows] = noisy_comp(comp_clean, sig_tot, rng) if args.noise > 0 else comp_clean
pH = None
t_pH, idx_pH, keep = RC.pH_layout(E, rec, mc_real, lay["ic_offset"])
if t_pH is not None:
    vals = np.asarray(mc_real["data_pH"], dtype=float).copy()
    ii = idx_pH[keep] - lay["ic_offset"]
    pb_ = g[ii, 0:5] * g[ii, 6:11]
    s_ = np.maximum(pb_.sum(1), 1e-12)
    ph_clean = 7.5 - 0.74 * pb_[:, 0] / s_ - 0.48 * pb_[:, 2] / s_
    sig_pH = float(mc_real.get("sigma_obs_pH", 0.15))
    vals[keep] = ph_clean + args.noise * sig_pH * rng.standard_normal(len(ph_clean))
    pH = {"t_days": t_pH.tolist(), "values": vals.tolist(), "sigma": sig_pH,
          "note": "lambda_ch5=%.2f（0 なら尤度に入らない）" % lam.get(5, 0.0)}

label = f"{tag}_syn_n{args.noise:g}_r{args.rng_seed}"
syn = {
    "label": label, "tag": tag, "condition": cond, "cultivation": cult,
    "source_root": str(Path(args.root).resolve().relative_to(RC.ROOT)) if str(Path(args.root).resolve()).startswith(
        str(RC.ROOT)) else str(args.root),
    "source_glob": glob, "source_runs": [r.name for r in runs], "source_run": best.name,
    "source_max_logL": float(np.load(best / "logL.npy").max()),
    "theta_true": theta.tolist(), "theta_names": RC.NAMES, "free_dims": rec["free_dims"],
    "prior_bounds_final": pb.tolist(), "noise": args.noise, "rng_seed": args.rng_seed,
    "sigma_obs": sig.tolist(), "sigma_obs_viability": sig_v, "sigma_obs_total": np.atleast_1d(sig_tot).tolist(),
    "settings": {k: a.get(k) for k in ("dt", "n_steps", "K_hill", "n_hill", "ic_day", "start_from_day",
                                        "use_exp_init", "lambda_pg", "lambda_late", "lambda_rare",
                                        "n_particles", "n_mutation_steps", "mutation", "use_de_mc",
                                        "mutation_throttle_floor", "max_stages")},
    "lambda_ch_effective": {str(k): v for k, v in lam.items()},
    "phi_init": rec["phi_init"],
    "days": lay["t_days"].tolist(), "likelihood_rows": rows.tolist(), "idx_likelihood": idx.tolist(),
    "composition": comp.tolist(), "composition_noise_free": comp_clean.tolist(),
    "viability": {str(float(d)): float(v) for d, v in zip(lay["t_days"], viab_all)},
    "viability_noise_free": viab_clean.tolist(),
    "total": None if total is None else total.tolist(),
    "pH": pH,
}

# ---- 検算 1: 合成データを差し込んだ「estimator と同じ経路」で尤度を組み、θ_true で評価
RC.install_synthetic_loaders(E, syn)
d2, t2, _, _, _ = E.load_experimental_data(E.DATA_DIR, cond, cult, a.get("start_from_day", 1), normalize=True,
                                           use_exp_init=a["use_exp_init"])
d2, t2 = d2[rows], t2[rows]
mc2 = E.load_multichannel_data(data_dir=E.DATA_DIR / "experiment_data", condition=cond, cultivation=cult,
                               days_filter=t2.tolist())
if t_pH is not None:
    mc2["data_pH"] = np.asarray(mc2["data_pH"])[keep]
    mc2["idx_pH"] = idx_pH[keep] - lay["ic_offset"]
ll = jax.jit(RC.build_likelihood(E, rec, d2, t2, idx, mc2))
ll_true = float(ll(jnp.array(theta)))
# 実データでの同じ点の log L（参考: 実データの max logL と一致するはず）
d_r, t_r = lay["data_real"][rows], lay["t_days"][rows]
mc_r = dict(mc_real)
if t_pH is not None:
    mc_r["data_pH"] = np.asarray(mc_real["data_pH"])[keep]
    mc_r["idx_pH"] = idx_pH[keep] - lay["ic_offset"]
ll_real = float(jax.jit(RC.build_likelihood(E, rec, d_r, t_r, idx, mc_r))(jnp.array(theta)))

# ---- 検算 2: ノイズの分布から期待される log L(θ_true)（同じ重み・正規化・切り捨てでモンテカルロ）
n_obs = len(rows)
W = np.ones((n_obs, 5))
W[:, 4] = a["lambda_pg"]
W[-2:, :] *= a["lambda_late"]  # n_late = 2（make_log_likelihood_jax_ode の既定）


def rare_w(c):
    w = W.copy()
    if a["lambda_rare"] < 1.0:
        w = w * np.where(c.mean(0) < 0.05, a["lambda_rare"], 1.0)[None, :]
    return w


def ll_numpy(c, v, t=None):
    w = rare_w(c)
    out = lam.get(1, 1.0) * -0.5 * np.sum(w * ((c - comp_clean) / sig) ** 2)
    out += lam.get(3, 2.0) * -0.5 * np.sum(((v - viab_clean) / sig_v) ** 2)
    if t is not None:
        out += lam.get(2, 0.5) * -0.5 * np.sum(((t - comp_clean) / sig_tot) ** 2)
    return out


ll_np = ll_numpy(comp[rows], viab_all[rows], None if total is None else total[rows])
mc_vals = None
if args.noise > 0:
    r2 = np.random.default_rng(12345)
    mc_vals = np.array([
        ll_numpy(noisy_comp(comp_clean, sig, r2),
                 np.clip(viab_clean + args.noise * sig_v * r2.standard_normal(n_obs), 0, 1),
                 None if total is None else noisy_comp(comp_clean, sig_tot, r2))
        for _ in range(args.n_mc)])
syn["checks"] = {
    "logL_truth_on_synthetic": ll_true,
    "logL_truth_numpy_replica": float(ll_np),
    "logL_truth_on_real_data": ll_real,
    "source_max_logL_real": syn["source_max_logL"],
    "expected_logL_truth_mc": None if mc_vals is None else {
        "mean": float(mc_vals.mean()), "sd": float(mc_vals.std()),
        "q05": float(np.percentile(mc_vals, 5)), "q95": float(np.percentile(mc_vals, 95)),
        "percentile_of_this_realisation": float((mc_vals < ll_true).mean() * 100), "n": args.n_mc},
    # 正規化・切り捨てを無視した χ² の近似（Σw/2 の分だけ下がる）。希少種の重み（λ_rare）は
    # 観測の平均で決まるので、この実現値の重みで数える
    "expected_logL_truth_chi2_approx": float(-0.5 * args.noise ** 2 * (
        lam.get(1, 1.0) * rare_w(comp[rows]).sum() + lam.get(3, 2.0) * n_obs)) if args.noise > 0 else 0.0,
}

out = Path(args.out) if args.out else RC.MAIN_DIR / "_runs" / "recovery" / label / "synthetic.json"
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(".json.tmp%d" % os.getpid())
with open(tmp, "w") as f:
    json.dump(syn, f, indent=1, ensure_ascii=False)
os.replace(tmp, out)  # 3 seed の job が同時に作っても壊れない
sha = hashlib.sha256(out.read_bytes()).hexdigest()[:12]

c = syn["checks"]
print(f"{label}: 真値 = {best.name} の MAP（実データの max logL {syn['source_max_logL']:.3f}、"
      f"同じ点を組み直した尤度で {ll_real:.3f}）")
print(f"  観測日 {lay['t_days'][rows].tolist()}  添字 {idx.tolist()}  σ_obs {np.round(sig, 4).tolist()}  σ_viab {sig_v:.3f}")
print(f"  log L(θ_true | 合成) = {ll_true:.6f}   （numpy の写し {ll_np:.6f}）")
if mc_vals is not None:
    e = c["expected_logL_truth_mc"]
    print(f"  期待値（MC {args.n_mc} 回）{e['mean']:.3f} ± {e['sd']:.3f}  [5%, 95%] = [{e['q05']:.3f}, {e['q95']:.3f}]"
          f"  この実現値は {e['percentile_of_this_realisation']:.0f} パーセンタイル"
          f"  （χ² 近似 {c['expected_logL_truth_chi2_approx']:.3f}）")
else:
    print("  無ノイズ: 全項 ≤ 0 なので 0 が最大。残差 max |合成 − 予測| = "
          f"{np.max(np.abs(comp[rows] - comp_clean)):.2e}")
print(f"  -> {out}  (sha256 {sha})")
