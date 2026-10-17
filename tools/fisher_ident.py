#!/usr/bin/env python3
"""同定性の独立な確認: MAP での尤度の曲率（観測 Fisher 情報量）（2026-10-12、共著者レビュー B2）。

補足 S4・図 3 の分類は事後標本の広がり（sd が箱の一様分布の sd の 0.8 倍未満なら「同定」）による。
ここでは標本を使わず、MAP での −log L の Hessian H（15 自由次元）から
  - Laplace 近似の sd_i = sqrt((H^{-1})_ii) と、箱の一様分布の sd との比（成分ごと）
  - H の固有値の分解: 曲率が「箱の一様分布の典型 sd で 1 nat 以上効く」方向の数（決まる方向の数）
を出し、事後の分類と比べる。尤度は run と同じ設定（tools/fit_ceiling.py と同じ組み方）。

  python3 tools/fisher_ident.py [--root docs/revision/generated/final_runs]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT / "docs/revision/generated/final_runs"))
ap.add_argument("--spec", default=None, help="JSON（make_supplementary_tables と同じ形: 条件 → 段 → glob）。ult の glob を使う")
ap.add_argument("--out", default=str(ROOT / "docs/revision/EML_submission/figures/provisional_2026-10-11/fisher_ident.json"))
args = ap.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT / "data_5species" / "main"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp  # noqa: E402

import estimate_paper_jax as E  # noqa: E402

GROUPS = {"CS": ("Commensal", "Static", "CS_ult_2c_raw_seed*"),
          "CH": ("Commensal", "HOBIC", "CH_ult_2c_raw_a33w_seed*"),
          "DS": ("Dysbiotic", "Static", "DS_ult_2c_raw_seed*"),
          "DH": ("Dysbiotic", "HOBIC", "DH_ult_2c_raw_seed*")}
if args.spec:  # raw の run など（2026-10-13）
    _sp = json.load(open(args.spec))
    GROUPS = {t: (c, u, _sp[t].get("final") or _sp[t]["ult"]) for t, (c, u, _p) in GROUPS.items()
              if t in _sp and ("final" in _sp[t] or "ult" in _sp[t])}  # "final": 最終群が ult でない条件（DH は p2）
NAMES = E.THETA_NAMES
UNIDENT_RATIO = 0.8
out = {}
for tag, (cond, cult, pat) in GROUPS.items():
    runs = sorted(Path(args.root).glob(pat))
    if not runs:  # 最終段がまだ無い条件（2026-10-16）
        print(f"{tag}: run なし（{pat}）→ 飛ばす")
        continue
    best = max(runs, key=lambda r: float(np.load(r / "logL.npy").max()))
    rec = json.load(open(best / "run_record.json"))
    a = rec["args"]
    E.DATA_DIR = E.data_dir_for(a)  # run が読んだデータ
    # 最終 MAP（final_theta_MAP/<tag>.json、磨いた点があればそれ）を優先。無ければ最良 seed の粒子の最良値（2026-10-17）
    _fm = ROOT / "docs/revision/generated/final_theta_MAP" / f"{tag}.json"
    th = json.load(open(_fm if _fm.exists() else best / "theta_MAP.json"))
    th = np.array([th[str(i)] for i in range(20)])
    pb = np.array(rec["prior_bounds_final"], dtype=float)
    data, t_days, sig, _, _ = E.load_experimental_data(E.DATA_DIR, cond, cult, 1, normalize=True, use_exp_init=True)
    _, idx = E.convert_days_to_model_time(t_days, a["dt"], a["n_steps"], day_scale=None)
    idx = np.clip(idx, 0, a["n_steps"])
    mc = E.load_multichannel_data(data_dir=E.DATA_DIR / "experiment_data", condition=cond, cultivation=cult,
                                  days_filter=t_days.tolist())
    lam = {int(k): float(v) for k, v in rec.get("lambda_ch_effective", {}).items()} or {1: 1.0, 2: 0.0, 3: 2.0, 5: 0.0}
    ll = E.make_log_likelihood_jax_ode(
        data=data, t_days=t_days, idx_sparse=idx, sigma_obs=np.array(rec["sigma_obs"]),
        phi_init=np.array(rec["phi_init"]), psi_init=rec.get("psi_init"), dt=a["dt"], n_steps=a["n_steps"], K_hill=a["K_hill"],
        n_hill=a["n_hill"], lambda_pg=a["lambda_pg"], lambda_late=a["lambda_late"],
        use_student_t=a.get("use_student_t", False), student_t_nu=a.get("student_t_nu", 5.0),
        lambda_rare=a["lambda_rare"], psi_fixed=None,
        data_total=mc.get("data_total"), sigma_obs_total=mc.get("sigma_obs_total"),
        data_viability=mc.get("data_viability"), sigma_obs_viability=mc.get("sigma_obs_viability", 0.10),
        lambda_ch=lam, two_channel=bool(a.get("two_channel", False)))
    l0 = float(jax.jit(ll)(jnp.array(th)))
    free = [i for i in range(20) if pb[i, 1] > pb[i, 0]]

    def nll_free(z):
        x = jnp.array(th).at[jnp.array(free)].set(z)
        return -ll(x)

    H = np.asarray(jax.jit(jax.hessian(nll_free))(jnp.array(th[free])))
    H = 0.5 * (H + H.T)
    unif = (pb[free, 1] - pb[free, 0]) / np.sqrt(12.0)
    # 箱の一様 sd で無次元化した曲率: 固有値 >= 1 なら、その方向に一様 sd だけ動くと log L が 0.5 nat 以上下がる
    Hs = H * np.outer(unif, unif)
    ev = np.linalg.eigvalsh(Hs)
    n_stiff = int(np.sum(ev >= 1.0 / UNIDENT_RATIO ** 2))
    # 成分ごとの Laplace sd（負の固有値は 0 に切って擬似逆行列）
    w, V = np.linalg.eigh(Hs)
    winv = np.where(w > 1e-8, 1.0 / np.maximum(w, 1e-8), 0.0)
    cov_s = (V * winv) @ V.T
    ratio = np.sqrt(np.clip(np.diag(cov_s), 0, None))  # Laplace sd / 一様 sd
    ratio = np.where(np.diag(cov_s) > 0, ratio, np.inf)
    S = np.concatenate([np.load(r / "samples.npy") for r in runs])
    post_ratio = S[:, free].std(0) / unif
    ident_post = [NAMES[free[k]] for k in range(len(free)) if post_ratio[k] < UNIDENT_RATIO]
    ident_lap = [NAMES[free[k]] for k in range(len(free)) if ratio[k] < UNIDENT_RATIO]
    agree = len(set(ident_post) & set(ident_lap))
    out[tag] = {"run": best.name, "logL_MAP": l0, "n_free": len(free), "eigenvalues_scaled": ev.tolist(),
                "n_stiff_directions": n_stiff, "identified_posterior": ident_post, "identified_laplace": ident_lap,
                "n_negative_eigen": int(np.sum(ev < 0))}
    print(f"{tag}: logL(MAP) {l0:.2f}  決まる方向（曲率）{n_stiff}/{len(free)}  負の固有値 {int(np.sum(ev < 0))}"
          f"  | 事後で同定 {len(ident_post)} {ident_post}  | Laplace で同定 {len(ident_lap)} {ident_lap}  一致 {agree}")
json.dump(out, open(args.out, "w"), indent=2)
