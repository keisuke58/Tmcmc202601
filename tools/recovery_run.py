#!/usr/bin/env python3
"""回復テストの推定を回す（共著者レビュー B2、2026-10-12h）。estimate_paper_jax.py は変更しない。

estimate_paper_jax をそのまま import し、データローダー（E.load_experimental_data / E.load_multichannel_data）を
合成データ（recovery_make_synthetic.py の JSON）を返す版に差し替えてから E.main() を呼ぶ。
σ・観測日・初期値・時刻の写し方・尤度の重み（λ_ch の自動調整を含む）は estimator の本物の経路を通る。

設定は実データの最終段（合成データの元 run の run_record）にそろえる:
  - 箱 = 元 run の prior_bounds_final（--override-bounds で 15 自由成分すべてを指定。b は estimator が 0 固定）
  - 粒子数・mutation 回数・DE-MC・throttle・λ（ch1/2/3/5・Pg・late・rare）・K_hill・n_hill・dt・n_steps・ic_day
  - **warm start はしない**（--init-from-dir なし）。実データの ult は p2 の事後から半分を初期化したが、
    回復テストでそれをやると「真値（= 実データの MAP）の近くから始める」ことになり、検証にならない。
    TMCMC は箱の一様分布（β = 0）から始める。--posterior-prior-nsigma も渡さない（箱は prior_bounds_final で固定）。

終わったら出力ディレクトリに recovery_record.json（真値・θ_true での log L・照合結果）を書く。

  python3 tools/recovery_run.py --synthetic <synthetic.json> --seed 42 --output-dir <dir> [--device gpu]
  python3 tools/recovery_run.py ... --dry-run        # TMCMC は回さず、渡す引数と θ_true での log L だけ確かめる
  python3 tools/recovery_run.py ... --tiny --device cpu  # 20 粒子・1 段の通し確認（結果は使わない）
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_common as RC  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--synthetic", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--device", default="gpu", choices=["cpu", "gpu", "auto"])
ap.add_argument("--n-particles", type=int, default=None, help="既定は元 run と同じ")
ap.add_argument("--n-mutation-steps", type=int, default=None, help="既定は元 run と同じ")
ap.add_argument("--max-stages", type=int, default=None, help="既定は元 run と同じ（30）")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--tiny", action="store_true", help="20 粒子・mutation 1・1 段（通し確認用。事後ではない）")
args = ap.parse_args()

syn_path = Path(args.synthetic).resolve()
syn = json.load(open(syn_path))
syn_sha = hashlib.sha256(syn_path.read_bytes()).hexdigest()
st = syn["settings"]
pb = np.array(syn["prior_bounds_final"], dtype=float)
theta_true = np.array(syn["theta_true"], dtype=float)

# ---- estimator に渡す引数（元 run の最終段と同じ）
n_part = args.n_particles or st["n_particles"]
n_mut = args.n_mutation_steps or st["n_mutation_steps"]
max_st = args.max_stages or st.get("max_stages") or 30
if args.tiny:
    n_part, n_mut, max_st = 20, 1, 1
ovr = ",".join(f"{i}:{float(pb[i, 0])!r}:{float(pb[i, 1])!r}" for i in syn["free_dims"])
lam_raw = None
# λ_ch は「自動調整前の値」を渡す（自動調整は estimator が同じ規則で掛け直す）。元 run の生の引数を合成データに残していないので
# run_record から読む
rec_src = None
src_dir = RC.ROOT / syn["source_root"] / syn["source_run"]
if (src_dir / "run_record.json").exists():
    rec_src = RC.read_record(src_dir)
    a_src = rec_src["args"]
    lam_raw = [a_src["lambda_ch1"], a_src["lambda_ch2"], a_src["lambda_ch3"], a_src["lambda_ch5"]]
else:
    raise SystemExit(f"元 run の run_record が見つからない: {src_dir}（--root を合わせて合成データを作り直す）")

argv = ["estimate_paper_jax.py",
        "--condition", syn["condition"], "--cultivation", syn["cultivation"],
        "--K-hill", repr(st["K_hill"]), "--n-hill", repr(st["n_hill"]),
        "--dt", repr(st["dt"]), "--n-steps", str(st["n_steps"]),
        "--start-from-day", str(st.get("start_from_day") or 1), "--ic-day", str(st.get("ic_day") or 1),
        "--lambda-pg", repr(st["lambda_pg"]), "--lambda-late", repr(st["lambda_late"]),
        "--lambda-rare", repr(st["lambda_rare"]),
        "--mutation", st["mutation"], "--seed", str(args.seed), "--device", args.device,
        "--mutation-throttle-floor", repr(st["mutation_throttle_floor"]),
        "--multichannel", "--lambda-ch1", repr(lam_raw[0]), "--lambda-ch2", repr(lam_raw[1]),
        "--lambda-ch3", repr(lam_raw[2]), "--lambda-ch5", repr(lam_raw[3]),
        "--n-particles", str(n_part), "--n-mutation-steps", str(n_mut), "--max-stages", str(max_st),
        "--override-bounds", ovr, "--output-dir", str(args.output_dir)]
if st.get("use_exp_init"):
    argv.append("--use-exp-init")
if st.get("use_de_mc"):
    argv.append("--use-de-mc")
for k, flag in (("de_mc_gamma", "--de-mc-gamma"), ("resample_method", "--resample-method")):
    if a_src.get(k) is not None:
        argv += [flag, str(a_src[k])]
for k in ("use_student_t", "estimate_b", "fix_psi", "use_flow", "waste_free", "replicate_sigma"):
    if a_src.get(k):
        raise SystemExit(f"元 run が {k} を使っている。回復テストの写しは未対応")
if (a_src.get("prior_scale") or 0) > 0 or a_src.get("box") is not None:
    raise SystemExit("元 run が --prior-scale / --box を使っている。回復テストの写しは未対応")

print("[recovery] estimator に渡す引数:\n  " + " ".join(argv), flush=True)

E = RC.import_estimator(args.device)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

RC.install_synthetic_loaders(E, syn)

# 尤度関数を横取りして、θ_true での値を記録する（estimator が実際に使う尤度そのもの）
_captured = {}
_orig_make = E.make_log_likelihood_jax_ode


def _capture_make(*a, **kw):
    fn = _orig_make(*a, **kw)
    _captured["fn"] = fn
    _captured["kw"] = {k: (np.asarray(v).tolist() if hasattr(v, "__len__") and not isinstance(v, dict) else v)
                       for k, v in kw.items() if k in ("sigma_obs", "phi_init", "idx_sparse", "lambda_ch",
                                                       "sigma_obs_viability", "n_steps", "dt", "K_hill", "n_hill")}
    ll_t = float(jax.jit(fn)(jnp.array(theta_true)))
    _captured["logL_truth"] = ll_t
    print(f"[recovery] estimator の尤度で log L(θ_true) = {ll_t:.6f}"
          f"（合成データ作成時 {syn['checks']['logL_truth_on_synthetic']:.6f}）", flush=True)
    if abs(ll_t - syn["checks"]["logL_truth_on_synthetic"]) > 1e-6:
        raise SystemExit("θ_true での log L が合成データ作成時と食い違う。設定の写しを確認すること")
    if args.dry_run:
        raise _DryRunDone()
    return fn


class _DryRunDone(Exception):
    pass


E.make_log_likelihood_jax_ode = _capture_make

sys.argv = argv
t0 = time.time()
try:
    E.main()
except _DryRunDone:
    print("[recovery] --dry-run: 尤度の照合まで。TMCMC は回していない")
    sys.exit(0)

# ---- 照合と記録
out = Path(args.output_dir)
rr = RC.read_record(out)
checks = {
    "prior_bounds_equal": bool(np.allclose(np.array(rr["prior_bounds_final"]), pb, atol=1e-12)),
    "sigma_obs_equal": bool(np.allclose(np.array(rr["sigma_obs"]), np.array(syn["sigma_obs"]))),
    "phi_init_equal": bool(np.allclose(np.array(rr["phi_init"]), np.array(syn["phi_init"]))),
    "lambda_ch_equal": {str(k): float(v) for k, v in rr["lambda_ch_effective"].items()}
    == {str(k): float(v) for k, v in syn["lambda_ch_effective"].items()},
    "free_dims_equal": rr["free_dims"] == syn["free_dims"],
}
logL = np.load(out / "logL.npy")
rec = {
    "synthetic": str(syn_path), "synthetic_sha256": syn_sha, "label": syn["label"], "tag": syn["tag"],
    "seed": args.seed, "tiny": args.tiny, "theta_true": syn["theta_true"],
    "logL_truth": _captured.get("logL_truth"), "max_logL": float(logL.max()),
    "max_logL_minus_truth": float(logL.max() - _captured.get("logL_truth", np.nan)),
    "checks": checks, "wall_s": time.time() - t0, "argv": argv,
}
json.dump(rec, open(out / "recovery_record.json", "w"), indent=2, ensure_ascii=False)
bad = [k for k, v in checks.items() if v is not True]
print(f"[recovery] max logL {logL.max():.3f}  log L(θ_true) {rec['logL_truth']:.3f}  "
      f"差 {rec['max_logL_minus_truth']:+.3f}  照合: {'OK' if not bad else 'NG ' + str(bad)}")
