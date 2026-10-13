#!/usr/bin/env python3
"""--two-channel の尤度の検査（2026-10-13）。CPU で 1 分ほど。

1. two_channel=False のとき、今までの尤度と同じ値（既定の経路が変わっていない）。
2. two_channel=True（Phase 2）の値が、simulate_0d_full から手で組んだ値と一致する:
   ch1 = 生きた菌の組成 vs φψ/Σφψ（重み・σ は ch1 と同じ）、ch2 = 全菌 vs φ/Σφ（ch1 と同じ重み）、ch3 = 生きた割合。
3. --fix-psi と一緒（Phase 1）: 種ごとの ψ が (n_obs, 5) で渡り、一様な ψ のときと値が変わる。
   一様な ψ を (n_obs, 5) に並べて渡すと、1 次元で渡したときと同じ値。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "--device", "cpu"]
sys.path.insert(0, str(ROOT / "data_5species" / "main"))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import estimate_paper_jax as E  # noqa: E402

ok = True


def check(c, msg):
    global ok
    print(("OK  " if c else "NG  ") + msg)
    ok &= bool(c)


cond = ("Commensal", "HOBIC")
E.DATA_DIR = E.DATA_DIRS["digitised"]  # 比べる相手（final_runs の CH）が読み取りデータの run なので
d = json.load(open(ROOT / "docs/revision/generated/final_theta_MAP/CH.json"))
th = jnp.array([d[str(i)] for i in range(20)])
data, t_days, sig, phi0, _ = E.load_experimental_data(E.DATA_DIR, *cond, 1, normalize=True, use_exp_init=True)
_, idx = E.convert_days_to_model_time(t_days, 1e-4, 2500, day_scale=None)
idx = np.clip(idx, 0, 2500)
phi0 = np.clip(phi0 / phi0.sum(), 0.001, 0.99)
mc = E.load_multichannel_data(data_dir=E.DATA_DIR / "experiment_data", condition=cond[0],
                              cultivation=cond[1], days_filter=t_days.tolist())
base = dict(data=data, t_days=t_days, idx_sparse=idx, sigma_obs=sig, phi_init=phi0, dt=1e-4, n_steps=2500,
            K_hill=0.0, n_hill=4.0, lambda_pg=5.0, lambda_late=3.0, lambda_rare=0.1,
            data_total=mc["data_total"], sigma_obs_total=mc["sigma_obs_total"],
            data_viability=mc["data_viability"], sigma_obs_viability=mc["sigma_obs_viability"])
lam = {1: 1.0, 2: 1.0, 3: 1.5, 5: 0.0}

# 1. 既定の経路（λ2=0 の論文の設定で、run の max logL と同じ値になるはず）
ll_old = float(E.make_log_likelihood_jax_ode(**base, lambda_ch={1: 1.0, 2: 0.0, 3: 1.5, 5: 0.0})(th))
rec = json.load(open(ROOT / "docs/revision/generated/final_runs/CH_ult_mut150_wide2_noph_sd4_seed7/run_record.json"))
check(abs(ll_old - rec["max_logL"]) < 1e-3, f"既定の経路: logL(MAP) {ll_old:.4f} = run の max logL {rec['max_logL']:.4f}")

# 2. 2 通路（Phase 2）を手で組んだ値と比べる
ll_new = float(E.make_log_likelihood_jax_ode(**base, lambda_ch=lam, two_channel=True)(th))
g = np.array(E.simulate_0d_full(th, n_steps=2500, dt=1e-4, phi_init=jnp.array(phi0), K_hill=0.0, n_hill=4.0))
phi = np.clip(g[idx, 0:5], 1e-10, 1 - 1e-10)
pb = phi * np.clip(g[idx, 6:11], 1e-10, 1.0)
nb, nt = pb / pb.sum(1, keepdims=True), phi / phi.sum(1, keepdims=True)
n_obs = data.shape[0]
w = np.ones((n_obs, 5))
w[:, 4] = 5.0
w[-2:, :] *= 3.0
rare = data.mean(0) < 0.05
w[:, rare] *= 0.1
ch1 = -0.5 * np.sum(w * ((data - nb) / sig) ** 2)
ch2 = -0.5 * np.sum(w * ((np.asarray(mc["data_total"]) - nt) / np.asarray(mc["sigma_obs_total"])) ** 2)
ch3 = -0.5 * np.sum(((np.asarray(mc["data_viability"]) - pb.sum(1) / phi.sum(1)) / mc["sigma_obs_viability"]) ** 2)
manual = ch1 + ch2 + 1.5 * ch3
check(abs(ll_new - manual) < 1e-6 * max(1, abs(manual)), f"2 通路 Phase 2: {ll_new:.4f} = 手で組んだ値 {manual:.4f}")
check(abs(ll_new - ll_old) > 1e-3, "2 通路では値が変わる")

# 3. Phase 1（ψ 固定）
live = np.interp(t_days, [1, 3, 6, 10, 15, 21], np.asarray(mc["data_viability"]) if len(mc["data_viability"]) == 6
                 else np.full(6, 0.5))
psi_sp = E.species_viability_from_data(*cond, t_days, np.clip(live, 0.01, 0.99))
check(psi_sp.shape == (n_obs, 5) and np.all((psi_sp >= 0.01) & (psi_sp <= 0.99)), f"種ごとの ψ: 形 {psi_sp.shape}")
b1 = dict(base)
b1.pop("data_viability"), b1.pop("sigma_obs_viability")
l_1d = float(E.make_log_likelihood_jax_ode(**b1, lambda_ch=lam, psi_fixed=live, two_channel=True)(th))
l_2d = float(E.make_log_likelihood_jax_ode(**b1, lambda_ch=lam, psi_fixed=np.tile(live[:, None], (1, 5)),
                                           two_channel=True)(th))
l_sp = float(E.make_log_likelihood_jax_ode(**b1, lambda_ch=lam, psi_fixed=psi_sp, two_channel=True)(th))
check(abs(l_1d - l_2d) < 1e-9, f"一様な ψ: 1 次元 {l_1d:.4f} = (n_obs, 5) {l_2d:.4f}")
check(abs(l_sp - l_1d) > 1e-3, f"種ごとの ψ で値が変わる: {l_sp:.4f} vs {l_1d:.4f}")
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
