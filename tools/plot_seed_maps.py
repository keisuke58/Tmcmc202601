#!/usr/bin/env python3
"""3 seed の MAP を重ねた組成の図（2026-10-11）。

図 2 では MAP（max logL 最大の 1 seed）の線と、3 seed の粒子を合わせた事後予測の帯を描く。
CS では MAP の線が帯の外に出たので、seed ごとの MAP を重ねて、どの seed がどこを見ているかを示す。

  python3 tools/plot_seed_maps.py [--root docs/revision/generated/final_runs] [--n-post 200]

各条件 1 列 × 5 種。線は seed ごとの MAP（組成 = φ_i / Σφ）、帯は 3 seed 合算の事後予測の 10–90%、
点は実測（正規化した中央値）。凡例に seed ごとの max logL と組成 RMSE を出す。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(ROOT / "docs/revision/generated/final_runs"))
ap.add_argument("--n-post", type=int, default=200, help="帯に使う事後標本の数（3 seed 合算から）")
ap.add_argument("--out", default=str(ROOT / "docs/revision/EML_submission/figures/provisional_2026-10-11/seed_maps_4cond"))
ap.add_argument("--spec", default=None, help='JSON {"CS": "<seed の前までの run 名>", ...}（例 "CS_ult_2c_raw_seed"）。既定は下の GROUPS')
ap.add_argument("--stage-label", default=None, help="列見出しの段の表記（--spec のとき、既定 ult）")
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

# 体裁: 本文・ほかの図と同じ Latin Modern（usetex）。LaTeX が無ければ通常の描画
try:
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage{lmodern}", "font.size": 9})
    plt.figure(); plt.text(0, 0, r"$x$"); plt.savefig("/dev/null", format="pdf"); plt.close()
except Exception:  # noqa: BLE001
    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 9})

import estimate_paper_jax as E  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_pred as CP  # noqa: E402

GROUPS = [
    ("CS", "Commensal", "Static", "CS_ult_2c_raw_seed", "ult"),
    ("CH", "Commensal", "HOBIC", "CH_ult_2c_raw_a33w_seed", "ult"),
    ("DS", "Dysbiotic", "Static", "DS_ult_2c_raw_seed", "ult"),
    ("DH", "Dysbiotic", "HOBIC", "DH_ult_2c_raw_seed", "ult"),
]
if args.spec:  # raw の run など（2026-10-13）
    _sp = json.load(open(args.spec))
    def _lab(prefix):  # run 名から段の表記（DH は p2 が最終、2026-10-18）
        for k, v in (("_ult_", "stage (iv)"), ("_p2_", "stage (iii)"), ("_p1_", "stage (ii)"), ("_pilot_", "stage (i)")):
            if k in prefix:
                return v
        return args.stage_label or "ult"
    GROUPS = [(t, c, u, _sp[t], _lab(_sp[t])) for (t, c, u, _p, _s) in GROUPS if t in _sp]
SEEDS = [42, 123, 7]
SEED_COL = {42: "#d62728", 123: "#1f77b4", 7: "#2ca02c"}
SP = ["S. oralis", "A. naeslundii", "Veillonella", "F. nucleatum", "P. gingivalis"]


def comp_traj(theta, rec):
    """生きた菌の組成と比べる予測（2 通路なら φψ/Σφψ、それ以外 φ/Σφ、tools/comp_pred.py）。"""
    a = rec["args"]
    g = E.simulate_0d_full(jnp.array(theta), n_steps=a["n_steps"], dt=a["dt"],
                           phi_init=jnp.array(rec["phi_init"]), psi_init=rec.get("psi_init"), K_hill=a["K_hill"], n_hill=a["n_hill"])
    return CP.fractions(np.asarray(g), CP.is_two_channel(rec))["viable"]


fig, axes = plt.subplots(5, 4, figsize=(13, 13), sharex=True, sharey="row")
summary = {}
for c, (tag, cond, cult, prefix, stage) in enumerate(GROUPS):
    runs = {s: Path(args.root) / f"{prefix}{s}" for s in SEEDS}
    if not all((r / "run_record.json").exists() for r in runs.values()):  # 最終段がまだ無い条件（2026-10-16）
        print(f"{tag}: run なし（{prefix}*）→ pending")
        axes[2, c].text(0.5, 0.5, "pending", transform=axes[2, c].transAxes, ha="center", va="center", color="0.5")
        axes[0, c].set_title(tag)
        continue
    recs = {s: json.load(open(r / "run_record.json")) for s, r in runs.items()}
    a = recs[SEEDS[0]]["args"]
    data, t_days, _, _, _ = E.load_experimental_data(E.data_dir_for(a), cond, cult, 1, normalize=True, use_exp_init=True)
    _, idx = E.convert_days_to_model_time(t_days, a["dt"], a["n_steps"], day_scale=None)
    idx = np.clip(idx, 0, a["n_steps"])
    day_per_step = (t_days[-1] - 0) / idx[-1]  # convert_days_to_model_time は線形（day 0 ↔ step 0）
    days_axis = np.arange(a["n_steps"] + 1) * day_per_step

    # 事後の帯（3 seed 合算から一様に抜く）
    S = np.concatenate([np.load(r / "samples.npy") for r in runs.values()])
    rng = np.random.default_rng(0)
    pick = S[rng.choice(len(S), size=min(args.n_post, len(S)), replace=False)]
    rec0 = recs[SEEDS[0]]
    sim = jax.jit(jax.vmap(lambda th: E.simulate_0d_full(th, n_steps=a["n_steps"], dt=a["dt"],
                                                         phi_init=jnp.array(rec0["phi_init"]), psi_init=rec0.get("psi_init"),
                                                         K_hill=a["K_hill"], n_hill=a["n_hill"])))
    P = CP.fractions(np.asarray(sim(jnp.array(pick))), CP.is_two_channel(rec0))["viable"]
    lo, med, hi = np.percentile(P, [10, 50, 90], axis=0)
    rmse_med = float(np.sqrt(np.mean((data - med[idx]) ** 2)))

    summary[tag] = {"posterior_median_rmse": rmse_med, "seeds": {}}
    for s in SEEDS:
        th = json.load(open(runs[s] / "theta_MAP.json"))
        th = np.array([th[str(i)] for i in range(20)])
        tr = comp_traj(th, recs[s])
        rmse = float(np.sqrt(np.mean((data - tr[idx]) ** 2)))
        mlog = float(np.load(runs[s] / "logL.npy").max())
        summary[tag]["seeds"][s] = {"max_logL": mlog, "rmse": rmse}
        for k in range(5):
            axes[k, c].plot(days_axis, tr[:, k], color=SEED_COL[s], lw=1.4,
                            label=f"seed {s}: logL {mlog:.2f}, RMSE {rmse:.3f}" if k == 0 else None)
    for k in range(5):
        ax = axes[k, c]
        ax.fill_between(days_axis, lo[:, k], hi[:, k], color="0.6", alpha=0.3, lw=0,
                        label=r"posterior 10--90\% (3 seeds)" if k == 0 else None)
        ax.plot(days_axis, med[:, k], color="0.3", lw=1.0, ls="--",
                label=f"posterior median, RMSE {rmse_med:.3f}" if k == 0 else None)
        ax.plot(t_days, data[:, k], "ko", ms=4)
        if c == 0:
            ax.set_ylabel(r"\textit{%s}" % SP[k] if plt.rcParams["text.usetex"] else SP[k])
        ax.set_ylim(-0.02, 1.0)
        ax.grid(alpha=0.2)
    axes[0, c].set_title(f"{tag} ({stage})")
    axes[0, c].legend(fontsize=6.5, loc="upper right")
    axes[-1, c].set_xlabel("Day")

fig.suptitle("MAP of each seed (lines), pooled posterior band (grey), data (dots)", fontsize=11)
fig.tight_layout()
fig.savefig(args.out + ".pdf")
fig.savefig(args.out + ".png", dpi=130)
json.dump(summary, open(args.out + ".json", "w"), indent=2)
print(json.dumps(summary, indent=2))
