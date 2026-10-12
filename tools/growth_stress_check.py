#!/usr/bin/env python3
"""推定した組成・生存率の動きを、拘束された材料点の成長応力に流す（EML 補足、2026-10-12）。

なぜ要るか
---------
本文の限界の節は「点モデルの体積分率が有限要素モデルの成長の変数を駆動できる」と書いている。
それを 1 図で示す。空間の FEM は使わず、**全方向を拘束した 1 点**の閉じた式だけを使う
（pde-fem-biofilm `ansys_usermat/apdl/closed_form_reference.py`。ANSYS のユーザー材料と一致を確認済み）。

モデル（Klempt et al. 2024 の成長則 Eq. 36 と、拘束点の閉じた式）
----------------------------------------------------------------
- 成長: F_g = (1+α) I、 dα/dT* = k_α φ_g(T*)、 T* = day / 21（Klempt 2024 と同じく 1 回の計算の長さで正規化）。
- 駆動 φ_g は 2 通り:
    living — 生きた体積 Σ φ_i ψ_i（群集の生きた割合は生存率の通路で推定に使っている）
    total  — 全体積 Σ φ_i（推定では割合しか使っていないので、全量はモデルの予測）
- 拘束点: F = I なので F_e = I/(1+α)、J_e = (1+α)^-3。偏差成分は 0、体積成分だけ:
    σ = K (J_e − 1) I、 K = E/(3(1−2ν))、 E = E_0 (φ_tot² + f)（Klempt 2024 の剛性）。
  圧力 p = −σ を E_0 で割って出す（E_0 の単位は Klempt 2024 で曖昧なので、条件どうしの比較だけを見る）。
- パラメータ: k_α = 1e-3、ν = 0.49、f = 1e-3（Klempt 2024 Table 2・pde-fem-biofilm の hp_clamped_check.py と同じ）。
  α ≪ 1 なので p は k_α と E_0 に比例する。

使い方（Zenodo README §3 と同じ run の群）:
    R=docs/revision/generated/final_runs
    python3 tools/growth_stress_check.py $R --p2-glob '{tag}_ult*' \
        --p2-glob-DS 'DS_p2_wide80_p18k_wide2_seed*' --p2-glob-DH 'DH_p2_nonarrow_w3_noph_p14k_seed*' \
        --out docs/revision/EML_submission/figures/provisional_2026-10-11/growth_stress
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_figures as MPF  # noqa: E402  (estimator と前進計算を同じ作法で読む)
import eml_style as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

K_ALPHA, NU, F_VOID = 1e-3, 0.49, 1e-3
DAY_END = 21.0


def pressure(alpha, phi_tot):
    """拘束点の圧力 p/E_0（閉じた式）。"""
    K = (phi_tot**2 + F_VOID) / (3.0 * (1.0 - 2.0 * NU))
    Je = (1.0 + alpha) ** -3
    return -K * (Je - 1.0)


def run_condition(g, n_samp, rng):
    rec = g["best"]["rec"]
    days = MPF.model_days(rec)
    keep = days <= DAY_END + 1e-9
    S = g["pooled"]
    sel = rng.choice(len(S), size=min(n_samp, len(S)), replace=False)
    X = MPF.simulate_full(S[sel], rec)[:, keep, :]
    Xm = MPF.simulate_full(g["best"]["theta"], rec)[:, keep, :]
    d = days[keep]
    T = d / DAY_END
    out = {"days": d}
    for name, Y in (("post", X), ("map", Xm)):
        phi, psi = Y[:, :, 0:5], Y[:, :, 6:11]
        tot = phi.sum(-1)
        liv = (phi * psi).sum(-1)
        res = {"phi_tot": tot, "live_frac": liv / tot}
        for drv, src in (("living", liv), ("total", tot)):
            # 台形則で α(T*) = k_α ∫ φ_g dT*
            inc = 0.5 * (src[:, 1:] + src[:, :-1]) * np.diff(T)[None, :]
            alpha = K_ALPHA * np.concatenate([np.zeros((len(src), 1)), np.cumsum(inc, axis=1)], axis=1)
            res[f"p_{drv}"] = pressure(alpha, tot)
        out[name] = res
    return out


def viability_data(rec):
    a = rec["args"]
    _d, t_days, _i = MPF.exp_data(*MPF.CK_of(rec), a)
    mc = MPF.EP.load_multichannel_data(
        data_dir=MPF.EP.DATA_DIR / "experiment_data",
        condition=a["condition"],
        cultivation=a["cultivation"],
        days_filter=list(t_days),
    )
    v = mc.get("data_viability")
    return (np.asarray(t_days, float), np.asarray(v, float)) if v is not None else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--p2-glob", default="{tag}_ult*")
    for t in MPF.CONDS:
        ap.add_argument(f"--p2-glob-{t}", default=None)
    ap.add_argument("--n", type=int, default=200, help="条件ごとの事後サンプル数（3 seed を合わせた中から）")
    ap.add_argument("--out", required=True, help="出力（拡張子なし）: .pdf/.png/.json")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    res, vdat = {}, {}
    for t in MPF.CONDS:
        pat = getattr(args, f"p2_glob_{t}") or args.p2_glob
        g = MPF.group(args.root, pat, t)
        if g is None:
            print(f"[{t}] run が無い: {pat}")
            continue
        res[t] = run_condition(g, args.n, rng)
        vdat[t] = viability_data(g["best"]["rec"])
        res[t]["runs"] = [r["dir"].name for r in g["runs"]]
        print(f"[{t}] {len(g['runs'])} runs, {min(args.n, len(g['pooled']))} samples")

    # ---- 数値（Day 21）
    summ = {"params": {"k_alpha": K_ALPHA, "nu": NU, "f_void": F_VOID, "T_star": "day/21"}}
    for t, r in res.items():
        e = {"runs": r["runs"]}
        for k in ("p_living", "p_total", "live_frac", "phi_tot"):
            v = r["post"][k][:, -1]
            e[k + "_day21"] = {"median": float(np.median(v)), "q05": float(np.quantile(v, 0.05)),
                               "q95": float(np.quantile(v, 0.95)), "map": float(r["map"][k][0, -1])}
        summ[t] = e
    ref = summ["CH"]["p_living_day21"]["median"] if "CH" in summ else None
    if ref:
        for t in res:
            summ[t]["p_living_day21_rel_CH"] = summ[t]["p_living_day21"]["median"] / ref
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(args.out + ".json", "w"), indent=1)

    # ---- 図: (a) 群集の生きた割合（推定に使ったデータ）(b) 成長の圧力（生きた体積で駆動）、細い破線は全体積で駆動
    st.apply()
    fig, axs = plt.subplots(1, 2, figsize=(st.DOUBLE_MM * st.MM, 62 * st.MM))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.86, wspace=0.28)
    for t, r in res.items():
        c, d = st.COND_COL[t], r["days"]
        for ax, k in ((axs[0], "live_frac"), (axs[1], "p_living")):
            y = r["post"][k] if k == "live_frac" else r["post"][k] * 1e3
            lo, mid, hi = np.quantile(y, [0.05, 0.5, 0.95], axis=0)
            ax.fill_between(d, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(d, mid, color=c, lw=1.2, label=rf"\textbf{{{t}}}")
        yt = r["map"]["p_total"][0] * 1e3
        axs[1].plot(d, yt, color=c, lw=0.6, ls=(0, (2, 1.5)), alpha=0.9)
        td, v = vdat[t]
        if td is not None:
            axs[0].scatter(td, v, s=10, color=c, edgecolors=st.INK, linewidths=0.35, zorder=5)
    axs[0].set_ylabel(r"live fraction $\sum_i\varphi_i\psi_i/\sum_i\varphi_i$")
    axs[1].set_ylabel(r"growth pressure $p/E_0$ $[10^{-3}]$")
    for ax in axs:
        ax.set_xlabel("Day")
        ax.set_xlim(0, 21.5)
        ax.set_xticks([0, 3, 6, 10, 15, 21])
        ax.grid(axis="y", color="#ECECEC", lw=0.35)
    axs[0].set_ylim(0, 1)
    axs[0].legend(loc="lower left", ncol=4, fontsize=7, handlelength=1.4, columnspacing=1.0, frameon=False)
    axs[1].text(0.02, 0.97, r"solid: driven by living volume $\sum\varphi_i\psi_i$" "\n"
                r"dashed: driven by total volume $\sum\varphi_i$ (MAP)",
                transform=axs[1].transAxes, va="top", fontsize=6.5, color=st.MUTE)
    axs[0].set_title("(a) community live fraction (data: dots)", fontsize=8, loc="left")
    axs[1].set_title("(b) constrained point, $F_g=(1+\\alpha)I$", fontsize=8, loc="left")
    st.save(fig, args.out)
    print(json.dumps({t: summ[t].get("p_living_day21_rel_CH") for t in res}, indent=1))


if __name__ == "__main__":
    main()
