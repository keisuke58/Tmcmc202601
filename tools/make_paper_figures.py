#!/usr/bin/env python3
"""論文（BMB 版）の図と表の数値を、論文パイプラインの run からまとめて作り直す。

ult が出たら、これ 1 本で原稿の図 5 枚と表の数値をそろえる:

  図（--out-dir、既定 docs/revision/BMB_submission/figures/。原稿と同じファイル名で上書き）
    paper_fig2_phi_transposed.pdf   Fig. 2  事後予測（docs/generate_fig2_phi_version.py の描画をそのまま使う）
    heatmap_A_4cond.pdf             MAP の A（4 条件）
    paper_posterior_violin_sharey.pdf  15 成分の事後（4 条件）
    phase1_vs_phase2_map.pdf        Phase 1 と Phase 2 の MAP（docs/generate_fig_phase_scatter.py をそのまま使う）
    umap_A_matrix_3d.pdf            事後の UMAP（umap-learn が無ければ飛ばす）
  数値（--out-dir/../generated/）
    paper_numbers.json              \\TBD を埋める数値（RMSE・R²・a45 区間・Pg 比・ρ・Δ_F・転移 RMSE・run 表）
    tables.tex                      Table 3（RMSE）・Table 4（R²）・pairwise・cross-prediction の tabular 本体

体裁は既存の図と同じ（thesis_style: usetex・lmodern・9 pt、色も同じ）。
前進モデルは run_record.json の実効値（K_hill・n_hill・dt・n_steps・phi_init）と estimator と同じモジュール
（estimate_paper_jax → hamilton_ode_jax_paper）で計算する。古い図のスクリプトは K_hill=0.05（ゲートあり）の
colab 版モデルを固定で使っていたので、計算部分だけここで置き換えた。

run の選び方:
  各条件で --p2-glob（既定 '{tag}_ult*'）と --p1-glob（既定 '{tag}_p1*'）に合う run（3 seed）を集め、
  MAP は max logL が最大の seed、事後（バイオリン・区間・UMAP・重心距離）は全 seed の粒子を合わせて使う。
  判定（tools/check_paper_runs.py）を通っていない群は使わないこと（ここでは判定しない）。

使い方（GPU サーバー、ult が終わったあと）:
    python3 tools/make_paper_figures.py data_5species/main/_runs/paper_gateoff \\
        --p2-glob '{tag}_ult*' --p1-glob '{tag}_p1*'
    # DS が箱を広げた系列なら: --p2-glob-DS 'DS_ult*wide*' のように条件ごとに上書きできる
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "data_5species" / "main"
_argv = sys.argv
sys.argv = [_argv[0], "--device", os.environ.get("FIG_DEVICE", "cpu")]
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN.parent))
sys.path.insert(0, str(ROOT / "docs"))
sys.path.insert(0, str(ROOT / "tools"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import estimate_paper_jax as EP  # noqa: E402
import jax.numpy as jnp  # noqa: E402

sys.argv = _argv
assert Path(EP._H.__file__).name == "hamilton_ode_jax_paper.py", EP._H.__file__

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDS = ["CS", "CH", "DS", "DH"]
CK = {
    "CS": ("Commensal", "Static"),
    "CH": ("Commensal", "HOBIC"),
    "DS": ("Dysbiotic", "Static"),
    "DH": ("Dysbiotic", "HOBIC"),
}
COND_COLORS = {"CS": "#2166AC", "CH": "#1B7837", "DS": "#FF8C00", "DH": "#B2182B"}
COND_TITLE = {
    "CS": "Commensal static",
    "CH": "Commensal HOBIC",
    "DS": "Dysbiotic static",
    "DH": "Dysbiotic HOBIC",
}
SP = ["So", "An", "Vei", "Fn", "Pg"]
SP_AXIS = ["S.o", "A.n", "Vei", "F.n", "P.g"]
# θ（20 次元）→ 15 成分。並びは原稿の式 (blocks) と古い図と同じ
KEEP_IDX = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]
KEEP_NAME = [
    "a11", "a12", "a22", "a33", "a34", "a44", "a13", "a14",
    "a23", "a24", "a55", "a15", "a25", "a35", "a45",
]  # fmt: skip
VIOLIN_ORDER = [
    "a11", "a22", "a33", "a44", "a55",
    "a12", "a13", "a14", "a15", "a23",
    "a24", "a25", "a34", "a35", "a45",
]  # fmt: skip
N_TRAJ = 100  # Fig. 2 の帯に使う事後サンプル数（古い図と同じ）
DAY_GRID = np.linspace(1, 21, 2501)  # 古い Fig. 2 の時間軸（Day 1 = 初期値）


# ----------------------------------------------------------------------------- runs
def pick_runs(root, pattern):
    dirs = [Path(p) for p in sorted(glob.glob(str(Path(root) / pattern)))]
    dirs = [d for d in dirs if (d / "run_record.json").exists()]
    return dirs


def load_run(d):
    rec = json.load(open(d / "run_record.json"))
    cfg = json.load(open(d / "config.json")) if (d / "config.json").exists() else {}
    th = json.load(open(d / "theta_MAP.json"))
    theta = np.array([th[str(i)] for i in range(20)], dtype=np.float64)
    return {
        "dir": d,
        "rec": rec,
        "cfg": cfg,
        "theta": theta,
        "samples": np.load(d / "samples.npy"),
        "logL": np.load(d / "logL.npy"),
    }


def group(root, pattern, tag):
    runs = [load_run(d) for d in pick_runs(root, pattern.format(tag=tag))]
    if not runs:
        return None
    best = max(runs, key=lambda r: float(r["rec"]["max_logL"]))
    pooled = np.concatenate([r["samples"] for r in runs], axis=0)
    return {"runs": runs, "best": best, "pooled": pooled}


# ----------------------------------------------------------------------------- model
_data_cache = {}


def exp_data(cond, cult, a):
    key = (cond, cult, a["dt"], a["n_steps"])
    if key not in _data_cache:
        data, t_days, _s, _p, _ = EP.load_experimental_data(
            EP.DATA_DIR,
            cond,
            cult,
            a["start_from_day"],
            normalize=True,
            use_exp_init=a["use_exp_init"],
        )
        _, idx = EP.convert_days_to_model_time(t_days, a["dt"], a["n_steps"], day_scale=None)
        _data_cache[key] = (
            np.asarray(data),
            np.asarray(t_days, dtype=float),
            np.clip(idx, 0, a["n_steps"]),
        )
    return _data_cache[key]


_sim_cache = {}


def _sim_fn(rec, full):
    """run の実効値で jit + vmap した前進計算を 1 回だけ作って使い回す（1 本ずつ呼ぶと再コンパイルでメモリが尽きる）。"""
    a = rec["args"]
    key = (full, a["n_steps"], a["dt"], tuple(rec["phi_init"]), a["K_hill"], a["n_hill"])
    if key not in _sim_cache:
        sim = EP.simulate_0d_full if full else EP.simulate_0d
        phi0 = jnp.array(rec["phi_init"], dtype=jnp.float64)

        def one(th):
            return sim(
                th,
                n_steps=a["n_steps"],
                dt=a["dt"],
                phi_init=phi0,
                K_hill=a["K_hill"],
                n_hill=a["n_hill"],
            )

        _sim_cache[key] = jax.jit(jax.vmap(one))
    return _sim_cache[key]


def simulate_full(thetas, rec):
    """θ（n, 20）→ 状態の軌道（n, n_steps+1, 12）。"""
    return np.asarray(_sim_fn(rec, True)(jnp.atleast_2d(jnp.array(thetas))))


def model_days(rec):
    """モデルの時間ステップ → 実験の日。estimator と同じ対応（convert_days_to_model_time）を使う。

    estimator は日をそのまま比例させる: t = day × day_scale、day_scale = 0.95·n_steps·dt / 最終日。
    つまり初期値（Day 1 の実測）はモデル時刻 0 = 「day 0」に置かれ、Day 21 が積分区間の 95% に来る。
    """
    a = rec["args"]
    _, t_days, idx = exp_data(*CK_of(rec), a)
    day_scale = 0.95 * a["n_steps"] * a["dt"] / float(np.max(t_days))
    days = np.arange(a["n_steps"] + 1) * a["dt"] / day_scale
    # 観測日のインデックスが estimator のもの（丸め）と一致するか確かめる
    k_chk = np.clip(np.round(np.asarray(t_days) * day_scale / a["dt"]).astype(int), 0, a["n_steps"])
    if np.max(np.abs(k_chk - np.asarray(idx))) > 1:
        raise RuntimeError(f"モデル時間と日の対応が estimator と合わない: {k_chk} vs {list(idx)}")
    return days


def CK_of(rec):
    a = rec["args"]
    return a["condition"], a["cultivation"]


def predict_fractions(theta, rec):
    """estimator の RMSE と同じ正規化で、観測日の組成を返す（観測日, 5）。"""
    a = rec["args"]
    data, _t, idx = exp_data(*CK_of(rec), a)
    traj = np.asarray(_sim_fn(rec, False)(jnp.atleast_2d(jnp.array(theta))))[0]
    p = np.clip(traj[idx, :], 1e-10, 1 - 1e-10)
    return p / p.sum(axis=1, keepdims=True), data


def theta_to_A(theta):
    A, _b = EP._H.theta_to_matrices(jnp.array(theta))
    return np.asarray(A)


# ----------------------------------------------------------------------------- Fig. 2
def write_fig2_cache(g2, cache_dir):
    """古い Fig. 2 のキャッシュと同じ形（phi_map, phibar_map, trajs, data_norm, d1n, t_days）で書く。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for tag, g in g2.items():
        best = g["best"]
        rec = best["rec"]
        days = model_days(rec)
        gm = simulate_full(best["theta"], rec)[0]
        phi = gm[:, 0:5]
        phibar = gm[:, 0:5] * gm[:, 6:11]
        rng = np.random.default_rng(42)
        S = g["pooled"]
        sel = rng.choice(len(S), size=min(N_TRAJ, len(S)), replace=False)
        trajs = simulate_full(S[sel], rec)[:, :, 0:5]

        def on_grid(y):
            return np.stack([np.interp(DAY_GRID, days, y[:, j]) for j in range(5)], axis=1)

        data, t_days, _idx = exp_data(*CK_of(rec), rec["args"])
        keep = t_days > 1.0 + 1e-9  # Day 1 は初期値として別に渡す
        d1 = np.asarray(rec["phi_init"], dtype=float)
        np.savez(
            cache_dir / f"fig2_phi_{tag}.npz",
            phi_map=on_grid(phi),
            phibar_map=on_grid(phibar),
            trajs=np.stack([on_grid(t) for t in trajs]),
            data_norm=data[keep],
            d1n=d1 / d1.sum(),
            t_days=t_days[keep],
        )


def make_fig2(g2, out_dir, cache_dir):
    import generate_fig2_phi_version as F2

    write_fig2_cache(g2, cache_dir)
    F2.CACHE_DIR = cache_dir
    F2.RUNS = Path("/")
    F2.P2_DIRS = {tag: str(g["best"]["dir"].resolve()) for tag, g in g2.items()}
    F2.FIG_DIR = out_dir
    F2.FIG_DIR2 = cache_dir  # 古いスクリプトは 2 か所に保存する。2 つめは作業用に回す
    # 欠けている条件があっても描けるように（その列は空になる。投稿用は 4 条件そろえてから）
    F2.CONDS = [t for t in CONDS if t in g2]
    F2.generate_fig2_phi_transposed()


# ----------------------------------------------------------------------------- heatmap
def make_heatmap(g2, out_path):
    import thesis_style as ts

    figsize = ts.use(width_frac=1.0, aspect=0.3)
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    As = {t: theta_to_A(g2[t]["best"]["theta"]) for t in CONDS if t in g2}
    vmax = max(np.abs(A[np.triu_indices(5)]).max() for A in As.values())
    cmap = plt.get_cmap("RdBu_r")
    im = None
    for ax, tag in zip(axes, CONDS):
        if tag not in As:
            ax.set_title(f"{COND_TITLE[tag]} (pending)")
            ax.axis("off")
            continue
        A = As[tag]
        M = np.full((5, 5), np.nan)
        iu = np.triu_indices(5)
        M[iu] = A[iu]
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax)
        for i, j in zip(*iu):
            col = "white" if abs(A[i, j]) > 0.6 * vmax else "black"
            # usetex では文字列中の改行が効かないので、名前と値を別々に置く
            ax.text(
                j,
                i - 0.18,
                rf"$\mathbf{{a}}_{{{i + 1}{j + 1}}}$",
                ha="center",
                va="center",
                fontsize=3.8,
                color=col,
            )
            ax.text(
                j,
                i + 0.2,
                rf"\textbf{{{A[i, j]:.2f}}}",
                ha="center",
                va="center",
                fontsize=3.8,
                color=col,
            )
        ax.set_xticks(range(5))
        ax.set_xticklabels(SP_AXIS, rotation=45, ha="right")
        ax.set_yticks(range(5))
        ax.set_yticklabels(SP_AXIS if tag == "CS" else [""] * 5)
        ax.set_title(COND_TITLE[tag])
        ax.tick_params(length=0)
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes), fraction=0.015, pad=0.02)
        cb.set_label(r"$A_{ij}$")
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------- violin
def make_violin(g2, out_path):
    import thesis_style as ts

    figsize = ts.use(width_frac=1.0, aspect=0.62)
    fig, axes = plt.subplots(3, 5, figsize=figsize, sharey=True)
    for k, name in enumerate(VIOLIN_ORDER):
        ax = axes.flat[k]
        col = KEEP_IDX[KEEP_NAME.index(name)]
        for pos, tag in enumerate(CONDS):
            if tag not in g2:
                continue
            x = g2[tag]["pooled"][:, col]
            parts = ax.violinplot([x], positions=[pos], showextrema=False, widths=0.8)
            for pc in parts["bodies"]:
                pc.set_facecolor(COND_COLORS[tag])
                pc.set_edgecolor(COND_COLORS[tag])
                pc.set_alpha(0.85)
            ax.plot(pos, g2[tag]["best"]["theta"][col], "o", color="k", ms=2.2, zorder=3)
        ax.axhline(0, color="k", lw=0.5, zorder=0)
        ax.set_title(rf"$a_{{{name[1:]}}}$")
        ax.set_xticks(range(4))
        ax.set_xticklabels(CONDS, rotation=45)
        if k % 5 == 0:
            ax.set_ylabel(r"$A_{ij}$")
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------- phase scatter
def make_phase_scatter(g1, g2, out_dir):
    import generate_fig_phase_scatter as PS

    PS.RUNS = Path("/")
    PS.P1 = {t: str(g1[t]["best"]["dir"].resolve()) for t in CONDS if t in g1}
    PS.P2 = {t: str(g2[t]["best"]["dir"].resolve()) for t in CONDS if t in g2}
    for t in CONDS:  # 欠けている条件は古い関数が (pending) と描く
        PS.P1.setdefault(t, "__missing__")
        PS.P2.setdefault(t, "__missing__")
    PS.FIG_DIR = out_dir
    PS.XLABEL = "Phase 1 (composition only)"
    PS.YLABEL = r"Phase 2 (with viability, pH)"
    PS.main()


# ----------------------------------------------------------------------------- UMAP
def make_umap(g2, out_path, n_per=1500):
    try:
        import umap
    except ImportError:
        print("umap-learn が無いので UMAP の図は飛ばす（pip install umap-learn）")
        return
    import thesis_style as ts

    rng = np.random.default_rng(0)
    X, lab = [], []
    for tag in CONDS:
        if tag not in g2:
            continue
        S = g2[tag]["pooled"][:, KEEP_IDX]
        sel = rng.choice(len(S), size=min(n_per, len(S)), replace=False)
        X.append(S[sel])
        lab += [tag] * len(sel)
    X = np.concatenate(X)
    lab = np.array(lab)
    E = umap.UMAP(n_components=3, random_state=42).fit_transform(X)
    figsize = ts.use(width_frac=0.5, aspect=1.0)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for tag in CONDS:
        m = lab == tag
        if m.any():
            ax.scatter(
                *E[m].T,
                s=1.5,
                color=COND_COLORS[tag],
                label=f"{tag} ({COND_TITLE[tag].replace('static', 'Static')})",
            )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend(loc="upper left", fontsize=6, markerscale=4)
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------- numbers
def r2_per_species(pred, data):
    out = []
    for j in range(5):
        y, f = data[:, j], pred[:, j]
        if y.max() < 0.02:  # 原稿の Table 4 と同じ基準（全時点で 2% 未満 → 定義しない）
            out.append(None)
            continue
        ss_tot = np.sum((y - y.mean()) ** 2)
        out.append(float(1 - np.sum((y - f) ** 2) / ss_tot) if ss_tot > 0 else None)
    return out


def fit_metrics(g):
    best = g["best"]
    pred, data = predict_fractions(best["theta"], best["rec"])
    rec, cfg = best["rec"], best["cfg"]
    return {
        "run": best["dir"].name,
        "seeds": [r["dir"].name for r in g["runs"]],
        "rmse": float(np.sqrt(np.mean((data - pred) ** 2))),
        "rmse_estimator": cfg.get("rmse"),
        "mae": float(np.mean(np.abs(data - pred))),
        "mean_accept": rec.get("mean_accept"),
        "max_logL": rec.get("max_logL"),
        "log_evidence": rec.get("log_evidence"),
        "n_stages": rec.get("n_stages"),
        "r2": r2_per_species(pred, data),
        "pg_ratio_D21_D15": pg_ratio(pred, best["rec"]),
    }


def pg_ratio(pred, rec):
    _d, t_days, _i = exp_data(*CK_of(rec), rec["args"])
    t = list(np.round(t_days).astype(int))
    return float(pred[t.index(21), 4] / pred[t.index(15), 4])


def posterior_summary(g):
    S = g["pooled"]
    out = {}
    for name, col in zip(KEEP_NAME, KEEP_IDX):
        q = np.percentile(S[:, col], [5, 50, 95])
        out[name] = {"map": float(g["best"]["theta"][col]), "q05": q[0], "q50": q[1], "q95": q[2]}
    return {k: {kk: float(vv) for kk, vv in v.items()} for k, v in out.items()}


def pairwise(g2):
    rows = []
    tags = [t for t in CONDS if t in g2]
    for i, c1 in enumerate(tags):
        for c2 in tags[i + 1 :]:
            A1, A2 = theta_to_A(g2[c1]["best"]["theta"]), theta_to_A(g2[c2]["best"]["theta"])
            rho = float(np.corrcoef(A1.ravel(), A2.ravel())[0, 1])
            dF = float(np.linalg.norm(A1 - A2) / max(np.linalg.norm(A1), np.linalg.norm(A2)))
            m1 = g2[c1]["pooled"][:, KEEP_IDX].mean(0)
            m2 = g2[c2]["pooled"][:, KEEP_IDX].mean(0)
            rows.append(
                {
                    "c1": c1,
                    "c2": c2,
                    "rho": rho,
                    "delta_F": dF,
                    "d15": float(np.linalg.norm(m1 - m2)),
                }
            )
    return rows


def cross_prediction(g2):
    """行 = MAP を取った条件、列 = データの条件（原稿の Table 6 と同じ向き）。"""
    tags = [t for t in CONDS if t in g2]
    M = {}
    for src in tags:
        M[src] = {}
        for tgt in tags:
            rec = g2[tgt]["best"]["rec"]  # データ・初期値・時間は相手の条件のもの
            pred, data = predict_fractions(g2[src]["best"]["theta"], rec)
            M[src][tgt] = float(np.sqrt(np.mean((data - pred) ** 2)))
    return M


def run_table(groups):
    rows = []
    for stage, gs in groups.items():
        for tag in CONDS:
            g = gs.get(tag)
            if g is None:
                continue
            for r in g["runs"]:
                rec = r["rec"]
                nf = max(len(rec.get("free_dims") or []), 1)
                m = rec.get("moves_per_particle_mean")
                rows.append(
                    {
                        "stage": stage,
                        "cond": tag,
                        "run": r["dir"].name,
                        "n_particles": rec["args"]["n_particles"],
                        "n_mutation_steps": rec["args"]["n_mutation_steps"],
                        "n_stages": rec["n_stages"],
                        "moves_per_dim": None if m is None else float(m) * rec["n_stages"] / nf,
                        "max_logL": rec["max_logL"],
                        "total_time_s": rec.get("total_time_s"),
                    }
                )
    return rows


def phase_consistency(g1, g2):
    """Phase 1 と Phase 2 の MAP の相関と RMSD（原稿の Discussion と Phase 散布図の数字）。"""
    out = {}
    for t in CONDS:
        if t in g1 and t in g2:
            m1 = g1[t]["best"]["theta"][KEEP_IDX]
            m2 = g2[t]["best"]["theta"][KEEP_IDX]
            out[t] = {
                "r": float(np.corrcoef(m1, m2)[0, 1]),
                "rmsd": float(np.sqrt(np.mean((m1 - m2) ** 2))),
            }
    return out


def channel_fit(g2):
    """Phase 2 の生存率・pH チャネルの当てはまり（estimator が config.json に書いた値）。"""
    out = {}
    for t, g in g2.items():
        mc = g["best"]["cfg"].get("multichannel_rmse") or {}
        out[t] = {
            "viability_rmse": (mc.get("ch3_viability") or {}).get("rmse"),
            "viability_r2": (mc.get("ch3_viability") or {}).get("r2"),
            "pH_rmse": (mc.get("ch5_pH") or {}).get("rmse"),
            "pH_r2": (mc.get("ch5_pH") or {}).get("r2"),
        }
    return out


def timing(groups):
    """段・条件ごとの 1 seed あたりの計算時間（時間）と GPU。原稿の Table 2 の最終段の行。"""
    out = {}
    for stage, gs in groups.items():
        for t, g in gs.items():
            hrs = [r["rec"].get("total_time_s") for r in g["runs"]]
            hrs = [h / 3600 for h in hrs if h is not None]
            dev = sorted({str(r["cfg"].get("device", "?")) for r in g["runs"]})
            out[f"{stage}_{t}"] = {
                "hours_mean": float(np.mean(hrs)) if hrs else None,
                "hours_max": float(np.max(hrs)) if hrs else None,
                "n_particles": g["best"]["rec"]["args"]["n_particles"],
                "devices": dev,
            }
    return out


def knockout(g2, n_samples):
    """論文の予測（Fn を除くと Pg の後期増加が消える）を Phase 2 の全 seed で計算する（tools/knockout_fn.py）。"""
    import knockout_fn as KO

    out = {}
    for t, g in g2.items():
        rows = [KO.run_dir(r["dir"], n_samples, 0) for r in g["runs"]]
        fr = {
            k: [r[k] for r in rows]
            for k in ("frac_surge_base", "frac_surge_noFn", "frac_surge_a45_0")
        }
        out[t] = {
            "per_seed": rows,
            "frac_surge_base": [min(fr["frac_surge_base"]), max(fr["frac_surge_base"])],
            "frac_surge_noFn": [min(fr["frac_surge_noFn"]), max(fr["frac_surge_noFn"])],
            "frac_surge_a45_0": [min(fr["frac_surge_a45_0"]), max(fr["frac_surge_a45_0"])],
            # 予測の確率 = 「そのまま」でサージが出て、「Fn を除く」と消える割合（seed ごとの差の範囲）
            "prob_suppressed": [
                min(b - n for b, n in zip(fr["frac_surge_base"], fr["frac_surge_noFn"])),
                max(b - n for b, n in zip(fr["frac_surge_base"], fr["frac_surge_noFn"])),
            ],
            "fn_D21_noFn_max": max(r["fn_D21_noFn_max"] for r in rows),
        }
    return out


def identifiability(root, ident_glob, tags):
    """広い箱 [-15, 20] での推定を事前分布なし（prior0）と N(0, 6^2)（prior6）で比べる（原稿 §6.8）。

    成分ごとに: 両者の中央値の差 / 事後 SD、90% 区間、箱の端 5% にある粒子の割合。
    「データで決まる」の目安: 中央値の差が 0.5 SD 以下 かつ 端の割合が 10% 未満（判定 4 と同じ 0.5 SD）。
    """
    out = {}
    for t in tags:
        g0 = group(root, ident_glob.format(tag=t, prior="0"), t)
        g6 = group(root, ident_glob.format(tag=t, prior="6"), t)
        if not (g0 and g6):
            continue
        pb = np.array(g0["best"]["rec"]["prior_bounds_final"], dtype=float)
        res = {}
        for name, col in zip(KEEP_NAME, KEEP_IDX):
            x0, x6 = g0["pooled"][:, col], g6["pooled"][:, col]
            sd = float(np.sqrt(0.5 * (x0.var() + x6.var())))
            lo, hi = pb[col]
            w = hi - lo
            edge = float(np.mean((x0 < lo + 0.05 * w) | (x0 > hi - 0.05 * w)))
            shift = float(abs(np.median(x0) - np.median(x6)) / max(sd, 1e-12))
            res[name] = {
                "median_prior0": float(np.median(x0)),
                "median_prior6": float(np.median(x6)),
                "q05_prior0": float(np.percentile(x0, 5)),
                "q95_prior0": float(np.percentile(x0, 95)),
                "shift_over_sd": shift,
                "edge_frac_prior0": edge,
                "identified": bool(shift <= 0.5 and edge < 0.10),
            }
        out[t] = res
    return out


def fmt(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def tables_tex(num):
    L = [
        "% generated by tools/make_paper_figures.py -- paste the rows into the tables of the manuscript",
        "",
    ]
    L.append("% Table tab:rmse (Phase 1 | Phase 2): RMSE & MAE & acc & maxlogL & lnZ")
    for tag in CONDS:
        p1, p2 = num["phase1"].get(tag), num["phase2"].get(tag)

        def part(p):
            if p is None:
                return "\\TBD{} & & & &"
            return (
                f"{fmt(p['rmse'])} & {fmt(p['mae'])} & {fmt(p['mean_accept'], 2)} & "
                f"${fmt(p['max_logL'], 1)}$ & ${fmt(p['log_evidence'], 1)}$"
            )

        L.append(f"{tag}  & {part(p1)} & {part(p2)}\\\\")
    L += ["", "% Table tab:r2_species (Phase 1 So..Pg | Phase 2 So..Pg); '$-$' = undefined"]
    for tag in CONDS:
        cells = []
        for ph in ("phase1", "phase2"):
            p = num[ph].get(tag)
            cells += (
                ["\\TBD{}"] * 5
                if p is None
                else [("$-$" if v is None else f"${v:.2f}$") for v in p["r2"]]
            )
        L.append(f"{tag}  & " + " & ".join(cells) + "\\\\")
    L += ["", "% Table tab:pairwise: C1 & C2 & rho & Delta_F & d_15D"]
    for r in num["pairwise"]:
        L.append(
            f"{r['c1']} & {r['c2']} & ${r['rho']:+.2f}$ & ${r['delta_F']:.2f}$ & ${r['d15']:.2f}$\\\\"
        )
    L += [
        "",
        "% Table tab:cross_prediction: rows = source MAP, columns = target data (CS CH DS DH)",
    ]
    cp = num["cross_prediction"]
    for src in CONDS:
        if src not in cp:
            continue
        cells = [
            (
                (f"\\textbf{{{cp[src][t]:.3f}}}" if t == src else f"{cp[src][t]:.3f}")
                if t in cp[src]
                else "--"
            )
            for t in CONDS
        ]
        L.append(f"{src} & " + " & ".join(cells) + " \\\\")
    L += [
        "",
        "% Table tab:runs: stage & cond & seed run & N_p & K & stages & moves/param & max lnL & hours",
    ]
    for r in num["runs"]:
        run_tt = r["run"].replace("_", "\\_")
        L.append(
            f"{r['stage']} & {r['cond']} & \\texttt{{{run_tt}}} & {r['n_particles']} & "
            f"{r['n_mutation_steps']} & {r['n_stages']} & {fmt(r['moves_per_dim'], 1)} & "
            f"${fmt(r['max_logL'], 1)}$ & {fmt((r['total_time_s'] or 0) / 3600, 1)}\\\\"
        )
    if num.get("identifiability"):
        L += [
            "",
            "% Table identifiability: param & (cond: median prior0 / prior6, shift/SD, edge, identified)",
        ]
        tags = list(num["identifiability"])
        for name in KEEP_NAME:
            cells = []
            for t in tags:
                d = num["identifiability"][t][name]
                mark = "" if d["identified"] else "$^\\ast$"
                cells.append(
                    f"${d['median_prior0']:+.2f}$ / ${d['median_prior6']:+.2f}${mark} & {d['shift_over_sd']:.2f}"
                )
            L.append(f"$a_{{{name[1:]}}}$ & " + " & ".join(cells) + "\\\\")
        L.append(
            "% $^\\ast$: weakly identified (median shifts by > 0.5 SD with the prior, or >= 10% of mass at the box bounds)"
        )
    if num.get("knockout"):
        L += [
            "",
            "% Knockout (Discussion, A testable prediction): frac. of posterior samples with Pg D21/D15 >= 1.5",
        ]
        for t, k in num["knockout"].items():
            L.append(
                f"% {t}: baseline {k['frac_surge_base'][0]:.2f}--{k['frac_surge_base'][1]:.2f}, "
                f"no Fn {k['frac_surge_noFn'][0]:.2f}--{k['frac_surge_noFn'][1]:.2f}, "
                f"a45=0 {k['frac_surge_a45_0'][0]:.2f}--{k['frac_surge_a45_0'][1]:.2f}, "
                f"P(suppressed) {k['prob_suppressed'][0]:.2f}--{k['prob_suppressed'][1]:.2f}"
            )
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", nargs="?", default=str(MAIN / "_runs" / "paper_gateoff"))
    ap.add_argument(
        "--p2-glob", default="{tag}_ult*", help="Phase 2（最終段）の run。{tag} は CS/CH/DS/DH"
    )
    ap.add_argument("--p1-glob", default="{tag}_p1*", help="Phase 1（組成だけの最終段）の run")
    for t in CONDS:
        ap.add_argument(f"--p2-glob-{t}", default=None, help=f"{t} だけ別の glob を使う")
        ap.add_argument(f"--p1-glob-{t}", default=None)
    ap.add_argument(
        "--out-dir", default=str(ROOT / "docs" / "revision" / "BMB_submission" / "figures")
    )
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["fig2", "heatmap", "violin", "phase", "umap", "knockout", "ident"],
    )
    ap.add_argument(
        "--ident-glob",
        default="{tag}_ident_prior{prior}_*",
        help="識別性の run（{tag} と {prior}=0/6 を埋める）",
    )
    ap.add_argument("--ident-tags", nargs="*", default=["DH", "DS"])
    ap.add_argument(
        "--knockout-tags", nargs="*", default=["DH", "DS"], help="ノックアウトを計算する条件"
    )
    ap.add_argument(
        "--knockout-n", type=int, default=500, help="ノックアウトに使う事後サンプル数（seed ごと）"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir.parent / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = gen_dir / "_cache"

    g1, g2 = {}, {}
    for tag in CONDS:
        p2 = getattr(args, f"p2_glob_{tag}") or args.p2_glob
        p1 = getattr(args, f"p1_glob_{tag}") or args.p1_glob
        a, b = group(args.root, p2, tag), group(args.root, p1, tag)
        if a:
            g2[tag] = a
        if b:
            g1[tag] = b
        print(
            f"{tag}: Phase 2 {len(a['runs']) if a else 0} run ({p2.format(tag=tag)})"
            f"{' MAP=' + a['best']['dir'].name if a else ''} / Phase 1 {len(b['runs']) if b else 0} run"
        )
    if not g2:
        print(f"Phase 2 の run が無い: {args.root}/{args.p2_glob}")
        return 1

    num = {
        "phase2": {t: fit_metrics(g) for t, g in g2.items()},
        "phase1": {t: fit_metrics(g) for t, g in g1.items()},
        "posterior_phase2": {t: posterior_summary(g) for t, g in g2.items()},
        "pairwise": pairwise(g2),
        "cross_prediction": cross_prediction(g2),
        "runs": run_table({"phase1": g1, "phase2": g2}),
        "phase_consistency": phase_consistency(g1, g2),
        "channel_fit_phase2": channel_fit(g2),
        "timing": timing({"phase1": g1, "phase2": g2}),
    }
    if "knockout" not in args.skip:
        num["knockout"] = knockout(
            {t: g2[t] for t in args.knockout_tags if t in g2}, args.knockout_n
        )
    if "ident" not in args.skip:
        num["identifiability"] = identifiability(args.root, args.ident_glob, args.ident_tags)
    for t, p in num["phase2"].items():
        if p["rmse_estimator"] is not None and abs(p["rmse"] - p["rmse_estimator"]) > 1e-6:
            print(
                f"警告 {t}: RMSE がestimator の値と合わない（{p['rmse']:.6f} vs {p['rmse_estimator']:.6f}）"
            )
    (gen_dir / "paper_numbers.json").write_text(
        json.dumps(num, indent=2, ensure_ascii=False, default=str)
    )
    (gen_dir / "tables.tex").write_text(tables_tex(num))
    print(f"数値: {gen_dir / 'paper_numbers.json'}\n表:   {gen_dir / 'tables.tex'}")

    if "fig2" not in args.skip:
        make_fig2(g2, out_dir, cache_dir)
    if "heatmap" not in args.skip:
        make_heatmap(g2, out_dir / "heatmap_A_4cond.pdf")
    if "violin" not in args.skip:
        make_violin(g2, out_dir / "paper_posterior_violin_sharey.pdf")
    if "phase" not in args.skip and g1:
        make_phase_scatter(g1, g2, out_dir)
    if "umap" not in args.skip:
        make_umap(g2, out_dir / "umap_A_matrix_3d.pdf")
    print(f"図: {out_dir}")
    for t in CONDS:
        if t in num["phase2"]:
            p = num["phase2"][t]
            a45 = num["posterior_phase2"][t]["a45"]
            print(
                f"  {t}: RMSE {p['rmse']:.3f}  Pg D21/D15 {p['pg_ratio_D21_D15']:.2f}  "
                f"a45 MAP {a45['map']:+.2f} 90% [{a45['q05']:+.2f}, {a45['q95']:+.2f}]"
            )
    for t, k in (num.get("knockout") or {}).items():
        print(
            f"  knockout {t}: surge base {k['frac_surge_base']}, no Fn {k['frac_surge_noFn']}, "
            f"a45=0 {k['frac_surge_a45_0']}, P(suppressed) {k['prob_suppressed']}"
        )
    for t, d in (num.get("identifiability") or {}).items():
        weak = [n for n, v in d.items() if not v["identified"]]
        print(f"  identifiability {t}: weakly identified {weak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
