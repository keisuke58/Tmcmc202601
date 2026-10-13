#!/usr/bin/env python3
"""合成データによる回復テスト（共著者レビュー B2「同定性が示されていない」への対応、2026-10-12h）の共通部品。

  - 実データの最終 run 群（final_runs）の読み込みと「max logL の seed」の選択
  - estimate_paper_jax.py の import（GPU/CPU の切り替えは import 前の sys.argv で決まるので、その前に呼ぶ）
  - 実データと同じ設定で尤度を組み直す（tools/fisher_ident.py と同じ組み方。データだけ差し替えられる）
  - 合成データ（recovery_make_synthetic.py の JSON）をローダーの戻り値に差し込む monkeypatch
  - 同定性の分類（tools/make_supplementary_tables.py の ident_table と同じ規則）

estimate_paper_jax.py は変更しない。データは E.load_experimental_data / E.load_multichannel_data
（モジュール大域の名前）を差し替えて渡す。E.main() はこの名前を実行時に引くので、そのまま合成データで走る。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "data_5species" / "main"
FINAL_ROOT = ROOT / "docs" / "revision" / "generated" / "final_runs"

NAMES = ["a11", "a12", "a22", "b1", "b2", "a33", "a34", "a44", "b3", "b4",
         "a13", "a14", "a23", "a24", "a55", "b5", "a15", "a25", "a35", "a45"]
# 本文の並び（make_supplementary_tables.py・make_eml_figs56.py と同じ）
ORDER = ["a11", "a12", "a22", "a33", "a34", "a44", "a13", "a14", "a23", "a24", "a55", "a15", "a25", "a35", "a45"]
B_DIMS = [3, 4, 8, 9, 15]
UNIDENT_RATIO = 0.8  # tools/check_paper_runs.py・make_supplementary_tables.py と同じ

# 最終群（2026-10-12 時点）。DS・DH は ult が判定を通ったら --glob で差し替える
GROUPS = {
    "CS": ("Commensal", "Static", "CS_ult_wide2_sd4_seed*"),
    "CH": ("Commensal", "HOBIC", "CH_ult_mut150_wide2_noph_sd4_seed*"),
    "DS": ("Dysbiotic", "Static", "DS_p2_wide80_p18k_wide2_seed*"),
    "DH": ("Dysbiotic", "HOBIC", "DH_p2_nonarrow_w3_noph_p14k_seed*"),
}
COND = {"CS": ("Commensal", "Static"), "CH": ("Commensal", "HOBIC"),
        "DS": ("Dysbiotic", "Static"), "DH": ("Dysbiotic", "HOBIC")}


# ---------------------------------------------------------------- 実 run
def load_group(root, glob):
    runs = sorted(Path(root).glob(glob))
    runs = [r for r in runs if (r / "samples.npy").exists() and (r / "run_record.json").exists()]
    if not runs:
        raise FileNotFoundError(f"run が無い: {root}/{glob}")
    return runs


def best_run(runs):
    """max logL が最大の seed（tools/fisher_ident.py・final_theta_MAP と同じ選び方）。"""
    return max(runs, key=lambda r: float(np.load(r / "logL.npy").max()))


def read_record(run):
    return json.load(open(Path(run) / "run_record.json"))


def read_map(run):
    th = json.load(open(Path(run) / "theta_MAP.json"))
    return np.array([th[str(i)] for i in range(20)], dtype=float)


# ---------------------------------------------------------------- estimator の import
def import_estimator(device="cpu"):
    """estimate_paper_jax を import して返す。device は import 時の sys.argv で決まる（estimator の仕様）。"""
    if "estimate_paper_jax" in sys.modules:
        return sys.modules["estimate_paper_jax"]
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    saved = sys.argv
    sys.argv = [saved[0], "--device", device]
    sys.path.insert(0, str(MAIN_DIR))
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        if device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        import estimate_paper_jax as E
    finally:
        sys.argv = saved
    return E


# ---------------------------------------------------------------- 観測日と時刻の対応（尤度と同じ）
def observation_layout(E, rec):
    """実データの読み込みから、尤度が使う観測日・時刻添字・初期値を run と同じ手順で作る。

    戻り値の rows は load_experimental_data が返す行（Day 1 は use_exp_init で除外済み）のうち
    尤度に入る行の番号。ic_day > 1 なら初期値の日までの行は尤度に入らない（estimator と同じ切り方）。
    """
    remember_real_loaders(E)
    a = rec["args"]
    cond, cult = a["condition"], a["cultivation"]
    data, t_days, sig, phi_exp, meta = _REAL_LOADERS["exp"](
        E.DATA_DIR, cond, cult, a.get("start_from_day", 1), normalize=True, use_exp_init=a["use_exp_init"])
    _, idx = E.convert_days_to_model_time(t_days, a["dt"], a["n_steps"], day_scale=None)
    idx = np.clip(idx, 0, a["n_steps"])
    ic_day = int(a.get("ic_day") or 1)
    rows = np.arange(len(t_days))
    offset = 0
    if ic_day > 1:
        k = int(np.where(np.asarray(t_days) == ic_day)[0][0])
        offset = int(idx[k])
        rows = rows[k + 1:]
    return dict(data_real=np.asarray(data), t_days=np.asarray(t_days, dtype=float), sigma_obs=sig,
                phi_init_exp=np.asarray(phi_exp), idx_full=np.asarray(idx), rows=rows,
                idx_lik=np.asarray(idx)[rows] - offset, ic_offset=offset, ic_day=ic_day)


def lambda_effective(rec):
    lam = {int(k): float(v) for k, v in rec.get("lambda_ch_effective", {}).items()}
    return lam or {1: 1.0, 2: 0.0, 3: 2.0, 5: 0.0}


def build_likelihood(E, rec, data, t_days, idx_lik, mc):
    """run_record の設定で尤度を組む（fisher_ident.py と同じ。data・mc だけ差し替え可能）。"""
    a = rec["args"]
    kw = {}
    if a.get("multichannel"):
        kw = dict(data_total=mc.get("data_total"), sigma_obs_total=mc.get("sigma_obs_total"),
                  data_viability=mc.get("data_viability"),
                  sigma_obs_viability=mc.get("sigma_obs_viability", 0.10), lambda_ch=lambda_effective(rec))
        if mc.get("data_pH") is not None:
            kw.update(data_pH=mc["data_pH"], idx_pH=mc["idx_pH"], sigma_obs_pH=mc.get("sigma_obs_pH", 0.15))
    if a.get("fix_psi"):
        raise NotImplementedError("fix_psi の段（pilot・p1）は回復テストの対象外（最終段は多チャネル）")
    return E.make_log_likelihood_jax_ode(
        data=data, t_days=t_days, idx_sparse=idx_lik, sigma_obs=np.asarray(rec["sigma_obs"]),
        phi_init=np.asarray(rec["phi_init"]), dt=a["dt"], n_steps=a["n_steps"], K_hill=a["K_hill"],
        n_hill=a["n_hill"], lambda_pg=a["lambda_pg"], lambda_late=a["lambda_late"],
        use_student_t=a.get("use_student_t", False), student_t_nu=a.get("student_t_nu", 5.0),
        lambda_rare=a["lambda_rare"], psi_fixed=None, two_channel=bool(a.get("two_channel", False)), **kw)


def pH_layout(E, rec, mc_real, ic_offset):
    """pH の観測時刻の添字（estimator と同じ: pH の時刻だけで day_scale を決める）。"""
    a = rec["args"]
    if mc_real.get("data_pH") is None:
        return None, None, None
    t = np.asarray(mc_real["t_pH_days"], dtype=float)
    _, idx = E.convert_days_to_model_time(t, a["dt"], a["n_steps"], day_scale=None)
    idx = np.clip(idx, 0, a["n_steps"])
    keep = np.ones(len(idx), dtype=bool) if ic_offset == 0 else idx > ic_offset
    return t, idx, keep


# ---------------------------------------------------------------- monkeypatch
_REAL_LOADERS = {}


def remember_real_loaders(E):
    _REAL_LOADERS.setdefault("exp", E.load_experimental_data)
    _REAL_LOADERS.setdefault("mc", E.load_multichannel_data)


def install_synthetic_loaders(E, syn):
    """E のデータローダーを合成データを返す版に差し替える。

    返すものは本物のローダーの戻り値のうち「観測値」だけを差し替えたもの。σ・観測日・初期値（Day 1 の絶対量）・
    メタデータは本物のまま（= 実データの推定と同じ観測演算子・ノイズ水準・時刻・初期状態）。
    """
    remember_real_loaders(E)
    real_exp, real_mc = _REAL_LOADERS["exp"], _REAL_LOADERS["mc"]
    cond, cult = syn["condition"], syn["cultivation"]
    comp = np.asarray(syn["composition"], dtype=float)
    syn_days = np.asarray(syn["days"], dtype=float)
    viab = {float(k): float(v) for k, v in syn["viability"].items()}
    total = np.asarray(syn["total"], dtype=float) if syn.get("total") is not None else None

    def load_experimental_data(data_dir, condition, cultivation, start_from_day=1, normalize=False,
                               use_exp_init=False):
        if (condition, cultivation) != (cond, cult):
            raise ValueError(f"合成データは {cond} {cult}、要求は {condition} {cultivation}")
        if not normalize:
            raise ValueError("合成データは正規化済み（normalize=True の経路）しか用意していない")
        data, t_days, sig, phi_exp, meta = real_exp(data_dir, condition, cultivation, start_from_day,
                                                   normalize=normalize, use_exp_init=use_exp_init)
        if not np.array_equal(np.asarray(t_days, dtype=float), syn_days):
            raise ValueError(f"観測日が合成データと違う: {list(t_days)} vs {list(syn_days)}")
        meta = dict(meta, synthetic=syn.get("label"))
        print(f"[recovery] 合成データを渡す: {syn.get('label')}  days={syn_days.tolist()}", flush=True)
        return comp.copy(), t_days, sig, phi_exp, meta

    def load_multichannel_data(data_dir, condition, cultivation, days_filter, normalize=False):
        res = real_mc(data_dir=data_dir, condition=condition, cultivation=cultivation, days_filter=days_filter,
                      normalize=normalize)
        days_filter = [float(d) for d in days_filter]
        if res.get("data_viability") is not None:
            res["data_viability"] = np.array([viab[d] for d in days_filter])
        if res.get("data_total") is not None and total is not None:
            pos = {d: i for i, d in enumerate(syn_days)}
            res["data_total"] = np.array([total[pos[d]] for d in days_filter])
        if res.get("data_pH") is not None and syn.get("pH") is not None:
            t_real = np.asarray(res["t_pH_days"], dtype=float)
            if not np.allclose(t_real, np.asarray(syn["pH"]["t_days"], dtype=float)):
                raise ValueError("pH の時刻が合成データと違う")
            res["data_pH"] = np.asarray(syn["pH"]["values"], dtype=float)
        return res

    E.load_experimental_data = load_experimental_data
    E.load_multichannel_data = load_multichannel_data


# ---------------------------------------------------------------- 分類
def classify(x, lo, hi):
    """同定（identified）/ 符号のみ（sign-only）/ 同定されない（n.i.）。ident_table と同じ規則。"""
    q05, q50, q95 = np.percentile(x, [5, 50, 95])
    unif = (hi - lo) / np.sqrt(12.0)
    ratio = float(np.std(x) / unif) if unif > 0 else float("nan")
    if ratio < UNIDENT_RATIO:
        c = "identified"
    elif q05 > 0 or q95 < 0:
        c = "sign-only"
    else:
        c = "n.i."
    return c, ratio, (float(q05), float(q50), float(q95))


def tex_name(nm):
    return "$a_{%s}$" % nm[1:]
