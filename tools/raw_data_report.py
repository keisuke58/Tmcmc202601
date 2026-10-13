#!/usr/bin/env python3
"""生データ（Heine 提供、2026-10-13）で初めてできる確認をまとめて出す（読み取りデータではできなかったもの）。

出力は生データから計算した数を含むので **git に入れない**（docs/revision/generated/raw_analysis/ は gitignore）。
共著者（Nils）が公開を了承するまでは、表・図は手元と原稿の下書きだけで使う。

1. 読み取りの誤差: 生データ − 読み取り値（組成 2 種・生きた割合・体積）の RMSE と最大の差、条件 × 通路。
2. 反復の構造: qPCR の列は「生物学的反復 × 技術反復 3」（列の位置で 3 つずつ、空欄は末尾だけ）。
   技術反復の SD と生物学的反復の SD の比（ICC）。尤度の σ がどちらの揺れを表すかの確認。
3. σ の比較: 推定が使う種ごとの σ（build_replicate_sigma、IQR/1.35）を、読み取り（合成の反復）・生データ
   （全測定）・生データ（技術反復を平均した生物学的反復）で並べる。
4. 正規の近似: 生物学的反復の平均 − 2SD が 0 を割る (日, 種) の割合（割合のデータに正規の尤度を使う妥当性）。
5. ψ_i^data の信頼区間: 生物学的反復のブートストラップ（生きた菌・全菌の qPCR、画像の生きた割合をそれぞれ
   独立に引き直す。PMA あり・なしの対応は不明なので対にしない）。式は tools/psi_from_data.py と同じ。
6. 種ごとの生きた体積: 体積（画像）× 生きた割合（画像）× 生きた菌の組成（qPCR）。モデルの φ_i ψ_i を
   体積と比べる新しい観測（推定には使わない、検証用）。
7. Day 1（初期値）の組成: 生データと読み取りの差。推定の phi_init は Day 1 から作るので、ここの差は全段に効く。

    python3 tools/raw_data_report.py            # → docs/revision/generated/raw_analysis/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_5species" / "experiment_data" / "raw_heine"
DIG = ROOT / "data_5species" / "digitised_dataset" / "experiment_data"
RDS = ROOT / "data_5species" / "raw_dataset" / "experiment_data"
OUT = ROOT / "docs" / "revision" / "generated" / "raw_analysis"
sys.path.insert(0, str(ROOT / "data_5species" / "main"))

SP = {"S. oralis": 0, "A. naeslundii": 1, "V. dispar": 2, "V. parvula": 2, "F. nucleatum": 3,
      "P. gingivalis_20709": 4, "P. gingivalis_W83": 4}
NAMES = ["So", "An", "Vei", "Fn", "Pg"]
CONDS = {"CS": ("Commensal", "Static"), "CH": ("Commensal", "HOBIC"),
         "DS": ("Dysbiotic", "Static"), "DH": ("Dysbiotic", "HOBIC")}
DAYS = [1, 3, 6, 10, 15, 21]
MIN_TOTAL = 0.02
N_BOOT = 2000
rng = np.random.default_rng(0)


def load():
    q = pd.read_csv(RAW / "qpcr_long.csv")
    q["sp"] = q.species.map(SP)
    q["bio"] = (q.rep - 1) // 3  # 技術反復 3 つで 1 つの生物学的反復（列の位置、空欄は末尾だけ）
    ld = pd.read_csv(RAW / "livedead_long.csv")
    v = pd.read_csv(RAW / "volume_long.csv")
    v = v[~v.objective_note.fillna("").str.startswith("40x")]
    return q, ld, v


def bio_table(q, c, cu, pool):
    """生物学的反復ごとの組成 (bio × 5)。日 → 配列。技術反復を平均してから 5 種にまとめ、行和 1。"""
    s = q[(q.condition == c) & (q.cultivation == cu) & (q.pool == pool)]
    m = s.groupby(["day", "bio", "sp"]).pct.mean().unstack(fill_value=0.0).reindex(columns=range(5), fill_value=0.0)
    out = {}
    for d in DAYS:
        if d in m.index.get_level_values(0):
            x = m.loc[d].values
            out[d] = x / x.sum(1, keepdims=True)
    return out


def median_comp(tab):
    return {d: (lambda z: z / z.sum())(np.median(x, 0)) for d, x in tab.items()}


def dig_comp(fn, c, cu):
    df = pd.read_csv(DIG / fn)
    df = df[(df.condition == c) & (df.cultivation == cu)]
    out = {}
    for _, r in df.iterrows():
        if r.species in SP:
            out.setdefault(int(r.day), np.zeros(5))[SP[r.species]] += r["median"]
    return {k: v / v.sum() for k, v in out.items()}


def raw_comp_all(q, c, cu, pool):
    """推定が今使う形（全測定の中央値を種ごとに → 5 種 → 行和 1）。"""
    s = q[(q.condition == c) & (q.cultivation == cu) & (q.pool == pool)]
    out = {}
    for (d, sp), x in s.groupby(["day", "species"]).pct:
        out.setdefault(int(d), np.zeros(5))[SP[sp]] += np.median(x)
    return {k: v / v.sum() for k, v in out.items()}


def sec1_digitisation(q, ld, v):
    memb = pd.read_csv(DIG / "fig2_membrane_distribution.csv")
    bp = pd.read_csv(DIG / "biofilm_boxplot_data.csv")
    rows = []
    for tag, (c, cu) in CONDS.items():
        for pool, fn in (("viable", "fig3_species_distribution_summary.csv"),
                         ("total", "figS1_total_species_distribution_summary.csv")):
            r, dg = raw_comp_all(q, c, cu, pool), dig_comp(fn, c, cu)
            dd = np.array([r[d] - dg[d] for d in DAYS if d in r and d in dg])
            k = np.unravel_index(np.argmax(np.abs(dd)), dd.shape)
            rows.append([tag, f"{pool} composition", np.sqrt(np.mean(dd**2)), np.abs(dd).max(),
                         f"Day {DAYS[k[0]]} {NAMES[k[1]]}"])
        s = ld[(ld.condition == c) & (ld.cultivation == cu)].groupby("day").live_pct.mean() / 100
        m = memb[(memb.condition == c) & (memb.cultivation == cu)].set_index("day").intact_membrane_pct / 100
        dd = (s - m).dropna()
        rows.append([tag, "live fraction", np.sqrt(np.mean(dd**2)), dd.abs().max(), f"Day {dd.abs().idxmax()}"])
        s = v[(v.condition == c) & (v.cultivation == cu)].groupby("day").volume.median() / 1e6
        m = bp[(bp.condition == c) & (bp.cultivation == cu)].set_index("day")["median"]
        dd = ((s - m) / m).dropna()
        rows.append([tag, "volume (rel.)", np.sqrt(np.mean(dd**2)), dd.abs().max(), f"Day {dd.abs().idxmax()}"])
    return pd.DataFrame(rows, columns=["cond", "channel", "rmse", "max_abs", "where"])


def sec2_structure(q):
    rows = []
    for tag, (c, cu) in CONDS.items():
        for pool in ("viable", "total"):
            s = q[(q.condition == c) & (q.cultivation == cu) & (q.pool == pool)]
            ratios, nb = [], []
            for (d, sp), x in s.groupby(["day", "species"]):
                g = x.groupby("bio").pct
                if g.ngroups < 2 or x.pct.mean() < 2:  # 2% 未満の種は比が不安定
                    continue
                tech = np.sqrt(np.mean(g.var(ddof=1).dropna()))
                bio = g.mean().std(ddof=1)
                if bio > 0:
                    ratios.append(tech / bio)
                nb.append(g.ngroups)
            rows.append([tag, pool, int(np.median(nb)), np.median(ratios), np.quantile(ratios, 0.9)])
    return pd.DataFrame(rows, columns=["cond", "pool", "n_bio(median)", "tech/bio SD (median)", "tech/bio SD (q90)"])


def sigma_from(df, c, cu, days):
    from _test_helpers import build_replicate_sigma  # 推定と同じ規則（IQR/1.35、下限 0.05、情報比の上限）
    tmp = OUT / "_tmp_rep.csv"
    df.to_csv(tmp, index=False)
    s = build_replicate_sigma(str(tmp), c, cu, days, SP, 5, 0.05, max_info_ratio=10.0)
    tmp.unlink()
    return s


def sec3_sigma(q):
    rows = []
    dig = pd.read_csv(DIG / "fig3_species_distribution_replicates.csv")
    for tag, (c, cu) in CONDS.items():
        s = q[(q.condition == c) & (q.cultivation == cu) & (q.pool == "viable")]
        allm = s.rename(columns={"pct": "distribution_pct"})
        bio = s.groupby(["condition", "cultivation", "day", "species", "bio"]).pct.mean().reset_index() \
            .rename(columns={"pct": "distribution_pct"})
        days = [3, 6, 10, 15, 21]
        for lab, df in (("digitised (synthetic reps)", dig), ("raw, all measurements", allm),
                        ("raw, biological reps", bio)):
            sg = sigma_from(df, c, cu, days)
            rows.append([tag, lab] + list(np.round(sg.mean(0), 3)))
    return pd.DataFrame(rows, columns=["cond", "source"] + NAMES)


def sec4_gauss(q):
    rows = []
    for tag, (c, cu) in CONDS.items():
        tab = bio_table(q, c, cu, "viable")
        n, neg = 0, 0
        for d, x in tab.items():
            if d == 1 or len(x) < 2:
                continue
            mu, sd = x.mean(0), x.std(0, ddof=1)
            ok = mu > 0.005
            n += ok.sum()
            neg += ((mu - 2 * sd) < 0)[ok].sum()
        rows.append([tag, n, neg, neg / max(n, 1)])
    return pd.DataFrame(rows, columns=["cond", "n(day,species)", "mean-2SD<0", "fraction"])


def psi_of(vc, tc, live):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tc >= MIN_TOTAL, vc / tc * live, np.nan)


def sec5_psi(q, ld):
    res = {}
    for tag, (c, cu) in CONDS.items():
        V, T = bio_table(q, c, cu, "viable"), bio_table(q, c, cu, "total")
        L = {d: x.live_pct.values / 100 for d, x in ld[(ld.condition == c) & (ld.cultivation == cu)].groupby("day")}
        res[tag] = {}
        for d in DAYS[1:]:
            pt = psi_of(np.median(V[d], 0) / np.median(V[d], 0).sum(), np.median(T[d], 0) / np.median(T[d], 0).sum(),
                        np.mean(L[d]))
            bs = []
            for _ in range(N_BOOT):
                v = V[d][rng.integers(len(V[d]), size=len(V[d]))]
                t = T[d][rng.integers(len(T[d]), size=len(T[d]))]
                l = L[d][rng.integers(len(L[d]), size=len(L[d]))]
                vm, tm = np.median(v, 0), np.median(t, 0)
                bs.append(psi_of(vm / vm.sum(), tm / tm.sum(), l.mean()))
            bs = np.array(bs)
            with np.errstate(all="ignore"):
                lo, hi = np.nanquantile(bs, 0.05, 0), np.nanquantile(bs, 0.95, 0)
            res[tag][d] = {"psi": pt.tolist(), "q05": lo.tolist(), "q95": hi.tolist()}
    return res


def sec6_livevol(q, ld, v):
    res = {}
    for tag, (c, cu) in CONDS.items():
        V = bio_table(q, c, cu, "viable")
        L = {d: x.live_pct.values / 100 for d, x in ld[(ld.condition == c) & (ld.cultivation == cu)].groupby("day")}
        W = {d: x.volume.values / 1e6 for d, x in v[(v.condition == c) & (v.cultivation == cu)].groupby("day")}
        res[tag] = {}
        for d in DAYS:
            if d not in V or d not in L or d not in W:
                continue
            bs = []
            for _ in range(N_BOOT):
                vv = V[d][rng.integers(len(V[d]), size=len(V[d]))]
                vm = np.median(vv, 0)
                bs.append(vm / vm.sum() * L[d][rng.integers(len(L[d]), size=len(L[d]))].mean()
                          * np.median(W[d][rng.integers(len(W[d]), size=len(W[d]))]))
            bs = np.array(bs)
            res[tag][d] = {"median": np.median(bs, 0).tolist(), "q05": np.quantile(bs, 0.05, 0).tolist(),
                           "q95": np.quantile(bs, 0.95, 0).tolist(),
                           "total_live": [float(np.median(bs.sum(1))), float(np.quantile(bs.sum(1), 0.05)),
                                          float(np.quantile(bs.sum(1), 0.95))]}
    return res


def sec7_day1(q):
    rows = []
    for tag, (c, cu) in CONDS.items():
        r = raw_comp_all(q, c, cu, "viable")[1]
        dg = dig_comp("fig3_species_distribution_summary.csv", c, cu)[1]
        rows.append([tag, "raw"] + list(np.round(r, 3)))
        rows.append([tag, "digitised"] + list(np.round(dg, 3)))
    return pd.DataFrame(rows, columns=["cond", "source"] + NAMES)


def figures(psi, lv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    for j, tag in enumerate(CONDS):
        ax = axs[0, j]
        for i in range(5):
            ds = [d for d in psi[tag] if not np.isnan(psi[tag][d]["psi"][i])]
            if not ds:
                continue
            y = np.array([psi[tag][d]["psi"][i] for d in ds])
            lo = np.array([psi[tag][d]["q05"][i] for d in ds])
            hi = np.array([psi[tag][d]["q95"][i] for d in ds])
            x = np.array(ds) + (i - 2) * 0.25
            ax.errorbar(x, y, [np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt="o-", color=col[i], ms=3,
                        capsize=2, lw=1, label=NAMES[i])
        ax.axhline(1, color="0.7", lw=0.5)
        ax.set_ylim(0, 1.6)
        ax.set_title(f"{tag}: species viability ψ_i^data (90% bootstrap)", fontsize=9)
        ax.set_xticks(DAYS[1:])
        if j == 0:
            ax.legend(fontsize=7, ncol=5)
        ax = axs[1, j]
        ds = sorted(lv[tag])
        bot = np.zeros(len(ds))
        for i in range(5):
            y = np.array([lv[tag][d]["median"][i] for d in ds])
            ax.bar(range(len(ds)), y, bottom=bot, color=col[i], edgecolor="k", lw=0.3)
            bot += y
        tl = np.array([lv[tag][d]["total_live"] for d in ds])
        ax.errorbar(range(len(ds)), tl[:, 0], [tl[:, 0] - tl[:, 1], tl[:, 2] - tl[:, 0]], fmt="none", color="k",
                    capsize=3)
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels(ds)
        ax.set_title(f"{tag}: live biovolume per species [10^6 µm³]", fontsize=9)
        ax.set_xlabel("Day")
    fig.suptitle("Raw data (MHH) — NOT for redistribution", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "raw_psi_livevolume.png", dpi=130)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q, ld, v = load()
    t1, t2, t3, t4, t7 = sec1_digitisation(q, ld, v), sec2_structure(q), sec3_sigma(q), sec4_gauss(q), sec7_day1(q)
    psi, lv = sec5_psi(q, ld), sec6_livevol(q, ld, v)
    figures(psi, lv)
    json.dump({"psi_boot": psi, "live_volume": lv}, open(OUT / "raw_report.json", "w"), indent=1)
    fmt = lambda df: df.to_markdown(index=False, floatfmt=".3f")  # noqa: E731
    md = ["# 生データでの確認（自動生成、git に入れない）", "",
          "## 1. 読み取りの誤差（生データ − 読み取り値）", fmt(t1), "",
          "## 2. 反復の構造（技術反復 SD / 生物学的反復 SD、2% 以上の種）", fmt(t2), "",
          "## 3. 推定の σ（生きた菌の組成、Day 3–21 の平均、build_replicate_sigma と同じ規則）", fmt(t3), "",
          "## 4. 正規の近似（生物学的反復の平均 − 2SD < 0）", fmt(t4), "",
          "## 7. Day 1（初期値）の生きた菌の組成", fmt(t7), "",
          "## 5・6. ψ_i^data の 90% 区間・種ごとの生きた体積 → raw_psi_livevolume.png、raw_report.json"]
    (OUT / "raw_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
