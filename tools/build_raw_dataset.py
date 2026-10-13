#!/usr/bin/env python3
"""生データ（raw_heine/*.csv、tools/convert_raw_heine.py の出力）から、推定が読む形の CSV 一式を作る（2026-10-13）。

出力先: data_5species/raw_dataset/experiment_data/（**生データから作ったファイルだけ**を置く。読み取りデータ
digitised_dataset/ からは何も複製しない（2026-10-13j、ユーザー指示）。pH・OD600 などは無いので推定では使われない）
  fig3_species_distribution_summary.csv / _replicates.csv     ← qPCR「only living cells」（PMA あり、生きた菌）
  figS1_total_species_distribution_summary.csv / _replicates.csv ← qPCR「all cells」（PMA なし、全菌）
  fig2_membrane_distribution.csv                               ← LIVE/DEAD（5 枚 = 1 バイオフィルム。intact = バイオフィルムの平均の平均、error = その SD）
  biofilm_boxplot_data.csv・boxplot_<条件>_<培養>.csv            ← LIVE/DEAD の画像ごとの体積（中央値・四分位・ひげ、×1e6 µm³）
推定は --data-source raw でこのフォルダを読む（既定 digitised は今までどおり）。
値の形は今の CSV と同じ（% 単位、summary は median/mean/std/min/max/q1/q3、replicates は replicate/distribution_pct）。
注意: qPCR の測定は反復 × 技術反復を区別せずに並べたもの（日・条件で 3〜18 個）。σ はそのまま全測定の SD から作られる。
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data_5species"
RAW = ROOT / "experiment_data" / "raw_heine"  # 生データ（gitignore、再配布しない）
OUT = ROOT / "raw_dataset" / "experiment_data"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)  # 以前の版で複製した読み取りデータを残さない
    OUT.mkdir(parents=True)
    q = pd.read_csv(RAW / "qpcr_long.csv")
    for pool, stem in (("viable", "fig3_species_distribution"), ("total", "figS1_total_species_distribution")):
        s = q[q.pool == pool].rename(columns={"rep": "replicate", "pct": "distribution_pct"})
        s[["condition", "cultivation", "day", "species", "replicate", "distribution_pct"]].to_csv(
            OUT / f"{stem}_replicates.csv", index=False)
        g = s.groupby(["condition", "cultivation", "day", "species"]).distribution_pct
        summ = pd.DataFrame({"median": g.median(), "mean": g.mean(), "std": g.std(), "min": g.min(),
                             "max": g.max(), "q1": g.quantile(0.25), "q3": g.quantile(0.75)}).reset_index()
        summ.to_csv(OUT / f"{stem}_summary.csv", index=False)
        print(f"{stem}: {len(s)} replicate rows, {len(summ)} summary rows")
    ld = pd.read_csv(RAW / "livedead_long.csv")
    # 5 枚ずつ同じバイオフィルム（N. Heine、2026-10-13）。error_pct はバイオフィルムの平均どうしの SD
    # （組成の σ と同じく生物学的反復のばらつき。画像 1 枚の SD はバイオフィルム内のばらつきを含んで大きすぎた）
    ld = ld.assign(biofilm=(ld.image - 1) // 5)
    bio = ld.groupby(["condition", "cultivation", "day", "biofilm"]).live_pct.mean().reset_index()
    g = bio.groupby(["condition", "cultivation", "day"]).live_pct
    m = pd.DataFrame({"intact_membrane_pct": g.mean(), "error_pct": g.std()}).reset_index()
    m["damaged_membrane_pct"] = 100 - m["intact_membrane_pct"]
    m[["condition", "cultivation", "day", "intact_membrane_pct", "damaged_membrane_pct", "error_pct"]].to_csv(
        OUT / "fig2_membrane_distribution.csv", index=False)
    print(f"fig2_membrane_distribution: {len(m)} rows")
    # 体積（LIVE/DEAD の画像ごと、µm³ → ×1e6 の単位で今の biofilm_boxplot_data.csv に合わせる）。
    # CH の Day 1 は 40 倍対物なので 63 倍に換算した値（objective_note = 63x-converted）を使う
    v = pd.read_csv(RAW / "volume_long.csv")
    v = v[~v.objective_note.fillna("").str.startswith("40x")]
    rows = []
    for (c, cu, d), x in v.groupby(["condition", "cultivation", "day"]).volume:
        x = x.values / 1e6
        q1, q3 = np.percentile(x, [25, 75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        rows.append([c, cu, d, np.median(x), q1, q3, x[x >= lo].min(), x[x <= hi].max(), ""])
    pd.DataFrame(rows, columns=["condition", "cultivation", "day", "median", "q1", "q3", "whisker_low",
                                "whisker_high", "fliers"]).to_csv(OUT / "biofilm_boxplot_data.csv", index=False)
    print(f"biofilm_boxplot_data: {len(rows)} rows")
    # 推定は boxplot_<condition>_<cultivation>.csv を biofilm_boxplot_data.csv より先に読むので、
    # 複製した読み取りの版をここで生データの版に置き換える（2026-10-13j）
    # pH（tools/convert_raw_ph.py の出力があるときだけ。反復の平均。推定には使わず ph_check の検証用）
    if (RAW / "ph_long.csv").exists():
        ph = pd.read_csv(RAW / "ph_long.csv")
        w = ph.groupby(["time_days", "condition"]).pH.mean().unstack()
        pd.DataFrame({"time_days": w.index, "pH_commensal": w.get("Commensal"),
                      "pH_dysbiotic": w.get("Dysbiotic")}).to_csv(OUT / "fig4A_pH_timeseries.csv", index=False)
        print(f"fig4A_pH_timeseries: {len(w)} rows")
    bp = pd.read_csv(OUT / "biofilm_boxplot_data.csv")
    for (c, cu), x in bp.groupby(["condition", "cultivation"]):
        x.to_csv(OUT / f"boxplot_{c}_{cu}.csv", index=False)


if __name__ == "__main__":
    main()
