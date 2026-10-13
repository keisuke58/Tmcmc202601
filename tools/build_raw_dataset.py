#!/usr/bin/env python3
"""生データ（raw_heine/*.csv、tools/convert_raw_heine.py の出力）から、推定が読む形の CSV 一式を作る（2026-10-13）。

出力先: data_5species/raw_dataset/experiment_data/（今の experiment_data/ の CSV を全部複製し、次の 6 つを生データで置き換える）
  fig3_species_distribution_summary.csv / _replicates.csv     ← qPCR「only living cells」（PMA あり、生きた菌）
  figS1_total_species_distribution_summary.csv / _replicates.csv ← qPCR「all cells」（PMA なし、全菌）
  fig2_membrane_distribution.csv                               ← LIVE/DEAD の画像ごとの値（intact = 画像の平均、error = 画像の SD）
  biofilm_boxplot_data.csv                                     ← LIVE/DEAD の画像ごとの体積（中央値・四分位・ひげ、×1e6 µm³）
推定は --data-source raw でこのフォルダを読む（既定 digitised は今までどおり）。
値の形は今の CSV と同じ（% 単位、summary は median/mean/std/min/max/q1/q3、replicates は replicate/distribution_pct）。
注意: qPCR の測定は反復 × 技術反復を区別せずに並べたもの（日・条件で 3〜18 個）。σ はそのまま全測定の SD から作られる。
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data_5species"
SRC = ROOT / "experiment_data"
RAW = SRC / "raw_heine"
OUT = ROOT / "raw_dataset" / "experiment_data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("*.csv"):
        shutil.copy(f, OUT / f.name)
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
    g = ld.groupby(["condition", "cultivation", "day"]).live_pct
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


if __name__ == "__main__":
    main()
