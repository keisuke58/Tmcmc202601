#!/usr/bin/env python3
"""LUH/MHH の体積・生きた割合（LIVE/DEAD、Heine 提供）を修論の FEM 用に書き出す（2026-10-13）。

公開してよい集計（Fig. 2A の箱ひげ図と同程度、--public-dir）:
  volume_summary.csv        条件 × 日: 画像数・バイオフィルム数・体積の中央値と四分位（画像）、バイオフィルム平均の平均と SD
  live_fraction_summary.csv 条件 × 日: 生きた割合のバイオフィルム平均の平均と SD
  live_volume_species.csv   条件 × 日 × 種: 種ごとの生きた体積（体積 × 生きた割合 × 生きた菌の組成）の中央値と 90% 区間
                            （tools/raw_data_report.py のブートストラップ、バイオフィルム単位）
公開しない（--private-dir、gitignore の場所に置くこと。データは請求に応じて提供、N. Heine）:
  biofilm_level.csv         条件 × 日 × バイオフィルム: 体積と生きた割合のバイオフィルム平均（5 枚 = 1 バイオフィルム）
体積の単位は 10^6 µm³（画像 1 枚の視野の体積）。CH の Day 1 は 40 倍対物の値を 63 倍に換算した値。

    python3 tools/export_volume_summary.py --public-dir <dir> --private-dir <gitignored dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raw_data_report as R  # noqa: E402

NAMES = ["So", "An", "Vei", "Fn", "Pg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-dir", required=True)
    ap.add_argument("--private-dir", required=True)
    a = ap.parse_args()
    pub, priv = Path(a.public_dir), Path(a.private_dir)
    pub.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    q, ld, v = R.load()
    v = v.assign(biofilm=(v.image - 1) // 5, vol=v.volume / 1e6)
    ld = ld.assign(biofilm=(ld.image - 1) // 5, live=ld.live_pct / 100)
    rows, lrows, brows = [], [], []
    for (c, cu, d), x in v.groupby(["condition", "cultivation", "day"]):
        bm = x.groupby("biofilm").vol.mean()
        rows.append([c, cu, d, len(x), len(bm), x.vol.median(), x.vol.quantile(0.25), x.vol.quantile(0.75),
                     bm.mean(), bm.std(ddof=1)])
        for b, val in bm.items():
            brows.append([c, cu, d, int(b) + 1, "volume_1e6um3", val])
    for (c, cu, d), x in ld.groupby(["condition", "cultivation", "day"]):
        bm = x.groupby("biofilm").live.mean()
        lrows.append([c, cu, d, len(x), len(bm), bm.mean(), bm.std(ddof=1)])
        for b, val in bm.items():
            brows.append([c, cu, d, int(b) + 1, "live_fraction", val])
    pd.DataFrame(rows, columns=["condition", "cultivation", "day", "n_images", "n_biofilms", "median_image",
                                "q1_image", "q3_image", "mean_biofilm", "sd_biofilm"]).round(4).to_csv(
        pub / "volume_summary.csv", index=False)
    pd.DataFrame(lrows, columns=["condition", "cultivation", "day", "n_images", "n_biofilms", "mean_biofilm",
                                 "sd_biofilm"]).round(4).to_csv(pub / "live_fraction_summary.csv", index=False)
    rep = R.OUT / "raw_report.json"
    if rep.exists():
        lv = json.load(open(rep))["live_volume"]
        srows = []
        for tag, (c, cu) in R.CONDS.items():
            for d, e in lv.get(tag, {}).items():
                for i, n in enumerate(NAMES):
                    srows.append([c, cu, int(d), n, e["median"][i], e["q05"][i], e["q95"][i]])
        pd.DataFrame(srows, columns=["condition", "cultivation", "day", "species", "median", "q05", "q95"]).round(
            4).to_csv(pub / "live_volume_species.csv", index=False)
    else:
        print(f"注意: {rep} が無いので live_volume_species.csv は書かない（先に tools/raw_data_report.py）")
    pd.DataFrame(brows, columns=["condition", "cultivation", "day", "biofilm", "quantity", "value"]).round(5).to_csv(
        priv / "biofilm_level.csv", index=False)
    print(f"公開: {pub}  /  非公開: {priv}")


if __name__ == "__main__":
    main()
