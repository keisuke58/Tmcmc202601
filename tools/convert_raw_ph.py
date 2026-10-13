#!/usr/bin/env python3
"""Heine 提供の pH（フロー反応器、Fig. 4A）を推定の道具が読む形にする（2026-10-13 準備、データは 10/14 着の予定）。

ファイルの形はまだ分からないので 2 段で使う:
  1. 中身を見る:   python3 tools/convert_raw_ph.py <xlsx/csv> --inspect
     （シート名と先頭の行を出す。どの列が時間・commensal・dysbiotic・反復かを決める）
  2. 変換する:     python3 tools/convert_raw_ph.py <xlsx/csv> --sheet <名前> --time-col <列> \
                       --commensal-cols <列,列,...> --dysbiotic-cols <列,列,...> [--time-unit h|d] [--skip-rows N]
     列は 0 始まりの番号か見出しの名前。反復の列は複数並べる（平均を取る）。

出力（raw_heine/ は gitignore、生データは再配布しない）:
  data_5species/experiment_data/raw_heine/ph_long.csv   condition, time_days, rep, pH
その後 `python3 tools/build_raw_dataset.py` が raw_dataset/experiment_data/fig4A_pH_timeseries.csv
（time_days, pH_commensal, pH_dysbiotic、反復の平均）を書く。pH は推定に使わず、tools/ph_check.py の検証だけ。
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data_5species" / "experiment_data" / "raw_heine"


def read_any(path, sheet=None, skip=0):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0, header=None, skiprows=skip)
    return pd.read_csv(p, header=None, skiprows=skip)


def col(df, spec, header_row):
    """列の指定（番号か、header_row 行目の見出し）→ 列番号。"""
    if str(spec).isdigit():
        return int(spec)
    hdr = [str(x).strip() for x in df.iloc[header_row].tolist()]
    if str(spec).strip() not in hdr:
        raise SystemExit(f"見出し '{spec}' が {header_row} 行目に無い: {hdr}")
    return hdr.index(str(spec).strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--header-row", type=int, default=0, help="見出しのある行（skip 後の 0 始まり）。データはその次の行から")
    ap.add_argument("--time-col")
    ap.add_argument("--time-unit", choices=["d", "h"], default="d")
    ap.add_argument("--commensal-cols")
    ap.add_argument("--dysbiotic-cols")
    a = ap.parse_args()

    if a.inspect:
        p = Path(a.path)
        sheets = pd.ExcelFile(p).sheet_names if p.suffix.lower().startswith(".xls") else [None]
        for sh in sheets:
            df = read_any(p, sh)
            print(f"## sheet {sh!r}: {df.shape[0]} 行 × {df.shape[1]} 列")
            print(df.head(12).to_string())
        return

    if not (a.time_col and a.commensal_cols and a.dysbiotic_cols):
        raise SystemExit("--time-col・--commensal-cols・--dysbiotic-cols が要る（先に --inspect で列を決める）")
    df = read_any(a.path, a.sheet, a.skip_rows)
    tc = col(df, a.time_col, a.header_row)
    groups = {"Commensal": [col(df, c, a.header_row) for c in a.commensal_cols.split(",")],
              "Dysbiotic": [col(df, c, a.header_row) for c in a.dysbiotic_cols.split(",")]}
    rows = []
    for _, r in df.iloc[a.header_row + 1:].iterrows():
        t = pd.to_numeric(r[tc], errors="coerce")
        if pd.isna(t):
            continue
        t = float(t) / (24.0 if a.time_unit == "h" else 1.0)
        for cond, cols in groups.items():
            for k, c in enumerate(cols, 1):
                v = pd.to_numeric(r[c], errors="coerce")
                if not pd.isna(v):
                    rows.append([cond, t, k, float(v)])
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "ph_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "time_days", "rep", "pH"])
        w.writerows(rows)
    d = pd.DataFrame(rows, columns=["condition", "time_days", "rep", "pH"])
    print(f"ph_long.csv: {len(rows)} 行")
    print(d.groupby("condition").agg(n=("pH", "size"), t_min=("time_days", "min"), t_max=("time_days", "max"),
                                     pH_min=("pH", "min"), pH_max=("pH", "max")).to_string())


if __name__ == "__main__":
    main()
