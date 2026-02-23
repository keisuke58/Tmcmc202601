from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_profiles(base_dir: Path) -> pd.DataFrame:
    files = [
        ("dh", base_dir / "depth_profiles_dh.csv"),
        ("cs", base_dir / "depth_profiles_cs.csv"),
        ("dh", base_dir / "depth_profiles_dh_2d.csv"),
        ("cs", base_dir / "depth_profiles_cs_2d.csv"),
    ]
    rows = []
    for cond, path in files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for odb in sorted(df["odb"].unique()):
            sub = df[df["odb"] == odb].copy()
            if "3DJob" in odb:
                dim = "3D"
            else:
                if "_2d" in path.name:
                    dim = "2D"
                else:
                    dim = "2D/1D"
            sub["cond"] = cond
            sub["dim"] = dim
            rows.append(sub)
    if not rows:
        raise SystemExit("no depth_profiles_*.csv found")
    df_all = pd.concat(rows, ignore_index=True)
    df_all["depth_frac"] = df_all["depth_frac"].astype(float)
    return df_all


def summarize_by_cond_dim_depth(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["S11", "S22", "S33", "S_Mises"]
    grouped = df.groupby(["cond", "dim", "depth_frac"])[metrics].agg(["mean", "std"])
    grouped = grouped.reset_index()
    cols = ["cond", "dim", "depth_frac"]
    for m in metrics:
        cols.append(f"{m}_mean")
        cols.append(f"{m}_std")
    grouped.columns = cols
    return grouped


def diff_between_conditions(summary: pd.DataFrame) -> pd.DataFrame:
    dh = summary[summary["cond"] == "dh"].set_index(["dim", "depth_frac"])
    cs = summary[summary["cond"] == "cs"].set_index(["dim", "depth_frac"])
    common_index = dh.index.intersection(cs.index)
    dh = dh.loc[common_index]
    cs = cs.loc[common_index]
    rows = []
    for (dim, depth), _ in dh.iterrows():
        r_dh = dh.loc[(dim, depth)]
        r_cs = cs.loc[(dim, depth)]
        row = {
            "dim": dim,
            "depth_frac": depth,
        }
        for m in ["S11", "S22", "S33", "S_Mises"]:
            row[f"{m}_diff_cs_minus_dh"] = float(
                r_cs[f"{m}_mean"] - r_dh[f"{m}_mean"]
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dim", "depth_frac"]).reset_index(drop=True)


def ratio_3d_over_2d(summary: pd.DataFrame) -> pd.DataFrame:
    s2d = summary[summary["dim"] == "2D"].set_index(["cond", "depth_frac"])
    s3d = summary[summary["dim"] == "3D"].set_index(["cond", "depth_frac"])
    common_index = s2d.index.intersection(s3d.index)
    if common_index.empty:
        return pd.DataFrame()
    s2d = s2d.loc[common_index]
    s3d = s3d.loc[common_index]
    rows = []
    for (cond, depth), _ in s2d.iterrows():
        r2 = s2d.loc[(cond, depth)]
        r3 = s3d.loc[(cond, depth)]
        row = {
            "cond": cond,
            "depth_frac": depth,
        }
        for m in ["S11", "S22", "S33", "S_Mises"]:
            num = r3[f"{m}_mean"]
            den = r2[f"{m}_mean"]
            if den == 0 or np.isnan(den):
                ratio = np.nan
            else:
                ratio = float(num / den)
            row[f"{m}_ratio_3D_over_2D"] = ratio
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cond", "depth_frac"]).reset_index(drop=True)


def main(argv: list[str]) -> int:
    base_dir = Path(".")
    df_all = load_profiles(base_dir)
    summary = summarize_by_cond_dim_depth(df_all)
    summary_path = base_dir / "abaqus_depth_summary.csv"
    summary.to_csv(summary_path, index=False)
    diff = diff_between_conditions(summary)
    diff_path = base_dir / "abaqus_depth_cond_diff.csv"
    if not diff.empty:
        diff.to_csv(diff_path, index=False)
    ratio = ratio_3d_over_2d(summary)
    ratio_path = base_dir / "abaqus_depth_3d_over_2d.csv"
    if not ratio.empty:
        ratio.to_csv(ratio_path, index=False)
    print("=== Summary: mean/std for S11, S22, S33, S_Mises ===")
    print(summary.to_string(index=False))
    if not diff.empty:
        print()
        print("=== Condition difference (cs - dh) ===")
        print(diff.to_string(index=False))
    if not ratio.empty:
        print()
        print("=== 3D / 2D ratio (per cond, depth_frac) ===")
        print(ratio.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
