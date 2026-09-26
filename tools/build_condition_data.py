#!/usr/bin/env python3
"""4条件の観測データと種ごとの sigma を experiment_data の CSV から組み立てる。

_runs/ にコミットされているのは Dysbiotic/HOBIC だけなので、他の3条件を
論文のパイプラインと同じ手順で作り直せるようにする。DH については
コミット済み data.npy と一致するかを検証する（既定で実行される）。

出所は species_distribution_data.csv（色名キー）。
fig3_species_distribution_summary.csv（種名キー）ではない — そちらを使うと
DH で正規化後 0.30 もずれる。

  data_abs[k, i] = total_vol_median[day_k] * median_pct[day_k, i] / 100
  data           = data_abs / row_sums

row_sums で割ると total_vol が約分されるので、結果は「各時点の種構成比」。
これが load_experimental_data(..., normalize=True) が返すものと同じ。

sigma は fig3_species_distribution_replicates.csv（種名キー）から、
ローダー（estimate_reduced_nishioka.py）と同じく日ごとの IQR/1.35 を
分率空間で平均し、下限 0.05 を当てる。

使い方:
    python3 tools/build_condition_data.py                 # 検証して表示
    python3 tools/build_condition_data.py --save out.npz  # npz に保存
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ED = ROOT / "data_5species" / "experiment_data"
DH_RUN = ROOT / "_runs" / "Dysbiotic_HOBIC_K0.05_n4.0_1k30"

# species_distribution_data.csv は色名キー
# （prepare_estimation_data.py の SPECIES_MAP_GENERAL と同じ）
COLOR = {"Blue": 0, "Green": 1, "Yellow": 2, "Orange": 2, "Purple": 3, "Red": 4}

# replicate CSV は種名キー。Vei と Pg は条件で種・株が違う
SPECIES = {
    "S. oralis": 0,
    "A. naeslundii": 1,
    "V. dispar": 2,
    "V. parvula": 2,
    "F. nucleatum": 3,
    "P. gingivalis_20709": 4,
    "P. gingivalis_W83": 4,
    "P. gingivalis": 4,
}
NAMES = ["So", "An", "Vei", "Fn", "Pg"]
DAYS = [1, 3, 6, 10, 15, 21]
CONDITIONS = [
    ("CS", "Commensal", "Static"),
    ("CH", "Commensal", "HOBIC"),
    ("DS", "Dysbiotic", "Static"),
    ("DH", "Dysbiotic", "HOBIC"),
]
MIN_SIGMA_FRAC = 0.05  # ローダーの _min_sigma_frac と同じ


def _total_volumes(cond, cult):
    out = {}
    with open(ED / f"boxplot_{cond}_{cult}.csv") as f:
        for r in csv.DictReader(f):
            if r["condition"] == cond and r["cultivation"] == cult:
                out[int(r["day"])] = float(r["median"])
    return out


def _species_medians(cond, cult):
    out = defaultdict(float)
    with open(ED / "species_distribution_data.csv") as f:
        for r in csv.DictReader(f):
            if r["condition"] != cond or r["cultivation"] != cult:
                continue
            i = COLOR.get(r["species"])
            if i is not None:
                out[(int(r["day"]), i)] += float(r["median"])
    return out


def sigma_per_species(cond, cult):
    """replicate から分率空間の sigma（日ごとの IQR/1.35 の平均、下限あり）。"""
    vals = defaultdict(list)
    with open(ED / "fig3_species_distribution_replicates.csv") as f:
        for r in csv.DictReader(f):
            if r["condition"] != cond or r["cultivation"] != cult:
                continue
            i = SPECIES.get(r["species"])
            if i is None:
                continue
            try:
                vals[(i, int(r["day"]))].append(float(r["distribution_pct"]))
            except ValueError:
                pass
    sig = np.full(5, MIN_SIGMA_FRAC)
    for i in range(5):
        per = []
        for d in DAYS:
            v = np.array(vals.get((i, d), []))
            if len(v) >= 3:
                per.append((np.percentile(v, 75) - np.percentile(v, 25)) / 1.35 / 100.0)
        if per:
            sig[i] = np.mean(per)
    return np.maximum(sig, MIN_SIGMA_FRAC)


def build(cond, cult):
    """(絶対体積, 正規化した構成比) を返す。"""
    tv = _total_volumes(cond, cult)
    med = _species_medians(cond, cult)
    abs_vol = np.zeros((len(DAYS), 5))
    for k, d in enumerate(DAYS):
        for i in range(5):
            abs_vol[k, i] = tv.get(d, 0.0) * med.get((d, i), 0.0) / 100.0
    return abs_vol, abs_vol / np.maximum(abs_vol.sum(1, keepdims=True), 1e-12)


def verify_dh():
    """DH をコミット済み data.npy と突き合わせる。"""
    abs_dh, norm_dh = build("Dysbiotic", "HOBIC")
    ref_abs = np.load(DH_RUN / "data.npy")
    ref_norm = ref_abs / ref_abs.sum(1, keepdims=True)
    d_abs = float(np.abs(abs_dh - ref_abs).max())
    d_norm = float(np.abs(norm_dh - ref_norm).max())
    print("=== DH をコミット済み data.npy と照合")
    print(f"  絶対体積 max|diff| = {d_abs:.2e}")
    print(f"  正規化後 max|diff| = {d_norm:.2e}")
    ok = d_norm < 1e-9
    print(f"  -> {'一致' if ok else '不一致'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", type=str, default=None, help="npz の保存先")
    args = ap.parse_args()

    if not verify_dh():
        print("\nDH が一致しないので組み立て方が違う。ここで止める。")
        return 1

    out = {}
    for tag, cond, cult in CONDITIONS:
        _, norm = build(cond, cult)
        sig = sigma_per_species(cond, cult)
        out[tag] = (norm, sig)
        print(f"\n=== {tag}  ({cond} / {cult})")
        header = "".join(f"{d:>9}" for d in DAYS)
        print(f"  {'':4}{header}    sigma   1/s^2比(Vei=1)")
        for i in range(5):
            row = "".join(f"{norm[k, i]:>9.4f}" for k in range(len(DAYS)))
            print(f"  {NAMES[i]:<4}{row}  {sig[i]:>7.4f}  {(sig[2] / sig[i]) ** 2:>11.2f}")
        pg = norm[:, 4]
        print(
            f"  Pg D15->D21 = {pg[-2]:.4f} -> {pg[-1]:.4f}  " f"比 {pg[-1] / max(pg[-2], 1e-9):.2f}"
        )

    print(
        "\n注: commensal 2条件は Fn も Pg も全時点 0.005（検出限界）。sigma が下限 0.05 に\n"
        "  当たるため 1/sigma^2 の重みが最大になる。ゲート ON では h(Fn) ~ 0.01 で\n"
        "  Pg の相互作用がほぼ殺されていた（A7 参照）。"
    )

    if args.save:
        np.savez(
            args.save,
            **{f"{t}_data": v[0] for t, v in out.items()},
            **{f"{t}_sigma": v[1] for t, v in out.items()},
        )
        print(f"\nsaved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
