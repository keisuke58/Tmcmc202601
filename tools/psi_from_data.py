#!/usr/bin/env python3
"""データから出した種ごとの生存率と、モデルの ψ_i（MAP）を並べる（EML 補足、2026-10-13）。

ψ_i^data(d) = (生きた菌の組成 PMA-qPCR の割合 / 全菌の組成の割合) × 群集の生きた割合（SYTO9/PI）
  - 生きた菌の組成: fig3_species_distribution_summary.csv（推定の組成の通路に使うもの）
  - 全菌の組成:     figS1_total_species_distribution_summary.csv（推定には使っていない）
  - 生きた割合:     fig2_membrane_distribution.csv（推定の生存率の通路）
  いずれも反復の中央値。全菌の割合が 2% 未満の日は比が不安定なので外す。
モデルの ψ_i は図 2 のキャッシュ（MAP の φ̄/φ）。days 3–21 の中央値を条件 × 種で出す。

    python3 tools/psi_from_data.py docs/revision/EML_submission/figures/generated/_cache \
        docs/revision/EML_submission/figures/provisional_2026-10-11/supp/supp_S_psi.tex
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data_5species" / "raw_dataset" / "experiment_data"  # 生データ（2026-10-13〜）
SP = {"S. oralis": 0, "A. naeslundii": 1, "V. dispar": 2, "V. parvula": 2, "F. nucleatum": 3,
      "P. gingivalis_20709": 4, "P. gingivalis_W83": 4, "P. gingivalis": 4}
NAMES = ["So", "An", "Vei", "Fn", "Pg"]
TAG = {"CS": ("Commensal", "Static"), "CH": ("Commensal", "HOBIC"),
       "DS": ("Dysbiotic", "Static"), "DH": ("Dysbiotic", "HOBIC")}
DAYS = [3, 6, 10, 15, 21]
MIN_TOTAL = 0.02
GRID = np.linspace(1, 21, 2501)  # 図 2 のキャッシュの時間軸


def comp(df, c, cu):
    d = df[(df.condition == c) & (df.cultivation == cu)]
    out = {}
    for _, r in d.iterrows():
        if r.species in SP:
            out.setdefault(int(r.day), np.zeros(5))[SP[r.species]] += r["median"]
    return {k: v / v.sum() for k, v in out.items()}


def data_psi(c, cu):
    v = comp(pd.read_csv(D / "fig3_species_distribution_summary.csv"), c, cu)
    t = comp(pd.read_csv(D / "figS1_total_species_distribution_summary.csv"), c, cu)
    m = pd.read_csv(D / "fig2_membrane_distribution.csv")
    out = {}
    for d in DAYS:
        live = m[(m.condition == c) & (m.cultivation == cu) & (m.day == d)].intact_membrane_pct.values[0] / 100
        with np.errstate(divide="ignore", invalid="ignore"):
            out[d] = np.where(t[d] >= MIN_TOTAL, v[d] / t[d] * live, np.nan)
    return out


def main(cache, out_tex):
    res = {}
    for tag, (c, cu) in TAG.items():
        dp = data_psi(c, cu)
        z = np.load(Path(cache) / f"fig2_phi_{tag}.npz")
        psi = z["phibar_map"] / z["phi_map"]
        mp = {d: psi[np.argmin(np.abs(GRID - d))] for d in DAYS}
        A = np.array([dp[d] for d in DAYS])
        B = np.array([mp[d] for d in DAYS])
        res[tag] = {"data": [None if np.all(np.isnan(A[:, i])) else float(np.nanmedian(A[:, i])) for i in range(5)],
                    "n_days": [int(np.sum(~np.isnan(A[:, i]))) for i in range(5)],
                    "model": [float(np.median(B[:, i])) for i in range(5)]}
    f = lambda x: "--" if x is None else f"{x:.2f}"  # noqa: E731
    L = [r"\begin{tabular}{l" + "cc" * 5 + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in NAMES) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(5)),
         " & " + " & ".join(["data", "model"] * 5) + r" \\", r"\midrule"]
    for tag, r in res.items():
        L.append(tag + " & " + " & ".join(f"{f(r['data'][i])} & {f(r['model'][i])}" for i in range(5)) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    Path(out_tex).write_text("\n".join(L) + "\n")
    json.dump(res, open(str(out_tex).replace(".tex", ".json"), "w"), indent=1)
    for tag, r in res.items():
        print(tag, "data", [f(x) for x in r["data"]], "n", r["n_days"], "model", [f(x) for x in r["model"]])


if __name__ == "__main__":
    main(*sys.argv[1:3])
