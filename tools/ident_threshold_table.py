#!/usr/bin/env python3
"""同定された成分の数が、しきい値（事後 sd / 箱の一様分布の sd）でどう変わるか（EML 補足、2026-10-12）。

本文・補足の分類は UNIDENT_RATIO = 0.8（check_paper_runs.py・make_supplementary_tables.py と同じ）。
ここでは 0.5・0.7・0.8・0.9 で数え直し、LaTeX の表（tabular だけ）を書く。
run の群は supp/spec_provisional.json の "ult"（DS・DH は暫定で段 (iii)）。

    python3 tools/ident_threshold_table.py docs/revision/generated/final_runs \
        docs/revision/EML_submission/figures/provisional_2026-10-11/supp/spec_provisional.json \
        docs/revision/EML_submission/figures/provisional_2026-10-11/supp/supp_S4_thresh.tex
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

CONDS = ["CS", "CH", "DS", "DH"]
THRESH = [0.5, 0.7, 0.8, 0.9]
SP = ["So", "An", "Vei", "Fn", "Pg"]
# θ の並び（20 次元）のうち A の 15 成分 → (i, j)
A_IDX = {0: (1, 1), 1: (1, 2), 2: (2, 2), 5: (3, 3), 6: (3, 4), 7: (4, 4), 10: (1, 3), 11: (1, 4),
         12: (2, 3), 13: (2, 4), 14: (5, 5), 16: (1, 5), 17: (2, 5), 18: (3, 5), 19: (4, 5)}


def ratios(root, pattern):
    ds = sorted(glob.glob(str(Path(root) / pattern)))
    S = np.concatenate([np.load(Path(d) / "samples.npy") for d in ds])
    rec = json.load(open(Path(ds[0]) / "run_record.json"))
    pb = np.array(rec.get("prior_bounds_final") or rec["prior_bounds"], dtype=float)
    out = {}
    for k in A_IDX:
        w = pb[k, 1] - pb[k, 0]
        if w > 0:
            out[k] = S[:, k].std() / (w / np.sqrt(12.0))
    return out


def main(root, spec, out):
    spec = json.load(open(spec))
    rows, sets = [], {}
    for t in CONDS:
        g = spec.get(t, {}).get("ult")
        if not g or not glob.glob(str(Path(root) / g)):  # 最終段がまだ無い条件（2026-10-16）
            rows.append((t, None))
            continue
        r = ratios(root, g)
        sets[t] = {c: sorted(k for k, v in r.items() if v < c) for c in THRESH}
        rows.append((t, [len(sets[t][c]) for c in THRESH]))
    lines = [r"\begin{tabular}{l" + "c" * len(THRESH) + "}", r"\toprule",
             "Threshold & " + " & ".join(f"{c:.1f}" + (r"$^\ast$" if c == 0.8 else "") for c in THRESH) + r" \\",
             r"\midrule"]
    for t, n in rows:
        if n is None:
            lines.append(f"{t} & " + r"\multicolumn{" + str(len(THRESH)) + r"}{l}{(pending)} \\")
            continue
        lines.append(f"{t} & " + " & ".join(str(v) for v in n) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines) + "\n")
    for t in sets:
        for c in THRESH:
            print(t, c, [f"a{A_IDX[k][0]}{A_IDX[k][1]}" for k in sets[t][c]])


if __name__ == "__main__":
    main(*sys.argv[1:4])
