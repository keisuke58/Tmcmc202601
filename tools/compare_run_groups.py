#!/usr/bin/env python3
"""2 つの run 群（3 seed 合算）の 15 成分を並べる（13f の箱 1.5 倍・13g の重みなしの対照表、2026-10-13）。

分類はそれぞれ自分の箱で、make_supplementary_tables.py の S4 と同じ規則:
  事後 sd < 0.8 × 箱の一様分布の sd → 同定（id）
  そうでなく 90% 区間が 0 を含まない → 符号だけ（s.o.）
  それ以外 → n.i.
「符号が同じ」は中央値の符号の一致（id・s.o. の成分だけ数える。n.i. は符号に意味がない）。

    python3 tools/compare_run_groups.py data_5species/main/_runs/paper_gateoff \
        'CH_p2_2c_seed*' 'CH_p2_2c_box15_seed*' --labels 2c,box15 --out docs/handoff/gpu_..._box15_CH
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

UNIDENT_RATIO = 0.8
# θ（20 次元）→ 15 成分（tools/make_paper_figures.py の KEEP_IDX・KEEP_NAME と同じ並び）
KEEP_IDX = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]
KEEP_NAME = ["a11", "a12", "a22", "a33", "a34", "a44", "a13", "a14",
             "a23", "a24", "a55", "a15", "a25", "a35", "a45"]  # fmt: skip


def load_group(root, pattern):
    ds = sorted(d for d in glob.glob(str(Path(root) / pattern)) if (Path(d) / "samples.npy").exists())
    if not ds:
        raise SystemExit(f"run が無い: {pattern}")
    S = np.concatenate([np.load(Path(d) / "samples.npy") for d in ds])
    rec = json.load(open(Path(ds[0]) / "run_record.json"))
    pb = np.array(rec.get("prior_bounds_final") or rec["prior_bounds"], dtype=float)
    out = {}
    for i, nm in zip(KEEP_IDX, KEEP_NAME):
        x = S[:, i]
        q05, q50, q95 = np.percentile(x, [5, 50, 95])
        unif = (pb[i, 1] - pb[i, 0]) / np.sqrt(12.0)
        if unif <= 0:
            cls = "fixed"
        elif x.std() < UNIDENT_RATIO * unif:
            cls = "id"
        else:
            cls = "s.o." if (q05 > 0 or q95 < 0) else "n.i."
        out[nm] = {"cls": cls, "q05": q05, "q50": q50, "q95": q95, "box": pb[i].tolist(),
                   "ratio": float(x.std() / unif) if unif > 0 else None}
    return [Path(d).name for d in ds], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("ref")
    ap.add_argument("alt")
    ap.add_argument("--labels", default="ref,alt")
    ap.add_argument("--out", default=None, help="拡張子なし。.md と .json を書く")
    a = ap.parse_args()
    la, lb = a.labels.split(",")
    ra, A = load_group(a.root, a.ref)
    rb, B = load_group(a.root, a.alt)

    f = lambda c: f"{c['q50']:+.2f} [{c['q05']:+.2f}, {c['q95']:+.2f}]"  # noqa: E731
    L = [f"- {la}: `{a.ref}`（{len(ra)} run）", f"- {lb}: `{a.alt}`（{len(rb)} run）", "",
         f"| 成分 | {la} 分類 | {la} 中央値 [5%, 95%] | {lb} 分類 | {lb} 中央値 [5%, 95%] | 分類 | 符号 |",
         "|---|---|---|---|---|---|---|"]
    same_cls = n_sign = same_sign = 0
    for nm in KEEP_NAME:
        x, y = A[nm], B[nm]
        sc = x["cls"] == y["cls"]
        same_cls += sc
        sg = ""
        if x["cls"] in ("id", "s.o.") and y["cls"] in ("id", "s.o."):
            n_sign += 1
            ok = np.sign(x["q50"]) == np.sign(y["q50"])
            same_sign += ok
            sg = "同じ" if ok else "**逆**"
        L.append(f"| {nm} | {x['cls']} | {f(x)} | {y['cls']} | {f(y)} | {'同じ' if sc else '**変化**'} | {sg} |")
    cnt = lambda G, c: sum(v["cls"] == c for v in G.values())  # noqa: E731
    L += ["", f"- 分類が同じ成分: {same_cls} / 15",
          f"- 符号が同じ成分（両方で id か s.o. のもの）: {same_sign} / {n_sign}",
          f"- {la}: id {cnt(A, 'id')}・s.o. {cnt(A, 's.o.')}・n.i. {cnt(A, 'n.i.')}",
          f"- {lb}: id {cnt(B, 'id')}・s.o. {cnt(B, 's.o.')}・n.i. {cnt(B, 'n.i.')}"]
    md = "\n".join(L) + "\n"
    print(md)
    if a.out:
        Path(a.out + ".md").write_text(md)
        json.dump({"ref": {"glob": a.ref, "runs": ra, "comp": A}, "alt": {"glob": a.alt, "runs": rb, "comp": B}},
                  open(a.out + ".json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
