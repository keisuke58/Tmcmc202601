#!/usr/bin/env python3
"""投稿用に本文の図を 1 枚ずつ別ファイルにする（EML の Guide for Authors、2026-10-14）。

本文の .tex から \\includegraphics を出てくる順に読み、Figure_1.pdf, Figure_2.pdf, ... として --out に写す
（本文がどの図を使っているかは .tex に従う。provisional → raw に差し替えたら、これを回し直すだけ）。
graphical abstract は graphical_abstract.pdf として一緒に写す。

確かめること（どれかに引っかかったら終了コード 1）:
  - 図は 6 枚まで（超えると審査なしで不採択）
  - PDF のフォントがすべて埋め込まれている（pdffonts の emb 列）
  - 埋め込みの画像（ビットマップ）は 300 dpi 以上（pdfimages -list の x-ppi・y-ppi）
  - graphical abstract は横長で、幅 1328 px 相当（13 cm で 260 dpi 弱）以上 → ベクタなら幅 13 cm 以上で読めることだけ見る

    python3 tools/export_submission_figs.py [--tex docs/revision/EML_submission/EML_manuscript_nishioka.tex] \
        [--out docs/revision/EML_submission/submission_figures]
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "docs" / "revision" / "EML_submission"
MAX_FIGS = 6
MIN_PPI = 300
GA_MIN_WIDTH_CM = 13.0


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def fonts_not_embedded(pdf):
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout.splitlines()
    bad = []
    for line in out[2:]:  # 見出し 2 行を飛ばす
        cols = line.split()
        # 列: name type [encoding...] emb sub uni object ID → 後ろから 5 番目が emb
        if len(cols) >= 6 and cols[-5] != "yes":
            bad.append(cols[0])
    return bad


def low_res_images(pdf):
    out = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True).stdout.splitlines()
    bad = []
    for line in out[2:]:
        cols = line.split()
        if len(cols) >= 14 and cols[2] in ("image", "smask") and cols[2] == "image":
            try:
                xppi, yppi = float(cols[12]), float(cols[13])
            except ValueError:
                continue
            if min(xppi, yppi) < MIN_PPI:
                bad.append(f"page {cols[0]} image {cols[1]}: {xppi:.0f}x{yppi:.0f} ppi")
    return bad


def page_size_cm(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out)
    w, h = float(m.group(1)), float(m.group(2))
    return w / 72 * 2.54, h / 72 * 2.54


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(SUB / "EML_manuscript_nishioka.tex"))
    ap.add_argument("--out", default=str(SUB / "submission_figures"))
    ap.add_argument("--ga", default=str(SUB / "figures" / "graphical_abstract.pdf"))
    a = ap.parse_args()
    tex = Path(a.tex)
    body = strip_comments(tex.read_text())
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    if len(figs) > MAX_FIGS:
        problems.append(f"図が {len(figs)} 枚（上限 {MAX_FIGS}）")
    for k, rel in enumerate(figs, 1):
        src = (tex.parent / rel)
        if src.suffix == "":
            src = src.with_suffix(".pdf")
        if not src.exists():
            problems.append(f"Figure {k}: {rel} が無い")
            continue
        dst = out / f"Figure_{k}{src.suffix}"
        shutil.copy2(src, dst)
        note = []
        if src.suffix == ".pdf":
            nf = fonts_not_embedded(src)
            if nf:
                problems.append(f"Figure {k}: 埋め込まれていないフォント {nf}")
            lr = low_res_images(src)
            if lr:
                problems.append(f"Figure {k}: 解像度が {MIN_PPI} ppi 未満 {lr}")
            w, h = page_size_cm(src)
            note.append(f"{w:.1f}x{h:.1f} cm")
        if "provisional" in rel:
            note.append("暫定の図（provisional）")
        print(f"Figure_{k}: {rel}  ({', '.join(note)})")
    ga = Path(a.ga)
    if ga.exists():
        shutil.copy2(ga, out / "graphical_abstract.pdf")
        w, h = page_size_cm(ga)
        print(f"graphical_abstract: {ga.relative_to(ROOT)}  ({w:.1f}x{h:.1f} cm)")
        if w <= h:
            problems.append("graphical abstract が横長でない")
        if w < GA_MIN_WIDTH_CM - 0.5:
            problems.append(f"graphical abstract の幅 {w:.1f} cm < {GA_MIN_WIDTH_CM} cm")
        nf = fonts_not_embedded(ga)
        if nf:
            problems.append(f"graphical abstract: 埋め込まれていないフォント {nf}")
    else:
        print("graphical abstract が無い（任意）")
    print(f"-> {out}")
    if problems:
        print("問題:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("問題なし")


if __name__ == "__main__":
    main()
