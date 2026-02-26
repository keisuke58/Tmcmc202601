#!/usr/bin/env python3
"""
Extract pages containing specific figure captions from a paper PDF.

Why:
- The paper PDF is binary; grepping doesn't work reliably.
- For quick comparison, it's convenient to split out the pages that contain
  Fig. 8–15 into separate small PDFs.

Output:
- Writes one PDF per page that contains any of the requested figure numbers.
- Also writes an index markdown listing which figures were found on which pages.

Example:
  python tmcmc/extract_paper_fig_pages.py \
    "tmcmc/Bayesian updating ... Kopie.pdf" \
    "tmcmc/_paper_figs_8_15" \
    --figs 8,9,10,11,12,13,14,15
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from pypdf import PdfReader, PdfWriter


@dataclass(frozen=True)
class PageHit:
    page_index0: int
    page_number1: int
    figs: List[int]


def _parse_figs(s: str) -> List[int]:
    xs: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        xs.append(int(part))
    xs = sorted(set(xs))
    return xs


def _find_fig_hits(reader: PdfReader, figs: Sequence[int]) -> List[PageHit]:
    # Match common caption styles seen in extracted text:
    # - "Fig. 8"
    # - "Fig. 11:"
    # - "Figure 9"
    patterns = {k: re.compile(rf"\b(Fig\.?|Figure)\s*{k}\b", flags=re.IGNORECASE) for k in figs}

    hits_by_page: Dict[int, List[int]] = {}
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        found: List[int] = []
        for k, pat in patterns.items():
            if pat.search(text):
                found.append(int(k))
        if found:
            hits_by_page[i] = sorted(set(found))

    out: List[PageHit] = []
    for i in sorted(hits_by_page.keys()):
        out.append(PageHit(page_index0=i, page_number1=i + 1, figs=hits_by_page[i]))
    return out


def _write_page_pdf(reader: PdfReader, page_index0: int, out_path: Path) -> None:
    w = PdfWriter()
    w.add_page(reader.pages[page_index0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        w.write(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract figure pages from a paper PDF.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--figs", type=str, default="8,9,10,11,12,13,14,15")
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    out_dir: Path = args.out_dir
    figs = _parse_figs(str(args.figs))

    reader = PdfReader(str(pdf_path))
    hits = _find_fig_hits(reader, figs)

    out_dir.mkdir(parents=True, exist_ok=True)

    index_lines: List[str] = []
    index_lines.append("## Extracted paper figure pages")
    index_lines.append("")
    index_lines.append(f"- **source**: `{pdf_path}`")
    index_lines.append(f"- **requested_figs**: `{','.join(map(str, figs))}`")
    index_lines.append("")

    if not hits:
        index_lines.append(
            "No figure captions found (text extraction might not include captions on this PDF)."
        )
        (out_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
        return 0

    index_lines.append("| PDF page | figures found | output file |")
    index_lines.append("|---:|---|---|")

    for h in hits:
        tag = "-".join(str(x) for x in h.figs)
        out_name = f"paper_page{h.page_number1:03d}_fig{tag}.pdf"
        out_path = out_dir / out_name
        _write_page_pdf(reader, h.page_index0, out_path)
        index_lines.append(f"| {h.page_number1} | {', '.join(map(str, h.figs))} | `{out_name}` |")

    (out_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"[extract_paper_fig_pages] wrote {len(hits)} PDFs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
