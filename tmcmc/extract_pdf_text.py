#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_text(pdf_path: Path, max_pages: int | None) -> str:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    if max_pages is not None:
        pages = pages[: max_pages]

    out: list[str] = []
    for i, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # pragma: no cover
            text = f"[extract_text failed on page {i}: {type(e).__name__}: {e}]"
        out.append(f"\n\n===== PAGE {i} =====\n")
        out.append(text)

    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract text from a PDF.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Only extract the first N pages (default: all pages).",
    )
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    out_path: Path = args.out

    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = extract_text(pdf_path, args.max_pages)
    out_path.write_text(text, encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()

