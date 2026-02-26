#!/usr/bin/env python3
"""
build_pdfs.py

Build PDFs for docs/*.tex and docs/auto_best_run/*.tex using latexmk.

Why this exists:
- Rocky/RHEL environments often have TeX (pdflatex) but not latexmk in PATH.
- We install latexmk in ~/.local/bin and want the build to be "one command".
"""

from __future__ import annotations

import os
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def _pick_latexmk() -> str:
    cmd = shutil.which("latexmk")
    if cmd:
        return cmd
    fallback = Path.home() / ".local" / "bin" / "latexmk"
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(
        "latexmk not found. Expected `latexmk` in PATH or `~/.local/bin/latexmk`."
    )


def _build_one(latexmk: str, tex_path: Path, *, force: bool, engine: str) -> None:
    workdir = tex_path.parent
    env = os.environ.copy()

    # User-space TeX package fallback:
    # Some minimal TeX installs on Rocky/RHEL miss certain .sty files.
    # We support extracting RPMs under ~/.local/texrpmroot and add it to TEXINPUTS.
    texmf_dist = Path.home() / ".local" / "texrpmroot" / "usr" / "share" / "texlive" / "texmf-dist"
    if texmf_dist.exists():
        # Make kpathsea find *all* file types (tex, cls, sty, enc, fonts, ...)
        env["TEXMFHOME"] = str(texmf_dist)
        texrpm_tex = texmf_dist / "tex"
        if texrpm_tex.exists():
            env["TEXINPUTS"] = str(texrpm_tex) + "//:" + env.get("TEXINPUTS", "")

    if engine not in {"pdflatex", "xelatex"}:
        raise ValueError(f"Unknown engine: {engine}")

    cmd = [latexmk]
    if engine == "xelatex":
        # Ensure a PDF is produced when using XeLaTeX.
        # `-xelatex` alone may stop at .xdv in some latexmk setups; `-pdfxe` forces PDF output.
        cmd += ["-pdfxe"]
    else:
        cmd += ["-pdf"]
    cmd += [
        "-silent",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    if force:
        # Force rebuild (useful after installing missing TeX packages).
        cmd.insert(1, "-g")
    p = subprocess.run(cmd, cwd=str(workdir), text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"latexmk failed for {tex_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force", action="store_true", default=False, help="Force rebuild (latexmk -g)"
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    docs = repo_root / "docs"
    auto = docs / "auto_best_run"

    latexmk = _pick_latexmk()

    targets: List[Path] = []
    targets += [docs / "tmcmc_flow_report.tex", docs / "tmcmc_flow_slides.tex"]
    targets += [docs / "tmcmc_flow_report_en.tex", docs / "tmcmc_flow_slides_en.tex"]
    targets += [auto / "auto_best_run_report.tex", auto / "auto_best_run_slides.tex"]
    targets += [auto / "auto_best_run_report_en.tex", auto / "auto_best_run_slides_en.tex"]

    built: List[Tuple[str, str]] = []
    for t in targets:
        if t.exists():
            # Docs under docs/: use xelatex for CJK. English versions under docs/ can use pdflatex.
            # Auto summaries: use pdflatex.
            if t.parent == docs:
                engine = (
                    "xelatex"
                    if t.name.endswith(".tex") and not t.name.endswith("_en.tex")
                    else "pdflatex"
                )
            else:
                engine = "pdflatex"
            _build_one(latexmk, t, force=bool(args.force), engine=engine)
            built.append((str(t), str(t.with_suffix(".pdf"))))

    if not built:
        print("[build_pdfs] No .tex targets found.")
        return 0

    print("[build_pdfs] Built PDFs:")
    for src, pdf in built:
        # print paths relative to repo root for readability
        try:
            src_rel = str(Path(src).resolve().relative_to(repo_root))
            pdf_rel = str(Path(pdf).resolve().relative_to(repo_root))
        except Exception:
            src_rel, pdf_rel = src, pdf
        print(f"- {src_rel} -> {pdf_rel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
