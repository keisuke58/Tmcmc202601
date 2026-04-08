"""
build_pptx.py — PDF page images → PPTX (16:9, lossless embed)
Usage: python3 build_pptx.py
"""

import glob
import os
from pptx import Presentation
from pptx.util import Inches, Emu

# 16:9 slide dimensions (same as beamer aspectratio=169)
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]  # completely blank

imgs = sorted(glob.glob("/tmp/slide_*.png"))
for img_path in imgs:
    slide = prs.slides.add_slide(blank_layout)
    slide.shapes.add_picture(img_path, 0, 0, width=SLIDE_W, height=SLIDE_H)

out = os.path.join(os.path.dirname(__file__), "slides_nishioka.pptx")
prs.save(out)
print(f"Saved {len(imgs)} slides → {out}")
