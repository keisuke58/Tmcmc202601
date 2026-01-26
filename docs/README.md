## tmcmc ドキュメント生成（LaTeX / スライド）

この `docs/` は `tmcmc/case2_tmcmc_linearization.py` を中心とした
TMCMC × TSM-ROM（線形化管理 + 解析微分/JIT）の説明資料テンプレです。

### 生成物

- `tmcmc_flow_report.tex`
  - 論文調のレポート（article）
- `tmcmc_flow_slides.tex`
  - Beamerスライド（PDFスライド。PowerPointの代替として使える）
- `tmcmc_slides.pptx.md`
  - Pandoc で PowerPoint（.pptx）に変換できる Markdown スライド原稿

### LaTeX ビルド例

レポート:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode tmcmc_flow_report.tex
```

スライド:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode tmcmc_flow_slides.tex
```

### まとめてPDFビルド（推奨）

`latexmk` の場所を自動検出して、レポート/スライドをまとめてPDF化します。

```bash
python3 docs/build_pdfs.py
```

### PowerPoint（.pptx）生成例（Pandoc）

Pandoc が使える環境なら:

```bash
cd docs
pandoc -t pptx -o tmcmc_flow_slides.pptx tmcmc_slides.pptx.md
```

※ `.pptx` はバイナリなので、このリポジトリ内では「原稿（.md）」のみを用意しています。

