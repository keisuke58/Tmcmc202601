# 論文 → PowerPoint 変換ツール

## 概要

`paper_summary_ja.md` などの Markdown 形式の論文サマリを、PowerPoint (.pptx) に自動変換するスクリプトです。

## セットアップ

```bash
pip install -r tools/requirements-pptx.txt
```

## 使い方

```bash
# デフォルト: docs/paper_summary_ja.md → docs/paper_slides.pptx
python tools/paper_to_pptx.py

# 入出力を指定
python tools/paper_to_pptx.py input.md output.pptx
```

## 対応形式

- **見出し**: `#`, `##`, `###` → スライドタイトル（セクション区切りは色付きバー）
- **箇条書き**: `-` または `*` → 本文
- **表**: `| col1 | col2 |` → PowerPoint ネイティブ表（ヘッダー色付き）
- **画像**: `![alt](path)` → スライドに挿入（相対パスは md ファイル基準）
- **番号付きリスト**: `1.`, `2.` など

## デザイン

- ネイビー・ティールのアカデミック配色
- セクション区切りスライド（見出しのみ）はダークスレート背景
- 表はヘッダー行をネイビーで強調

## 出力例

- 口頭発表用の骨子スライド
- 論文の構成に沿った発表資料（図も自動挿入）
- 口頭試問対策の補助資料

画像パスは Markdown からの相対パスで解決します（例: `../FEM/figures/paper_final/Fig08.png`）。
