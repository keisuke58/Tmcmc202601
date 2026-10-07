# Bulletin of Mathematical Biology（BMB）投稿用

投稿先 1 番目の **Bulletin of Mathematical Biology**（Springer、Editorial Manager: https://www.editorialmanager.com/bmab/ 、ユーザー名 `knishioka`）に出すためのファイル一式。

| ファイル | 中身 |
|---|---|
| `BMB_manuscript_nishioka.tex` | 原稿（sn-jnl v3.1、`sn-mathphys-ay` = BMB の著者・年の引用、通し行番号）。赤字 `[...]`（`\TBD`）は ult の数値・共著者確認待ち |
| `BMB_cover_letter.tex` | Cover letter の下書き |
| `references_ikm.bib` | 文献 |
| `sn-jnl.cls`, `sn-mathphys-ay.bst` | Springer Nature のテンプレート（Editorial Manager でのコンパイルに必要、そのまま一緒にアップロード） |
| `figures/` | 本文で使う図だけ |

コンパイル: `pdflatex → bibtex → pdflatex ×2`（`BMB_manuscript_nishioka`）

起点は共著者に回した 2026-09-16 版（commit 015d627、article クラス）。書き直しの中身は `../manuscript_changes.md`、
投稿手順とチェックリストは `../journal_submission_guide.md`。
2 番目の投稿先（Journal of Biological Dynamics）に回すときは、このフォルダを複製して別名にする。
