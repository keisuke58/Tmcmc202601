# 投稿の日にやること（EML、VSI: AI & CM）— 2026-10-19 に点検

共著者の返事（締切 10/30）が全部そろってから。上から順に。

## 0. 点検の結果（10-19）

| 項目 | 規程 | 状態 |
|---|---|---|
| 本文の語数 | < 4000 | 3915（detex、図の説明込み）✔ |
| 要旨 | ≤ 250 語 | 238 ✔ |
| 図 | ≤ 6 枚、1 枚ずつ別ファイル、フォント埋め込み | 4 枚、`tools/export_submission_figs.py` で「問題なし」✔ |
| Highlights | 3〜5 項目、各 85 字以内 | 5 項目、72〜80 字 ✔（`highlights.txt`） |
| Graphical abstract | 横長、幅 13 cm で読める | 13.1 × 5.1 cm ✔。**「4–9 of 15」→「5–9 of 15」に直した（10-19）** |
| キーワード | 1〜7 | 6 ✔ |
| 責任著者 | 1 人 | Nishioka（stud.uni-hannover.de と keio.jp）✔ |
| Funding | DFG は Nils 指定の文言そのまま、ERC は EU の定型文 | ✔ |
| 利益相反 | 本文に定型文＋ツールの .docx | 本文 ✔、**.docx はまだ（下の 2）** |
| 生成 AI の申告 | 参考文献の前、決まった書式 | ✔（10-12 の決定どおり） |
| データの記述 | 本文と投稿システムで同じ文 | 本文 ✔、システム用の文は下の 3 |
| CRediT | 全員分 | 案のみ（`\TBD`）→ 返事で確定 |
| 補足の `\TBD` | なし | `supp_S_lambda2c.tex` はある（`\TBD` は無いとき用の予備）✔ |

## 1. 返事を受けて直すもの

- [ ] 原稿: CRediT の `\TBD{draft, ...}` の行を消し、直しがあれば反映。
- [ ] 原稿: 名前・所属の直し（あれば）。
- [ ] カバーレター: `\date{\TBD{date}}` → 投稿日、`\TBD{confirm}` を消す。
- [ ] 補足: 1 日外しの予測（13f-5）の段落と表（結果が届いていれば）。
- [ ] 全部ビルドし直して `\TBD` が 0 個か確かめる: `grep -n 'TBD{' *.tex`（定義の行を除く）。
- [ ] `python3 tools/export_submission_figs.py` をもう一度（「問題なし」を確認）。

## 2. 本人の作業（Web で）

- [ ] **利益相反の .docx**: https://declarations.elsevier.com/ で作る（全員「なし」でよいかは共著者メールで確認済みになる）。
- [ ] **Zenodo を公開**して DOI 10.5281/zenodo.23320423 を有効にする（今は予約だけ）。
      その前に `docs/revision/zenodo/MANIFEST.txt` で実験データが入っていないことを確認（`build_archive.sh` は入っていると止まる）。
- [x] Elsevier のアカウント（10-19、keisuke58@keio.jp・慶應の大学ログイン、あと約 1 年半有効）。投稿は https://submit.elsevier.com/EML（Editorial Manager ではない新しいシステム）。プロフィールに Gmail を 2 つ目のアドレスで。
- [ ] 特集号を選ぶ: "VSI: AI & CM"（Artificial Intelligence and Computational Mechanics for Materials Design）。
- [ ] SSRN のプレプリントの選択: **しない**（共著者全員の同意が要るため、聞いていない）。

## 3. 投稿システムに貼る文

**Data availability（投稿システムの声明。本文の "Data and code availability" と同じ内容）**
> Code, run configurations, convergence records and posterior samples are archived at Zenodo (doi:10.5281/zenodo.23320423). The experimental data are those of Heine et al. (2025) and are available from those authors on request; they are not redistributed with the archive.

（選択肢があれば「Data will be made available on request」ではなく「I have shared the link to my data/code at the Attach File step」でもなく、**"Other"** を選んでこの文を貼る。）

## 4. 上げるファイル（順に）

| 種類 | ファイル |
|---|---|
| Cover letter | `EML_cover_letter.pdf` |
| Manuscript（LaTeX 一式） | `EML_manuscript_nishioka.tex`・`references_ikm.bib`・`.bbl`、図（下）、`elsarticle` 以外の独自のスタイルは無し |
| Figures | `submission_figures/Figure_1.pdf` 〜 `Figure_4.pdf` |
| Graphical abstract | `submission_figures/graphical_abstract.pdf` |
| Highlights | `highlights.txt`（名前に "highlights" が入っていること） |
| Supplementary material | `EML_supplementary.pdf` |
| Declaration of interest | 上の 2 で作った .docx |
| 査読者の候補（求められたら） | 未定 → 必要なら Junker 先生・Meisam と相談 |
