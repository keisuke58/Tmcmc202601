# BMB 投稿の準備状況（2026-10-08）

投稿予定 **10/16**（Editorial Manager: https://www.editorialmanager.com/bmab/ 、ユーザー `knishioka`）。

## できているもの

| 項目 | 状態 |
|---|---|
| 原稿（sn-jnl・通し行番号） | ✅ コンパイル通る（22 ページ、エラー 0） |
| 速度の数字 | ✅ 「~200×」は表の計算ミス（21600 s ÷ 147 s = 147×）。要旨・序論・結論を「~150×」、表を 147×、注を「1 段あたり 170×・合計 147×」に直した（2026-10-08） |
| 要旨 | ✅ 218 語（上限 250）。DS の a45 を「主な山の中でだけ正」に直した（2026-10-08） |
| Keywords | ✅ 6 個 |
| Declarations（Funding・Competing interests・Ethics・Data・Code・Author contributions） | ✅ 節はある（Author contributions は共著者の確認待ち） |
| Cover letter | ✅ 下書き。DS の言い方を本文とそろえた |
| 補足資料 `BMB_supplementary.tex` | ✅ 骨組み（S1 箱・S2 run・S3 Bayes 因子と DS の山・S4 ±2SD・S5 λ）。本文の参照番号を S1・S4 に決めた |
| ORCID・EM 登録・責任著者の大学アドレス | ✅ |

## ult の結果が出たら埋めるもの（10/10〜12）

- 本文の数値の `\TBD`（約 35 か所）: RMSE、a45 の区間、ノックアウトの確率、時間・速度（`~200×` を含む）、Phase 1 と 2 の比較、相関など
- §Results 冒頭の「前の run の数字」の注記を消す
- 補足資料 S1・S2 の表（`make_paper_figures.py` の出力）、S3 の Bayes 因子の表（CS・CH の a45=0 mut120 待ち）、S3 の DS の山ごとの表
- 図 5 枚の差し替え（`make_paper_figures.py`、glob は 08q のもの）

## 決めてほしいこと（ユーザー）

1. **本文 330 行目「組成データの和が 1 という制約を無視している」の扱い** — いまは「感度解析で確認（M.S. と相談）」の `\TBD`。
   感度解析を回す時間はないので、案:「先行研究との比較のためこの近似を残し、限界として議論で触れる」と書く。Meisam に一言確認するか。
2. **±2SD の節（補足 S4）** — ±2SD の ult は投稿に間に合わない見込み。節と本文の参照を消すか、「査読時に追加」とするか。
3. **λ 感度（補足 S5）** — 走行中。10/12 までに出なければ本文の段落ごと次の版に回すか。
4. **公開 GitHub の Heine の CSV**（`experiment_data/`）— Heine 氏に断るか、repo から外すか。Zenodo には入れない。
5. **推薦査読者**（任意）— 候補（共著者・IKM・MHH・SIIRI との共著なし、2026-10-08 確認）:
   - Hermann J. Eberl（University of Guelph, Mathematics & Statistics）heberl@uoguelph.ca — バイオフィルムの数理モデル
   - Isaac Klapper（Temple University, Mathematics）メールは Temple の名簿で確認 — バイオフィルムの連続体モデル、バイオフィルムモデルのベイズ推定
   - Costas Papadimitriou（University of Thessaly, Mechanical Engineering）costasp@uth.gr — TMCMC・ベイズ UQ
   - 予備: Katharine Coyte（University of Manchester）— 微生物群集の相互作用の生態学
7. **pH の関係式（本文 329 行目、`\TBD{source of the pH relation}`）** — 出典を調べたところ、`evaluator.py` のコメントは
   「Heine 2025 の検証データへの事後回帰（R²=0.71）」。つまり**同じ pH データから回帰で作った式を、その pH データの尤度に使っている**。
   しかも回帰の切片は 6.95（`docs/make_pptx.py`）なのに実装は 7.5 で、違う理由が記録にない。査読で突かれやすい。案:
   (a) 正直に書く:「経験的な観測演算子で、以前の推定の組成に対する回帰から得た。重み λ=0.3 と小さい」＋切片の違いを確認して説明
   (b) pH チャネルを外した p2 を回して結論が変わらないことを示す（GPU に余裕があれば）
6. **Zenodo の DOI** — 最終の run が出たら作る（`experiment_data/` を除く）。

## 共著者から待つもの

- 10/13 に最終版を送る → 10/15 までに意見・承認（Author contributions・所属・Funding の確認を含む）
- Cover letter の「All authors have approved」は承認が出てから `\TBD` を外す

## 日程

| 日 | やること |
|---|---|
| 10/10〜11 | ult 4 条件そろう → 判定 |
| 10/11〜12 | 図表・数値を埋める、補足資料を仕上げる、Zenodo |
| 10/13 | 共著者へ最終版（Gmail の下書き r3514969574136866139） |
| 10/15 | 共著者の意見の締め切り |
| 10/16 | 投稿 |
