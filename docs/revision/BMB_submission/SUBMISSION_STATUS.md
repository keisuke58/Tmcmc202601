> **2026-10-09: 投稿先を変更。** Meisam・Junker 先生の判断で、BMB ではなく **Extreme Mechanics Letters の特集号「AI & CM」（VSI: AI & CM、締切 2027-04-01）**に出す。
> EML は本文 4000 語未満・図 6 枚まで（要確認）なので、計算が終わったら短縮版を作る。BMB 版はこのフォルダに残す（短縮版の元）。10/13・10/16 の日程は取り消し。

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

## ⚠️ 結果の変化（2026-10-09、pH を外した p2）— 原稿の主張に関わる

- **DH の a45**: 組成だけの段（pilot・p1）では +3.0 [+0.8, +5.2] と安定していたが、生存率を足した p2（pH なし）では
  当てはまりの良い 2 seed で中央値 +0.7〜+1.2、下限 −0.5 の近くに 2〜4 割の質量（90% 区間は [−0.16, +7.5] / [+0.16, +13]）。
  a45 の下限を −5 に広げて 10000 粒子で回し直し中。**負側に出れば「DH の a45 > 0」は Phase 1 の結果として書き、Phase 2 では決まらない、と直す**。
  題名の "a testable prediction" の扱いも含めて、結果が出たら Meisam と決める。
- **DH の a35（Vei–Pg）**: p2 で −4〜−10（Phase 1 は +0.3）。原稿 621 行・787 行の「DH で Vei–Pg が正」は Phase 2 では成り立たない → ult 後に書き直す。
- **DS の a33 の二峰**: 5000 粒子・箱を広げた p2 では消えた（単峰 −0.7）。原稿に入れた「DS の a45 を山ごとに」の文は ult 後に削る。
- **DS の a45**: 中央値 +1.8〜+2.8 だが 90% 区間は 0 を含む（[−1.5, +14.8]）。
- 良い点: pH を外して RMSE が DH 0.140 → 0.087、CH 0.149 → 0.075、CS 0.20 → 0.17 に改善。

## 決めてほしいこと（ユーザー）

1. ✅（2026-10-09）**和が 1 の制約** — Discussion に Limitations の段落を新設して書いた（10/13 の最終版で Meisam も見る）。元のメモ: **本文 330 行目「組成データの和が 1 という制約を無視している」の扱い** — いまは「感度解析で確認（M.S. と相談）」の `\TBD`。
   感度解析を回す時間はないので、案:「先行研究との比較のためこの近似を残し、限界として議論で触れる」と書く。Meisam に一言確認するか。
2. ✅（2026-10-09）**±2SD** — 投稿に間に合わないので、補足 S4 と本文の参照を削除（査読で聞かれたら sd2 の run で答える）。元のメモ: **±2SD の節（補足 S4）** — ±2SD の ult は投稿に間に合わない見込み。節と本文の参照を消すか、「査読時に追加」とするか。
3. ✅ **λ 感度（補足 S5）** — 全群 PASS（2026-10-09）。本文の段落と補足の表に記入済み。a45 の中央値は変わらず、区間の下端が 0 に届く
4. ✅（2026-10-09 ユーザー判断）**公開 GitHub の Heine のデータ（`data_5species/experiment_data/`、79 ファイル）は最新版から外す。**
   - **やるのは 10/12（ult と図が全部そろってから）、GPU 側で** `git rm -r --cached data_5species/experiment_data` ＋ `.gitignore` に追加 → commit・push。
     `--cached` なので GPU サーバーのファイルは残る（計算は止まらない）。今やると、GPU 側が pull したときにファイルが消えて計算が壊れる。
   - 他の clone（手元の PC など）は pull するとファイルが消えるので、先に `experiment_data/` を別の場所に退避しておく。
   - 過去のコミットには残る（履歴の書き換えは他の clone を壊すのでしない）。Zenodo には入れない。
5. **推薦査読者**（任意）— 候補（共著者・IKM・MHH・SIIRI との共著なし、2026-10-08 確認）:
   - Hermann J. Eberl（University of Guelph, Mathematics & Statistics）heberl@uoguelph.ca — バイオフィルムの数理モデル
   - Isaac Klapper（Temple University, Mathematics）メールは Temple の名簿で確認 — バイオフィルムの連続体モデル、バイオフィルムモデルのベイズ推定
   - Costas Papadimitriou（University of Thessaly, Mechanical Engineering）costasp@uth.gr — TMCMC・ベイズ UQ
   - 予備: Katharine Coyte（University of Manchester）— 微生物群集の相互作用の生態学
7. ✅ **pH** — 推定から外し、全種の回帰式（6.10 + 0.16 So + 0.55 An + 0.30 Vei + 1.22 Fn + 1.88 Pg）による答え合わせに戻した（ユーザー判断 2026-10-08、08r）。DH・CH は p2 から回し直し。R² などは `\TBD`
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

## ult が出たら（GPU 側、3 コマンド）

```bash
python3 tools/make_paper_figures.py data_5species/main/_runs/paper_gateoff <glob 群>   # 図 5 枚 + generated/paper_numbers.json
python3 tools/fill_manuscript.py docs/revision/generated/paper_numbers.json --extra extra.json   # 本文の表 4 つ・Table 2 の行・数値マクロ
python3 tools/make_supplementary_tables.py data_5species/main/_runs/paper_gateoff          # 補足 S1〜S3
```
`extra.json` には pH の答え合わせ（ph_r2・ph_rmse・ph_n_samples・ph_n_points）、gingipain の r、Zenodo の DOI を入れる。
表は `docs/revision/generated/*.tex` を原稿が `\input` するので、生成すればそのまま反映される。本文中の数（a45 の区間など）は、結果を見てからクラウド側で文ごと書き直す（主張が変わりうるため自動にはしない）。
