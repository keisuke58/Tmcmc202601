# Extreme Mechanics Letters 版の構成案（2026-10-09）

投稿先: Extreme Mechanics Letters、特集号 **VSI: AI & CM**（"Artificial Intelligence and Computational Mechanics for Materials Design"、締切 2027-04-01）。
元原稿: `../BMB_submission/BMB_manuscript_nishioka.tex`（本文 約 8800 語、図 14・表 5）。

## 1. 投稿規程（Guide for Authors、2026-10-09 確認）

| 項目 | 規程 | 備考 |
|---|---|---|
| 本文の長さ | **4000 語未満** | 超えると**審査なしで不採択**。参考文献・図の説明・Declarations が数に入るかは未確認 → 本文＋図の説明で 3800 語に収める |
| 図 | **6 枚まで** | 表を数に含むかは未確認 → 本文の表は 0〜1 にして、残りは補足へ |
| Highlights | 3〜5 項目、各 85 字以内 | 別ファイル（ファイル名に "highlights"） |
| 原稿ファイル | Word か LaTeX（PDF は不可） | Elsevier の `elsarticle` |
| 特集号の選択 | 投稿システムで "VSI: AI & CM" を選ぶ | |
| 公開までの目安 | 投稿から 6〜8 週で online | |

未確認（投稿前に Guide for Authors の全文で確認）: 図の数に表を含むか／要旨の語数／補足資料の扱い／graphical abstract が必須か。

## 2. 打ち出し方の変更

- BMB 版は「生物学の予測」が主役。EML の特集は「AI と計算力学で材料を設計」なので、**計算力学の枠組みを主役**にする:
  拡張 Hamilton 原理による熱力学的に整合な多相連続体モデル × GPU での並列ベイズ推定（TMCMC）× 段階的な尤度と収束・同定性の確認。
- バイオフィルムは「生物由来の材料」（特集の対象に biological materials あり）として位置づける。Fn を除く予測は、モデルを使った「設計的な問い」の例として最後に短く残す。
- 題名の案:
  1. *Bayesian calibration of a Hamilton-principle continuum model of multispecies biofilms on the GPU*
  2. *GPU-accelerated Bayesian inference of interaction parameters in a variational multiphase biofilm model*
  （最終の結果、特に DH の a45 を見てから決める）

## 3. 本文の語数配分（合計 3800 語）

| 節 | 語数 | 中身（元原稿の節） | 元の語数 |
|---|---|---|---|
| Abstract | 150 | 書き直し | 240 |
| 1 Introduction | 450 | 問題（多種間相互作用は直接測れない）、連続体モデル、ベイズ同定の難しさ、本研究の 3 点 | 440 |
| 2 Model | 550 | 状態変数、拡張 Hamilton 原理（式は 3 本に絞る）、対称な相互作用行列 | 104+360+487 |
| 3 Bayesian inference | 800 | 尤度（組成＋生存率）、4 段の推定、TMCMC の要点、GPU（vmap・約 150 倍）、収束と箱の規則 | 319+666+500+581 |
| 4 Results | 1300 | 当てはまり（4 条件）、相互作用行列、事後と同定性、a45 と Bayes 因子、pH の答え合わせ、Fn を除く予測 | 3200 超 |
| 5 Discussion & Conclusions | 550 | 計算力学としての意義、限界（和が 1・有効パラメータ・pH）、展望（FEM への伝播など） | 1550 |
| **合計** | **3800** | | **約 8800** |

## 4. 図 6 枚の候補

| 図 | 中身 | 元 |
|---|---|---|
| 1 | モデルと推定の流れ（5 種の模式図＋相互作用ネットワーク＋4 段の推定・GPU）| 元の TikZ ネットワーク図と Algorithm を 1 枚にまとめる（新規作図） |
| 2 | 事後予測の当てはまり（4 条件 × 5 種、組成と生存率）| `paper_fig2_phi_transposed.pdf` |
| 3 | MAP の相互作用行列（4 条件のヒートマップ）| `heatmap_A_4cond.pdf` |
| 4 | 15 成分の事後分布（4 条件）＋ a45 を強調 | `paper_posterior_violin_sharey.pdf` |
| 5 | 同定性: 広い箱 × 事前分布の有無の比較、段ごとの事後の変化（Phase 1 vs 2）| `phase1_vs_phase2_map.pdf` ＋ 同定性の図（新規） |
| 6 | 予測と答え合わせ: Fn を除いたときの Pg（ノックアウト）＋ pH の答え合わせ | 新規（2 パネル） |

外すもの: UMAP（補足へ）、計算時間の表（本文 1 文＋補足）。

## 5. 補足資料に回すもの

- 変分原理からの導出の全部、2.5 生物学的な相互作用ネットワークの表（エビデンス付き）
- 事前分布の箱と広げた成分（S1）、run の一覧と収束の判定（S2）、Bayes 因子（S3）、λ 感度（S5）
- Algorithm（TMCMC の擬似コード）、計算時間の表
- 種ごとの R²、Phase 1 と 2 の比較の数値、pairwise 比較・UMAP、条件間の転移の表
- 同定性の詳細（ident run）、DS の多峰性、pH の回帰式の選び方

## 6. 段取り

1. 今: `elsarticle` の雛形を作り、Introduction・Model・Methods を短縮して書き始める（結果に依存しない部分）
2. ult がそろったら: Results・Discussion・題名・Highlights を結果に合わせて書く
3. Meisam に短縮版を送る → MHH を含む共著者に回す → 投稿

## 7. 進み具合（2026-10-09）

`EML_manuscript_nishioka.tex`（elsarticle、通し行番号）に、結果に依存しない部分を書いた。コンパイル通る（8 ページ、エラー 0）。

| 節 | 語数（数式を除く概算） | 予算 |
|---|---|---|
| Abstract | 146（結果の 1〜2 文を足して約 180） | 150 |
| 1 Introduction | 348（主結果の 1 文を足す） | 450 |
| 2 Model | 221 | 550 |
| 3 Bayesian inference | 611 | 800 |
| 小計 | 1180 | 1800 |

→ 結果・考察・結論に **約 2600 語**使える（予算 1850 より余裕あり）。
Funding・謝辞は BMB 版と同じ文言（DFG は Nils 指定の文言、ERC Gen-TSM も）。

## 8. メモ（2026-10-09）

- **graphical abstract は図 1（`figures/fig1_pipeline.tex`、Times）から作る。** 本文の図 6 枚はすべて結果に使う。
  Elsevier の graphical abstract の規格（推奨サイズ・横長比率・文字の大きさ）を投稿前に確認して、文字を減らした版にする。
- 当てはまりの図（4 条件 × 5 種）は必須。
- 採択率は非公表。IF 約 4.6（第三者サイト）、掲載まで 0〜6 か月がほとんど。特集号の招待経由（Junker 先生）。
- 必要なら、編者（Moreno-Mateos 氏ら）に scope が合うかを Junker 先生経由で事前に問い合わせる。

## 9. 特集号の募集要項（2026-10-07 公開、要点）

- 題名: Artificial Intelligence and Computational Mechanics for Materials Design。締切 2027-04-01（それまでいつでも投稿可）。
- 投稿: Editorial Manager（https://www.editorialmanager.com/eml）で article type **"VSI: AI & CM"** を選ぶ。
- 求めるもの: 新しいデータ駆動・機械学習の手法、または材料の**逆設計（inverse design）・逆同定（inverse characterization）**へのデータ駆動手法の応用。
  「物理にもとづく導き・機構の制約・領域知識が機械学習の成功に不可欠」と明記。対象に **soft biological materials / biological materials** を含む。
- キーワード: Materials Design; Materials Discovery; Artificial Intelligence; Scientific Machine Learning; Computational Mechanics; Generative AI
- 編者: Vahidullah Tac（Stanford）、Miguel Angel Moreno-Mateos（FAU）、Mokarram Hossain（Swansea）、Yihui Zhang（Tsinghua）。テーマの適否は編者に問い合わせ可。

### 合わせ方
- 本論文は「物理（拡張 Hamilton 原理）に縛られたモデルを、データから**確率的に逆同定**する」研究 → 募集要項の inverse characterization ＋ physics-based guidance にそのまま当てはまる。
- 弱いのは「AI / 機械学習」と「設計」。対策:
  1. ベイズ推定（TMCMC）を **probabilistic / physics-constrained inference** として位置づけ、JAX（微分可能・GPU の科学計算）を scientific ML の道具として書く。
  2. 「Fn を除いたら Pg の増加が消える」予測を、**群集の組成を変える設計的な問い（in-silico design of the consortium）**として書く。
  3. 同定性の分析（どの相互作用がデータで決まるか）を、逆同定の信頼性を示す方法論の結果として前面に。
  4. DeepONet 代理モデル（1 サンプル約 80 倍）は **使わない**: 2026-02 に旧ソルバー（`improved_5species_jit`、20 パラメータ・旧モデル）で学習したもので、
     論文の順モデル（ゲートなし・15 成分）と別物。載せるなら学習し直しが必要 → 今回は見送り（展望で 1 文触れる程度）。
- キーワード案: inverse characterization; physics-constrained Bayesian inference; scientific machine learning; multiphase continuum; living materials; GPU

## 10. 進み具合（2026-10-09 夜）

- 要旨・序論・キーワードを特集号向けに直した: 「物理に縛られた逆同定（physics-constrained inverse characterization）」、
  「物理と、データで何が決まるかの確認に導かれたデータ駆動手法」、「校正したモデルを in-silico の試験台にする（1 種を除いたら？）」。
- キーワード: Inverse characterization / Physics-constrained Bayesian inference / Scientific machine learning / Extended Hamilton principle / Living materials / GPU

## 11. 進み具合（続き）

- `EML_cover_letter.tex`: 特集号の編者 4 名宛て、Junker 先生への招待に応じた投稿であること、特集との合い方を明記。題名・結果 2 文・日付・承認・DOI は `\TBD`。
- `figures/graphical_abstract.tex`: 13 × 5 cm（横長）、左「生きた材料（5 種・15 個の未知数）」→ 中「物理に縛られたベイズ推定（段階・GPU・確認）」→ 右「同定された相互作用」（結果待ち）。
  Elsevier の規格（531 × 1328 px 以上、5 × 13 cm で読める）に合わせ、提出時は `pdftoppm -r 270` で PNG/TIFF にする。
- `EML_supplementary.tex`（6 ページ、エラー 0）: BMB 版の補足に、本文から外した S1 モデルの導出・S2 文献上の相互作用（図と表）・S3 尤度の重み・Algorithm・計算時間の表を足した。
  S 番号は結果がそろってから振り直す（本文の参照も）。

## 12. タイトル（2026-10-12 決定）

**決定:** Identifiability of species interactions in a living material: Bayesian calibration of an extended Hamilton principle multiphase model
（H7 案。"physics-constrained" は PINN 型と誤解されるため不採用、モデルの出自として extended Hamilton principle を明記（IKM の枠組み）、"calibration" でモデルのパラメータであることを示す（共著者レビュー D1）。物理制約付き学習（PINN 等）と誤解されるため。本文・カバーレターの同語も "model-based" に変更）

### 以下は検討時の候補

方向は **A（同定性＝「データで何が決まるか」を前に出す）** で、ここから選ぶ。本文の `\title` はまだ旧案のまま。

| # | 案 | メモ |
|---|---|---|
| A | What can time-course data tell us about species interactions in a living material? Physics-constrained Bayesian inverse characterization of an oral biofilm | 元の A。長め |
| A' | Physics-constrained Bayesian inverse characterization of interspecies interactions in an oral biofilm | A の後半だけ（短縮版） |
| 1 | Which species interactions can be inferred from biofilm time courses? Physics-constrained Bayesian inverse characterization of a living material | 問いを具体的に |
| 2 | Inferring what the data determine: physics-constrained Bayesian identification of species interactions in an oral biofilm | 問いの形を避ける |
| 3 | Identifiable and unidentifiable species interactions in a living material: a physics-constrained Bayesian inverse characterization | **おすすめ**（問いの形を避けるならこれ） |
| 4 | From composition to interaction: what time-course data reveal about a five-species biofilm through physics-constrained Bayesian inference | 逆問題の向き。長め |
| 5 | How much of a biofilm's interaction network do time-course data determine? Physics-constrained Bayesian inverse characterization | **おすすめ**（「一部しか決まらない」という結論に一番素直） |

参考（方向が違う案）: B「GPU-accelerated Bayesian inverse characterization of a variational multiphase biofilm model: staged inference with explicit convergence and identifiability checks」（手法を前に）、C「Identifying species interactions in a living material by physics-constrained Bayesian inference」（短い）。

### 12.1 サブタイトル（案 5 をメインにする場合、2026-10-12、未決定）

メイン: How much of a biofilm's interaction network do time-course data determine?

| # | サブタイトル | メモ |
|---|---|---|
| s1 | Physics-constrained Bayesian inverse characterization | 今の案。特集号の言葉そのまま |
| s2 | Physics-constrained Bayesian inverse characterization of a five-species oral biofilm | **おすすめ**。対象まで書く（検索でも拾われやすい） |
| s3 | GPU-accelerated Bayesian inference of a variational multiphase model | 手法とモデルを前に |
| s4 | Staged Bayesian inference with explicit convergence and identifiability checks | 判定と同定性の方法論を前に |

GPU・判定の話は要旨とハイライトで伝わるので、タイトルからは外してよい。

## 13. 2026-09-25 の共著者レビュー（`LUH_summer_2026/1030_Masterarbeit/notes/paper_review_20260925_coauthors.md`）の対応状況（2026-10-12 点検）

- 済（10-12 追記: C15 交絡の一文も本文に追加。前回の点検では漏れていた）: A0（ゲートなしで全部やり直し）・A1（全 run で ln Z ≤ max ln L を確認）・A2（MAE は載せない）・A3（R² の誤った説明は無い）・A4（`Vei–Pg` など無向で表記）・B3（全段 3 seed＋判定）・B4（限界の節）・B5（正規化・時間・初期値を本文に）・C1〜C2・C3・C5（UMAP 外した）・C6（pH は推定に使っていない「check」）・C8（So–Vei は正、機構ではないと明記）・C9〜C12（該当の表・図は EML に無い）・C13（150× に統一、表記 a_ij）・D2（著者順）・**D3（西岡に MHH・NIFE を追加、2026-10-12）**・N3（尤度の重みを本文に）・N4
- 一部: B1（同じデータでの箱の絞り込み。±4σ で事後を切らないことは書いたが、広い箱との感度比較は段 (iii) が全箱に戻ることで代える）・B2（同定性は 3 seed と箱幅比で分類。profile likelihood や合成データでの再現テストは無い）
- 未: D1（タイトルが「実験で分かった相互作用」と読めないように。候補 5 の "interaction network" は要注意 → サブタイトルか本文で effective と断る）・D5（Zenodo DOI、投稿直前）

## 14. 方針変更（2026-10-12）: 焦らずいい論文にする

- 今週投稿はやめる。DS・DH の ult が判定を通ってから。Meisam には連絡済み（10/12）。
- 時間の対応は「t は接種からの時間、Day 1 の組成は最初の測定として t=0 の初期状態に使う」と書く（ずれではなく解釈として。Day 1 から始めた場合との差は最大 0.008）。
- 同定性（B2）: Fisher 情報量（MAP での曲率）で決まる方向の数を補足 S6 に追加済み（CS 4・CH 4・DS 8・DH 5 vs 事後 4・6・8・9）。次は合成データでの回収テスト（GPU、本線の後）。


## 15. GitHub 公開とデータ（2026-10-12）

- **実験データ（Heine et al. 2025）は Zenodo で再配布しない（ユーザー判断）。** `docs/revision/zenodo/MANIFEST.txt` から CSV 10 本と `ph_check_bands.npz` を外し、README §6 に入手方法を書いた。
- 原稿の Data availability は Zenodo だけを挙げる（GitHub の URL を外した、d768666）。Zenodo の README・.zenodo.json・CITATION.cff からも GitHub を外した。
- 🔴 **ただしリポジトリ `keisuke58/Tmcmc202601` は public で、`data_5species/experiment_data/`（85 ファイル）が入っている。**
- **対応（おすすめ・ユーザー了承）: 非公開（private）にする。ただし今はまだやらない。**
  - 今切り替えると止まるもの:
    1. IKMHIWI03 の `run_calibrated.ps1`（pde-fem-biofilm）が GitHub API と raw URL を**認証なし**で叩いて final_theta_MAP・phi_init を取っている → 失敗する
    2. GPU クラスタの `git pull` の認証方式が不明 → 失敗すると古いコードで走る（前例あり）
  - **切り替える時期:** DS・DH の ult が終わり、IKMHIWI03 の FEM 再計算（CH・CS・DS・DH）が済んだあと。
  - **その前に:** GPU 側と IKMHIWI03 に、認証付きで pull・取得できるか確認を頼む（handoff）。`run_calibrated.ps1` はトークン（環境変数）対応が要るかもしれない。
  - 手順: GitHub → Settings → General → Danger Zone → Change visibility → Private。MHH の了承が取れたら public に戻してもよい。
  - 消すだけでは履歴に残るので、ファイル削除ではなく非公開で対応する。

### 15.1 実験データを Zenodo に入れるかの確認（2026-10-12 決定）
- 共著者への「final numbers」再送メール（DS・DH の ult が通ったあと）に、次の一文を足して MHH の了解を取る（ユーザー判断）:
  > One question for Nils, Katharina, Meike and Szymon: may we include the experimental data used in the paper (species fractions per replicate, viability and pH from Heine et al. 2025) in the Zenodo archive, citing your article, so that the results can be reproduced from the archive alone? If you prefer not to, we would include only the summary values the inference uses (daily medians and standard deviations).
- 了解 → `docs/revision/zenodo/MANIFEST.txt` の §2 と `ph_check_bands.npz` を戻し、README §6・.zenodo.json の「not included」を直して再ビルド。原稿の Data availability も「archived at Zenodo, including the experimental data of Heine et al.」に。
- 不可 → 中央値・標準偏差だけの要約表を作って入れる（案 A）か、今のまま（案 B）。
- 根拠: Heine et al. 2025（Front. Oral Health, CC BY）の Data Availability は "will be made available by the authors, without undue reservation"。公開保存はされていない。手元の CSV はレプリケート単位で論文の図を超える。

## 16. 共著者メールは延期（2026-10-12 ユーザー判断）
- 10/13 8:00 の予約送信は**取り消す**（本人が Gmail の「予約済み」から）。下書き r-76615614630915311 は残す。
- **今週中（〜10/18）に MHH を含む共著者へ送る。** それまでに入れたいもの: DS・DH の最終段（または段 (iii) で確定の書き方）、回収テスト（12j、seed 42 だけでも）、TBD の整理。
- 送る直前に、最新の PDF 4 本（原稿・補足・カバーレター・共著者への回答）を作り直して添付する。本文は「DS・DH は暫定」の書き方を結果に合わせて直す。

## 17. 合格・不合格の意味（2026-10-13 メモ）
**合格 = 同じ設定で 3 seed 回し、次を全部通った群。** 1 つでも落ちたら不合格で、その群の数字は使わず回し直す。

| 本文 | GPU 側の番号 | 見るもの | 落ちる意味 |
|---|---|---|---|
| (a) | 0 | 保存した粒子の log L を計算し直して一致 | 結果が壊れている・取り違え |
| (b) | 1 | テンパリングが β = 1 まで到達 | 事後分布になっていない |
| (c) | 2・2b | 粒子の移動：1 段あたり 2 回以上・次元あたり 5 回以上 | 混合不足 |
| (d) | 3 | 3 seed の max log L の差 ≤ 1 | 探索不足（いい場所を見つけていない seed がある） |
| (e) | 4 | 3 seed の中央値の差 ≤ 0.5 事後 SD（同定された係数だけ） | seed ごとに違う答え |

- 別枠: **箱の端の印**（外側 5% に 20% 以上）→ その端だけ広げて回し直す。
- 同定の判定（しきい値 0.8）は §補足 S_ident・表 S_thresh。

### 状況（2026-10-13 朝）
- DS: p18k ult は (d) 1.16・(e) a23 0.75 で不合格。予備 ult（N_MUT 250、3400-3402、3090）が 10/14〜15 ごろ終わる見込み。
- DH: 仮 ult は (d) 0.14 は通ったが (e) a24 0.63 で不合格 → 使わない（ユーザー判断）。本線 p2 p20k → gate 3407 → ult（約 25 h、4090）、予備 p25k → gate 3408 → ult（約 47 h、3090）。最終は早くて 10/15〜16。
- 回収テスト: 6 本投入（3409-3412・3419・3420）。seed 42（4090）は 10/13 昼〜夕方。
- 共著者メールは今週中（§16）。DS・DH が間に合わなければ「段 (iii)、(e) 未達」の版で送るか判断。

### 17.1 「惜しい不合格」の扱い（2026-10-13 方針）
- **基準を後から緩めて合格扱いにはしない。** 基準は本文に先に書き、CS・CH にそのまま当てた。DS・DH だけ変えると「ゴールを動かした」になる。
- **落ちた所を書いたうえで使うのは可**（今の DH 段 (iii) の書き方と同じ）。
- DH 仮 ult（`DH_ult_nonarrow_w3_noph_p14k_sd4_prov`）: (d) 0.14 で合格、(e) a24 0.63・a13 0.58・a34 0.55（基準 0.5）→ **ぎりぎりアウト**。中身は 3 seed でそろう（RMSE 0.082–0.085、a35 中央値 −7.5/−6.2/−7.4、a45 は全 seed で 90% 区間が 0 をまたぐ）。
- DS p18k ult: (d) 1.16・(e) a23 0.75・a44 が箱の端で切られる → ぎりぎりではない。使わない。
- **代わりの手**:
  - DS: 段 (iv) が間に合わなければ、判定を全部通った**段 (iii)** を使う（今の本文と同じ）。
  - DH: 本線 ult（10/15〜16）を待つ。落ちたら、仮 ult か段 (iii) を「(a)–(d) は満たす、(e) は a24 0.63（基準 0.5）、ただし RMSE・a35・a45・同定される成分は 3 seed で一致」と書いて使う。仮 ult を使うなら GPU 側に `final_runs/` へ `_prov` で上げてもらう。

## 18. ψ_i と 2 通路の組成（2026-10-13）
- **データから種ごとの生存率が出せる**: ψ_i^data =（生きた菌の組成 PMA ÷ 全菌の組成 Fig. S1）× 群集の生きた割合（SYTO9/PI）。`tools/psi_from_data.py` → 補足 `sec:S_psi`・表 `tab:S_psi`。
  - データの ψ_i は多くが群集の生きた割合（0.3〜0.9）の近く。例外は commensal の Fn・Pg（0.04〜0.10、ほぼ死んでいる、全菌でも数 %）。
  - モデルの ψ_i（MAP）は 0.03〜0.96 の両極端で、CS 以外は合わない（CH So 0.26 vs 0.42、DS An 0.07 vs 0.57、DH Pg 0.07 vs 0.29 など）。尤度が群集の生きた割合しか見ていないため。
  - 図 2 の破線（種ごとの φψ）は外した（d35b57b）。本文の仮定の文と限界の節に一句ずつ（98980fd）。
- **2 通路の尤度の案**: 生きた菌の組成 ↔ φψ/Σφψ、全菌の組成 ↔ φ/Σφ、生きた割合はそのまま。ψ_i が種ごとに制約され、モデルとして筋が通る。
  - コスト: 4 条件 × 段 (i)〜(iv) のやり直し＋判定、GPU で 1〜2 週間以上。図・数値もほぼ全部作り直し。
  - **ユーザーの考え: 精度が上がるなら待つべき。** → まず CPU のパイロット（`tools/two_channel_pilot.py`、今の MAP から 2 通路の二乗誤差の和を L-BFGS-B で最小化、4 条件）で、両方の組成と生きた割合を同時に合わせられるかを確かめてから決める。
  - 判断の目安: 最適化後も各誤差が今と同程度 → 実現できる、待つ価値あり（Meisam・Felix に相談）。どれかが大きく悪化 → 構造上無理、今の版で出し、2 通路は修論・改訂で。
  - CH の今の MAP: 生きた菌↔φ 0.070、全菌↔φ 0.055、生きた菌↔φψ 0.144、生きた割合 0.113。
- **前にやった「2 通路」との違い（2026-10-13 確認）**: estimator には ch2（全菌の組成）が 3 月からある（f47a778）。ただし `make_log_likelihood_jax_ode` では **ch1（生きた菌）も ch2（全菌）も同じ φ/Σφ と比べている**（docstring は「Ch1: φ·ψ」と書くが、コードは fix-ψ のときだけ一様な ψ を掛ける）。3 月の Phase 1 で ch2 を入れて「衝突した」ので重み 0 にした（`run_phase2_free_psi.sh` 14 行目）のは、同じ量に少し違う 2 つの目標を課したため。**生きた菌 ↔ φψ、全菌 ↔ φ の形は試していない。** 論文・修論の run はすべて λ2 = 0。

## 19. 3 つの版の違い（2026-10-13 メモ）

| | 前の版（修論・3〜4 月） | 今の版（EML・結果あり） | 新しい案（2 通路） |
|---|---|---|---|
| 段 | Phase 1（ψ 固定）→ Phase 2（ψ 自由）、1 seed | 同じ 2 段を 4 段に分け、**各段 3 seed で判定 (a)〜(e)** | 4 段と判定はそのまま |
| Phase 1 の ψ | 全種同じ（群集の生きた割合） | 同じ | **種ごと**（データから出した ψ_i、`psi_from_data.py`） |
| 生きた菌の組成（PMA）の比べる相手 | φ | φ | **φψ** |
| 全菌の組成（Fig. S1） | 使わない | 使わない | **φ と比べる** |
| 群集の生きた割合 | 使う | 使う | 使う |
| pH | HOBIC で尤度に入っていた可能性が高い（重み 0.3） | 入れない（答え合わせだけ） | 入れない |
| 設定・記録 | config・ログなし | 全部あり（Zenodo） | 残す |
| 収束の確認 | なし | seed 間の一致まで | 同じ |
| 箱 | 事後 ±2σ に絞る | ±4σ、端にたまれば広げる | 同じ |

- **前の版の問題**: 収束未確認（1 seed）、再現できない、pH を入れて「予測」と言っていた可能性、±2σ で事後を切っていた → 今の版で直した。
- **今の版の残る問題**: 生きた菌のデータを φ と比べる（種によらない生存率の仮定、commensal の Fn・Pg で崩れる）、全菌のデータを使わない → ψ_i が決まらず 0/1 に張り付く（図 2 の破線を外した理由）。ほかに、尤度の重み（λ_Pg 5、λ_late 3、λψ が条件ごとに 1.5/2.0/3.0）は査読で聞かれうる。
- **新しい案で期待すること**: ψ_i が種ごとに決まり、生きた量と全体の量の両方がデータと合う。同定される成分が増える可能性。ただし構造上両方は合わない可能性もある → CPU パイロット（`tools/two_channel_pilot.py`、§18）で確認中。

### 18.1 2 通路のパイロットの結果（2026-10-13、CPU、MAP から L-BFGS-B 80 回・1 回だけ、収束前）
RMSE（Day 3〜21、重みなし）。`docs/revision/generated/two_channel_pilot/`。

| | 今の MAP: 生きた菌↔φ | 生きた菌↔φψ | 全菌↔φ | 生きた割合 | → 2 通路の最適: 生きた菌↔φψ | 全菌↔φ | 生きた割合 |
|---|---|---|---|---|---|---|---|
| CS | 0.161 | 0.122 | 0.115 | 0.076 | **0.108** | **0.104** | **0.028** |
| CH | 0.070 | 0.144 | 0.055 | 0.113 | **0.064** | 0.083 | **0.047** |
| DS | 0.029 | 0.114 | 0.038 | 0.061 | 0.047 | 0.052 | **0.034** |
| DH | 0.076 | 0.091 | 0.038 | 0.108 | **0.067** | 0.058 | **0.049** |

- **2 通路で同時に合わせられる。** 生きた菌の組成は今の φ の当てはまりと同程度かそれ以上（DS だけ少し悪い）、全菌も 0.05〜0.10、群集の生きた割合は 4 条件とも半分以下。収束前なので改善の下限。
- ψ（日の中央値）: CS 0.92/0.90/0.46/0.06/0.04（データ 0.81/0.69/0.64/0.06/0.10 に近い）、CH 0.35/0.66/0.62/0.14/0.12（データ 0.42/0.37/0.39/0.04/0.05、前より近い）。DS・DH はまだ離れる。
- **重大: CS の「組成と生存率は同時に合わない」（本文の結論の 3 点目・限界の節・補足 S_ceiling）は、生きた菌のデータを φ と比べていたことによる見かけの可能性が高い。** 2 通路では CS で生きた菌 0.108 と生きた割合 0.028 が同時に出る。

## 20. 決定: 2 通路で回し直してから投稿（2026-10-13）
- ユーザー決定: **案 1**（2 通路で推定し直してから投稿）。GPU の古い尤度のジョブは**全部止める**。Meisam には**まだ伝えない**（CH の最初の結果が出てから）。
- 手直し: 2 通路では ch2（全菌）にも ch1 と同じ重み（λ_Pg 5・終盤 ×3・珍しい種 ×0.1）。`test_two_channel.py`・`test_paper_pipeline.py` とも OK。gate にも `TWO_CHANNEL=1`。
- GPU 側への指示: handoff `2026-10-13b_two_channel_go.md`（止めるジョブ、4 条件の段ごとの粒子数・箱、CH → CS → DS・DH の順）。
- 待つ間: 本文の書き換えの下書き（尤度の式、全菌のデータ、CS の不一致の主張を外す、S_psi を結果に、図 2 に φψ を戻すか）。回収テストは 2 通路の CH・CS ult が通ってから同じ尤度で。
- 見込み: CH・CS は 2〜3 日、DS・DH は 1〜2 週間（判定のやり直し次第）。共著者メールはそのあと。

### 20.1 「2 通路よりよく当てる」の検討（2026-10-13、GPU 側メモ）
- ユーザー指示: 改善案（2 通路）よりよく当てはまる実装を。クラウドにも相談（handoff `gpu_2026-10-13_fit_improve.md`）。
- §18.1 のパイロットは 80 反復・1 回で収束前。DS・DH の ψ_i はデータから離れたまま。
- `tools/two_channel_pilot.py` に `--psi-data W`（ψ_i を ψ_i^data に寄せる項）と `--seed` を追加。既定は変えない。
- CPU で 4 条件 × W∈{0,1}、再スタート 2・250 反復を掃引中 → `docs/revision/generated/two_channel_pilot/improve/`。
- 判断の目安: W=0 で収束しても ψ が離れるなら ch4（`LAMBDA_PSI_DATA`）を estimator に足して CH の p2 から 1 系列試す。
  W=0 で近づくなら 2 通路のまま（二重数えを避ける）。2 通路の GPU ジョブは止めない。
- **20.2（2026-10-13c、GPU 側の相談への返事）**: ψ^data を尤度に入れない（ch1÷ch2×ch3 から作るので二重計上。2 通路でも合わなければ構造上の結果として書く、ψ^data は確認用）。パイロットは推定と同じ尤度（重み・σ）と p2 の実際の箱で回し直す。**日・種ごとの σ（反復の q25/q75）は改訂・修論の候補**（今回は尤度を変えない）。
- **20.3 判定の見直し（2026-10-13、ユーザー決定「案 1」、2 通路の結果を見る前）**: (d) は各 seed の上位粒子を L-BFGS-B で磨いた max logL の幅 ≤ 1（粒子の最良値は粒子数と偶然で揺れる。DS ult p5k は磨くと幅 0.13）。(e) は 0.5 SD のまま、R̂ を並べる。4 条件すべてに同じく、今の尤度の結果には遡らない。本文の (d) の定義を変更。handoff 13e。
- **20.4 なぜ DS・DH は遅いか（2026-10-13 メモ）**: 前進計算の重さではない（1 粒子 × 1 mutation × 1 段あたり 22〜24 µs で 4 条件同じ、pilot）。
  1. テンパリングの段数が約 2 倍（pilot: CH 5・CS 6 段 / DS・DH 10 段、2.5〜2.9 h / 5.1〜5.3 h）。Fn・Pg が 10〜27% で情報が多く尤度の山が鋭いので、β の刻みが細かくなる。
  2. 必要な粒子数が 3〜10 倍（p2: CH 2000 / DS 18000 / DH 14000〜20000）。事後が複雑（DS の a33 の 2 つの山、Vei–Pg の符号、細く曲がった形）で、少ないと seed が別の山・場所に行って (d)(e) に落ちる。GPU の並列で 9 倍の粒子でも時間は約 2.4 倍（CH p2 7.4 h / DS p2 17.6 h）。
  3. 今の尤度で回し直しが多かった。
  2 通路では、ψ のあいまいさが減って事後が素直になる（粒子・回し直しが減る）可能性と、データが増えて段数が増える可能性の両方がある。CH の結果が最初の手がかり。
- **20.5 (e) は 0.5 SD のまま（2026-10-13 ユーザー決定）**: さらに緩める案（0.75 SD ≈ R̂ 1.07、R̂ < 1.1 ≈ 0.9 SD）は採らない。理由: 直前に (d) を磨いた値に変えたばかりで、続けて緩めると「通るまで基準を下げた」に見える。0.5 SD は R̂ ≈ 1.03 で、推奨（1.01）と慣習（1.1）の中間。ぎりぎりで落ちたら §17.1 のとおり、落ちた所と R̂ を明記して使う。

## 21. 組成データは図からの読み取り（2026-10-13 判明）
- `fig3_species_distribution_replicates.csv`（8 個 × 時点）は著者提供ではなく、Fig. 3A の箱ひげ図の読み取り値（中央値・四分位・ひげ）から `extract_and_plot_fig3.py` の `generate_replicates`（seed 42）で作った人工の 8 点。回し直すと 960 行が完全一致。Fig. S1（全菌）も 10/09 に図から抽出。
- 本文・補足の「著者提供の反復値（8 個）」を「箱ひげ図からの読み取り（反復 9）、反復値そのものは入手していない」に訂正。σ_i は読み取った箱の広がりから（最小 0.05）。
- 5 月のデータ依頼（Szymon → Nils・Doll-Nikutta）は **FISH-CLSM の画像**で、共有は了承済み（ファイルは MHH 内部の N: ドライブ）。**qPCR の反復値は頼んでいない** → Nils 宛て（Szymon CC）の依頼の下書きを Gmail に作った（r8592691890676242677、送るのは本人）。届いたら最終の run はそれで。
