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
  4. 任意: リポジトリの DeepONet 代理モデル（1 サンプル約 80 倍）を 1 段落で触れるか、補足に入れる（機械学習の要素を足せる。ただし語数と相談）。
- キーワード案: inverse characterization; physics-constrained Bayesian inference; scientific machine learning; multiphase continuum; living materials; GPU
