# GPU 側 → クラウド側（2026-10-08）: DS p1 投げ直し済み / 事前体積の交絡で ln Z の解釈に問題 / 指示をあおぎたいこと 3 件

`2026-10-08h.md` の指示は実行済み。あわせて査読用の Bayes 因子を用意する過程で、
**原稿の ln Z の解釈が事前分布の体積と交絡している**ことが分かった。§3 が本題。

---

## 1. 2026-10-08h の指示: DS p1 を投げ直した（完了）

先置きの DS p1（3214-3216、2000 粒子・PREV が seed ごと）を qdel し、指示どおり投げ直した。

| Job | SEED | GPU | 状態 |
|---|---|---|---|
| 3246 | 42 | vancouver01-gpu/2 (RTX4090) | R |
| 3253 | 7 | vancouver03-gpu/0 (RTX4090) | R |
| 3254 | 123 | vancouver03-gpu/1 (RTX4090) | R |
| 3255 | — | 通知（`afterany:3246:3253:3254`, `RUN_GLOB=DS_p1_wide80_p3k_*`） | H |

```
STAGE=p1 TAG=DS N_PART=3000 N_MUT=100 RUNTAG=wide80_p3k
PREV=_runs/paper_gateoff/DS_pilot_wide80_seed123   （3 seed とも指示どおり同一）
OVERRIDE=5:-15:20;6:-15:20;19:-15:20
walltime 22:00:00
```

**3 本とも RTX4090 に乗った。** `vancouver03`（RTX4090 x2）が PBS に登録されているのに
`claude.md` のノード構成に書かれておらず、丸ごと空いていた。NFS ホーム・`klempt_fem2` の
JAX 0.6.2・CUDA を確認してから投入した。ノード一覧に追記済み。

`gpufree` が `nvidia-smi` のメモリしか見ないため、**他ユーザーの放置 jupyterlab が PBS 上で
確保しているだけの GPU が「空き」に見える**（4090 を 4 枚、走行 280〜1200 時間・walltime 無制限）。
PBS の実割り当てと実プロセスを突き合わせる `gpualloc` を作って `claude.md` に書いた。
なお Torque は GPU を 1 枚単位で排他確保するので、`qsub -l gpus=1` で投げている限り
他ユーザーと同じ GPU に相乗りすることはない（空きが無ければ Q で止まるだけ）。

### a33 の下の山の重み（指示どおり報告する）

`tools/a33_mode_weight.py` を追加した。pilot（= 今回の比較対象）はこうなっている:

| run | n | **w(a33 < −5)** | 中央値 | 5% | 95% | maxlogL |
|---|---|---|---|---|---|---|
| DS_pilot_wide80_seed42 | 1000 | **0.657** | −8.10 | −14.22 | 1.71 | −1.63 |
| DS_pilot_wide80_seed7 | 1000 | **0.443** | −0.07 | −13.61 | 1.95 | −1.49 |
| DS_pilot_wide80_seed123 | 1000 | **0.139** | 0.87 | −11.37 | 1.80 | −0.91 |

**pilot の重みは 0.14〜0.66（幅 0.52）** で、「山への配分が seed ごとに違う」という診断と一致。
p1（3000 粒子）が終わったら同じ表を出し、幅が ±0.10 以内に収まるかで判定する。

---

## 2. ジョブの棚卸しとパイプラインテスト

走行 21 本は全部必要な群で、削るものは無い（DS p1 3、p2 CH/CS/DH 9、a45=0 pilot 9）。
査読用の a45=0 は既存の完了分と合わせて **4 条件 × 3 seed がちょうど揃う**。
`tools/test_paper_pipeline.py` は全項目 OK。

---

## 3. 本題: 事前体積の交絡で ln Z の条件間比較が成り立たない

査読用に `tools/bayes_factor_a45.py` を書いたとき、比較の前提を照合する過程で見つけた。

### 何が起きているか

原稿が自分自身と矛盾している。

- §Constrained prior: 「the bounds $[l_k,u_k]$ are **condition-specific** and iteratively narrowed」
- §Model evidence: 「ln Z is higher for commensal conditions **despite the same 15-parameter prior**,
  because the posterior concentrates pathogen entries near zero, effectively reducing the Occam factor.
  This evidence contrast provides a model-based signature of ecological complexity.」

一様事前分布なので ln Z には $-\sum_k \ln(u_k-l_k)$ が加算で入る。実測（ゲート OFF pilot, seed42）:

| 条件 | 自由次元 | **ln(事前体積)** | a45 の箱（幅） |
|---|---|---|---|
| CS | 15 | **12.92** | [−1, 1]（2） |
| CH | 15 | **14.20** | [−1, 2]（3） |
| DH | 15 | **21.12** | [−0.5, 6.0]（6.5） |
| DS | 15 | **24.47** | [−15, 20]（35） |

**CS–DS の ln Z 差のうち 11.6 nat は事前体積だけで説明がつき、差そのものと同程度。**
DS の事前体積は CS の e^11.6 ≈ 10^5 倍。つまり「commensal の方が ln Z が高い」のは
主に箱が狭いからで、Occam factor / 生態学的複雑さという説明は支持できない。

DS の箱が広いのは 2026-10-07d で a33 の事後が論文の箱から外れていたのを直した結果なので、
この交絡は**ゲート OFF の新しい run 系列で初めて効く**。

### 原稿に入れた変更（コミット `9ab1009`、latexmk で PDF 通過）

ユーザー判断は「(b) 原稿に箱の幅を書く」（条件間比較を避ける）。それに沿って:

1. §Constrained prior に条件別の a45 の箱と ln 事前体積を明記（`sec:prior` ラベル追加）
2. §TMCMC に「ln Z は $-\sum\ln(u-l)$ を含む」「比較は同一の箱の入れ子モデル内に限る」を追記
3. §Model evidence から生態学的結論を取り下げ、交絡を 11.6 nat と定量して述べる
4. a45=0 の入れ子比較（ln B）の読み方を追記。**DS は箱が広いので ln(35/2) ≈ 2.9 nat
   不利**になることを明記。数値表は全 seed 完了後に挿入（`% TODO` 済み）
5. Effective dimensionality の $r_i$ を $\Delta_{\text{prior}}=6$ 固定から各パラメータの箱幅に変更

### Bayes 因子ツール（`tools/bayes_factor_a45.py`）

ln B = ln Z(a45 自由) − ln Z(a45 = 0)。**前提が崩れていたら数字を出さずに止める**作りにした
（データ・尤度・a45 以外の事前分布の一致、`free_dims` の差が a45 だけか、`beta_final=1`、
logL と粒子の整合）。a45 の箱の幅も必ず併記する。

作る途中で自分のバグを 1 つ踏んだので記録しておく: 最初 `run_record.json` を読んでいたが、
このファイルには `condition` / `n_hill` / `K_hill` / `n_mutation_steps` / `n_particles` が
入っておらず、**照合 10 項目のうち 7 項目が黙って素通り**していた。`config.json`（上位集合）に
切り替え、鍵が無い場合も「照合できない」で止めるようにした。

現時点で出ている分（seed が揃っていないので解釈はしない）:

| 条件 | seed | ln Z 自由 | ln Z a45=0 | ln B |
|---|---|---|---|---|
| CH | 42 | −9.744 | −9.682 | **−0.062** |
| CS | 42 | −15.040 | −15.238 | **+0.198** |
| CS | 123 | −15.217 | −15.326 | **+0.109** |

いずれも Kass–Raftery で「差とは言えない」域。**Commensal では a45 を入れても証拠が改善しない**
という論文の筋（a45 は DH だけ正）と整合する。

なお **CH / CS は mutation 回数が非対称**（ベースライン 120 対 a45=0 側 80）。モデルの違いでは
ないので止めていないが、ln Z の精度に効くのでツールは警告を出す。DH・DS は 80 対 80 で対称。

---

## 4. 指示をあおぎたいこと

### (1) `r_i` の数値をどう差し替えるか ← 一番効く

分母を各パラメータの箱幅にすると、実箱での最大は **0.296** なので
「All 15 parameters satisfy $r_i < 0.43$」は生き残る。しかし条件の順位が変わる:

| 条件 | mean r（Δ=6 固定） | **mean r（実箱）** |
|---|---|---|
| DH | 0.136 | **0.190** |
| DS | 0.221 | **0.194** |
| CH | 0.106 | **0.249** |
| CS | 0.089 | **0.233** |

**「DS achieving the tightest posteriors (mean r=0.07)」は使えなくなる**（実箱では 4 条件ほぼ横並び）。
原稿の 0.07 はゲート ON の旧 run 由来で私の数値はゲート OFF pilot なので、**勝手な差し替えはしていない**
（`% TODO` でマークだけした）。順位を書かず「全条件で $r_i < 0.3$」とまとめる案でよいか、
それとも別の書き方にするか。

### (2) CH / CS の a45=0 を mutation 120 で回し直すか

ln B の非対称（120 対 80）を消すなら pilot 3 本 ×2 条件 = 6 本。
`ln B` がどちらも |0.2| 未満なので結論は動かないと思うが、査読で「サンプラー設定が違う」と
言われうる。回すなら枠は空いている。

### (3) DS の ln B を論文の箱でも 1 組取るか

(b) の方針では不要（箱の幅を明記して条件間比較を避ける）。念のため確認。
DS だけ ln(35/2) ≈ 2.9 nat 不利な数字が出ることになる。

---

## 5. 状態

- コミット: `5165a76`（gpualloc）、`3189974`（ツール 2 本）、`9ab1009`（原稿）を push 済み
- DS p1（3246 / 3253 / 3254）走行中、終了したら判定 0〜4 と a33 の重みを報告する
- 通知ジョブ 3255 が afterany で待機
