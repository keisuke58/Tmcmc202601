# DH gate-off a35 prior-check 調査ログ (2026-09-30)

Dysbiotic HOBIC, gate-off (`--K-hill 0.0 --n-hill 2.0`) で
**a35 (`theta[18]`, Vei→Pg) が prior 無しだと箱の端に張り付く（非同定）のか**を確かめた記録。
指示書 2026-09-29「copaam 側の Claude への指示書」に対応する。

- スクリプト: `data_5species/main/dh_prior_check_job.sh`
- 本体: `data_5species/main/estimate_reduced_nishioka_jax.py`
- 共通設定: `--box -15 20`, `--dt 1e-4 --n-steps 2500`, `--mutation rw --device gpu`,
  b (`theta[3,4,8,9,15]`) は 0 固定で探索次元 15
- `a35 = theta[18]` は `estimate_reduced_nishioka.py` の `param_names`（a11,a12,a22,b1,b2,
  a33,a34,a44,b3,b4,a13,a14,a23,a24,a55,b5,a15,a25,**a35**,a45）で確認済み

## 1. 最初の run (3012 / 3013) は walltime で全損した

50 粒子、walltime 2h。**TMCMC 本体は 115〜119 秒で正常終了**していたが、その後の
MAP polishing が終わらず 7231s で kill、結果は 1 ファイルも保存されなかった。

### 原因: polish の FD 勾配コスト見積もりが 200 倍以上外れていた

`estimate_reduced_nishioka_jax.py:486-488` のコメントは「1 eval ≈ 5-10ms」を前提にしていたが、

- 20 次元 × `jac="3-point"` = **勾配 1 回あたり 40 eval**、`maxiter=50` × 2 start で 4000-5000 eval
- 1 eval は `n_steps=2500` の逐次 `lax.scan`。GPU ではカーネル起動レイテンシ律速で、
  **単一 θ の forward は 50 粒子を vmap した場合と同じ壁時計時間かかる**
  （TMCMC が速いのはコストを粒子間で償却しているから）
- 実測は取っていないが、walltime から逆算して **1 eval ≳ 1.4 秒**

### 併発している問題（未修正）

1. **保存が polish の後ろにある**（`:516-526`）。kill されると正常終了した TMCMC まで全損する。
   → samples/logL/config は polish の前に一度書くべき。
2. **polish が固定次元も動かしている**。b は 0 固定なのに `bounds_list` は 20 次元全部で、
   FD が 10 eval/勾配 無駄に増える。自由 15 次元だけにすべき。
3. **polish が事前分布を無視している**。`_neg_logL` は `log_likelihood` のみ。
   加えて engine 側の `theta_MAP` も `argmax(logL)`（`tmcmc_nuts_engine.py:372`）。
   → **sigma6 run の「MAP」は事後最大ではなく最尤点**で、今回の比較目的に合っていない。

暫定対処として `--no-polish` を追加し walltime を短縮した（3014/3015 以降）。

## 2. 50 粒子 run (3014 / 3015) の結果: a35 は端に張り付かない

出力: `_runs/dh_gateoff_{noprior,sigma6}_20260930/`

| | noprior (3014) | sigma6 (3015) |
|---|---|---|
| a35 MAP | +2.051 | +0.001 |
| a35 mean ± sd | +1.94 ± 0.31 | +0.03 ± 0.02 |
| a35 min / max | +1.43 / +2.39 | −0.02 / +0.08 |
| 箱の端 5% 以内の粒子 | 0% | 0% |
| max logL | −91.0 | −111.2 |
| stages / time | 10 / 114.6s | 9 / 116.8s |

- **prior 無しでも a35 は +2 付近にまとまり、端（−15 / +20）から 13 以上離れている。**
  自由 15 成分のどれも端 5% 以内に 1 粒子も無い（最も端寄りは noprior の a23 = +14.7）。
- → 当初の「prior が無いと a35 が piling する」という仮説は、この設定では支持されない。

### ただし信用しきれない点

- **sigma6 の事後が異常に狭い**。自由 15 次元の粒子間距離は平均 0.076、次元ごとの sd は
  0.003-0.036。50 粒子が実質 1 点に潰れている（degeneracy ではなく unique 50/50 だが早期収束）。
- **2 つの run が別のモードにいる**。sigma6 は logL が 20 悪く、a35 以外も全く別の場所
  （a13: +8.7→+0.9、a12: −8.4→+1.3、a23: +14.7→−2.5）。
- **20 nats の差は prior では説明できない**。N(0, 6²) は箱 [−15, 20] に対しほぼ無情報で、
  noprior の平均ベクトルに対する log-prior ペナルティを足しても 6 nats 程度。
  **sigma6 側が収束していない疑いが強い。**
- 15 次元に対し 50 粒子は少なすぎる。

## 3. 1000 粒子 × seed 42/7/123 の再実行 (3016-3021)

出力: `_runs/dh_gateoff_{noprior,sigma6}_1000p_seed{42,7,123}_20260930/`
3 ノード（stuttgart01/02/03）に 2 本ずつ分散、GPU は PBS_GPUFILE 経由で全 6 本別デバイス
（`exec_gpus` と `nvidia-smi --query-compute-apps` で競合ゼロを確認済み）。

全 6 本とも正常終了（172-204s、9-11 stages、accept 0.39-0.54）。

### a35 (`theta[18]`) の事後（箱 `[-15, 20]`）

| run | a35 MAP | mean ± sd | 95% 区間 | 端 5% 以内 | max logL |
|---|---|---|---|---|---|
| noprior seed42 | −0.96 | +3.82 ± 4.09 | [−1.00, +17.49] | 2.1% | −79.1 |
| noprior seed7 | **−14.11** | −13.82 ± 0.82 | [−14.93, −11.71] | **78.5%** | −85.2 |
| noprior seed123 | −2.88 | −4.19 ± 0.87 | [−6.22, −2.64] | 0.0% | −78.8 |
| sigma6 seed42 | −3.57 | −0.03 ± 3.59 | [−4.41, +9.05] | 0.0% | −74.0 |
| sigma6 seed7 | +1.74 | −0.14 ± 1.70 | [−2.73, +2.30] | 0.1% | −69.5 |
| sigma6 seed123 | +2.07 | +1.84 ± 0.43 | [+0.94, +2.63] | 0.0% | −68.5 |

### 結論が 50 粒子のときから逆転した

1. **prior 無しでは a35 は実際に箱の端に張り付く。** seed7 では粒子の **78.5%** が下端
   (−15) の 5% 以内、MAP も −14.11 で端そのもの。seed42 では逆に 95% 区間が
   [−1.0, +17.5] と箱の半分に広がる。**MAP は seed で −0.96 / −14.11 / −2.88 と全く定まらない。**
   → 50 粒子 run の「端に張り付かない」は単なるサンプル不足のアーティファクトだった。
2. **非同定は a35 だけの話ではない。** noprior では他成分の方が激しく端に溜まる
   （seed123: a34 95.9%, a11 92.1%; seed7: a33 90.8%, a11 85.1%）。
   gate-off + 箱 [−15, 20] という設定自体が全体として under-determined。
3. **弱情報 prior は効いている。** sigma6 では端 5% 以内が最大でも 3.7%（a25）で、
   a35 は全 seed で 0.1% 以下。しかも **max logL が noprior より良い**
   (−68.5〜−74.0 vs −78.8〜−85.2)。prior は情報を足すだけでなく、探索を
   まともな領域に保ってサンプラーの収束を助けている。
4. **50 粒子 sigma6 の「logL が 20 悪い」は収束不全だった**（当時の疑い通り）。
   1000 粒子では順位が逆転している。
5. **ただし sigma6 も seed 依存は残る。** a35 の mean は −0.03 / −0.14 / +1.84、
   sd は 3.59 / 1.70 / 0.43 とばらつき、max logL も 5 nats 幅がある。
   **多峰でまだ収束しきっていない。** 1000 粒子 / max-stages 30 でも足りない。

## 4. 今後に向けて

### 現時点で言えること

- **指示書の仮説は支持された**: prior 無しの a35 は箱の端に張り付き、MAP は seed 依存で
  全く定まらない。弱情報 prior N(0, 6²) はこれを解消する。
- **ただし a35 固有の問題ではない**。noprior では a33 / a11 / a34 の方が激しく端に溜まる。
  論文で「a35 に prior を置く」と書くなら、なぜ a35 だけなのかの説明が要る。
  実際には A 全体に置いており（`--prior-scale` は自由 15 成分すべてに掛かる）、
  記述はそれに合わせるべき。
- **まだ答えられない**: sigma6 でも seed 間で a35 の事後が一致していない（mean −0.03 〜 +1.84）。
  「prior 下での a35 の値」を論文に載せるには収束が足りない。
- 点推定の比較には上記 1 章の MAP 定義の問題も残っている（`theta_MAP` = `argmax(logL)`）。

### 次にやること（優先順）

1. **収束を詰める**: 粒子 5000、`--max-stages` 増、`--n-mutation-steps` 増で seed 間一致を確認。
   1000 粒子が 3 分だったので 5000 でも 15-20 分程度の見込み（未実測）。
2. **a35 のプロファイル尤度**: a35 を格子上（例 −15 … +20 を 36 点）に固定して残り 14 次元を
   最適化し logL の曲率を見る。箱にも prior にも依存しない非同定の直接証拠になる。
   既存スクリプトで回せるかは未確認。noprior で平坦なら図として論文に使える。
3. **polish を事後最大に直す**: `logL + log_prior`、自由 15 次元のみ、時間予算で打ち切り。
   FD をやめて `jax.grad` を使う案もあるが、2500 ステップ scan の reverse-mode の
   コンパイル時間は未実測。polish だけ CPU で回す方が速い可能性もある。
4. **保存を polish の前に移す**（1 章の問題 1）。

### GPU 運用の教訓

- **GPU で単一 θ の scan を回す最適化ループは作らない**。vmap で償却できない形は
  GPU の利点が消え、CPU より遅くなりうる。
- 長い後処理を挟むなら、**主結果は後処理の前に保存する**。
- PBS の walltime 超過は無警告で全損するので、未実測の処理には広めの walltime を取る。

## 5. データの出所（2026-09-30 追記）

**論文の図表と論文 MAP は `fig3_species_distribution_summary.csv` 側で作られている。**
`species_distribution_data.csv`（色キー）は 2月時点のレガシーで、DH では正規化後で
最大 0.30 ずれ、実測 Pg の Day21/Day15 が 7.76（fig3 は 2.74）になる。

根拠:

1. loader の fig3 優先は `d4bbd07`(2026-03-14) で入った。コミット済みの
   `_runs/Dysbiotic_HOBIC_K0.05_n4.0_1k30/data.npy` は `9c7fdca`(2026-02-27) で、
   それ以前なのでレガシー側になるのは必然。
2. 論文 MAP の `ultimate_10000p` は 2026-04-19 実行。`run_ultimate_10000p.sh:79-93` は
   `--external-data` を使わず `--condition/--cultivation` で loader を通すので fig3 側。
3. 論文図の生成スクリプトも fig3 を直読み
   (`docs/regenerate_all_figures.py:70`, `data_5species/main/plot_paper_fig2.py:55`)。

したがって **estimator の既定 loader で回している GPU run は、論文と同じデータ**。
`_extdata/*_legacy.json` はレガシー側での再現用であり、論文との比較には使わない。

論文 MAP はさらに `--multichannel --lambda-ch1 1.0 --lambda-ch3 2.0`（DH は
`--lambda-ch5 0.3`）、`--posterior-prior-nsigma 2`、Phase 2 からの warm start で
得られている。単一チャネルの RMSE で論文値 0.087 と一致しないのはこのため。
実行ログは `.gitignore` の `*.log` で残っていない。
