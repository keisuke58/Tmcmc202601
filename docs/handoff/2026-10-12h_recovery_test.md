# クラウド側 → GPU 側（2026-10-12h）: 合成データによる回復テスト（共著者レビュー B2「同定性が示されていない」への対応）

**本線（DS ult・DH p2/ult、12g の予備）は止めない。こちらは枠が空いたら。** CH・CS は最終なので先に、DS・DH は ult が判定を通ってから。

## 1. 何をするか

補足資料（supplement）S4 の同定性の分類（同定 / 符号のみ / n.i.、sd 比 < 0.8 の規則）が「データと尤度の性質」であって
サンプラーの癖ではないことを、**真値が分かっている合成データ**で確かめる。

- 真値 θ_true = 最終 run 群のうち max logL の seed の MAP（CH は `CH_ult_mut150_wide2_noph_sd4_seed7`、CS は `CS_ult_wide2_sd4_seed42`）
- 実データと**同じ**前進モデル・観測演算子（観測日への写し、clip と正規化）・σ_obs（種ごと）・σ_viab・初期値（Day 1）・尤度の重み（λPg 5、λlate 3、λrare 0.1、λch3 は自動調整後 1.5）で合成データを作り、
  種を固定した Gaussian ノイズを足す（組成は 0 で切って行和 1 に正規化し直す。実データと同じく「割合」が観測）
- **estimate_paper_jax.py は変更していない。** `tools/recovery_run.py` が estimator を import し、データローダー 2 つ
  （`load_experimental_data`・`load_multichannel_data`）の戻り値の観測値だけを合成データに差し替えて `main()` を呼ぶ（σ・日・初期値は本物の経路のまま）
- 設定は実データの ult と同じ: 箱 = 元 run の `prior_bounds_final`、粒子 5000・N_MUT 150・DE-MC・throttle 1.0
- **warm start はしない**（箱の一様分布から TMCMC）。実データの ult は p2 の事後から半分を初期化したが、回復テストでそれをやると真値の近くから始めることになり検証にならない
- 見るもの: (a) 実データで「同定」の成分が合成でも同定され、90% 区間に真値を含み、誤差が小さいか　(b) 実データで「n.i.」の成分が合成でも n.i.（事後 ≈ 箱）か　(c) 全 15 成分の被覆率

## 2. 新しいファイル（既存のファイルは触っていない）

| ファイル | 役割 |
|---|---|
| `tools/recovery_common.py` | 共通部品（最終 run の読み込み、尤度の組み直し＝fisher_ident.py と同じ、ローダーの差し替え、分類の規則） |
| `tools/recovery_make_synthetic.py` | 合成データ（JSON 1 つ）を作り、θ_true での log L を検算 |
| `tools/recovery_run.py` | estimator を合成データで回す（`--dry-run` で尤度の照合だけ、`--tiny` で 20 粒子 1 段の通し確認） |
| `tools/recovery_job.sh` | PBS ジョブ（資源の要求は `paper_gateoff_job.sh` と同じ）。合成データが無ければ先に作る |
| `tools/recovery_evaluate.py` | 評価: JSON・LaTeX の表・図（make_eml_figs56.py の図 4 と同じ書式に真値の縦線） |

## 3. GPU 側でやること

```bash
cd ~/Tmcmc202601 && git pull
# (1) 合成データを先に作って検算（CPU、各 20 秒）。無ノイズの検算も同時に
for T in CH CS; do
  JAX_PLATFORMS=cpu python3 tools/recovery_make_synthetic.py --tag $T --noise 1 --out data_5species/main/_runs/recovery/${T}_syn_n1_r0/synthetic.json
  JAX_PLATFORMS=cpu python3 tools/recovery_make_synthetic.py --tag $T --noise 0 --out data_5species/main/_runs/recovery/${T}_syn_n0_r0/synthetic.json
done
# (2) estimator の経路で尤度が一致するか（TMCMC は回さない）
cd data_5species/main
JAX_PLATFORMS=cpu python3 ../../tools/recovery_run.py --synthetic _runs/recovery/CH_syn_n1_r0/synthetic.json \
    --seed 42 --output-dir /tmp/rec_dry --device cpu --dry-run
# (3) 投入（CH・CS × seed 42/7/123 = 6 本。ノードは明示、celtic03 は使わない）
for T in CH CS; do for S in 42 7 123; do
  qsub -l nodes=1:ppn=1:gpus=1:stuttgart02 -l walltime=20:00:00 -v TAG=$T,SEED=$S ../../tools/recovery_job.sh
done; done
```

- 本数の上限（15）に入らなければ、本線の終わりに `-W depend=afterany:<jobid>` で依存させる。ノードは空いているもの（stuttgart01〜03 / celtic01・02）。
- 出力: `data_5species/main/_runs/recovery/{CH,CS}_syn_n1_r0_seed{42,7,123}/`（いつもの 5 ファイル＋`recovery_record.json`）。
- 余裕があれば無ノイズ版（`NOISE=0`、seed 42 の 1 本ずつ）も。こちらはノイズの実現値に左右されない「純粋な同定性」の確認になる。
- **DS・DH**: ult が判定を通ったら、その群を `SRC_GLOB` で渡す（例 `-v TAG=DS,SEED=42,SRC_GLOB='DS_ult_..._seed*'`）。
  元 run の粒子数（DS 18000 など）がそのまま使われるので、walltime は**その ult の実測 × 1.5**。
  final_runs に置く前なら `SRC_ROOT=$HOME/Tmcmc202601/data_5species/main/_runs/paper_gateoff` も渡す。

### 所要時間の見込み

実データの ult（5000 粒子・N_MUT 150）の実測（run_record の `total_time_s`）: **CH 9.7〜9.8 h（7 段）、CS 7.7〜9.0 h（6〜7 段）**、1 段 ≈ 1.3〜1.4 h。
回復テストは warm start なしなので段数が 1〜3 増えうる → **1 本 10〜14 h を見込み、walltime 20 h**（CS ult が 8 h で切れた前例あり、10-10）。
合成データの作成・dry-run は CPU で各 20 秒以下。

## 4. 評価（GPU 側でもクラウド側でも。CPU で数秒）

```bash
R=data_5species/main/_runs/recovery
python3 tools/recovery_evaluate.py \
  --case "CS=$R/CS_syn_n1_r0/synthetic.json=$R/CS_syn_n1_r0_seed*" \
  --case "CH=$R/CH_syn_n1_r0/synthetic.json=$R/CH_syn_n1_r0_seed*" \
  --out docs/revision/EML_submission/figures/provisional_2026-10-11/recovery
```

- 終わった run は final_runs と同じ要領で `docs/revision/generated/recovery_runs/` に上げてほしい（5 ファイル＋`recovery_record.json`、合成データの `synthetic.json` も）。評価はそれを `--case` に渡せばクラウド側でも回せる。
- 各 run の `recovery_record.json` の `checks`（箱・σ・初期値・λ の照合）がすべて true であること、`beta_final` = 1 であることを確認。
  `max_logL_minus_truth` が大きく負（例 −2 より下）なら探索不足（真値より悪い点しか見つけていない）→ 報告。
- 判定の読み方: (a) 実データで同定の成分は全部が区間に真値を含み、合成でも同定　(b) n.i. は合成でも n.i.。
  ずれた成分は表で斜体、区間に真値が無い成分は †。

## 5. 補足資料（supplement、`EML_supplementary.tex` の同定性の節）に入れるもの

- 表: `recovery.tex`（成分 × 条件: 真値、中央値 [5%, 95%]、sd 比、分類 合成/実）。`\input` するだけ（booktabs・xcolor）
- 図: `recovery.pdf`（箱の中の 50%/90% 区間、黒の縦線が真値）
- 本文 2〜3 文の案: “To test whether the classification reflects the information content of the data rather than the sampler, we generated synthetic data from the MAP of each condition with the same observation operator, noise level, time points and initial state, and repeated the final-stage inference (three seeds). Entries classified as identified on the real data were recovered (true value inside the 90 % interval for k/n), and entries classified as not identified remained at the width of the prior box.”
- 書き方の注意: 真値を固定した 1 回の合成データでは、90% 区間の被覆は成分ごとに独立な 0.9 の試行ではない（成分間の相関、ベイズの被覆は事前で平均した性質）。**主張は分類の一致で、被覆率は目安。**
  また箱は実データの ult の箱（p2 の事後の ± 4σ）なので、真値は構成上その中にある。この点は 1 文で断る。

## 6. CPU での事前確認（クラウド側、2026-10-12）

- 無ノイズ: θ_true での log L は **CH・CS とも 0.000000**（全項 ≤ 0 なので最大。残差 0）。同じ点の実データでの log L は −8.646（CH）・−15.810（CS）で run の max logL と一致（尤度の組み直しが正しい）。
- ノイズあり（rng 0）: log L(θ_true) は **CH −8.90、CS −12.49**。同じノイズを 2000 回引き直した分布は CH −29.6 ± 28.6（5〜95%: −73.4〜−8.4）、CS −32.4 ± 28.1 で、実現値はその 94・88 パーセンタイル（分布は希少種の重み λrare が観測の平均で 0.1 / 1 に切り替わるので裾が重い）。
- estimator の経路（`recovery_run.py --dry-run`）でも log L(θ_true) = −8.897766 で一致。`--tiny`（20 粒子・1 段、CPU 90 秒）で保存・照合（箱・σ・初期値・λ）まで通った。
- 評価スクリプトは実データの最終 run の粒子を「回復の事後」の代わりに入れて出力を確認した（表は pdflatex で通る、図は usetex で描ける）。この模擬では分類一致 15/15（当然）、被覆 CH 13/15・CS 14/15。
