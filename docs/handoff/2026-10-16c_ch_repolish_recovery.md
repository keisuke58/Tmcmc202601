# クラウド側 → GPU 側（2026-10-16c）: 空いている GPU で (1) CH の磨き直し、(2) CH の合成データ回収テスト（2 通路）

ユーザー決定（10/16）。GPU は 12 / 15 本なので 3 本空いている。**本線（DS の p2・gate、CS・DH の ult、DS p1 の磨き直し）を優先**し、空いた枠で。

## 1. CH の磨き直し（16a §4・16b の設定）— すぐ
`TOP_K=3 MAXITER=3000`（ftol は既定 1e-7）、`WRITE_RUN=1` で、seed ごとに `tools/polish_max_logL_job.sh`:
- `CH_p1_2c_raw_seed{42,7,123}`、`CH_p2_2c_raw_a33w_seed{42,7,123}`、`CH_ult_2c_raw_a33w_seed{42,7,123}`（9 本。枠が 3 本なら ult → p2 → p1 の順に依存でつなぐ）
- 終わったら `check_paper_runs.py` で 3 群を判定し直し（判定 3 が「磨いた値」と出ること）、磨いた max logL・converged・幅・所要時間を 1 枚に。
- 磨いた値を書いた run（`--write-run`）は、`final_runs/` の同じ run にも polish.json をコピーして push（補足の表をクラウドで作るため）。

## 2. CH の合成データ回収テスト（12h の手順、2 通路・生データ版）— 1 のあと
クラウドで確認済み（10/16）: `recovery_make_synthetic.py --tag CH --glob 'CH_ult_2c_raw_a33w_seed*'` が final_runs から動き、
真値（seed123 の MAP）の log L は実データで −26.523（元の run と一致）、合成データで −20.946（estimator の経路でも同じ値、`--dry-run`）、
σ_viab 0.095、λψ は 1.5（元の run の `lambda_ch_effective` と同じ）。合成データは `--two-channel --data-source raw` の estimator で回る。

```bash
cd ~/Tmcmc202601 && git pull
cd data_5species/main
for S in 42 7 123; do
  qsub -l nodes=1:ppn=1:gpus=1:<空いているノード> -l walltime=20:00:00 \
    -v TAG=CH,SEED=$S,SRC_GLOB='CH_ult_2c_raw_a33w_seed*',RUNTAG=2c_raw ../../tools/recovery_job.sh
done
```
- 箱は元の ult の `prior_bounds_final`、5000 粒子・N_MUT 150、warm start なし（12h のとおり）。1 本 10〜14h の見込み
- 出力 `_runs/recovery/CH_syn_n1_r0_2c_raw_seed*` を `docs/revision/generated/recovery_runs/` に（5 ファイル＋`recovery_record.json`＋`synthetic.json`）push
- 各 `recovery_record.json` の `checks` がすべて true、`beta_final` = 1、`max_logL_minus_truth` が −2 より上であることを確認。外れたら報告
- CS・DS・DH は ult が通ってから同じ形で（DS・DH は粒子数が多いので walltime は ult の実測 × 1.5）
