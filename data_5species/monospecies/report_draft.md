# レポート化（教授提出用）Draft

## 目的
- Monospecies biofilm の Hamilton ODE モデルに対し、TMCMC による温度別のパラメータ推定（σ）を行い、dynamic nutrient coupling `c(t)=c_scale*(1-φψ)` のスケール（`c_scale=10` と `100`）での違いを定量比較する（温度セットは Liu et al., 2019 を参考）
- 推定結果（MAP/事後分布）と再現性情報（実行条件、ソルバー設定）を添えて、図表＋誤差指標（RMSE 等）まで含んだ提出用データ一式を作る

## 最終成果物（提出パッケージ）
- 図
  - 8温度×1枚：観測点（replicate + mean±std）、MAP フィット、posterior predictive 95% CI
  - `c_scale=10` と `c_scale=100` の双方（必要なら static c=100 も）
- 表（CSV）
  - 温度×条件（c_scale 等）ごとの推定結果（MAP/CI）と誤差指標をまとめた “master CSV”
  - 実行条件（n_particles, mutation, seed, max_stages, dt, n_newton, dynamic_c, c_scale）を必ず列として含める
- レポート本文（LaTeX）
  - 方法：モデル、ソルバー、推定（TMCMC）、誤差指標、可視化
  - 結果：c_scale=10 vs 100 の比較、温度依存の傾向、残差/カバレッジ
  - 付録：再現コマンド、ディレクトリ構造、生成物一覧

## 解析フロー（再現手順）
### 1) 推定（TMCMC）
- 推定スクリプト：`estimate_monospecies_tmcmc.py`
- dynamic c の切替：
  - `--dynamic-c --c-scale 10`
  - `--dynamic-c --c-scale 100`
- ソルバー調整（推定と後処理で一致させる）：
  - `--dt 1e-4`
  - `--n-newton 6`

例（全温度、c_scale=10）：
```bash
python3 estimate_monospecies_tmcmc.py \
  --mutation rw --n_particles 4000 --max_stages 80 --seed 42 \
  --dynamic-c --c-scale 10 --dt 1e-4 --n-newton 6 \
  --outdir _tmcmc_results_dyncc10_reest_all_<timestamp>
```

### 2) 後処理（誤差計算＋図の生成＋提出用CSV）
- 後処理スクリプト：`postprocess_monospecies_tmcmc.py`
- 入力：TMCMC outdir（複数指定可）
- 出力：
  - 各 outdir 内に `monospecies_compiled_metrics.csv` と 8panel png
  - monospecies 直下に `monospecies_master_compiled_metrics.csv`

例（c_scale=10/100 をまとめて集計）：
```bash
python3 postprocess_monospecies_tmcmc.py \
  --tmcmc_dir _tmcmc_results_dyncc10_reest_all_<timestamp> \
  --tmcmc_dir _tmcmc_results_dyncc100_reest_all_<timestamp> \
  --n_pp_samples 200 --seed 0
```

## 指標定義（本文に必ず明記）
- 変換：観測時刻の `φ̄(t)=φ(t)ψ(t)` に対し、`log10(CFU/mL) ≈ a*φ̄ + b` の最小二乗で `a,b` をプロファイルアウト
- MAP 誤差（平均データ）：
  - RMSE（mean vs MAP予測）
  - MAE（mean vs MAP予測）
  - R²（mean vs MAP予測）
- 観測ノイズ込み指標（noise floor を使用）：
  - `σ_obs = max(std, noise_floor)` を用いた（近似）log-likelihood / NLL
  - `WRMSE = sqrt(mean(((pred-obs)/σ_obs)^2))`
- 予測区間（posterior predictive 95% CI）：
  - 観測時刻における 2.5/97.5 percentile
  - replicate が CI に入る割合（coverage）
  - mean が CI に入る割合（coverage）

## 図の仕様（教授向けの最低限）
- パネル：8温度（4, 8, 15, 20, 25, 35, 37, 40℃）
- 各パネルに明記する情報：
  - 温度、MAP σ、`c(t)=c_scale*(1-φψ)` の c_scale、（必要なら dt/n_newton）
- 図注に書く：
  - 橙点：replicate
  - エラーバー：mean±std
  - 青線：MAP σ のモデル予測（affine mapping 適用後）
  - 青帯：posterior predictive 95% CI

## レポート構成案（LaTeX）
1. Introduction
   - 背景（biofilm、温度依存、モデル化の意義）
   - 目的（c_scale の比較も含む）
2. Model
   - 状態変数（φ, φ0, ψ, γ）と φ̄=φψ
   - dynamic nutrient coupling `c(t)=c_scale*(1-φψ)`
3. Numerical Method
   - implicit Euler + Newton（dt, n_newton）
   - 収束安定性の扱い（clip 等）
4. Parameter Estimation
   - 推定対象：σ（温度ごと1パラメータ）
   - TMCMC（mutation, n_particles, max_stages, seed）
   - プロファイル尤度（a,b の解析最適化）
5. Evaluation Metrics
   - RMSE/MAE/R²、weighted 指標、coverage
6. Results
   - c_scale=10 の結果（表＋図）
   - c_scale=100 の結果（表＋図）
   - 比較（差分、傾向、どちらがデータに整合的か）
7. Discussion
   - c_scale の物理的解釈、限界、今後（多種系への拡張等）
8. Reproducibility (Appendix)
   - コマンド、ディレクトリ、生成物一覧

## 本文に埋め込む表・図の候補
- Table: 推定結果（temp, σ_MAP, mean±std, CI, stages, time）
- Table: 誤差指標（RMSE/MAE/R²/WRMSE/coverage）
- Figure: 8panel（c_scale=10）
- Figure: 8panel（c_scale=100）
- Figure: σ_MAP vs temperature（c_scale=10/100 の比較）

## 依存ファイル（参照）
- 推定：`estimate_monospecies_tmcmc.py`
- ソルバー：`hamilton_monospecies_jax.py`
- 可視化：`plot_monospecies_8panel.py`
- 後処理：`postprocess_monospecies_tmcmc.py`
- データ：`raw data.xlsx`（sheet `St`）
