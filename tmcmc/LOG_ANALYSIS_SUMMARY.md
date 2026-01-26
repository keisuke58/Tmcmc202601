# ログ分析サマリー

## 実行前チェック結果

### ✓ 依存関係
- numpy: OK
- scipy: OK
- matplotlib: OK
- numba: OK (オプション、パフォーマンス向上に使用)

### ✓ 必要なファイル
すべての必要なファイルが存在します。

### ✓ 過去の実行履歴
- **成功した実行**: 14件
- **失敗した実行**: 4件（`_failed`ディレクトリに保存）

## 最新の成功した実行

### m1_20260117_100253
- **実行日時**: 2026-01-17 10:02:54
- **モード**: DEBUG
- **設定**: 
  - n_particles: 2000
  - n_stages: 30
  - n_mutation_steps: 5
  - n_chains: 1
  - sigma_obs: 0.01
  - cov_rel: 0.005
- **ステータス**: 正常終了（エラーなし）
- **実行時間**: 約4時間43分（Stage 1のみで約3時間42分）

### m1_check_np1000_ns20_sig002
- **実行日時**: 2026-01-16 09:04:53 - 15:28:30
- **モード**: debug
- **設定**:
  - n_particles: 1000
  - n_stages: 20
  - n_mutation_steps: 3
  - sigma_obs: 0.02
- **ステータス**: **PASS**
- **結果**:
  - RMSE_total (MAP): 0.1002
  - MAE_total (MAP): 0.0815
  - max_abs (MAP): 0.2257
  - rom_error_final: 1.024e-15
  - ESS_min: 795.8
  - accept_rate_mean: 0.5007
  - beta_final: 1
  - beta_stages: 8

## ログから確認された問題

### エラー
- **最新の成功した実行（m1_20260117_100253）**: エラーなし
- **subprocess.log**: エラーなし
- **ImportError/ModuleNotFoundError**: 見つかりませんでした

### 注意点
1. **実行時間**: M1の計算には時間がかかります（数時間）
   - Stage 1のみで約3-4時間
   - 全30ステージ完了には相当な時間が必要

2. **リソース使用**: 
   - メモリ使用量が大きい可能性があります
   - CPU使用率も高い可能性があります

3. **失敗した実行**: 
   - `_failed`ディレクトリに4件の失敗記録があります
   - 主にパラメータスイープ実行での失敗のようです

## 推奨設定

### デバッグ/テスト実行
```bash
--n-particles 500
--n-stages 10
--n-mutation-steps 3
```

### 本番実行
```bash
--n-particles 2000
--n-stages 30
--n-mutation-steps 5
```

## 実行準備完了

すべてのチェックが完了し、実行可能な状態です。

### 実行方法

**Windows (バッチファイル)**:
```cmd
cd tmcmc_docs\tmcmc
run_m1.bat
```

**Python直接実行**:
```cmd
cd tmcmc_docs
python tmcmc\run_pipeline.py --mode debug --models M1
```

**オプション付き実行**:
```cmd
python tmcmc\run_pipeline.py --mode debug --models M1 --n-particles 1000 --n-stages 20
```

## 実行中の監視

実行中は以下のログファイルを確認できます：
- `tmcmc/_runs/<run_id>/run.log`: メインログ
- `tmcmc/_runs/<run_id>/subprocess.log`: サブプロセスログ
- `tmcmc/_runs/<run_id>/pipeline.log`: パイプラインログ

実行完了後：
- `tmcmc/_runs/<run_id>/REPORT.md`: 実行レポート
- `tmcmc/_runs/<run_id>/figures/`: 生成された図


