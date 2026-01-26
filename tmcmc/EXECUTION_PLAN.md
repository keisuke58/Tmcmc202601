# M1計算実行計画

## ステップ1: テスト実行（100パーティクル）

まず、軽量な設定でテスト実行を行い、問題がないことを確認します。

### 実行コマンド
```cmd
cd C:\Users\nishioka\Neuer Ordner\tmcmc_docs
python tmcmc\run_pipeline.py --mode debug --models M1 --n-particles 100 --n-stages 10 --n-mutation-steps 3 --run-id m1_test_100
```

### 設定
- n_particles: 100
- n_stages: 10
- n_mutation_steps: 3
- n_chains: 1
- 予想実行時間: 10-30分程度

### 確認方法
```cmd
python tmcmc\check_run_status.py m1_test_100
```

または、REPORT.mdが生成されるまで待つ：
```cmd
dir tmcmc\_runs\m1_test_100\REPORT.md
```

## ステップ2: 本番実行（5000パーティクル）

テスト実行が成功したら、5000パーティクルで本番実行を行います。

### 実行方法1: バッチファイル（推奨）
```cmd
cd C:\Users\nishioka\Neuer Ordner\tmcmc_docs\tmcmc
run_m1_5000.bat
```

### 実行方法2: Python直接実行
```cmd
cd C:\Users\nishioka\Neuer Ordner\tmcmc_docs
python tmcmc\run_pipeline.py --mode debug --models M1 --n-particles 5000 --n-stages 30 --n-mutation-steps 5 --n-chains 1 --run-id m1_5000_<timestamp>
```

### 設定
- n_particles: 5000
- n_stages: 30
- n_mutation_steps: 5
- n_chains: 1
- 予想実行時間: 10-20時間程度（パーティクル数に比例）

### 実行状況の確認
```cmd
python tmcmc\check_run_status.py <run_id>
```

または、ログファイルを直接確認：
```cmd
type tmcmc\_runs\<run_id>\run.log
```

## 注意事項

1. **実行時間**: 5000パーティクルの実行には長時間（10-20時間）かかります
2. **リソース**: メモリとCPUを大量に使用します
3. **中断**: Ctrl+Cで中断できますが、途中結果は保存されません
4. **結果**: 実行完了後、`tmcmc/_runs/<run_id>/REPORT.md`に結果が保存されます

## 実行後の確認

実行完了後、以下を確認：
- `REPORT.md`: 実行結果のサマリー
- `figures/`: 生成された図
- `results_MAP_linearization.npz`: 数値結果


