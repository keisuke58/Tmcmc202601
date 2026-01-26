# 5000パーティクル実行情報

## 実行開始

5000パーティクルでM1のみの計算を開始しました。

## 実行設定

- **モデル**: M1のみ
- **パーティクル数**: 5000
- **ステージ数**: 30
- **Mutation steps**: 5
- **チェーン数**: 1
- **sigma_obs**: 0.01
- **cov_rel**: 0.005
- **seed**: 42
- **mode**: debug
- **debug_level**: MINIMAL

## 予想実行時間

5000パーティクルでは、**10-20時間程度**かかる可能性があります。

## 実行状況の確認方法

### 1. 最新の実行IDを確認

```cmd
cd "C:\Users\nishioka\Neuer Ordner\tmcmc_docs"
python tmcmc\find_latest_run.py
```

### 2. 実行状況を確認

```cmd
python tmcmc\check_running.py
```

### 3. ログファイルを直接確認

実行IDが分かったら：

```cmd
type tmcmc\_runs\<run_id>\run.log
```

または、メモ帳で開く：

```cmd
notepad tmcmc\_runs\<run_id>\run.log
```

## 実行中の注意事項

1. **長時間実行**: 10-20時間かかる可能性があります
2. **リソース使用**: メモリとCPUを大量に使用します
3. **進捗確認**: ログファイルで進捗を確認できます
4. **完了後**: `tmcmc/_runs/<run_id>/REPORT.md`に結果が保存されます

## 実行結果の確認

実行完了後、以下のファイルが生成されます：

- `REPORT.md`: 実行結果のサマリー
- `figures/`: 生成された図
- `results_MAP_linearization.npz`: 数値結果
- `run.log`: 実行ログ


