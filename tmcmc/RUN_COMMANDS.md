# M1計算実行コマンド

## 現在のディレクトリ
```
C:\Users\nishioka\Neuer Ordner\tmcmc_docs
```

## ステップ1: テスト実行（100パーティクル）

```cmd
cd "C:\Users\nishioka\Neuer Ordner\tmcmc_docs"
python tmcmc\run_pipeline.py --mode debug --models M1 --n-particles 100 --n-stages 10 --n-mutation-steps 3 --run-id m1_test_100
```

## ステップ2: 本番実行（5000パーティクル）

テスト実行が成功したら、以下のコマンドで実行：

```cmd
cd "C:\Users\nishioka\Neuer Ordner\tmcmc_docs"
python tmcmc\run_pipeline.py --mode debug --models M1 --n-particles 5000 --n-stages 30 --n-mutation-steps 5 --n-chains 1 --run-id m1_5000_<timestamp>
```

または、バッチファイルを使用：

```cmd
cd "C:\Users\nishioka\Neuer Ordner\tmcmc_docs\tmcmc"
run_m1_5000.bat
```

## 実行状況の確認

```cmd
cd "C:\Users\nishioka\Neuer Ordner\tmcmc_docs"
python tmcmc\check_run_status.py <run_id>
```

## ログの確認

```cmd
type tmcmc\_runs\<run_id>\run.log
```


