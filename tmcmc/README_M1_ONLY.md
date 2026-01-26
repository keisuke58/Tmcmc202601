# M1のみで計算する方法

このドキュメントでは、M1モデルのみでTMCMC計算を実行する方法を説明します。

## Windowsでの実行方法

### 方法1: バッチファイルを使用（推奨）

```cmd
cd tmcmc_docs\tmcmc
run_m1.bat
```

オプションを指定する場合：

```cmd
run_m1.bat --n-particles 3000 --n-stages 40
```

### 方法2: Pythonスクリプトを直接実行

```cmd
cd tmcmc_docs
python tmcmc\run_pipeline.py --mode debug --models M1
```

### 方法3: case2_tmcmc_linearization.pyを直接実行

```cmd
cd tmcmc_docs
python tmcmc\case2_tmcmc_linearization.py --mode debug --models M1
```

## 主なオプション

- `--models M1`: M1モデルのみを実行（デフォルトはM1,M2,M3）
- `--mode debug|paper|sanity`: 実行モード
- `--n-particles N`: パーティクル数（デフォルト: 2000）
- `--n-stages N`: ステージ数（デフォルト: 30）
- `--n-chains N`: チェーン数（デフォルト: 1）
- `--sigma-obs VAL`: 観測ノイズ（デフォルト: 0.01）
- `--cov-rel VAL`: 相対共分散（デフォルト: 0.005）

## 実行結果

実行結果は以下のディレクトリに保存されます：

```
tmcmc_docs\tmcmc\_runs\<run_id>\
```

- `REPORT.md`: 実行レポート
- `figures/`: 生成された図
- `*.npy`: データファイル
- `*.json`: 設定ファイル

## 例

### 基本的な実行（デフォルト設定）

```cmd
cd tmcmc_docs\tmcmc
run_m1.bat
```

### より多くのパーティクルで実行

```cmd
run_m1.bat --n-particles 5000 --n-stages 40
```

### 論文モードで実行

```cmd
run_m1.bat --mode paper
```


