# バックグラウンド実行ガイド

Windows環境で `run_pipeline.py` をバックグラウンドで実行する方法です。

## 重要な注意事項

⚠️ **PCをシャットダウンするとプロセスは終了します**
- ターミナルやIDEを閉じても実行は継続します
- しかし、PCをシャットダウンするとプロセスは停止します
- 長時間実行が必要な場合は、実行が完了するまでPCを起動したままにしてください

## 使用方法

### 1. バックグラウンド実行の開始

#### PowerShellから実行:
```powershell
cd "C:\Users\nishioka\Neuer Ordner\tmcmc_docs"
.\tmcmc\run_pipeline_bg.ps1 -Mode debug -Seed 42 -Models M1 -NParticles 5000 -NStages 30
```

#### バッチファイルから実行:
```cmd
cd "C:\Users\nishioka\Neuer Ordner\tmcmc_docs\tmcmc"
run_pipeline_bg.bat
```

#### 引数の例:
```powershell
# デフォルト設定
.\tmcmc\run_pipeline_bg.ps1

# カスタム設定
.\tmcmc\run_pipeline_bg.ps1 `
    -Mode debug `
    -Seed 42 `
    -Models "M1,M2,M3" `
    -NParticles 5000 `
    -NStages 30 `
    -NMutationSteps 5 `
    -NChains 1
```

### 2. 実行状況の確認

```powershell
.\tmcmc\check_pipeline_bg.ps1
```

または

```cmd
check_pipeline_bg.bat
```

これにより以下が表示されます:
- プロセスの実行状態
- PID（プロセスID）
- CPU使用率とメモリ使用量
- 最新のログの最後の10行

### 3. ログのリアルタイム監視

```powershell
.\tmcmc\watch_pipeline_bg.ps1
```

または

```cmd
watch_pipeline_bg.bat
```

Ctrl+C で監視を終了できます。

### 4. プロセスの停止

```powershell
.\tmcmc\stop_pipeline_bg.ps1
```

または

```cmd
stop_pipeline_bg.bat
```

## ファイルの場所

- **PIDファイル**: `tmcmc/run_pipeline_bg.pid` - 実行中のプロセスIDが保存されます
- **ステータスファイル**: `tmcmc/run_pipeline_bg_status.txt` - 実行開始時刻などの情報
- **ログファイル**: `tmcmc/_runs/<run_id>/run.log` - 実行ログ

## Linux/Mac の screen や tmux との違い

Windowsでは `screen` や `tmux` は標準では利用できません。このスクリプトは以下の機能を提供します:

- ✅ ターミナルを閉じても実行が継続
- ✅ バックグラウンドでの実行
- ✅ ログの監視
- ❌ PCをシャットダウンしても継続（これは不可能です）

## WSL (Windows Subsystem for Linux) を使用する場合

WSLがインストールされている場合は、Linuxの `screen` や `tmux` を使用することもできます:

```bash
# WSLで実行
wsl
cd /mnt/c/Users/nishioka/Neuer\ Ordner/tmcmc_docs
screen -S tmcmc
python tmcmc/run_pipeline.py --mode debug --models M1 --n-particles 5000
# Ctrl+A, D でデタッチ
```

## トラブルシューティング

### プロセスが見つからない

PIDファイルが残っているがプロセスが実行されていない場合:
```powershell
.\tmcmc\check_pipeline_bg.ps1
```
実行時にPIDファイルの削除を提案されます。

### ログファイルが見つからない

実行が開始されていないか、実行ディレクトリが作成されていない可能性があります。
`tmcmc/_runs/` ディレクトリを確認してください。

### 複数のプロセスを実行したい

各実行に異なる `--run-id` を指定してください:
```powershell
.\tmcmc\run_pipeline_bg.ps1 -RunId "run1" -Models M1
.\tmcmc\run_pipeline_bg.ps1 -RunId "run2" -Models M2
```

ただし、PIDファイルは1つしか管理できないため、複数実行時は手動でPIDを管理する必要があります。
