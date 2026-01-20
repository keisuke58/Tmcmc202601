# バックグラウンド実行ガイド
## サーバーでPCの電源が切れても継続実行する方法

---

## 🎯 概要

`sweep_m1.sh`をサーバーでバックグラウンド実行し、PCの電源が切れても継続実行できるようにする方法です。

3つの方法を提供：
1. **screen**（推奨）- セッション管理が簡単
2. **tmux** - screenの代替、より高機能
3. **nohup** - シンプルだが、セッション管理ができない

---

## 🚀 クイックスタート

### 方法1: screen（推奨）

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc

# バックグラウンド実行
bash run_sweep_background.sh screen

# セッションに接続（進捗確認）
screen -r sweep_m1_<timestamp>

# セッションから切断（実行は継続）
# Ctrl+A, その後 D を押す
```

### 方法2: tmux

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc

# バックグラウンド実行
bash run_sweep_background.sh tmux

# セッションに接続
tmux attach -t sweep_m1_<timestamp>

# セッションから切断
# Ctrl+B, その後 D を押す
```

### 方法3: nohup

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc

# バックグラウンド実行
bash run_sweep_background.sh nohup

# ログを確認
tail -f sweep_logs/sweep_<timestamp>.log
```

---

## 📋 詳細な使い方

### 1. screen を使う方法（推奨）

#### インストール（必要な場合）

```bash
# CentOS/RHEL
sudo yum install screen

# Ubuntu/Debian
sudo apt-get install screen
```

#### 実行

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc
bash run_sweep_background.sh screen
```

#### セッション管理

```bash
# セッション一覧を表示
screen -ls

# セッションに接続
screen -r sweep_m1_<timestamp>

# セッションから切断（実行は継続）
# セッション内で: Ctrl+A, その後 D

# セッションを終了（実行も停止）
# セッション内で: exit または Ctrl+D
```

#### メリット
- ✅ セッションに再接続して進捗を確認できる
- ✅ 実行中の出力をリアルタイムで見られる
- ✅ 簡単に操作できる

---

### 2. tmux を使う方法

#### インストール（必要な場合）

```bash
# CentOS/RHEL
sudo yum install tmux

# Ubuntu/Debian
sudo apt-get install tmux
```

#### 実行

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc
bash run_sweep_background.sh tmux
```

#### セッション管理

```bash
# セッション一覧を表示
tmux ls

# セッションに接続
tmux attach -t sweep_m1_<timestamp>

# セッションから切断（実行は継続）
# セッション内で: Ctrl+B, その後 D

# セッションを終了（実行も停止）
# セッション内で: exit または Ctrl+D
```

#### メリット
- ✅ screenより高機能
- ✅ 複数のウィンドウを管理できる
- ✅ セッション管理が柔軟

---

### 3. nohup を使う方法

#### 実行

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc
bash run_sweep_background.sh nohup
```

#### ログ確認

```bash
# リアルタイムでログを確認
tail -f sweep_logs/sweep_<timestamp>.log

# 最後の100行を表示
tail -n 100 sweep_logs/sweep_<timestamp>.log
```

#### プロセス確認

```bash
# プロセスが実行中か確認
ps -p <PID>

# 全てのsweepプロセスを確認
ps aux | grep sweep_m1.sh
```

#### メリット
- ✅ シンプル
- ✅ 追加ソフトウェア不要
- ❌ セッションに再接続できない（ログのみ）

---

## 📊 実行状態の確認

### ステータス確認スクリプト

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc
bash check_sweep_status.sh
```

このスクリプトは以下を表示：
- 実行中のscreen/tmuxセッション
- 実行中のnohupプロセス
- 最近のログファイル
- ディスク使用量

### 手動確認

```bash
# screenセッション確認
screen -ls

# tmuxセッション確認
tmux ls

# プロセス確認
ps aux | grep sweep_m1.sh

# ログ確認
ls -lht sweep_logs/
tail -f sweep_logs/sweep_<timestamp>.log
```

---

## 🔍 ログファイルの場所

全てのログは `sweep_logs/` ディレクトリに保存されます：

```
sweep_logs/
├── sweep_20260119_120000.log  # 実行ログ
├── sweep_20260119_120000.pid  # プロセスID
├── sweep_20260119_130000.log
└── ...
```

---

## ⚠️ 注意事項

### 1. サーバーの再起動

- **screen/tmux**: サーバーが再起動されるとセッションは失われます
- **nohup**: プロセスは終了します
- **対策**: システムサービス（systemd）を使うか、サーバー再起動後に再実行

### 2. ディスク容量

長時間実行する場合、ログファイルと結果ファイルが大きくなる可能性があります：

```bash
# ディスク使用量を確認
du -sh tmcmc/_runs/
du -sh sweep_logs/
```

### 3. メモリ使用量

高粒子数（10000-20000）の場合、メモリ使用量が大きくなります：

```bash
# メモリ使用量を確認
free -h
top
htop
```

### 4. ネットワーク切断

- **screen/tmux**: ネットワークが切断されても実行は継続
- **nohup**: ネットワークが切断されても実行は継続
- 再接続後、screen/tmuxセッションに再接続可能

---

## 🛠️ トラブルシューティング

### セッションに接続できない

```bash
# screenセッションが応答しない場合
screen -D -r <session_name>  # 強制再接続

# tmuxセッションが応答しない場合
tmux kill-session -t <session_name>  # セッションを強制終了（注意！）
```

### プロセスが停止している

```bash
# プロセスIDを確認
cat sweep_logs/sweep_<timestamp>.pid

# プロセスが実行中か確認
ps -p <PID>

# 実行されていない場合、ログを確認
tail -n 100 sweep_logs/sweep_<timestamp>.log
```

### ログファイルが大きくなりすぎる

```bash
# ログをローテート（古いログを圧縮）
gzip sweep_logs/sweep_*.log

# または、古いログを削除（注意！）
find sweep_logs/ -name "*.log" -mtime +30 -delete
```

---

## 🎯 推奨ワークフロー

### 1. 実行開始

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc
bash run_sweep_background.sh screen
```

### 2. セッションに接続して確認

```bash
screen -r sweep_m1_<timestamp>
# 実行が開始されていることを確認
# Ctrl+A, D で切断
```

### 3. 定期的に進捗を確認

```bash
# ステータス確認
bash check_sweep_status.sh

# ログを確認
tail -f sweep_logs/sweep_<timestamp>.log

# または、セッションに再接続
screen -r sweep_m1_<timestamp>
```

### 4. 完了後の確認

```bash
# 最良実行を確認
cat tmcmc/_runs/sweep_m1_*/best_run_id.txt

# サマリーを確認
cat tmcmc/_runs/sweep_m1_*/sweep_summary.csv
```

---

## 📝 まとめ

### 推奨方法: screen

- ✅ セッション管理が簡単
- ✅ 進捗をリアルタイムで確認できる
- ✅ 再接続が容易

### 実行コマンド

```bash
cd /home/nishioka/IKM_Hiwi/tmcmc
bash run_sweep_background.sh screen
```

### 確認コマンド

```bash
# ステータス確認
bash check_sweep_status.sh

# セッションに接続
screen -ls
screen -r <session_name>
```

---

_作成日時: 2026-01-19_
