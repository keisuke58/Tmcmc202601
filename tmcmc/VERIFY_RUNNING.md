# 実行確認ガイド
## PCを閉じても実行が継続するか確認

---

## ✅ 現在の状態

screenセッションが正常に作成されています：
- **セッション名**: `sweep_m1_20260120_214522`
- **PID**: 275846
- **ログファイル**: `/home/nishioka/IKM_Hiwi/tmcmc/sweep_logs/sweep_20260120_214522.log`

---

## 🔍 実行確認方法

### 1. セッションが実行中か確認

```bash
# screenセッション一覧を表示
screen -ls

# 実行中なら以下のように表示されます：
# There is a screen on:
#   275846.sweep_m1_20260120_214522    (Attached/Detached)
```

### 2. ログを確認（実行中か確認）

```bash
# リアルタイムでログを確認
tail -f /home/nishioka/IKM_Hiwi/tmcmc/sweep_logs/sweep_20260120_214522.log

# 最後の50行を表示
tail -n 50 /home/nishioka/IKM_Hiwi/tmcmc/sweep_logs/sweep_20260120_214522.log
```

### 3. セッションに接続して確認

```bash
# セッションに接続（実行中の出力を見られる）
screen -r sweep_m1_20260120_214522

# 切断する場合: Ctrl+A, その後 D を押す
```

### 4. プロセスが実行中か確認

```bash
# sweep_m1.shが実行中か確認
ps aux | grep sweep_m1.sh | grep -v grep

# または、run_pipeline.pyが実行中か確認
ps aux | grep run_pipeline.py | grep -v grep
```

---

## 💻 PCを閉じても大丈夫？

### ✅ はい、大丈夫です！

**理由**:
1. **screenセッションはサーバー上で実行されている**
   - PCの電源を切っても、サーバー側のscreenセッションは継続
   - SSH接続が切れても、screenセッションは継続

2. **実行はサーバー上で行われる**
   - `/home/nishioka/IKM_Hiwi/tmcmc/` はサーバーのパス
   - 全ての処理はサーバー上で実行される

3. **ログもサーバーに保存される**
   - ログファイルはサーバー上に保存される
   - PCを閉じても、ログは継続して書き込まれる

---

## 🔄 再接続後の確認方法

PCを再度開いて、サーバーにSSH接続した後：

```bash
# 1. サーバーにSSH接続
ssh nishioka@<サーバーアドレス>

# 2. screenセッションを確認
screen -ls

# 3. セッションに接続（必要に応じて）
screen -r sweep_m1_20260120_214522

# 4. ログを確認
tail -f /home/nishioka/IKM_Hiwi/tmcmc/sweep_logs/sweep_20260120_214522.log
```

---

## ⚠️ 注意事項

### サーバーが再起動された場合

- **screenセッションは失われます**
- サーバー再起動後は、再度実行する必要があります

### ネットワーク切断の場合

- **SSH接続が切れても、screenセッションは継続**
- 再接続後、`screen -r` でセッションに再接続可能

---

## 📊 実行状態の確認コマンド

```bash
# ステータス確認スクリプトを使用
cd /home/nishioka/IKM_Hiwi/tmcmc
bash check_sweep_status.sh
```

このスクリプトは以下を表示：
- 実行中のscreen/tmuxセッション
- 実行中のプロセス
- 最近のログファイル
- ディスク使用量

---

## 🎯 まとめ

### ✅ PCを閉じてもOK

- screenセッションはサーバー上で実行されている
- PCの電源を切っても、実行は継続
- SSH接続が切れても、実行は継続

### 🔍 確認方法

```bash
# セッション確認
screen -ls

# ログ確認
tail -f /home/nishioka/IKM_Hiwi/tmcmc/sweep_logs/sweep_20260120_214522.log

# セッションに接続
screen -r sweep_m1_20260120_214522
```

---

_作成日時: 2026-01-20_
