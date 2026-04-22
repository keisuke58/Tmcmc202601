# GitHub 運用ガイド（CPU環境 ↔ GPUサーバー）

## 概要

- **今の環境（CPU）**: 開発・編集
- **GPUサーバー**: 学習・TMCMC・重い計算
- **GitHub**: コード共有・バージョン管理

---

## 1. 初回セットアップ（CPU環境で一度だけ）

### 1.1 GitHub にリポジトリを作成

1. https://github.com/new にアクセス
2. リポジトリ名を入力（例: `IKM_Hiwi`）
3. **Private** または **Public** を選択
4. 「Create repository」をクリック
5. **README や .gitignore は追加しない**（既にローカルにあるため）

### 1.2 リモートを追加してプッシュ

```bash
cd /home/nishioka/IKM_Hiwi

# リモート追加（YOUR_USERNAME を自分の GitHub ユーザー名に置換）
git remote add origin https://github.com/YOUR_USERNAME/IKM_Hiwi.git

# 現在の変更をコミット（必要なら）
git add .
git status   # 確認
git commit -m "feat: initial push to GitHub"

# プッシュ
git push -u origin master
```

**認証**: 初回は GitHub のユーザー名とパスワード（または Personal Access Token）を聞かれる。

---

## 2. GPUサーバーで作業するとき

### 2.1 初回：クローン

```bash
# 適当な作業ディレクトリへ
cd ~
git clone https://github.com/YOUR_USERNAME/IKM_Hiwi.git
cd IKM_Hiwi
```

### 2.2 仮想環境・依存関係のセットアップ

```bash
# 例: venv
python -m venv .venv
source .venv/bin/activate   # Linux
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
# GNN 用: pip install -r Tmcmc202601/gnn/requirements-gnn.txt
```

### 2.3 作業フロー

```bash
# 1. 最新を取得（CPU環境で push した変更を取り込む）
git pull origin master

# 2. 計算・学習を実行
python Tmcmc202601/tmcmc/program2602/run_pipeline.py
# など

# 3. 結果をコミットしてプッシュ（コード変更のみ推奨）
git add Tmcmc202601/...
git commit -m "feat: add GPU-accelerated TMCMC results"
git push origin master
```

**注意**: `.gitignore` で `*.npy`, `*.pt`, `runs/` などは除外済み。大きな結果ファイルは GitHub に上げない。

---

## 3. CPU環境に戻ったとき

```bash
cd /home/nishioka/IKM_Hiwi
git pull origin master   # GPUサーバーで push した変更を取得
```

---

## 4. よく使うコマンド一覧

| 操作 | コマンド |
|------|----------|
| 最新を取得 | `git pull origin master` |
| 変更を確認 | `git status` |
| コミット | `git add .` → `git commit -m "メッセージ"` |
| プッシュ | `git push origin master` |
| ブランチ作成 | `git checkout -b feature/xxx` |

---

## 5. トラブルシュート

### プッシュ時に認証エラー

- **HTTPS**: Personal Access Token を使う（Settings → Developer settings → Personal access tokens）
- **SSH**: `git remote set-url origin git@github.com:YOUR_USERNAME/IKM_Hiwi.git` に変更し、SSH 鍵を登録

### コンフリクト

```bash
git pull origin master
# CONFLICT が出たら、ファイルを編集して
git add .
git commit -m "fix: resolve merge conflict"
git push origin master
```

### 大きなファイルを誤って add した

```bash
git reset HEAD ファイル名
# または .gitignore に追加してから
git rm --cached ファイル名
```

---

## 6. 推奨ワークフロー

```
[CPU環境] 編集 → git add/commit → git push
                    ↓
              [GitHub]
                    ↓
[GPUサーバー] git pull → 実行 → (コード変更があれば) git push
```

**ポイント**: データ・モデル重み（`*.npy`, `*.pt`）は GitHub に上げず、別途共有（共有ストレージ、scp、云々）する。
