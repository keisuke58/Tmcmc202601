# ツールセットアップガイド

DVC, pre-commit, Zotero, Hydra, MLflow の導入・運用ガイド。

> **関連**: [魔改造計画 Issue #94](https://github.com/keisuke58/Tmcmc202601/issues/94)

---

## 1. DVC（データ・チェックポイント管理）

### インストール

```bash
pip install dvc
# リモートに S3/GDrive を使う場合
pip install dvc[s3]   # または dvc[gdrive]
```

### 初期化（Tmcmc202601 で）

```bash
cd Tmcmc202601
dvc init
git add .dvc .dvcignore
git commit -m "chore: add DVC"
```

### トラッキング対象の例

```bash
# 学習データ（大きい場合）
dvc add gnn/data/train_gnn_N10000.npz
dvc add data_5species/experiment_data/

# チェックポイント（条件別）
dvc add gnn/data/checkpoints/
dvc add deeponet/checkpoints_Dysbiotic_HOBIC_50k/

# 実行結果（再生成に時間がかかる場合）
dvc add data_5species/_runs/Dysbiotic_HOBIC_20260226_022824/
```

### リモート設定（オプション）

```bash
# ローカル共有ストレージ
dvc remote add -d storage /path/to/shared/storage

# Google Drive
dvc remote add -d storage gdrive://<folder_id>

# S3
dvc remote add -d storage s3://bucket/path
```

### よく使うコマンド

| コマンド | 用途 |
|----------|------|
| `dvc add <path>` | ファイル/ディレクトリをトラッキング |
| `dvc push` | リモートへアップロード |
| `dvc pull` | リモートから取得 |
| `dvc status` | 変更状況確認 |

---

## 2. pre-commit（コミット前チェック）

### インストール

```bash
pip install pre-commit
cd Tmcmc202601
pre-commit install
```

### 初回実行（全ファイル）

```bash
pre-commit run --all-files
```

### フック内容（.pre-commit-config.yaml）

- **ruff**: Lint + 自動修正（line-length=100）
- **black**: フォーマット（line-length=100）
- **pre-commit-hooks**: trailing whitespace, large file チェック, debug 文検出

### スキップ（緊急時）

```bash
git commit -m "..." --no-verify
```

---

## 3. Zotero（文献管理）

### セットアップ

1. **インストール**: https://www.zotero.org/download/
2. **ブラウザ連携**: Zotero Connector を Chrome/Firefox に導入
3. **BibTeX 連携**: 設定 → 引用 → BibTeX を有効化

### Tmcmc202601 との連携

論文の `references_ikm.bib` を Zotero からエクスポート可能：

1. Zotero でコレクション作成（例: Tmcmc202601）
2. 右クリック → コレクションのエクスポート
3. 形式: **BibTeX**、文字コード: **UTF-8**
4. `data_5species/docs/references_ikm.bib` にマージ

### 推奨プラグイン

| プラグイン | 用途 |
|------------|------|
| Better BibTeX | 自動 citation key、BibTeX 同期 |
| Zotero PDF Translate | PDF 内翻訳 |
| scite | 引用コンテキスト表示 |

---

## 4. Hydra（設定管理）

### インストール

```bash
pip install hydra-core
```

### 設定ファイル例

`configs/gnn_train.yaml` を参照。既存の argparse を Hydra に移行する場合：

```python
# 例: gnn/train.py の Hydra 対応
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

def main():
    with initialize_config_dir(config_dir="configs", version_base=None):
        cfg = compose(config_name="gnn_train")
        # cfg.data.path, cfg.model.hidden, ...
```

### オーバーライド（CLI）

```bash
python train.py model.hidden=256 train.epochs=500
```

### 段階的導入

既存の `argparse` を残しつつ、`--config-path` で Hydra をオプション有効にする方式も可能。

---

## 5. MLflow（実験トラッキング）

### インストール

```bash
pip install mlflow
```

### ローカル UI 起動

```bash
cd Tmcmc202601
mlflow ui
# http://localhost:5000
```

### GNN 学習への組み込み例

```python
import mlflow

def main_train(args):
    mlflow.set_experiment("gnn_aij")
    with mlflow.start_run():
        mlflow.log_params({
            "hidden": args.hidden,
            "layers": args.layers,
            "lr": args.lr,
            "dropout": args.dropout,
        })
        for epoch in range(args.epochs):
            train_loss = train_epoch(...)
            val_loss = validate(...)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        mlflow.pytorch.log_model(model, "model")
```

### リモートサーバー（オプション）

```bash
# バックエンド + アーティファクトストア
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

環境変数で指定：

```bash
export MLFLOW_TRACKING_URI=http://gpu-server:5000
python gnn/train.py ...
```

---

## 6. 依存関係まとめ

| ツール | pip パッケージ | 備考 |
|-------|----------------|------|
| DVC | `dvc` | オプション: dvc[s3], dvc[gdrive] |
| pre-commit | `pre-commit` | フックに ruff, black 含む |
| Zotero | — | デスクトップアプリ |
| Hydra | `hydra-core` | OmegaConf 同梱 |
| MLflow | `mlflow` | PyTorch 連携: mlflow[pytorch] |

### requirements-tools.txt（新規）

```
dvc>=3.0
pre-commit>=3.0
hydra-core>=1.3
mlflow>=2.0
```

---

## 7. 導入順序の推奨

1. **pre-commit** — 即効性あり、コード品質向上
2. **DVC** — データ・チェックポイントの肥大化対策
3. **MLflow** — GNN / DeepONet の実験比較
4. **Hydra** — 設定の一元化（既存 argparse と併用可）
5. **Zotero** — 論文執筆時に BibTeX 連携
