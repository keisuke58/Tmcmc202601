# GitHub Actions ワークフロー一覧

## CI / 品質

| ワークフロー | トリガー | 説明 |
|-------------|----------|------|
| **Lint** | push/PR (`.py`) | Ruff, Black, pre-commit, mypy |
| **Test** | push/PR (`.py`) | pytest × Python 3.9/10/11, coverage |
| **CodeQL** | push/PR (main), 週次 | セキュリティスキャン |
| **Security** | push/PR (`.py`), 週次 | Bandit, pip-audit |
| **Commitlint** | PR | コミットメッセージ規約 (feat:, fix:) |
| **Python Check** | push/PR (`.py`) | 構文チェック |

## プロジェクト管理

| ワークフロー | トリガー | 説明 |
|-------------|----------|------|
| **Stale** | 週次, 手動 | 放置 Issue/PR のラベル・クローズ |
| **Release Drafter** | push/PR (main) | リリースノート自動生成 |
| **Auto Release** | タグ push (`v*`, `paper-*`) | PDF + 再現性ログをリリースに添付 |
| **Auto-label** | PR | 変更ファイルに応じてラベル付与 |
| **Welcome** | Issue/PR 作成時 | ウェルカムメッセージ |

## 科学計算向け

| ワークフロー | トリガー | 説明 |
|-------------|----------|------|
| **Reproducibility** | deps 変更, 月次 | pip freeze, 環境スナップショット |
| **Benchmark** | FEM/core 変更, 週次 | 実行時間記録 |
| **Figure Regression** | プロット関連 PR | 図のピクセル差分検出 |

## その他

| ワークフロー | トリガー | 説明 |
|-------------|----------|------|
| **LaTeX Build** | `.tex` 変更 | 論文 PDF ビルド |
