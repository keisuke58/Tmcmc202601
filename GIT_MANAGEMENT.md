# Git 管理方針

## 基本方針

### コミットするもの
- **ソースコード**: `.py`, `.sh`, `.md`, `.tex` など
- **設定ファイル**: `config.py`, `.gitignore` など
- **ドキュメント**: `docs/` 内の LaTeX ソース（`.tex`）と生成PDF（論文関連）
- **README / ドキュメント**: `PLAN.md`, `README.md` など

### コミットしないもの（`.gitignore` で除外）
- **実験結果**: `tmcmc/_runs/` (30MB+)
- **生成図表**: `tmcmc/_paper_figs_8_15/`, `tmcmc/*.png`, `tmcmc/*.pdf`
- **中間ファイル**: `tmcmc/case2_*_results/`, `tmcmc/*.csv`, `tmcmc/*.json`
- **Python キャッシュ**: `__pycache__/`, `*.pyc`
- **仮想環境**: `.venv/`, `venv/`

## コミット戦略

### 推奨: 機能単位のコミット
```bash
# 例: 新機能追加
git add tmcmc/new_feature.py tmcmc/config.py
git commit -m "feat: add new feature for X"

# 例: バグ修正
git add tmcmc/bugfix.py
git commit -m "fix: resolve issue with Y"

# 例: ドキュメント更新
git add docs/README.md PLAN.md
git commit -m "docs: update documentation"
```

### コミットメッセージの形式（推奨）
- `feat: 新機能`
- `fix: バグ修正`
- `docs: ドキュメント`
- `refactor: リファクタリング`
- `test: テスト追加`
- `chore: その他（設定変更など）`

## ブランチ戦略

### シンプル版（推奨）
- **`master` ブランチのみ**: 小規模プロジェクトならこれで十分
- 直接 `master` にコミット

### 機能開発版（必要に応じて）
- **`master`**: 安定版
- **`feature/xxx`**: 新機能開発用
- **`fix/xxx`**: バグ修正用

```bash
# 機能開発の例
git checkout -b feature/new-algorithm
# ... 開発 ...
git add .
git commit -m "feat: implement new algorithm"
git checkout master
git merge feature/new-algorithm
```

## 初回コミットの進め方

### ステップ1: 現在の状態を確認
```bash
git status
```

### ステップ2: コードファイルを追加（段階的に）
```bash
# コアコードから
git add tmcmc/*.py tmcmc/*.sh
git add tmcmc/config.py
git commit -m "feat: initial TMCMC codebase"

# ドキュメント
git add docs/*.tex PLAN.md
git commit -m "docs: add documentation"

# その他（必要に応じて）
git add Biofilm/ biofilm_project/
git commit -m "feat: add biofilm analysis code"
```

### ステップ3: 確認
```bash
git log --oneline
git status
```

## 日常的な運用

### 作業前
```bash
git status  # 変更確認
```

### 作業後
```bash
git add <変更ファイル>
git commit -m "feat: 変更内容"
```

### 定期的な確認
```bash
git log --oneline -10  # 最近のコミット確認
git diff HEAD~1        # 直前の変更確認
```

## 注意事項

1. **大きなファイルは避ける**: 100MB 以上のファイルは Git LFS を検討
2. **機密情報は含めない**: API キー、パスワードなど
3. **実験結果は別管理**: `tmcmc/_runs/` は git に入れず、必要なら別途バックアップ
4. **PDF は慎重に**: 論文PDFは保持するが、生成図は除外

## トラブルシューティング

### 間違って大きなファイルを追加してしまった
```bash
git rm --cached tmcmc/_runs/large_file.json
# .gitignore に追加してから再コミット
```

### コミットを取り消したい（まだ push していない場合）
```bash
git reset --soft HEAD~1  # コミット取り消し、変更は保持
# または
git reset --hard HEAD~1  # コミットと変更を完全に取り消し（注意！）
```
