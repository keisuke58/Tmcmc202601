# リファクタリングプラン

## 📋 現状分析

### 主要な問題点

1. **巨大な単一ファイル**
   - `case2_tmcmc_linearization.py`: **5,586行**
   - 複数の責務が混在（MCMC実装、可視化、設定、ユーティリティ）

2. **クラス・関数の混在**
   - `DebugLogger` (206行目〜)
   - `MCMCConfig` (524行目〜)
   - `ExperimentConfig` (542行目〜)
   - `PlotManager` (1,167行目〜)
   - `LogLikelihoodEvaluator` (1,692行目〜)
   - `TMCMCResult` (2,140行目〜)
   - 複数のMCMC実装関数（`run_TMCMC`, `run_adaptive_MCMC`, `run_multi_chain_TMCMC`など）

3. **重複コードの可能性**
   - 類似ファイル: `case2_tmcmc_refined_JIT.py`, `improved1207_paper_jit.py`
   - 設定の重複（`config.py`と`case2_tmcmc_linearization.py`内の定数）

4. **依存関係の複雑さ**
   - 多数のimport（29個のimport文）
   - 循環依存のリスク

---

## 🎯 リファクタリング目標

1. **モジュール化**: 単一責任の原則に基づいてファイルを分割
2. **再利用性向上**: 共通機能を独立したモジュールに抽出
3. **保守性向上**: コードの理解と変更を容易に
4. **テスト容易性**: 各モジュールを独立してテスト可能に

---

## 📦 提案する新しい構造

```
tmcmc/
├── __init__.py
├── config.py                    # ✅ 既存（設定管理）
├── case2_tmcmc_linearization.py  # ⚠️ リファクタリング対象
│
├── core/                       # 🆕 コア機能
│   ├── __init__.py
│   ├── tmcmc.py               # TMCMCアルゴリズム実装
│   ├── mcmc.py                # 汎用MCMC実装（adaptive MCMCなど）
│   └── evaluator.py           # LogLikelihoodEvaluator
│
├── utils/                      # 🆕 ユーティリティ
│   ├── __init__.py
│   ├── io.py                   # ファイルI/O（save_json, write_csv, etc.）
│   ├── timing.py              # TimingStats
│   ├── health.py              # LikelihoodHealthCounter
│   └── validation.py          # 入力検証関数
│
├── visualization/              # 🆕 可視化
│   ├── __init__.py
│   ├── plot_manager.py        # PlotManager
│   └── diagnostics.py         # 診断プロット関数
│
├── debug/                      # 🆕 デバッグ機能
│   ├── __init__.py
│   ├── logger.py              # DebugLogger
│   └── events.py              # イベントロギング
│
└── main/                       # 🆕 エントリーポイント
    ├── __init__.py
    └── case2_main.py          # main()関数とCLI処理
```

---

## 🔄 リファクタリング手順

### Phase 1: ユーティリティの抽出（低リスク）

**優先度: 高 | リスク: 低**

1. **`utils/io.py`** に以下を移動:
   - `_save_npy()`
   - `_save_likelihood_meta()`
   - `save_json()`
   - `write_csv()`
   - `_code_crc32()`

2. **`utils/timing.py`** に以下を移動:
   - `TimingStats` クラス
   - `timed()` コンテキストマネージャー

3. **`utils/health.py`** に以下を移動:
   - `LikelihoodHealthCounter` クラス

4. **`utils/validation.py`** に以下を移動:
   - `_validate_tmcmc_inputs()`
   - その他の検証関数

**期待される効果:**
- 約300-400行のコードを分離
- 他のスクリプトからも再利用可能

---

### Phase 2: デバッグ機能の分離（中リスク）

**優先度: 高 | リスク: 中**

1. **`debug/logger.py`** に以下を移動:
   - `DebugLogger` クラス（206-523行目）
   - Slack通知関連のコード（131-167行目）

2. **`debug/events.py`** に以下を移動:
   - イベントロギング関連のヘルパー関数

**期待される効果:**
- 約400行のコードを分離
- デバッグ機能の独立性向上

---

### Phase 3: 可視化機能の分離（低リスク）

**優先度: 中 | リスク: 低**

1. **`visualization/plot_manager.py`** に以下を移動:
   - `PlotManager` クラス（1,167-1,691行目）
   - プロット関連のヘルパー関数

**期待される効果:**
- 約500行のコードを分離
- 可視化ロジックの独立性向上

---

### Phase 4: コアMCMC実装の分離（高リスク）

**優先度: 高 | リスク: 高**

1. **`core/evaluator.py`** に以下を移動:
   - `LogLikelihoodEvaluator` クラス（1,692-2,037行目）

2. **`core/tmcmc.py`** に以下を移動:
   - `run_TMCMC()` 関数（2,316-3,479行目）
   - `run_multi_chain_TMCMC()` 関数（3,547-3,796行目）
   - `TMCMCResult` データクラス（2,140-2,160行目）
   - TMCMC関連のヘルパー関数:
     - `reflect_into_bounds()`
     - `choose_subset_size()`
     - `should_do_fom_check()`

3. **`core/mcmc.py`** に以下を移動:
   - `run_adaptive_MCMC()` 関数（2,043-2,139行目）
   - `run_two_phase_MCMC_with_linearization()` 関数（3,802-3,944行目）
   - `MCMCConfig` データクラス（524-541行目）

**期待される効果:**
- 約1,500行のコードを分離
- MCMCアルゴリズムの独立性向上
- テスト容易性の向上

**注意点:**
- 依存関係の管理が重要
- 段階的にテストしながら進める

---

### Phase 5: メイン関数の分離（中リスク）

**優先度: 中 | リスク: 中**

1. **`main/case2_main.py`** に以下を移動:
   - `main()` 関数（4,003-5,560行目）
   - `parse_args()` 関数（602-680行目）
   - `ExperimentConfig` データクラス（542-589行目）
   - その他のヘルパー関数:
     - `select_sparse_data_indices()`
     - `log_likelihood_sparse()`
     - `compute_phibar()`
     - `compute_MAP_with_uncertainty()`
     - `generate_synthetic_data()`

**期待される効果:**
- 約1,500行のコードを分離
- エントリーポイントの明確化

---

### Phase 6: 設定の整理（低リスク）

**優先度: 低 | リスク: 低**

1. **`config.py`** に以下を統合:
   - `case2_tmcmc_linearization.py` 内の定数（172-200行目）を削除
   - `config.py` の既存設定を活用

**期待される効果:**
- 設定の一元管理
- 重複の削減

---

## 📊 リファクタリング後のファイルサイズ見積もり

| ファイル | 現在 | リファクタリング後 | 削減 |
|---------|------|------------------|------|
| `case2_tmcmc_linearization.py` | 5,586行 | ~500行（orchestrationのみ） | -91% |
| `core/tmcmc.py` | - | ~1,200行 | - |
| `core/mcmc.py` | - | ~400行 | - |
| `core/evaluator.py` | - | ~350行 | - |
| `visualization/plot_manager.py` | - | ~500行 | - |
| `debug/logger.py` | - | ~400行 | - |
| `utils/*.py` | - | ~400行（合計） | - |
| `main/case2_main.py` | - | ~1,500行 | - |

---

## ✅ リファクタリングのベストプラクティス

### 1. 段階的アプローチ
- 一度にすべてを変更しない
- 各Phaseを完了してから次へ進む
- 各Phase後にテストを実行

### 2. テスト戦略
- 既存のテストが通ることを確認
- 新しいモジュールごとにユニットテストを追加
- 統合テストで全体の動作を確認

### 3. 後方互換性
- 既存のimportパスを維持（`__init__.py`で再エクスポート）
- 段階的に非推奨警告を追加
- ドキュメントを更新

### 4. コードレビュー
- 各Phase完了後にレビュー
- 依存関係の確認
- パフォーマンス影響の評価

---

## 🚀 実装の優先順位

### 即座に実施（Phase 1-2）
- ✅ ユーティリティの抽出（低リスク、高効果）
- ✅ デバッグ機能の分離（中リスク、高効果）

### 短期（Phase 3-4）
- ✅ 可視化機能の分離
- ✅ コアMCMC実装の分離（注意深く）

### 中期（Phase 5-6）
- ✅ メイン関数の分離
- ✅ 設定の整理

---

## 📝 注意事項

1. **既存の動作を維持**
   - リファクタリング中も既存の機能が動作することを確認
   - 回帰テストを実行

2. **依存関係の管理**
   - 循環依存を避ける
   - `__init__.py`で適切にエクスポート

3. **ドキュメントの更新**
   - 新しい構造の説明
   - 移行ガイドの作成

4. **パフォーマンス**
   - import時間への影響を最小化
   - 必要に応じて遅延importを検討

---

## 🔍 追加の改善提案

### 1. 型ヒントの強化
- すべての関数・メソッドに型ヒントを追加
- `mypy`での型チェックを有効化

### 2. ドキュメント文字列の統一
- Google/NumPyスタイルのdocstringを統一
- APIドキュメントの自動生成

### 3. エラーハンドリングの改善
- カスタム例外クラスの導入
- エラーメッセージの明確化

### 4. ロギングの統一
- ロギングレベルの統一
- 構造化ログの検討

---

## 📅 推定工数

| Phase | 工数 | リスク |
|-------|------|--------|
| Phase 1: ユーティリティ | 2-3時間 | 低 |
| Phase 2: デバッグ機能 | 2-3時間 | 中 |
| Phase 3: 可視化 | 2-3時間 | 低 |
| Phase 4: コアMCMC | 4-6時間 | 高 |
| Phase 5: メイン関数 | 3-4時間 | 中 |
| Phase 6: 設定整理 | 1-2時間 | 低 |
| **合計** | **14-21時間** | - |

---

## 🎯 成功基準

1. ✅ すべての既存テストが通過
2. ✅ ファイルサイズが適切な範囲（<1000行/ファイル）
3. ✅ モジュール間の依存関係が明確
4. ✅ コードの可読性が向上
5. ✅ 新機能の追加が容易になる

---

## 📚 参考資料

- [Python Packaging Guide](https://packaging.python.org/)
- [Clean Code Principles](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- [Refactoring: Improving the Design of Existing Code](https://refactoring.com/)

---

**作成日**: 2025-01-XX  
**最終更新**: 2025-01-XX  
**ステータス**: 📋 計画段階
