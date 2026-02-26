# Tmcmc202601 魔改造計画

**作成日**: 2026-02-26
**目的**: コード品質・パフォーマンス・運用性の抜本的強化

---

## 1. 現状サマリ

| 領域 | 状態 | 課題 |
|------|------|------|
| **TMCMC** | 動作済み | 90h/1000p、並列化なし |
| **DeepONet** | ~80× 高速化 | CS/CH/DS の MAP 外挿誤差大 |
| **GNN** | Phase 1 完了 | Phase 2 HMP 統合、Phase 3 TMCMC 統合 |
| **Project E** | Phase 1 動作 | Dysbiotic_HOBIC MAE 悪化、合成データ拡張 |
| **FEM** | パイプライン完成 | hamilton_pde_jaxfem.py に多数 TODO |
| **CI** | py_compile のみ | テスト・lint・型チェックなし |

---

## 2. 魔改造ロードマップ

### Phase A: 基盤強化（1–2 週間）

#### A1. テスト・CI 拡張

- [ ] `pytest` 導入、`_tests/` にユニットテスト追加
  - `data_5species/core/tmcmc.py`: 事前分布サンプリング、重み計算
  - `data_5species/core/evaluator.py`: 尤度計算の数値検証
  - `FEM/material_models.py`: DI → E(x) マッピング
- [ ] GitHub Actions に `pytest`、`ruff`（または `flake8`）を追加
- [ ] `.cursorrules` 準拠: `black --line-length 100`、型ヒント推奨

#### A2. ロギング・設定の統一

- [ ] `print()` を `logging` に置換（`data_5species/`, `FEM/`）
- [ ] `config.py` または `model_config/` にマジックナンバー集約
- [ ] デバッグ用 `breakpoint()` / `pdb` の除去（CI で検出済み）

#### A3. 型ヒント・Docstring

- [ ] 公開 API に NumPy スタイル docstring 追加
- [ ] `typing` / `npt.NDArray` で主要関数に型ヒント
- [ ] `mypy` は numba/scipy を除外して段階導入

---

### Phase B: パフォーマンス（2–4 週間）

#### B1. TMCMC 並列化

- [ ] `evaluator.py` の尤度計算を `multiprocessing` または `joblib` で並列化
- [ ] 粒子ごとの ODE 評価をバッチ化（JAX `vmap` 検討）
- [ ] 目標: 1000 粒子 90h → 10–20h（4–8 コア想定）

#### B2. DeepONet 精度改善

- [ ] **Commensal_Static / HOBIC**: MAP 周辺 importance sampling 強化（実装済み preset の検証）
- [ ] **Dysbiotic_Static**: v2 checkpoint の本番適用
- [ ] 学習データ範囲外検出 → フォールバック ODE の自動切替

#### B3. JAX-FEM 時間積分

- [ ] `hamilton_pde_jaxfem.py` の TODO 解消:
  - Backward-Euler 時間積分ループ
  - Nutrient PDE カップリング
  - Volume constraint (φ₀ = 1 − Σφᵢ)

---

### Phase C: ML パイプライン統合（3–4 週間）

#### C1. GNN Phase 2 → Phase 3

- [ ] `download_hmp.py` TODO 解消、HMP oral 16S 取得・前処理
- [ ] `predict_hmp.py` → prior JSON 出力
- [ ] `estimate_reduced_nishioka.py` に `--use-gnn-prior --gnn-prior-json` 追加
- [ ] informed prior で TMCMC 収束速度検証

#### C2. Project E 安定化

- [ ] `docs/ISSUE_Dysbiotic_HOBIC_MAE_degradation.md` の対策実装
- [ ] `--posterior-frac` のチューニング、合成データ拡張の自動化
- [ ] 500 epochs 以上の学習をデフォルトに

#### C3. DeepONet × GNN ハイブリッド

- [ ] GNN prior → TMCMC → DeepONet サロゲートで E2E 高速化
- [ ] 新条件（HMP 由来）に対する amortized inference 検証

---

### Phase D: 運用・ドキュメント（1–2 週間）

#### D1. Makefile 拡張

```makefile
# 追加候補
test:        ## Run pytest
lint:        ## ruff check + black --check
format:      ## black . && ruff check --fix
repro:       ## Full pipeline (tmcmc → multiscale → eigenstrain)
```

#### D2. 環境の一元化

- [ ] `requirements.txt` を階層化:
  - `requirements-core.txt`: numpy, scipy（TMCMC 最小）
  - `requirements-jax.txt`: JAX, jax-fem（FEM/DeepONet）
  - `requirements-gnn.txt`: torch, torch-geometric（既存）
- [ ] `pyproject.toml` または `setup.cfg` でプロジェクトメタデータ統一

#### D3. 再現性ガイド強化

- [ ] `REPRODUCIBIITY.md` に GPU サーバー向け手順追加
- [ ] `docs/GITHUB_WORKFLOW.md`（IKM_Hiwi 版）との連携
- [ ] チェックポイント・結果のバージョン管理方針（Git LFS / DVC）

---

## 3. 優先度マトリクス

| 項目 | 影響 | 工数 | 優先度 |
|------|------|------|--------|
| A1 テスト・CI | 高 | 中 | **P0** |
| B1 TMCMC 並列化 | 高 | 高 | **P0** |
| A2 ロギング | 中 | 低 | P1 |
| C1 GNN Phase 3 | 高 | 中 | P1 |
| B2 DeepONet 精度 | 中 | 中 | P2 |
| B3 JAX-FEM TODO | 中 | 高 | P2 |
| C2 Project E | 中 | 低 | P2 |
| D1–D3 運用 | 中 | 低 | P2 |

---

## 4. 禁止事項（.cursorrules 準拠）

- 裸の `except:` ブロック
- 本番コードへの `print()` 残存
- マジックナンバー（定数化）
- 未使用インポート
- 500 行超の単一ファイル（分割検討）
- 循環依存

---

## 5. 参考リンク

- [ARCHITECTURE.md](../ARCHITECTURE.md) — モジュール依存関係
- [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) — 再現手順
- [deeponet/DEEPONET_ACHIEVEMENT_STATUS.md](../deeponet/DEEPONET_ACHIEVEMENT_STATUS.md) — DeepONet 現状
- [gnn/WIKI.md](../gnn/WIKI.md) — GNN 口頭試問対策
- [IKM_Hiwi docs/GITHUB_WORKFLOW.md](../../docs/GITHUB_WORKFLOW.md) — CPU↔GPU 運用
