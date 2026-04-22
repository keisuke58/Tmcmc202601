# IKM_Hiwi 具体的実装計画

**作成日**: 2026-02-27  
**参照**: [AMBITIONS_2026.md](AMBITIONS_2026.md)

---

## 現状サマリ（実装計画用）

| 項目 | 現状 | 備考 |
|------|------|------|
| **TMCMC 並列化** | `n_jobs=12` で ProcessPoolExecutor 使用済み | estimate_reduced_nishioka.py L2448 |
| **print()** | 約 200 ファイル | 優先: core, main, FEM パイプライン |
| **裸 except** | 15 ファイル | 修正対象リストあり |
| **Makefile** | tmcmc, lint, check のみ | test, format, repro, paper 未実装 |
| **CI** | lint.yml, test.yml あり | test は FEM/tests のみ、data_5species 未含む |
| **GNN download_hmp** | スタブのみ | R/HMP16SData 統合 TODO |

---

## Phase 1: 基盤強化（1–2 週間）

### 1.1 print() → logging 置換

**優先順位**: P0（本番パイプライン）

| 順序 | 対象ディレクトリ/ファイル | 件数目安 | 作業 |
|------|---------------------------|----------|------|
| 1 | `data_5species/core/` | 2 | evaluator, nishioka_model |
| 2 | `data_5species/main/estimate_reduced_nishioka.py` | 31 | メイン推定スクリプト |
| 3 | `data_5species/main/` (他) | ~50 | 図生成・分析スクリプト |
| 4 | `FEM/` パイプライン系 | ~80 | multiscale, run_*, plot_* |
| 5 | `deeponet/`, `gnn/`, `project_e/` | ~40 | ML 系 |
| 6 | `tmcmc/program2602/` | ~100 | デバッグ・検証スクリプト（後回し可） |

**手順（1ファイルあたり）**:
1. ファイル先頭に `import logging` と `logger = logging.getLogger(__name__)`
2. `print("...")` → `logger.info("...")` または `logger.debug(...)` / `logger.warning(...)`
3. 進捗表示は `logger.info`、デバッグは `logger.debug`

**一括検索コマンド**:
```bash
rg "print\(" Tmcmc202601 --type py -l | head -50
```

---

### 1.2 裸 except 修正

**対象ファイル一覧**（15 箇所）:

| ファイル | 行 | 修正方針 |
|----------|-----|----------|
| `tmcmc/program2602/add_spaghetti_plot.py` | 120 | `except Exception as e:` + `logger.warning(...)` |
| `data_5species/docs/create_presentation.py` | 215 | 同上 |
| `deeponet/generate_fig22_posterior_comparison.py` | 88 | 同上 |
| `FEM/external_tooth_models/.../utils.py` | 555 | 外部→`except OSError` 等に限定 |
| `tmcmc/program2602/scan_m2_results.py` | 49 | `except (ValueError, KeyError) as e:` |
| `tmcmc/program2602/run_debug_M4_M5.py` | 148 | `except Exception as e:` |
| `tmcmc/program2602/find_sigma001_runs.py` | 35, 74 | 同上 |
| `tmcmc/program2602/check_execution_status.py` | 83, 146, 165, 186 | 同上 |
| `tmcmc/program2602/check_running.py` | 78 | 同上 |
| `tmcmc/program2602/check_process_output.py` | 45 | 同上 |
| `tmcmc/program2602/check_bg_status.py` | 103, 117 | 同上 |
| `data_5species/main/generate_report.py` | 337 | 同上 |
| `data_5species/main/generate_ideal_prediction.py` | 52 | 同上 |
| `data_5species/docs/pptx_to_pdf.py` | 40, 54, 68, 95, 140, 180, 223 | 同上 |

**修正テンプレート**:
```python
# Before
except:
    pass

# After
except (ValueError, KeyError, OSError) as e:
    logger.warning("Operation failed: %s", e)
```

---

### 1.3 Makefile 拡張

**追加ターゲット**:

```makefile
# Tmcmc202601/Makefile に追加

.PHONY: test format repro paper

test:  ## Run pytest (FEM + data_5species)
	$(PYTHON_TMCMC) -m pytest FEM/tests/ data_5species/ -v --tb=short -x
	$(PYTHON_TMCMC) -m pytest tmcmc/program2602/test_*.py -v --tb=short -x -k "not slow"

format:  ## Black + ruff fix
	black --line-length 100 .
	ruff check . --line-length 100 --fix

repro:  ## Full pipeline: tmcmc → multiscale → eigenstrain
	$(MAKE) tmcmc-quick
	$(MAKE) multiscale
	$(MAKE) hybrid
	$(MAKE) eigenstrain

paper:  ## Generate paper figures (LaTeX 前提)
	$(MAKE) all-figures
	cd data_5species/docs && latexmk -pdf
```

**既存 lint の拡張**:
```makefile
lint:  ## Ruff + Black check
	ruff check . --line-length 100
	black --check --line-length 100 .
```

---

### 1.4 CI 拡張

**test.yml に data_5species テスト追加**:

```yaml
# .github/workflows/test.yml に job 追加または既存修正

- name: Run data_5species tests
  run: |
    python -m pytest data_5species/ -v --tb=short -x \
      --ignore=data_5species/_runs/ \
      -k "not slow and not integration"
  timeout-minutes: 5
  continue-on-error: true  # 初回は失敗許容
```

**lint.yml**: 既に ruff, black, mypy あり。pytest は test.yml で実行。

---

## Phase 2: TMCMC 高速化（2–4 週間）

### 2.1 現状の並列化確認

- `run_TMCMC(..., n_jobs=12)` で ProcessPoolExecutor 使用
- ボトルネック: ODE 求解（TSM）が粒子ごとに独立 → 並列化済み
- 残り: **ODE 求解そのものの高速化**

### 2.2 実装タスク

| タスク | 内容 | ファイル | 工数目安 |
|--------|------|----------|----------|
| 2.2.1 | JAX vmap で ODE バッチ評価 | `data_5species/core/evaluator.py` | 3日 |
| 2.2.2 | improved_5species_jit の vmap 対応 | `tmcmc/improved_5species_jit.py` | 2日 |
| 2.2.3 | DeepONet サロゲートをデフォルトオプションに | `estimate_reduced_nishioka.py` | 0.5日 |
| 2.2.4 | チェックポイント自動保存間隔の短縮 | `TMCMCCheckpointManager` | 0.5日 |

**2.2.1 詳細**:
- `LogLikelihoodEvaluator.__call__` は 1 粒子ずつ呼ばれる
- `run_TMCMC` 内で `evaluate_particles_parallel` がバッチで呼ぶ
- 各ワーカーが 1 粒子の ODE を解く → ワーカー数分の並列
- **vmap 化**: theta の (n_particles, n_params) を一括で JAX に渡し、`jax.vmap(solve_ode)(theta_batch)` で GPU 並列

**2.2.2 前提**:
- `improved_5species_jit.py` が Numba JIT の場合、JAX 化が必要
- 既に JAX 版がある場合は、`jax.vmap` でラップ

---

### 2.3 中間目標（90h → 10h）

1. `n_jobs` を CPU 数に合わせて自動設定（現状 12 固定）
2. DeepONet サロゲートを `--use-deeponet` で全条件に適用
3. 粒子数 1000 → 500 で試行し、ESS が足りるか検証

---

## Phase 3: GNN Phase 2 → 3（2–3 週間）

### 3.1 download_hmp.py 実装

| ステップ | 内容 | 成果物 |
|----------|------|--------|
| 3.1.1 | R スクリプトで HMP16SData::V35() 取得 | `gnn/scripts/fetch_hmp_oral.R` |
| 3.1.2 | R → CSV export、Python で読み込み | `gnn/data/hmp_oral.csv` |
| 3.1.3 | 5 菌種への OTU マッピング | `gnn/otu_mapping.json` |
| 3.1.4 | `download_hmp.py` から R を subprocess 呼び出し | 1 コマンドで取得 |

**3.1.1 R スクリプト例**:
```r
# gnn/scripts/fetch_hmp_oral.R
if (!require("BiocManager")) install.packages("BiocManager")
BiocManager::install("HMP16SData", update=FALSE)
library(HMP16SData)
d <- V35()
meta <- as.data.frame(colData(d))
oral <- meta[grepl("oral", meta$HMP_BODY_SUBSITE, ignore.case=TRUE), ]
# ... export to CSV
```

### 3.2 predict_hmp.py → prior JSON

| ステップ | 内容 |
|----------|------|
| 3.2.1 | GNN で HMP oral サンプルの prior パラメータ予測 |
| 3.2.2 | `--output-prior-json` で JSON 出力 |
| 3.2.3 | `estimate_reduced_nishioka.py --use-gnn-prior --gnn-prior-json prior.json` で読み込み |

### 3.3 estimate_reduced_nishioka 拡張

```python
# 追加引数
parser.add_argument("--use-gnn-prior", action="store_true")
parser.add_argument("--gnn-prior-json", type=str, default=None)
```

- prior_bounds を JSON から読み込み、一様事前の代わりに使用
- または、正規事前の mean/cov を指定

---

## Phase 4: 論文完全再現（1–2 週間）

### 4.1 図表の自動生成

| 図 | 生成スクリプト | 出力 |
|-----|----------------|------|
| 図1 | `generate_pub_plots.py` | `fig1_*.png` |
| 図2 | `generate_extra_figures_generic.py` | 等 |
| 図3–4 | `extract_and_plot_fig*.py` | 等 |
| FEM 図 | `generate_pipeline_summary.py` | 等 |

**make paper で一括実行**:
```makefile
paper:
	$(MAKE) tmcmc-quick
	$(MAKE) all-figures
	cd data_5species/docs && latexmk -pdf nishioka_latex20260218.tex
```

### 4.2 数値の自動読み込み

- 表の数値は CSV から `\input` または Python で生成した `.tex` 断片を include
- 手動コピペを排除

---

## Phase 5: 運用・ドキュメント（1 週間）

### 5.1 requirements 階層化

```
Tmcmc202601/
├── requirements-core.txt   # numpy, scipy (TMCMC 最小)
├── requirements-jax.txt    # jax, jaxlib, equinox (FEM/DeepONet)
├── requirements-gnn.txt    # torch, torch-geometric (既存)
└── requirements-tools.txt  # 既存
```

### 5.2 pyproject.toml

```toml
[project]
name = "tmcmc-biofilm"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
]
```

### 5.3 REPRODUCIBILITY.md 拡張

- GPU サーバー向け手順
- Docker オプション（`Dockerfile` 追加）
- チェックポイントのバージョン管理方針

---

## 実装順序（推奨）

| 週 | タスク |
|----|--------|
| **W1** | 1.2 裸 except 修正、1.3 Makefile 拡張、1.4 CI 拡張 |
| **W2** | 1.1 print→logging（core + estimate_reduced_nishioka） |
| **W3** | 1.1 続き（FEM パイプライン）、2.3 中間目標 |
| **W4** | 2.2 JAX vmap 検討、3.1 download_hmp |
| **W5–6** | 3.2–3.3 GNN 統合、4.1 論文図表 |
| **W7** | 5.1–5.3 運用・ドキュメント |

---

## チェックリスト（週次）

- [ ] `make test` が通る
- [ ] `make lint` が通る
- [ ] `make format` でフォーマット統一
- [ ] 裸 except ゼロ
- [ ] core + main の print ゼロ
- [ ] CI 全 job 緑
