# GNN × 口腔菌叢相互作用ネットワーク (Project B, Issue #39)

HMP 16S データから菌種間相互作用 a_ij を GNN で予測し、TMCMC の informed prior として活用する。

## パイプライン

```
HMP 16S (菌叢組成) → co-occurrence network → GNN → a_ij 予測
                                                      ↓
                                              Hamilton ODE の prior として使う
                                              → informed TMCMC (faster convergence)
```

## ディレクトリ構成

```
gnn/
├── README.md                 # 本ファイル
├── generate_training_data.py # 合成データ生成 (θ → φ, a_ij)
├── graph_builder.py          # 組成 → PyG Data 変換
├── gnn_model.py              # GNN モデル (edge regression)
├── train.py                  # 学習スクリプト
├── download_hmp.py           # HMP データ取得 (Phase 2)
├── predict_hmp.py            # HMP 組成 → a_ij 予測 → prior JSON
├── scripts/extract_hmp_oral.R # HMP 口腔データ抽出 (R)
└── data/                     # 生成データ・チェックポイント
```

## フェーズ

### Phase 1: 合成データでブートストラップ (現在)

- TMCMC と同様に θ をサンプリング → Hamilton ODE で φ(t) を計算
- 各サンプル: ノード特徴 = 組成統計 (φ_mean, φ_std, ...), エッジラベル = a_ij
- GNN で a_ij を回帰予測
- **利点**: HMP データ不要で検証可能

### Phase 2: HMP データ統合

- HMP oral 16S をダウンロード・前処理
- co-occurrence network を構築
- 5 菌種へのマッピング (So, An, Vd, Fn, Pg)
- Phase 1 で学習した GNN を転移 or 再学習

### Phase 3: TMCMC 統合

- GNN の a_ij 予測 → prior の中心値として設定
- `--use-gnn-prior --gnn-prior-json` で informed TMCMC を実行

```bash
# 一括実行 (HMP デモ or 実データ)
./run_phase2_tmcmc.sh [species_abundance.csv]
```

## セットアップ

```bash
pip install torch torch-geometric numpy scipy
# or
pip install -r requirements-gnn.txt
```

## 使い方

### 推奨: 一括実行 (N=10k + 過学習対策)

```bash
./train_10k.sh
# → データ生成済みなら学習 → 終了後 eval_detailed 自動実行
```

### 手動実行

```bash
# 1. 合成データ生成 (10k samples)
python generate_training_data.py --n-samples 10000 --condition Dysbiotic_HOBIC

# 2. 学習 (dropout=0.2, weight_decay=1e-2, patience=100)
python train.py --data data/train_gnn_N10000.npz --epochs 1000 \
  --dropout 0.2 --weight-decay 1e-2 --patience 100

# 3. 評価 (per-edge 詳細)
./eval_detailed.sh data/train_gnn_N10000.npz data/checkpoints/best.pt
```

### 過学習対策 (N=1k で train≪val の場合)

- **データ増量**: N=10k 推奨
- **正則化**: `--dropout 0.2 --weight-decay 1e-2`
- **Early stopping**: `--patience 100`

## 参考

- Issue #39: 公開データ × ML プロジェクトアイデア
- **[SPECIES_MAPPING.md](SPECIES_MAPPING.md)**: 5 菌種マッピング表（HMP 16S → So, An, Vd, Fn, Pg）
- interaction_graph.json: 5 菌種の相互作用構造
- DeepONet: 同様の合成データ生成フロー
- HMP16SData: https://github.com/waldronlab/HMP16SData

## 口頭試問対策

- **[WIKI.md](WIKI.md)** — 査読・口頭試問対策（村松先生視点の想定質問含む）
