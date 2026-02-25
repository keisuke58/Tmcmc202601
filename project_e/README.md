# Project E: VAE × TMCMC — Amortized Posterior Inference

TMCMC 事後分布を VAE で近似し、新データ y_obs に対して 1 回の forward pass で事後サンプルを生成する amortized inference を実現する。

## 目的

- **Phase 1**: 既存 TMCMC posterior（4条件分）を VAE で学習し、事後分布を圧縮・近似
- **Phase 2**: 合成データ拡張で amortized inference を訓練し、任意の y_obs に対して高速に θ をサンプリング

## パイプライン

```
y_obs (6×5 種組成時系列)  →  VAE Encoder  →  z (latent)
                                              ↓
θ (20 params)  ←  VAE Decoder  ←  z
```

学習データ: 4条件 × (data.npy, samples.npy) = (y_obs, θ) ペア

## ディレクトリ構成

```
project_e/
├── README.md              # 本ファイル
├── load_posterior_data.py # TMCMC 事後データローダー
├── vae_model.py           # VAE モデル (PyTorch)
├── train.py               # 学習スクリプト
├── eval.py                # 評価・可視化
└── data/                  # 学習データ・チェックポイント
```

## データソース

| 条件 | data.npy | samples.npy |
|------|----------|-------------|
| Commensal_Static | commensal_static_posterior | commensal_static_posterior |
| Commensal_HOBIC | commensal_hobic_posterior | commensal_hobic_posterior |
| Dysbiotic_Static | dysbiotic_static_posterior | dysbiotic_static_posterior |
| Dysbiotic_HOBIC | dysbiotic_hobic_1000p | dh_baseline |

- y_obs: (6, 5) 正規化種組成（6日 × 5菌種）
- θ: (300, 20) 事後サンプル（条件あたり 300 粒子）

## セットアップ

```bash
pip install torch numpy
# or
pip install -r ../requirements-gnn.txt  # torch 含む
```

## 使い方

### Phase 1: 既存 posterior のみ

```bash
# 1. データ確認
python load_posterior_data.py --check

# 2. 学習 — 500 epochs 推奨
python train.py --epochs 500 --latent-dim 16 --batch-size 64

# 3. 評価・可視化
python eval.py --checkpoint data/checkpoints/best.pt --plot
```

### Phase 2: 合成データ拡張（amortized inference）

```bash
# 1. 合成データ生成（4条件 × 5000サンプル）
#    Dysbiotic_HOBIC は自動で事後サンプル 50% を使用（MAE 悪化対策）
python generate_synthetic_data.py --n-samples 5000 --all-conditions

# 2. 合成 + posterior で学習
python train.py --epochs 500 --synthetic data/synthetic_all_N20000.npz
```

**注意**: 50 epochs の短い学習では MAE が高くなる。500 epochs 以上で精度向上を期待。

**Dysbiotic_HOBIC MAE 悪化**: 合成 θ 分布が事後とずれる場合、`docs/ISSUE_Dysbiotic_HOBIC_MAE_degradation.md` 参照。`--posterior-frac 0.5` で事後ベース合成を増やせる。

## 参考

- `docs/ISSUE_Project_E_Data_Survey.md`: データ調査結果
- `gnn/generate_training_data.py`: 合成データ生成（Phase 2 用）
