# TSM サロゲートとのハイブリッド — 役割分担案

> **Issue:** [#78](https://github.com/keisuke58/Tmcmc202601/issues/78)
> **参照:** Fritsch et al. 2025 (arXiv:2512.15145), Geisler et al. 2025 (D2)

---

## 現状の比較

| 項目 | Fritsch et al. 2025 | 本研究 (Nishioka) |
|------|---------------------|-------------------|
| サロゲート | TSM (Time-Separated Stochastic Mechanics) | DeepONet |
| 種数 | 2種・4種 | 5種 |
| 利点 | 理論的保証、追加コストほぼゼロ | 3.8M× 高速化、勾配取得可能 |
| 課題 | 高次元・強非線形で精度低下の可能性 | ブラックボックス、外挿リスク |

---

## ハイブリッド設計案

### 案 A: 階層的 ROM

```
θ → [低次元・線形部分] → TSM (解析的) → φ_lin
                    → [残差・非線形] → DeepONet → φ_res
                    → φ = φ_lin + φ_res
```

- **TSM が得意:** パラメータ摂動に対する平均・分散の 1 次 Taylor 伝播
- **DeepONet が得意:** 非線形・高次項、Hill ゲート等

### 案 B: 条件分岐

```
if θ が prior の中心付近:
    → TSM で高速評価（解析的）
else:
    → DeepONet で評価
```

- 事後分布の大部分は prior 中心付近 → TSM でカバー
- 裾は DeepONet で補完

### 案 C: アンサンブル

```
log L = α · log L_TSM + (1-α) · log L_DeepONet
```

- α は検証データでチューニング
- 両者の長所を重み付きで結合

---

## 実装タスク（案 A を想定）

1. **TSM 部分の切り出し**
   - `BiofilmTSM5S` または `BiofilmTSM_Analytical` の線形化部分のみ使用
   - 出力: μ(θ), Σ(θ) の 1 次近似

2. **DeepONet の残差学習**
   - ターゲット: φ_full - φ_TSM_lin
   - TSM が捉えきれない非線形部分を学習

3. **TMCMC 統合**
   - 尤度評価で φ = φ_TSM + φ_res を使用
   - 勾配が必要な場合は DeepONet 部分のみ AD

---

## 参照コード

- TSM: `data_5species/core/` の `BiofilmTSM5S`, `BiofilmTSM_Analytical`
- DeepONet: `deeponet/surrogate_tmcmc.py`
- Fritsch et al.: arXiv:2512.15145（TSM をバイオフィルムの ROM として使用）
