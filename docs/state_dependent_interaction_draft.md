# 状態依存相互作用 A(pH, metabolites) 設計ドラフト

**目的**: P. gingivalis の終末 surge を改善するため、相互作用行列を環境変数に依存させる。

---

## 1. 背景

現状の Hamilton ODE では $\mathbf{A}$ は時間不変。しかし：
- 終末で pH 低下、代謝物蓄積
- P. gingivalis は低 pH で活性化（gingipain 等）
- F. nucleatum の代謝産物が P. gingivalis 増殖を促進

→ $\mathbf{A}$ を $\mathbf{A}(c, \mathrm{pH}, \mathbf{m})$ に拡張することで終末 surge を再現可能。

---

## 2. 提案形式

### 2.1 単純な拡張

$$A_{ij}(t) = A_{ij}^{(0)} \cdot f_{\mathrm{pH}}(\mathrm{pH}(t)) \cdot f_{\mathrm{met}}(\mathbf{m}(t))$$

- $f_{\mathrm{pH}}$: pH が低下すると Pg 関連の $A_{35}$, $A_{45}$ が増大
- $f_{\mathrm{met}}$: 乳酸等の蓄積で Vei→Pg 経路が活性化

### 2.2 閾値型

$$\mathbf{A}(t) = \mathbf{A}_{\mathrm{commensal}} + (\mathbf{A}_{\mathrm{dysbiotic}} - \mathbf{A}_{\mathrm{commensal}}) \cdot \sigma(\mathrm{pH} - \mathrm{pH}_{\mathrm{crit}})$$

- $\sigma$: シグモイド
- pH が閾値を下回ると regime shift

### 2.3 実装上の課題

- pH, 代謝物の時系列データが必要
- Heine et al. のデータに pH が含まれるか要確認
- パラメータ増加 → 同定可能性の低下

---

## 3. 次のステップ

1. Heine2025 の Supplementary で pH/代謝物データの有無を確認
2. ~~簡易版: 時間 $t$ に比例する $A_{35}(t)$, $A_{45}(t)$ で surge を再現できるか検証~~ → **実装済み**: `tools/test_time_dependent_A.py`（`make test-time-dependent-A`）
3. パイロット実装: `core_hamilton_1d` に `A_t-dependent` オプションを追加
