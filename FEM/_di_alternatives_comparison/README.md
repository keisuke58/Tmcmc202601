# DI Alternative Models Comparison

学術的妥当性順に実装した DI 代替モデルの比較結果。

## 実装順序（学術的妥当性）

1. **Shannon DI** (既存) — 生態学の標準、情報エントロピー
2. **Simpson DI** — dominance に敏感、生態学で広く使用
3. **Voigt mixing** — 有効媒質理論、変分整合的（Rule of mixtures）
4. **Gini evenness** — 不均等度の直接表現
5. **Pielou evenness** — 種数に正規化された均等度
6. **Reuss mixing** — 直列負荷の下限
7. **φ_Pg, Virulence** — メカニズムベース（参考）

## 結果サマリ（Synthetic: Commensal vs Dysbiotic S.oralis）

| Model | E_comm [Pa] | E_dysb [Pa] | Ratio | 方向 |
|-------|-------------|-------------|-------|------|
| Gini | 1000 | 166 | **6.0×** | ✓ |
| Simpson | 1000 | 197 | **5.1×** | ✓ |
| Shannon | 1000 | 239 | 4.2× | ✓ |
| Pielou | 1000 | 239 | 4.2× | ✓ |
| Voigt | 522 | 880 | 0.6× | ✗ |
| Reuss | 46 | 161 | 0.3× | ✗ |
| φ_Pg | 712 | 998 | 0.7× | ✗ |
| Virulence | 643 | 998 | 0.6× | ✗ |

## 解釈

- **DI-based (Shannon, Simpson, Gini, Pielou)**: 多様性低下 → E↓（正しい方向）
- **Voigt/Reuss**: 組成ベース。dysbiotic So は E_So が高いため E↑（多様性仮説と逆）
- **φ_Pg, Virulence**: 病原菌ベース。So 優占では φ_Pg が低く差別化不可

## 結論

多様性仮説（diversity loss → stiffness reduction）に基づく場合、**Gini が最も高い差別化能**を示す。Shannon と数学的に等価な Pielou は同程度。Simpson は dominance に敏感で Shannon よりやや高い差別化。

Voigt/Reuss は変分原理と整合的だが、種特異的弾性率 E_i に依存し、多様性ではなく組成を反映する。

## ファイル

- `fig_di_alternatives_comparison.png` — 比較図
- `comparison_results.json` — 数値結果
