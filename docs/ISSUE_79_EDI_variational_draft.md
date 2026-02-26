# E(DI) の変分導出 — Lagrangian 候補の検討

> **Issue:** [#79](https://github.com/keisuke58/Tmcmc202601/issues/79)
> **参照:** Junker & Balzani 2021 (A1), Junker et al. 2025 (A5)

---

## 現状

- E(DI) = E_max(1-r)² + E_min·r, r = DI/DI_scale
- 経験的構成則として仮定。変分原理からの導出は未実施。

---

## 候補 1: DI を内部変数とする自由エネルギー

DI を秩序パラメータと見なし、Landau 型の自由エネルギーを導入:

$$\Psi(\mathrm{DI}) = \Psi_0 + \frac{a}{2}(\mathrm{DI} - \mathrm{DI}_*)^2 + \frac{b}{4}(\mathrm{DI} - \mathrm{DI}_*)^4$$

弾性率を「ひずみに対する 2 次微分」と解釈する場合、DI と ε の結合項が必要:

$$\Psi(\varepsilon, \mathrm{DI}) = \frac{1}{2} E(\mathrm{DI}) \varepsilon^2 + \Psi_{\mathrm{DI}}(\mathrm{DI})$$

ここで E(DI) は DI の関数として与え、変分原理から DI の発展方程式を導出する。

**課題:** DI の「力学」— DI は生態学 ODE の結果であり、独立した内部変数としての発展則が不明。

---

## 候補 2: 有効媒質としての解釈

バイオフィルムを「有効弾性媒質」と見なし、組成 φ から有効 E を混合則で与える:

$$E_{\mathrm{eff}} = \sum_i f(\varphi_i) E_i$$

Shannon entropy は組成の関数: DI = -Σ φ_i log φ_i。したがって E_eff = g(DI) は間接的に組成から決まる。

**変分との接続:** Hamilton 原理では φ が状態変数。E は φ の関数として E(φ) = E(DI(φ)) と書ける。これを弾性項の係数として Lagrangian に組み込む。

---

## 候補 3: 散逸ポテンシャルからの導出

Junker & Balzani の拡張 Hamilton 原理では、散逸汎関数 D が重要。DI が「構造の乱れ」を表すと解釈し、散逸が DI に依存すると仮定:

$$D = D(\dot{\varphi}, \dot{\psi}, \mathrm{DI})$$

高い DI → 多様 → 散逸が大きい（粘性が高い）? または逆?

**要検討:** バイオフィルム力学では、低 DI（dysbiosis）が soft に対応。散逸との関係は未整理。

---

## 次のステップ

1. Junker & Balzani 2021 の Section 2–3 を精読し、内部変数と弾性項の結合の具体例を確認
2. Klempt et al. 2025 の体積拘束・散逸の扱いと、E(DI) を追加する自然な挿入点を検討
3. 最小限の 1D 例（DI と ε の 2 変数）で変分から E(DI) 型の関係が導出できるか試算
