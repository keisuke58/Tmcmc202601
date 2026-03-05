# 16S rRNA + AFM 同時測定による DI→E 検証手法

**参照**: Ohmura et al. (2024), Pattem et al. (2018, 2021)

---

## 1. 背景

現在の DI→E constitutive law は以下の形式で与えられる:

$$E = E_{\max}(1-r)^2 + E_{\min}\, r, \quad r = \mathrm{clamp}(\mathrm{DI}/s, 0, 1)$$

- \(E_{\max} \approx 909\) Pa（commensal）
- \(E_{\min} \approx 32\) Pa（dysbiotic）
- パラメータ範囲は Pattem 2018/2021, Gloag 2019 の文献値と整合

**課題**: 組成（→ DI）と弾性率を**同一サンプル**で測定したデータがなく、現象論的仮定に依存している。

---

## 2. Ohmura et al. (2024) の手法

- **目的**: 生きた 3D バイオフィルムで、多糖類分布と局所弾性率の空間相関を同時に可視化。
- **技術**: 蛍光プローブによる多糖類イメージング + AFM ナノインデンテーション。
- **応用可能性**: 5 種口腔バイオフィルムでは、16S rRNA による種組成定量を組み合わせる必要がある。

---

## 3. 提案する測定フロー

### 3.1 サンプル調製

- HOBIC 培地、4 条件（CS, CH, DS, DH）で 5 種口腔バイオフィルムを培養。
- 培養時間: 既存 TMCMC 較正と整合する時間点（例: 24 h, 48 h, 72 h）。

### 3.2 経路 A: 16S rRNA → DI

1. サンプルから DNA 抽出。
2. 16S rRNA アンプリコンシーケンス（V3–V4 等）。
3. 種組成 \(\varphi_i\)（So, An, Vd, Fn, Pg）を定量。
4. Dysbiosis Index を計算:
   $$\mathrm{DI} = 1 - \frac{H}{H_{\max}}, \quad H = -\sum_i \varphi_i \ln \varphi_i$$

### 3.3 経路 B: AFM ナノインデンテーション → E

1. **同一サンプル**（または同一培養から得た隣接サンプル）を AFM 用に調製。
2. ナノインデンテーション（Pattem 2018 と同様のプロトコル）。
3. 力-変位曲線から Young 弾性率 \(E\) を算出。

### 3.4 同一サンプル化の工夫

- **オプション 1**: 同一ウェル内の隣接領域を 16S 用と AFM 用に分ける。
- **オプション 2**: 複数ウェルを同一条件で培養し、一部を 16S、一部を AFM に用いる（バッチ内変動を考慮）。
- **オプション 3**: Ohmura 型の空間分解測定が可能であれば、同一視野で組成 + 弾性率を取得。

---

## 4. データ解析

- (DI, E) ペアをプロットし、\(E = E_{\max}(1-r)^2 + E_{\min}\, r\) でフィット。
- 既存パラメータ（\(E_{\max}\), \(E_{\min}\), \(s\)）との一致度を評価。
- RMSE、相関係数、信頼区間を報告。

---

## 5. 参考文献

- Ohmura et al. (2024). Soft Matter. Spatially resolved correlation...
- Pattem et al. (2018). Scientific Reports. AFM nanoindentation: low-sucrose 14.35 kPa vs high-sucrose 0.55 kPa.
- Pattem et al. (2021). Hydrated conditions: 10.4 vs 2.8 kPa.
- Gloag et al. (2019). Dual-species biofilm storage modulus \(G' = 160\) Pa.
