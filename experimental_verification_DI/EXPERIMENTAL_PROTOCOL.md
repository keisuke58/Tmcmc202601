# 16S rRNA + AFM 同時測定 実験プロトコル

**目的**: 同一サンプルで種組成（→ DI）と弾性率 E を測定し、DI→E constitutive law を実験的に検証する。

**参照**: Ohmura et al. (2024) *Soft Matter* — 多糖類分布と局所弾性率の空間相関同時測定

---

## 1. サンプル調製

### 1.1 培養条件

| 条件 | 培養 | 状態 | 想定 DI 範囲 |
|------|------|------|-------------|
| CS | Static | Commensal | 0.05–0.20 |
| CH | HOBIC | Commensal | 0.10–0.25 |
| DS | Static | Dysbiotic | 0.60–0.85 |
| DH | HOBIC | Dysbiotic | 0.70–0.95 |

- **HOBIC**: Hannover Oral Biofilm Chamber（Kommerein 2018, Heine 2025）
- **5 種**: S. oralis, A. naeslundii, V. dispar, F. nucleatum, P. gingivalis

### 1.2 サンプル数

- **推奨**: 各条件 5 サンプル × 4 条件 = **20 サンプル**（最低 15）
- サンプルサイズ推定: `make exp-verify-design` で確認済み（n≥15 で RMSE≤100 Pa, R²≥0.9）

---

## 2. 測定フロー（同一サンプル）

```
サンプル（バイオフィルム付着基質）
    │
    ├─→ (A) 16S rRNA シーケンス
    │       - DNA 抽出
    │       - 16S V3–V4 リージョン PCR
    │       - Illumina MiSeq / NovaSeq
    │       - DADA2 等で OTU/ASV クラスタリング
    │       - 種組成 φ_i (5 種) → DI = 1 - H/H_max
    │
    └─→ (B) AFM ナノインデンテーション
            - 同一サンプルの別領域（または同一領域の非破壊測定後）
            - カンチレバー: 春定数 0.1–1 N/m（Pattem 2018 参照）
            - インデンテーション深さ: 500 nm–2 μm
            - 複数点測定（5–10 点/サンプル）→ 平均 E, 標準偏差
```

### 2.1 同一サンプル性の確保

- **オプション A**: サンプルを 2 分割し、半分を 16S、半分を AFM に（隣接領域で組成は近似）
- **オプション B**: AFM 測定後、同一領域から DNA 抽出（破壊的）
- **オプション C**: 並列培養で同一条件の複数サンプルを用意し、16S 用と AFM 用に分ける（条件レベルでの対応）

---

## 3. データ記録フォーマット

### 3.1 必須列（CSV）

| 列 | 説明 | 単位 |
|----|------|------|
| di | Dysbiosis Index | [0, 1] |
| E | Young 弾性率 | Pa |
| E_err | 測定誤差（標準偏差）| Pa（オプション）|
| condition | 培養条件 | CS, CH, DS, DH |
| sample_id | サンプル識別子 | 例: DH_S01 |

### 3.2 テンプレート

`data/di_e_pairs.csv` に配置。テンプレートは `data/di_e_template.csv` を参照。

```csv
di,E,E_err,condition,sample_id
0.05,850.2,120.5,commensal_static,S01
0.82,45.1,8.2,dysbiotic_static,S02
```

---

## 4. 解析パイプライン

実データ取得後:

```bash
# フィッティングと検証レポート
make exp-verify
# または
python -m experimental_verification_DI.run_validation --data experimental_verification_DI/data/di_e_pairs.csv
```

---

## 5. 参考文献

- Ohmura, Wakita, Nishimura (2024). *Soft Matter*. 多糖類–弾性率の空間相関
- Pattem et al. (2018). *ACS Biomater. Sci. Eng.* AFM ナノインデンテーション
- Pattem et al. (2021). *Langmuir*. 水和バイオフィルムの弾性
- Gloag et al. (2019). *mBio*. デュアル種バイオフィルムのレオロジー
- Heine et al. (2025). *Peri-implantitis*. HOBIC 5 種データ
- Kommerein et al. (2018). HOBIC フロー chamber

---

## 6. チェックリスト（実験担当者向け）

- [ ] 培養条件 4 種（CS, CH, DS, DH）のサンプル調製
- [ ] 各条件 5 サンプル以上（推奨 20 サンプル）
- [ ] 16S rRNA シーケンス（5 種の組成取得）
- [ ] AFM ナノインデンテーション（弾性率 E 取得）
- [ ] 同一サンプルまたは同一条件での (DI, E) ペア記録
- [ ] CSV で `data/di_e_pairs.csv` に保存
- [ ] `make exp-verify` でフィット検証
