# 実験データ格納ディレクトリ

16S rRNA + AFM 同時測定で得られた (DI, E) ペアをここに配置する。

## フォーマット

CSV 形式。必須列: `di`, `E`

| 列 | 説明 |
|----|------|
| di | Dysbiosis Index [0, 1] |
| E | Young 弾性率 [Pa] |
| E_err | 測定誤差（標準偏差）[Pa]（オプション）|
| condition | 培養条件（オプション）|
| sample_id | サンプル識別子（オプション）|

例: `di_e_template.csv` を参照

## 使い方

```bash
python -m experimental_verification_DI.run_validation --data data/di_e_pairs.csv
```
