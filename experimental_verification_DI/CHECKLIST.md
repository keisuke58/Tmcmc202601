# 実験検証 チェックリスト

**目標**: DI→E constitutive law の実験的検証（10/10 達成に必須）

---

## Phase 1: 準備（コード側・完了済み）

- [x] 文献データの整理（Pattem, Gloag）
- [x] 実験デザイン（サンプル数推定: n≥20 推奨）
- [x] データフォーマット・テンプレート
- [x] フィッティングパイプライン（`make exp-verify`）
- [x] 合成データでの検証（R²=0.989, e_max/e_min 誤差 5% 以内）

## Phase 2: 実験準備（実験担当者）

- [ ] 文献調査の深化（Ohmura 2024, Pattem プロトコル詳細）
- [ ] 既存 HOBIC データとの整合性確認
- [ ] 共同研究者・実験施設の確保
- [ ] 試薬・基質の調達

## Phase 3: データ取得

- [ ] サンプル調製（4 条件 × 5 サンプル以上）
- [ ] 16S rRNA シーケンス
- [ ] AFM ナノインデンテーション
- [ ] (DI, E) ペアの記録・CSV 保存

## Phase 4: 解析・論文反映

- [ ] `run_validation --data data/di_e_pairs.csv` 実行
- [ ] 検証レポートの確認
- [ ] 論文 Methods に実験検証を追加
- [ ] Limitations の更新

---

## クイックスタート（実データ取得後）

```bash
# 1. データを配置
cp your_data.csv experimental_verification_DI/data/di_e_pairs.csv

# 2. 検証実行
cd Tmcmc202601 && make exp-verify

# 3. 出力確認
# _validation_output/validation_report.json
# _validation_output/fig_E_DI_validation.png
```
