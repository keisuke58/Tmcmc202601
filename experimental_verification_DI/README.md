# DI→E 実験検証プロジェクト

**目的**: Dysbiosis Index (DI) から弾性率 \(E\) への constitutive law を、同一サンプルでの実験データで直接検証する。

**現状**: 現象論的モデル。文献（Pattem 2018/2021, Gloag 2019, Ohmura 2024）との整合性のみで、組成と弾性率の同時測定データによる検証は未実施。

**目標**: 10/10 評価に向けた最重要項目。同一サンプルでの 16S rRNA + AFM 測定により、DI→E を直接検証する。

---

## フォルダ構成

```
experimental_verification_DI/
├── README.md           # 本ファイル（概要）
├── EXPERIMENTAL_PROTOCOL.md  # 16S + AFM 同時測定の実験プロトコル
├── CHECKLIST.md        # 実験検証チェックリスト
├── PLAN.md             # 詳細ロードマップ・Phase 1–3
├── METHODOLOGY.md      # 16S + AFM 同時測定の手法（Ohmura 2024 等）
├── ISSUE_TEMPLATE.md   # GitHub Issue 作成用テンプレート
├── data_loader.py      # (DI, E) CSV の読み込み・保存
├── fit_constitutive.py # E(DI) モデルのフィッティング
├── synthetic_data.py   # 合成データ生成（検証用）
├── run_validation.py   # 検証パイプライン実行
├── experiment_design.py  # サンプルサイズ推定（Phase 1）
├── literature_data.py   # 文献データ（Pattem, Gloag）
├── plot_E_DI_with_literature.py  # 論文 Fig 11 風プロット
├── data/               # 実験データ格納
│   ├── di_e_template.csv
│   └── README.md
├── tests/              # ユニットテスト
└── _validation_output/ # 実行結果（図・レポート）
```

---

## クイックスタート

```bash
# 合成データで検証パイプラインを実行
cd Tmcmc202601
python -m experimental_verification_DI.run_validation

# 実データでフィット（CSV を data/ に配置後）
python -m experimental_verification_DI.run_validation --data experimental_verification_DI/data/di_e_pairs.csv

# 4 条件（CS, CH, DS, DH）を模した合成データ
python -m experimental_verification_DI.run_validation --condition-aware

# サンプルサイズ推定（実験デザイン）
python -m experimental_verification_DI.run_validation --design

# 文献オーバーレイ図（論文 Fig 11 風）
python -m experimental_verification_DI.plot_E_DI_with_literature

# Makefile 経由
make exp-verify          # 検証実行
make exp-verify-design   # サンプルサイズ推定
make exp-verify-literature  # 文献整合性レポート + E(DI) 図
make exp-verify-test     # テスト

# テスト実行
python -m pytest experimental_verification_DI/tests/ -v
```

## クイックリンク

- **詳細プラン**: [PLAN.md](PLAN.md)
- **手法**: [METHODOLOGY.md](METHODOLOGY.md)
- **GitHub Issue**: [New Issue](https://github.com/keisuke58/Tmcmc202601/issues/new?template=experimental_verification_DI.md) から「Experimental Verification (DI→E)」テンプレートを選択して作成
- **全体ロードマップ**: `../docs/ROADMAP_10_10.md`

---

## 達成条件（10/10 の条件）

| 項目 | 達成条件 |
|------|----------|
| **実験検証** | DI→E の constitutive law が、同一サンプルでの実験データで検証されている |
| **予測の妥当性** | 未使用データでの予測精度が定量的に示されている |
