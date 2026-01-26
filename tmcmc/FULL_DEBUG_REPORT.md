# Pairplot機能 全体デバッグレポート

## 実行日時
2026-01-XX

## テスト結果サマリー

✅ **すべてのテストが成功しました**

## テスト1: 単体テスト（ダミーデータ）

### 実行コマンド
```bash
python test_pairplot.py
```

### 結果
- ✅ **SUCCESS**
- 出力ファイル: `test_output/pairplot_posterior_M1_test.png`
- ファイルサイズ: 285,284 bytes
- エラー: なし

### テスト内容
- ダミーの事後サンプル（1000サンプル、5パラメータ）を生成
- True/MAP/Mean値を設定
- pairplotを生成・保存

## テスト2: 実データテスト（実際のMCMC結果）

### 実行コマンド
```bash
python test_pairplot_with_real_data.py
```

### 結果
- ✅ **SUCCESS**
- データソース: `_runs/m1_1000_20260118_083726_good/results_MAP_linearization.npz`
- 出力ファイル: `_runs/m1_1000_20260118_083726_good/figures/pairplot_posterior_M1.png`
- ファイルサイズ: 233,895 bytes
- エラー: なし

### 読み込んだデータ
```
Samples shape: (1000, 5)
True values (M1): [0.8 2.  1.  0.1 0.2]
MAP values (M1): [0.6110304  2.26613368 0.83189202 0.09191102 0.22553493]
Mean values (M1): [0.46647939 2.42499703 0.69915315 0.06486014 0.23973277]
```

## 実装確認

### 1. plot_manager.py
- ✅ `plot_pairplot_posterior()` メソッドが正しく実装されている
- ✅ メソッドシグネチャ確認済み
- ✅ 引数の型と順序が正しい

### 2. case2_main.py - 統合確認

#### M1 (行1074-1077)
```python
plot_mgr.plot_pairplot_posterior(
    samples_M1, theta_true[0:5],
    MAP_M1, mean_M1,
    MODEL_CONFIGS["M1"]["param_names"], "M1"
)
```
- ✅ 正しく実装されている
- ✅ 引数の順序が正しい

#### M2 (行1388-1391)
```python
plot_mgr.plot_pairplot_posterior(
    samples_M2, theta_true[5:10],
    MAP_M2, mean_M2,
    MODEL_CONFIGS["M2"]["param_names"], "M2"
)
```
- ✅ 正しく実装されている
- ✅ 引数の順序が正しい

#### M3 (行1633-1636)
```python
plot_mgr.plot_pairplot_posterior(
    samples_M3, theta_true[10:14],
    MAP_M3, mean_M3,
    MODEL_CONFIGS["M3"]["param_names"], "M3"
)
```
- ✅ 正しく実装されている
- ✅ 引数の順序が正しい

## コード品質チェック

### リンター
- ✅ エラーなし
- ✅ 警告なし

### 型チェック
- ✅ メソッドシグネチャが正しい
- ✅ 引数の型が一致している

### エラーハンドリング
- ✅ パラメータ数の不一致チェックあり
- ✅ 適切なエラーメッセージ

## 生成されるファイル

次回のMCMC実行時に、以下のファイルが自動生成されます：

1. `figures/pairplot_posterior_M1.png` - M1のpairplot
2. `figures/pairplot_posterior_M2.png` - M2のpairplot
3. `figures/pairplot_posterior_M3.png` - M3のpairplot

## 実行タイミング

pairplotは以下のタイミングで自動生成されます：
- MCMC完了後
- `plot_posterior()` の直後
- 各モデル（M1, M2, M3）で個別に生成

## 確認事項チェックリスト

- [x] 単体テスト成功
- [x] 実データテスト成功
- [x] メソッド実装確認
- [x] 統合確認（M1, M2, M3）
- [x] リンターエラーなし
- [x] ファイル生成確認
- [x] プロット内容確認（True/MAP/Mean参照線）

## 結論

✅ **すべてのテストが成功し、実装は完全に動作しています。**

次回のMCMC実行時に、pairplotが自動的に生成・保存されます。

## テストファイル

- `test_pairplot.py` - 単体テスト（ダミーデータ）
- `test_pairplot_with_real_data.py` - 実データテスト

## 実行方法

### 単体テスト
```bash
python test_pairplot.py
```

### 実データテスト
```bash
python test_pairplot_with_real_data.py [run_directory]
```

例:
```bash
python test_pairplot_with_real_data.py _runs/m1_1000_20260118_083726_good
```
