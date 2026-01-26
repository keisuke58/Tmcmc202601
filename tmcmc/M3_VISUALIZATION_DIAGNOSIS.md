# M3可視化問題の診断結果

## 実行したテスト

### 1. 自己一致テスト（`test_m3_self_consistency.py`）

**目的**: 同じ`theta_true`を使って、データ生成と可視化が一致するか確認

**結果**: ✅ **完全一致**

```
Step 3: 自己一致検証
  data vs phibar_dense[idx_sparse]:
    Max difference: 0.00e+00
    Mean difference: 0.00e+00
  ✅ PASS: data完全一致（期待通り）

最終判定
✅ PASS: 完全一致
  → データ生成と可視化は正しく実装されています
  → ズレが見える場合は「推定の問題」です
  → MAPやmeanを使った比較でズレるのは正常です
```

**結論**: データ生成と可視化のコードは正しく実装されています。

---

## 3つの原因の診断結果

### 原因①: 比較している「線」がM3の真のphibarじゃない

**診断結果**:
- ✅ `theta_true`で生成したphibar: 完全一致（期待通り）
- ⚠ MAPで生成したphibar: ズレる（正常、MAP ≠ theta_true）
- ⚠ 再計算したphibar: ズレる（正常、内部入力が違う）
- ⚠ φやψを描画: ズレる（正常、phibarではない）

**推奨事項**:
- データ生成時の図（`TSM_simulation_M3_with_data.png`）では、`theta_true`で生成したphibarを使用しているため、データ点と線が一致するはず
- MAP/meanフィットの図では、データ点と線がズレるのは正常（MAP/mean ≠ theta_true）

---

### 原因②: 観測点インデックスと描画インデックスの不一致

**診断結果**:
- ✅ すべてのインデックスが有効範囲内
- ✅ データ点とモデル値の差はノイズ範囲内（正常）

```
観測点インデックスの確認:
  idx_sparse shape: (20,)
  idx_sparse range: [37, 750]
  phibar_dense shape: (751, 4)
  phibar_dense length: 751
  ✅ すべてのインデックスが有効範囲内

データ点とモデル値の一致確認:
  Species 0-3: すべてノイズ範囲内（正常）
```

**結論**: インデックスの問題はありません。

---

### 原因③: M3内部でM1/M2を「真値でなく再計算」している

**診断結果**:
- ⚠ M1/M2が再計算された場合、phibarに差が生じる（正常）

**推奨事項**:
- データ生成時と描画時で`theta_base`が同じか確認
- M1/M2が再計算されていないか確認

---

## 重要な発見

### 1. データ生成時の図は正しい

`generate_synthetic_data`関数（`main/case2_main.py`の457行目）では：
```python
# CRITICAL FIX: Pass pre-computed phibar to ensure plot uses the same phibar as data generation
plot_mgr.plot_TSM_simulation(t_arr, x0, config["active_species"], name, data, idx_sparse, phibar=phibar)
```

**`phibar=phibar`を渡しているため、データ生成時に使ったphibarと同じphibarで描画されます。**

### 2. MAP/meanフィットの図ではズレるのは正常

MAP/meanフィットの図（`M3_MAP_fit`、`M3_MEAN_fit`）では：
- データ点: `theta_true`で生成
- 線: MAP/meanで生成

**MAP/mean ≠ theta_true**のため、ズレるのは正常です。

---

## 推奨される次のステップ

### 1. 実際の図を確認

`TSM_simulation_M3_with_data.png`を確認して：
- データ点と線が一致しているか確認
- もしズレている場合、何を描画しているか確認

### 2. コードの確認ポイント

1. **データ生成時の図**:
   - `generate_synthetic_data`で`phibar=phibar`を渡しているか確認 ✅
   - `plot_TSM_simulation`で`phibar`パラメータを使用しているか確認 ✅

2. **MAP/meanフィットの図**:
   - MAP/meanで生成した`x0`から`phibar`を計算しているか確認
   - データ点は`theta_true`で生成されているか確認

### 3. 問題の切り分け

もし`TSM_simulation_M3_with_data.png`でズレが見える場合：

1. **データ生成時のphibarを保存**:
   ```python
   np.save("phibar_M3_data_gen.npy", phibar)
   ```

2. **描画時に同じphibarを使用**:
   ```python
   phibar_saved = np.load("phibar_M3_data_gen.npy")
   plot_mgr.plot_TSM_simulation(..., phibar=phibar_saved)
   ```

3. **比較**:
   - 保存したphibarと描画時に計算したphibarを比較
   - 差があれば、原因を特定

---

## まとめ

### ✅ 確認できたこと

1. **自己一致テスト**: 完全一致 ✅
2. **データ生成と可視化のコード**: 正しく実装されている ✅
3. **インデックスの問題**: なし ✅

### ⚠️ 注意点

1. **MAP/meanフィットの図**: データ点と線がズレるのは正常（MAP/mean ≠ theta_true）
2. **データ生成時の図**: データ点と線が一致するはず（同じ`theta_true`を使用）

### 🎯 結論

**「データ点生成の時点でズレが大きく見える」場合の原因は、ほぼ確実に「推定の問題」ではなく「可視化 or 比較対象の問題」です。**

具体的には：
- データ生成時の図（`TSM_simulation_M3_with_data.png`）でズレが見える場合 → **可視化のバグ**
- MAP/meanフィットの図でズレが見える場合 → **正常（MAP/mean ≠ theta_true）**

---

## 参考

- `test_m3_self_consistency.py`: 自己一致テスト
- `diagnose_m3_visualization_issues.py`: 3つの原因の診断
- `main/case2_main.py`: データ生成と可視化の実装
- `visualization/plot_manager.py`: 可視化関数の実装
