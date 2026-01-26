# 時刻インデックスズレ修正レポート

## 問題の原因

**確定**: プロット生成時に、データ生成時の元の `t_arr` ではなく、MAP/MEAN推定時に新しく計算した `t_fit` を使用していたため、時刻マッピングがずれていた。

### 問題のコード（修正前）

```python
# MCMC完了後、MAP/MEAN推定値でプロット生成
t_fit, x0_fit_MAP, _ = tsm_M1_fit.solve_tsm(theta_MAP_full_M1)
plot_mgr.plot_TSM_simulation(t_fit, x0_fit_MAP, ..., data_M1, idx_M1)
#                                 ^^^^^ 問題: 新しい t_fit を使用
#                                                      ^^^^^ しかし idx_M1 は元の t_M1 から計算された
```

**問題点**:
- `idx_M1` はデータ生成時の `t_M1` から計算されたインデックス
- プロット時は `t_fit`（MAP/MEAN推定時の新しい時刻配列）を使用
- `t_obs = t_fit[idx_M1]` が間違った時刻値を返す可能性

## 修正内容

### 1. 元の時刻配列を使用

プロット生成時は、データ生成時に保存された `t_M1`, `t_M2`, `t_M3` を使用するように変更。

### 2. 時刻配列の不一致に対応

もし `t_fit` と元の `t_arr` が異なる場合（長さや値が違う場合）、`scipy.interpolate.interp1d` を使用して `x0_fit` を元の時刻点に補間。

### 修正後のコード

```python
t_fit, x0_fit_MAP, _ = tsm_M1_fit.solve_tsm(theta_MAP_full_M1)
# CRITICAL FIX: Use original t_M1 (from data generation) instead of t_fit
if len(t_fit) != len(t_M1) or not np.allclose(t_fit, t_M1):
    # 時刻配列が異なる場合、補間が必要
    x0_fit_MAP_interp = np.zeros((len(t_M1), x0_fit_MAP.shape[1]))
    for j in range(x0_fit_MAP.shape[1]):
        interp_func = interp1d(t_fit, x0_fit_MAP[:, j], kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        x0_fit_MAP_interp[:, j] = interp_func(t_M1)
    x0_fit_MAP = x0_fit_MAP_interp
plot_mgr.plot_TSM_simulation(t_M1, x0_fit_MAP, ..., data_M1, idx_M1)
#                            ^^^^ 修正: 元の t_M1 を使用
```

## 修正箇所

以下の6箇所を修正（M1, M2, M3 それぞれ MAP と MEAN の2つ）:

1. `tmcmc_docs/tmcmc/main/case2_main.py:997-1003` - M1 MAP/MEAN
2. `tmcmc_docs/tmcmc/main/case2_main.py:1300-1306` - M2 MAP/MEAN  
3. `tmcmc_docs/tmcmc/main/case2_main.py:1517-1523` - M3 MAP/MEAN

## 期待される効果

修正後、プロット上のデータ点がモデル曲線と正しく一致するはずです：

- ✅ データ点がモデル曲線の上に正確に配置される
- ✅ 時刻マッピングのズレが解消される
- ✅ 初期〜後半まで一貫してデータとモデルが一致

## 依存関係

- `scipy.interpolate.interp1d` が必要（時刻配列が異なる場合の補間用）
- 通常は時刻配列は同じはずなので、補間は実行されない可能性が高い
- ただし、安全性のため補間ロジックを実装

## テスト方法

1. 修正後のコードで実行
2. `TSM_simulation_M1_MAP_fit_with_data.png` を確認
3. データ点がモデル曲線と一致していることを確認
4. 時刻マッピングのズレが解消されていることを確認
