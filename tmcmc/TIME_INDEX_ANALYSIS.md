# 時刻インデックスズレ分析レポート

## 問題の症状

図 `TSM_simulation_M1_with_data.png` から観察される症状：
- **全データ点が同じ方向に少しだけ平行移動している**（ランダム誤差ではない）
- 点は曲線の形に沿っているが、**常に少し前/後ろの値を拾っている感じ**
- 初期〜中盤は合うが、後半で系統的にズレる

## 優先度付き原因分析

### ① サンプリング時刻（index ↔ time）のズレ 【最優先・確定的】

**根拠：**
- 全データ点が同じ方向に平行移動 → これはランダム誤差ではなく、**系統的なインデックスオフセット**
- 点が曲線の形に沿っている → モデル自体は正しいが、**時刻マッピングが1ステップずれている**
- 典型的な off-by-one エラー / 正規化ミス

**コード上の問題箇所：**

1. **`select_sparse_data_indices()` の実装** (`tmcmc_docs/tmcmc/main/case2_main.py:190-206`)
   ```python
   def select_sparse_data_indices(n_total: int, n_obs: int) -> np.ndarray:
       start_idx = int(0.1 * n_total)
       indices = np.linspace(start_idx, n_total - 1, n_obs)
       indices = np.floor(indices).astype(int)
   ```
   - `np.linspace(start, end, n)` は `[start, end]` の両端を含む
   - `np.floor()` で切り捨て → 最後のインデックスが `n_total - 1` になる
   - **問題の可能性**: `t_arr` の長さと `n_total` の定義が一致しているか？

2. **データ生成時のインデックス使用** (`tmcmc_docs/tmcmc/main/case2_main.py:361-369`)
   ```python
   idx_sparse = select_sparse_data_indices(len(t_arr), exp_config.n_data)
   phibar = compute_phibar(x0, config["active_species"])
   data[:, i] = phibar[idx_sparse, i] + rng.standard_normal(...)
   ```
   - `phibar[idx_sparse]` で直接インデックスアクセス
   - **問題の可能性**: `t_arr` と `x0` の長さが一致しているか？

3. **プロット時の時刻マッピング** (`tmcmc_docs/tmcmc/visualization/plot_manager.py:82`)
   ```python
   t_obs = t_arr[idx_sparse]
   ```
   - データ生成時とプロット時で同じ `idx_sparse` を使用
   - **問題の可能性**: `t_arr` のインデックスが0始まりか1始まりか？

**確認すべき点：**
- `t_arr` の長さ = `x0.shape[0]` か？
- `t_arr[0]` は 0.0 か、それとも最初の時間ステップの値か？
- `solve_tsm()` が返す `t_arr` のインデックス範囲は `[0, n_time-1]` か？

### ② 観測量の定義ズレ（φ vs φ̄=φψ） 【中優先】

**根拠：**
- 初期〜中盤は合うが、後半で系統的にズレる → ψ影響の典型
- しかし、**両種とも同じ方向にズレる**のは説明しにくい

**コード確認：**
- `compute_phibar()` は正しく実装されている（`tmcmc_docs/tmcmc/visualization/helpers.py:20-48`）
- データ生成時も `phibar = compute_phibar(x0, active_species)` を使用
- **問題の可能性は低い**が、念のため確認

### ③ 初期条件の微差 【低優先】

**根拠：**
- t≈0 近傍はほぼ合っている → 主因ではない

### ④ パラメータ不適合 【低優先】

**根拠：**
- 両種とも同じ方向にズレるのは説明しにくい
- パラメータ不適合なら、曲線の形自体が合わないはず

## 推奨される修正手順

### Step 1: インデックス整合性の確認

```python
# デバッグコードを追加
t_arr, x0, sig2 = tsm.solve_tsm(theta_true)
print(f"t_arr shape: {t_arr.shape}, x0 shape: {x0.shape}")
print(f"t_arr[0] = {t_arr[0]}, t_arr[-1] = {t_arr[-1]}")
print(f"t_arr length: {len(t_arr)}")

idx_sparse = select_sparse_data_indices(len(t_arr), exp_config.n_data)
print(f"idx_sparse: {idx_sparse}")
print(f"idx_sparse min: {idx_sparse.min()}, max: {idx_sparse.max()}")
print(f"t_arr[idx_sparse]: {t_arr[idx_sparse]}")

# データ生成時のインデックス範囲チェック
phibar = compute_phibar(x0, active_species)
print(f"phibar shape: {phibar.shape}")
print(f"phibar[idx_sparse] shape: {phibar[idx_sparse].shape}")
```

### Step 2: 時刻マッピングの検証

```python
# プロット時の時刻とデータ生成時の時刻が一致するか確認
t_obs_data_gen = t_arr[idx_sparse]  # データ生成時
t_obs_plot = t_arr[idx_sparse]       # プロット時
assert np.allclose(t_obs_data_gen, t_obs_plot), "時刻マッピング不一致！"
```

### Step 3: オフセット修正のテスト

もしインデックスオフセットが確認された場合：

```python
# 修正案1: idx_sparse を ±1 シフトしてテスト
idx_sparse_shifted = idx_sparse + 1  # または -1
# ただし、境界チェックが必要

# 修正案2: 時刻ベースの補間を使用
from scipy.interpolate import interp1d
t_obs_target = np.linspace(t_arr[0], t_arr[-1], n_obs)
phibar_interp = interp1d(t_arr, phibar, axis=0, kind='linear')
phibar_at_obs = phibar_interp(t_obs_target)
```

## 実際のデータ確認結果

実際のデータファイルから確認：
```python
t_arr shape: (2501,), data shape: (20, 2), idx shape: (20,)
t_arr[0]=0.000000, t_arr[-1]=0.025000
idx min=250, max=2500, len(t_arr)=2501
First 5 idx: [250 368 486 605 723]
First 5 t_arr[idx]: [0.0025  0.00368 0.00486 0.00605 0.00723]
```

**インデックス範囲は正常** - 問題は別の場所にある可能性が高い。

## 追加の可能性

### 時刻ステップの不一致

`t_arr` は `dt * step` で生成されているが、**データ生成時とプロット時の `t_arr` が異なる可能性**：

1. **データ生成時**: `t_arr, x0, sig2 = tsm.solve_tsm(theta_true)` → この `t_arr` で `idx_sparse` を計算
2. **プロット時**: 別の `t_arr` が使われている可能性

### モデル評価時の時刻不一致

MCMC実行中、各 `theta` で `solve_tsm()` を呼ぶが、**返される `t_arr` の長さや時刻値が異なる可能性**：
- データ生成: `t_arr` が2501要素
- MCMC評価: `t_arr` が異なる長さ/時刻値

## 次のステップ

1. **即座に確認**: プロット生成時の `t_arr` とデータ生成時の `t_arr` が一致しているか
2. **デバッグ出力追加**: 
   ```python
   # プロット生成時に追加
   print(f"Plot t_arr: shape={t_arr.shape}, [0]={t_arr[0]}, [-1]={t_arr[-1]}")
   print(f"Data t_arr: shape={data_t_arr.shape}, [0]={data_t_arr[0]}, [-1]={data_t_arr[-1]}")
   print(f"idx_sparse: {idx_sparse}")
   print(f"t_arr[idx_sparse] vs data_t_arr[idx_sparse]:")
   print(f"  Plot: {t_arr[idx_sparse][:5]}")
   print(f"  Data: {data_t_arr[idx_sparse][:5]}")
   ```
3. **修正テスト**: 
   - プロット時にデータ生成時の `t_arr` を保存して使用
   - または、時刻ベースの補間を使用して時刻の不一致を吸収
