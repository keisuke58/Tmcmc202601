# GPU 側 → クラウド側（2026-10-08 01:00）: 下流（図表生成）を先に通した。2 枚だけ生成元が無い

GPU は 4 条件で埋まっていて待ちなので、その間に **ult が出てから図を作る段**を先に通した。
結果として、放っておくと締切前に詰まる不具合が 3 つ見つかったので直した。

## 1. `docs/regenerate_all_figures.py` の 3 つの不具合（修正済み・commit 済み）

1. **run ディレクトリが 2026-03 の名前でハードコードされていて、8 個すべてこのマシンに存在しない**
   ```
   P2_DIRS = {"CS": "jax_ode_nuts_Commensal_Static_20260320_043505", ...}
   ```
   名前のとおり `nuts`＝**ゲート ON の旧 run** を指していた。
   → `--p1-glob` / `--p2-glob` で受けるようにした（`{cond}` が CS/CH/DS/DH に展開される。
   既定は `{cond}_ult_*` / `{cond}_p1_*`）。同じ段で複数 seed が当たったら **max_logL 最大**を採り、
   他の seed と値も표示して選択が見えるようにした。
2. **前進モデルが推定側と別系統**: `from hamilton_ode_jax import simulate_0d` だったが、
   `estimate_paper_jax.py` は `hamilton_ode_jax_paper` を使う。実質差分 222 行。
   **2026-09 の事故（推定と別のモデルで評価した）と同じ型**なので差し替えた。
3. **ゲートが ON でハードコードされていた**: `run_ode` が `K_hill=0.05, n_hill=4.0` 固定。
   ゲート OFF（K_hill=0）の事後に対してゲート ON で前進計算していたことになる。
   → `K_hill` / `n_hill` / `phi_init` と**ソルバ分岐**を各 run の `config.json` から読むようにした。
   多チャネル（ch3 生存率・ch5 pH）で `fix_psi` でない run は `simulate_0d_full` を使う
   （`estimate_paper_jax.py` の `_need_full` と同じ条件）。p2 / ult はこちらに該当する。

### 通した確認（CPU のみ、GPU は使っていない）

```
$ ~/miniforge3/envs/klempt_fem2/bin/python3 ~/Tmcmc202601/docs/regenerate_all_figures.py \
      --p2-glob '{cond}_pilot_mut80_*' --p1-glob '{cond}_pilot_mut80_*' --figs 3,4,5
  [p2] CS: CS_pilot_mut80_seed123  max_logL=-5.86  (others: ...seed7 -5.97, ...seed42 -5.97)
  [p2] CH: CH_pilot_mut80_seed7    max_logL=-3.19  (others: ...)
  [p2] DS: DS_pilot_mut80_seed7    max_logL=-0.85  (others: ...)
  [p2] DH: DH_pilot_mut80_seed7    max_logL=-3.92  (others: ...)
  Saved: docs/figures/paper_posterior_violin.pdf
  Saved: docs/figures/heatmap_A_4cond.pdf
  Saved: docs/figures/phase1_vs_phase2_map.pdf
```
Fig 2（事後予測、ODE を 4 条件 × 101 回）は CPU だと時間がかかるので別途回している。
**ult が出たら `--p2-glob '{cond}_ult_*'` に変えるだけ**で原稿用の図になる。

## 2. `tools/knockout_fn.py` は修正不要（CPU で完走を確認）

`DH_p2_mut80_seed7` に `--n-samples 8` で当てたら、内部の自己照合
（`simulate_masked` が `simulate_0d` と一致するか）も通って完走した。結果は論文の予測どおりの向き:

| run | K | Pg D21/D15 そのまま | Fn を除く | a45=0 | サージ(≥1.5)の割合 |
|---|---|---|---|---|---|
| DH_p2_mut80_seed7 | 0.00 | 2.34 [2.03, 2.60] | 1.05 [0.57, 2.40] | 1.01 [0.70, 2.73] | 100% / 12% / 25% |

（実測の DH は 2.74。ult が出たら `RUN_GLOB='<TAG>_ult_*'` で回す、は 2026-10-07g §4 のまま）

## 3. 判断してほしいこと: 原稿の図 5 枚のうち 2 枚は生成元がリポジトリに無い

`docs/revision/BMB_submission/BMB_manuscript_nishioka.tex` が実際に読んでいる図は 5 枚
（`fig_ph_validation.pdf` と `fig_heine_kegg_sign_comparison.png` は `%` でコメントアウト済み）。

| 図 | ファイル | 生成元 |
|---|---|---|
| Fig 2 | `paper_fig2_phi_transposed.pdf` | `docs/generate_fig2_phi_version.py`（**未点検**。同じ 3 不具合がある可能性） |
| Fig 3 | `paper_posterior_violin_sharey.pdf` | **無い**。この名前を書くコードがリポジトリに存在しない |
| Fig 4 | `heatmap_A_4cond.pdf` | `regenerate_all_figures.py`（修正済み） |
| Fig 5 | `phase1_vs_phase2_map.pdf` | `regenerate_all_figures.py`（修正済み） |
| Fig 6 | `umap_A_matrix_3d.pdf` | **無い**。`data_5species/experiment_data/fig.ipynb`（notebook）だけ |

Fig 3 と Fig 6 は、ult の事後で作り直す手段が今のところ無い。

**お願い**: この 2 枚について、(a) GPU 側がスクリプトを書き起こす（`regenerate_all_figures.py` の
fig3 を `sharey` 版にし、UMAP は notebook から切り出す）か、(b) クラウド側が持っている生成元を
push するか、どちらにするか指示してほしい。(a) なら GPU の待ち時間で進められる。
`docs/generate_fig2_phi_version.py` の点検も (a) に含めてよい。
