# GPU 側 → クラウド側（2026-10-08 01:10）: `make_paper_figures.py` は 5 枚すべて通った。umap-learn を入れた

2026-10-08a §5 / 5' の `tools/make_paper_figures.py` を既存 run（pilot mut80 × 4 条件）で通した。
**原稿の図 5 枚と `paper_numbers.json` / `tables.tex` がすべて出た。**

```
$ ~/miniforge3/envs/klempt_fem2/bin/python3 tools/make_paper_figures.py \
      data_5species/main/_runs/paper_gateoff \
      --p2-glob '{tag}_pilot_mut80_*' --p1-glob '{tag}_pilot_mut80_*' \
      --skip knockout ident --out-dir <scratch>
  heatmap_A_4cond.pdf                 592 KB
  paper_fig2_phi_transposed.pdf      2.8 MB   （ODE を回す Fig 2 も完走）
  paper_posterior_violin_sharey.pdf   432 KB
  phase1_vs_phase2_map.pdf            191 KB
  umap_A_matrix_3d.pdf               1.8 MB
  CS: RMSE 0.118  Pg D21/D15 0.99  a45 MAP -0.06 90% [-0.89, +0.89]
  CH: RMSE 0.069  Pg D21/D15 1.00  a45 MAP -0.11 90% [-0.85, +1.86]
  DS: RMSE 0.018  Pg D21/D15 1.32  a45 MAP +2.42 90% [+1.61, +2.48]
  DH: RMSE 0.062  Pg D21/D15 2.83  a45 MAP +1.95 90% [+0.88, +5.35]
```

（出力は scratchpad に出した。**原稿の `BMB_submission/figures/` は触っていない**。
数字は pilot の run なので中身は使わない。通ることの確認だけ。）

## 入れたもの（`klempt_fem2` env）

UMAP が無くて Fig 6 が作れなかったので入れた:

```
umap-learn 0.5.12, pynndescent 0.6.0, scikit-learn 1.7.2,
joblib 1.6.0, threadpoolctl 3.7.0, cloudpickle 3.1.2, tqdm 4.70.1
```

JAX との共存を確認済み（`numpy 2.2.6 / numba 0.65.1 / jax 0.6.2`、CPU で前進計算が通る）。
TeX 側（cm-super・dvipng）は既に入っていた。

## 2026-10-08 01:00 の報告（gpu_2026-10-08_downstream.md）のうち、不要になった項目

`docs/regenerate_all_figures.py` の 3 不具合（run 名のハードコード・前進モデルが
`hamilton_ode_jax`・`K_hill=0.05` 固定）は直して commit したが、**原稿の図は
`make_paper_figures.py` で作るので、こちらが正規の経路**。`regenerate_all_figures.py` は
修正済みのまま残してある（Fig 3 / Fig 6 の生成元が無いという相談も解決済み）。

## ult が出てからの手順（確認）

```bash
python3 tools/make_paper_figures.py data_5species/main/_runs/paper_gateoff \
    --p2-glob '{tag}_ult*' --p1-glob '{tag}_p1*' \
    --p2-glob-DH 'DH_ult*nonarrow*' --p1-glob-DH 'DH_p1_mut80*' \
    --p2-glob-DS 'DS_ult*wide80*'   --p1-glob-DS 'DS_p1*wide80*' \
    --ident-glob '{tag}_ident_prior{prior}_mut80_seed*'
```
判定を通った群だけに glob を合わせる。出力（figures と generated）を commit して push する。
