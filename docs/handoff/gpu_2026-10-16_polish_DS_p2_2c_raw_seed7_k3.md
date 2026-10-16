# GPU 側 → クラウド側: max logL の磨き直し（2026-10-16 / DS_p2_2c_raw_seed7_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_p2_2c_raw_seed7`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
対象の run が無い: data_5species/main/_runs/paper_gateoff/DS_p2_2c_raw_seed7

警告: polish_max_logL.py が終了コード 1 で終わった（上は途中までの出力）
```

- HEAD: `530730b`、ホスト stuttgart02、2026-10-16 10:07:36 JST
