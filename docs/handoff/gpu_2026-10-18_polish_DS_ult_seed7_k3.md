# GPU 側 → クラウド側: max logL の磨き直し（2026-10-18 / DS_ult_seed7_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_ult_2c_raw_seed7`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_ult_2c_raw_seed7                         -96.936    -96.317   0.619   203  yes


JSON: docs/handoff/gpu_2026-10-18_polish_DS_ult_seed7_k3.json
```

- HEAD: `acdf18b`、ホスト stuttgart02、2026-10-18 23:41:19 JST
