# GPU 側 → クラウド側: max logL の磨き直し（2026-10-18 / DS_ult_seed42_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_ult_2c_raw_seed42`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_ult_2c_raw_seed42                        -96.796    -96.279   0.517   381  yes


JSON: docs/handoff/gpu_2026-10-18_polish_DS_ult_seed42_k3.json
```

- HEAD: `4f30328`、ホスト stuttgart01、2026-10-19 02:05:33 JST
