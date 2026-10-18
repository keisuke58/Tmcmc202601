# GPU 側 → クラウド側: max logL の磨き直し（2026-10-18 / DS_pilot_a45zero_seed7_k3）

`tools/polish_max_logL_job.sh` が自動で書いた。対象 `DS_pilot_2c_raw_a45zero_m120_seed7`、上位 3 粒子、L-BFGS-B 最大 3000 反復。

```
JAX devices: [CudaDevice(id=0)]
run                                           粒子の最良       磨いた値       差    反復   収束
DS_pilot_2c_raw_a45zero_m120_seed7          -63.328    -57.781   5.547   584  yes


JSON: docs/handoff/gpu_2026-10-18_polish_DS_pilot_a45zero_seed7_k3.json
```

- HEAD: `2b313e6`、ホスト vancouver03、2026-10-18 20:38:34 JST
