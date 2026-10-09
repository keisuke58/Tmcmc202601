# GPU 側報告 2026-10-09（gate: DH ult は**投入しなかった**）

前段 `DH_p2_nonarrow_w2_noph_p14k_seed*` の判定が通らなかった（または片側の端の印がある）ので、`nonarrow_w2_noph_p14k_sd4` の ult は投入していない
（FAIL の段からは進めない、docs/paper_gateoff_pipeline.md §5）。

- 前段の run: **0 / 3**（足りない場合は walltime か crash で落ちた seed がある）
- 判定: **（結論行が出なかった。下の出力をそのまま読むこと）**
- 片側の端の印（←）: **0**（1 つでもあれば投入しない。箱を広げて p2 を回し直す）

```
判定できない: data_5species/main/_runs/paper_gateoff に run が無い（空の PASS を返さない）
```
