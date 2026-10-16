# GPU 側報告 2026-10-16（gate: DS ult は**投入しなかった**）

前段 `DS_p2_2c_raw_seed*` の判定が通らなかった（または片側の端の印がある）ので、`2c_raw` の ult は投入していない
（FAIL の段からは進めない、docs/paper_gateoff_pipeline.md §5）。

- 前段の run: **0 / 3**（足りない場合は walltime か crash で落ちた seed がある）
- 判定: **（結論行が出なかった。下の出力をそのまま読むこと）**
- 片側の端の印（←）: **0**（1 つでもあれば投入しない。箱を広げて p2 を回し直す）

```
判定できない: data_5species/main/_runs/paper_gateoff に run が無い（空の PASS を返さない）
```
