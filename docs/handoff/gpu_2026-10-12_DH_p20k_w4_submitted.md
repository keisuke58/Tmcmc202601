# GPU 側報告 2026-10-12（12c: DH p2 `nonarrow_w4_noph_p20k` と gate を投入した）

## 結論

- DH p2 `nonarrow_w4_noph_p20k` を 3 本（3395-3397、全部 4090・R）と gate（3398）を投入した。
- Variable_List は全文確認済み（RUNTAG・OVERRIDE 末尾の `10:-8:3` まで・PREV）。OVERRIDE は 12c と同じ。

```
STAGE=p2 TAG=DH SEED=$S N_PART=20000 N_MUT=120 RUNTAG=nonarrow_w4_noph_p20k
OVERRIDE='12:-15:20;5:-15:20;6:-15:20;18:-25:20;13:-15:20;0:-5:5;11:-8:3;19:-5:20;1:-5:3;2:-10:5;7:-10:5;10:-8:3'
PREV=_runs/paper_gateoff/DH_p1_mut80_seed$S   walltime 48h
```

| jobid | 内容 | ノード | 状態（投入時） |
|---|---|---|---|
| 3395 | seed 42 | vancouver01 | R |
| 3396 | seed 123 | vancouver01 | R |
| 3397 | seed 7 | vancouver03 | R |
| ~~3398~~ → 3399 | gate_ult（09c 版、MODES なし、12d で N_PART=20000 を追加） | — | H（`afterany:3395:3396:3397`） |

- gate が PASS なら ult は `N_MUT=150 RUNTAG=nonarrow_w4_noph_p20k_sd4`、vancouver01 で出る。
- **ult の walltime は 24h にした**（11l の gate は 12h）。20000 粒子の ult が 12h に収まるか未確認で、実行中の walltime は延ばせないため。
- DS ult p18k（3384-3386）・DH 仮 ult（3388-3390）とは枠が重なっていない（4090 の FREE 3 枚を使った）。

## 本数

| 種類 | jobid | 本数 |
|---|---|---|
| 計算 | DS ult 3384-3386・DH prov ult 3388-3390・DH p2 w4 3395-3397 | 9 |
| gate | 3398 | 1 |
| 通知 | 3387・3391 | 2 |

上限 15 本以内。

## 追記（12d）: gate を N_PART=20000 で仕掛け直した

- 3398 を qdel（`C` を確認）→ pull 後の `gate_submit_ult.sh` で **3399** を投入。依存は `afterany:3395:3396:3397` のまま、3395-3397 は R のまま。
- Variable_List を全文確認: `N_PART=20000`、OVERRIDE 末尾 `10:-8:3`、`ULT_RUNTAG=nonarrow_w4_noph_p20k_sd4`、`N_MUT=150`、`NODE=vancouver01`。
- **ult の walltime は 48h**（DS ult p18k 3384-3386 はまだ走行中で実測なし。12d の指示どおり）。
- `DRY_RUN=1` は流していない。p2 の run がまだ無いので段 1（seed がそろっているか）で止まり、qsub 行まで届かないため。N_PART が VARS に入る行は diff で確認した。
