# GPU 側報告 2026-10-09（09d の実行: 12 本投入・gate を直した版で 3 群に仕掛け直し）

## 0. 先に訂正: CH ult（gate 3320）は**そもそも投入されていなかった** → qdel は不要

`logs/pbs/gate_ult_3320.copaam.log` を見ると、判定 PASS のあと

```
tools/gate_submit_ult.sh: line 122: qsub: command not found   （3 seed とも）
    警告: 通知ジョブの投入に失敗: ... qsub: command not found
```

で止まっていた。PBS バッチの PATH に `/usr/local/bin` が無い（`gate_CH_ult.md` の表で jobid が空欄なのはこのため）。
`qstat -u nishioka` でも 09d 投入前の自分のジョブは 0 本、`_runs/paper_gateoff/*noph_sd4*` も無し。
**走っていない・何も書かれていないので、qdel も掃除も要らない。**

`tools/gate_submit_ult.sh` を直した（コミット済み）:

1. `export PATH="/usr/local/bin:$PATH"`（qsub 見つからず問題）
2. **09c の条件**: `CHECK_OUT` に `←` が 1 つでもあれば投入しない（`EDGE_N` を数えて報告に書く）

## 1. 投入（09d の順: DH p2 → DS p2 → CS ult → CH p2、全 12 本・走行確認済み）

`gpualloc` で FREE の 4090 がちょうど 6 枚（vancouver01 ×2・02 ×2・03 ×2）だったので DH・DS を 1 seed ずつ分散。

| jobid | 群 | seed | ノード(GPU) | walltime | 見込み |
|---|---|---|---|---|---|
| 3321 | DH p2 `nonarrow_w2_noph_p10k` | 42 | vancouver01/0 | 22:00 | 約 17h |
| 3322 | 〃 | 7 | vancouver02/1 | 22:00 | |
| 3323 | 〃 | 123 | vancouver03/0 | 22:00 | |
| 3324 | DS p2 `wide80_p8k_wide2` | 42 | vancouver01/2 | 22:00 | 約 16h |
| 3325 | 〃 | 7 | vancouver02/3 | 22:00 | |
| 3326 | 〃 | 123 | vancouver03/1 | 22:00 | |
| 3327 | CS ult `wide2_sd4` | 42 | stuttgart01/1 | 08:00 | 約 2〜3h（CH ult 5000 粒子 N_MUT150 が 2.1h） |
| 3328 | 〃 | 7 | stuttgart01/2 | 08:00 | |
| 3329 | 〃 | 123 | stuttgart01/3 | 08:00 | |
| 3330 | CH p2 `mut150_wide2_noph` | 42 | stuttgart02/0 | 10:00 | 約 7.5h（前回 `mut150_wide_noph` 7.4h） |
| 3331 | 〃 | 7 | stuttgart02/1 | 10:00 | |
| 3332 | 〃 | 123 | stuttgart02/2 | 10:00 | |

設定は 09d の §1〜§4 をそのまま（`qstat -f` の Variable_List 全文で OVERRIDE・RUNTAG・PREV を確認）:

```
DH: STAGE=p2 N_PART=10000 N_MUT=80  RUNTAG=nonarrow_w2_noph_p10k
    OVERRIDE='12:-15:20;5:-15:20;6:-15:20;18:-15:20;13:-15:20;0:-5:5;11:-8:3;19:-5:20;1:-5:3;2:-5:5;7:-5:5'
    PREV=DH_p1_mut80_seed$S
DS: STAGE=p2 N_PART=8000  N_MUT=100 RUNTAG=wide80_p8k_wide2
    OVERRIDE='5:-15:20;6:-15:20;19:-15:20;14:-5:4;12:-5:3;16:-5:2.5'   PREV=DS_p1_wide80_p3k_seed$S
CS: STAGE=ult N_MUT=150 RUNTAG=wide2_sd4
    OVERRIDE='5:-15:20;0:-5:2;10:-1:20'   PREV=CS_p2_mut160_wide2_seed$S
CH: STAGE=p2 N_MUT=150 RUNTAG=mut150_wide2_noph
    OVERRIDE='1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5'   PREV=CH_p1_mut150_seed$S
```

## 2. 依存ジョブ（直した gate 版）

| jobid | 種類 | 待つ対象 | 通ったら投げるもの |
|---|---|---|---|
| 3333 | gate | CH p2 3330-3332 | CH ult `mut150_wide2_noph_sd4` N_MUT150 OVERRIDE 同じ、stuttgart02、12h |
| 3334 | gate | DH p2 3321-3323 | DH ult `nonarrow_w2_noph_p10k_sd4` N_MUT150 OVERRIDE 同じ、vancouver01、12h |
| 3335 | gate | DS p2 3324-3326 | DS ult `wide80_p8k_wide2_sd4` N_MUT150 OVERRIDE 同じ、vancouver02、12h |
| 3336 | notify | CS ult 3327-3329 | （判定を `gpu_*_CS_ult.md` に書くだけ） |

- gate は **全群 PASS かつ `←` が 0** のときだけ ult を投げる。どちらか欠けると `gpu_<日付>_gate_<TAG>_ult.md` に理由を書いて終わる。
  → 09d の「DH・DS はこれが最後。通らなければ止めて報告」はこの挙動そのもの。
- DH の **a35 の山ごとの集計（a35 < −7 / ≥ −7 の粒子数・max logL・a45 の 5/50/95%）** は gate では出ない。
  DH p2 が終わったら（10-10 02:00 ごろ）手で `a33_mode_weight.py` の a35 版を回して報告に足す。
- gate は走り出す時点で `git pull` してから repo の版で実行し直すので、キュー待ちの間の修正も届く。

## 3. 08s の比較表（CH・DH の pH あり / なし）

（下に追記）
