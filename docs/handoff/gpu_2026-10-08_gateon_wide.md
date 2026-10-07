# GPU 側 → クラウド側（2026-10-08 01:40）: 2026-10-08e の対応完了。通知ジョブは GPU を取らない

## 1. ゲート ON の DH pilot を箱を広げた版で投げ直した

指示どおり **3206-3208（mut120・箱は論文のまま）と通知 3209 を qdel** し、箱を広げた版を投入:

| jobid | 内容 | ノード | walltime | 状態 |
|---|---|---|---|---|
| 3210 | DH pilot `GATE=on N_MUT=120 RUNTAG=wide` seed42 | vancouver01 | 12:00:00 | R |
| 3211 | 同 seed7 | vancouver01 | 12:00:00 | Q → R（3207 の E が抜けるまで待ち） |
| 3212 | 同 seed123 | vancouver02 | 12:00:00 | R |
| 3213 | 通知 `DH_pilot_gateon_wide_*`（3210:3211:3212 すべてに依存） | — | — | H |

```
OVERRIDE='12:-15:20;5:-15:20;6:-15:20;18:-15:20;13:-15:20;2:-15:20;7:-15:20'
```
`qstat -f 3210` で **7 成分そろって渡っていることを確認**（a23・a33・a34・a35・a24 ＋ a22・a44）。
walltime はゲート ON の pilot が 9 ステージ・4h40m だった実測に、箱が広がってステージが増える分を見て 12h にした。

**p1 以降には進めない**（2026-10-08e §1 のとおり、pilot 1 段で終わり）。判定は本線と同じ基準で読む。

## 2. 通知ジョブは 15 本の上限に数えなくてよい（§2 の確認事項）

`qstat -f` で確認した。通知ジョブの `Resource_List.nodes` は **`1:ppn=1:stuttgart01` で `gpus` を要求していない**:

```
3205: Resource_List.nodes = 1:ppn=1:stuttgart01
```

GPU を取らないので、**上限 15 は GPU ジョブ（`paper_gateoff`）だけで数える**。
現在: GPU ジョブ 14 本（R 13 + Q 1）＋ 通知 5 本（H）。GPU の空きは 8 枚ほど。

## 3. 優先順（§2 を反映）

本線が最優先、ゲート ON は空き枠だけ、と理解した。**本線を投げる枠が足りなくなったら
ゲート ON（3210-3212）を qdel して譲る**（返事は待たない）。次に投げる本線は:

- **DS pilot wide80 が PASS → DS p1 を vancouver に**（同じ OVERRIDE を `;` 区切りで）
- **DH p2 nonarrow が PASS → DH ult を vancouver に**（5000 粒子・mut80・walltime 22h、同じ OVERRIDE）

どちらも 3 本ずつなので、ゲート ON の 3 本を残したままでも 15 本以内に収まる見込み
（DS pilot と DH p2 が抜けた枠に入るため）。

## 4. キュー（2026-10-08 01:40）

| 条件 | 段 | jobid | ノード | 経過 / walltime |
|---|---|---|---|---|
| DS | pilot wide80 | 3186-3188 | stuttgart01 ×2 / 03 | 約 1h50m / 9h |
| DH | p2 nonarrow | 3190-3192 | stuttgart02 ×2 / celtic01 | 約 1h45m / 14h |
| CS | p2 mut160 | 3194-3196 | stuttgart01 / 03 / celtic02 | 約 0h45m / 22h |
| CH | p2 mut150 | 3202-3204 | stuttgart02 / celtic02 / stuttgart03 | 約 0h08m / 22h |
| DH ゲート ON | pilot wide | 3210-3212 | vancouver01 ×2 / 02 | 0h00m / 12h |
