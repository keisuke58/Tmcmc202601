# GPU 側報告 2026-10-16（DS を優先: p2 → 磨き → gate → ult を依存でつないだ）

ユーザー指示（10/16）: **枠が空いたら DS を優先**。人手を待たずに DS が進むよう、DS p2 の後ろを依存ジョブでつないだ（16a §1・16b §2 の案のとおり）。

| jobid | 内容 | 依存 | ノード・walltime |
|---|---|---|---|
| 3597 | 磨き DS_p2_2c_raw_seed42（`TOP_K=3 MAXITER=3000`、ftol 既定、`WRITE_RUN=1`） | afterany:3519 | stuttgart02・24h |
| 3598 | 磨き DS_p2_2c_raw_seed7 | afterany:3520 | stuttgart02・24h |
| 3599 | 磨き DS_p2_2c_raw_seed123 | afterany:3521 | stuttgart02・24h |
| 3600 | DS の gate（`POLISH=0`） | afterany:3597:3598:3599 | stuttgart01・2h |

- gate 3600 の設定は旧 3529 と同じ（`TWO_CHANNEL=1 DATA_SOURCE=raw PREV_RUNTAG=2c_raw ULT_RUNTAG=2c_raw`、N_MUT 150、N_PART 18000、
  OVERRIDE `5:-15:20;6:-15:20;19:-15:20;14:-5:4;12:-5:3;16:-5:2.5;7:-5:2.5;10:-5:1.5`、ult は 36h）。
  **ult の投入先だけ stuttgart02 ×3 → `vancouver03;vancouver03;vancouver02`（4090）に変えた**。CS の ult（〜13:00 JST に終わる見込み）が空ける枠
- **旧 gate 3529 は qdel した**（hold 中で一度も走っていないので、誤報の push は無い）。DS p2 の通知 3530 はそのまま
- 磨きは p2 の seed ごとに、その seed の p2 が空ける stuttgart02 の GPU を引き継ぐので、本数は増えない（上限 15 本を超えない）
- DS p1 の磨き直し（3585〜3587）の判定 3 は、p2 → ult の流れを止めない（16a §2）。結果は別に報告する
- 回収テスト（16c §2）など、DS 以外で枠を使うものは、DS が ult まで進んでから投入する
