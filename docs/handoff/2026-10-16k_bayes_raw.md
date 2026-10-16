# クラウド側 → GPU 側（2026-10-16k）: ベイズ因子（a45 の有無）を生データ・2 通路で出し直す（24 本）

ユーザー決定（10/16、案 A）。本文 4.3 節・補足 Table S8 の ln B は、まだ読み取りデータ・1 通路の値。
生データ・2 通路の段 (i)（pilot）で、全モデルと a45 = 0 のモデルを 3 seed ずつ回し、TMCMC の log_evidence から ln B を出す。

## 設定
- 今の `*_pilot_2c_raw_seed*`（n_mut 40）は判定 (c)（移動 ≥ 5 回/次元）に落ちている（CS 2.9〜3.0、CH 2.7〜2.8）。
  ln Ẑ を比べるには両方のモデルが判定を通っている必要があるので、**全モデルも n_mut 120 で回し直す**（10/08 の `a45zero_m120` と同じ考え方）。
- 粒子 1000（今の pilot と同じ）、`N_MUT=120`、`DATA_SOURCE=raw TWO_CHANNEL=1`。walltime 10h（1 本 5h 前後の見込み）。
- OVERRIDE は今の raw の pilot と同じで、a45 = 0 のほうは `19:0:0` を足す（DS は a45 の行を置き換え）。

| 条件 | 全モデル（RUNTAG `2c_raw_m120`） | a45 = 0（RUNTAG `2c_raw_a45zero_m120`） |
|---|---|---|
| CS | なし | `19:0:0` |
| CH | なし | `19:0:0` |
| DS | `5:-15:20;6:-15:20;19:-15:20` | `5:-15:20;6:-15:20;19:0:0` |
| DH | なし | `19:0:0` |

```bash
cd ~/Tmcmc202601/data_5species/main && git pull
declare -A OVF=([CS]="" [CH]="" [DS]="5:-15:20;6:-15:20;19:-15:20" [DH]="")
declare -A OVZ=([CS]="19:0:0" [CH]="19:0:0" [DS]="5:-15:20;6:-15:20;19:0:0" [DH]="19:0:0")
for T in CS CH; do for S in 42 7 123; do   # 1 回目: CS・CH（12 本）。DS・DH は 2 回目に同じループで T を DS DH に
  qsub -l walltime=10:00:00 -v STAGE=pilot,TAG=$T,SEED=$S,N_MUT=120,DATA_SOURCE=raw,TWO_CHANNEL=1,RUNTAG=2c_raw_m120${OVF[$T]:+,OVERRIDE=${OVF[$T]}} paper_gateoff_job.sh
  qsub -l walltime=10:00:00 -v STAGE=pilot,TAG=$T,SEED=$S,N_MUT=120,DATA_SOURCE=raw,TWO_CHANNEL=1,RUNTAG=2c_raw_a45zero_m120,OVERRIDE=${OVZ[$T]} paper_gateoff_job.sh
done; done
```
（`OVERRIDE` の区切りは `;` のままでよい。job が `,` に直す）

## 優先度
- **CS・CH を先に（12 本）、DS・DH はそのあと（12 本）**（追記 10/16、ユーザー）。CS・CH は最終段まで確定しているので、
  ln B が出ればすぐ本文・表 S8 の CS・CH の行を確定できる。DS・DH は ult がそろうときにまとめて入れる。
  段 (i) の箱は後の段に依存しないので、DS・DH の分は DS の ult を待たずに、GPU が空いたら入れてよい。
- **本線（DS p2 → 磨き → gate → ult、DH ult、CH の磨き・回収）を先に**。空いた GPU に順に入れる（同時 15 本の枠の中で）。
  10/26 の週に共著者へ送る版に入れたいので、**10/23 までに終わる**のが目安。
- 1 本目が終わったら、run_record の `log_evidence` が数値（None でない）、`args.override_bounds` に `19:0:0`、`moves_per_particle` が 5 回/次元を超えることを確認。

## 終わったら
- 24 本を `docs/revision/generated/final_runs/` に（15d と同じ形）。`check_paper_runs.py` の結果（両群の判定）と、
  条件ごとに `ln Ẑ_full`・`ln Ẑ_{a45=0}`（3 seed の平均と幅）・ln B を `gpu_<日付>_bayes_raw.md` に。
- クラウド側で `spec_raw.json` の `pilot` を `*_pilot_2c_raw_m120_seed*` に、`a45zero` を `*_pilot_2c_raw_a45zero_m120_seed*` にして表 S8 と本文を作り直す。
