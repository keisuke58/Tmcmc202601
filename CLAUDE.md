# Tmcmc202601 — プロジェクト指示

@claude.md

## ジョブ投入のルール（必ず守る）

**GPU / 重い計算は `qsub`（PBS/Torque, server: copaam）でしか投入しない。**
`ssh` して裏で直接走らせることはしない。

**`celtic03` は使わない。** ノードを明示して投入する（使えるのは `stuttgart01-03`、
`celtic01` / `celtic02`、`vancouver01` / `vancouver02` が各 4 枚、`vancouver03` が 2 枚）。
**`vancouver01-02` は RTX4090 で一番速い**（stuttgart は 3090、celtic は 2080Ti）。
急ぐときは vancouver から埋める（2026-10-08 に許可）。

```bash
cd ~/Tmcmc202601/data_5species/main
qsub -l nodes=1:ppn=1:gpus=1:stuttgart01 -l walltime=05:00:00 \
     -v STAGE=ident,TAG=DH,SEED=42 paper_gateoff_job.sh
qstat -u nishioka    # 自分のジョブを確認
```

複数本は stuttgart01 → 02 → 03 → celtic01 → celtic02 と振り分ける。ノード指定を省くと
Torque が celtic03 にも割り当てる。**celtic03 は GPU が 1 枚バスから落ちていて
（`0000:1A:00.0`、`nvidia-smi` 自体がエラー）、割り当てられた GPU ジョブは `cuInit(0)` が
`CUDA_ERROR_NO_DEVICE` で失敗して起動時に死ぬ**（2026-10-05 時点・管理者連絡済み）。
celtic01 / celtic02 は正常。

理由と効果:

- **ターミナルを閉じても、ssh が切れても、Claude のセッションが終わってもジョブは死なない。**
  Torque の下で走るのでプロセスの親が端末やシェルではない（`setsid`/`nohup` も不要）
- 共有サーバーなので GPU の二重確保を防げる。`-l nodes=1:ppn=N:gpus=1` で Torque が空き GPU を割り当てる
- **同時 15 本まで**（2026-10-08 に 10 → 15 に緩和。論文投稿〜10-16 を急ぐため）。
  投入前に `qstat -u nishioka` で本数を確認し、`qstat -a` と `gpufree` で他ユーザーの
  使用状況も見る。他ユーザーが混んでいるときは控える

Claude Code の設定でバックグラウンドシェルを残すことはできない（そういう設定キーは存在しない）。
ジョブの寿命は `qsub` で担保する、がこのリポジトリの方針。

## 投入前のチェック

```bash
cd ~/Tmcmc202601 && git pull
python3 tools/test_paper_pipeline.py                                  # 全項目 OK を確認
python3 tools/check_paper_runs.py data_5species/main/_runs/paper_gateoff   # 既存 run の判定
```

判定に FAIL がある群は解釈しない・次の段に進めない（`docs/paper_gateoff_pipeline.md` §5）。
