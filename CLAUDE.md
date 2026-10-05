# Tmcmc202601 — プロジェクト指示

@claude.md

## ジョブ投入のルール（必ず守る）

**GPU / 重い計算は `qsub`（PBS/Torque, server: copaam）でしか投入しない。**
`ssh` して裏で直接走らせることはしない。

```bash
cd ~/Tmcmc202601/data_5species/main
qsub -l walltime=05:00:00 -v STAGE=ident,TAG=DH,SEED=42 paper_gateoff_job.sh
qstat -u nishioka    # 自分のジョブを確認
```

理由と効果:

- **ターミナルを閉じても、ssh が切れても、Claude のセッションが終わってもジョブは死なない。**
  Torque の下で走るのでプロセスの親が端末やシェルではない（`setsid`/`nohup` も不要）
- 共有サーバーなので GPU の二重確保を防げる。`-l nodes=1:ppn=N:gpus=1` で Torque が空き GPU を割り当てる
- **同時 10 本まで。** 投入前に `qstat -u nishioka` で本数を確認する

Claude Code の設定でバックグラウンドシェルを残すことはできない（そういう設定キーは存在しない）。
ジョブの寿命は `qsub` で担保する、がこのリポジトリの方針。

## 投入前のチェック

```bash
cd ~/Tmcmc202601 && git pull
python3 tools/test_paper_pipeline.py                                  # 全項目 OK を確認
python3 tools/check_paper_runs.py data_5species/main/_runs/paper_gateoff   # 既存 run の判定
```

判定に FAIL がある群は解釈しない・次の段に進めない（`docs/paper_gateoff_pipeline.md` §5）。
