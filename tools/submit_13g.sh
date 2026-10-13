#!/bin/bash
# ============================================================
# 13g（尤度の重みの感度）の投入（2026-10-13）。copaam で手で叩く。DRY_RUN=1 で qsub せず内容だけ出す。
#
#   2c の p2 を重みなし（λ_Pg = λ_late = λ_rare = 1）で 3 seed。λψ は自動調整のまま。
#   粒子数・mutation・OVERRIDE・PREV（2c の p1）は 2c の p2 と同じ。RUNTAG は 2c_now。
#
#     TAG=CH NODES='vancouver03;vancouver03;vancouver02' tools/submit_13g.sh
#     TAG=DH NODES='vancouver01;vancouver01;vancouver02' tools/submit_13g.sh
#
# 終わったら（notify が判定 (a)–(e) を push する）、2c の p2 との対照:
#   python3 tools/compare_run_groups.py data_5species/main/_runs/paper_gateoff \
#       'CH_p2_2c_seed*' 'CH_p2_2c_now_seed*' --labels 2c,now --out docs/revision/generated/weight_sensitivity/CH_gpu_class
#   python3 tools/fit_metrics_2c.py data_5species/main/_runs/paper_gateoff \
#       'CH_p2_2c_seed*' 'CH_p2_2c_now_seed*' --out docs/revision/generated/weight_sensitivity/CH_gpu_fit
# ============================================================
set -euo pipefail

MAIN="$HOME/Tmcmc202601/data_5species/main"
REPO="$HOME/Tmcmc202601"
R=_runs/paper_gateoff
TAG="${TAG:?TAG=CH|DH}"
NODES="${NODES:?NODES='n1;n2;n3'（seed 42 / 7 / 123 の順）}"
DRY_RUN="${DRY_RUN:-0}"

# 13b の 2c の p2（gpu_2026-10-13_2c_p2_submit.md と同じ）
declare -A P2_PART=([CH]=2000 [DH]=20000)
declare -A P2_MUT=([CH]=150 [DH]=120)
declare -A P2_WALL=([CH]=16:00:00 [DH]=48:00:00)
declare -A P2_OVR=([CH]='1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5'
                   [DH]='12:-15:20;5:-15:20;6:-15:20;18:-25:20;13:-15:20;0:-5:5;11:-8:3;19:-5:20;1:-5:3;2:-10:5;7:-10:5;10:-8:3')
[ -n "${P2_PART[$TAG]:-}" ] || { echo "TAG は CH か DH（13g）"; exit 1; }

q() {  # qsub か、DRY_RUN なら表示だけ
  if [ "$DRY_RUN" = "1" ]; then
    echo "qsub $*" >&2
    echo "DRY$RANDOM.copaam"
  else
    qsub "$@"
  fi
}

IFS=';' read -r -a NODE_ARR <<< "$NODES"
cd "$MAIN"
DEP=""
i=0
for S in 42 7 123; do
  N="${NODE_ARR[$i]:-${NODE_ARR[-1]}}"; i=$((i + 1))
  PREV="$R/${TAG}_p1_2c_seed$S"
  [ -f "$PREV/samples.npy" ] || { echo "2c の p1 が無い: $PREV"; exit 1; }
  J="$(q -l "nodes=1:ppn=1:gpus=1:$N" -l "walltime=${P2_WALL[$TAG]}" \
    -v "STAGE=p2,TAG=$TAG,SEED=$S,TWO_CHANNEL=1,LAMBDA_PG=1,LAMBDA_LATE=1,LAMBDA_RARE=1,RUNTAG=2c_now,N_PART=${P2_PART[$TAG]},N_MUT=${P2_MUT[$TAG]},OVERRIDE=${P2_OVR[$TAG]},PREV=$PREV" \
    paper_gateoff_job.sh)"
  echo "$TAG now seed$S $N $J"
  DEP="$DEP:$J"
done
cd "$REPO"
echo "notify $(q -W "depend=afterany$DEP" -v "RUN_GLOB=${TAG}_p2_2c_now_seed*,LABEL=${TAG}_p2_now" tools/notify_cloud_runs.sh)"
