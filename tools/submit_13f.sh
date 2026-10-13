#!/bin/bash
# ============================================================
# 13f（質の確認 4・5）の投入（2026-10-13）。copaam で手で叩く。DRY_RUN=1 で qsub せず内容だけ出す。
#
#   4. 箱を 1.5 倍に広げた p2（CH・CS、3 seed）。2c の p2 が終わってから（箱は seed 42 の p2 から作る）
#        PART=box TAG=CH NODES='vancouver03;vancouver03;vancouver02' tools/submit_13f.sh
#   5. 1 日を外した pilot → p1 → p2（CH、seed 42、afterok で連鎖）。2c の p2 の結果を待たずに投げられる
#        PART=holdout DAY=10 NODE=stuttgart01 tools/submit_13f.sh
#
# 粒子数・mutation・箱は 13b の 2c と同じ（下の表）。終わったら notify_cloud_runs.sh が報告を push する。
# 評価: holdout は tools/holdout_eval.py、box は 2c の p2 との対照（ident_threshold_table.py の規則）。
# ============================================================
set -euo pipefail

MAIN="$HOME/Tmcmc202601/data_5species/main"
REPO="$HOME/Tmcmc202601"
R=_runs/paper_gateoff
PART="${PART:?PART=box|holdout}"
DRY_RUN="${DRY_RUN:-0}"
FACTOR="${FACTOR:-1.5}"

# 13b の 2c の設定（p1 の N_MUT、p2 の N_MUT・OVERRIDE）
declare -A P1_MUT=([CH]=150 [CS]=120)
declare -A P2_MUT=([CH]=150 [CS]=160)
declare -A P2_OVR=([CH]='1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5' [CS]='5:-15:20;0:-5:2;10:-1:20')

q() {  # qsub か、DRY_RUN なら表示だけ（jobid の代わりに DRY<n> を返す）
  if [ "$DRY_RUN" = "1" ]; then
    echo "qsub $*" >&2
    echo "DRY$RANDOM.copaam"
  else
    qsub "$@"
  fi
}

case "$PART" in
  box)
    TAG="${TAG:?TAG=CH|CS}"
    NODES="${NODES:?NODES='n1;n2;n3'（seed 42 / 7 / 123 の順）}"
    WALL="${WALL:-16:00:00}"
    REF="${REF_RUN:-$MAIN/$R/${TAG}_p2_2c_seed42}"  # 箱を作る run（既定は 2c の p2 の seed 42）
    [ -f "$REF/run_record.json" ] || { echo "2c の p2 がまだ無い: $REF"; exit 1; }
    OVR="$(python3 "$REPO/tools/widen_box.py" "$REF" --factor "$FACTOR")"
    echo "OVERRIDE（${FACTOR} 倍）: $OVR" >&2
    IFS=';' read -r -a NODE_ARR <<< "$NODES"
    cd "$MAIN"
    DEP=""
    i=0
    for S in 42 7 123; do
      N="${NODE_ARR[$i]:-${NODE_ARR[-1]}}"; i=$((i + 1))
      J="$(q -l "nodes=1:ppn=1:gpus=1:$N" -l "walltime=$WALL" \
        -v "STAGE=p2,TAG=$TAG,SEED=$S,TWO_CHANNEL=1,RUNTAG=2c_box15,N_PART=2000,N_MUT=${P2_MUT[$TAG]},OVERRIDE=$OVR,PREV=$R/${TAG}_p1_2c_seed$S" \
        paper_gateoff_job.sh)"
      echo "$TAG box15 seed$S $N $J"
      DEP="$DEP:$J"
    done
    cd "$REPO"
    echo "notify $(q -W "depend=afterany$DEP" -v "RUN_GLOB=${TAG}_p2_2c_box15_seed*,LABEL=${TAG}_p2_box15" tools/notify_cloud_runs.sh)"
    ;;
  holdout)
    DAY="${DAY:?DAY=10|15}"
    TAG="${TAG:-CH}"
    S="${SEED:-42}"
    NODE="${NODE:?NODE=<host>}"
    RT="2c_ho${DAY}"
    COMMON="TAG=$TAG,SEED=$S,TWO_CHANNEL=1,RUNTAG=$RT,HOLDOUT_DAY=$DAY"
    G="nodes=1:ppn=1:gpus=1:$NODE"
    cd "$MAIN"
    J1="$(q -l "$G" -l walltime=10:00:00 -v "STAGE=pilot,$COMMON" paper_gateoff_job.sh)"
    J2="$(q -l "$G" -l walltime=14:00:00 -W "depend=afterok:$J1" \
      -v "STAGE=p1,$COMMON,N_MUT=${P1_MUT[$TAG]},PREV=$R/${TAG}_pilot_${RT}_seed$S" paper_gateoff_job.sh)"
    J3="$(q -l "$G" -l walltime=16:00:00 -W "depend=afterok:$J2" \
      -v "STAGE=p2,$COMMON,N_PART=2000,N_MUT=${P2_MUT[$TAG]},OVERRIDE=${P2_OVR[$TAG]},PREV=$R/${TAG}_p1_${RT}_seed$S" paper_gateoff_job.sh)"
    echo "$TAG ho$DAY seed$S $NODE pilot $J1 / p1 $J2 / p2 $J3"
    cd "$REPO"
    echo "notify $(q -W "depend=afterany:$J3" -v "RUN_GLOB=${TAG}_p2_${RT}_seed*,LABEL=${TAG}_p2_ho${DAY}" tools/notify_cloud_runs.sh)"
    ;;
  *) echo "PART=box|holdout"; exit 1 ;;
esac
