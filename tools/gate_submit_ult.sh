#!/bin/bash
#PBS -N gate_ult
#PBS -l nodes=1:ppn=1
#PBS -q default
#PBS -j oe
#PBS -o logs/pbs/${PBS_JOBNAME}_${PBS_JOBID}.log

# ============================================================
# 前段（p2）が全 seed 終わったら判定し、PASS なら次の段（ult）を自分で qsub する。
# 判定が FAIL なら投入せず、理由を docs/handoff/ に書いて push する。
#
# なぜジョブにするか: 投入が Claude のセッションに依存すると、端末を閉じた時点で
# 止まる。Torque の依存ジョブにすれば、セッションが終わっていても夜のうちに進む
# （このリポジトリの方針: ジョブの寿命は qsub で担保する）。
#
# 使い方（前段の 3 本に依存させる）:
#   qsub -W depend=afterany:3307.copaam:3308.copaam:3309.copaam \
#        -v TAG=CH,PREV_RUNTAG=mut150_wide_noph,ULT_RUNTAG=mut150_wide_noph_sd4,\
# N_MUT=150,OVERRIDE='1:-5:2.5;11:-5:1;16:-5:1',NODE=stuttgart02,WALLTIME=12:00:00 \
#        tools/gate_submit_ult.sh
#
# 環境変数:
#   TAG           CS|CH|DS|DH（必須）
#   PREV_RUNTAG   p2 の RUNTAG（必須。PREV は <TAG>_p2_<PREV_RUNTAG>_seed<S>）
#   ULT_RUNTAG    ult の RUNTAG（必須）
#   N_MUT         ult の mutation 回数（既定 150）
#   OVERRIDE      ult の箱の上書き（p2 と同じものを渡す）
#   MODES         判定 4 で山ごとに見る多峰の成分と谷（check_paper_runs.py --modes に渡す。
#                 例 'a34:-1.5;a35:-10'。qsub -v はカンマで変数を区切るので区切りは ";"。空なら従来どおり）
#   SEEDS         既定 "42 7 123"
#   NODE          投入先ホスト（既定 stuttgart02）
#   WALLTIME      既定 12:00:00
#   DRY_RUN=1     qsub せず、投げる内容を出すだけ
# ============================================================
set -uo pipefail

REPO="$HOME/Tmcmc202601"

# PBS は qsub 時点のスクリプトを複製するので、キューで待つ間の修正は届かない
# （2026-10-06 の ident 通知が古い版で空振りした）。走り出したら最新で実行し直す。
if [ -z "${GATE_REEXEC:-}" ]; then
  cd "$REPO" || exit 1
  # shellcheck source=tools/git_sync.sh
  source "$REPO/tools/git_sync.sh"
  git_sync_latest
  GATE_REEXEC=1 exec bash "$REPO/tools/gate_submit_ult.sh"
fi

cd "$REPO" || exit 1
source "$REPO/tools/git_sync.sh"

TAG="${TAG:?TAG must be CS|CH|DS|DH}"
PREV_RUNTAG="${PREV_RUNTAG:?PREV_RUNTAG (p2 の RUNTAG) が必要}"
ULT_RUNTAG="${ULT_RUNTAG:?ULT_RUNTAG が必要}"
N_MUT="${N_MUT:-150}"
OVERRIDE="${OVERRIDE:-}"
MODES="${MODES:-}"
MODES="${MODES//;/,}"
SEEDS="${SEEDS:-42 7 123}"
NODE="${NODE:-stuttgart02}"
WALLTIME="${WALLTIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON="${PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3}"
[ -x "$PYTHON" ] || PYTHON=python3
export JAX_PLATFORMS=cpu
unset LD_LIBRARY_PATH
# PBS バッチの PATH には /usr/local/bin が無く、qsub が見つからない（2026-10-09 の CH gate 3320 は
# 判定 PASS のあと qsub: command not found で 3 本とも投入できていなかった）
export PATH="/usr/local/bin:$PATH"

MAIN="data_5species/main"
RUNS_ROOT="$MAIN/_runs/paper_gateoff"
GLOB="${TAG}_p2_${PREV_RUNTAG}_seed*"
mkdir -p logs/pbs

echo "=== gate_ult $TAG  prev=$GLOB  -> ult $ULT_RUNTAG  $(date)"

# 1. 前段の run が seed 分そろっているか（walltime や crash で落ちた seed はディレクトリが無い）
FOUND="$(ls -d $RUNS_ROOT/$GLOB 2>/dev/null | wc -l)"
WANT="$(echo $SEEDS | wc -w)"
echo "    前段の run: $FOUND / $WANT"

# 2. 判定
CHECK_OUT="$("$PYTHON" tools/check_paper_runs.py "$RUNS_ROOT" --glob "$GLOB" --modes "$MODES" 2>&1)"
VERDICT="$(echo "$CHECK_OUT" | grep -E "^(全群 PASS|FAIL を含む群)" | tail -1)"
echo "$CHECK_OUT"
echo "    判定: ${VERDICT:-（結論行なし）}"

# 片側の端の印（←）は判定 0〜4 に入っていない。p2 で新しく端に積む成分があれば、ult へは進まず
# p2 の箱を広げて回し直す（08k / 08m の規則、2026-10-09c で gate にも追加）
EDGE_N="$(echo "$CHECK_OUT" | grep -c '←')"
echo "    片側の端の印（←）: $EDGE_N"

OK=0
[ "$FOUND" = "$WANT" ] && echo "$VERDICT" | grep -q '^全群 PASS' && [ "$EDGE_N" = "0" ] && OK=1

TODAY="$(date +%Y-%m-%d)"
REPORT="docs/handoff/gpu_${TODAY}_gate_${TAG}_ult.md"

if [ "$OK" != "1" ]; then
  {
    echo "# GPU 側報告 ${TODAY}（gate: ${TAG} ult は**投入しなかった**）"
    echo
    echo "前段 \`$GLOB\` の判定が通らなかった（または片側の端の印がある）ので、\`$ULT_RUNTAG\` の ult は投入していない"
    echo "（FAIL の段からは進めない、docs/paper_gateoff_pipeline.md §5）。"
    echo
    echo "- 前段の run: **$FOUND / $WANT**（足りない場合は walltime か crash で落ちた seed がある）"
    echo "- 判定: **${VERDICT:-（結論行が出なかった。下の出力をそのまま読むこと）}**"
    echo "- 片側の端の印（←）: **$EDGE_N**（1 つでもあれば投入しない。箱を広げて p2 を回し直す）"
    echo
    echo '```'
    echo "$CHECK_OUT"
    echo '```'
  } > "$REPORT"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: $REPORT を書いたが commit / push はしない"
  else
    git add "$REPORT" && git commit -q -m "handoff: gate ${TAG} ult は判定が通らず投入せず（${TODAY}）" && git_push_current
  fi
  echo "=== 投入せずに終了"
  exit 0
fi

# 3. PASS → ult を投入
SUBMITTED=""
for S in $SEEDS; do
  PREV="_runs/paper_gateoff/${TAG}_p2_${PREV_RUNTAG}_seed${S}"
  VARS="STAGE=ult,TAG=${TAG},SEED=${S},N_MUT=${N_MUT},RUNTAG=${ULT_RUNTAG},PREV=${PREV}"
  [ -n "$OVERRIDE" ] && VARS="${VARS},OVERRIDE=${OVERRIDE}"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: qsub -l nodes=1:ppn=1:gpus=1:${NODE} -l walltime=${WALLTIME} -v ${VARS} paper_gateoff_job.sh"
    continue
  fi
  JID="$(cd "$MAIN" && qsub -l "nodes=1:ppn=1:gpus=1:${NODE}" -l "walltime=${WALLTIME}" -v "$VARS" paper_gateoff_job.sh)"
  echo "    投入: $JID  seed=$S"
  SUBMITTED="${SUBMITTED} ${JID}"
done
[ "$DRY_RUN" = "1" ] && exit 0

# 4. 投入した ult に通知ジョブを付ける（終わったらまた判定が handoff に出る）
DEP="afterany"
for J in $SUBMITTED; do DEP="${DEP}:${J}"; done
NOTIFY="$(qsub -W "depend=${DEP}" -v "RUN_GLOB=${TAG}_ult_${ULT_RUNTAG}_seed*,LABEL=${TAG}_ult" tools/notify_cloud_runs.sh 2>&1)" \
  && echo "    通知ジョブ: $NOTIFY" || echo "    警告: 通知ジョブの投入に失敗: $NOTIFY"

{
  echo "# GPU 側報告 ${TODAY}（gate: ${TAG} p2 が PASS → ult を投入した）"
  echo
  echo "前段 \`$GLOB\` は **$FOUND / $WANT** そろい、判定は **$VERDICT**、片側の端の印（←）は 0。"
  echo "09a §1 の指示どおり、返事を待たずに ult を投入した（Claude のセッションに依存しないよう"
  echo "Torque の依存ジョブとして仕掛けておいたもの）。"
  echo
  echo "| jobid | seed | ノード | walltime |"
  echo "|---|---|---|---|"
  i=0; for S in $SEEDS; do i=$((i+1)); echo "| $(echo $SUBMITTED | cut -d' ' -f$i) | $S | $NODE | $WALLTIME |"; done
  echo
  echo "\`STAGE=ult TAG=${TAG} N_MUT=${N_MUT} RUNTAG=${ULT_RUNTAG} OVERRIDE='${OVERRIDE}' PREV=${TAG}_p2_${PREV_RUNTAG}_seed\$S\`"
  echo "（\`ULT_NSIGMA\` は既定 4、\`N_PART\` は既定 5000）"
  echo
  echo "判定の全文:"
  echo
  echo '```'
  echo "$CHECK_OUT"
  echo '```'
} > "$REPORT"
git add "$REPORT" && git commit -q -m "handoff: gate ${TAG} p2 PASS → ult 投入（${TODAY}）" && git_push_current
echo "=== done $(date)"
