#!/bin/bash
#PBS -N weights_raw
#PBS -l nodes=1:ppn=1
#PBS -q default
#PBS -j oe
#PBS -o /home/nishioka/Tmcmc202601/logs/pbs/weights_raw_${PBS_JOBID}.log

# ============================================================
# 14a: 尤度の重みの感度（raw・2 通路、決定論的な最適化）を 1 本ずつ回す。
# tools/two_channel_pilot_v2.py は中で JAX を CPU に固定しているので GPU は確保しない。
#
# 使い方（ノードは必ず明示。celtic03 は使わない）:
#   qsub -l nodes=1:ppn=1:vancouver02 -l walltime=08:00:00 -v TAG=CH,VAR=B tools/weights_raw_job.sh
#   VAR=B: 重みなし（λ_pg=λ_late=λ_rare=1）、VAR=C: λψ=2.0
#   VAR=TABLE: 8 本のあとに表を作って push する（-W depend=afterany:... で最後に）
# ============================================================
set -uo pipefail

REPO="$HOME/Tmcmc202601"
VAR="${VAR:?VAR を渡す（B|C|TABLE）}"
PYTHON="${PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3}"
O="${OUTDIR:-docs/revision/generated/weight_sensitivity/raw}"
# 14c: INIT_GLOB を渡すとその run 群の MAP から始める（例 data_5species/main/_runs/paper_gateoff/CH_p2_2c_raw_a33w_seed*）
INIT_GLOB="${INIT_GLOB:-}"

cd "$REPO" || exit 1
unset LD_LIBRARY_PATH
# 1 コアしか確保していないので XLA の CPU スレッドも 1 本にする（同じノードの GPU ジョブの邪魔をしない）
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

declare -A OV=(
 [CH]="5:-15:20;1:-5:2.5;11:-5:1;16:-5:1;0:-5:1.5;6:-5:1;7:-1:5;13:-5:1;14:-5:1.5;17:-5:1;18:-5:1"
 [CS]="5:-15:20;10:-1:20;0:-10:2;6:-5:1;11:-5:1;12:-5:3;13:-5:1;16:-5:1;17:-5:1;18:-5:1"
 [DS]="5:-15:20;6:-15:20;19:-15:20;14:-5:4;12:-5:3;16:-5:2.5;7:-5:2.5;10:-5:1.5"
 [DH]="12:-15:20;5:-15:20;6:-15:20;18:-25:20;13:-15:20;11:-8:3;19:-5:20;2:-10:5;7:-10:5;10:-8:3;0:-10:5;1:-10:3;17:-5:2.5"
)

if [ "$VAR" = "TABLE" ]; then
  # shellcheck source=tools/git_sync.sh
  source "$REPO/tools/git_sync.sh"
  git_sync_latest
  TEX=docs/revision/EML_submission/figures/raw/supp/supp_S_lambda2c.tex
  "$PYTHON" tools/weight_sens_table.py "$O" --out "$TEX" 2>&1 | tee "$O/table.out"
  git add "$O"/*.json "$O"/*.log "$TEX" 2>/dev/null
  case "$O" in
    *raw_p2init*) MSG="14c: 重みの感度（raw・2 通路、生データの p2 の MAP から）の A・B・C と表 supp_S_lambda2c.tex" ;;
    *)            MSG="14a: 重みの感度（raw・2 通路）の B・C 8 本と表 supp_S_lambda2c.tex" ;;
  esac
  git -c user.name="Keisuke Nishioka" -c user.email="kei128608@gmail.com" \
    commit -q -m "$MSG" || exit 0
  git_push_current
  exit 0
fi

TAG="${TAG:?TAG を渡す（CS|CH|DS|DH）}"
case "$VAR" in
  A) EXTRA=() ;;
  B) EXTRA=(--lambda-pg 1 --lambda-late 1 --lambda-rare 1) ;;
  C) EXTRA=(--lambda-ch3 2.0) ;;
  *) echo "VAR は A|B|C|TABLE"; exit 2 ;;
esac
OUT="$O/${TAG}_${VAR}.json"
[ -s "$OUT" ] && { echo "既にある: $OUT"; exit 0; }

echo "weights_raw: TAG=$TAG VAR=$VAR host=$(hostname) HEAD=$(git rev-parse --short HEAD) start=$(date)"
"$PYTHON" -u tools/two_channel_pilot_v2.py --tag "$TAG" --data-source raw --override "${OV[$TAG]}" \
  --restarts 1 --maxiter 150 ${INIT_GLOB:+--init-glob "$INIT_GLOB"} "${EXTRA[@]}" --out "$OUT" > "$O/${TAG}_${VAR}.log" 2>&1
echo "終了コード $? end=$(date)"
