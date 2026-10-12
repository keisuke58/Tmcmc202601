#!/bin/bash
#PBS -N recovery
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q default
#PBS -j oe
#PBS -o logs/pbs/${PBS_JOBNAME}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# ============================================================
# 合成データによる回復テスト（共著者レビュー B2、2026-10-12h）の推定を GPU で回す PBS ジョブ。
# 資源の要求は data_5species/main/paper_gateoff_job.sh（本番の ult）と同じ。
#
# 中身:
#   1. 合成データ（tools/recovery_make_synthetic.py）が無ければ作る（CPU で 20 秒ほど。乱数の種固定なので
#      3 seed の job が同時に作っても同じ内容。書き込みは原子的）
#   2. tools/recovery_run.py で estimate_paper_jax.py をそのまま（ローダーだけ差し替えて）回す。
#      箱 = 元 run の prior_bounds_final、粒子数・mutation・λ は元 run と同じ、warm start はしない
#
# 使い方（CH, seed 42。ノードは必ず明示、celtic03 は使わない）:
#   cd ~/Tmcmc202601/data_5species/main
#   qsub -l nodes=1:ppn=1:gpus=1:stuttgart02 -l walltime=20:00:00 \
#        -v TAG=CH,SEED=42 ../../tools/recovery_job.sh
#
# 環境変数:
#   TAG       CS|CH|DS|DH（必須）
#   SEED      TMCMC の seed（既定 42。本番と同じ 42 / 7 / 123 の 3 本を投げる）
#   NOISE     合成データのノイズ倍率（既定 1。0 で無ノイズ）
#   RNG       合成データの乱数の種（既定 0。3 seed で同じ合成データを使うこと）
#   SRC_GLOB  元にする最終 run 群（空なら recovery_common.GROUPS の既定。DS・DH は ult が通ったら渡す）
#   SRC_ROOT  元 run の置き場所（既定 docs/revision/generated/final_runs）
#   N_PART    粒子数の上書き（空なら元 run と同じ）
#   N_MUT     mutation 回数の上書き（空なら元 run と同じ）
#   RUNTAG    出力ディレクトリ名の末尾に付ける識別子
#   OVERWRITE=1  既存の出力を上書き（既定は止まる）
# 出力: data_5species/main/_runs/recovery/<TAG>_syn_n<NOISE>_r<RNG>[_<RUNTAG>]_seed<SEED>/
#       （samples.npy・logL.npy・theta_MAP.json・run_record.json・config.json・recovery_record.json）
#       合成データは _runs/recovery/<TAG>_syn_n<NOISE>_r<RNG>/synthetic.json
# ============================================================
set -euo pipefail

TAG="${TAG:?TAG must be CS|CH|DS|DH}"
SEED="${SEED:-42}"
NOISE="${NOISE:-1}"
RNG="${RNG:-0}"
SRC_GLOB="${SRC_GLOB:-}"
N_PART="${N_PART:-}"
N_MUT="${N_MUT:-}"
RUNTAG="${RUNTAG:-}"

REPO="$HOME/Tmcmc202601"
SRC_ROOT="${SRC_ROOT:-$REPO/docs/revision/generated/final_runs}"
cd "$REPO/data_5species/main"
mkdir -p logs/pbs _runs/recovery
PYTHON=/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3

# ラベルは recovery_make_synthetic.py と同じ書式（{noise:g}）
LABEL="$("$PYTHON" -c "import sys; print(f'{sys.argv[1]}_syn_n{float(sys.argv[2]):g}_r{int(sys.argv[3])}')" "$TAG" "$NOISE" "$RNG")"
SYN="_runs/recovery/${LABEL}/synthetic.json"
SUFFIX=""
[ -n "$RUNTAG" ] && SUFFIX="_${RUNTAG}"
OUTDIR="_runs/recovery/${LABEL}${SUFFIX}_seed${SEED}"

if [ -d "$OUTDIR" ] && [ "${OVERWRITE:-0}" != "1" ]; then
  echo "既に存在する: $OUTDIR  （RUNTAG を変えるか OVERWRITE=1 を渡す）"
  exit 1
fi

# GPU の割り当ては PBS に任せる（paper_gateoff_job.sh と同じ）
if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
  export CUDA_VISIBLE_DEVICES="$(grep -oE 'gpu[0-9]+' "$PBS_GPUFILE" | head -1 | grep -oE '[0-9]+')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

echo "=== recovery $TAG seed=$SEED noise=$NOISE rng=$RNG  $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "    SYN=$SYN  OUT=$OUTDIR  git=$(git rev-parse --short HEAD)  $(date)"

# 1. 合成データ（無ければ作る。CPU で）
if [ ! -f "$SYN" ]; then
  GEN=(--tag "$TAG" --noise "$NOISE" --rng-seed "$RNG" --root "$SRC_ROOT" --out "$SYN")
  [ -n "$SRC_GLOB" ] && GEN+=(--glob "$SRC_GLOB")
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PYTHON" "$REPO/tools/recovery_make_synthetic.py" "${GEN[@]}"
fi

# 2. 推定（estimator はそのまま、ローダーだけ合成データに差し替え）
RUN=(--synthetic "$SYN" --seed "$SEED" --output-dir "$OUTDIR" --device gpu)
[ -n "$N_PART" ] && RUN+=(--n-particles "$N_PART")
[ -n "$N_MUT" ] && RUN+=(--n-mutation-steps "$N_MUT")
"$PYTHON" "$REPO/tools/recovery_run.py" "${RUN[@]}"

echo "=== done $(date)  -> $OUTDIR"
