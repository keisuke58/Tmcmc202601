#!/bin/bash
#PBS -N knockout_fn
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q default
#PBS -j oe
#PBS -o knockout_fn_${PBS_JOBID}.log

# ============================================================
# tools/knockout_fn.py を GPU で回す（論文の予測「Fn を除くと Pg サージが消える」を事後サンプルで検証）。
# 各 run の事後サンプル N_SAMPLES 個で、そのまま / Fn を除く / a45=0 の 3 通りを前進計算する。
#
# 使い方（ノードは必ず明示。celtic03 は使わない）:
#   qsub -l nodes=1:ppn=1:gpus=1:stuttgart01 -l walltime=08:00:00 \
#        -v RUN_GLOB='D*_*mut80*',LABEL=dh_ds tools/knockout_fn_job.sh
# 結果: docs/handoff/gpu_<日付>_knockout_<LABEL>.{md,json} を書いて push する
# ============================================================
set -uo pipefail

REPO="$HOME/Tmcmc202601"
RUN_GLOB="${RUN_GLOB:?RUN_GLOB を渡す（例 'D*_*mut80*'）}"
LABEL="${LABEL:-knockout}"
N_SAMPLES="${N_SAMPLES:-500}"
PYTHON="${PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3}"

cd "$REPO" || exit 1
git pull --rebase --quiet origin "$(git rev-parse --abbrev-ref HEAD)" || echo "警告: pull に失敗。手元の版で続ける"

if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
  export CUDA_VISIBLE_DEVICES="$(grep -oE 'gpu[0-9]+' "$PBS_GPUFILE" | head -1 | grep -oE '[0-9]+')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

TODAY="$(date +%Y-%m-%d)"
REPORT="docs/handoff/gpu_${TODAY}_knockout_${LABEL}.md"
JSON="docs/handoff/gpu_${TODAY}_knockout_${LABEL}.json"
# KNOCKOUT_DEVICE=gpu: estimator の早期デバイス判定に gpu を渡す（既定 cpu だと GPU が隠れる）
OUT="$(KNOCKOUT_DEVICE=gpu "$PYTHON" tools/knockout_fn.py data_5species/main/_runs/paper_gateoff \
  --glob "$RUN_GLOB" --n-samples "$N_SAMPLES" --json "$JSON" 2>&1 \
  | grep -v -E '^(INFO|WARNING):' | tail -40)"
echo "$OUT" | grep -q 'JAX devices: \[Cuda\|JAX devices: \[cuda' || echo "警告: GPU で走っていない可能性（出力の JAX devices を確認）"

{
  echo "# GPU 側 → クラウド側: Fn ノックアウト（${TODAY} / ${LABEL}）"
  echo
  echo "\`tools/knockout_fn_job.sh\` が自動で書いた。対象 \`${RUN_GLOB}\`、各 run の事後サンプル ${N_SAMPLES} 個。"
  echo
  echo '```'
  echo "$OUT"
  echo '```'
  echo
  echo "- HEAD: \`$(git rev-parse --short HEAD)\`、ホスト $(hostname)、$(date '+%Y-%m-%d %H:%M:%S %Z')"
} > "$REPORT"

git add "$REPORT"
[ -f "$JSON" ] && git add "$JSON"
git -c user.name="Keisuke Nishioka" -c user.email="kei128608@gmail.com" \
  commit -q -m "docs: Fn ノックアウト（${TODAY} / ${LABEL}）" || exit 0
git pull --rebase --quiet origin "$(git rev-parse --abbrev-ref HEAD)" || true
git push --quiet origin "$(git rev-parse --abbrev-ref HEAD)" || echo "警告: push に失敗"
