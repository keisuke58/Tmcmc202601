#!/bin/bash
#PBS -N polish_maxlogL
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q default
#PBS -j oe
#PBS -o polish_maxlogL_${PBS_JOBID}.log

# ============================================================
# tools/polish_max_logL.py を GPU で回す（判定 5 を「磨いた max logL」で読み直す）。
# 上位 TOP_K 粒子を 1 回の L-BFGS-B でまとめて磨く。勾配は ODE を通すので CPU では遅い。
#
# 使い方（ノードは必ず明示。celtic03 は使わない）:
#   qsub -l nodes=1:ppn=1:gpus=1:stuttgart01 -l walltime=08:00:00 \
#        -v RUN_GLOB='DH_ident*mut80*',LABEL=dh_ident tools/polish_max_logL_job.sh
# 結果: docs/handoff/gpu_<日付>_polish_<LABEL>.{md,json} を書いて push する
# ============================================================
set -uo pipefail

REPO="$HOME/Tmcmc202601"
RUN_GLOB="${RUN_GLOB:?RUN_GLOB を渡す（例 'DH_ident*mut80*'）}"
LABEL="${LABEL:-polish}"
TOP_K="${TOP_K:-20}"
MAXITER="${MAXITER:-300}"
PYTHON="${PYTHON:-/home/nishioka/miniforge3/envs/klempt_fem2/bin/python3}"

cd "$REPO" || exit 1
# shellcheck source=tools/git_sync.sh
source "$REPO/tools/git_sync.sh"
git_sync_latest

if [ -n "${PBS_GPUFILE:-}" ] && [ -f "${PBS_GPUFILE}" ]; then
  export CUDA_VISIBLE_DEVICES="$(grep -oE 'gpu[0-9]+' "$PBS_GPUFILE" | head -1 | grep -oE '[0-9]+')"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

TODAY="$(date +%Y-%m-%d)"
REPORT="docs/handoff/gpu_${TODAY}_polish_${LABEL}.md"
JSON="docs/handoff/gpu_${TODAY}_polish_${LABEL}.json"
# 出力は必ず先にファイルへ流す。PBS は実行中ジョブの stdout をバッファするので、
# `OUT="$(...)"` でため込むと walltime で殺されたときに何も残らない
# （2026-10-07 のジョブ 3129 は 8 時間走って進捗ゼロのログだけを残した）。
# python -u で行ごとに吐かせ、tee で随時 RAWLOG に落とす。
RAWLOG="polish_maxlogL_${PBS_JOBID:-local}_raw.log"
echo "生ログ: $REPO/$RAWLOG（実行中も読める）"
# POLISH_DEVICE=gpu: estimator の早期デバイス判定に gpu を渡す（既定 cpu だと GPU が隠れる）
POLISH_DEVICE=gpu "$PYTHON" -u tools/polish_max_logL.py data_5species/main/_runs/paper_gateoff \
  --glob "$RUN_GLOB" --top-k "$TOP_K" --maxiter "$MAXITER" --json "$JSON" 2>&1 \
  | tee "$RAWLOG"
RC="${PIPESTATUS[0]}"
OUT="$(grep -v -E '^(INFO|WARNING):' "$RAWLOG" | tail -40)"
[ "$RC" -eq 0 ] || OUT="$OUT

警告: polish_max_logL.py が終了コード $RC で終わった（上は途中までの出力）"
grep -q 'JAX devices: \[Cuda\|JAX devices: \[cuda' "$RAWLOG" || echo "警告: GPU で走っていない可能性（出力の JAX devices を確認）"

{
  echo "# GPU 側 → クラウド側: max logL の磨き直し（${TODAY} / ${LABEL}）"
  echo
  echo "\`tools/polish_max_logL_job.sh\` が自動で書いた。対象 \`${RUN_GLOB}\`、上位 ${TOP_K} 粒子、L-BFGS-B 最大 ${MAXITER} 反復。"
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
  commit -q -m "docs: max logL の磨き直し（${TODAY} / ${LABEL}）" || exit 0
git_push_current
