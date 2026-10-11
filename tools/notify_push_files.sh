#!/bin/bash
#PBS -N notify_files
#PBS -l nodes=1:ppn=1
#PBS -q default
#PBS -j oe
#PBS -o logs/pbs/${PBS_JOBNAME}_${PBS_JOBID}.log

# ============================================================
# 監視対象のジョブが終わったら、そのジョブが書いたファイルを commit して push する。
# notify_cloud_runs.sh は _runs/ の判定用。こちらは fit_ceiling のように
# docs/handoff/ に結果を直接書くだけのジョブ向け（クラウド側は git pull で読む）。
#
# 使い方（依存ジョブとして投入する）:
#   qsub -W depend=afterany:3369.copaam \
#        -v "FILES=docs/handoff/gpu_2026-10-11_fit_ceiling_CH_part1.txt,MSG=fit_ceiling CH (1) 完了" \
#        tools/notify_push_files.sh
#
# 環境変数:
#   FILES  push するファイル（repo からの相対パス。複数は ";" 区切り。qsub -v はカンマで変数を区切るため）
#   MSG    commit メッセージ（既定 "handoff: ジョブ結果"）
# ============================================================
set -uo pipefail

REPO="$HOME/Tmcmc202601"
cd "$REPO" || exit 1
source "$REPO/tools/git_sync.sh"
export PATH="/usr/local/bin:$PATH"

FILES="${FILES:?FILES が必要}"
MSG="${MSG:-handoff: ジョブ結果}"

git_sync_latest
ADDED=0
IFS=';' read -ra LIST <<< "$FILES"
for F in "${LIST[@]}"; do
  if [ -f "$F" ]; then git add "$F" && ADDED=1 && echo "add: $F"
  else echo "警告: $F が無い（ジョブが書く前に死んだ可能性）"; fi
done

if [ "$ADDED" = "1" ] && ! git diff --cached --quiet; then
  git commit -q -m "$MSG（自動通知 $(date +%Y-%m-%d)）" && git_push_current
else
  echo "commit するものが無い"
fi
