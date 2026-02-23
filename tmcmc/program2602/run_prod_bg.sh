#!/usr/bin/env bash
set -euo pipefail

# Background runner + log watcher for long production runs.
#
# Usage:
#   ./tmcmc/run_prod_bg.sh start
#   ./tmcmc/run_prod_bg.sh watch
#   ./tmcmc/run_prod_bg.sh status
#
# You can override RUN_ID and args via environment variables:
#   RUN_ID=prod_np5000_ns60 ./tmcmc/run_prod_bg.sh start
#
# Default command (as requested):
#   python tmcmc/run_pipeline.py --mode paper --run-id prod_np5000_ns60 --seed 42 \
#     --n-particles 5000 --n-stages 60 --n-mutation-steps 8 --n-chains 5

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-"${ROOT_DIR}/tmcmc/_runs"}"
RUN_ID="${RUN_ID:-prod_np5000_ns60}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/run.log"
PID_FILE="${RUN_DIR}/run.pid"

CMD=(
  # -u / PYTHONUNBUFFERED=1: flush logs to run.log immediately
  env PYTHONUNBUFFERED=1 python -u "${ROOT_DIR}/tmcmc/run_pipeline.py"
  --mode paper
  --run-id "${RUN_ID}"
  --seed 42
  --n-particles 5000
  --n-stages 60
  --n-mutation-steps 8
  --n-chains 5
)

is_running() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

cmd_start() {
  mkdir -p "${RUN_DIR}"

  if is_running; then
    echo "Already running: RUN_ID=${RUN_ID}"
    echo "  RUN_DIR=${RUN_DIR}"
    echo "  PID=$(cat "${PID_FILE}")"
    echo "  LOG=${LOG_FILE}"
    exit 0
  fi

  echo "Starting background run..."
  echo "  RUN_ID=${RUN_ID}"
  echo "  RUN_DIR=${RUN_DIR}"
  echo "  LOG=${LOG_FILE}"
  echo "  CMD: ${CMD[*]}"
  echo

  # Start in background, save pid.
  nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"

  echo "Started."
  echo "  PID=$(cat "${PID_FILE}")"
  echo
  echo "Watch progress:"
  echo "  ./tmcmc/run_prod_bg.sh watch"
  echo
  echo "Check status:"
  echo "  ./tmcmc/run_prod_bg.sh status"
}

cmd_watch() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "Log not found yet: ${LOG_FILE}"
    echo "Run start first:"
    echo "  ./tmcmc/run_prod_bg.sh start"
    exit 1
  fi
  echo "Tailing log: ${LOG_FILE}"
  echo "(Ctrl-C to stop watching)"
  tail -f "${LOG_FILE}"
}

cmd_status() {
  echo "RUN_ID=${RUN_ID}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "LOG=${LOG_FILE}"
  if [[ -f "${PID_FILE}" ]]; then
    echo "PID=$(cat "${PID_FILE}")"
  else
    echo "PID=missing"
  fi

  if is_running; then
    echo "STATUS=running"
  else
    echo "STATUS=not_running"
  fi

  if [[ -f "${RUN_DIR}/REPORT.md" ]]; then
    echo "REPORT=present (${RUN_DIR}/REPORT.md)"
  else
    echo "REPORT=missing"
  fi
}

case "${1:-}" in
  start)  cmd_start ;;
  watch)  cmd_watch ;;
  status) cmd_status ;;
  *)
    echo "Usage: $0 {start|watch|status}"
    exit 2
    ;;
esac

