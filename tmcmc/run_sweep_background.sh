#!/usr/bin/env bash
# Background execution wrapper for sweep_m1.sh
# This script ensures the sweep continues even if the terminal is disconnected
#
# Usage:
#   bash run_sweep_background.sh [method]
#   method: screen (default), tmux, or nohup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/sweep_m1.sh"
METHOD="${1:-screen}"

# Create logs directory
LOGS_DIR="${SCRIPT_DIR}/sweep_logs"
mkdir -p "${LOGS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOGS_DIR}/sweep_${TIMESTAMP}.log"
PID_FILE="${LOGS_DIR}/sweep_${TIMESTAMP}.pid"

echo "Starting sweep in background mode: ${METHOD}"
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"

case "${METHOD}" in
  screen)
    if ! command -v screen &> /dev/null; then
      echo "ERROR: screen is not installed. Install with: sudo yum install screen (or apt-get install screen)"
      exit 1
    fi
    SESSION_NAME="sweep_m1_${TIMESTAMP}"
    echo "Creating screen session: ${SESSION_NAME}"
    screen -dmS "${SESSION_NAME}" bash -c "cd '${SCRIPT_DIR}' && echo 'Sweep started at \$(date)' > '${LOG_FILE}' && bash '${SWEEP_SCRIPT}' 2>&1 | tee -a '${LOG_FILE}' && echo 'Sweep completed at \$(date)' >> '${LOG_FILE}'"
    # Get screen session PID
    SCREEN_PID=$(screen -list | grep "${SESSION_NAME}" | head -1 | awk -F'.' '{print $1}' | tr -d ' ')
    if [ -n "${SCREEN_PID}" ]; then
      echo "${SCREEN_PID}" > "${PID_FILE}"
    else
      echo "session" > "${PID_FILE}"
    fi
    echo ""
    echo "✅ Screen session created: ${SESSION_NAME}"
    echo "To attach: screen -r ${SESSION_NAME}"
    echo "To list sessions: screen -ls"
    echo "To detach: Press Ctrl+A, then D"
    ;;
    
  tmux)
    if ! command -v tmux &> /dev/null; then
      echo "ERROR: tmux is not installed. Install with: sudo yum install tmux (or apt-get install tmux)"
      exit 1
    fi
    SESSION_NAME="sweep_m1_${TIMESTAMP}"
    echo "Creating tmux session: ${SESSION_NAME}"
    tmux new-session -d -s "${SESSION_NAME}" "cd '${SCRIPT_DIR}' && echo 'Sweep started at \$(date)' > '${LOG_FILE}' && bash '${SWEEP_SCRIPT}' 2>&1 | tee -a '${LOG_FILE}' && echo 'Sweep completed at \$(date)' >> '${LOG_FILE}'"
    # Get tmux session PID
    TMUX_PID=$(tmux list-sessions -F "#{session_id} #{session_name}" 2>/dev/null | grep "${SESSION_NAME}" | awk '{print $1}' | sed 's/%//' || echo "")
    if [ -n "${TMUX_PID}" ]; then
      echo "${TMUX_PID}" > "${PID_FILE}"
    else
      echo "session" > "${PID_FILE}"
    fi
    echo ""
    echo "✅ Tmux session created: ${SESSION_NAME}"
    echo "To attach: tmux attach -t ${SESSION_NAME}"
    echo "To list sessions: tmux ls"
    echo "To detach: Press Ctrl+B, then D"
    ;;
    
  nohup)
    echo "Starting with nohup..."
    cd "${SCRIPT_DIR}"
    nohup bash "${SWEEP_SCRIPT}" > "${LOG_FILE}" 2>&1 &
    SWEEP_PID=$!
    echo $SWEEP_PID > "${PID_FILE}"
    echo ""
    echo "✅ Sweep started with nohup (PID: ${SWEEP_PID})"
    echo "Log file: ${LOG_FILE}"
    echo "To check status: tail -f ${LOG_FILE}"
    echo "To check if running: ps -p ${SWEEP_PID}"
    ;;
    
  *)
    echo "ERROR: Unknown method: ${METHOD}"
    echo "Available methods: screen, tmux, nohup"
    exit 1
    ;;
esac

echo ""
echo "📋 Summary:"
echo "  Method: ${METHOD}"
echo "  Log: ${LOG_FILE}"
if [ -f "${PID_FILE}" ]; then
  echo "  PID/Session: $(cat ${PID_FILE})"
else
  echo "  PID/Session: N/A"
fi
echo ""
echo "💡 Tips:"
echo "  - The sweep will continue even if you disconnect"
echo "  - Check progress: tail -f ${LOG_FILE}"
echo "  - Monitor resource usage: htop or top"
