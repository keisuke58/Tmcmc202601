#!/usr/bin/env bash
# Check the status of running sweeps
#
# Usage:
#   bash check_sweep_status.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/sweep_logs"

echo "=== Sweep Status Check ==="
echo ""

# Check screen sessions
if command -v screen &> /dev/null; then
  SCREEN_SESSIONS=$(screen -ls 2>/dev/null | grep "sweep_m1_" || true)
  if [ -n "${SCREEN_SESSIONS}" ]; then
    echo "📺 Screen sessions:"
    echo "${SCREEN_SESSIONS}"
    echo ""
  fi
fi

# Check tmux sessions
if command -v tmux &> /dev/null; then
  TMUX_SESSIONS=$(tmux ls 2>/dev/null | grep "sweep_m1_" || true)
  if [ -n "${TMUX_SESSIONS}" ]; then
    echo "🖥️  Tmux sessions:"
    echo "${TMUX_SESSIONS}"
    echo ""
  fi
fi

# Check nohup processes
if [ -d "${LOGS_DIR}" ]; then
  PID_FILES=("${LOGS_DIR}"/sweep_*.pid)
  if [ -e "${PID_FILES[0]}" ]; then
    echo "🔄 Nohup processes:"
    for pid_file in "${PID_FILES[@]}"; do
      if [ -f "${pid_file}" ]; then
        PID=$(cat "${pid_file}" 2>/dev/null || echo "")
        if [ -n "${PID}" ] && ps -p "${PID}" > /dev/null 2>&1; then
          LOG_FILE="${pid_file%.pid}.log"
          echo "  PID: ${PID}"
          echo "  Log: ${LOG_FILE}"
          echo "  Status: Running"
          echo ""
        fi
      fi
    done
  fi
fi

# Show recent log files
if [ -d "${LOGS_DIR}" ]; then
  echo "📄 Recent log files:"
  ls -lht "${LOGS_DIR}"/sweep_*.log 2>/dev/null | head -5 || echo "  No log files found"
  echo ""
fi

# Check running sweep processes
echo "🔍 Running sweep processes:"
ps aux | grep -E "[s]weep_m1\.sh|[r]un_sweep_background" || echo "  No sweep processes found"
echo ""

# Show disk usage
if [ -d "${SCRIPT_DIR}/tmcmc/_runs" ]; then
  echo "💾 Disk usage of _runs directory:"
  du -sh "${SCRIPT_DIR}/tmcmc/_runs" 2>/dev/null || echo "  Cannot check disk usage"
  echo ""
fi

echo "=== End of Status Check ==="
