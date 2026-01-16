#!/usr/bin/env bash
set -euo pipefail

# 24h monitor for TMCMC runs:
# - hourly:   tmcmc/audit_runs.py -> _runs/_audit/{runs_audit.md,runs_audit.csv} (+ timestamped logs)
# - every N:  disk + memory snapshot -> _runs/_monitor/resource_log.tsv
# - daily via cron: start once per day, run for 24 hours, then exit

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-"${ROOT_DIR}/tmcmc/_runs"}"
PYTHON_BIN="${PYTHON_BIN:-python}"

AUDIT_DIR="${RUNS_ROOT}/_audit"
MON_DIR="${RUNS_ROOT}/_monitor"
mkdir -p "${AUDIT_DIR}" "${MON_DIR}"

# Prevent double-start (e.g. cron overlap)
LOCK_DIR="${MON_DIR}/.lock_monitor_24h"
if mkdir "${LOCK_DIR}" 2>/dev/null; then
  :
else
  echo "monitor_24h.sh: lock exists (${LOCK_DIR}); another monitor is running. Exiting."
  exit 0
fi
cleanup() { rm -rf "${LOCK_DIR}" || true; }
trap cleanup EXIT INT TERM

AUDIT_INTERVAL_SEC="${AUDIT_INTERVAL_SEC:-3600}"     # 1 hour
RESOURCE_INTERVAL_SEC="${RESOURCE_INTERVAL_SEC:-600}" # 10 minutes
SLEEP_TICK_SEC="${SLEEP_TICK_SEC:-30}"               # loop tick

START_EPOCH="$(date +%s)"
END_EPOCH="$((START_EPOCH + 24*3600))"

HOST="$(hostname 2>/dev/null || echo unknown)"

resource_log="${MON_DIR}/resource_log.tsv"
if [[ ! -f "${resource_log}" ]]; then
  printf "ts_iso\tepoch\thost\tdf_size_B\tdf_used_B\tdf_avail_B\tmem_total_B\tmem_used_B\tmem_avail_B\tload1\tload5\tload15\n" > "${resource_log}"
fi

write_resource_snapshot() {
  local ts_iso epoch
  ts_iso="$(date -Iseconds)"
  epoch="$(date +%s)"

  # Disk snapshot (bytes)
  local df_line filesystem size used avail pcent mount
  df_line="$(df -P -B1 "${RUNS_ROOT}" | awk 'NR==2{print $0}')"
  filesystem="$(awk '{print $1}' <<<"${df_line}")"
  size="$(awk '{print $2}' <<<"${df_line}")"
  used="$(awk '{print $3}' <<<"${df_line}")"
  avail="$(awk '{print $4}' <<<"${df_line}")"
  pcent="$(awk '{print $5}' <<<"${df_line}")"
  mount="$(awk '{print $6}' <<<"${df_line}")"

  # Memory snapshot (bytes)
  local mem_total mem_used mem_avail
  mem_total="$(free -b | awk '/^Mem:/{print $2}')"
  mem_used="$(free -b | awk '/^Mem:/{print $3}')"
  mem_avail="$(free -b | awk '/^Mem:/{print $7}')"

  # Load average
  local load1 load5 load15
  read -r load1 load5 load15 _ < /proc/loadavg || { load1="0"; load5="0"; load15="0"; }

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${ts_iso}" "${epoch}" "${HOST}" \
    "${size}" "${used}" "${avail}" \
    "${mem_total}" "${mem_used}" "${mem_avail}" \
    "${load1}" "${load5}" "${load15}" \
    >> "${resource_log}"

  # Compact human-readable line (optional)
  printf "[%s] disk: used=%s avail=%s (%s on %s)  mem: used=%s avail=%s  load=%s %s %s\n" \
    "${ts_iso}" "${used}" "${avail}" "${pcent}" "${mount}" "${mem_used}" "${mem_avail}" "${load1}" "${load5}" "${load15}" \
    >> "${MON_DIR}/monitor.log"
}

run_audit() {
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local md_out csv_out log_out
  md_out="${AUDIT_DIR}/runs_audit.md"
  csv_out="${AUDIT_DIR}/runs_audit.csv"
  log_out="${AUDIT_DIR}/audit_${ts}.log"

  {
    echo "=== audit start: $(date -Iseconds) ==="
    echo "RUNS_ROOT=${RUNS_ROOT}"
    echo "PYTHON_BIN=${PYTHON_BIN} ($(command -v "${PYTHON_BIN}" 2>/dev/null || echo not-found))"
    "${PYTHON_BIN}" "${ROOT_DIR}/tmcmc/audit_runs.py" \
      --runs-root "${RUNS_ROOT}" \
      --md-out "${md_out}" \
      --csv-out "${csv_out}"
    echo "=== audit end:   $(date -Iseconds) ==="
  } >> "${log_out}" 2>&1
}

echo "monitor_24h.sh: start $(date -Iseconds) host=${HOST}"
echo "monitor_24h.sh: RUNS_ROOT=${RUNS_ROOT}"
echo "monitor_24h.sh: AUDIT_INTERVAL_SEC=${AUDIT_INTERVAL_SEC} RESOURCE_INTERVAL_SEC=${RESOURCE_INTERVAL_SEC}"
echo "monitor_24h.sh: will stop at $(date -Iseconds -d "@${END_EPOCH}") (epoch=${END_EPOCH})"

next_audit="${START_EPOCH}"
next_resource="${START_EPOCH}"

# Run immediately once at start.
write_resource_snapshot || true
run_audit || true
next_resource="$((START_EPOCH + RESOURCE_INTERVAL_SEC))"
next_audit="$((START_EPOCH + AUDIT_INTERVAL_SEC))"

while true; do
  now="$(date +%s)"
  if [[ "${now}" -ge "${END_EPOCH}" ]]; then
    break
  fi

  if [[ "${now}" -ge "${next_resource}" ]]; then
    write_resource_snapshot || true
    next_resource="$((next_resource + RESOURCE_INTERVAL_SEC))"
  fi

  if [[ "${now}" -ge "${next_audit}" ]]; then
    run_audit || true
    next_audit="$((next_audit + AUDIT_INTERVAL_SEC))"
  fi

  sleep "${SLEEP_TICK_SEC}"
done

echo "monitor_24h.sh: done  $(date -Iseconds)"
