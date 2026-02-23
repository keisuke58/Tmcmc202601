#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep runner for M1.
#
# What it does:
# - Runs tmcmc/run_pipeline.py for a grid of (sigma_obs, cov_rel, n_particles)
# - Creates run_ids like: sig0020_cov0005_np0500_ns20
# - Stores ALL run outputs under: tmcmc/_runs/<sweep_prefix>/<run_id>/
# - Appends a compact summary to: tmcmc/_runs/<sweep_prefix>/sweep_summary.csv
#
# Notes:
# - stdout/stderr of each run is already persisted by run_pipeline.py into each run directory.
# - This script only orchestrates + summarizes.
# - Runs are executed in parallel (configurable) and best run is auto-selected at the end.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${RUNS_ROOT:-"${ROOT_DIR}/tmcmc/_runs"}"

MODE="${MODE:-debug}"
DEBUG_LEVEL="${DEBUG_LEVEL:-MINIMAL}"
MODELS="${MODELS:-M1}"
N_STAGES="${N_STAGES:-30}"  # Updated: 20 → 30 for better convergence

# Grids (space-separated lists)
SIGMA_LIST="${SIGMA_LIST:-0.01}"
COVREL_LIST="${COVREL_LIST:-0.02}"
NP_LIST="${NP_LIST:-2000 3000 5000}"  # Updated: 300,500,1000 → 2000,3000,5000 for better ESS

# Parallelism
# - Keep this modest on shared servers. Typical: 2-4.
# - This system has 12 cores, 62GB RAM, so 4-8 is safe.
MAX_JOBS="${MAX_JOBS:-8}"

# Prefix folder (under runs root) to store sweep outputs
SWEEP_PREFIX="${SWEEP_PREFIX:-sweep_m1_$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${RUNS_ROOT}/${SWEEP_PREFIX}"
mkdir -p "${SWEEP_DIR}"

ROWS_DIR="${SWEEP_DIR}/rows"
mkdir -p "${ROWS_DIR}"

SUMMARY_CSV="${SWEEP_DIR}/sweep_summary.csv"
SUMMARY_HEADER="run_id,mode,models,n_particles,n_stages,sigma_obs,cov_rel,exit_code,status,ess_min,rmse_total_map,map_error,rom_error_final,run_dir,report_path"

require_runs_root_flag() {
  # Guard: if run_pipeline.py doesn't support --runs-root, the sweep can't isolate outputs.
  if ! python "${ROOT_DIR}/tmcmc/run_pipeline.py" --help 2>/dev/null | grep -q -- "--runs-root"; then
    echo "ERROR: tmcmc/run_pipeline.py does not support --runs-root; cannot safely run sweep." >&2
    echo "       Please update run_pipeline.py or run without parallel isolation." >&2
    exit 2
  fi
}

wait_one() {
  # wait for one background job to finish (portable fallback for older bash without wait -n)
  if wait -n 2>/dev/null; then
    return 0
  fi
  local pids=()
  # jobs is a bash builtin; mapfile is bash builtin.
  mapfile -t pids < <(jobs -pr)
  if ((${#pids[@]} > 0)); then
    wait "${pids[0]}"
  fi
}

python_json() {
  # Usage: python_json <json_file> <python_expr_returning_value>
  # Example: python_json metrics.json 'd["errors"]["m1_map_error"]'
  local json_file="$1"
  local expr="$2"
  python - "$json_file" "$expr" <<'PY'
import json, sys
from pathlib import Path
path, expr = sys.argv[1], sys.argv[2]
if not Path(path).exists():
    print("missing")
    raise SystemExit(0)
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)
try:
    v = eval(expr, {"__builtins__": {}}, {"d": d})
except Exception:
    v = None
if v is None:
    print("missing")
else:
    print(v)
PY
}

extract_report_field() {
  # Best-effort extraction from REPORT.md (avoid jq dependency).
  # Usage: extract_report_field <report.md> <regex>
  local report="$1"
  local regex="$2"
  python - "$report" "$regex" <<'PY'
import re, sys
from pathlib import Path
path, pat = sys.argv[1], sys.argv[2]
if not Path(path).exists():
    print("missing")
    raise SystemExit(0)
txt = open(path, "r", encoding="utf-8").read()
m = re.search(pat, txt, flags=re.MULTILINE)
print(m.group(1) if m else "missing")
PY
}

normalize_tag() {
  # Turn "0.02" -> "0020", "0.005" -> "0005"
  python - "$1" <<'PY'
import sys
s = sys.argv[1].strip()
f = float(s)
print(f"{int(round(f*1000)):04d}")
PY
}

parse_key_metrics_row() {
  # Parse the table row starting with "| M1 |" from REPORT.md and return:
  # rmse_total_map,rom_error_final,ess_min
  local report="$1"
  python - "$report" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("missing,missing,missing")
    raise SystemExit(0)
rmse = rom = ess = "missing"
for line in path.read_text(encoding="utf-8").splitlines():
    if line.startswith("| M1 |"):
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        # cols: Model, RMSE_total(MAP), MAE_total(MAP), max_abs(MAP), rom_error_final, ESS_min, accept_rate_mean, beta_final, beta_stages
        if len(cols) > 1: rmse = cols[1]
        if len(cols) > 4: rom  = cols[4]
        if len(cols) > 5: ess  = cols[5]
        break
print(f"{rmse},{rom},{ess}")
PY
}

run_one() (
  # Run each job in an isolated subshell so failures don't kill the parent sweep.
  set +e
  set -u -o pipefail

  local sigma="$1"
  local cov="$2"
  local np="$3"

  local sig_tag cov_tag run_id run_dir exit_code report metrics status rmse_total_map rom_final ess_min map_error triple
  sig_tag="$(normalize_tag "${sigma}")"
  cov_tag="$(normalize_tag "${cov}")"

  run_id="sig${sig_tag}_cov${cov_tag}_np$(printf '%04d' "${np}")_ns$(printf '%02d' "${N_STAGES}")"
  run_dir="${SWEEP_DIR}/${run_id}"
  exit_code=0
  status="missing"
  rmse_total_map="missing"
  rom_final="missing"
  ess_min="missing"
  map_error="missing"
  report="${run_dir}/REPORT.md"
  metrics="${run_dir}/metrics.json"

  # Always write a row even on unexpected failures.
  trap '
    if [[ "${exit_code}" -ne 0 ]]; then
      status="FAIL"
    fi
    echo "${run_id},${MODE},${MODELS},${np},${N_STAGES},${sigma},${cov},${exit_code},${status},${ess_min},${rmse_total_map},${map_error},${rom_final},${run_dir},${report}" \
      > "${ROWS_DIR}/${run_id}.csv"
  ' EXIT

  if [[ -f "${run_dir}/REPORT.md" ]]; then
    echo "[skip] ${run_id} (already has REPORT.md)"
  else
    echo "[run ] ${run_id}"
    python "${ROOT_DIR}/tmcmc/run_pipeline.py" \
      --runs-root "${SWEEP_DIR}" \
      --mode "${MODE}" \
      --debug-level "${DEBUG_LEVEL}" \
      --run-id "${run_id}" \
      --models "${MODELS}" \
      --n-particles "${np}" \
      --n-stages "${N_STAGES}" \
      --sigma-obs "${sigma}" \
      --cov-rel "${cov}"
    exit_code=$?
    if [[ "${exit_code}" -ne 0 ]]; then
      echo "[fail] ${run_id} (exit_code=${exit_code})"
      status="FAIL"
      return 0
    fi
  fi

  # Parse outputs (best-effort)
  status="$(extract_report_field "${report}" '^- \\*\\*status\\*\\*: \\*\\*(PASS|WARN|FAIL)\\*\\*' || echo missing)"
  if [[ "${exit_code}" -ne 0 ]]; then
    status="FAIL"
  fi
  triple="$(parse_key_metrics_row "${report}" || echo 'missing,missing,missing')"
  rmse_total_map="${triple%%,*}"
  triple="${triple#*,}"
  rom_final="${triple%%,*}"
  ess_min="${triple#*,}"
  map_error="$(python_json "${metrics}" '((d.get("errors") or {}).get("m1_map_error"))' || echo missing)"
)

select_best() {
  local summary_csv="$1"
  python - "$summary_csv" <<'PY'
import csv, math, sys
from pathlib import Path

path = Path(sys.argv[1])
rows = []
with path.open("r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    fieldnames = r.fieldnames or []
    for row in r:
        rows.append(row)

def fnum(x: str) -> float:
    try:
        if x is None: return math.inf
        x = x.strip()
        if x == "" or x.lower() == "missing": return math.inf
        return float(x)
    except Exception:
        return math.inf

status_rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
def key(row):
    # If the process failed, force FAIL regardless of parsed status.
    try:
        if int((row.get("exit_code") or "0").strip() or "0") != 0:
            status = "FAIL"
        else:
            status = (row.get("status") or "missing").strip()
    except Exception:
        status = (row.get("status") or "missing").strip()
    rank = status_rank.get(status, 3)
    rmse = fnum(row.get("rmse_total_map") or "missing")
    map_err = fnum(row.get("map_error") or "missing")
    ess = fnum(row.get("ess_min") or "missing")
    # Prefer: PASS > WARN > FAIL, then lowest RMSE, then lowest MAP error, then highest ESS
    return (rank, rmse, map_err, -ess)

if not rows:
    print("No rows to select from.")
    raise SystemExit(0)

best = min(rows, key=key)
out_dir = path.parent
(out_dir / "best_run_id.txt").write_text(best.get("run_id", "missing") + "\n", encoding="utf-8")
header = fieldnames if fieldnames else list(best.keys())
line = ",".join(best.get(h, "") for h in header)
(out_dir / "best_row.csv").write_text(",".join(header) + "\n" + line + "\n", encoding="utf-8")
print(f"BEST run_id={best.get('run_id')} status={best.get('status')} rmse={best.get('rmse_total_map')} map_error={best.get('map_error')} ess={best.get('ess_min')}")
PY
}

require_runs_root_flag

echo "Sweep dir: ${SWEEP_DIR}"
echo "Grid:"
echo "  sigma_obs: ${SIGMA_LIST}"
echo "  cov_rel:   ${COVREL_LIST}"
echo "  particles: ${NP_LIST}"
echo "Parallel:"
echo "  max_jobs:  ${MAX_JOBS}"
echo ""

# Launch runs in parallel with a simple job limiter.
running=0
for sigma in ${SIGMA_LIST}; do
  for cov in ${COVREL_LIST}; do
    for np in ${NP_LIST}; do
      run_one "${sigma}" "${cov}" "${np}" &
      running=$((running + 1))
      if [[ "${running}" -ge "${MAX_JOBS}" ]]; then
        # wait for at least one job to finish
        wait_one
        running=$((running - 1))
      fi
    done
  done
done
wait

# Assemble summary.csv deterministically
echo "${SUMMARY_HEADER}" > "${SUMMARY_CSV}"
files=("${ROWS_DIR}"/*.csv)
if [[ -e "${files[0]}" ]]; then
  # Stable order for reproducibility
  LC_ALL=C printf '%s\n' "${files[@]}" | LC_ALL=C sort | while IFS= read -r f; do
    cat "$f" >> "${SUMMARY_CSV}"
  done
fi

select_best "${SUMMARY_CSV}"

echo ""
echo "Done."
echo "Summary: ${SUMMARY_CSV}"
