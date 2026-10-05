#!/bin/bash
#PBS -N notify_cloud
#PBS -l nodes=1:ppn=1
#PBS -q default
#PBS -j oe
#PBS -o notify_cloud_${PBS_JOBID}.log

# ============================================================
# 監視対象の run が終わったら、判定結果を docs/handoff/ に書いて push する。
# クラウド側 Claude は git pull でこれを読む（チャット経由の連絡はしない、が取り決め）。
#
# 使い方: 依存ジョブとして投入する（監視対象が全部終わってから走る）
#   qsub -W depend=afterany:3080.copaam:3081.copaam tools/notify_cloud_runs.sh
#
# 環境変数:
#   RUNS_ROOT  判定するディレクトリ（既定 data_5species/main/_runs/paper_gateoff）
#   RUN_GLOB   今回の run を見分ける glob（既定 *mut80*）
#   DRY_RUN=1  commit / push せず、書く内容を標準出力に出すだけ
# ============================================================
set -uo pipefail

REPO="$HOME/Tmcmc202601"
RUNS_ROOT="${RUNS_ROOT:-data_5species/main/_runs/paper_gateoff}"
RUN_GLOB="${RUN_GLOB:-*mut80*}"
# 同じ日に複数回通知するときに report が上書きされないようにする識別子（波ごとに渡す）
LABEL="${LABEL:-runs}"
DRY_RUN="${DRY_RUN:-0}"
cd "$REPO" || exit 1

TODAY="$(date +%Y-%m-%d)"
REPORT="docs/handoff/gpu_${TODAY}_${LABEL}.md"
CSV="docs/handoff/gpu_${TODAY}_${LABEL}.csv"
CHECK_OUT="$(python3 tools/check_paper_runs.py "$RUNS_ROOT" --glob "$RUN_GLOB" 2>&1)"
VERDICT="$(echo "$CHECK_OUT" | grep -E "^(全群 PASS|FAIL を含む群)" | tail -1)"
[ -z "$VERDICT" ] && VERDICT="（判定スクリプトが結論行を出さなかった。出力をそのまま読むこと）"

# 今回の run の 1 行サマリ（終わっていない run はディレクトリが無い）
FOUND="$(ls -d $RUNS_ROOT/$RUN_GLOB 2>/dev/null | wc -l)"
MOVES="$(python3 - "$RUNS_ROOT" "$RUN_GLOB" <<'PY'
import json, sys, glob, os
root, pat = sys.argv[1], sys.argv[2]
rows = []
for d in sorted(glob.glob(os.path.join(root, pat))):
    f = os.path.join(d, "run_record.json")
    if not os.path.exists(f):
        rows.append(f"| {os.path.basename(d)} | run_record.json が無い（異常終了） | | | |")
        continue
    r = json.load(open(f))
    n_free = max(len(r.get("free_dims") or []), 1)
    m = r.get("moves_per_particle_mean")
    m = float(m) if m is not None else float(r["mean_accept"]) * r["args"]["n_mutation_steps"]
    rows.append(
        f"| {os.path.basename(d)} | {r['n_stages']} | {r['mean_accept']:.3f} | "
        f"{m:.1f}/stage ({m * r['n_stages'] / n_free:.1f}/次元) | {r['max_logL']:.2f} |"
    )
print("\n".join(rows) if rows else "| （出力ディレクトリが見つからない） | | | | |")
PY
)"

# RMSE・Pg D21/D15・a33/a45 の事後を回収する（CSV も残す）。
# eval_gateoff_runs.py は使わない: colab_package の前進モデル（n_hill=2・K_hill=0 固定）で
# 予測するので、論文パイプラインの run（とくにゲート ON）では数字が別物になる。
# eval_paper_runs.py は run_record の実効値と estimator と同じモジュールで予測し、
# estimator の RMSE と照合する。
EVAL_OUT="$(timeout 1800 python3 tools/eval_paper_runs.py "$RUNS_ROOT" --glob "$RUN_GLOB" \
  --csv "$REPO/$CSV" 2>&1 | grep -v -E '^(INFO|WARNING):' | tail -40)"
[ -z "$EVAL_OUT" ] && EVAL_OUT="（eval_paper_runs.py が出力なし）"

{
  echo "# GPU 側 → クラウド側: run 終了の自動通知（${TODAY} / ${LABEL}）"
  echo
  echo "\`tools/notify_cloud_runs.sh\` が PBS の依存ジョブとして自動で書いた。"
  echo "監視対象: \`$RUNS_ROOT/$RUN_GLOB\`（出力ディレクトリ $FOUND 個）"
  echo
  echo "## 結論"
  echo
  echo "**$VERDICT**"
  echo
  echo "## 今回の run"
  echo
  echo "| run | stages | 受理率 | 1 粒子の移動 | max logL |"
  echo "|---|---|---|---|---|"
  echo "$MOVES"
  echo
  echo "判定 2b の目標は **5 回/次元**。下回っていれば混合不足なので解釈しない。"
  echo
  echo "## 回収した数字（tools/eval_paper_runs.py）"
  echo
  echo "RMSE（estimator の値と照合）・Pg D21/15・a33/a45 の事後 5/50/95% と箱の端の割合（run 自身の箱）。"
  echo "**判定を通っていない群の数字は使わない**（seed ごとに別の領域を見ているだけなので比較にならない）。"
  echo
  echo '```'
  echo "$EVAL_OUT"
  echo '```'
  echo
  echo "全列は \`${CSV}\` にある。"
  echo
  echo "## check_paper_runs.py の出力（そのまま）"
  echo
  echo '```'
  echo "$CHECK_OUT"
  echo '```'
  echo
  echo "## 直近の run の git"
  echo
  echo "- HEAD: \`$(git rev-parse --short HEAD)\` $(git log -1 --format=%s)"
  echo "- 通知を書いた時刻: $(date '+%Y-%m-%d %H:%M:%S %Z')"
} > "$REPORT"

if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY_RUN: $REPORT に書いた内容 ==="
  cat "$REPORT"
  rm -f "$REPORT" "$CSV"
  exit 0
fi

# LATEST.md の「GPU 側からの報告」表の先頭に 1 行挿す
python3 - "$REPORT" "$TODAY" "$VERDICT" <<'PY'
import pathlib, sys
rep, today, verdict = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path("docs/handoff/LATEST.md")
s = p.read_text()
head = "| 日付 | ファイル | 要点 |\n|---|---|---|\n"
name = rep.split("/")[-1]
row = f"| {today} | [{name}]({name}) | **自動通知**: run 終了。{verdict} |\n"
if head in s and row not in s:
    p.write_text(s.replace(head, head + row, 1))
PY

git add "$REPORT" docs/handoff/LATEST.md
[ -f "$CSV" ] && git add "$CSV"
git -c user.name="Keisuke Nishioka" -c user.email="kei128608@gmail.com" \
  commit -q -m "docs: run 終了の自動通知（${TODAY}）— ${VERDICT}" || {
    echo "commit するものが無い"; exit 0; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git pull --rebase --quiet origin "$BRANCH" || echo "警告: pull --rebase が失敗。手で解消が必要"
if git push --quiet origin "$BRANCH"; then
  echo "push 済み: $BRANCH <- $(git rev-parse --short HEAD)"
else
  echo "警告: push が失敗。コミットはローカルに残っている"
fi
