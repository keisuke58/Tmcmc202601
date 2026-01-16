#!/usr/bin/env bash
set -euo pipefail

# Git 自動コミットスクリプト
# - 変更を検出して自動コミット
# - コミットメッセージは変更内容から自動生成
# - ログを残す

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${ROOT_DIR}/.git_auto_commit"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/auto_commit_$(date +%Y%m%d).log"
LOCK_FILE="${LOG_DIR}/.lock"

# ロックファイルで重複実行を防止
if [[ -f "${LOCK_FILE}" ]]; then
  pid=$(cat "${LOCK_FILE}" 2>/dev/null || echo "")
  if ps -p "${pid}" > /dev/null 2>&1; then
    echo "$(date -Iseconds): Another instance is running (pid=${pid}). Exiting." >> "${LOG_FILE}"
    exit 0
  fi
fi
echo $$ > "${LOCK_FILE}"
trap "rm -f ${LOCK_FILE}" EXIT INT TERM

log() {
  echo "$(date -Iseconds): $*" | tee -a "${LOG_FILE}"
}

# Git リポジトリか確認
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  log "ERROR: Not a git repository"
  exit 1
fi

# 変更があるか確認
if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
  log "No changes detected. Exiting."
  exit 0
fi

log "Changes detected. Preparing commit..."

# 変更内容を分析してコミットメッセージを生成
changed_files=$(git diff --name-only HEAD 2>/dev/null || echo "")
untracked=$(git ls-files --others --exclude-standard | head -10)
staged=$(git diff --cached --name-only 2>/dev/null || echo "")

# 変更タイプを判定
msg_parts=()
if echo "${changed_files}" | grep -qE "\.(py|sh)$"; then
  msg_parts+=("code")
fi
if echo "${changed_files}" | grep -qE "\.(tex|md)$"; then
  msg_parts+=("docs")
fi
if echo "${changed_files}" | grep -qE "\.(json|yaml|yml|toml)$"; then
  msg_parts+=("config")
fi
if echo "${changed_files}" | grep -q "\.gitignore"; then
  msg_parts+=("chore")
fi

# 変更されたファイル数
file_count=$(echo -e "${changed_files}\n${untracked}\n${staged}" | grep -v '^$' | sort -u | wc -l)

# コミットメッセージ生成
if [[ ${#msg_parts[@]} -eq 0 ]]; then
  msg_type="chore"
else
  msg_type="${msg_parts[0]}"
fi

# 主要な変更ファイル名から詳細を抽出
main_file=$(echo -e "${changed_files}\n${untracked}\n${staged}" | grep -v '^$' | head -1 | xargs basename 2>/dev/null || echo "files")
if [[ "${file_count}" -eq 1 ]]; then
  commit_msg="${msg_type}: update ${main_file}"
else
  commit_msg="${msg_type}: update ${file_count} files (${main_file} and others)"
fi

# タイムスタンプを追加
commit_msg="${commit_msg} [auto-commit $(date +%Y%m%d_%H%M%S)]"

log "Commit message: ${commit_msg}"

# すべての変更をステージング
git add -A

# コミット実行
if git commit -m "${commit_msg}"; then
  log "SUCCESS: Committed changes"
  log "Files: ${file_count} changed"
  
  # コミットハッシュを記録
  commit_hash=$(git rev-parse --short HEAD)
  log "Commit hash: ${commit_hash}"
  
  # 簡易サマリーを生成
  {
    echo "=== Auto Commit Summary ==="
    echo "Time: $(date -Iseconds)"
    echo "Commit: ${commit_hash}"
    echo "Message: ${commit_msg}"
    echo "Files changed: ${file_count}"
    echo ""
    echo "Changed files:"
    git diff --name-only HEAD~1 HEAD 2>/dev/null | head -10 | sed 's/^/  - /'
    echo ""
  } >> "${LOG_FILE}"
  
  exit 0
else
  log "ERROR: Failed to commit (maybe no changes to commit?)"
  exit 1
fi
