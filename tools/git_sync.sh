#!/bin/bash
# ============================================================
# 無人ジョブ（PBS）から repo を最新に揃えるための共通関数。source して使う。
#
# なぜ `git pull --rebase` を直に呼ばないか:
#   2026-10-07 の磨き直しジョブ 3129 が `fatal: Cannot rebase onto multiple branches.` で
#   pull に失敗し、古い版のコードで 8 時間走って walltime で死んだ。`git pull` は FETCH_HEAD に
#   複数の候補が入ると（detached HEAD で枝名が `HEAD` になる・refspec が増える等）この形で落ちる。
#   fetch と rebase を分けて明示の `origin/<branch>` に当てれば、この失敗は起きない。
#
# 失敗しても呼び出し側は続行してよい（戻り値 1）。ただし失敗は必ず標準出力に出す。
# ============================================================

# 現在の枝名を返す。detached HEAD なら空を返す（その場合は同期しない）。
git_current_branch() {
  local b
  b="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)" || return 1
  [ -n "$b" ] || return 1
  printf '%s\n' "$b"
}

# origin/<branch> に rebase して repo を最新にする。
git_sync_latest() {
  local branch err
  branch="$(git_current_branch)" || {
    echo "警告: detached HEAD なので pull しない（手元の版で続ける）"; return 1; }

  # 前のジョブが途中で死んで rebase が残っていると、以降の git が全部失敗する
  if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
    echo "警告: 中断された rebase が残っていたので abort する"
    git rebase --abort 2>&1 | sed 's/^/  /'
  fi

  if ! err="$(git fetch --quiet origin "$branch" 2>&1)"; then
    echo "警告: fetch に失敗。手元の版で続ける"; echo "$err" | sed 's/^/  /'; return 1
  fi
  # autoStash: このリポジトリの作業ツリーは普段から汚れている（nife などの手元の変更、
  # _runs/ の出力）。素の rebase は "cannot rebase: You have unstaged changes" で必ず落ちるので、
  # 退避してから rebase し、あとで戻す。
  if ! err="$(git -c rebase.autoStash=true rebase --quiet "origin/$branch" 2>&1)"; then
    echo "警告: rebase に失敗。手元の版で続ける"; echo "$err" | sed 's/^/  /'
    git rebase --abort 2>/dev/null
    return 1
  fi
  echo "同期済み: $branch <- $(git rev-parse --short HEAD)"
  return 0
}

# commit 済みの内容を origin に送る（送る前に rebase で揃える）。
git_push_current() {
  local branch
  branch="$(git_current_branch)" || { echo "警告: detached HEAD なので push しない"; return 1; }
  git_sync_latest >/dev/null || echo "警告: push 前の同期に失敗。そのまま push を試す"
  if git push --quiet origin "$branch"; then
    echo "push 済み: $branch <- $(git rev-parse --short HEAD)"
    return 0
  fi
  echo "警告: push が失敗。コミットはローカルに残っている"
  return 1
}
