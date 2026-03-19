#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# ワークツリー・ブランチ作成スクリプト
# 
# 標準的なワークツリーセットアップツール。規約に従った新しい開発用ブランチと
# ワークツリーを作成し、自動的に WORKTREE_SCOPE.md テンプレートを配置します。
#
# 【使用方法】
#   1. 標準的な使い方（デフォルトパス）
#      bash scripts/tools/create_worktree.sh my-feature
#      → ブランチ: work/my-feature-<YYYYMMDD> (※注: 本スクリプトは日付1付与しない)
#      → ワークツリー: .worktrees/my-feature
#
#   2. カスタムワークツリーパス指定
#      bash scripts/tools/create_worktree.sh feature-xyz .worktrees/custom-path
#      → ワークツリー: .worktrees/custom-path
#
# 【前提条件】
#   - リポジトリが clean 状態（git status: 何も表示されない）
#   - main ブランチが存在
#   - origin と接続可能（git fetch 可能）
#
# 【副作用】
#   - ブランチが新規作成される（既存の場合は既存を使用）
#   - .worktrees/ ディレクトリが自動作成される
#   - WORKTREE_SCOPE.md テンプレートが配置される
#   - 初回コミット・push は行わない（ユーザーが WORKTREE_SCOPE.md 編集後に実施）
#
# 【エラー処理】
#   - ブランチが既存でも警告して進行（既存ブランチ上のワークツリー作成）
#   - ワークツリーパスが既存の場合は git worktree add エラーで停止
#   - WORKTREE_SCOPE_TEMPLATE.md がない場合は警告（パス指定が必要）
#
# 【関連ドキュメント】
#   - documents/FILE_CHECKLIST_OPERATIONS.md: チェックリスト1（新規ブランチ開始）
#   - documents/tools/TOOLS_DIRECTORY.md: ツール詳細目録
#   - documents/tools/WORKTREE_TOOL_UNIFICATION.md: 結合分析・推奨ツール
#
# ═══════════════════════════════════════════════════════════════════════════

# Create a git branch and worktree, and populate WORKTREE_SCOPE.md from template.
# Usage: create_worktree.sh <branch-name> [worktree-path]

BRANCH=${1:-}
if [ -z "$BRANCH" ]; then
  echo "Usage: $0 <branch-name> [worktree-path]" >&2
  exit 2
fi
WT_PATH=${2:-.worktrees/${BRANCH}}

echo "Using branch: $BRANCH"
echo "Worktree path: $WT_PATH"

# ensure origin/main is available
git fetch origin --prune

# if branch does not exist locally, create from origin/main
if ! git rev-parse --verify refs/heads/$BRANCH >/dev/null 2>&1; then
  if git ls-remote --exit-code --heads origin $BRANCH >/dev/null 2>&1; then
    echo "Branch exists on origin, creating local branch tracking origin/$BRANCH"
    git branch --track $BRANCH origin/$BRANCH || true
  else
    echo "Creating new branch $BRANCH from origin/main"
    git branch $BRANCH origin/main
  fi
fi

# create worktree
mkdir -p "$WT_PATH"
git worktree add "$WT_PATH" refs/heads/$BRANCH

# copy WORKTREE_SCOPE_TEMPLATE.md if present
TEMPLATE=documents/WORKTREE_SCOPE_TEMPLATE.md
if [ -f "$TEMPLATE" ]; then
  cp "$TEMPLATE" "$WT_PATH/WORKTREE_SCOPE.md"
  echo "Copied WORKTREE_SCOPE_TEMPLATE.md to $WT_PATH/WORKTREE_SCOPE.md"
else
  echo "Template not found at $TEMPLATE; create WORKTREE_SCOPE.md manually in $WT_PATH"
fi

echo "Worktree ready at $WT_PATH"
