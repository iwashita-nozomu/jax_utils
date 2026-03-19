#!/usr/bin/env bash
set -euo pipefail
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
