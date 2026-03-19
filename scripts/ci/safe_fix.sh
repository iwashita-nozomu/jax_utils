#!/usr/bin/env bash
set -euo pipefail
# Safe auto-fix driver
# Usage: safe_fix.sh [--commit]

COMMIT=false
if [ "${1:-}" = "--commit" ]; then
  COMMIT=true
fi

python3 scripts/ci/safe_file_extractor.py
SAFE_LIST=/tmp/safe_files.txt
if [ ! -s "$SAFE_LIST" ]; then
  echo "No safe files to fix. Exiting."
  exit 0
fi

echo "Running ruff --fix on safe files..."
while read -r f; do
  echo "Fixing: $f"
  ruff --fix "$f" || true
done < "$SAFE_LIST"

echo "Running black on safe files..."
xargs -a "$SAFE_LIST" -r black --line-length 100 || true

echo "Collecting ruff report for fixed files..."
sed 's@^/workspace/@@' "$SAFE_LIST" | xargs -r ruff --format json > reports/static-analysis/ruff_after_fix_run_safe.json || true

if [ "$COMMIT" = true ]; then
  echo "Committing changes..."
  git add $(sed 's@^/workspace/@@' "$SAFE_LIST") reports/static-analysis/ruff_after_fix_run_safe.json || true
  if git diff --cached --quiet; then
    echo "No changes to commit"
  else
    git commit -m "fix(ruff): apply safe import/format fixes"
    git push -u origin HEAD
  fi
fi

echo "Safe fix complete. Reports in reports/static-analysis/ruff_after_fix_run_safe.json"
