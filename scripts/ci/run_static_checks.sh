#!/usr/bin/env bash
set -euo pipefail
# Run static checks and collect reports into reports/static-analysis
# Usage: run_static_checks.sh [report_dir]

REPORT_DIR=${1:-reports/static-analysis}
mkdir -p "$REPORT_DIR"
echo "Running ruff..."
ruff --format json . > "$REPORT_DIR/ruff.json" || true
echo "Running pyright..."
pyright > "$REPORT_DIR/pyright.txt" || true
echo "Running black (check)..."
black --line-length 100 --check . || true
echo "Running pytest (junit xml)..."
mkdir -p "$(dirname "$REPORT_DIR")/test-results"
pytest --junitxml="$REPORT_DIR/../test-results/junit.xml" || true
echo "Static checks complete. Reports in $REPORT_DIR"
