#!/usr/bin/env bash
set -euo pipefail
# Collect and archive reports for CI artifacts

OUTDIR=${1:-reports/artifacts}
mkdir -p "$OUTDIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
ARCHIVE="$OUTDIR/static_analysis_${TS}.tar.gz"
tar -czf "$ARCHIVE" reports/static-analysis || true
echo "Reports archived to $ARCHIVE"
