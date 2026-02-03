#!/usr/bin/env bash
set -euo pipefail

# このスクリプトは pytest のログを実行ごとのディレクトリに保存します。
# 保存場所は python/tests/logs/[YYYYMMDD]-[HHMMSS]/ です。

ROOT_DIR="/workspace"
LOG_ROOT="${ROOT_DIR}/python/tests/logs"
RUN_DIR="${LOG_ROOT}/$(date +%Y%m%d)-$(date +%H%M%S)"

mkdir -p "${RUN_DIR}"

set +e
PYTHONPATH="${ROOT_DIR}/python" /usr/bin/python3 -m pytest -q > "${RUN_DIR}/pytest.txt" 2>&1
EXIT_CODE=$?
set -e

echo "${EXIT_CODE}" > "${RUN_DIR}/exit_code.txt"

echo "logs_dir=${RUN_DIR}"
exit "${EXIT_CODE}"
