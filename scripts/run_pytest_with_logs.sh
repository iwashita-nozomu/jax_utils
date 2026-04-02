#!/usr/bin/env bash
set -euo pipefail

# このスクリプトは pytest のログを実行ごとのディレクトリに保存します。
# 保存場所は python/tests/logs/[YYYYMMDD]-[HHMMSS]/ です。

ROOT_DIR="/workspace"
LOG_ROOT="${ROOT_DIR}/python/tests/logs"
RUN_DIR="${LOG_ROOT}/[$(date +%Y%m%d)]-[$(date +%H%M%S)]"
RAW_LOG="${RUN_DIR}/pytest.raw.txt"
JSON_LOG="${RUN_DIR}/pytest.jsonl"

export PYTHONPATH="${ROOT_DIR}/python:${PYTHONPATH:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-}"

mkdir -p "${RUN_DIR}"

set +e
set -o pipefail
/usr/bin/python3 -m pytest -q -s 2>&1 | tee "${RAW_LOG}"
EXIT_CODE=${PIPESTATUS[0]}
set +o pipefail
set -e

RAW_LOG_PATH="${RAW_LOG}" JSON_LOG_PATH="${JSON_LOG}" /usr/bin/python3 - << 'PY'
import json
import os
from pathlib import Path

raw_log = Path(os.environ["RAW_LOG_PATH"])
json_log = Path(os.environ["JSON_LOG_PATH"])

with raw_log.open() as f_in, json_log.open("w") as f_out:
	for line in f_in:
		stripped = line.strip()
		if not stripped:
			continue
		try:
			obj = json.loads(stripped)
		except Exception:
			continue
		if isinstance(obj, dict):
			f_out.write(json.dumps(obj) + "\n")
PY

echo "${EXIT_CODE}" > "${RUN_DIR}/exit_code.txt"

echo "logs_dir=${RUN_DIR}"
exit "${EXIT_CODE}"
