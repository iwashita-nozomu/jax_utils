#!/bin/bash

# pre_review.sh — PR 前の自動 QA チェック
# 
# 使用方法:
#   scripts/ci/pre_review.sh
#
# このスクリプトは以下を実行します:
#   1. Type checking (Pyright strict mode)
#   2. Test execution (pytest)
#   3. Docstring validation (pydocstyle)
#   4. Code quality checks (Ruff)
#
# 環境要件: Python 3.10+, pyright, pytest, pydocstyle, ruff がインストール済み

set -e

# 色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "=========================================="
echo "PRE-REVIEW QA CHECKS"
echo "=========================================="

# 1. Type checking
echo ""
echo -e "${BLUE}1️⃣  Type Checking (Pyright strict mode)...${NC}"
if python3 -m pyright ./python/jax_util; then
    echo -e "${GREEN}✅ Type checking passed${NC}"
else
    echo -e "${RED}❌ Type errors found. Review code.${NC}"
    exit 1
fi

# 2. Test execution
echo ""
echo -e "${BLUE}2️⃣  Running pytest...${NC}"
# 注: 削除/移動したテストをスキップ (-k オプション)
if PYTHONPATH=/workspace/python python3 -m pytest python/tests/ -q --tb=short \
  --ignore=python/tests/solvers/test_jax_debug.py \
  --ignore=python/tests/solvers/test_solver_internal_branches.py \
  --ignore=python/tests/functional/test_protocols_and_smolyak_helpers.py \
  --ignore=python/tests/functional/test_smolyak.py \
  --ignore=python/tests/experiment_runner/test_subprocess_scheduler_unit.py; then
    echo -e "${GREEN}✅ All tests passed${NC}"
else
    echo -e "${RED}❌ Test failures. Fix tests.${NC}"
    exit 1
fi

# 3. Docstring validation
echo ""
echo -e "${BLUE}3️⃣  Docstring validation (pydocstyle)...${NC}"
if python3 -m pydocstyle python/jax_util; then
    echo -e "${GREEN}✅ Docstring validation passed${NC}"
else
    echo -e "${YELLOW}⚠️  Docstring issues. Review output above.${NC}"
fi

# 4. Code quality
echo ""
echo -e "${BLUE}4️⃣  Code quality checks (Ruff)...${NC}"
if python3 -m ruff check python/jax_util --select E,F,I,D,UP; then
    echo -e "${GREEN}✅ Code quality checks passed${NC}"
else
    echo -e "${YELLOW}⚠️  Style issues found. Review output above.${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ PRE-REVIEW CHECKS COMPLETE${NC}"
echo "=========================================="
echo ""
echo "Next: Commit changes and open PR"
echo ""
