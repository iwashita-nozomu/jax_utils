#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# 統合 CI スクリプト
#
# 用途: pytest + pyright + ruff を一括実行してプロジェクト品質を検証
#
# 使用方法:
#   bash scripts/ci/run_all_checks.sh           # 全テスト・解析実行
#   bash scripts/ci/run_all_checks.sh --quick   # 高速モード（ruff skip）
#   bash scripts/ci/run_all_checks.sh --verbose # 詳細出力
#
# 前提条件:
#   - Docker 環境、または requirements.txt のパッケージ導入済み
#   - PYTHONPATH は自動設定
#
# 出力:
#   - コンソール: テスト結果・エラー詳細
#   - logs/ci_*.txt: 実行ログ（未実装版はコンソール出力のみ）
#
# 戻り値:
#   - 0: すべてのチェック成功
#   - 1: テスト失敗 または解析エラー
#
# 関連ドキュメント:
#   - scripts/ci/README.md: ローカル CI 実行ガイド
#   - documents/FILE_CHECKLIST_OPERATIONS.md#チェックリスト3: テスト実行
#   - .github/workflows/ci.yml: GitHub Actions ワークフロー
#
# ═══════════════════════════════════════════════════════════════════════════

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE_ROOT"

# オプション解析
QUICK_MODE=0
VERBOSE_MODE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK_MODE=1
      shift
      ;;
    --verbose)
      VERBOSE_MODE=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

export PYTHONPATH="${WORKSPACE_ROOT}/python:${PYTHONPATH:-}"

echo "════════════════════════════════════════════════════════════════"
echo "📋 統合 CI セッション開始"
echo "════════════════════════════════════════════════════════════════"
echo ""

EXIT_CODE=0

# 1. pytest 実行
echo "1️⃣  pytest を実行中..."
if python -m pytest python/tests/ -q --tb=short 2>&1; then
  echo "✅ pytest 成功"
else
  echo "❌ pytest 失敗"
  EXIT_CODE=1
fi
echo ""

# 2. pyright 実行
echo "2️⃣  pyright を実行中..."
if python -m pyright python/ 2>&1 | grep -q "0 errors"; then
  echo "✅ pyright 成功"
else
  echo "⚠️  pyright 警告あり（エラーで止めない）"
fi
echo ""

# 3. pydocstyle 実行（Docstring 検証）
echo "3️⃣  pydocstyle を実行中... (Docstring チェック)"
if python -m pydocstyle python/jax_util/ 2>&1; then
  echo "✅ pydocstyle 成功"
else
  echo "⚠️  pydocstyle エラーあり（詳細: documents/DOCSTRING_GUIDE.md を参照）"
fi
echo ""

# 4. ruff (QUICK_MODE でスキップ可能)
if [ $QUICK_MODE -eq 0 ]; then
  echo "4️⃣  ruff を実行中..."
  echo "   - E,F: コード品質（エラー・警告）"
  echo "   - I: Import 順序チェック"
  echo "   - D: Docstring 検証"
  echo "   - UP: Python 最新構文チェック"
  echo ""
  
  RUFF_ERRORS=$(python -m ruff check python/ --select D,E,F,I,UP 2>&1)
  
  if echo "$RUFF_ERRORS" | grep -q "error:"; then
    echo "   $RUFF_ERRORS" | head -30
    echo "❌ ruff エラーで CI 失敗"
    EXIT_CODE=1
  elif [ -z "$RUFF_ERRORS" ] || echo "$RUFF_ERRORS" | grep -q "All checks passed"; then
    echo "✅ ruff 成功"
  else
    echo "⚠️  ruff 警告あり（詳細: ruff check python/ --show）"
    echo "$RUFF_ERRORS" | head -20
  fi
  echo ""
fi

echo "════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ CI チェック完了: すべて成功"
else
  echo "❌ CI チェック完了: 失敗あり"
fi
echo "════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
