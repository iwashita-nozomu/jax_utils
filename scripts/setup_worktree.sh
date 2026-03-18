#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 作業用ブランチとワークツリーのセットアップスクリプト
# 使用方法: ./scripts/setup_worktree.sh <branch-name> [description]
# 例: ./scripts/setup_worktree.sh protocol-improvements "Protocol型アノテーション改善"
# ═══════════════════════════════════════════════════════════════════════════

set -e

if [ $# -lt 1 ]; then
    echo "❌ ブランチ名が必要です"
    echo ""
    echo "使用方法:"
    echo "  $0 <branch-name> [description]"
    echo ""
    echo "例:"
    echo "  $0 protocol-improvements 'Protocol型アノテーション改善'"
    echo "  $0 train-api-refactor 'train.pyのAPI改善'"
    exit 1
fi

BRANCH_NAME="work/$1-$(date +%Y%m%d)"
DESCRIPTION="${2:-作業用ブランチ}"
WORKTREE_DIR=".worktrees/$(echo $1 | tr '/' '-')-$(date +%Y%m%d)"

echo "════════════════════════════════════════════════════════════════════════"
echo "ワークツリーのセットアップ"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "📋 設定内容:"
echo "  ブランチ名: $BRANCH_NAME"
echo "  説明: $DESCRIPTION"
echo "  ワークツリー: $WORKTREE_DIR"
echo ""

# mainから新しいブランチを作成
echo "1️⃣  ブランチを作成中..."
cd /workspace

# origin/main の最新を取得
git fetch origin main

# 既存ブランチの確認
if git branch | grep -q "$BRANCH_NAME"; then
    echo "   ⚠️  ブランチ既存: $BRANCH_NAME"
else
    # ブランチを作成（メインワークツリーはチェックアウトしない）
    git branch "$BRANCH_NAME" origin/main
    echo "   ✅ ブランチ作成: $BRANCH_NAME"
fi

echo ""

# ワークツリーを登録
echo "2️⃣  ワークツリーを登録中..."

# メインワークツリーが main ブランチにいることを確認
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "   ℹ️  メインワークツリーを main に戻しています..."
    git checkout main
fi

mkdir -p ".worktrees"
git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"

echo "   ✅ ワークツリー作成: $WORKTREE_DIR"
echo ""

# スコープファイルを作成（規約: worktree-lifecycle.md テンプレートをコピー）
echo "3️⃣  スコープファイルを作成中..."
cp documents/WORKTREE_SCOPE_TEMPLATE.md "$WORKTREE_DIR/WORKTREE_SCOPE.md"
echo "   ✅ スコープファイル: $WORKTREE_DIR/WORKTREE_SCOPE.md"
echo ""

# Git設定を表示
echo "4️⃣  登録確認:"
git branch -v | grep "$BRANCH_NAME"
echo ""
git worktree list
echo ""

echo "════════════════════════════════════════════════════════════════════════"
echo "✅ セットアップが完了しました"
echo ""
echo "📌 次のステップ:"
echo ""
echo "  1. ワークツリーに移動:"
echo "     cd $WORKTREE_DIR"
echo ""
echo "  2. スコープファイルを確認・編集（重要）:"
echo "     less WORKTREE_SCOPE.md"
echo ""
echo "  3. 規約を確認:"
echo "     bash ../scripts/view_conventions.sh"
echo ""
echo "  4. チェックリストを実行:"
echo "     pyright python/"
echo "     pytest python/tests/"
echo ""
echo "  5. 作業実施"
echo ""
echo "  6. コミット・プッシュ:"
echo "     git add -A"
echo "     git commit -m 'category: 説明'"
echo "     git push origin $BRANCH_NAME"
echo ""
echo "  7. ワークツリーを削除（完了後、carry-over の確認後）:"
echo "     git worktree remove $WORKTREE_DIR"
echo "     git branch -d $BRANCH_NAME"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
