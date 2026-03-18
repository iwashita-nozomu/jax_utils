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

# スコープファイルを作成（規約: worktree-lifecycle.md に従う）
echo "3️⃣  スコープファイルを作成中..."
create_scope_file() {
    local scope_file="$1/WORKTREE_SCOPE.md"
    
    cat > "$scope_file" << EOF
# WORKTREE_SCOPE

このファイルは、worktree の目的、変更範囲、参照必須文書をまとめたものです。
詳細は [documents/worktree-lifecycle.md](#) を参照。

## Worktree Summary

- **Branch:** \`$BRANCH_NAME\`
- **Worktree path:** \`$WORKTREE_DIR\`
- **Purpose:** $DESCRIPTION
- **Owner or agent:** （作業者名）

## Editable Directories

以下のディレクトリで変更を行って問題ありません。

- \`python/jax_util/\`
- \`python/tests/\`
- \`documents/\`
- \`notes/\`

## Read-Only Or Avoid Directories

以下のディレクトリは参照のみで、原則として変更しないでください。

- \`Archive/\`
- \`references/\`
- \`.worktrees/\` （他の worktree）
- \`experiments/\` （実験結果、状況による）

## Required References Before Editing

**必ず確認してください：**

1. [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
   - worktree の作成・削除・吸い出し規約
2. [documents/coding-conventions-project.md](/workspace/documents/coding-conventions-project.md)
   - プロジェクト全体の方針
3. [documents/conventions/python/04_type_annotations.md](/workspace/documents/conventions/python/04_type_annotations.md)
   - 型アノテーション（全実装必須）
4. [documents/conventions/python/09_file_roles.md](/workspace/documents/conventions/python/09_file_roles.md)
   - 責務分離・ファイル配置
5. [documents/conventions/common/05_docs.md](/workspace/documents/conventions/common/05_docs.md)
   - ドキュメント形式（Markdown 等）

## Main Carry-Over Targets

このワークツリーを削除する前に、以下を \`main\` へ持ち帰ってください：

- \`notes/\` に関する判断・結果は \`main\` へ commit または merge
- 重要な観測・考察は \`notes/worktrees/\` に記録
- 参照必須の文書は worktree 選定時に整理

## Required Checks Before Commit

確認・テスト実行：

- [ ] \`pyright python/\` （型チェック全体）
- [ ] \`pytest python/tests/\` （ユニットテスト）
- [ ] \`markdownlint\` （Markdown 形式）
- [ ] 規約に従った記述か（.scope.md 削除前に確認）

## Additional Rules

### 規約確認スクリプト

このワークツリー内で利用可能：

\`\`\`bash
# 規約表示
bash ../scripts/view_conventions.sh

# ガイド表示
bash ../scripts/guide.sh

# クイックスタート
less ../QUICK_START.md
\`\`\`

### コミット規約

コミットメッセージは以下の形式に従ってください：

\`\`\`
<category>: <説明>

<詳細（必要に応じて）>
\`\`\`

**category の種類:**
- \`feat:\` 新機能追加
- \`fix:\` バグ修正
- \`docs:\` ドキュメント更新
- \`refactor:\` コード改善（機能変更なし）
- \`test:\` テスト追加・修正
- \`chore:\` 設定変更

### 削除時のチェックリスト

ワークツリーを削除する前に、このファイルの "Carry-Over Targets" を確認し、
[documents/worktree-lifecycle.md#7-worktree-を閉じる前のチェック](worktree-lifecycle.md) 
に記載のチェックリストを完了してください。

EOF
    echo "   ✅ スコープファイル: $scope_file"
}

create_scope_file "$WORKTREE_DIR"
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
