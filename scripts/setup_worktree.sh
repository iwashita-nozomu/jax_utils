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

# スコープファイルを作成
echo "3️⃣  スコープファイルを作成中..."
create_scope_file() {
    local scope_file="$1/.scope.md"
    
    cat > "$scope_file" << EOF
# 作業スコープ: $DESCRIPTION

**ブランチ:** \`$BRANCH_NAME\`  
**ワークツリー:** \`$WORKTREE_DIR\`  
**作成日:** $(date '+%Y-%m-%d %H:%M:%S')

---

## 📋 作業概要

$DESCRIPTION

---

## 🎯 目標

1. [ ] 目標1
2. [ ] 目標2
3. [ ] 目標3

---

## 📝 実装詳細

### 修正対象ファイル

- [ ] ファイル1
- [ ] ファイル2

### 確認項目

- [ ] 規約確認（\`bash ../scripts/view_conventions.sh\`）
- [ ] テスト実行（\`make test\`）
- [ ] 型チェック（\`pyright\`）
- [ ] Linting（\`pylint\`）

---

## 🔍 確認チェックリスト

### コード品質
- [ ] 型注釈が完全（Protocol/関数引数等）
- [ ] エラーメッセージが明確
- [ ] ドキュメント文字列が充実
- [ ] テストカバレッジ ≥ 80%

### 規約確認
- [ ] Python規約 (04_type_annotations.md)
- [ ] 責務分離 (09_file_roles.md)
- [ ] 命名規則 (11_naming.md)
- [ ] ドキュメント (common/05_docs.md)

### Git・コミット
- [ ] コミットメッセージが規約準拠
- [ ] 関連ファイルをすべてコミット
- [ ] 不要なログ・デバッグコード削除

---

## 📚 参考資料

- 規約一覧: \`bash ../scripts/view_conventions.sh\`
- ガイド: \`bash ../scripts/guide.sh\`
- クイックスタート: \`less ../QUICK_START.md\`

---

## 💬 メモ

（進捗・判明したことなどを記録）

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
echo "  2. スコープファイルを確認・編集:"
echo "     less .scope.md"
echo ""
echo "  3. 規約を確認:"
echo "     bash ../scripts/view_conventions.sh"
echo ""
echo "  4. 作業実施"
echo ""
echo "  5. コミット・プッシュ:"
echo "     git add -A"
echo "     git commit -m 'category: 説明'"
echo "     git push origin $BRANCH_NAME"
echo ""
echo "  6. ワークツリーを削除（完了後）:"
echo "     git worktree remove $WORKTREE_DIR"
echo "     git branch -d $BRANCH_NAME"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
