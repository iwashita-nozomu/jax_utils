#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# プロジェクト規約・ワークツリー統合ガイド
# ═══════════════════════════════════════════════════════════════════════════

set -e

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_ROOT"

clear

cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║          JAX Utils - 作業用ワークツリーセットアップガイド                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

📋 目的: 規約に従った新しい作業ブランチとワークツリーを作成します

════════════════════════════════════════════════════════════════════════════

【ステップ1】 規約の確認

  ✓ 実行コマンド:
    bash scripts/view_conventions.sh

  ✓ 主要規約:
    - 型アノテーション (04_type_annotations.md)
    - 責務分離 (09_file_roles.md)
    - ニューラルネット (18_neuralnetwork.md)
    - ドキュメント (common/05_docs.md)

════════════════════════════════════════════════════════════════════════════

【ステップ2】 ワークツリーの作成

  ✓ 実行コマンド:
    bash scripts/setup_worktree.sh <branch-name> [description]

  ✓ 例:
    bash scripts/setup_worktree.sh protocol-improvements \
      "Protocol型アノテーション改善"
    
    bash scripts/setup_worktree.sh train-api-refactor \
      "train.pyのAPI改善"

  ✓ 自動作成される内容:
    • ブランチ: work/<branch-name>-<YYYYMMDD>
    • ワークツリー: .worktrees/<branch-name>-<YYYYMMDD>

════════════════════════════════════════════════════════════════════════════

【ステップ3】 ワークツリーで作業開始

  cd .worktrees/<branch-name>-<YYYYMMDD>
  
  # 規約確認（オプション）
  bash ../scripts/view_conventions.sh

════════════════════════════════════════════════════════════════════════════

【ステップ4】 作業完了後のプッシュ

  git add -A
  git commit -m "category: 説明"
  git push origin work/<branch-name>-<YYYYMMDD>

  📌 コミットメッセージ例:
    • feat: 新機能追加
    • fix: バグ修正
    • docs: ドキュメント更新
    • refactor: コード改善
    • test: テスト追加

════════════════════════════════════════════════════════════════════════════

【ステップ5】 ワークツリーのクリーンアップ

  # ワークツリーを削除
  git worktree remove .worktrees/<branch-name>-<YYYYMMDD>

  # ローカルブランチを削除（マージ後）
  git branch -d work/<branch-name>-<YYYYMMDD>

════════════════════════════════════════════════════════════════════════════

🔍 ワークツリー・ブランチの確認

  # 現在のワークツリー一覧
  git worktree list

  # ブランチ一覧
  git branch -v

  # リモートブランチ一覧
  git branch -r

════════════════════════════════════════════════════════════════════════════

📚 规约キーワードマップ

  型安全性:
    ← 04_type_annotations.md
    ← 07_type_checker.md

  コード構成:
    ← 09_file_roles.md
    ← 10_dependencies.md

  命名规則:
    ← 11_naming.md (Python)
    ← 02_naming.md (共通)

  テスト・実装:
    ← documents/coding-conventions-project.md

  ドキュメント:
    ← 05_docs.md (共通)

════════════════════════════════════════════════════════════════════════════

✨ 完全なワークフロー例

  # 1. 規約確認
  bash scripts/view_conventions.sh

  # 2. ワークツリー作成
  bash scripts/setup_worktree.sh feature-xyz "新機能XYZ実装"

  # 3. ワークツリーに移動
  cd .worktrees/feature-xyz-20260318

  # 4. 作業実施
  # ... ファイル編集 ...

  # 5. テスト・確認
  make test

  # 6. コミット
  git add -A
  git commit -m "feat: 新機能XYZ実装完了"

  # 7. プッシュ
  git push origin work/feature-xyz-20260318

  # 8. クリーンアップ（マージ後）
  git worktree remove .worktrees/feature-xyz-20260318
  git branch -d work/feature-xyz-20260318

════════════════════════════════════════════════════════════════════════════

EOF

# ブランチとワークツリーの状況を表示
echo "【現在の状況】"
echo ""
echo "ブランチ:"
git branch -v | head -5
echo ""
echo "ワークツリー:"
git worktree list | head -5
echo ""
