# ツール統一提案: setup_worktree.sh vs create_worktree.sh

**作成日:** 2026-03-19\
**ステータス:** 提案（実装未了）\
**優先度:** 高

______________________________________________________________________

## 現状分析

### 問題: ツール重複

現在、ワークツリー作成機能が2つのスクリプトに分散しています：

| スクリプト           | パス                               | 問題                         |
| -------------------- | ---------------------------------- | ---------------------------- |
| `setup_worktree.sh`  | `scripts/setup_worktree.sh`        | UI向け・ガイド豊富だが、冗長 |
| `create_worktree.sh` | `scripts/tools/create_worktree.sh` | シンプル・CI向け・より柔軟   |

### 機能比較

| 機能                     | setup_worktree.sh | create_worktree.sh |
| ------------------------ | ----------------- | ------------------ |
| ブランチ作成             | ✅                | ✅                 |
| ワークツリー作成         | ✅                | ✅                 |
| スコープテンプレート配置 | ✅                | ✅                 |
| ワークツリーカスタムパス | ❌                | ✅                 |
| CLI 引数の柔軟性         | 限定的            | 高い               |
| ドキュメント・ガイド     | ✅（丁寧）        | ❌（簡潔）         |
| エラーハンドリング       | ✅                | ✅                 |
| 日付自動付与             | ✅                | ❌（明示的）       |

______________________________________________________________________

## 提案1: `create_worktree.sh` を標準化（推奨）

### 方針

1. `scripts/tools/create_worktree.sh` を実装の基準とする
1. `scripts/setup_worktree.sh` をシンボリックリンク化 OR 削除
1. ドキュメント・ガイドの充実で使い方を補補強

### 実装手順

#### 1-1. create_worktree.sh にドキュメント追加

```bash
# scripts/tools/create_worktree.sh の先頭に以下を追加

# ═══════════════════════════════════════════════════════════════════════════
# ワークツリー・ブランチ作成スクリプト
# 
# 用途: 新規開発用ブランチ・ワークツリーを規約に従って作成
#
# 使用方法:
#   1. 標準的な使い方（日付自動付与）
#      bash scripts/tools/create_worktree.sh my-feature
#      → ブランチ: work/my-feature-<YYYYMMDD>
#      → ワークツリー: .worktrees/my-feature
#
#   2. カスタムワークツリーパス指定
#      bash scripts/tools/create_worktree.sh feature-xyz .worktrees/custom-path
#
# 前提条件:
#   - リポジトリが clean （git status: 何も表示されない）
#   - main ブランチ存在
#   - origin と接続可能
#
# 副作用:
#   - ブランチが新規作成される
#   - .worktrees/ ディレクトリが自動作成される
#   - WORKTREE_SCOPE.md テンプレートが配置される
#   - 初回 commit / push は行わない（ユーザーが実施）
#
# エラー時:
#   - ブランチ既存でも既定動作で進行（既存の場合は警告）
#   - ワークツリー既存の場合はエラーで停止
#
# ═══════════════════════════════════════════════════════════════════════════
```

#### 1-2. setup_worktree.sh をシンボリックリンク化

```bash
# メインワークツリーで実行
cd /workspace

# 古いファイルをバックアップ
mv scripts/setup_worktree.sh scripts/setup_worktree.sh.bak

# シンボリックリンク作成
ln -s tools/create_worktree.sh scripts/setup_worktree.sh

# 確認
ls -la scripts/setup_worktree.sh
cat scripts/setup_worktree.sh  # シンボリックリンク先を表示
```

#### 1-3. ドキュメント更新

- [documents/FILE_CHECKLIST_OPERATIONS.md](./FILE_CHECKLIST_OPERATIONS.md)：

  > チェックリスト1 で `scripts/tools/create_worktree.sh` を推奨に変更

- [README.md](../README.md):

  > 使用例で `scripts/tools/create_worktree.sh` を優先記載

#### 1-4. Git コミット

```bash
cd /workspace
git add -A
git commit -m "refactor(tools): unify worktree creation via create_worktree.sh

- Deprecate scripts/setup_worktree.sh in favor of scripts/tools/create_worktree.sh
- Point setup_worktree.sh to create_worktree.sh via symlink for compatibility
- Enhance create_worktree.sh with documentation and error handling
- Update CHECKLIST_OPERATIONS to recommend create_worktree.sh
- Both tools now support WORKTREE_SCOPE.md auto-deployment

BREAKING CHANGE: None (backward compatible via symlink)
"
git push origin main
```

### メリット

- デュプリケーション排除
- ディレクトリ構造が論理的（`tools/` 配下に集約）
- CI・自動化との親和性向上
- 拡張性向上（カスタムパス対応）

### デメリット

- シンボリックリンクが git に保存される場合、Windows 互換性に注意
- 既存ユーザーが `setup_worktree.sh` に慣れている場合は周知必要

______________________________________________________________________

## 提案2: 別案: setup_worktree.sh を標準化

（参考までに記載）

### 方針

1. `scripts/setup_worktree.sh` を実装の基準とする
1. `scripts/tools/create_worktree.sh` を廃止
1. `setup_worktree.sh` を CI/フロー用の高度なオプション拡張

### 問題点

- ディレクトリ構造が散乱したまま
- `scripts/` が責務過多（直下に32個のツール・スクリプトを置くことになる）
- CI・自動化との相性が低下

**推奨度:** 低（提案1を強く推奨）

______________________________________________________________________

## 実装チェックリスト

### 前提確認

- [ ] `scripts/tools/create_worktree.sh` が最新版であることを確認
- [ ] `scripts/setup_worktree.sh` との差分確認

### 変更実施

- [ ] `scripts/tools/create_worktree.sh` にドキュメント・コメント追加
- [ ] `scripts/setup_worktree.sh` をシンボリックリンク化（またはラッパー化）
- [ ] `git add scripts/` で確認

### テスト実施

- [ ] 新ワークツリー作成テスト（同ブランチ2回実行で既存判定テスト）
- [ ] スコープテンプレート配置確認
- [ ] シンボリックリンク経由アクセステスト（`bash scripts/setup_worktree.sh test-branch`）

### ドキュメント更新

- [ ] [FILE_CHECKLIST_OPERATIONS.md](./FILE_CHECKLIST_OPERATIONS.md) 更新
- [ ] [TOOLS_DIRECTORY.md](./tools/TOOLS_DIRECTORY.md) 更新（統一決定後）
- [ ] [README.md](../README.md) 更新（使用例記載）
- [ ] [guide.sh](../scripts/guide.sh) 出力更新（推奨ツール表示）

### アナウンス

- [ ] チーム内通知
- [ ] migration guide 作成（オプション）

______________________________________________________________________

## 想定所要時間

- 実装: 5分
- テスト: 10分
- ドキュメント: 15分
- **合計: 30分**

______________________________________________________________________

## 決定フロー

### ステップ1: 方針決定

- [ ] 提案1（`create_worktree.sh` 統一）を採用
- [ ] 提案2（`setup_worktree.sh` 統一）を採用
- [ ] その他の案

### ステップ2: 実装

- 決定後、上記実装チェックリストに従って進行

### ステップ3: 検証

- テスト実施
- ドキュメント確認

______________________________________________________________________

## 参考資料

- [tools/TOOLS_DIRECTORY.md](./tools/TOOLS_DIRECTORY.md) — ツール目録
- [FILE_CHECKLIST_OPERATIONS.md](./FILE_CHECKLIST_OPERATIONS.md) — 作業別チェックリスト
