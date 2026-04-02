# レビュー手順とポリシー

目的: コード/ドキュメント変更の品質を担保し、安全に main へ反映するための標準手順を定める。

## 適用範囲

- main ブランチへの直接のコード変更を禁止します。ドキュメント変更だけを例外として許可します。

## 役割

- レビュワー (Reviewer): 技術的妥当性、スタイル、テストを確認。
- インテグレータ (Integrator): レビュー完了後の統合・マージ担当（自動化可、ただしポリシーを満たす場合のみ）。
- 監査者 (Auditor): 証跡・結果（reports/）の確認と保管。

## レビュー前チェックリスト（PR 作成時）

- `pytest` の該当テストを追加・更新し、ローカルで通ること。
- `ruff`、`pyright`、`black` を実行し、重大な警告がないこと。
- ドキュメント変更がある場合は `documents/` に追記します。図・ユースケースが必要な変更では、それらも含めます。
- `WORKTREE_SCOPE.md` が該当ワークツリーに存在すること（ワークツリー運用時）。

## PR フロー

2. コマンドファースト：まず再現コマンド（`scripts/ci/*`）を作成し、README に手順を記載。
1. 自動チェックを実行（`scripts/ci/run_static_checks.sh` 等）。出力は `reports/` に保管。
1. PR を作成し、テンプレートに以下を記載：目的、変更点、再現手順、チェック結果のリンク。
1. レビュワーをアサイン、CI が成功するまで対応。
1. 承認後、インテグレータがマージ。ただし以下はマージ禁止/制限:
   - main への直接コード修正は禁止（ドキュメントは可）。
   - 自動修正は `safe_file_extractor.py` により安全判定されたファイルのみを対象とする。

## ブランチスコープ

- PR の説明に `変更スコープ` を明記し、変更が複数サブモジュールにまたがる場合は分割を検討すること。

## 詳細設計の配置

- PR で設計文書を追加/更新する場合、`documents/design/<submodule>/README.md` に変更の要約を追記すること。

## マージ条件

- 少なくとも1名の承認（重要箇所は2名推奨）
- コンフリクトがないこと
- 変更ログ（CHANGELOG または PR 説明）あり

## 自動化と自動修正ポリシー

- SAFE_COUNT が 0 の場合、手動でブランチ所有者と調整し、再実行する。
- 自動化エージェントは PR を作成できるが、main へのマージは人間の承認を必要とする（ドキュメント例外あり）。

## レビュー期限・優先度

- マイナー修正: 3 営業日以内にレビュー。

## レビューテンプレート（PR に貼る）

2. 変更内容(一覧):
1. 再現手順 / コマンド:
1. テスト: 追加/更新したテスト列挙
1. 影響範囲: モジュール、API、ユーザー影響
1. 関連 Issue / 設計ドキュメント:
1. artifacts/reports へのリンク:

## エビデンス保存

- 重大なレビュー結果は `reviews/` 配下に記録する（`CONFLICT_REPORT.md` 等）。

______________________________________________________________________

## レビュー履歴管理 — 最新のみ保持

`reviews/` ディレクトリに蓄積したレビュー結果を整理する手順。定期的（月 1 回）に実施してください。

### 手順

#### **1. 現在のレビューファイルを確認**

```bash
# reviews/ 配下の全ファイルリスト
ls -lt reviews/*.md | head -20

# 古さルー確認（3ヶ月以上前のファイル）
find reviews/ -type f -name "*.md" -mtime +90 | sort
```

#### **2. git 履歴で保存確認**

`reviews/` ファイルは Git の履歴に保存されるため、削除しても参照可能：

```bash
# 特定レビュー の全バージョンを表示
git log --follow --oneline -- reviews/COMPREHENSIVE_CODE_REVIEW_20260321.md

# 過去バージョンを表示
git show HEAD~5:reviews/COMPREHENSIVE_CODE_REVIEW_20260321.md

# 削除前に確認
git log --all --full-history -- reviews/DETAILED_THOROUGH_CODE_REVIEW_20260321.md
```

#### **3. 古いレビューを削除**

```bash
# 確認済みなら削除
rm reviews/COMPREHENSIVE_CODE_REVIEW_20260321.md
rm reviews/DETAILED_THOROUGH_CODE_REVIEW_20260321.md

# git で追跡削除
git rm reviews/COMPREHENSIVE_CODE_REVIEW_20260321.md
git rm reviews/DETAILED_THOROUGH_CODE_REVIEW_20260321.md

# コミット
git commit -m "chore: archive old reviews to git history

Remove stale review reports:
- COMPREHENSIVE_CODE_REVIEW_20260321.md (3+ months old, kept in git log)
- DETAILED_THOROUGH_CODE_REVIEW_20260321.md (3+ months old, kept in git log)

Reference: git log --follow -- reviews/<filename>"

# Push
git push origin main
```

### **保持ポリシー**

| ファイル | 保持期間 | 処理 |
|---------|--------|------|
| 最新のレビュー | 常時 | 保持 |
| 参照中のレビュー | 実行中 | 保持 |
| 過去レビュー（1-3ヶ月）| 3ヶ月 | **保持** |
| 古いレビュー（3ヶ月+） | 3ヶ月以上 | **git 履歴に移動** |
| テンプレート・ガイド | 常時 | 保持 |

### **Git 履歴からの復元**

削除後に必要になった場合：

```bash
# 削除ファイルの最後のバージョンを復元
git show HEAD@{n}:reviews/FILENAME.md > /tmp/FILENAME.md

# または、git log で特定コミット を検索
git log -p -- reviews/ | grep -B5 -A5 "削除内容"

# 復元ブランチを作成
git checkout <commit> -- reviews/FILENAME.md
```

______________________________________________________________________

## 例外と逸脱の扱い
