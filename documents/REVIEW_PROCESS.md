# レビュー手順とポリシー

目的: コード、文書、workflow の変更品質を担保し、安全に main へ反映するための標準手順を定めます。

## 適用範囲

- main への直接のコード変更を禁止します。文書変更だけを例外として許可します。
- repo-wide review、workflow 改造、実験系の完了判定にもこの文書を適用します。

## 役割

- Reviewer: 技術的妥当性、スタイル、テスト、証拠を確認します。
- Integrator: review 完了後の統合とマージを担当します。
- Auditor: `reviews/` と `reports/` の証跡を確認します。

## Repo-Wide Review の入口

- repo 全体レビュー、棚卸し、workflow 改造後の確認では `agents/skills/project-review.md` を最上位入口にします。
- inventory と整合確認では `comprehensive-review`、継続運用の drift 確認では `project-health` を併用します。
- scope に experiments、reports、benchmark protocol、research docs が入る場合は `agents/skills/research-perspective-review.md` を追加します。
- 文書整理を伴う場合は `docs-completeness-review`、`md-style-check`、`docs-consistency-review` を追加します。
- worktree を使う場合は `worktree-health` を追加します。
- findings は `fix now`、`follow-up`、`delete-ok` に分けます。

## 変更前に固定すること

- 変更スコープ
- 受け入れ条件
- 必要な review family
- evidence の保存先

## 実行チェック

- Python baseline の確認は `bash scripts/ci/run_all_checks.sh --quick` を使います。
- Python 差分を含む場合は、必要に応じて `pyright <path>`、`ruff check <path>`、`pytest <target>` を追加実行します。
- Markdown 差分を含む場合は、変更したファイルに対して `mdformat <paths>` と `python3 scripts/tools/check_markdown_lint.py <paths>` を実行します。
- 規約や workflow を触った場合は `python3 scripts/check_convention_consistency.py` を実行します。
- 実験結果やユーザー向け report を含む場合は `critical-review` と `report-review` を省略しません。

## Review Flow

1. 変更スコープと必要な review family を先に決めます。
1. 再現コマンドと evidence の保存先を先に決めます。
1. 変更を実装し、対象に応じたチェックを実行します。
1. findings-first で review し、必要なら追加検証、再実行、書き直しを行います。
1. evidence を `reviews/` または `reports/` に残し、Integrator が統合します。

## マージ条件

- 必要な review family が完了していること
- 対象に応じたチェック結果が保存されていること
- コンフリクトがないこと
- 変更理由と影響範囲が追えること

## エビデンス保存

- 現在有効な review 文書は `reviews/` に置きます。
- project-wide な分析や自動生成レポートは `reports/` に置きます。
- 一時メモは `notes/` に置き、正本の代わりにしません。
- 古い review や report は Git 履歴を正本とし、root や `documents/` に dated summary を残しません。

## 履歴管理

- `reviews/` には現在有効な review 文書だけを残します。
- 重複、テスト用、役目を終えた review 文書は削除し、必要な場合は Git 履歴から参照します。
- `reports/` には再利用価値のある report だけを残し、途中経過や completion summary は残しません。
