# Worktree Health

## Purpose

現在の worktree が、scope、branch、未コミット差分、conflict risk の観点で健全かを確認します。

## Use This For

- `WORKTREE_SCOPE.md` の存在と内容確認
- editable directories / runtime output directories の逸脱確認
- worktree の clean / dirty 状態確認
- conflict risk や carry-over 漏れの確認
- 削除前の健全性チェック

## Must Read Before Reviewing

- `documents/worktree-lifecycle.md`
- `documents/WORKTREE_SCOPE_TEMPLATE.md`
- 対象 worktree の `WORKTREE_SCOPE.md`

## Inputs

- 対象 worktree
- `git status`
- `git worktree list`
- 必要なら branch / merge 状態

## Outputs

- health findings
- scope drift、missing scope、runtime output drift、cleanup risk の指摘
- `safe to continue`, `needs scope fix`, `needs cleanup`, `delete-ok` の判定

## Mandatory Checklist

- `WORKTREE_SCOPE.md` がある
- editable directories の外に repo edit が出ていない
- runtime output が許可ディレクトリに収まっている
- worktree の目的と branch 名が一致している
- carry-over すべき note や result が取り残されていない
- conflict risk や stale branch の兆候がない

## Default Commands

- `git worktree list`
- `git status --short`
- `python3 scripts/agent_tools/validate_role_write_scope.py ...`
- `bash scripts/tools/check_worktree_scopes.sh`

## Boundary

- repo 全体レビューは `agents/skills/project-review.md` を使います。
- 文書内容の review は docs 系 skill を使います。

## Implementation Surface

- `.github/skills/24-worktree-health/`
