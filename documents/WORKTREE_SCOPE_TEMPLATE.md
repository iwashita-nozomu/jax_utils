# WORKTREE_SCOPE Template

このファイルは、他環境へ渡す worktree や、変更範囲を限定して使う worktree のためのテンプレートです。
実際に使うときは、このファイルを worktree root に `WORKTREE_SCOPE.md` として置きます。

## Worktree Summary

- Branch:
- Worktree path:
- Purpose:
- Owner or agent:

## Editable Directories

- `path/to/dir`
- `another/path`

## Read-Only Or Avoid Directories

- `path/to/avoid`
- `another/path`

## Required References Before Editing

- `documents/...`
- `notes/...`
- `reviews/...`

## Additional Rules

- ここに、この worktree 固有の制約を書きます。
- 例: テストは触らない、結果 JSON は commit しない、runner だけ変更する、など。
