# WORKTREE_SCOPE

このファイルは、`work/smolyak-improvement-20260318` の作業範囲を定義します。

## Worktree Summary

- Branch: `work/smolyak-improvement-20260318`
- Worktree path: `./.worktrees/work-smolyak-improvement-20260318`
- Purpose: Smolyak 積分器の改良（実装・検証・必要な文書更新）
- Owner or agent: GitHub Copilot / 開発担当者

## Editable Directories

- `python/jax_util/functional/`
- `python/tests/functional/`
- `documents/`（Smolyak 関連の規約・設計・運用文書のみ）
- `notes/experiments/`（実験メモ）

## Read-Only Or Avoid Directories

- `python/jax_util/solvers/archive/`
- `Archive/`
- 大容量の生成物ディレクトリ（必要最小限の結果以外）

## Required References Before Editing

- [documents/worktree-lifecycle.md](documents/worktree-lifecycle.md)
- [documents/coding-conventions-project.md](documents/coding-conventions-project.md)
- [documents/coding-conventions-experiments.md](documents/coding-conventions-experiments.md)
- [documents/coding-conventions-testing.md](documents/coding-conventions-testing.md)

## Main Carry-Over Targets

- `notes/experiments/...`
- `notes/worktrees/...`
- `documents/...`（必要な仕様更新のみ）

## Required Checks Before Commit

- `pyright`
- `pytest python/tests/functional`
- `pytest python/tests/experiment_runner`
- `markdownlint`（変更した Markdown のみ）

## Additional Rules

- 変更は Smolyak 改良に必要な最小範囲に限定します。
- 複雑な分岐を避け、シンプルなコードを優先します。
- 生成物は原則 results 側で管理し、`main` には最小限のみ持ち帰ります。
