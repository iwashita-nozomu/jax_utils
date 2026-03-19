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

- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
- [documents/coding-conventions-project.md](/workspace/documents/coding-conventions-project.md)
- `documents/...`
- `notes/...`
- `reviews/...`

## Main Carry-Over Targets

- `notes/...`
- `notes/worktrees/...`

## Required Checks Before Commit

- `pyright`
- `markdownlint`
- `pytest ...`

## Additional Rules

ニューラルネットワークの実装、テスト、簡易的な実験まで行う

/python/jax_util/neuralnetwork 、/python/tests/neuralnetwork のみ編集

- ここに、この worktree 固有の制約を書きます。
- 例: テストは触らない、結果 JSON は commit しない、runner だけ変更する、など。
- 例: `pyright python/tests/experiment_runner` を必ず追加で実行する。
- 例: 変更した Markdown は `.markdownlint.json` を基準に確認する。
