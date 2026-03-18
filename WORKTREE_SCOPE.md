# Smolyak Improvement Worktree Scope

## Worktree Summary

- Branch: `work/smolyak-improvement-20260318`
- Worktree path: `/workspace/.worktrees/work-smolyak-improvement-20260318`
- Purpose: corrected `main` を起点に Smolyak 実装を改造する
- Owner or agent: Codex

## Editable Directories

- `python/jax_util/functional/`
- `python/tests/functional/`
- `experiments/functional/smolyak_scaling/`
- `experiments/functional/smolyak_hlo/`
- `notes/experiments/`
- `notes/worktrees/`

## Read-Only Or Avoid Directories

- `python/jax_util/neuralnetwork/`
- `python/jax_util/experiment_runner/`
- `reviews/`
- `documents/`
- `experiments/functional/smolyak_scaling/results/`
- `experiments/functional/smolyak_hlo/results/`

## Required References Before Editing

- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
- [documents/coding-conventions-project.md](/workspace/documents/coding-conventions-project.md)
- [documents/conventions/python/07_type_checker.md](/workspace/documents/conventions/python/07_type_checker.md)
- [worktree_results_smolyak_scaling_tuned_2026-03-18.md](/workspace/notes/worktrees/worktree_results_smolyak_scaling_tuned_2026-03-18.md)
- [worktree_work_smolyak_improvement_2026-03-18.md](/workspace/notes/worktrees/worktree_work_smolyak_improvement_2026-03-18.md)

## Main Carry-Over Targets

- `notes/experiments/`
- `notes/worktrees/`

## Required Checks Before Commit

- `pyright python/jax_util/functional python/tests/functional/test_smolyak.py python/tests/functional/test_integrate.py`
- `JAX_PLATFORMS=cpu PYTHONPATH=python python3 -m pytest python/tests/functional/test_smolyak.py python/tests/functional/test_integrate.py -q`
- `markdownlint` if Markdown changes

## Additional Rules

- この worktree の出発点は、`results/functional-smolyak-scaling-tuned` branch の Smolyak module を `main` に戻した状態である
- merge conflict で混ぜた helper API を前提にしない
- 新しい helper や公開 API を増やす場合は、この worktree で意図を明示して追加する
- raw JSON / JSONL / log / SVG / HTML は既存ファイルを上書きせず、必要な新規結果だけを意図的に追加する
- `main` に残す判断、観測、次の方針は `notes/` に置いてから worktree を閉じる
