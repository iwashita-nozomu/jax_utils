# Smolyak Validation Branch

- Branch: `results/smolyak-validation-20260328`
- Status: `active`
- Worktree: `/workspace/.worktrees/results-smolyak-validation-20260328`
- Purpose: `main` の `12dee48` を基準に、standalone 化後の `experiment_runner` を使った Smolyak validation と結果保存を行う。

## Read This First

- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
- [documents/experiment_runner.md](/workspace/documents/experiment_runner.md)
- [notes/worktrees/worktree_work_smolyak_improvement_2026-03-18.md](/workspace/notes/worktrees/worktree_work_smolyak_improvement_2026-03-18.md)

## Current Scope

- 実験コード: `experiments/smolyak_experiment/`
- 結果保存: `experiments/smolyak_experiment/results/`, `experiments/functional/smolyak_scaling/results/`
- 積分器改造: `python/jax_util/functional/`

## Carry-Over Targets

- `notes/worktrees/worktree_smolyak_validation_2026-03-28.md`
- `notes/experiments/smolyak_scaling_experiment.md`
- `notes/experiments/results/`

## Notes

- raw JSONL と途中結果はこの branch 側に残す。
- `main` へは要約と最小 final JSON を持ち帰る。
