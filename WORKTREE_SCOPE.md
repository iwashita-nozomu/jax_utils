# WORKTREE_SCOPE

## Worktree Summary

- Branch: `work/smolyak-integrator-opt-20260328`
- Worktree path: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328`
- Purpose: `SmolyakIntegrator` の最適化と、`experiments/smolyak_experiment/README.md` を含む実験入口の整理を進める。
- Owner or agent: Codex / user pair

## Editable Directories

- `WORKTREE_SCOPE.md`
- `python/jax_util/functional/`
- `experiments/smolyak_experiment/`
- `notes/worktrees/`
- `notes/branches/`

## Read-Only Or Avoid Directories

- `experiments/smolyak_experiment/results/`
- `experiments/functional/smolyak_scaling/results/`
- `python/tests/functional/`
- `documents/`
- `agents/`
- `scripts/`

## Required References Before Editing

- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
- [documents/BRANCH_SCOPE.md](/workspace/documents/BRANCH_SCOPE.md)
- [notes/branches/work_smolyak_integrator_opt_20260328.md](/workspace/notes/branches/work_smolyak_integrator_opt_20260328.md)
- [notes/worktrees/worktree_smolyak_integrator_opt_2026-03-28.md](/workspace/notes/worktrees/worktree_smolyak_integrator_opt_2026-03-28.md)
- [notes/experiments/smolyak_scaling_experiment.md](/workspace/notes/experiments/smolyak_scaling_experiment.md)

## Main Carry-Over Targets

- `notes/worktrees/worktree_smolyak_integrator_opt_2026-03-28.md`
- `notes/experiments/smolyak_scaling_experiment.md`
- `notes/branches/work_smolyak_integrator_opt_20260328.md`

## Working Notes During Execution

- Action log path: `notes/worktrees/worktree_smolyak_integrator_opt_2026-03-28.md`
- Experiment memo path: `notes/experiments/smolyak_scaling_experiment.md`
- Branch summary path: `notes/branches/work_smolyak_integrator_opt_20260328.md`
- worktree 内でも、最終配置と同じ相対パスで下書きする

## Required Checks Before Commit

- `python3 -m pytest python/tests/functional/test_smolyak.py`
- `python3 experiments/smolyak_experiment/run_smolyak_experiment_simple.py --size smoke --max-cases 1 --max-workers 1`
- `git diff --check`

## Additional Rules

- raw JSONL や rendered report はこの branch に常設せず、`results/*` worktree で保持する。
- `python/tests/functional/test_smolyak.py` の大きな責務変更は `work/smolyak-integrator-lead-20260328` と整合を取ってから carry-over する。
- scope 更新、編集開始、テスト実行、実験開始/停止、carry-over 判断は action log に逐次追記する。
