# Experiment Runner Refactor Branch

- Branch: `work/experiment-runner-refactor-20260330`
- Status: `archived`
- Retention: `delete-ok`
- Worktree: closed
- Purpose: `python/experiment_runner/` とその接続面の改造を、この branch で集中的に進める。

## Read This First

- [documents/experiment_runner.md](/workspace/documents/experiment_runner.md)
- [experiment_runner.md](/workspace/notes/themes/experiment_runner.md)
- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)

## Current Scope

- `python/experiment_runner/`
- `python/tests/experiment_runner/`
- `experiments/smolyak_experiment/`
- `documents/experiment_runner.md`

## Carry-Over Targets

- `notes/worktrees/worktree_experiment_runner_refactor_2026-03-30.md`
- `notes/themes/experiment_runner.md`
- `notes/knowledge/environment_setup.md`
- `notes/knowledge/experiment_operations.md`

## Notes

- raw 実験結果は必要に応じて branch 側へ残す。
- `main` へ統合するときは test と document を同時に更新する。

## Current Landing Points

- 当初の carry-over target にあった `notes/experiments/experiment_runner_usage.md` と個別 theme note は、main では 1 本の恒久 note へ整理した。
- 現在の main 側の受け皿は [experiment_runner.md](/workspace/notes/themes/experiment_runner.md) です。
- 実務ルールは [environment_setup.md](/workspace/notes/knowledge/environment_setup.md) と [experiment_operations.md](/workspace/notes/knowledge/experiment_operations.md) に吸収しました。
