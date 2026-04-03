# Main Branch Summary

## Branch

- `main`

## Role

- 再生成可能な code、文書、最小限の実験雛形を集約する既定 branch
- 長時間実験の生成物は直接は置かず、results branch と note への入口を整える branch

## What To Read First

- 全体方針:
  - [coding-conventions-project.md](/workspace/documents/coding-conventions-project.md)
  - [coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md)
- 実験 note の入口:
  - [notes/README.md](/workspace/notes/README.md)
  - [notes/branches/README.md](/workspace/notes/branches/README.md)

## Current Position

- Smolyak 実験コードそのものは `main` に入っている
- 旧版結果の要約は [legacy_smolyak_results_20260316.md](/workspace/notes/experiments/legacy_smolyak_results_20260316.md) にある
- tuned 実験と runner modularization の判断は worktree note と branch note から辿れる

## Interpretation

- `main` は結果ファイル本体の保管場所ではなく、再現のための code と、判断履歴への入口を保つ場所として扱う
- そのため branch note を厚めに残し、results branch へ飛ばないと分からない状態を減らします

## Branch Inventory And Knowledge Landing

### Inventory

| Branch | State | Knowledge Landing | Retention |
| --- | --- | --- | --- |
| `work/experiment-runner-module-20260316` | merged | [experiment_runner.md](/workspace/notes/themes/experiment_runner.md) | `delete-ok` |
| `work/experiment-runner-generalization-20260317` | merged | [experiment_runner.md](/workspace/notes/themes/experiment_runner.md) | `delete-ok` |
| `work/experiment-runner-refactor-20260330` | stale branch, core findings merged | [experiment_runner.md](/workspace/notes/themes/experiment_runner.md) | `delete-ok` |
| `results/smolyak-experiment-20260321` | archived / stale | [experiment_directory_planning.md](/workspace/notes/knowledge/experiment_directory_planning.md) | `delete-ok` |
| `work/smolyak-improvement-20260318` | archived / stale | [work_smolyak_improvement_20260318.md](/workspace/notes/branches/work_smolyak_improvement_20260318.md) | `delete-ok` |
| `results/functional-smolyak-scaling` | archived raw result branch | [legacy_smolyak_results_20260316.md](/workspace/notes/experiments/legacy_smolyak_results_20260316.md) | `persistent` |
| `results/functional-smolyak-scaling-tuned` | fully absorbed | [smolyak_integrator.md](/workspace/notes/themes/smolyak_integrator.md) | `delete-ok` |
| `work/smolyak-tuning-20260316` | merged knowledge retained | [smolyak_integrator.md](/workspace/notes/themes/smolyak_integrator.md) | `delete-ok` |
| `work/jaxutil-test-expansion-20260317` | merged | branch note のみで十分 | `delete-ok` |
| `work/editing-20260316` | archived | branch note と worktree note を保持 | `delete-ok` |
| `work/nn-develop-20260318` | active | active worktree 側で継続 | `keep-while-active` |
| `results/smolyak-validation-20260328` | active | active worktree 側で継続 | `keep-while-active` |
| `work/smolyak-integrator-lead-20260328` | active | active worktree 側で継続 | `keep-while-active` |
| `work/smolyak-integrator-opt-20260328` | active | active worktree 側で継続 | `keep-while-active` |
| `feature/fix/reports/merge/*` の保守 branch | mostly merged or low-signal | `main` と Git 履歴を参照 | `delete-ok` |
| `template-python-module` | upstream template tracking | `template/python-module` | `persistent` |

### Knowledge Landing Map

- experiment runner の設計判断:
  - [experiment_runner.md](/workspace/notes/themes/experiment_runner.md)
- 実験ディレクトリと results branch 運用:
  - [experiment_directory_planning.md](/workspace/notes/knowledge/experiment_directory_planning.md)
- Smolyak 積分器の本質的な観測:
  - [smolyak_integrator.md](/workspace/notes/themes/smolyak_integrator.md)
- 実験運用の実務ルール:
  - [experiment_operations.md](/workspace/notes/knowledge/experiment_operations.md)
  - [environment_setup.md](/workspace/notes/knowledge/environment_setup.md)
