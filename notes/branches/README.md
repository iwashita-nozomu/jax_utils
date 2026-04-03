# Branch Notes

`notes/branches/` は、branch ごとの要約と関連 note への入口をまとめるディレクトリです。

- 詳細な実験考察は `notes/experiments/`
- 削除済み worktree 由来の判断は `notes/worktrees/`
- 日付ごとの流れは `diary/`

に残し、このディレクトリでは

- その branch が何のために存在したか
- どこまで取り込まれたか
- どの note を先に読むべきか

を短く整理します。

## Format

- 1 branch 1 file を基本とします。
- タイトルは branch 名そのものではなく、topic-first で付けます。
- branch 名、役割、現在の状態、読むべき note、主要な知見を最初に書きます。
- 過去 branch の note は、履歴説明を増やさず、状態と知見の受け皿だけ更新します。
- active な branch は、対応する worktree の action log と carry-over 先を必ずここから辿れるようにします。

## Retention Labels

- `persistent`: 継続的な入口または raw 結果の保管先として残します。
- `keep-while-active`: 対応する worktree や実験が動いている間は残します。
- `delete-ok`: 知見の吸収が終わっており、削除してよい branch です。

## Index

| Branch | Note | Role | Status | Retention |
| --- | --- | --- | --- | --- |
| `main` | [main.md](/workspace/notes/branches/main.md) | 既定のコード・文書の統合先 | active | `persistent` |
| `results/smolyak-validation-20260328` | [results_smolyak_validation_20260328.md](/workspace/notes/branches/results_smolyak_validation_20260328.md) | Smolyak validation と結果保存 | active | `keep-while-active` |
| `results/smolyak-experiment-20260321` | [results_smolyak_experiment_20260321.md](/workspace/notes/branches/results_smolyak_experiment_20260321.md) | `smolyak_experiment` と配置設計 | archived | `delete-ok` |
| `work/experiment-runner-refactor-20260330` | [work_experiment_runner_refactor_20260330.md](/workspace/notes/branches/work_experiment_runner_refactor_20260330.md) | experiment_runner 改造 | archived | `delete-ok` |
| `work/nn-develop-20260318` | [work_nn_develop_20260318.md](/workspace/notes/branches/work_nn_develop_20260318.md) | neuralnetwork 層別訓練開発 | active | `keep-while-active` |
| `work/smolyak-improvement-20260318` | [work_smolyak_improvement_20260318.md](/workspace/notes/branches/work_smolyak_improvement_20260318.md) | Smolyak 実験簡略化と運用整理 | archived | `delete-ok` |
| `work/smolyak-integrator-lead-20260328` | [work_smolyak_integrator_lead_20260328.md](/workspace/notes/branches/work_smolyak_integrator_lead_20260328.md) | Smolyak integrator 方向付け | active | `keep-while-active` |
| `work/smolyak-integrator-opt-20260328` | [work_smolyak_integrator_opt_20260328.md](/workspace/notes/branches/work_smolyak_integrator_opt_20260328.md) | Smolyak integrator 最適化 | active | `keep-while-active` |
| `results/functional-smolyak-scaling` | [results_functional_smolyak_scaling.md](/workspace/notes/branches/results_functional_smolyak_scaling.md) | 旧版 Smolyak の完了結果と途中結果 | archived | `persistent` |
| `results/functional-smolyak-scaling-tuned` | [results_functional_smolyak_scaling_tuned.md](/workspace/notes/branches/results_functional_smolyak_scaling_tuned.md) | tuned Smolyak の継続実験 | archived | `delete-ok` |
| `work/smolyak-tuning-20260316` | [work_smolyak_tuning_20260316.md](/workspace/notes/branches/work_smolyak_tuning_20260316.md) | tuned 積分器の設計・HLO 解析 | archived | `delete-ok` |
| `work/experiment-runner-module-20260316` | [work_experiment_runner_module_20260316.md](/workspace/notes/branches/work_experiment_runner_module_20260316.md) | 実験 runner のモジュール化 | archived | `delete-ok` |
| `work/editing-20260316` | [work_editing_20260316.md](/workspace/notes/branches/work_editing_20260316.md) | 汎用編集用 worktree | archived | `delete-ok` |
| `work/jaxutil-test-expansion-20260317` | [work_jaxutil_test_expansion_20260317.md](/workspace/notes/branches/work_jaxutil_test_expansion_20260317.md) | `jax_util` 全体の test coverage 拡張 | archived | `delete-ok` |

## Branches Without Dedicated Notes

次の branch は履歴価値はあるが、専用 note を増やすより `main` 側の inventory と topic note で追う方が自然です。

| Branch | Current Reading | Retention |
| --- | --- | --- |
| `work/experiment-runner-generalization-20260317` | [experiment_runner.md](/workspace/notes/themes/experiment_runner.md) | `delete-ok` |
| `feature/experiment-runner-ci-tests` | `git log` と CI 履歴 | `delete-ok` |
| `fix/docs-guides-20260319` | 文書本体と `main` | `delete-ok` |
| `fix/line-length-20260318` | 静的解析 report と `main` | `delete-ok` |
| `fix/ruff-imports-20260318` | 静的解析 report と `main` | `delete-ok` |
| `fix/type-annotations-20260318` | 型注釈の現行 code と `main` | `delete-ok` |
| `reports/static-analysis-20260318` | `reports/` と `main` | `delete-ok` |
| `merge/docs-guides-20260319` | merge 履歴のみ | `delete-ok` |
| `template-python-module` | upstream template と `template/python-module` | `persistent` |

## Workflow

1. worktree を切ったら、同時にこのディレクトリへ branch summary を作る。
1. `WORKTREE_SCOPE.md` で指定した action log と carry-over target を、この summary から辿れるようにする。
1. branch を閉じる前に、関連 note と final JSON のリンクを更新する。
