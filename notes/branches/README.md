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
- 過去 branch の既存メモ本文は書き換えず、このディレクトリに補助的な summary を足します。
- active な branch は、対応する worktree の action log と carry-over 先を必ずここから辿れるようにします。

## Index

| Branch                                     | Note                                                                                                                 | Role                                 | Status   |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | -------- |
| `main`                                     | [main.md](/workspace/notes/branches/main.md)                                                                         | 既定のコード・文書の統合先           | active   |
| `results/smolyak-validation-20260328`      | [results_smolyak_validation_20260328.md](/workspace/notes/branches/results_smolyak_validation_20260328.md)           | Smolyak validation と結果保存        | active   |
| `work/experiment-runner-refactor-20260330` | [work_experiment_runner_refactor_20260330.md](/workspace/notes/branches/work_experiment_runner_refactor_20260330.md) | experiment_runner 改造               | active   |
| `work/nn-develop-20260318`                 | [work_nn_develop_20260318.md](/workspace/notes/branches/work_nn_develop_20260318.md)                                 | neuralnetwork 層別訓練開発           | active   |
| `work/smolyak-integrator-lead-20260328`    | [work_smolyak_integrator_lead_20260328.md](/workspace/notes/branches/work_smolyak_integrator_lead_20260328.md)       | Smolyak integrator 方向付け          | active   |
| `work/smolyak-integrator-opt-20260328`     | [work_smolyak_integrator_opt_20260328.md](/workspace/notes/branches/work_smolyak_integrator_opt_20260328.md)         | Smolyak integrator 最適化            | active   |
| `results/functional-smolyak-scaling`       | [results_functional_smolyak_scaling.md](/workspace/notes/branches/results_functional_smolyak_scaling.md)             | 旧版 Smolyak の完了結果と途中結果    | archived |
| `results/functional-smolyak-scaling-tuned` | [results_functional_smolyak_scaling_tuned.md](/workspace/notes/branches/results_functional_smolyak_scaling_tuned.md) | tuned Smolyak の継続実験             | active   |
| `work/smolyak-tuning-20260316`             | [work_smolyak_tuning_20260316.md](/workspace/notes/branches/work_smolyak_tuning_20260316.md)                         | tuned 積分器の設計・HLO 解析         | archived |
| `work/experiment-runner-module-20260316`   | [work_experiment_runner_module_20260316.md](/workspace/notes/branches/work_experiment_runner_module_20260316.md)     | 実験 runner のモジュール化           | archived |
| `work/editing-20260316`                    | [work_editing_20260316.md](/workspace/notes/branches/work_editing_20260316.md)                                       | 汎用編集用 worktree                  | archived |
| `work/jaxutil-test-expansion-20260317`     | [work_jaxutil_test_expansion_20260317.md](/workspace/notes/branches/work_jaxutil_test_expansion_20260317.md)         | `jax_util` 全体の test coverage 拡張 | archived |

## Workflow

1. worktree を切ったら、同時にこのディレクトリへ branch summary を作る。
1. `WORKTREE_SCOPE.md` で指定した action log と carry-over target を、この summary から辿れるようにする。
1. branch を閉じる前に、関連 note と final JSON のリンクを更新する。
