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
- 過去 branch の既存メモ本文は原則として書き換えず、このディレクトリに補助的な summary を足します。

## Index

| Branch | Note | Role | Status |
| --- | --- | --- | --- |
| `main` | [main.md](/workspace/notes/branches/main.md) | 既定のコード・文書の統合先 | active |
| `results/functional-smolyak-scaling` | [results_functional_smolyak_scaling.md](/workspace/notes/branches/results_functional_smolyak_scaling.md) | 旧版 Smolyak の完了結果と途中結果 | archived |
| `results/functional-smolyak-scaling-tuned` | [results_functional_smolyak_scaling_tuned.md](/workspace/notes/branches/results_functional_smolyak_scaling_tuned.md) | tuned Smolyak の継続実験 | active |
| `work/smolyak-tuning-20260316` | [work_smolyak_tuning_20260316.md](/workspace/notes/branches/work_smolyak_tuning_20260316.md) | tuned 積分器の設計・HLO 解析 | archived |
| `work/experiment-runner-module-20260316` | [work_experiment_runner_module_20260316.md](/workspace/notes/branches/work_experiment_runner_module_20260316.md) | 実験 runner のモジュール化 | archived |
| `work/editing-20260316` | [work_editing_20260316.md](/workspace/notes/branches/work_editing_20260316.md) | 汎用編集用 worktree | active |
| `work/jaxutil-test-expansion-20260317` | [work_jaxutil_test_expansion_20260317.md](/workspace/notes/branches/work_jaxutil_test_expansion_20260317.md) | `jax_util` 全体の test coverage 拡張 | archived |
