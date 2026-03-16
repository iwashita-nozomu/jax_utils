# Experiment Result JSON Archive

このディレクトリには、`main` に持ち帰る最小限の final JSON を置きます。

- 目的は、後から別の図や集計を再生成できるようにすることです。
- raw な JSONL、巨大ログ、途中経過の全ファイルまでは置きません。
- branch を代表する final JSON、あるいは partial でも再解析価値の高い JSON を選んで置きます。

各 JSON について、対応する note から

- branch 名
- 元の results branch
- 元データの所在
- その JSON を持ち帰った理由

が辿れるようにします。

## Current Files

- [tuned_smolyak_partial_results_20260316.json](/workspace/notes/experiments/results/tuned_smolyak_partial_results_20260316.json)
  - source branch:
    - `results/functional-smolyak-scaling-tuned`
  - source file:
    - `/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T132125Z_partial.json`
  - reason:
    - raw JSONL 相当の `cases` 情報を含み、後から別の図や集計を再生成できるため
