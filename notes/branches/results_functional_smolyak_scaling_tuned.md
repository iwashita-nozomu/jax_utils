# Tuned Smolyak Results Branch Summary

## Branch

- `results/functional-smolyak-scaling-tuned`

## Role

- tuned Smolyak 積分器と、新しい experiment runner を使った継続実験の結果を保持する branch
- 旧版より explicit grid 展開を減らした実装で、実行限界と bottleneck を再評価している

## Primary Notes

- current partial results:
  - [tuned_smolyak_partial_results_20260316.md](/workspace/notes/experiments/tuned_smolyak_partial_results_20260316.md)
- tuning worktree 由来の吸い出し:
  - [worktree_smolyak_tuning_2026-03-16.md](/workspace/notes/worktrees/worktree_smolyak_tuning_2026-03-16.md)
- runner modularization の吸い出し:
  - [worktree_experiment_runner_module_2026-03-16.md](/workspace/notes/worktrees/worktree_experiment_runner_module_2026-03-16.md)
- 旧版との比較用 note:
  - [legacy_smolyak_results_20260316.md](/workspace/notes/experiments/legacy_smolyak_results_20260316.md)

## Current Findings

この branch の partial では、`547` ケース時点で `ok=99`, `failed=448` だった。重要なのは、`level=1` の成功ケースが全 dtype で `num_points=1` に留まっているにもかかわらず、次元とともに `integrator_init_seconds` と `process_rss_mb` が急増していることである。したがって、今のボトルネックは積分本体より、初期化・lowering・compile 側にあると読むのが自然である。

case 順は途中で `dtype -> level -> dimension` から `level -> dimension -> dtype` を経て、現在は `dimension -> level -> dtype` に見直している。これは、途中停止しても frontier と dtype 比較を読みやすくするための変更である。

## Interpretation

- この branch は「Smolyak 数値精度の最終比較」より、「今の tuned 実装がどこで詰まるか」を見極める基盤になっている
- 実験 runner の改善で GPU 配布や JSONL 永続化はかなり安定したが、積分器の初期化・compile cost がまだ支配的

## Status

- active
- 継続観測中
