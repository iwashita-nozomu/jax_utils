# Smolyak 実験基盤チェックポイント

日付: 2026-03-20\
関連 worktree: `work/smolyak-improvement-20260318`

## Source

- 当日の worktree で `experiment_runner` と `experiments/smolyak_experiment/` を接続して行った smoke / small run の観測。
- 実装本体は現在 [experiment_runner](/workspace/python/experiment_runner) と [smolyak_experiment](/workspace/experiments/smolyak_experiment) に統合済み。

## Summary

- JAX の fork 問題を避けるため、child process は spawn 前提で扱う方針が有効だった。
- `TaskContext["environment_variables"]` で GPU 割当と JAX メモリ設定を渡す構成が、runner と worker の責務分離としてうまく機能した。
- case ごとの JSONL 逐次保存と file lock により、長時間 run の partial 回収がしやすくなった。

## Observations

- smoke run では数ケース規模を短時間で回せた。
- small run では case 数が増えても scheduler と context 受け渡しは安定していた。
- worker の pickle 検証を早めに入れておくと、pool 起動前に失敗を検出できる。

## Implementation Carry-Over

- spawn context と picklable check は [jax_context.py](/workspace/python/experiment_runner/jax_context.py) と [runner.py](/workspace/python/experiment_runner/runner.py) に反映済み。
- 環境変数の反映は [context_utils.py](/workspace/python/experiment_runner/context_utils.py) に分離済み。
- host memory / GPU slot の同時監視は [resource_scheduler.py](/workspace/python/experiment_runner/resource_scheduler.py) に統合済み。

## Consideration

- medium 以上の長時間 run は `main` ではなく `results/*` branch の worktree で回すのが自然。
- smoke / verified のような短時間 run は `main` 側に近い作業ツリーで試し、長時間生成物だけを results branch に逃がすと扱いやすい。
