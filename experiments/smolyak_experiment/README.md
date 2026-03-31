# Smolyak Experiment Module

`experiments/smolyak_experiment/` は、Smolyak 積分器のスケーリング実験を
`experiment_runner` 上で動かすための実験コードです。

長時間 run は `main` ではなく `results/*` branch の worktree で回し、
このディレクトリの `results/` は生成物の一時置き場として扱います。

## Files

- `cases.py`: 次元・レベル・dtype の直積からケースを作り、リソース見積もりも返します。
- `runner_config.py`: 実験サイズごとの設定を持つ `SmolyakExperimentConfig` を定義します。
- `results_aggregator.py`: JSONL から dtype / dimension / level ごとの集計を行います。
- `run_smolyak_experiment_simple.py`: 実験の実行入口です。

## Run

```bash
python3 experiments/smolyak_experiment/run_smolyak_experiment_simple.py --size smoke
python3 experiments/smolyak_experiment/run_smolyak_experiment_simple.py --size small
python3 experiments/smolyak_experiment/run_smolyak_experiment_simple.py --size verified --max-workers 1
python3 experiments/smolyak_experiment/run_smolyak_experiment_simple.py --size medium
```

各 run は `experiments/smolyak_experiment/results/<size>/` に
`results_<timestamp>.jsonl` と `final_results_<timestamp>.json` を保存します。
この `results/` 配下は `.gitignore` で生成物扱いにします。

run は 1 回の fresh 実行で完走させる前提です。途中で止まった場合は同じ
`jsonl` や `final_results` に継ぎ足さず、新しい timestamp の run を 0 から
やり直します。JSONL は run 中の progress 記録であり、resume 入力ではありません。

## Notes

- worker は `TaskContext["environment_variables"]` を `apply_environment_variables()` で適用してから JAX を import します。
- `SmolyakIntegrator.integrate()` 自体は module 側で JIT せず、実験コード側で `eqx.filter_jit()` をかけます。
- HLO ダンプは `jax_util.hlo.dump.dump_hlo_jsonl` が利用可能な場合だけ補助的に出します。
- size preset は `smoke`, `small`, `verified`, `medium`, `large` の 5 種です。
- raw JSONL や trace を `main` に常設しない運用は [documents/coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md) に従います。
