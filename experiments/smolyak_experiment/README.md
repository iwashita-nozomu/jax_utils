# Smolyak Experiment Module

`experiments/smolyak_experiment/` は、Smolyak 積分器のスケーリング実験を
`experiment_runner` 上で動かすための実験コードです。

従来は `results/*` branch の worktree を強く想定していましたが、新標準では branch 分離は必須ではありません。
このディレクトリの `results/` は legacy な生成物置き場として残っています。

## Layout Note

- この topic は簡素化前の layout で、`results/` と複数 helper file を持っています。
- 新規 experiment の標準構成は [experiments/README.md](/workspace/experiments/README.md) を参照してください。
- 1 回の run に対する Markdown report の正本は [experiments/report/](/workspace/experiments/report/README.md) に置きます。
- 新規 experiment でこの legacy layout を再利用することを禁止します。

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

現在の legacy 実装では、各 run は `experiments/smolyak_experiment/results/` 配下に
`<run_id>.jsonl` と `<run_id>.json` を保存し、report を生成する場合は `<run_id>_report/` も生成します。
この `results/` 配下は `.gitignore` で生成物扱いにします。

run は 1 回の fresh 実行で完走させる前提です。途中で止まった場合は同じ
`jsonl` や legacy の最終 JSON に継ぎ足さず、新しい run_id の run を 0 から
やり直します。JSONL は run 中の progress 記録であり、resume 入力ではありません。
README にない ad hoc な 1 case 実行や subset 実行は debug / smoke に限り、
正式な比較結果や carry-over の正本には使いません。

## Notes

- worker は `TaskContext["environment_variables"]` を `apply_environment_variables()` で適用してから JAX を import します。
- `SmolyakIntegrator.integrate()` 自体は module 側で JIT せず、実験コード側で `eqx.filter_jit()` をかけます。
- HLO ダンプは `from jax_util.hlo import dump` を使って補助的に出します。
- size preset は `smoke`, `small`, `verified`, `medium`, `large` の 5 種です。
- raw JSONL や trace を `main` に常設しない運用は [documents/coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md) に従います。
- naming rule や出力先を変えた場合は、この README と `documents/coding-conventions-experiments.md` を同時に更新します。
