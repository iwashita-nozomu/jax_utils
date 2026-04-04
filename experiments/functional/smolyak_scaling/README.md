# Smolyak Scaling Experiments

`run_smolyak_scaling.py` は、`Smolyak` 積分器と same-budget `Monte Carlo` を同じ runner で比較するための baseline 実験機です。

## What It Runs

- ケース軸
  - `dimension`
  - `level`
  - `dtype`
  - `integration_method in {smolyak, monte_carlo}`
  - `execution_variant in {single, vmap}`
- `Monte Carlo` の sample count は、同じ `dimension, level` の `Smolyak` `num_evaluation_points` に合わせます。
- 各ケースは fresh child process で実行します。
- GPU 実行は `StandardFullResourceScheduler` で GPU ごとに worker slot を割り当てます。
- child 側は結果を JSONL へ直書きし、parent 側は `timeout / skipped / no_completion` などの制御結果だけを補完します。
- `timeout` が出たら、同じ `dimension × dtype × integration_method × execution_variant` の高レベルは skip します。

## Output

- `<run>.jsonl`
  - case 単位の逐次結果
- `<run>.json`
  - 実験全体の集計結果

主な記録項目:

- `num_terms`, `num_points`, `num_evaluation_points`, `same_budget_num_points`
- `first_call_ms`, `compile_ms`, `warm_runtime_ms`, `throughput_integrals_per_second`
- `integrator_init_seconds`, `sampling_seconds`, `device_transfer_seconds`
- `lowering_seconds`, `first_execute_seconds`, `warm_execute_seconds`
- `memory_checkpoints`, `device_memory_stats`, `process_rss_mb`
- `failure_kind`, `runner_failure_kind`, `failure_source`, `failure_phase`

## Runner Integration

- runner は `python/experiment_runner/` の現行 API を使います。
- `TaskContext` から
  - `environment_variables`
  - `runner_metadata`
  - `run_config`
  を受けます。
- `StandardWorker` の既定 initializer で child 先頭の環境初期化を行います。
- `ExecutionResult` は runner 状態制御に使い、実験 payload は JSONL に残します。

## Layout Note

- この topic は簡素化前の layout で、`results/` と補助 script を持っています。
- 新規 experiment の標準構成は [experiments/README.md](/workspace/experiments/README.md) を参照してください。
- 1 回の run に対する Markdown report の正本は [experiments/report/](/workspace/experiments/report/README.md) に置きます。
- 新規 experiment でこの legacy layout を再利用することを禁止します。

## Usage

CPU canary:

```bash
PYTHONPATH=python python3 experiments/functional/smolyak_scaling/run_smolyak_scaling.py \
  --platform cpu \
  --dimensions 1:2 \
  --levels 1:2 \
  --dtypes float32 \
  --integration-methods smolyak,mc \
  --execution-variants single,vmap \
  --num-accuracy-problems 8 \
  --num-repeats 1
```

GPU smoke:

```bash
PYTHONPATH=python python3 experiments/functional/smolyak_scaling/run_smolyak_scaling.py \
  --platform gpu \
  --gpu-indices 0 \
  --workers-per-gpu 1 \
  --dimensions 1:1 \
  --levels 1:2 \
  --dtypes float32 \
  --integration-methods smolyak,mc \
  --execution-variants single,vmap \
  --num-accuracy-problems 64 \
  --num-repeats 1
```

Full baseline:

```bash
PYTHONPATH=python python3 experiments/functional/smolyak_scaling/run_smolyak_scaling.py \
  --platform gpu \
  --gpu-indices 0,1,2 \
  --workers-per-gpu 1 \
  --dimensions 1:100 \
  --levels 1:50 \
  --dtypes all \
  --integration-methods all \
  --execution-variants all \
  --timeout-seconds 300
```

## Notes

- `single` と `vmap` は別ケースとして記録します。
- compile 時間が長くても baseline として残す方針なので、timeout は `300s` を既定にしています。
- 途中失敗した run は診断用として残し、resume はしません。再実行は新しい output で 0 から行います。
- topic の naming rule や report 置き場を変更した場合は、この README と `documents/coding-conventions-experiments.md` を同時に更新します。
