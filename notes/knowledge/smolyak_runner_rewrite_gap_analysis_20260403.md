# Smolyak Runner Rewrite Gap Analysis

## Context

`smolyak_scaling` は現在 `StandardRunner` / `StandardFullResourceScheduler` ベースへ置き換わっている。

- current file:
  - [experiments/functional/smolyak_scaling/run_smolyak_scaling.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/run_smolyak_scaling.py)
- richer reference implementation before replacement:
  - `git show 25f2345:experiments/functional/smolyak_scaling/run_smolyak_scaling.py`

このメモの目的は、runner 置換後に実験コードから落ちた機能を整理し、書き直し時に何を戻すべきかを固定することである。

## Checkpoint

比較を始める前に、現在の branch 状態は GitHub へ push 済み。

- pushed branch:
  - `GitHub/work/smolyak-integrator-opt-20260328`
- latest checkpoint:
  - `7338c4a chore(results): refresh latest smolyak scaling snapshot`

## What The New Runner Already Gives Us

現行の runner stack を使う利点は明確で、ここは実験コードから削ってよい。

- `StandardRunner`
  - fresh child process 実行
  - parent-side timeout
  - normalized `ExecutionResult`
  - `on_case_started` / `on_case_finished`
  - `skip_controller`
  - `monitor`
- `StandardFullResourceScheduler`
  - GPU slot / host memory / worker slot 管理
  - `runner_metadata`
  - `environment_variables`
  - GPU visibility と CPU affinity の割当
- `result_io`
  - JSONL append / read helpers
- `context_utils.apply_environment_variables`
  - child 側 env 適用

要するに、今後の rewrite では

- child CLI
- host-managed subprocess loop
- worker slot serialization
- fallback JSONL write の専用実装

は不要である。

## Critical Regressions

現行実装は、単に機能が減っただけではなく、現在の積分器 API と齟齬がある。

### 1. Removed Integrator Fields Are Still Referenced

現行 `run_smolyak_scaling.py` は次を待っている。

- `integrator.term_levels`
- `integrator.term_num_points`

しかし現在の `SmolyakIntegrator` は dynamic term generation へ移っており、これらは存在しない。

したがって現行 baseline はそのままでは壊れている。

### 2. Wrapper `integrate(f, integrator)` Path Is Not Used

現行コードは

- `current_integrator.integrate(...)`

を直接呼んでいる。現状の batching / wrapper policy は `integrate(f, integrator)` 側へ寄せつつあるので、ここは wrapper 経由へ戻すべきである。

## Lost Experiment Axes

reference implementation にあったが current implementation で落ちた主な軸は次。

### 1. `integration_method`

落ちたもの:

- `smolyak`
- `monte_carlo`

current は `Smolyak` 固定で、same-budget Monte Carlo 比較が消えている。

### 2. `execution_variant`

落ちたもの:

- `single`
- `vmap`

current 実装は 1 ケースの中で

- accuracy 用 batched eval
- repeated single

を混ぜており、case 軸として `single` / `vmap` を切っていない。

その結果、

- timeout frontier
- summary
- failure table

が variant ごとに取れない。

### 3. Same-Budget Metadata

落ちたもの:

- `same_budget_num_points`
- `num_samples`
- `dense_integrand_matrix_upper_bound_bytes`

same-budget 比較の前提が JSONL に残らない。

## Lost Measurement Detail

reference implementation にあったが current で落ちた計測項目は多い。

### Timing

落ちたもの:

- `first_call_ms`
- `compile_ms`
- `warm_runtime_ms`
- `throughput_integrals_per_second`
- `measurement_problem_count`
- `vmap_batch_size`
- `timing_probe_seconds`
- `warmup_seconds`
- `measured_runtime_seconds`

current は

- `warmup_seconds`
- `batched_integral_seconds`
- `avg_integral_seconds`

のような旧式の混ざった指標に戻っている。

### Memory / Transfer

落ちたもの:

- `memory_checkpoints`
  - `after_init`
  - `after_transfer`
  - `after_benchmark`
  - `after_execute`
  - `after_host_copy`
- `coeff_inputs_device_nbytes`
- `measured_values_device_nbytes`

現在の runner rewrite でも、これらは実験コード側で再導入が必要である。

### Failure Diagnostics

落ちたもの:

- `runner_failure_kind`
- `failure_source`
- `host_oom` / `oom` / `timeout` の parent-child 区別
- `num_skipped`

current は `ExecutionResult` を使っているが、JSON schema へ十分に落としていない。

## Lost CLI Surface

reference implementation にあり、current 実装で落ちた CLI / config は次。

- `--integration-methods`
- `--execution-variants`
- `--chunk-size`
- `--xla-memory-fraction`
- `--xla-allocator`
- `--xla-tf-gpu-allocator`
- `--xla-use-cuda-host-allocator`
- `--xla-memory-scheduler`
- `--xla-gpu-enable-while-loop-double-buffering`
- `--xla-latency-hiding-scheduler-rerun`
- `--jax-compiler-enable-remat-pass`
- `--monitor-port`
- `--monitor-bind-host`
- `--monitor-sample-interval-seconds`
- `--monitor-enable-http`

ここで重要なのは、runner が進化したことで不要になったのではなく、単に current rewrite で省かれただけ、という点である。

## Lost Control Logic

### 1. Timeout Frontier Skip

reference 側で入れていた

- timeout した `(dimension, dtype, integration_method, execution_variant)` について
- 同 `dimension` の `level >= timed_out_level` を skip

は current 実装では消えている。

runner には now `skip_controller` があるので、これは runner 標準経路で戻すべきである。

### 2. Monitor Wiring

current `run_smolyak_scaling.py` は `StandardRunner` を使っているが、`monitor` は配線していない。

runner 更新後は

- `StandardRunner(..., monitor=...)`

で素直に入れられるので、ここは current code の不足である。

## Lost Summary Structure

reference implementation では summary / frontier が

- `integration_method`
- `execution_variant`
- `dtype_name`

の 3 軸で整理されていた。

current 実装では

- `dtype` のみ

へ後退している。

これにより

- `smolyak vs monte_carlo`
- `single vs vmap`
- `skipped vs failed`

の比較がトップレベル JSON だけでは追えない。

## Rewrite Guidance

## Keep

- `StandardRunner`
- `StandardFullResourceScheduler`
- `StandardWorker`
- `result_io`
- `apply_environment_variables`

## Restore In Experiment Code

### Case Schema

- `integration_method`
- `execution_variant`
- `same_budget_num_points`

### Result Schema

- detailed timing fields
- memory checkpoints
- normalized failure metadata
- `num_skipped`

### CLI / Config

- XLA flags
- monitor flags
- `chunk_size`
- `integration_methods`
- `execution_variants`

### Control

- timeout frontier `SkipController`

## Do Not Reintroduce

- legacy subprocess scheduler
- child JSON CLI protocol
- manual worker slot serialization

## Recommended Rewrite Order

1. fix current correctness regression
   - remove `term_levels` / `term_num_points` assumptions
   - route through `integrate(f, integrator)`
2. restore case axes
   - `integration_method`
   - `execution_variant`
3. restore detailed metrics
4. restore `SkipController`
5. restore monitor / XLA CLI
6. restore summary / frontier schema

## Bottom Line

runner rewrite 自体は正しい方向だが、current `smolyak_scaling` は

- runner に任せてよい部分

と

- baseline として必要だった experiment-specific logic

を一緒に削りすぎている。

次の書き直しでは

- process / resource orchestration は runner に委譲
- case schema, metrics, fairness, summaries は experiment code に戻す

という線引きに戻すのがよい。
