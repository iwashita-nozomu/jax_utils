# Experiment Runner Usage

この note は、`python/experiment_runner/` を実験コードからどう使うかを、実務向けにまとめた利用ガイドです。

対象 reader は次です。

- 新しく `experiment_runner` を使い始める人
- `main` へ持ち帰った後に既存実験を移植する人
- GPU 実験で env 設定と runner の責務を切り分けたい人

## 1. 何を使い分けるか

`experiment_runner` には、いま大きく 4 つの入口があります。

### 1.1 `StandardRunner`

標準の実験実行入口です。

- case ごとに fresh child process を起動します
- JAX / CUDA の import-sensitive な state を case 間で持ち越しにくいです
- 一般的な実験はまずこれを使う前提にします

使う主な型:

- `StandardWorker`
- `StandardScheduler`
- `StandardRunner`

### 1.2 `StandardFullResourceScheduler`

GPU、host memory、worker slot を同時に見たいときの標準 scheduler です。

- GPU 実験ではこちらを基本にします
- `resource_estimate(case)` をもとに割当を決めます
- `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` を `TaskContext["environment_variables"]` に入れます

使う主な型:

- `FullResourceCapacity`
- `FullResourceEstimate`
- `GPUDeviceCapacity`
- `GPUEnvironmentConfig`
- `StandardFullResourceScheduler`

### 1.3 `subprocess_scheduler`

host 側で `Popen` を直接持ちたい実験向けです。

- benchmark 系
- worker slot を明示したい run
- pid と slot を host で管理したい run
- `RuntimeMonitor` と素直につなぎたい run

使う主な API:

- `build_worker_slots()`
- `apply_worker_environment()`
- `run_cases_with_subprocess_scheduler()`

### 1.4 `RuntimeMonitor`

軽量 monitor です。

- HTML GUI
- JSON API
- worker event
- runtime snapshot

`subprocess_scheduler` と組み合わせるのが現状いちばん自然です。

## 2. 最初に守るルール

### 2.1 実験コード側で env を直接いじらない

これはかなり重要です。

実験コード側では、次の env を直接設定しない方針に寄せます。

- `CUDA_VISIBLE_DEVICES`
- `NVIDIA_VISIBLE_DEVICES`
- `JAX_PLATFORMS`
- `XLA_PYTHON_CLIENT_PREALLOCATE`
- `XLA_PYTHON_CLIENT_MEM_FRACTION`
- `XLA_PYTHON_CLIENT_ALLOCATOR`
- `TF_GPU_ALLOCATOR`
- その他 `XLA_*` / JAX allocator 系 env

理由:

- GPU 割当の責務は runner / scheduler 側に寄せたい
- 実験コードが横から env を変えると、scheduler の契約とぶつかる
- JAX は import 後の env 変更に弱く、バグの原因が追いにくい

実験コードが持つべきなのは「意図」です。

- この case は GPU を 1 枚使う
- この実験は CPU で動かしたい
- この run では preallocation を切りたい

実際の env 組み立ては runner 側に寄せます。

### 2.2 task / worker / case / context は picklable にする

`StandardRunner` は `spawn` で fresh child process を起動します。

そのため、次を守る必要があります。

- task は top-level function か top-level class に置く
- local lambda を worker に直接渡さない
- case は JSON-like な dict や dataclass に寄せる
- `TaskContext` は辞書で、pickle 可能な値だけを入れる

### 2.3 JAX import より前に env を適用する

GPU 実験では、worker child 側で JAX import 前に env を適用する必要があります。

標準パターン:

```python
from experiment_runner.context_utils import apply_environment_variables

def task(case, context):
    apply_environment_variables(context)
    import jax
    ...
```

## 3. 最小の CPU 例

CPU 実験だけなら、最初はこれで十分です。

```python
from experiment_runner.runner import (
    StandardResourceCapacity,
    StandardRunner,
    StandardScheduler,
    StandardWorker,
)


def task(case: int, context: dict[str, object]) -> None:
    del context
    print(f"run case={case}")


cases = [0, 1, 2, 3]

scheduler = StandardScheduler(
    resource_capacity=StandardResourceCapacity(max_workers=4),
    cases=cases,
)
worker = StandardWorker(task)
runner = StandardRunner(scheduler)
runner.run(worker)
```

これで分かること:

- FIFO 実行になる
- `max_workers` 本まで並列になる
- case ごとに fresh process が使われる

## 4. GPU 実験の基本形

GPU 実験では `StandardFullResourceScheduler` を使います。

### 4.1 worker 側

```python
from experiment_runner.context_utils import apply_environment_variables
from experiment_runner.protocols import TaskContext
from experiment_runner.resource_scheduler import FullResourceEstimate


def task(case: dict[str, object], context: TaskContext) -> None:
    apply_environment_variables(context)

    import jax
    import jax.numpy as jnp

    device = jax.devices("gpu")[0]
    with jax.default_device(device):
        x = jnp.ones((1024, 1024))
        y = jnp.tanh(x @ x.T)
        jax.block_until_ready(y)


def resource_estimate(case: dict[str, object]) -> FullResourceEstimate:
    del case
    return FullResourceEstimate(
        host_memory_bytes=512 * 1024 * 1024,
        gpu_count=1,
        gpu_memory_bytes=2 * 1024 * 1024 * 1024,
        gpu_slots=1,
    )
```

### 4.2 scheduler / runner 側

```python
from experiment_runner.resource_scheduler import (
    FullResourceCapacity,
    GPUEnvironmentConfig,
    StandardFullResourceScheduler,
)
from experiment_runner.runner import StandardRunner, StandardWorker


cases = [{"case_id": i} for i in range(8)]

capacity = FullResourceCapacity.from_system(
    max_workers=2,
    gpu_max_slots=1,
)

scheduler = StandardFullResourceScheduler(
    resource_capacity=capacity,
    cases=cases,
    estimate_builder=resource_estimate,
    gpu_environment_config=GPUEnvironmentConfig(
        memory_fraction=0.4,
    ),
)

worker = StandardWorker(task)
runner = StandardRunner(scheduler)
runner.run(worker)
```

この構成で scheduler がやること:

- 空いている GPU を選ぶ
- `CUDA_VISIBLE_DEVICES` を child に渡す
- scheduler 内部の GPU memory / slot 帳簿を更新する

### 4.3 `disable_gpu_preallocation` と `GPUEnvironmentConfig`

単純に preallocation を切りたいだけなら:

```python
scheduler = StandardFullResourceScheduler(
    ...,
    disable_gpu_preallocation=True,
)
```

より明示的に allocator を調整したいなら:

```python
from experiment_runner.resource_scheduler import GPUEnvironmentConfig

gpu_env = GPUEnvironmentConfig(
    disable_preallocation=True,
    memory_fraction=0.4,
    xla_client_allocator="platform",
    tf_gpu_allocator="cuda_malloc_async",
    use_cuda_host_allocator=False,
)
```

使い分け:

- `disable_gpu_preallocation=True`
  - まず簡単に安全側へ寄せたいとき
- `GPUEnvironmentConfig(...)`
  - 比較実験したいとき
  - allocator 方針を note とテストに残したいとき

## 5. `context_builder` は何に使うか

`context_builder` は env を好き勝手に入れる場所ではなく、case ごとの metadata を child に渡す場所です。

良い使い方:

- `case_id`
- 出力 JSONL の path
- 実験種別
- task 内の分岐キー
- allocator 以外の harmless な実験パラメータ

例:

```python
def context_builder(case: dict[str, object]) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "result_path": f"/tmp/case_{case['case_id']}.json",
        "task_key": "smolyak_integrate",
    }
```

避けたい使い方:

- `CUDA_VISIBLE_DEVICES`
- `JAX_PLATFORMS`
- `XLA_*`

runner が管理する env と衝突しやすいからです。

## 6. `subprocess_scheduler` を使う場面

次のような run では、`subprocess_scheduler` の方が向いています。

- host が pid を直接追いたい
- worker slot を固定したい
- child の stdout completion record を直接扱いたい
- monitor と素直につなぎたい

基本形:

```python
from pathlib import Path

from experiment_runner.subprocess_scheduler import (
    build_worker_slots,
    run_cases_with_subprocess_scheduler,
)


worker_slots = build_worker_slots(
    "gpu",
    gpu_indices=[0, 1],
    workers_per_gpu=1,
)

results = run_cases_with_subprocess_scheduler(
    cases=cases,
    worker_slots=worker_slots,
    timeout_seconds=3600,
    build_child_command=build_child_command,
    build_parent_failure_result=build_parent_failure_result,
    fallback_jsonl_output_path=Path("results.jsonl"),
    cwd=Path.cwd(),
)
```

`StandardRunner` と違って host が child process を直接持つので、運用観測はしやすいです。

## 7. Monitor の使い方

現状の monitor は軽量な HTML + JSON API です。

### 7.1 基本

```python
from experiment_runner.monitor import RuntimeMonitor

monitor = RuntimeMonitor.for_run(
    bind_host="127.0.0.1",
    port=8765,
    sample_interval_seconds=1.0,
)
monitor.start()
try:
    ...
finally:
    monitor.stop()
```

### 7.2 `subprocess_scheduler` とつなぐ

```python
results = run_cases_with_subprocess_scheduler(
    ...,
    monitor=monitor,
)
```

### 7.3 主な endpoint

- `GET /`
- `GET /api/v1/snapshot`
- `GET /api/v1/history`
- `GET /api/v1/events`
- `GET /healthz`

### 7.4 `run` と `daemon`

`RuntimeMonitor` には 2 つの作り方があります。

- `RuntimeMonitor.for_run(...)`
  - run に付属する monitor
  - event を厚めに持つ
- `RuntimeMonitor.for_daemon(...)`
  - 常駐寄り
  - history と event を軽めに持つ

## 8. いまの推奨パターン

2026-03-31 時点では、次が基本です。

### 8.1 通常の実験

- `StandardWorker`
- `StandardFullResourceScheduler`
- `StandardRunner`

### 8.2 host 側で pid / slot を強く見たい run

- `build_worker_slots()`
- `run_cases_with_subprocess_scheduler()`
- `RuntimeMonitor`

### 8.3 GPU env の扱い

- 実験コードは env を直接セットしない
- scheduler / subprocess helper が組み立てる
- JAX allocator 系は `GPUEnvironmentConfig` に寄せる

## 9. テストの回し方

通常:

```bash
pytest -q python/tests/experiment_runner
```

型チェック:

```bash
pyright python/experiment_runner python/tests/experiment_runner
```

重い GPU 比較:

```bash
RUN_HEAVY_TESTS=1 pytest -q -s python/tests/experiment_runner/test_runner_gpu.py
```

この GPU テストは、同じ GPU に複数 worker を重ねたときの allocator profile 比較にも使えます。

## 10. よくある失敗

### 10.1 local lambda を worker に渡す

`spawn` child に送れず、pickle エラーになります。

### 10.2 JAX import 後に env を変える

`CUDA_VISIBLE_DEVICES` が効かず、意図しない GPU 競合になります。

### 10.3 実験コード側と scheduler 側の両方で env をいじる

どちらが最終値か分からなくなり、再現性が落ちます。

### 10.4 `gpu_count` / `gpu_memory_bytes` の見積もりが雑すぎる

scheduler の帳簿と実使用量がずれます。

## 11. `main` に持ち帰るときの利用ルール

`main` に取り込んだ後は、次をチームの共通ルールにしたいです。

1. GPU 実験は `StandardFullResourceScheduler` か `subprocess_scheduler` のどちらかに寄せる。
1. 実験コード側では runtime env を直接セットしない。
1. JAX allocator 設定は `GPUEnvironmentConfig` に集約する。
1. monitor を使う場合は `RuntimeMonitor` を入口にする。
1. 新しい実験 script はまずこの note の CPU / GPU 最小例を土台にする。

## 12. 関連 note

- [experiment_runner.md](/workspace/documents/experiment_runner.md)
- [experiment_runner_realtime_monitor.md](/workspace/notes/themes/experiment_runner_realtime_monitor.md)
- [experiment_runner_main_integration.md](/workspace/notes/themes/experiment_runner_main_integration.md)
