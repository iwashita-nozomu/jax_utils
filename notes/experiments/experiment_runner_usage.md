# Experiment Runner Usage

この note は、`python/experiment_runner/` を実験コードからどう使うかを、実務向けにまとめた利用ガイドです。
実験全般の標準手順は [experiment-workflow.md](/workspace/documents/experiment-workflow.md) を正本とし、この note では runner、scheduler、monitor、GPU env の使い分けに集中します。

対象 reader は次です。

- 新しく `experiment_runner` を使い始める人
- `main` へ持ち帰った後に既存実験を移植する人
- GPU 実験で env 設定と runner の責務を切り分けたい人

研究の問い、比較設計、実験 note の体裁は次を正本にします。

- [research-workflow.md](/workspace/documents/research-workflow.md)
- [experiment-report-style.md](/workspace/documents/experiment-report-style.md)

## 0. 現状把握から始める

新しい実験を始めるときは、いきなり long run を投げず、まず現状を固定します。

### 0.1 最初に固定すること

少なくとも次の 4 つを run 前に 1 回言語化します。

- `Question:`
  - 何を確かめたいか。速度、精度、メモリ、failure pattern のどれを見たいか。
- `Comparison Target:`
  - main 実装、旧 script、別 scheduler、別 allocator 設定のどれと比べるか。
- `Metrics:`
  - 何を JSON と note に残すか。少なくとも時間、成功率、failure kind、主要誤差を含めます。
- `Stop Condition:`
  - smoke が通ればよいのか、verified run まで必要か、正式な比較表を作るのか。

ここが曖昧なまま実験を始めると、

- debug run と正式 run が混ざる
- partial run を結論に使ってしまう
- 比較条件が毎回ずれる

という問題が起きやすいです。

### 0.2 run 前に確認するもの

新しい topic でも既存 topic でも、まず次を確認します。

- topic の `README.md`
  - 標準コマンド、出力先、carry-over 方針を確認する。
- 直近の experiment note
  - 既知の failure pattern、まだ言えないこと、再実行理由を確認する。
- final JSON または JSONL schema
  - 後段の集計や report 生成が何を前提にしているかを見る。
- `git status --short`
  - 生成物とコード変更を混ぜないため、作業ツリーの状態を先に確認する。

`Interpretation:`
現状把握の目的は「前回どこまで分かっていて、今回どこから再開するか」を固定することです。
コードを書く前に context を揃えるほど、後から note が書きやすくなります。

### 0.3 入口の決め方

最初に、今回の run がどの種類かを決めます。

- 通常の実験
  - `StandardWorker`
  - `StandardScheduler` または `StandardFullResourceScheduler`
  - `StandardRunner`
- host 側で pid / slot / stdout completion を直接見たい run
  - `build_worker_slots()`
  - `run_cases_with_subprocess_scheduler()`
  - 必要なら `RuntimeMonitor`

基準は次です。

- worker を case ごとの fresh process として安全に回したい
  - `StandardRunner` 系を先に検討する。
- host で child process の状態を強く観測したい
  - `subprocess_scheduler` を使う。

1 case だけの debug run は許容されますが、正式な比較 evidence は、
README と note に書いた protocol に従う fresh run だけに限定します。

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

### 2.4 実験 script 側で mini-runner を足さない

`experiment_runner` を使うなら、実験 script 側で次を重複実装しません。

- 独自の `Popen` 管理
- GPU slot の帳簿
- `CUDA_VISIBLE_DEVICES` の手組み
- partial run を前提にした resume protocol

実験 script 側が持つべきなのは、問い、case、metric、resource estimate、final JSON への集計です。

### 2.5 spot run を正式結果にしない

1 case だけの単発実行や、その場しのぎの subset 実行は debug / smoke には使ってよいです。
ただし、それを benchmark の結論や carry-over の正本に使いません。

- 正式結果は、README と note に書いた protocol に従う run だけ
- partial run は診断材料
- final JSON は完走 run から作る

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

### 8.4 実験の標準的な進め方

`experiment_runner` を使う実験は、次の順番で進めるのを基本にします。

1. 現状把握
   - topic README、既存 note、final JSON を見て `Question:` と `Comparison Target:` を固定する。
1. smoke
   - 小さい CPU run か極小ケースで、import、pickle、JSONL 追記、集計導線だけを確認する。
1. verified
   - 本番に近い backend と env で、worker 数を絞った narrow run を通す。
1. formal run
   - case range、timeout、allocator 方針、出力先を固定した fresh run を 1 回で流す。
1. note 化
   - final JSON、raw JSONL、主要 figure、failure kind を note から辿れるようにする。

`How to read this workflow:`
smoke は「動くか」を見る段階であり、formal run は「比較できるか」を見る段階です。
この 2 つを混ぜないことが、再現性と review のしやすさに直結します。

### 8.5 どの段階で止めるか

実験は、目的ごとに止めどころを変えてよいです。

- runner 配線確認
  - smoke が通れば十分です。
- env / GPU slot / allocator 確認
  - verified まで通してから判断します。
- benchmark / 比較表 / report
  - formal run と final JSON まで必要です。

partial run は診断には使えますが、

- 比較表の根拠
- carry-over の正本
- `main` に持ち帰る final JSON の代替

には使いません。

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
1. 新しい実験は、まずこの note の `0. 現状把握から始める` と `8.4 実験の標準的な進め方` に従って段階を切る。
1. 新しい実験 script はまずこの note の CPU / GPU 最小例を土台にする。

## 12. 関連 note

- [experiment_runner.md](/workspace/documents/experiment_runner.md)
- [experiment_runner_realtime_monitor.md](/workspace/notes/themes/experiment_runner_realtime_monitor.md)
- [experiment_runner_main_integration.md](/workspace/notes/themes/experiment_runner_main_integration.md)
