# `experiment_runner` 設計方針

この文書は、`python/jax_util/experiment_runner/` の最小設計を整理します。
現在は [protocols.py](/workspace/.worktrees/work-experiment-runner-generalization-20260317/python/jax_util/experiment_runner/protocols.py) を正として、[runner.py](/workspace/.worktrees/work-experiment-runner-generalization-20260317/python/jax_util/experiment_runner/runner.py) の標準実装をそれに揃えます。

## 1. 基本抽象

`experiment_runner` では、`ResourceEstimate`、`ResourceCapacity`、`Worker[T, U]`、
`Scheduler[T]`、`Runner[T, U]` を基本抽象とします。

### 1.1 `ResourceEstimate`

- `ResourceEstimate` は各ケースを起動したときの資源見積もりです。
- 典型例は必要 GPU 数、推定メモリ量、優先度などです。
- 最小 protocol では runner へ直接見せず、concrete scheduler の内部表現として使います。

### 1.2 `ResourceCapacity`

- `ResourceCapacity` は全体として使える資源を表します。
- 現在の最小要件は `max_workers` です。

### 1.3 `Worker[T, U]`

- `Worker` は単一の `task` を持ちます。
- protocol 上の実行入口は `__call__(case, context) -> int` です。
- `task` は `Callable[[T, TaskContext], U]` です。
- 複数の実験関数を回したい場合は、`task` 側で `context` を見て分岐します。

### 1.4 `Scheduler[T]`

- `Scheduler` は `resource_capacity` を持ちます。
- `next_case()` は次に流すべき `case` と `TaskContext` を返します。
- `on_finish(...)` はケース完了を受けて scheduler 内部状態を更新します。
- case 順序最適化、resource 見積もり、task 切り替え用の `context` 管理は scheduler の責務です。

### 1.5 `Runner[T, U]`

- `Runner` は `Scheduler` に従って `Worker` を起動します。
- 標準実装は `ProcessPoolExecutor` により worker を別プロセスで実行します。
- runner 自身は実行機構だけを持ち、順序決定や task 切り替えは持ちません。

## 2. 標準実装

- [runner.py](/workspace/.worktrees/work-experiment-runner-generalization-20260317/python/jax_util/experiment_runner/runner.py) には `StandardWorker`、`StandardResourceCapacity`、`StandardCompletion`、`StandardScheduler`、`StandardRunner` を置きます。
- `StandardWorker` は `task(case, context)` を呼び、成功時は `0`、例外時は `1` を返します。
- `StandardScheduler` は FIFO で case を返す最小 scheduler です。
- `StandardScheduler` は optional な `context_builder` を受け取り、`TaskContext` の組み立てを標準機能として持ちます。
- `StandardScheduler` は `on_finish(...)` で `StandardCompletion` を記録します。
- `StandardRunner` は `max_workers` 本の child process を使ってケースを並列実行します。

## 3. task 切り替えのルール

- worker は単一の `task` だけを受け取ります。
- 異なる実験関数を回したいときは、`Scheduler.next_case()` が返す `TaskContext` に切り替えキーを入れます。
- `task` 側は `context["task_key"]` などを見て分岐します。
- これにより、runner と worker の構造を増やさずに複数系統の実験を扱えます。

## 4. プロセス実行の前提

- `StandardRunner` は worker を別プロセスで実行します。
- したがって、worker、task、case、context はプロセス間で受け渡せる形であることを前提にします。
- top-level class / function と pickle 可能な dataclass を使うのが基本です。

## 5. 今後の拡張

- resource-aware な順序最適化は scheduler 側へ追加します。
- GPU 固有差分が必要になっても、runner を増やすのではなく scheduler と resource 表現で吸収する方針です。
- [gpu_runner.py](/workspace/.worktrees/work-experiment-runner-generalization-20260317/python/jax_util/experiment_runner/gpu_runner.py) には、環境から GPU 一覧を読み取り、1 プロセス 1 GPU を仮定する `GPUResourceCapacity` と `StandardGPUScheduler` を置きます。
- `ResourceEstimate` を本格利用する scheduler は別ファイルで追加してよいです。
