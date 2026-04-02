# `experiment_runner` 設計方針

この文書は、`python/experiment_runner/` の最小設計を整理します。
現在の正本は [protocols.py](/workspace/python/experiment_runner/protocols.py)、
[runner.py](/workspace/python/experiment_runner/runner.py)、
[resource_scheduler.py](/workspace/python/experiment_runner/resource_scheduler.py)、
[monitor.py](/workspace/python/experiment_runner/monitor.py)、
[subprocess_scheduler.py](/workspace/python/experiment_runner/subprocess_scheduler.py) です。

## 1. 基本抽象

`experiment_runner` では、`ResourceEstimate`、`ResourceCapacity`、`Worker[T, U]`、`Scheduler[T]`、`Runner[T, U]` を基本抽象とします。

### 1.1 `ResourceEstimate`

- `ResourceEstimate` は各ケースを起動したときの資源見積もりです。
- 典型例は必要 GPU 数、推定メモリ量、優先度などです。
- 最小 protocol では runner へ直接見せず、concrete scheduler の内部表現として使います。
- 見積もりロジック自体は framework 側へ押し込まず、実際の task を定義する実験コードと同じ場所で実装する方針です。

### 1.2 `ResourceCapacity`

- `ResourceCapacity` は全体として使える資源を表します。
- 現在の最小要件は `max_workers` です。

### 1.3 `Worker[T, U]`

- `Worker` は単一の `task` を持ちます。
- protocol 上の実行入口は `__call__(case, context) -> int` です。
- `task` は `Callable[[T, TaskContext], U]` です。
- 複数の実験関数を回したい場合は、`task` 側で `context` を見て分岐します。
- resource-aware な実験では、worker が `resource_estimate(case)` を追加で持ってよく、task と見積もりを同じ実装単位に寄せるのを推奨します。

### 1.4 `Scheduler[T]`

- `Scheduler` は `resource_capacity` を持ちます。
- `next_case()` は次に流すべき `case` と `TaskContext` を返します。
- `on_finish(...)` はケース完了を受けて scheduler 内部状態を更新します。
- case 順序最適化、resource 見積もり、task 切り替え用の `context` 管理は scheduler の責務です。

### 1.5 `Runner[T, U]`

- `Runner` は `Scheduler` に従って `Worker` を起動します。
- 標準実装は spawn child process により case ごとに fresh worker process を起動します。
- runner 自身は実行機構だけを持ち、順序決定や task 切り替えは持ちません。

## 1.6 責務境界

`experiment_runner` を使うときは、次の境界を崩さない方針にします。

runner / scheduler 側の責務:

- fresh child process の起動と終了
- case queue の進行
- resource estimate に基づく slot / GPU 割当
- child に渡す `TaskContext["environment_variables"]` の構築
- GPU 可視性や allocator 系 env の反映
- worker event や runtime state の観測面

experiment code 側の責務:

- 研究の問い
- case 定義
- difficulty range
- resource estimate の意味付け
- metric 計算
- final JSON への集計
- result interpretation と note 化

したがって、experiment script 側で独自の mini-runner、独自の GPU slot 管理、独自の env wiring を重ねるのは避けます。

## 2. 標準実装

- [runner.py](/workspace/python/experiment_runner/runner.py) には `StandardWorker`、`StandardResourceCapacity`、`StandardCompletion`、`StandardScheduler`、`StandardRunner` を置きます。
- `StandardWorker` は `task(case, context)` を呼び、成功時は `0`、例外時は `1` を返します。
- `StandardScheduler` は FIFO で case を返す最小 scheduler です。
- `StandardScheduler` は optional な `context_builder` を受け取り、`TaskContext` の組み立てを標準機能として持ちます。
- `StandardScheduler` は `on_finish(...)` で `StandardCompletion` を記録します。
- `StandardRunner` は `max_workers` 本までの fresh child process を case 単位で起動し、ケース完了ごとに process を終了させます。

## 3. task 切り替えのルール

- worker は単一の `task` だけを受け取ります。
- 異なる実験関数を回したいときは、`Scheduler.next_case()` が返す `TaskContext` に切り替えキーを入れます。
- `task` 側は `context["task_key"]` などを見て分岐します。
- これにより、runner と worker の構造を増やさずに複数系統の実験を扱えます。

## 4. プロセス実行の前提

- `StandardRunner` は worker を case ごとの別プロセスで実行します。
- したがって、worker、task、case、context はプロセス間で受け渡せる形であることを前提にします。
- top-level class / function と pickle 可能な dataclass を使うのが基本です。

## 5. 今後の拡張

- resource-aware な順序最適化は scheduler 側へ追加します。
- GPU 固有差分は専用 runner を増やさず、`resource_scheduler.py` 内の scheduler と resource 表現で吸収する方針です。
- [resource_scheduler.py](/workspace/python/experiment_runner/resource_scheduler.py) には、1 task = 1 process を前提に host memory と GPU ごとの slot / memory を同時に見る `FullResourceCapacity`、`FullResourceEstimate`、`StandardFullResourceScheduler` を置きます。
- [monitor.py](/workspace/python/experiment_runner/monitor.py) には、軽量な runtime snapshot 保持、worker event 記録、`GET /` と `/api/v1/*` を返す小さな HTTP surface を置きます。
- [subprocess_scheduler.py](/workspace/python/experiment_runner/subprocess_scheduler.py) には、host が worker slot と child process を直接管理したい benchmark 向けに、`WorkerSlot`、`build_worker_slots()`、`run_cases_with_subprocess_scheduler()` を置きます。
- `StandardFullResourceScheduler` に渡す `estimate_builder` は、task 実装に隣接した関数、または `task.resource_estimate(case)` のような bound method として定義するのを基本にします。
- `StandardFullResourceScheduler.from_worker(...)` を使うと、worker が持つ `resource_estimate(case)` をそのまま scheduler へ渡せます。
- `FullResourceCapacity.from_system(...)` は、`max_workers` を CPU 数、`host_memory_bytes` を物理メモリ量、`gpu_devices` を可視 GPU とそのメモリ容量から自動検出する入口です。
- GPU を割り当てる scheduler は `CUDA_VISIBLE_DEVICES` と `NVIDIA_VISIBLE_DEVICES` を `TaskContext["environment_variables"]` にまとめて載せ、worker 側で [context_utils.py](/workspace/python/experiment_runner/context_utils.py) の `apply_environment_variables()` を呼んで反映します。
- JAX / XLA の標準 env は [xla_env.py](/workspace/python/jax_util/xla_env.py) を正本とし、`experiment_runner` はその env dict を child へ運ぶ役割に留めます。
- 既定では JAX allocator 系の環境変数は足しません。`StandardFullResourceScheduler` は「GPU 可視性の制御」だけを基本責務にして、allocator / preallocation の調整は明示的な opt-in に寄せます。
- JAX の GPU メモリ挙動を調整したい実験では、`jax_util.xla_env.build_gpu_env(...)` または `GPUEnvironmentConfig(...)` を使って `XLA_PYTHON_CLIENT_PREALLOCATE`、`XLA_PYTHON_CLIENT_MEM_FRACTION`、`XLA_PYTHON_CLIENT_ALLOCATOR`、`TF_GPU_ALLOCATOR` などを `environment_variables` 経由で worker へ渡します。
- 実験 script 側では `CUDA_VISIBLE_DEVICES`、`JAX_PLATFORMS`、`XLA_*` などの runtime env を直接組み立てず、`jax_util.xla_env` で env dict を作って runner / scheduler へ渡します。
- `StandardRunner` は case ごとに fresh child process を使うため、`CUDA_VISIBLE_DEVICES` や `JAX_PLATFORMS` のような import-sensitive な環境変数が前ケースの JAX state に汚染されにくいです。
- 既存の multi-GPU 実験で host が pid を直接管理したい場合は、引き続き `subprocess_scheduler.py` を併用できます。
- 現在は `python/experiment_runner/` の standalone module として置いています。
- 今後さらに別リポジトリへ分離する場合も、`Worker` / `Scheduler` / `Runner` / `TaskContext` の境界は保ち、experiment 側のコードから見える契約を先に安定化します。

## 5.1 runner を使うならやらなくてよいこと

- `run_*.py` の中で `Popen` を直接並べること
- GPU ごとの空き管理を script 側で持つこと
- `CUDA_VISIBLE_DEVICES` を script 側で case ごとに差し替えること
- case ごとの fresh process を自前で管理すること
- env を script 本体の if 文に埋め込むこと

これらは runner / scheduler 層へ寄せる前提です。

## 5.2 Spot Run を正規運用にしない

`experiment_runner` は case ごとの実行を安定させるためのものであり、ad hoc な `spot run` を正式な比較手段にするための仕組みではありません。

- 単発 case の debug run は許容します。
- ただし、その結果を正式な benchmark evidence や carry-over の正本にしません。
- 正式な比較は、README や note に書いた protocol に従う 1 回の run と final JSON を単位に扱います。

## 6. リアルタイム資源モニタ設計

### 6.1 目的

- 実験実行中に CPU / host memory / GPU utilization / GPU memory / worker 状態をリアルタイムに確認できるようにする。
- GUI と機械可読 API を同時に提供する。
- 既定構成は軽量に保ち、重い外部監視基盤を必須にしない。

### 6.2 既定方針

- 既定実装は `python/experiment_runner/` 内に軽量 monitor を持つ。
- GUI はローカルポートに bind する簡易 HTTP サーバで提供する。
- API は同じサーバから JSON を返す。
- 既定 bind は `127.0.0.1` とし、外部公開は前提にしない。
- `Prometheus` / `Grafana` 連携は将来の optional 機能に留め、v1 の必須要件にはしない。

### 6.3 データ取得

- GPU 情報は既存方針と合わせて `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` を基本 backend にする。
- host 側の CPU / memory / pid 情報は Linux 前提で `/proc` 読み取りを基本にし、v1 では Python 依存を増やさない。
- `nvidia-smi` は表示ツールとして常駐させるのではなく、monitor thread が一定間隔で短命プロセスとして呼ぶ collector として使う。
- 推奨サンプリング間隔は 1.0 秒、必要時のみ 0.5 秒まで下げる。

### 6.4 公開インタフェース

- `GET /`
  - 軽量な HTML ダッシュボードを返す。
- `GET /api/v1/snapshot`
  - 最新 1 件の監視スナップショットを JSON で返す。
- `GET /api/v1/history?limit=N`
  - 直近 N 件の時系列を JSON で返す。
- `GET /api/v1/events`
  - case start / finish / timeout / worker_terminated を JSON 配列で返す。
- `GET /healthz`
  - monitor thread が稼働中かどうかだけを返す軽量 endpoint。

### 6.5 サンプル構造

- snapshot は少なくとも次を持つ。
  - `timestamp`
  - `runner`
    - `pending_cases`
    - `running_cases`
    - `completed_cases`
    - `max_workers`
  - `host`
    - `cpu_percent`
    - `memory_total_bytes`
    - `memory_available_bytes`
    - `memory_used_bytes`
  - `gpus`
    - `gpu_id`
    - `uuid`
    - `utilization_gpu_percent`
    - `utilization_memory_percent`
    - `memory_total_bytes`
    - `memory_used_bytes`
    - `temperature_c`
    - `power_watts`
  - `workers`
    - `case_id`
    - `worker_label`
    - `pid`
    - `state`
    - `gpu_ids`
    - `started_at`
    - `elapsed_seconds`

### 6.6 runner への接続

- `subprocess_scheduler.py` 系では host が `Popen` を直接持つため、worker 起動時に pid と slot 情報を monitor registry へ即時登録できる。
- `StandardRunner` 系では case ごとに fresh child process を起動するため、pid registration path を後から足すこと自体は難しくない。ただし v1 は `subprocess_scheduler.py` 側を monitor の基準挙動とし、`StandardRunner` 側は scheduler の pending/running/completed 数と GPU 割当情報を主表示にする。
- `StandardRunner` で pid 単位の監視が必要になった場合は、worker 起動直後に host へ `register_worker(case_id, pid, gpu_ids, started_at)` を送る軽量 registration path を追加する。

### 6.7 保持方式

- メモリ上には固定長 ring buffer だけを保持する。
- 永続化が必要な場合は `monitor_<run_id>.jsonl` へ append-only で書く。
- GUI は polling ベースで `snapshot` と `history` を取得し、WebSocket は v1 では使わない。

### 6.8 非目標

- v1 では認証付き公開 API、複数ノード集約、長期保存 DB、Grafana 前提の構成は含めない。
- v1 では MIG ごとの詳細 GPU process attribution までは保証しない。
