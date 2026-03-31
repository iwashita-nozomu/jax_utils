# Experiment Runner Real-Time Monitor

この note は、`experiment_runner` に載せる軽量リアルタイム資源モニタの設計を固めるための theme note です。

対象は単一ホスト上の experiment run です。

目的は、実行中の CPU / host memory / GPU utilization / GPU memory / worker 状態を、軽い GUI と機械可読 API の両方で見えるようにすることです。

## Known

Source: NVIDIA の `nvidia-smi` 公式ドキュメントでは、`--query-gpu=...` と `--format=csv,noheader,nounits` により、スクリプト向けの GPU 情報取得が明示されています。

Source: 同じ文書では `--loop` と `--loop-ms` も提供されていますが、`nvidia-smi` 自体は表示ツールでもあるため、monitor 実装では human-readable table を直接 parse するより、query mode を短周期で呼ぶ方が安全です。

Source: 同文書では GPU 指定に index, UUID, PCI bus ID が使えますが、再起動間で順序が安定しないため UUID か PCI bus ID を推奨しています。

Source: `psutil` 公式ドキュメントでは、`cpu_percent(interval=None)` の最初の呼び出しは意味のない `0.0` を返すため無視すべきであり、複数の process 情報取得には `oneshot()` が有効です。

Interpretation: この repo はすでに [resource_scheduler.py](/workspace/python/experiment_runner/resource_scheduler.py) で `nvidia-smi` を使って GPU 検出を行っているため、v1 monitor の GPU backend も同じ依存で揃えるのが自然です。

Interpretation: [subprocess_scheduler.py](/workspace/python/experiment_runner/subprocess_scheduler.py) 系は host が `Popen` を直接持つため pid 追跡が簡単です。

Interpretation: [runner.py](/workspace/python/experiment_runner/runner.py) の `StandardRunner` は case ごとに fresh child process を起動するため、将来的には pid registration path を追加しやすいです。ただし v1 monitor では `subprocess_scheduler.py` 側を先に基準挙動にする方が軽いです。

## Design

### Core Shape

Worked: v1 は `experiment_runner` 本体に軽量 monitor service を内蔵し、重い外部監視基盤を必須にしません。

Worked: monitor は host 側の thread 1 本で動かします。

Worked: monitor は「collector」「state store」「HTTP surface」の 3 層だけに分けます。

- collector
  - host `/proc`
  - `nvidia-smi --query-gpu=...`
  - runner / scheduler の内部状態
- state store
  - 最新 snapshot 1 件
  - 固定長 ring buffer
  - event log
- HTTP surface
  - GUI
  - JSON API
  - health endpoint

### Lightweight Rules

Worked: v1 では `Prometheus + Grafana` や WebSocket を既定構成にしません。

Worked: GUI は軽量な polling ベースにします。

Worked: サーバは `127.0.0.1` bind を既定にします。

Worked: 永続化は append-only JSONL を optional に留めます。

Did Not Work: `nvidia-smi` の通常 table 出力や `watch nvidia-smi` を直接 GUI backend にする案は、parse が脆く、CPU / runner 状態と統合しにくいため採りません。

### Data Sources

Worked: GPU metrics は `nvidia-smi --query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits` を基本にします。

Worked: host metrics は Linux 前提で `/proc/stat`, `/proc/meminfo`, `/proc/<pid>/stat`, `/proc/<pid>/status` を読む方針にします。

Coding Pattern: CPU 使用率は前回 snapshot との差分で計算します。

Pitfall: `psutil` を使う場合でも最初の `cpu_percent()` は無視が必要です。

### API Surface

Worked: GUI と API は同一ポートの単一サーバから出します。

Worked: API は v1 では JSON over HTTP を基本にします。

Likely: もし non-HTTP API も欲しくなったら、次の追加は Unix domain socket か monitor JSONL tail のどちらかで十分です。

提案 endpoint:

- `GET /`
  - 軽量 dashboard HTML
- `GET /api/v1/snapshot`
  - 最新 snapshot
- `GET /api/v1/history?limit=60`
  - 直近 history
- `GET /api/v1/events`
  - case start / finish / timeout / worker_terminated
- `GET /healthz`
  - monitor thread の生存確認

### Snapshot Schema

Worked: snapshot は最低限、以下の形に揃えます。

```json
{
  "timestamp": "2026-03-30T12:34:56.789Z",
  "runner": {
    "pending_cases": 120,
    "running_cases": 4,
    "completed_cases": 38,
    "max_workers": 4
  },
  "host": {
    "cpu_percent": 83.2,
    "memory_total_bytes": 540431955968,
    "memory_available_bytes": 201304780800,
    "memory_used_bytes": 339127175168
  },
  "gpus": [
    {
      "gpu_id": 0,
      "uuid": "GPU-...",
      "utilization_gpu_percent": 96,
      "utilization_memory_percent": 71,
      "memory_total_bytes": 25769803776,
      "memory_used_bytes": 18374686464,
      "temperature_c": 68,
      "power_watts": 247.5
    }
  ],
  "workers": [
    {
      "case_id": 42,
      "worker_label": "gpu-0-w0",
      "pid": 12345,
      "state": "running",
      "gpu_ids": [0],
      "started_at": "2026-03-30T12:33:12.123Z",
      "elapsed_seconds": 104.7
    }
  ]
}
```

### Integration With Current Runner Types

Worked: `subprocess_scheduler.py` 系では、`Popen` 直後に pid, worker_label, gpu_index, case_id を monitor registry へ登録できます。

Worked: `on_case_started` と `on_case_finished` はそのまま event log へ流せます。

Open: `StandardRunner` 側で pid registration を正式に持たせるかはまだ決めていません。

Likely: `StandardRunner` v1 では scheduler の `pending/running/completed` と GPU 割当を主表示にし、pid 単位表示は後回しにするのが軽いです。

Idea: `StandardRunner` でも pid 単位表示が必要なら、child が起動直後に `register_worker(case_id, pid, gpu_ids, started_at)` を host へ送る軽量 registration path を追加します。

### Sampling Policy

Worked: 既定 interval は 1.0 秒にします。

Worked: 0.5 秒より細かい polling は既定では許しません。

Pitfall: `nvidia-smi --loop-ms` をそのまま常駐させるとプロセス管理と parse が複雑になるため、monitor 側で 1 回ずつ query する方が扱いやすいです。

### Implementation Order

Worked: 実装順は次で固定するのが安全です。

1. subprocess scheduler で使える monitor core を先に作る
1. 軽量 HTML + `/api/v1/*` を乗せる
1. JSONL sink を付ける
1. `StandardRunner` の registration path を必要なら追加する

## Extension Ideas

Idea 1: `--monitor` / `--monitor-port` / `--monitor-bind` を実験 CLI 共通オプションにして、script 側の起動コードを定型化する。

Idea 2: Docker 実行時に `0.0.0.0` bind と公開 URL を自動表示し、`docker run -p` の例を startup banner に出す。

Idea 3: monitor snapshot を `monitor_<run_id>.jsonl` へ定期保存し、run 後にタイムライン解析へ再利用できるようにする。

Idea 4: worker 一覧に `stdout_tail` / `stderr_tail` の短い ring buffer を持たせ、異常時の一次切り分けをブラウザ上で済ませる。

Idea 5: GPU ごとの utilization と memory_used の短期 sparkline を HTML 側に追加し、チャートライブラリなしでも負荷の波形を見えるようにする。

Idea 6: timeout や `worker_terminated` が続いたときに UI 上で warning banner を出し、event 数の閾値超過を強調する。

Idea 7: `StandardRunner` 用の child registration path を追加し、fresh child process と case_id の対応を monitor へ送れるようにする。

Idea 8: `/api/v1/snapshot` に scheduler 内部の資源残量を加え、host memory、GPU memory、slot 数の残りを直接見えるようにする。

Idea 9: `/api/v1/history` に単純な downsampling を入れて、長時間 run でもブラウザ負荷を増やさずに長い履歴を返せるようにする。

Idea 10: JSON API に加えて Unix domain socket の軽量ローカル API を optional で足し、同一ホスト内の補助ツールからポート公開なしで読めるようにする。

## Open

Open: `StandardRunner` 側で pid registration を入れるなら、worker protocol を増やすか、context 経由の side channel を使うかを決める必要があります。

Open: host metrics を `/proc` のみで押し切るか、将来 `psutil` を許容するかは、実装量と保守性を見て判断してよいです。

Open: GUI の描画は純粋な HTML + 小さな inline JS で十分か、最小限の chart helper を持つかを決める必要があります。

Open: Docker コンテナ内で monitor を起動するときは、既定 bind を `127.0.0.1` のままにするか、`--monitor-bind 0.0.0.0` を推奨するかを CLI と運用文書で揃える必要があります。

Open: GPU metrics は現在 `nvidia-smi` 依存で十分ですが、将来 MIG や process attribution まで見たくなったときに NVML へ切り替えるかどうかは判断が必要です。

Open: 長時間 run の履歴をその場で全部返すと JSON が肥大化するため、history retention と downsampling の既定値をどこで切るかを決める必要があります。

Open: event log が増えたときに、`case_started` と `case_finished` をそのまま全部返すか、warning / failure 系を別 endpoint に分けるかは API 整理の余地があります。

Open: resource scheduler と接続するなら、`snapshot` に scheduler 内部の available host memory、GPU memory、slot 残量をどの粒度で出すかを決める必要があります。

Open: monitor JSONL を保存する場合、実験結果 JSONL と同じディレクトリへ置くか、`monitor/` 下へ分離するかで run 後の整理しやすさが変わります。

Open: ブラウザ UI で複数 run を切り替えたい要求が出た場合、単一 monitor per process のままいくか、run registry を持つ小さな multiplexer を作るかの判断が必要です。

Open: alerting を入れるなら、UI banner だけで済ませるか、stderr 通知、JSON event、将来の webhook を見据えた構造にするかを先に決めた方が API がぶれません。

Likely: 短期では `subprocess_scheduler` の運用観測を安定させ、その後に `StandardRunner` へ寄せる順番がいちばん安全です。

Likely: 中期では monitor snapshot と final result JSON を並べて見られる補助 report を作ると、timeout や OOM の読み解きがかなり楽になります。

Likely: Docker / local / remote host のどこで起動しても同じ URL 構成で使えるようにしておくと、後から CLI と docs を広げやすいです。

Consideration: この monitor は observability 基盤そのものを作るのではなく、実験 runner のデバッグ時間を減らす道具として育てる方が scope を保ちやすいです。

Consideration: したがって次の拡張も、「長時間 run の異常切り分けが楽になるか」「GPU / host memory / worker 状態の関係が読みやすくなるか」を基準に優先順位を付けるのがよいです。

## After Carry-Over Work

この worktree の内容を `main` へ持ち帰った後は、次の作業を順に片付けると抜けが少ないです。

1. まず、monitor 差分と runner 整理差分を同じ commit にするか分けるかを決めます。

1. `WORKTREE_SCOPE.md` とこの theme note を `main` から辿れるようにし、どの worktree で設計したかが追える状態にします。

1. `python/experiment_runner/monitor.py` の public API を確定し、`run` 用と `daemon` 用のどちらを v1 の正式入口にするかを決めます。

1. `subprocess_scheduler.py` 側の monitor 統合を基準挙動として固定し、`StandardRunner` 側は未実装なら未実装と分かる形で docs に残します。

1. `documents/experiment_runner.md` と `notes/experiments/experiment_runner_usage.md` を、実装済みの endpoint、mode、interval 変更 API に合わせて再確認します。

1. 実験 script 側では monitor を有効化する CLI option をどう出すかを決めます。

1. `smolyak_scaling` と今後の run script のどちらから先に monitor 起動オプションを載せるかを決め、片方にまず絞って導入します。

1. local 実行で `GET /`, `GET /api/v1/snapshot`, `GET /api/v1/history`, `GET /api/v1/events`, `GET /healthz` が素直に見えることを smoke test します。

1. Docker 実行では `--monitor-bind 0.0.0.0` と `-p host:container` の組み合わせを確認し、ブラウザから見える手順を短い例で残します。

1. `daemon` mode の既定値として、`sample_interval_seconds`, `history_limit`, `event_limit` をどこまで軽くするかを一度固定します。

1. 取得間隔を動的変更できる API を本当に外へ見せるか、当面は Python API のみに留めるかを決めます。

1. monitor snapshot を JSONL に保存する場合は、保存先、ファイル名規約、run 終了後の整理手順を決めます。

1. `StandardRunner` 用 registration path を後続タスクにする場合は、必要な worker protocol 変更を task として切り出します。

1. 複数 run、Docker 常駐 monitor、将来の Bash launcher は scope 外アイデアとして残し、v1 に入れないものを明記します。

1. 最後に、`pytest -q python/tests/experiment_runner`, `pyright python/experiment_runner`, `git diff --check` を `main` 上でも再実行して、持ち帰り時の崩れがないことを確認します。

### Priority Cut

持ち帰り直後は、次の優先度で進めると判断がぶれにくいです。

P0:

- commit を分けるかどうかを最初に決め、monitor 差分と runner 整理差分の境界を固定します。
- `WORKTREE_SCOPE.md` とこの note を `main` 側から辿れる状態にし、設計経緯を追えるようにします。
- `monitor.py` の public API を確定し、`run` 用と `daemon` 用のどちらを v1 の正式入口にするかを決めます。
- `subprocess_scheduler.py` の monitor 統合を v1 基準挙動として固定し、`StandardRunner` 側の未実装点は docs に明記します。
- docs を実装済み endpoint と mode に合わせて揃え、`main` 上で `pytest`, `pyright`, `git diff --check` を再実行します。

P1:

- 実験 script 側で monitor を有効化する CLI option の出し方を決めます。
- `smolyak_scaling` と今後の run script のどちらから先に monitor 起動を載せるかを決め、片方に先行導入します。
- local の endpoint smoke test と、Docker の `--monitor-bind` / `-p` 動作確認を短い手順にまとめます。
- `daemon` mode の既定値として `sample_interval_seconds`, `history_limit`, `event_limit` を固定します。
- interval 変更 API を Python 内部 API に留めるか、外部に見せるかを決めます。
- monitor JSONL を保存する場合の保存先と命名規約を決めます。

P2:

- `StandardRunner` 用 registration path を後続 task に切り出し、必要な worker protocol 変更を整理します。
- 複数 run の切り替え、Docker 常駐 monitor、将来の Bash launcher は v1 から切り離した拡張案として管理します。
- 長時間 run 向けの history downsampling や alerting など、便利化の枝を別タスクで育てます。

## References

- NVIDIA, `nvidia-smi` documentation:
  - https://docs.nvidia.com/deploy/nvidia-smi/index.html
- psutil API reference:
  - https://psutil.readthedocs.io/en/latest/api.html
- Local implementation references:
  - [resource_scheduler.py](/workspace/python/experiment_runner/resource_scheduler.py)
  - [runner.py](/workspace/python/experiment_runner/runner.py)
  - [subprocess_scheduler.py](/workspace/python/experiment_runner/subprocess_scheduler.py)
  - [experiment_runner_main_integration.md](/workspace/notes/themes/experiment_runner_main_integration.md)
