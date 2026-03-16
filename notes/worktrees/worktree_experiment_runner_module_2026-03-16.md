# Experiment Runner Modularization Worktree Extraction

## Context

- branch:
  - `work/experiment-runner-module-20260316`
- extracted_to_main_from:
  - `/workspace/.worktrees/work-experiment-runner-20260316`
- base:
  - `work/smolyak-tuning-20260316`
- scope:
  - 実験 runner の責務をモジュールへ切り出す
  - 積分器アルゴリズムそのものは、この branch の一次対象にしない

## Goal

- 実験 runner の共通部分を `smolyak_scaling` から分離する
- 以後の実験で
  - case 生成
  - worker 配布
  - fresh process 実行
  - JSON/JSONL 保存
  - 失敗分類
  - metadata 保存
  を再利用できるようにする

## Conventions

- `Idea:`
  - まだ実装していない設計案を書く
- `Decision:`
  - この branch で採用した方針を書く
- `Source:`
  - 既存コード・既存結果・文献など、根拠となる対象を書く
- `Interpretation:`
  - Source からの解釈を書く

## Non-Goals

- Smolyak 積分器の数値アルゴリズム変更
- 実験結果の可視化仕様変更
- `main` 側の大規模なディレクトリ再編

## First Split Candidates

- case definition
  - `dimension`, `level`, `dtype_name`, `case_id`
- worker definition
  - `worker_label`, `gpu_index`, `gpu_slot`, `cpu_affinity`
- execution policy
  - `workers_per_gpu`
  - `timeout_seconds`
  - `platform`
- result writer
  - JSONL append
  - final JSON save
  - metadata injection
- failure mapper
  - `oom`
  - `host_oom`
  - `timeout`
  - `worker_terminated`
  - generic `error`

## Design Direction

### Decision

- 最終配置は `python/jax_util/experiment_runner/` にする
- `smolyak_scaling` から先に使い始めて、runner 共通部として育てる
- 最初の目標は
  - `smolyak_scaling`
  - 将来の別実験 runner
  が同じ部品を使える状態にすること

### Idea

- 候補配置:
  - `experiments/_runner/`
  - または `experiments/common/runner/`
- まずは次の分離から始める
  - `config.py`
  - `workers.py`
  - `results.py`
  - `failures.py`

## Current Implementation

### Decision

- 最終的な切り出し先は `python/jax_util/experiment_runner/` に統一した
- host は
  - case queue
  - free worker slot
  - timeout
  - fallback failure record
  だけを管理する
- child は
  - `case + run_config + worker_slot`
  を受け取り、
  - 実験実行
  - JSONL 追記
  - stdout completion message
  を担当する

### Source

- 実装先:
  - `/workspace/.worktrees/work-experiment-runner-20260316/python/jax_util/experiment_runner/subprocess_scheduler.py`
  - `/workspace/.worktrees/work-experiment-runner-20260316/python/jax_util/experiment_runner/__init__.py`
  - `/workspace/.worktrees/work-experiment-runner-20260316/experiments/functional/smolyak_scaling/run_smolyak_scaling.py`

### Interpretation

- static な round-robin 割当より、host が free slot を見て都度 dispatch する方が、GPU の空き管理という責務に素直
- child が completion を明示送信することで、`Popen.wait()` だけに依存せず、異常終了と正常終了を切り分けやすくなる
- JSONL の owner を child に寄せることで、成功ケースの永続化は child の責務に閉じられる

## Validation

### Source

- `pytest -q python/tests/experiment_runner/test_subprocess_scheduler.py -s`
- `/bin/python3 python/tests/experiment_runner/test_subprocess_scheduler.py`
- `pyright python/jax_util/experiment_runner python/tests/experiment_runner experiments/functional/smolyak_scaling/run_smolyak_scaling.py`
- 出力:
  - `selected_gpu_indices: [0, 1]`
  - `worker_labels: ['gpu-0-w0', 'gpu-1-w0']`
  - `work_seconds: [2.50..., 2.50...]`

### Interpretation

- 2 GPU に対して host scheduler が別 slot へ 1 ケースずつ dispatch できている
- child 側の GPU 負荷は数秒持続するため、`nvidia-smi` でも観測しやすい
- `gpu_device_count == 1`, `visible_gpu_ids == [0]` を child が返しているので、各 child は `CUDA_VISIBLE_DEVICES` により 1 GPU だけを見ている
- 直接実行でも test file 自身が `sys.path` を補うので、`PYTHONPATH` なしで runner test を起動できる
- `pyright` では対象コードに型エラーが残っていない

## Current Constraints

### Source

- 現行 runner:
  - `/workspace/.worktrees/work-experiment-runner-20260316/experiments/functional/smolyak_scaling/run_smolyak_scaling.py`
- 現行 tuning note:
  - `/workspace/.worktrees/work-experiment-runner-20260316/notes/experiments/smolyak_tuning_20260316.md`

### Interpretation

- 現在の `run_smolyak_scaling.py` は
  - runner 共通ロジック
  - Smolyak 固有ロジック
  - benchmark 条件
  が 1 ファイルに混ざっている
- そのため
  - multiprocessing だけを直したい
  - JSONL の書き方だけ変えたい
  - 別の積分器へ横展開したい
  ときに修正範囲が広い

## Immediate Next Steps

1. `run_smolyak_scaling.py` の責務を section 単位で棚卸しする
2. runner 共通ロジックと Smolyak 固有ロジックの境界を決める
3. 内部モジュールの最小単位を切る
4. 既存 CLI を壊さない形で import 経由へ置き換える

## Addendum

### Quick Reference

- branch:
  - `work/experiment-runner-module-20260316`
- final landing place:
  - `python/jax_util/experiment_runner/`
- main changes:
  - host scheduler / child execution の分離
  - JSONL 逐次保存
  - completion record を child が明示送信する runner 方式
  - multi-GPU child probe test の追加
- validation:
  - GPU child probe
  - direct execution
  - `pyright`
- final status:
  - 成果は `main` と `results/functional-smolyak-scaling-tuned` に統合済み
