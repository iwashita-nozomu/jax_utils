# Experiment Runner Modularization 2026-03-16

## Context

- branch:
  - `work/experiment-runner-module-20260316`
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

- まずは `experiments` 配下の内部モジュールとして切り出す
- いきなり `python/jax_util/...` へは入れない
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
