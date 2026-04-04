# Smolyak スケーリング実験

日付: 2026-03-20\
関連 worktree: `work/smolyak-improvement-20260318`

## Purpose

Smolyak 積分器の次元、level、dtype に対するスケーリングを系統的に観測する。
特に、初期化時間、積分時間、誤差、failure pattern を分離して記録する。

## Current Implementation

- 実験コード本体: [experiments/functional/smolyak_scaling](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling)
- 汎用実行基盤: [python/experiment_runner](/workspace/python/experiment_runner)
- 積分器本体: [smolyak.py](/workspace/python/jax_util/functional/smolyak.py)

## Size Presets

- `smoke`: 最小構成の配線確認
- `small`: CPU 上の短時間確認
- `verified`: 少し広げた CPU 確認
- `medium`: GPU を含む中規模 run
- `large`: 長時間の本実行に近い構成

実際の引数と既定値は [run_smolyak_scaling.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/run_smolyak_scaling.py) を参照する。

## Output Shape

- per-case の raw record は JSONL に逐次保存する。
- 集約結果は `final_results_<timestamp>.json` に保存する。
- 各 case は `case_params` と `smolyak` の nested schema を持つ。
- `smolyak` には status、init time、integrate time、誤差、error text などを残す。

## Branch Strategy

- smoke / verified の短時間 run は作業中 worktree で確認してよい。
- medium / large の長時間 run は `results/*` branch の専用 worktree へ分ける。
- raw JSONL や trace は results branch に残し、`main` には note と必要最小限の final JSON を持ち帰る。

## Consideration

- この experiment は benchmark ではなく、多条件 sweep と failure analysis を含む experiment として扱う。
- benchmark 的な前後比較だけをしたい場合は、同じ topic 配下に短時間 benchmark を別途置く方が読みやすい。
