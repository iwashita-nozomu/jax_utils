# Smolyak Scaling Tuned Results Worktree Extraction

## Context

- branch:
  - `results/functional-smolyak-scaling-tuned`
- extracted_to_main_from:
  - `/workspace/.worktrees/results-functional-smolyak-scaling-tuned`
- branch_head:
  - `c6011615e8f13fc664c19fd1ad46cf2199318eed`
- scope:
  - tuned Smolyak scaling の継続実験
  - partial result と timing report の整理
  - Smolyak HLO 実験 README の保持

## Merged To Main

- [tuned_smolyak_partial_results_20260318.md](/workspace/notes/experiments/tuned_smolyak_partial_results_20260318.md)
- [tuned_smolyak_timing_timeout_report_20260318.md](/workspace/notes/experiments/tuned_smolyak_timing_timeout_report_20260318.md)
- [README.md](/workspace/experiments/functional/smolyak_hlo/README.md)
- [results](/workspace/experiments/functional/smolyak_hlo/results)
- [results](/workspace/experiments/functional/smolyak_scaling/results)
- [run_smolyak_hlo_case.py](/workspace/experiments/functional/smolyak_hlo/run_smolyak_hlo_case.py)
- [run_smolyak_scaling.py](/workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py)
- [render_smolyak_scaling_report.py](/workspace/experiments/functional/smolyak_scaling/render_smolyak_scaling_report.py)

## Results Branch Retention

- raw JSON / JSONL / log / rendered report asset は `main` にも `results/functional-smolyak-scaling-tuned` branch にも残る
- worktree を削除しても、commit 済みの tracked file は `main` と `origin/results/functional-smolyak-scaling-tuned` の両方から再取得できる

## Decision

- report Markdown は summary にせず、そのまま `main` に保持する
- raw 結果と実験スクリプトも `main` に取り込み、results branch 側にも保持する
- `python/jax_util/functional/smolyak.py` は merge conflict を解消し、実験 branch の plan 実装と `main` の helper API を統合した
- 上記を確認したうえで、`/workspace/.worktrees/results-functional-smolyak-scaling-tuned` は削除してよい
