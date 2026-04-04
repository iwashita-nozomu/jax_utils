# Smolyak Research Loop 085

Date: 2026-04-01

## Goal

gaussian high-level push to level 15. Push the integrator toward the 50D level-15 target on smooth isotropic baseline.

## Executed Calculations

- Family: `gaussian`
- DTypes: `float64`
- Dimensions: `20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50`
- Levels: `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`
- Executed dimensions: `50`
- Executed levels: `15`
- Requested modes: `auto,indexed,batched`
- Chunk sizes: `16384`
- Batch size: `64`
- Timeout per case: `180` seconds
- Matrix command: `python3 -m experiments.smolyak_experiment.run_smolyak_mode_matrix --platform gpu --dimensions 50 --levels 15 --dtypes float64 --families gaussian --requested-modes auto,indexed,batched --chunk-sizes 16384 --batch-size 64 --warm-repeats 1 --timeout-seconds 180 --workers-per-gpu 1 --quiet --output-dir /workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push`

## Primary Outputs

- Matrix run dir: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z`
- Matrix JSONL: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z/results.jsonl`
- Matrix summary JSON: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z/summary.json`
- Matrix Markdown report: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z/report.md`
- Raw CSV: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z/mode_matrix_raw_cases.csv`
- Frontier CSV: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z/mode_matrix_frontier.csv`

## Result Summary

- Cases requested: `3`
- Cases succeeded: `0`
- Cases failed: `3`
- Actual mode counts: `{}`
- Auto highest level in this loop: `15`
- Auto max successful dimension at that level: `None`
- Auto first failure dimension at that level: `50`

## Monte Carlo Compare

No Monte Carlo compare case was run.

## Frontier Snapshot

- `auto`: highest level `15`, max successful dimension `none`, first failure `50`, last-success storage `none` bytes, last-success batch runtime `none` ms, last-success batch peak GPU memory `none` MiB.
- `indexed`: highest level `15`, max successful dimension `none`, first failure `50`, last-success storage `none` bytes, last-success batch runtime `none` ms, last-success batch peak GPU memory `none` MiB.
- `batched`: highest level `15`, max successful dimension `none`, first failure `50`, last-success storage `none` bytes, last-success batch runtime `none` ms, last-success batch peak GPU memory `none` MiB.

## Figures

- Auto Frontier Gap gaussian float64 chunk=16384: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z/auto_frontier_gap_gaussian_float64_c16384.svg`
- Fastest Success Mode gaussian float64 chunk=16384: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z/fastest_success_mode_gaussian_float64_c16384.svg`
- Frontier gaussian float64 chunk=16384: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_085_gaussian_high_level_push/report_20260401T100520Z/frontier_gaussian_float64_c16384.svg`

## Critical Review

- Failure modes observed: {'error': 3}
- All three modes failed in the same place: `multi_indices(d=50, q=64)` attempted to allocate the full isotropic term table before any GPU-heavy integration work began.
- The raw error was `Unable to allocate 2.13 PiB for an array with shape (47855699958816, 50) and data type uint8`, so the blocker is combinatorial term enumeration, not an execution-mode threshold.
- This loop therefore does not support "batched is too slow" or "indexed storage is too large" as the primary explanation at `50D level15`; the implementation fails strictly earlier than that.

## Measurement Improvements

- Add at least one Monte Carlo compare case for the hardest successful auto-mode cell.
- Keep logging failure-onset dimensions so implementation bugs are not confused with true frontier limits.
- Add a theoretical term-count table alongside the empirical frontier because for very high `d, level` the host-side combinatorics dominate before the run reaches GPU execution.

## Next Step

- Use the critical-review findings to choose the next implementation or measurement change before rerunning.
