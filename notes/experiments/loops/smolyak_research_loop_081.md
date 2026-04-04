# Smolyak Research Loop 081

Date: 2026-04-01

## Goal

gaussian high-level push to level 11. Push the integrator toward the 50D level-11 target on smooth isotropic baseline.

## Executed Calculations

- Family: `gaussian`
- DTypes: `float64`
- Dimensions: `20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50`
- Levels: `1,2,3,4,5,6,7,8,9,10,11`
- Executed dimensions: `50`
- Executed levels: `4`
- Requested modes: `auto,indexed,batched`
- Chunk sizes: `16384`
- Batch size: `64`
- Timeout per case: `180` seconds
- Matrix command: `python3 -m experiments.smolyak_experiment.run_smolyak_mode_matrix --platform gpu --dimensions 50 --levels 4 --dtypes float64 --families gaussian --requested-modes auto,indexed,batched --chunk-sizes 16384 --batch-size 64 --warm-repeats 1 --timeout-seconds 180 --workers-per-gpu 1 --quiet --output-dir /workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push`

## Primary Outputs

- Matrix run dir: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z`
- Matrix JSONL: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/results.jsonl`
- Matrix summary JSON: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/summary.json`
- Matrix Markdown report: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/report.md`
- Raw CSV: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/mode_matrix_raw_cases.csv`
- Frontier CSV: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/mode_matrix_frontier.csv`

## Result Summary

- Cases requested: `3`
- Cases succeeded: `2`
- Cases failed: `1`
- Actual mode counts: `{'indexed': 2}`
- Auto highest level in this loop: `4`
- Auto max successful dimension at that level: `50`
- Auto first failure dimension at that level: `None`

## Monte Carlo Compare

- Compare JSON: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/compare/compare_smolyak_vs_mc_1775038440.json`
- Compare target: `d=50`, `level=4`, `mode=indexed`
- Same-budget: Smolyak more accurate = `False`
- Warm runtime: Smolyak faster = `False`
- Smolyak absolute error = `1.6753275085751413`
- Monte Carlo same-budget absolute error = `1.3238012899143177e-05`
- Monte Carlo matched-error absolute error = `0.012206118776594973`

## Frontier Snapshot

- `auto`: highest level `4`, max successful dimension `50`, first failure `none`, last-success storage `66498068` bytes, last-success batch runtime `7.931914005894214` ms, last-success batch peak GPU memory `432` MiB.
- `indexed`: highest level `4`, max successful dimension `50`, first failure `none`, last-success storage `66498068` bytes, last-success batch runtime `7.948052982101217` ms, last-success batch peak GPU memory `330` MiB.
- `batched`: highest level `4`, max successful dimension `none`, first failure `50`, last-success storage `none` bytes, last-success batch runtime `none` ms, last-success batch peak GPU memory `none` MiB.

## Figures

- Auto Frontier Gap gaussian float64 chunk=16384: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/auto_frontier_gap_gaussian_float64_c16384.svg`
- Fastest Success Mode gaussian float64 chunk=16384: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/fastest_success_mode_gaussian_float64_c16384.svg`
- Frontier gaussian float64 chunk=16384: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/frontier_gaussian_float64_c16384.svg`
- GPU Util gaussian float64 mode=auto chunk=16384: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_081_gaussian_high_level_push/report_20260401T101058Z/median_batch_avg_gpu_util_gaussian_float64_auto_c16384.svg`

## Critical Review

- Failure modes observed: {'timeout': 1}
- `auto` reached the current loop cap at level 4; the next loop should raise level or dimension limits.
- Monte Carlo matched or beat Smolyak on same-budget error for the chosen compare case.
- Warm-runtime speed is still a concern versus Monte Carlo on the chosen compare case.

## Measurement Improvements

- Increase the level or dimension cap in the next loop because the present frontier saturated the loop bounds.

## Next Step

- Use the critical-review findings to choose the next implementation or measurement change before rerunning.

