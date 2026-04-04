# Smolyak Research Loop 011

Date: 2026-04-01

## Goal

shifted_laplace_product frontier to level 4. Map the all-mode frontier for non-smooth cusp product.

## Executed Calculations

- Family: `shifted_laplace_product`
- DTypes: `float64`
- Dimensions: `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50`
- Levels: `1,2,3,4`
- Executed dimensions: `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`
- Executed levels: `1,2,3,4`
- Requested modes: `auto,points,indexed,batched`
- Chunk sizes: `16384`
- Batch size: `32`
- Timeout per case: `120` seconds
- Matrix command: `python3 -m experiments.smolyak_experiment.run_smolyak_mode_matrix --platform gpu --dimensions 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --levels 1,2,3,4 --dtypes float64 --families shifted_laplace_product --requested-modes auto,points,indexed,batched --chunk-sizes 16384 --batch-size 32 --warm-repeats 1 --timeout-seconds 120 --workers-per-gpu 1 --quiet --output-dir /workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier`

## Primary Outputs

- Matrix run dir: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T095250Z`
- Matrix JSONL: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T095250Z/results.jsonl`
- Matrix summary JSON: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T095250Z/summary.json`
- Matrix Markdown report: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T095250Z/report.md`
- Raw CSV: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T095250Z/mode_matrix_raw_cases.csv`
- Frontier CSV: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T095250Z/mode_matrix_frontier.csv`

## Result Summary

- Cases requested: `240`
- Cases succeeded: `211`
- Cases failed: `29`
- Actual mode counts: `{'batched': 36, 'indexed': 57, 'points': 118}`
- Auto highest level in this loop: `4`
- Auto max successful dimension at that level: `15`
- Auto first failure dimension at that level: `None`

## Monte Carlo Compare

- Compare JSON: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/compare/compare_smolyak_vs_mc_1775037568.json`
- Same-budget: Smolyak more accurate = `False`
- Warm runtime: Smolyak faster = `False`
- Smolyak absolute error = `0.00010164679635270727`
- Monte Carlo same-budget absolute error = `5.476176851946644e-07`
- Monte Carlo matched-error absolute error = `1.6490640508590603e-05`

## Critical Review

- Failure modes observed: {'error': 15, 'oom': 14}
- `auto` reached the current loop cap at level 4; the next loop should raise level or dimension limits.
- 32 auto-mode cells had median average GPU utilization below 10%; batching or compile amortization is still weak there.
- Monte Carlo matched or beat Smolyak on same-budget error for the chosen compare case.
- Warm-runtime speed is still a concern versus Monte Carlo on the chosen compare case.

## Measurement Improvements

- Increase the level or dimension cap in the next loop because the present frontier saturated the loop bounds.

## Next Step

- Use the critical-review findings to choose the next implementation or measurement change before rerunning.

