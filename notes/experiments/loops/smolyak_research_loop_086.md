# Smolyak Research Loop 086

Date: 2026-04-01

## Goal

Permutation-aware batching and lazy index decoding after-action loop. Push the integrator toward the `100D level20` target, explicitly sweep the trailing batched suffix width `1..5`, and measure whether the new data-structure work actually extends the executable frontier.

## Implementation Changes

- Added `max_vectorized_suffix_ndim` to `SmolyakIntegrator` so batched mode can sweep the number of trailing computational axes vectorized together instead of hard-coding `3`.
- Added `batched_axis_order_strategy` with `original` and `length`. The new `length` strategy sorts each term's computational axes by local 1D rule length so the longest axes land in the batched suffix.
- Added per-term batched metadata arrays:
  - `batched_term_rule_offsets`
  - `batched_term_rule_lengths`
  - `batched_term_axis_strides`
  - `batched_term_inverse_axis_permutations`
- Updated the batched execution path to decode points in computational order and then restore original coordinate order before evaluating the integrand.
- Preserved these new controls across `initialize_smolyak_integrator(...)` and `refine()`.
- Extended experiment CLIs so mode-matrix, GPU sweep, GPU batch scaling, and MC compare can record the requested suffix width and axis-order strategy.

## Theory Note

- The permutation-group investigation is in `notes/experiments/smolyak_permutation_group_theory_20260401.md`.
- The key conclusion is that a full suffix permutation group is not an exact symmetry in the general weighted / anisotropic case. Exact reuse is only guaranteed inside equal-length axis blocks.
- Therefore the implementation here does not introduce a blind full `S_n` traversal. It uses a stabilizer-safe computational reordering (`length`) plus the existing exact lazy mixed-radix decode.

## Executed Calculations

### 1. Batched suffix-width probe

- Purpose: determine whether suffix width `1..5` and axis ordering materially change batched throughput.
- Family: `gaussian`
- DType: `float64`
- Dimensions: `50,100`
- Level: `20`
- Requested mode: `batched`
- Weight schedule: `linear`
- Weight scale: `1.25`
- Batch size: `1000`
- Warm repeats: `3`
- Strategies: `original,length`
- Widths: `1,2,3,4,5`
- Aggregate CSV: `notes/experiments/results/smolyak_batched_suffix_width_probe_20260401.csv`

### 2. High-level indexed / lazy-indexed frontier

- Purpose: test whether lazy index decoding removes enough host/GPU pressure to make `100D level20` practical.
- Family: `gaussian`
- DType: `float64`
- Dimensions: `50,60,70,80,90,100`
- Levels: `15,16,17,18,19,20`
- Requested modes: `indexed,lazy-indexed`
- Weight schedule: `linear`
- Weight scale: `1.0`
- Batch size: `1000`
- Warm repeats: `1`
- GPU workers: `2`
- Run dir: `/tmp/smolyak_100d20_highlevel_indexed_modes/report_20260401T122500Z`
- Main outputs:
  - `/tmp/smolyak_100d20_highlevel_indexed_modes/report_20260401T122500Z/report.md`
  - `/tmp/smolyak_100d20_highlevel_indexed_modes/report_20260401T122500Z/mode_matrix_raw_cases.csv`
  - `/tmp/smolyak_100d20_highlevel_indexed_modes/report_20260401T122500Z/mode_matrix_frontier.csv`
  - `/tmp/smolyak_100d20_highlevel_indexed_modes/report_20260401T122500Z/mode_matrix_cross_mode.csv`

### 3. Batched frontier with the new best config

- Purpose: test how far the improved batched mode can be pushed once width/order are tuned.
- Family: `gaussian`
- DType: `float64`
- Dimensions: `50,60,70,80,90,100`
- Levels: `1..20`
- Requested mode: `batched`
- Weight schedule: `linear`
- Weight scale: `1.0`
- Batch size: `1000`
- Warm repeats: `1`
- Axis order: `length`
- Suffix width: `4`
- Resume dir used for salvage:
  - `/tmp/smolyak_100d20_batched_frontier_w4/report_20260401T120926Z`
- Resume continuation dir:
  - `/tmp/smolyak_100d20_batched_frontier_w4_resume/report_20260401T123332Z`
- I stopped the continuation after repeated `level18` timeouts because the frontier pattern was already clear and GPU time was better spent on implementation feedback.

### 4. Same-budget Monte Carlo comparison at the new target

- Purpose: verify whether the new executable `100D level20` regime is also numerically competitive.
- Case: `gaussian`, `dimension=100`, `level=20`, `dtype=float64`, `requested-mode=lazy-indexed`
- Weight schedule: `linear`
- Weight scale: `1.0`
- Compare JSON:
  - `/tmp/smolyak_compare_gpu1_lazy100d20/compare_smolyak_vs_mc_1775046879.json`

### 5. Indexed rerun for contaminated level-20 failures

- The initial `indexed` run reported `50D/60D level20` failures caused by CUDA backend initialization, not the integrator itself.
- I reran those two cases separately on GPU 2:
  - `/tmp/smolyak_indexed_l20_rerun_gpu2/report_20260401T124058Z/report.md`
- Corrected outcome:
  - `50D level20 indexed`: true `oom`
  - `60D level20 indexed`: `success`

## Quantitative Results

### Batched width/order sweep

- `100D level20` weighted probe, `original` order, width `4`:
  - batch warm runtime `2154.17 ms`
  - throughput `464.22 integrals/s`
  - dominant Pstate `P5`
- `100D level20` weighted probe, `length` order, width `4`:
  - batch warm runtime `82.59 ms`
  - throughput `12108.70 integrals/s`
  - dominant Pstate `P2`
- This is a `26.08x` throughput improvement from the ordering change plus width tuning.
- The same pattern also held at `50D level20`:
  - `original,width4`: `2098.32 ms`, `476.57 integrals/s`
  - `length,width4`: `82.63 ms`, `12102.08 integrals/s`
- Width `4` was the best point in both `50D` and `100D`. Width `5` remained strong but was slightly slower. Width `1` and `2` underused the vectorized suffix.

### High-level indexed / lazy-indexed frontier

- `lazy-indexed` succeeded on all `36/36` requested cases:
  - dimensions `50,60,70,80,90,100`
  - levels `15,16,17,18,19,20`
- `indexed` remained memory-limited:
  - level `15`: successes at all six dimensions
  - level `16`: successes at `50,60,80`; OOM at `70,90,100`
  - level `17`: successes at `60,80,90`; OOM at `50,70,100`
  - level `18`: success only at `60`; OOM elsewhere
  - level `19`: OOM at all six dimensions
  - level `20`: success at `60,70,80,90,100`; OOM at `50`
- The corrected `indexed` level-20 picture matters: the earlier `50/60` failures were not both real. After rerun, `60D` survives but `50D` still OOMs.

### 100D level20, 1000-integrand VMAP target

- `lazy-indexed`, `100D level20`, `gaussian`, `float64`:
  - points `5,238,345`
  - terms `2,087`
  - storage `27.97 MiB`
  - warm batch runtime for `1000` integrands `607.63 ms`
  - throughput `1645.73 integrals/s`
  - average GPU utilization `97.75%`
  - dominant Pstate `P2`
- `indexed`, same mathematical case:
  - storage `2066.21 MiB`
  - warm batch runtime `561.02 ms`
  - throughput `1782.46 integrals/s`
  - average GPU utilization `97.22%`
- So lazy decoding cuts storage by `73.86x` at only `8.3%` runtime penalty on this target case.
- The throughput target of “`1000` VMAP integrals in under one minute” is therefore satisfied very comfortably. The current lazy-indexed path does it in about `0.61` seconds.

### Batched frontier after tuning

- With `batched_axis_order_strategy=length` and `max_vectorized_suffix_ndim=4`, batched mode succeeded across all tested dimensions up to level `17`.
- Representative level-17 results:
  - `50D`: `16547.85 ms`, `60.43 integrals/s`, GPU util `97.93%`, Pstate `P2`
  - `100D`: `6510.62 ms`, `153.60 integrals/s`, GPU util `97.88%`, Pstate `P2`
- At level `18`, repeated timeouts appeared immediately:
  - `50D level18`: timeout
  - `60D level18`: timeout
  - `70D level18`: timeout
- This means the width/order tuning makes batched mode far healthier than before, but it still does not compete with lazy-indexed on the `100D level20` frontier.

### Monte Carlo comparison

- `100D level20 lazy-indexed` compare result:
  - Smolyak absolute error `1.0788436474691952e-01`
  - Same-budget Monte Carlo absolute error mean `3.59776961381098e-07`
  - Error ratio `299,864.57x` against Smolyak
  - Smolyak single-integrand warm runtime `112.42 ms`
  - Same-budget MC warm runtime `14.45 ms`
- Matched-accuracy MC found that even `1` Monte Carlo sample already beat this Smolyak error level on average:
  - MC matched-error absolute error `6.217788044772675e-04`
  - MC matched-error warm runtime `0.256 ms`

## Critical Review

- The implementation work succeeded on its main systems goal: `100D level20` is now executable in a memory-light mode.
- The biggest win is structural, not numerical:
  - `lazy-indexed` removes the gigantic explicit index table
  - `length`-ordered batching removes a major batched-mode throughput bottleneck
- The biggest remaining weakness is accuracy. On smooth Gaussian `100D level20`, the current weighted sparse rule is still dramatically worse than Monte Carlo at equal evaluation budget.
- Batched mode is no longer “broken,” but it is still the wrong frontier mode for the hardest cases. Its tuned runtime is much better, yet it times out already at `level18` in the unweighted linear schedule.
- Indexed mode is still useful as a speed reference, but its storage blows up fast enough that it cannot be the long-term solution.
- The `indexed` non-monotonic success pattern across dimensions is not a trustworthy “physics” result by itself. Part of it comes from OOM, and part from the weighted-set geometry. The corrected rerun removed a backend-init artifact, but the main conclusion remains: indexed storage is too fragile at this regime.

## Measurement Improvements

- The suffix width sweep should remain part of the default measurement workflow whenever batched mode is studied. Width `3` was not enough; width `4` was consistently best in the tested regime.
- High-level frontier runs should record whether failures are:
  - backend initialization
  - OOM
  - timeout
  - numerical error
- For `indexed`, rerunning suspect failures is worthwhile because backend init noise can masquerade as frontier collapse.
- For `lazy-indexed`, future loops should add explicit integer-decode profiling because the remaining runtime cost is now dominated more by decode/gather overhead than by storage.

## Next Step

- Keep `lazy-indexed` as the default frontier candidate for `50D+` and `level20` pushes.
- Do not invest further in unrestricted permutation-group enumeration; the theory note shows that it is not the correct exact abstraction in the general weighted case.
- Instead, the next implementation loop should target accuracy:
  - dimension-adaptive or interaction-adaptive index sets
  - better anisotropic weights than the current linear schedule
  - improved 1D rule nesting or error indicators
- Batched mode should continue to use width `4` and `length` order as the baseline tuned configuration until a better sweep disproves it.
