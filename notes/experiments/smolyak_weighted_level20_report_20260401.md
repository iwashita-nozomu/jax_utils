# Weighted Smolyak 50D Level 20 Report

Date: 2026-04-01

## Goal

This loop focused on a narrower goal than the earlier isotropic campaign:

1. make `50D level 20` executable at all,
2. push `vmap(integrate(f))` for `1000` integrands under `1 minute`,
3. keep the implementation readable enough that later adaptive work can be layered on top.

The outcome is mixed but useful. We achieved execution and throughput, but the simple weighted sparse-grid rules tested here were still substantially less accurate than Monte Carlo on the studied `50D` families.

## Implementation Changes

### 1. Weighted admissible index sets in the integrator

The main integrator change is in [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py). The integrator now accepts `dimension_weights`, and the term plan is built from a weighted budget

`sum_j w_j (l_j - 1) <= level - 1`

instead of the isotropic `|l|_1 <= d + level - 1` whenever weights are provided.

This is the change that turned `50D level 20` from immediate host-side memory failure into a tractable constructor.

### 2. Requested materialization mode is now an API choice

The integrator also now accepts `requested_materialization_mode in {auto, points, indexed, batched}`. This replaces the earlier experiment-side global threshold mutation and makes mode comparison explicit and reproducible.

This cleaned up:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py)
- [run_smolyak_mode_matrix.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/run_smolyak_mode_matrix.py)
- [compare_smolyak_vs_mc.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/compare_smolyak_vs_mc.py)
- [report_smolyak_gpu_sweep.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/report_smolyak_gpu_sweep.py)
- [report_smolyak_gpu_batch_scaling.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/report_smolyak_gpu_batch_scaling.py)

### 3. Experiment scripts now preserve more reusable data

The GPU sweep and batch-scaling reports now emit raw CSV in addition to JSON and Markdown, so later loops can re-plot without re-running the whole experiment.

### 4. Runner-level support for weighted mode matrices

[run_smolyak_mode_matrix.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/run_smolyak_mode_matrix.py) now accepts the same weight configuration as the other experiment scripts, which is necessary because the weighted frontier and the isotropic frontier are qualitatively different problems.

## Literature-Guided Design Frame

The literature points in a consistent direction: if the target is high-dimensional sparse-grid quadrature that remains executable, the first move is not “more isotropic Smolyak”, but “smaller admissible sets and better work decomposition”.

Primary references used in this loop:

1. Bungartz and Griebel, *Sparse Grids* (Acta Numerica, 2004).  
   Link: https://doi.org/10.1017/S0962492904000182

2. Gerstner and Griebel, *Dimension-Adaptive Tensor-Product Quadrature* (Computing, 2003).  
   Link: https://doi.org/10.1007/s00607-003-0015-5

3. Haji-Ali, Harbrecht, Peters, Siebenmorgen, *Novel results for the anisotropic sparse grid quadrature* (Journal of Complexity, 2018).  
   Link: https://doi.org/10.1016/j.jco.2018.02.003

4. Chen, *Sparse quadrature for high-dimensional integration with Gaussian measure* (ESAIM: M2AN, 2018).  
   Link: https://doi.org/10.1051/m2an/2018012

5. Murarasu et al., *Compact data structure and scalable algorithms for the sparse grid technique* (PPoPP, 2011).  
   Link: https://doi.org/10.1145/1941553.1941559

6. Hupp et al., *Global communication schemes for the sparse grid combination technique* (2014).  
   Link: https://doi.org/10.3233/978-1-61499-381-0-564

The implementation lesson I took from these papers is:

- weighted admissible sets are essential, not optional,
- dimension-adaptive frontier growth is the next real algorithmic step,
- compact integer-coded layouts are better suited to accelerators than object-heavy recursive structures,
- combination-technique style decomposition is a plausible future route once one monolithic sparse grid stops being the right execution unit.

## Experiments Run

### 1. Constructor sweep at `50D level 20`

Results file:

- [smolyak_weight_constructor_sweep_fast_20260401.csv](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/results/smolyak_weight_constructor_sweep_fast_20260401.csv)

Key outcomes:

| Scheme | Scale | Status | Terms | Points | Mode | Storage bytes | Init ms |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| isotropic | 1.0 | fail | - | - | - | - | - |
| log2 | 1.0 | ok | 629650 | 307490398 | batched | 903325024 | 40843.48 |
| log2 | 2.0 | ok | 2201 | 67691 | points | 30733608 | 218.27 |
| sqrt | 2.0 | ok | 3006 | 140714 | points | 61660432 | 402.90 |
| linear | 1.0 | ok | 2087 | 5238345 | batched | 19716320 | 1442.66 |
| linear | 1.25 | ok | 366 | 22706 | points | 9796048 | 167.72 |
| linear | 1.5 | ok | 225 | 13235 | points | 5733352 | 96.58 |
| linear | 2.0 | ok | 97 | 4373 | points | 1937432 | 89.12 |

The isotropic constructor failed immediately with an attempted allocation of about `2.01 EiB`, so the weighted index-set change is not a cosmetic improvement. It is the only reason `50D level 20` is executable at all in the current codebase.

### 2. Mode comparison at `50D level 20`, Gaussian, `linear 1.25`

Results:

- [mode_matrix_raw_cases.csv](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/weighted_level20_mode_matrix_gaussian_linear125/report_20260401T110339Z/mode_matrix_raw_cases.csv)
- [report.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/weighted_level20_mode_matrix_gaussian_linear125/report_20260401T110339Z/report.md)

Key table:

| Requested mode | Actual mode | Points | Storage bytes | Batch warm ms | Batch ints/s | Avg GPU util | Dominant Pstate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| auto | points | 22706 | 9796048 | 4.5695 | 218842.59 | 74.2 | P2 |
| points | points | 22706 | 9796048 | 4.6172 | 216580.05 | 38.0 | P2 |
| indexed | indexed | 22706 | 2984248 | 4.5477 | 219891.78 | 40.4 | P2 |
| batched | batched | 22706 | 532000 | 2165.1519 | 461.86 | 38.9 | P5 |

Interpretation:

- `indexed` is the best practical mode here. It cuts storage by about `3.3x` relative to `points` while matching throughput.
- `batched` saves storage again, but the runtime penalty is catastrophic at this point count.
- `auto` currently lands on `points`, which is acceptable in a clean child process, but the explicit `indexed` path is more robust for high-batch experiments because it separates execution from dense point materialization.

### 3. 1000-integrand throughput target

Main result:

- [report.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/weighted_level20_batch_scaling_gaussian_linear125_indexed/report_20260401T110140Z/report.md)
- [summary.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/weighted_level20_batch_scaling_gaussian_linear125_indexed/report_20260401T110140Z/summary.json)
- [raw_results.csv](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/weighted_level20_batch_scaling_gaussian_linear125_indexed/report_20260401T110140Z/raw_results.csv)

For `50D level20`, `linear 1.25`, `indexed`, `gaussian`, the `1000`-integrand batch achieved:

- batch warm runtime `4.5984 ms`,
- throughput `217464.91 integrals/s`,
- average GPU utilization `48.86%`,
- peak GPU utilization `72%`,
- dominant Pstate `P2`.

This clears the original throughput target by a wide margin.

![Batch Throughput](../../experiments/smolyak_experiment/results/weighted_level20_batch_scaling_gaussian_linear125_indexed/report_20260401T110140Z/throughput.svg)

![Batch GPU Util](../../experiments/smolyak_experiment/results/weighted_level20_batch_scaling_gaussian_linear125_indexed/report_20260401T110140Z/gpu_util.svg)

![Batch Pstate](../../experiments/smolyak_experiment/results/weighted_level20_batch_scaling_gaussian_linear125_indexed/report_20260401T110140Z/pstate.svg)

Reading guide:

- x-axis is `vmap` batch size,
- throughput and speedup plots are log-scale on y,
- lower Pstate number is better,
- utilization should be read together with runtime because short kernels can still leave average utilization modest.

### 4. Monte Carlo comparison on multiple families

Aggregate CSV:

- [smolyak_weighted_level20_compare_sweep_20260401.csv](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/results/smolyak_weighted_level20_compare_sweep_20260401.csv)

Same weighted configuration:

- `weight_scheme=linear`
- `weight_scale=1.25`
- `requested_mode=indexed`
- `num_points=22706`

Summary:

| Family | Smolyak abs error | Same-budget MC abs error | Error ratio | Smolyak warm ms | MC warm ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| gaussian | 1.9831e-01 | 7.3282e-05 | 2706.09 | 1.4123 | 0.2912 |
| anisotropic_gaussian | 3.5225e-02 | 8.7191e-05 | 403.997 | 1.3681 | 1.3418 |
| shifted_anisotropic_gaussian | 7.1957e-03 | 7.2036e-05 | 99.89 | 1.3486 | 1.3388 |
| balanced_exponential | 2.8809e+00 | 1.1181e-01 | 25.77 | 1.3840 | 1.3268 |
| absolute_sum | 9.5092e+00 | 3.7861e-03 | 2511.07 | 1.1416 | 0.2437 |

This is the most important negative result of the loop:

- the weighted rule made `50D level20` executable and fast,
- but the tested weighted admissible sets were still much less accurate than same-budget Monte Carlo on every family in this sweep.

### 5. Weight tradeoff sweep

Aggregate CSV:

- [smolyak_weight_tradeoff_sweep_20260401.csv](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/results/smolyak_weight_tradeoff_sweep_20260401.csv)

I compared three weight schedules under forced `indexed` execution:

- `linear 1.25`
- `log2 2.0`
- `sqrt 2.0`

All of them still met the throughput target for batch size `1000`.
However, increasing the point count by switching from `linear 1.25` to `log2 2.0` or `sqrt 2.0` did not improve accuracy reliably. On the studied Gaussian families, the absolute error was often even worse.

Key rows:

| Family | Scheme | Points | Batch1000 warm ms | Batch1000 ints/s | Smolyak/MC error ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| gaussian | linear 1.25 | 22706 | 4.5674 | 218944.21 | 2706.09 |
| gaussian | log2 2.0 | 67691 | 10.0166 | 99834.12 | 5732.21 |
| gaussian | sqrt 2.0 | 140714 | 17.4217 | 57399.52 | 8891.98 |
| anisotropic_gaussian | linear 1.25 | 22706 | 16.3659 | 61102.83 | 403.997 |
| anisotropic_gaussian | log2 2.0 | 67691 | 40.7228 | 24556.24 | 20016.26 |
| anisotropic_gaussian | sqrt 2.0 | 140714 | 69.4594 | 14396.89 | 10631.56 |

This means the next algorithmic bottleneck is not “too few points because of implementation limits”.
It is “the current weighted sparse-grid rule is not selecting the right points for these 50D test families”.

### 6. Dimension-by-dimension GPU sweep from `1D` to `50D`

Results:

- [report.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/weighted_level20_gpu_sweep_anisotropic_linear125_indexed/report_20260401T110717Z/report.md)
- [summary.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/weighted_level20_gpu_sweep_anisotropic_linear125_indexed/report_20260401T110717Z/summary.json)
- [raw_results.csv](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/weighted_level20_gpu_sweep_anisotropic_linear125_indexed/report_20260401T110717Z/raw_results.csv)

Configuration:

- family `anisotropic_gaussian`
- `anisotropic_alpha_start=1.4`
- `anisotropic_alpha_stop=0.2`
- `weight_scheme=linear`
- `weight_scale=1.25`
- `requested_mode=indexed`
- `vmap_batch_size=1000`

Selected rows:

| Dimension | Points | Storage bytes | Single warm ms | Batch warm ms | Batch ints/s | Speedup | Avg GPU util | Dominant Pstate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1032 | 27352 | 0.2421 | 2.4062 | 415593.92 | 100.60 | 66.0 | P2 |
| 10 | 22562 | 749208 | 0.3276 | 14.1111 | 70865.97 | 23.21 | 62.4 | P2 |
| 20 | 22706 | 1314448 | 1.0146 | 7.8861 | 126804.76 | 128.65 | 71.0 | P2 |
| 30 | 22706 | 1871048 | 1.0116 | 31.2084 | 32042.68 | 32.41 | 70.0 | P3 |
| 40 | 22706 | 2427648 | 1.3977 | 13.1424 | 76089.49 | 106.35 | 91.0 | P2 |
| 50 | 22706 | 2984248 | 1.0492 | 39.3043 | 25442.48 | 26.69 | 74.0 | P3 |

Two observations matter here:

1. the weighted admissible set saturates quickly, so the point count stabilizes near `22706` from around `d=20` onward,
2. even with the point count fixed, runtime and utilization still move substantially with dimension because the integrand arithmetic and gather pattern both scale with `d`.

The best batch throughput over the full sweep was `415593.92 integrals/s` at `d=1`, but the more relevant high-dimensional point is that `d=50` still stays at `39.30 ms` for `1000` integrals.

![Sweep Runtime](../../experiments/smolyak_experiment/results/weighted_level20_gpu_sweep_anisotropic_linear125_indexed/report_20260401T110717Z/runtime.svg)

![Sweep Throughput](../../experiments/smolyak_experiment/results/weighted_level20_gpu_sweep_anisotropic_linear125_indexed/report_20260401T110717Z/throughput.svg)

![Sweep Speedup](../../experiments/smolyak_experiment/results/weighted_level20_gpu_sweep_anisotropic_linear125_indexed/report_20260401T110717Z/vmap_speedup.svg)

![Sweep GPU Util](../../experiments/smolyak_experiment/results/weighted_level20_gpu_sweep_anisotropic_linear125_indexed/report_20260401T110717Z/gpu_util.svg)

![Sweep Pstate](../../experiments/smolyak_experiment/results/weighted_level20_gpu_sweep_anisotropic_linear125_indexed/report_20260401T110717Z/pstate.svg)

## Critical Review

### What worked

1. The integrator can now construct and execute `50D level20`.
2. `1000`-integrand `vmap` is no longer a stretch goal; it is achieved comfortably.
3. `indexed` materialization is the right default candidate for this high-batch regime.
4. The experiment suite is materially better than before: requested mode is explicit, weight schedules are explicit, and CSV outputs are preserved.

### What did not work

1. Accuracy did not improve enough to justify the weighted rule as a scientific result.
2. The simple monotone weight schedules tested here are too blunt.
3. Even when more points were added, the sparse-grid rule often moved further away from Monte Carlo rather than toward it.

### Likely reason

The literature-backed interpretation is that simple coordinate weights are not enough.
The next useful algorithmic ingredients are probably:

1. dimension-adaptive admissible-frontier growth,
2. interaction truncation / ANOVA-style screening,
3. possibly combination-technique execution for the selected anisotropic tensor components,
4. weight selection driven by actual integrand structure rather than a hand-coded monotone schedule.

## Current Conclusion

The current status is:

- `50D level20` execution: achieved,
- `1000`-integrand `vmap` under `1 minute`: achieved by a wide margin,
- high-accuracy deterministic quadrature at the same budget as Monte Carlo: not achieved.

The most honest statement is therefore:

> We solved the implementation-side feasibility problem for `50D level20`, but not the numerical-quality problem. The current weighted Smolyak integrator is a strong execution substrate, not yet a competitive high-accuracy rule.

The next loop should stay implementation-focused, but the implementation target should change from “more materialization tuning” to “adaptive admissible-set construction with integrand-informed prioritization”.
