# Smolyak Lazy-Indexed Decode Note

## Goal

`indexed` mode had been storing a full rule-index table with shape `(dimension, num_evaluation_points)`. For the weighted `50D level20` case this meant holding `50 x 22706` rule indices in memory before any integration call. The goal of this loop was to replace that pre-enumerated table with a compact execution plan that decodes rule indices on demand.

## Prior Work That Informed The Change

- Alin Murarasu et al., "Compact data structure and scalable algorithms for the sparse grid technique" (PPoPP 2011), DOI: <https://doi.org/10.1145/1941553.1941559>
  The key idea is to store sparse-grid points through compact integer encodings and bijections rather than pointer-heavy structures. That maps directly onto "global point id -> decoded sparse-grid location" instead of storing every decoded index tuple.
- Thomas Gerstner and Michael Griebel, "Dimension-Adaptive Tensor-Product Quadrature" (2003), DOI: <https://doi.org/10.1007/s00607-003-0015-5>
  This paper reinforces that admissible index sets should be expanded lazily from a frontier, not blindly pre-expanded everywhere.
- Markus Haji-Ali et al., "Novel results for the anisotropic sparse grid quadrature" (2018), DOI: <https://doi.org/10.1016/j.jco.2018.02.003>
  This supports the weighted admissible-set side we already added: if the active set is smaller, the decode path has fewer terms to scan.
- Jihong Chen, "Sparse quadrature for high-dimensional integration with Gaussian measure" (2018), DOI: <https://doi.org/10.1051/m2an/2018012>
  The relevant design lesson is again "admissible set first, quadrature realization second". The implementation should keep a compact set representation and derive execution data late.
- Béla Vajnovszki and Jean-Luc Vernay, "Restricted compositions and permutations: From old to new Gray codes" (2011), DOI: <https://doi.org/10.1016/j.ipl.2011.03.022>
  This is not a sparse-grid paper, but it is directly relevant to the user's intuition: bounded compositions can be traversed by small local edits. That makes Gray-code-like streaming of multi-indices a plausible next step beyond the present mixed-radix decoder.

## Implementation

Code change: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py)

- Added `requested_materialization_mode="lazy-indexed"`.
- Added `term_point_offsets`, the exclusive prefix sum of `term_num_points`.
- Added `_lazy_indexed_plan_integral(...)`, which performs:
  1. chunked traversal over global point ids,
  2. `searchsorted` from global point id to term id,
  3. term-local mixed-radix decode with `term_axis_strides`,
  4. on-the-fly `take` into `rule_nodes` and `rule_weights`.
- Kept existing `indexed` mode intact for direct A/B comparison.
- Updated experiment/report entry points so `lazy-indexed` can be selected from the command line.

This is intentionally the conservative first step. It removes the dense index table, but it does not yet implement Gray-code incremental updates across neighboring points.

## Validation

- `python3 -m pytest python/tests/functional/test_smolyak.py -q`
  Result: `21 passed`
- `python3 -m py_compile ...`
  Result: passed
- `git diff --check`
  Result: passed

New tests confirm:

- `lazy-indexed` matches the existing batched path numerically.
- `refine()` preserves `requested_materialization_mode="lazy-indexed"`.
- The integrator reports `materialization_mode == "lazy-indexed"` and does not allocate dense point/index tables.

## Benchmark Setup

Executed command:

```bash
python3 -m experiments.smolyak_experiment.run_smolyak_mode_matrix \
  --platform gpu \
  --dimensions 50 \
  --levels 20 \
  --dtypes float64 \
  --families gaussian \
  --requested-modes indexed,lazy-indexed,batched \
  --weight-scheme linear \
  --weight-scale 1.25 \
  --batch-size 1000 \
  --chunk-sizes 16384 \
  --warm-repeats 5 \
  --gpu-indices 0 \
  --workers-per-gpu 1 \
  --timeout-seconds 240 \
  --output-dir /tmp/smolyak_lazy_indexed_mode_matrix
```

Run directory:

- [/tmp/smolyak_lazy_indexed_mode_matrix/report_20260401T112748Z/report.md](/tmp/smolyak_lazy_indexed_mode_matrix/report_20260401T112748Z/report.md)
- [/tmp/smolyak_lazy_indexed_mode_matrix/report_20260401T112748Z/mode_matrix_raw_cases.csv](/tmp/smolyak_lazy_indexed_mode_matrix/report_20260401T112748Z/mode_matrix_raw_cases.csv)

Same-budget MC comparison:

```bash
python3 -m experiments.smolyak_experiment.compare_smolyak_vs_mc \
  --platform gpu \
  --dimension 50 \
  --level 20 \
  --dtype float64 \
  --family gaussian \
  --weight-scheme linear \
  --weight-scale 1.25 \
  --requested-mode lazy-indexed \
  --warm-repeats 5 \
  --skip-matched-accuracy \
  --output-dir /tmp/smolyak_lazy_indexed_compare
```

Output:

- [/tmp/smolyak_lazy_indexed_compare/compare_smolyak_vs_mc_1775042954.json](/tmp/smolyak_lazy_indexed_compare/compare_smolyak_vs_mc_1775042954.json)

## Quantitative Result

Case: weighted Gaussian, `dimension=50`, `level=20`, `dtype=float64`, `batch_size=1000`

| Mode | Storage bytes | Init ms | Batch warm ms | Throughput (integrals/s) | Avg GPU util | Peak GPU util | Dominant Pstate | Abs. error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `indexed` | 2,987,184 | 97.49 | 4.556 | 219,473.31 | 75.50 | 78 | `P2` | 1.9831e-01 |
| `lazy-indexed` | 534,936 | 77.18 | 4.847 | 206,305.75 | 57.25 | 78 | `P2` | 1.9831e-01 |
| `batched` | 534,936 | 80.86 | 2120.48 | 471.59 | 39.08 | 40 | `P5` | 1.9831e-01 |

Derived comparisons:

- `lazy-indexed` cuts storage by about `5.58x` relative to `indexed`.
- `lazy-indexed` also reduces constructor time by about `20.8%`.
- The steady-state runtime penalty versus `indexed` is only about `6.4%`.
- `lazy-indexed` remains about `437x` faster than `batched` on the same case.

## Critical Review

This change solved a real implementation problem, but not the mathematical one.

- The implementation problem:
  `indexed` paid a memory cost proportional to `dimension x num_points`. `lazy-indexed` removes that table and keeps only term metadata plus a global point-to-term prefix sum.
- The mathematical problem:
  the weighted `50D level20` Gaussian result still has absolute error `1.9831e-01`, while same-budget Monte Carlo at `22706` samples is around `7.33e-05` mean absolute error. So this loop improves execution structure, not quadrature quality.

That means the next bottleneck is no longer "can we avoid storing the index table?" The next bottlenecks are:

- richer admissible sets than the current fixed weighted rule,
- dimension-adaptive or interaction-adaptive refinement,
- possibly Gray-code-style local-update traversal if decode arithmetic becomes the next hot path.

## Takeaway

The user's intuition was good: the full index table was unnecessary. A repeated arithmetic rule over term-local mixed-radix coordinates is enough to reproduce the same quadrature points and weights. In the present codebase, that translates into a useful new middle ground:

- much smaller than `indexed`,
- dramatically faster than `batched`,
- numerically identical to the existing Smolyak rule.

It is a solid implementation improvement, but it does not by itself make `50D level20` accurate against Monte Carlo. That next step will require changing the index set, not just the decode path.
