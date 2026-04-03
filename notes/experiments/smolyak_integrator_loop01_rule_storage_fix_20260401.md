# Smolyak Integrator Loop 01: Rule-Storage Fix

Date: 2026-04-01

## Goal

The first implementation-focused loop was to stop guessing about the current frontier and instead measure where the integrator actually fails.
The initial target run was an all-mode Gaussian frontier sweep on GPU:

```bash
python3 -m experiments.smolyak_experiment.run_smolyak_mode_matrix \
  --platform gpu \
  --dimensions 1,2,3,...,50 \
  --levels 1,2,3,4,5,6,7,8,9,10 \
  --dtypes float64 \
  --families gaussian \
  --requested-modes auto,points,indexed,batched \
  --chunk-sizes 16384 \
  --batch-size 32 \
  --warm-repeats 1 \
  --timeout-seconds 180 \
  --workers-per-gpu 1
```

The run was intentionally failure-inclusive: every requested case was launched in its own child process and no feasibility pre-filter skipped large cases.

## What Failed First

The first meaningful observation came from the early `auto`, `level=1`, `float64`, `gaussian` cases in the pre-fix run.

Observed before the fix:

- `d=20, level=1`: success, `actual_mode=points`, `num_evaluation_points=1`, `storage=16 MiB`, `init_ms=1551.66`
- `d=21, level=1`: success, `storage=32 MiB`, `init_ms=2974.69`
- `d=22, level=1`: success, `storage=64 MiB`, `init_ms=6374.50`
- `d=23, level=1`: success, `storage=128 MiB`, `init_ms=12632.96`
- `d=24, level=1`: success, `storage=256 MiB`, `init_ms=26610.39`
- `d=25, level=1`: OOM
- `d=26, level=1`: OOM
- `d=27+`: timeout under the same run configuration

This pattern was immediately suspicious because `level=1` on an isotropic Smolyak construction should be mathematically trivial.
Most importantly, the recorded `num_evaluation_points` was still `1`.
So the failure was not caused by quadrature-point growth.

## Diagnosis

The problem turned out to be in the integrator implementation itself.

Before the fix, the code used:

- `dimension + level - 1` as the multi-index `1`-norm limit
- the same quantity as the maximum 1D difference-rule level to precompute

Those two quantities are not the same.

For the Smolyak index set on positive integer levels:

- the `1`-norm bound is `q = d + l - 1`
- but the largest 1D level appearing on any single axis is only `l`

In other words, the implementation was precomputing many 1D difference rules that can never appear in the actual term set.
At `level=1`, the code was still building rule storage up to level `d`, which is completely unnecessary.

## Code Fix

The integrator was changed so that:

- `_max_multi_index_norm(d, l)` returns `d + l - 1`
- `_max_difference_rule_level(d, l)` returns `l`
- term-plan construction uses the multi-index norm
- rule-storage construction uses the true maximum 1D level

This is a mathematical correction, not only an optimization.

## Quantitative Effect

After the fix, direct integrator construction for `level=1` became:

- `d=12`: `rule_levels=1`, `num_evaluation_points=1`, `storage=0.000458 MiB`
- `d=24`: `rule_levels=1`, `num_evaluation_points=1`, `storage=0.000870 MiB`
- `d=50`: `rule_levels=1`, `num_evaluation_points=1`, `storage=0.001762 MiB`

The contrast at `d=24, level=1` is the key number:

- before fix: `256 MiB`, `init_ms=26610.39`
- after fix: about `0.00087 MiB`

That is not a marginal improvement.
It changes the interpretation of the frontier entirely: the previous ceiling was dominated by a storage bug, not by the underlying Smolyak rule.

## Validation

Regression testing after the fix:

- `python3 -m pytest python/tests/functional/test_smolyak.py -q`
- result: `15 passed`

An additional test now checks that `level=1` only precomputes level-1 rules.

## Why This Matters For The 50D Level-10 Goal

This loop shows that the existing frontier had been underestimating the implementation by a large margin.
Any claim about the old `d`/`level` ceiling would have mixed up:

- true sparse-grid work growth
- unnecessary rule-storage blow-up caused by an implementation bug

With this fix in place, the next loops can finally ask the right question:

"After removing avoidable storage inflation, what now limits `indexed`, `batched`, and `auto` at higher dimension and higher level?"

## Next Steps

1. Re-run the all-mode Gaussian frontier sweep on the fixed code.
2. Export updated JSONL, CSV, and SVG reports from the mode-matrix pipeline.
3. Measure whether the next ceiling comes from:
   compile time,
   materialized-plan growth,
   indexed access overhead,
   batched execution shape,
   or actual quadrature-point explosion.
4. Extend the same loop structure to adversarial analytic families such as shifted Laplace products and balanced exponentials.
