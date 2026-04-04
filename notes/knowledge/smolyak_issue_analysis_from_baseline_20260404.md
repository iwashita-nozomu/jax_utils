# Smolyak Issue Analysis From Baseline 2026-04-04

## Scope

- Source run: `experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260403T145050Z_baseline_rewrite.jsonl`
- Focus: `integration_method == "smolyak"` only
- Purpose: separate intrinsic `Smolyak` bottlenecks from runner or environment noise

## High-Level Picture

The dominant `Smolyak` issue in the previous baseline was not numerical accuracy failure. It was frontier collapse driven by

- `integrator_init` cost at high level
- timeout at high `(dimension, level)`
- lower `vmap` reach for higher precision, especially `float64`

The data do **not** suggest a correctness crisis in the successful region. Instead, they suggest a performance-and-initialization crisis.

## Status Breakdown

- Total `Smolyak` records: `7636`
- `ok`: `1805`
- `failed`: `321`
- `skipped`: `5510`

This means the current `Smolyak` experiment behavior is already dominated by frontier skipping rather than uncontrolled hard failure.

## Failure Structure

### Failure kinds

- `timeout`: `187`
- `python_exception`: `72`
- `oom`: `62`

### Failure phases

- `timeout` failures are parent-side and have no phase tag
- among child-side failures:
  - `integrator_init`: `71`
  - `jax_import`: `63`

Interpretation:

- The main intrinsic hard failures are in `integrator_init`
- `jax_import` failures are not mathematical or algorithmic; they are environment noise

Representative non-intrinsic noise:

- `3D level1 float16 single` failed at `jax_import` with `Unable to initialize backend 'cuda': no supported devices found`

Representative intrinsic init failures:

- `2D level24 float32 single`: GPU OOM in `_difference_rule_storage_device(...)`
- `2D level27 float32 single`: cuFFT plan creation failure during Clenshaw-Curtis weight construction

## Frontier

### Single

- `float16`: max successful `(dimension, level_at_max_dimension) = (20, 5)`
- `bfloat16`: `(20, 5)`
- `float32`: `(20, 5)`
- `float64`: `(20, 4)`

### Vmap

- `float16`: `(20, 5)`
- `bfloat16`: `(20, 4)`
- `float32`: `(20, 4)`
- `float64`: `(20, 4)`

Interpretation:

- `float64` is already the weakest dtype
- `vmap` loses one level earlier than `single` in several dtypes
- the gap is modest for low precision, but becomes material for `float32/float64`

## Performance Shape

### Median successful-case performance

#### Single

- `float16`: `init 8.116 s`, `warm_runtime 149.760 ms`, `throughput 6.677 /s`
- `bfloat16`: `init 8.200 s`, `warm_runtime 139.276 ms`, `throughput 7.180 /s`
- `float32`: `init 8.209 s`, `warm_runtime 141.273 ms`, `throughput 7.079 /s`
- `float64`: `init 8.367 s`, `warm_runtime 150.092 ms`, `throughput 6.666 /s`

#### Vmap

- `float16`: `init 7.938 s`, `warm_runtime 193.705 ms`, `throughput 5162.502 /s`
- `bfloat16`: `init 7.415 s`, `warm_runtime 190.866 ms`, `throughput 5239.278 /s`
- `float32`: `init 6.973 s`, `warm_runtime 231.248 ms`, `throughput 4328.189 /s`
- `float64`: `init 5.565 s`, `warm_runtime 584.388 ms`, `throughput 1711.278 /s`

### Single vs vmap overlap

For overlapping successful `(dimension, level, dtype)` cases:

- median `single_runtime / vmap_runtime = 0.380`
- median `vmap_throughput / single_throughput = 380.064`

Interpretation:

- `vmap` is doing the right job throughput-wise
- but `float64 vmap` pays a very large runtime penalty compared to lower precision
- `single` is fundamentally not the path to optimize first for throughput

## Heaviest Successful Cases

Representative heavy successes:

- `vmap float16 d=2 l=25`: `init 191.23 s`, `warm 10570.70 ms`, `points 872,415,457`
- `single bfloat16 d=2 l=26`: `init 166.85 s`, `warm 10630.08 ms`, `points 1,811,939,575`
- `single float64 d=4 l=21`: `init 36.80 s`, `warm 33769.32 ms`, `points 4,932,501,666`
- `vmap float64 d=2 l=23`: `init 35.10 s`, `warm 34077.69 ms`, `points 201,326,776`

Interpretation:

- low dimension does not mean cheap
- huge `num_evaluation_points` in low dimension and high level is enough to dominate both init and runtime

## Accuracy

Among successful cases:

- median `mean_abs_err`:
  - `single`: `4.666349977899831e-04`
  - `vmap`: `4.2739021815463605e-04`
- max `mean_abs_err`:
  - `single`: `1.1893545031759722e-01`
  - `vmap`: `3.8869287068470086e-02`

Interpretation:

- successful `Smolyak` cases are not failing because of accuracy collapse
- the dominant issue is computational reach, not numerical unreliability

## Main Smolyak-Side Problems

### 1. `integrator_init` is still too expensive

Evidence:

- most intrinsic hard failures happen in `integrator_init`
- OOM and cuFFT plan failures appear there
- successful cases still show `~8 s` median init

Implication:

- 1D rule construction and storage preparation remain a first-class bottleneck

### 2. High-level low-dimensional cases are still pathological

Evidence:

- `2D level24+` and `4D level21` already become extreme
- successful heavy cases show point counts in the hundreds of millions to billions

Implication:

- "dimension frontier" alone hides severe low-dimensional high-level cost cliffs

### 3. `float64 vmap` is the weakest practical path

Evidence:

- lowest `vmap` frontier among dtypes
- median `warm_runtime_ms = 584.388`, far above lower-precision `vmap`
- median throughput `1711 /s`, much lower than `float16/bfloat16`

Implication:

- any `vmap`-oriented optimization should be validated on `float64`

### 4. Timeout frontier is the main runtime wall

Evidence:

- `5510` skips vs only `321` failures
- earliest skip levels track closely behind earliest failure levels

Implication:

- skip is working
- but the method still reaches a time wall very early in higher dimensions

### 5. Some failures are not Smolyak issues

Evidence:

- `jax_import` failures from missing CUDA backend

Implication:

- do not over-interpret raw failure counts without separating environment noise

## Prioritized Next Steps

### Priority A

- reduce `integrator_init` cost
- especially 1D rule generation and storage preparation

### Priority B

- improve high-level execution cost for low-dimensional large-budget cases
- these are already frontier-defining

### Priority C

- target `float64 vmap` explicitly when evaluating HLO or batching changes

### Priority D

- keep filtering out environment noise when reading frontier data

## Bottom Line

The previous baseline says the current `Smolyak` implementation is **accurate enough where it succeeds**, but is still constrained by

- initialization overhead
- very large point counts at modest dimension
- poor `float64 vmap` scaling
- timeout frontier at higher `(dimension, level)`

So the next `Smolyak` work should stay focused on performance structure, not on changing the quadrature rule itself.
