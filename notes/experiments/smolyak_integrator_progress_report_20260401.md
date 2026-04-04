# Smolyak Integrator Progress Report

Date: 2026-04-01

## Scope

This report summarizes the implementation-focused loop work after the earlier experiment-heavy phase.
The target remained the same throughout:

- keep Monte Carlo as the baseline
- improve the Smolyak integrator implementation itself
- push toward high-dimension, high-level execution on GPU
- identify the real limiting mechanism instead of guessing it

The most important outcome of this cycle is that the current ceiling is now much better localized.
For moderate frontier cases such as `d=50`, `level=4`, the implementation can run on GPU in `indexed` mode.
For the headline target `d=50`, `level=15`, the current plain isotropic Smolyak implementation fails before GPU execution becomes the main issue: term-set enumeration itself explodes on the host side.

## What Changed In Code

### 1. Shared analytic family handling

`experiments/smolyak_experiment/families.py` now centralizes the analytic integrand families used by:

- `compare_smolyak_vs_mc.py`
- `run_smolyak_mode_matrix.py`
- `run_smolyak_research_loops.py`

This removed duplicated family logic and exposed a real bug:

- anisotropic / shifted families had been written with inconsistent point-shape assumptions
- `balanced_exponential` used Python / NumPy conversion inside a JIT-traced path

Both were fixed.
This mattered because the loop runner is meant to be failure-inclusive; if the family layer itself is inconsistent, frontier results become uninterpretable.

### 2. Smolyak integrator cleanup

`python/jax_util/functional/smolyak.py` was simplified further:

- removed the old recursive `_smolyak_plan_integral` path and its helper chain because the current implementation uses only `points`, `indexed`, and `batched`
- kept the file focused on the actual execution paths

This is a readability improvement, but it also reduces the risk of confusing a dead path with the active implementation.

### 3. Constructor-side memory fix for high-dimension indexed runs

A second implementation bug appeared when pushing `d=50`, `level=4`.
The constructor computed the suffix-vectorization configuration by converting `self.term_rule_lengths` back from JAX to NumPy:

- the term lengths already existed as `term_rule_lengths_np`
- but the code first moved them into JAX arrays
- then called `np.asarray(...)` on the JAX array again

On a stressed GPU path this caused an avoidable OOM during construction.
The suffix configuration is now chosen directly from the original NumPy plan data before those arrays are wrapped as JAX arrays.

This change is small but important because it moved the `50D level4 auto` case from failure to success in the clean rerun.

## Quantitative Loop Results

### Loop 001: Gaussian frontier to level 4

Note:
`notes/experiments/loops/smolyak_research_loop_001.md`

Result:

- requested cases: `240`
- succeeded: `240`
- failed: `0`
- actual mode counts: `points=120`, `indexed=60`, `batched=60`
- auto frontier at this loop cap: `d=15`, `level=4`

Monte Carlo comparison on the hardest successful auto cell in this loop (`d=15`, `level=4`) was not favorable to Smolyak:

- Smolyak absolute error: `6.82e-3`
- Monte Carlo same-budget absolute error: `3.97e-4`
- Monte Carlo matched-error absolute error: `4.58e-3`

Interpretation:

- the implementation was stable on this smooth low-level regime
- but same-budget accuracy was not yet enough to claim superiority over Monte Carlo
- GPU utilization on many auto cells remained low, so this loop was more useful as a stability baseline than as a final performance argument

### Loop 011: Shifted Laplace product frontier to level 4

Note:
`notes/experiments/loops/smolyak_research_loop_011.md`

Result:

- requested cases: `240`
- succeeded: `211`
- failed: `29`
- failure counts: `oom=14`, `error=15`
- actual mode counts among successes: `points=118`, `indexed=57`, `batched=36`
- auto frontier at this loop cap: `d=15`, `level=4`

Monte Carlo comparison on the hardest successful auto cell (`d=15`, `level=4`) again favored Monte Carlo:

- Smolyak absolute error: `1.016e-4`
- Monte Carlo same-budget absolute error: `5.48e-7`
- Monte Carlo matched-error absolute error: `1.65e-5`

Interpretation:

- the non-smooth family exposed the real weakness of plain Smolyak much more clearly than the Gaussian baseline
- the successful auto frontier alone is not enough; error quality deteriorated sharply on this cusp-like family
- `batched` was clearly less robust than `points` and `indexed`
- some of the ugliest runtime instability was later traced to leaked child worker processes from interrupted loops, so these failure counts combine true mode weakness with temporarily polluted GPU state

This aligns with the literature review: non-smooth or localized difficulty is where adaptive sparse-grid methods are expected to matter, while plain isotropic Smolyak is weakest.

### Loop 016: Balanced exponential frontier to level 4

This loop did not produce a clean final frontier note in this cycle because it first exposed a new implementation bug:

- `balanced_exponential.eval_one()` converted the traced `scale` argument through NumPy / Python
- this caused `TracerArrayConversionError`

After the fix, the rerun started to produce mixed success/failure records instead of all failures, which confirmed that the original all-failure run was a pure implementation bug rather than a mathematical limitation.
The rerun was then interrupted so leaked child workers could be cleaned up and GPU resources could be redirected to the more valuable `50D` target checks.

Interpretation:

- the loop was still useful because it revealed a real correctness bug in the experiment family layer
- this is exactly why the loop process should remain critical and failure-inclusive

### Loop 085: Gaussian `d=50`, `level=15` target push

Note:
`notes/experiments/loops/smolyak_research_loop_085.md`

Result:

- requested cases: `3` (`auto`, `indexed`, `batched`)
- succeeded: `0`
- failed: `3`
- failure kind: `error`

The key point is not just that the cases failed, but why they failed.
All three cases died in `multi_indices(...)` while allocating the term table.
The raw failure text shows:

- `shape = (47855699958816, 50)`
- `dtype = uint8`
- requested allocation roughly `2.13 PiB`

This is the decisive result of the cycle.
At `d=50`, `level=15`, plain isotropic Smolyak needs

- `comb(50 + 15 - 1, 50) = comb(64, 50) = 47,855,699,958,816` terms

The supporting growth table is stored in:

- `notes/experiments/results/smolyak_term_count_growth_20260401.csv`

Interpretation:

- the current limit is not primarily GPU FLOPs
- the current limit is not primarily dense point materialization
- the current limit is the combinatorial size of the Smolyak term set itself

This means "make the same isotropic full-term Smolyak plan faster" is no longer the main bottleneck for the `50D level15` target.
Without changing the mathematical term-selection strategy, the target is not realistic.

### Clean `d=50`, `level=4` anchor after constructor fix

After moving suffix-configuration selection to the host-side NumPy plan, the clean rerun of the `d=50`, `level=4` Gaussian case showed:

- `auto`: success, actual mode `indexed`
- `indexed`: success, actual mode `indexed`
- `batched`: timeout at `180 s`
- both successful runs reported `577,826` evaluation points
- storage about `66.5 MB`
- batch warm runtime about `7.93 ms`
- average GPU utilization about `54%` to `62%`
- dominant Pstate `P2`

However, the same-budget Monte Carlo compare on this `d=50`, `level=4` success cell still favored Monte Carlo strongly:

- Smolyak absolute error: `1.6753`
- Monte Carlo same-budget absolute error: `1.3238e-5`
- Monte Carlo matched-error absolute error: `1.2206e-2`
- Smolyak warm runtime: about `2.65 ms`
- Monte Carlo same-budget warm runtime: about `2.43 ms`

Interpretation:

- the implementation can handle `50D` on GPU in a moderate-level regime
- the earlier `auto` failure at this point was an implementation artifact, not a hard mathematical ceiling
- the fix improved the practical frontier even though it did not change the fundamental `50D level15` impossibility result
- successful execution at `50D` is not the same as competitive accuracy; at `level=4`, plain isotropic Smolyak still loses badly to Monte Carlo on this Gaussian baseline

## What The Data Now Supports

The current data supports five claims.

1. The previous frontier was understated by implementation bugs.

The earlier level-1 rule-storage bug and the later constructor-side NumPy/JAX roundtrip bug both created artificial ceilings.
Fixing them moved practical frontier points outward.

2. Moderate high-dimension runs are feasible.

`d=50`, `level=4` is now a real successful GPU regime in `indexed` mode.
This is important because it shows the codebase is not fundamentally broken at high dimension.

3. Non-smooth families remain a weak point.

The shifted Laplace product loop reached the loop cap in auto mode but lost badly to Monte Carlo on same-budget accuracy.
That is a mathematical warning, not just an implementation warning.

4. The `50D level15` headline target is blocked by term enumeration, not just execution speed.

This is the largest conclusion.
The failure at `d=50`, `level=15` happened before any meaningful GPU computation.
The existing implementation still assumes full isotropic term-set materialization.
At that target, the term count alone is already catastrophic.

5. Measurement hygiene changes the empirical frontier.

Interrupted parent loops left child workers alive on the GPUs.
Those leaked workers distorted later runs with spurious OOM and backend failures.
After cleanup, the `d=50`, `level=4` Gaussian auto case moved from failure to success.
So child-process cleanup is not bookkeeping; it directly affects the validity of the frontier data.

## Difference From The Literature

The literature review already pointed to the likely gap:

- classical Smolyak theory assumes favorable mixed regularity
- modern sparse-grid practice relies heavily on dimension adaptivity, local adaptivity, or weighting
- GPU-native integration papers focus on work decompositions that fit the device instead of only porting a mathematically convenient rule

Our current implementation is still:

- non-adaptive in dimension
- non-adaptive in local refinement
- based on the full isotropic term set for the requested level

So the present implementation is still behind the adaptive sparse-grid literature in the most important way.
The new data now makes that gap quantitative instead of merely conceptual.

## Practical Next Steps

If the goal remains "something like `50D level15`", the next changes should not focus only on low-level kernel tuning.
The priority order should be:

1. Replace full `multi_indices` materialization with a streamed or implicit term iterator so the code can at least start traversing very large term sets without allocating the full table.
2. Introduce weighted or dimension-adaptive term selection, because the full isotropic term set is the real blocker.
3. Keep `indexed` as the practical high-dimension execution mode for the current implementation.
4. Treat `batched` as a separate stability/performance problem, especially on non-smooth families.
5. Add explicit cleanup for interrupted loop processes so aborted parent runs do not leave child workers behind and corrupt later measurements.

## Bottom Line

This cycle substantially improved the implementation and, more importantly, improved the honesty of the measurement process.

The code is better in three concrete ways:

- family handling is consistent and JIT-safe
- dead integrator paths are gone
- moderate `50D` indexed execution is cleaner and more stable

But the cycle also made the main limitation unmistakable:

- plain isotropic Smolyak with full term enumeration will not reach `50D level15`

To get meaningfully closer to that target, the next loop must change how terms are selected or traversed, not only how already-selected terms are executed.
