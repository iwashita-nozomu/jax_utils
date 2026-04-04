# Smolyak vs Monte Carlo Theoretical Evaluation 2026-04-04

## Scope

- Data source: `experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260403T145050Z_baseline_rewrite.jsonl`
- Focus: compare `Smolyak` and `Monte Carlo` using previously recorded results
- Comparison set:
  - successful cases only
  - overlap on identical `(dimension, level, dtype, execution_variant)`

## Theory Expectations

### Monte Carlo

For standard Monte Carlo integration on a bounded-variance integrand:

- error scale is expected to be `O(N^{-1/2})`
- this rate is largely dimension-independent
- time is expected to scale roughly linearly in sample count `N`

### Smolyak

For mixed-smooth integrands, sparse-grid / Smolyak rules are expected to converge faster than Monte Carlo with respect to evaluation budget.

- For sufficiently smooth mixed-derivative structure, the error should decrease faster than `N^{-1/2}`
- For analytic tensor-friendly integrands such as the box exponential used here, Smolyak should have a substantial accuracy advantage per evaluation point
- The trade-off is more complicated setup cost and a sharper high-level frontier

This matches the literature summary already collected in:

- `notes/knowledge/smolyak_modification_methods_from_literature_20260403.md`

## Empirical Overlap Results

Number of successful overlap pairs:

- total: `1551`
- `single`: `901`
- `vmap`: `650`

### Accuracy ratio

Median ratio `Smolyak mean_abs_err / MonteCarlo mean_abs_err`:

- `single`: `0.08984`
- `vmap`: `0.10666`

Interpretation:

- In the successful overlap region, Smolyak is typically about `9x` to `11x` more accurate than Monte Carlo at the same nominal budget

### Runtime ratio

Median ratio `Smolyak warm_runtime_ms / MonteCarlo warm_runtime_ms`:

- `single`: `209.08`
- `vmap`: `79.86`

Interpretation:

- In raw runtime, Smolyak is much more expensive
- The accuracy advantage does not come for free; it is bought with substantial setup and execution cost

## Log-Log Trend Fits

These are descriptive fits over successful cases only, not asymptotic theorems.

### Error slope vs `num_evaluation_points`

Fitted slope in `log10(mean_abs_err)` vs `log10(num_evaluation_points)`:

- `Smolyak single`: `-0.5347`
- `Smolyak vmap`: `-0.3089`
- `Monte Carlo single`: `-0.3913`
- `Monte Carlo vmap`: `-0.3813`

Interpretation:

- `Monte Carlo` is roughly in the neighborhood of the expected algebraic decay, though the fit is noisy because it mixes dimensions and truncates at the success frontier
- `Smolyak single` shows a steeper empirical decay than Monte Carlo in the observed region
- `Smolyak vmap` looks flatter, which is better explained by frontier truncation and success-set bias than by a true loss of quadrature order

## Runtime Scaling Fits

Fitted slope in `log10(warm_runtime_ms)` vs `log10(num_evaluation_points)`:

- `Smolyak single`: `0.5018`
- `Smolyak vmap`: `0.4666`
- `Monte Carlo single`: `0.1878`
- `Monte Carlo vmap`: `0.3986`

Interpretation:

- None of these are pure algorithmic exponents; they are success-region fits contaminated by compile amortization, batching, skips, and frontier truncation
- The main practical point is that Smolyak runtime grows strongly with budget, and Monte Carlo grows more gently over the surviving region

## Frontier Reading

From the previous partial run:

- `Smolyak single`
  - reaches `20D level5` for `float16/bfloat16/float32`
  - reaches `20D level4` for `float64`
- `Smolyak vmap`
  - reaches `20D level5` for `float16`
  - reaches `20D level4` for `bfloat16/float32/float64`
- `Monte Carlo single`
  - reaches `20D level5` for `float16/bfloat16/float32`
  - reaches `20D level4` for `float64`
- `Monte Carlo vmap`
  - reaches `20D level5` for `float16/bfloat16/float32`
  - reaches `20D level4` for `float64`

Interpretation:

- Over the observed partial frontier, the methods appear superficially similar in reach
- But this is misleading because the previous Monte Carlo implementation had major initialization failures outside the visible frontier

## Overall Assessment

### Smolyak

Strength:

- clearly better accuracy per budget on the current analytic benchmark

Weakness:

- very high runtime cost
- much heavier setup cost
- sharp frontier from timeout and initialization cost

### Monte Carlo

Strength:

- very low runtime in the successful region
- much cheaper setup
- dimension-agnostic theory remains attractive

Weakness:

- materially worse accuracy at the same budget in the current benchmark
- previous implementation was structurally unfair because full-sample materialization caused hard failures

## Bottom Line

From the previous results alone:

- `Smolyak` is the better accuracy method on this benchmark
- `Monte Carlo` is the cheaper runtime method on this benchmark
- the practical decision boundary is dominated by whether the extra `~9x-11x` accuracy is worth the `~80x-200x` runtime cost in the overlap region

So the correct next-step interpretation is not "replace Smolyak with Monte Carlo" but:

- preserve Smolyak as the accuracy-oriented method
- reduce Smolyak setup/runtime cost
- make Monte Carlo structurally fair before drawing a final performance conclusion
