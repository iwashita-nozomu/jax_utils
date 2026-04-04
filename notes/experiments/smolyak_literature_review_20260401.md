# Smolyak / Monte Carlo / GPU Literature Review

Date: 2026-04-01

## Goal

This note fixes the comparison frame before the next implementation-focused experiment loops.
The practical target is a Smolyak integrator that remains useful as dimension and level grow, ideally toward `d=50`, `level=10`, while preserving a quantitative comparison against Monte Carlo on GPU.

## Primary Sources

1. Hans-Joachim Bungartz and Michael Griebel, "Sparse grids", *Acta Numerica* 13, 2004.
   Link: https://doi.org/10.1017/S0962492904000182
   Cambridge page: https://www.cambridge.org/core/journals/acta-numerica/article/sparse-grids/47EA2993DB84C9D231BB96ECB26F615C
   Relevance:
   The canonical survey. It states the standard sparse-grid promise under mixed-regularity assumptions and contrasts sparse grids with full tensor constructions.

2. Thomas Gerstner and Michael Griebel, "Dimension-Adaptive Tensor-Product Quadrature", *Computing* 71(1), 2003, pp. 65-87.
   Link: https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=200902241888115136
   Relevance:
   Important for us because our current implementation is not dimension-adaptive. This paper is a natural benchmark for what "state of the art beyond plain Smolyak" looks like.

3. Erich Novak and Henryk Wozniakowski, *Tractability of Multivariate Problems, Volume II. Standard Information for Functionals*, EMS Press, 2010.
   Link: https://doi.org/10.4171/084
   EMS page: https://ems.press/books/etm/83
   Relevance:
   Gives the right theoretical lens for high-dimensional integration. The main message is that tractability depends strongly on the function class and weighting assumptions; high dimension alone does not guarantee sparse-grid success.

4. Markus Holtz, *Sparse Grid Quadrature in High Dimensions with Applications in Finance and Insurance*, Springer, 2011.
   Link: https://doi.org/10.1007/978-3-642-16004-2
   Springer page: https://link.springer.com/book/10.1007/978-3-642-16004-2
   Relevance:
   A practical sparse-grid quadrature reference with emphasis on effective dimension, smoothing, and when sparse grids can beat Monte Carlo or quasi-Monte Carlo in application settings.

5. John D. Jakeman and Stephen G. Roberts, "Local and Dimension Adaptive Sparse Grid Interpolation and Quadrature", arXiv:1110.0010, 2011.
   Link: https://doi.org/10.48550/arXiv.1110.0010
   arXiv page: https://arxiv.org/abs/1110.0010
   Relevance:
   Directly relevant to the "what is missing from our implementation?" question. They emphasize local adaptivity and dimension selection for discontinuous or unevenly important variables.

6. Ioannis Sakiotis et al., "PAGANI: A Parallel Adaptive GPU Algorithm for Numerical Integration", arXiv:2104.06494, 2021.
   Link: https://doi.org/10.48550/arXiv.2104.06494
   arXiv page: https://arxiv.org/abs/2104.06494
   Relevance:
   A GPU-native multidimensional integration reference. It is not a sparse-grid paper, but it is valuable as a comparison point for GPU utilization strategy.

7. R. Wyrzykowski et al., "Vectorized algorithm for multidimensional Monte Carlo integration on modern GPU, CPU and MIC architectures", *The Journal of Supercomputing*, 2017.
   Link: https://link.springer.com/article/10.1007/s11227-017-2172-x
   Relevance:
   Useful baseline for the Monte Carlo side. It explicitly states the standard `O(N^{-1/2})` error behavior and focuses on high-throughput GPU execution.

## What The Literature Actually Supports

### 1. Why sparse grids can win

Bungartz and Griebel summarize the classical sparse-grid story: if the target function has enough mixed regularity, sparse grids can reduce tensor-product complexity dramatically. The survey emphasizes complexity of roughly `O(N (log N)^(d-1))` degrees of freedom instead of `O(N^d)` for full tensor constructions, with corresponding accuracy guarantees under bounded mixed derivatives.

For our project, this is the core theoretical justification for trying Smolyak at all.
If our integrands remain smooth, reasonably isotropic in mixed derivatives, and not dominated by sharp local features, a deterministic sparse-grid rule can beat plain Monte Carlo on equal sample budget and often on equal accuracy.

### 2. Why sparse grids do not automatically solve high dimension

Novak and Wozniakowski are the most important correction to any over-optimistic reading.
Their tractability perspective says that "high dimensional integration" is not one problem; it is a family of problems whose hardness changes dramatically with weights, effective dimension, and available regularity.

The practical reading for us is:

- `d=50` is not impressive by itself if the effective interaction order is low.
- `d=50` can still be hopeless if all directions matter equally and the regularity assumptions needed by sparse grids are weak or false.
- claiming success requires identifying the function class, not only reporting a dimension/level pair.

### 3. Why adaptive methods matter

Gerstner and Griebel (2003) and Jakeman and Roberts (2011) both point in the same direction: once variable importance is uneven, or the integrand contains localized difficulty, adaptivity matters.

This is especially important for our current implementation because we are optimizing materialization and batching, but not yet adapting:

- not adaptive in dimension
- not adaptive in local refinement
- not adaptive in rule selection based on observed difficulty

So even if we push the current code to higher `d` and `level`, we are not yet matching the strongest sparse-grid literature.

### 4. Why GPU performance is not only about FLOPs

The PAGANI paper is helpful here. Their GPU argument is not simply "move quadrature to GPU". It is about choosing an algorithm whose work decomposition fits the GPU well.

That matters because our current bottlenecks are likely to be:

- plan storage growth
- JAX compile latency
- irregular memory access in indexed/batched modes
- under-filled kernels for easy cases

not only raw floating-point throughput.

This means poor GPU utilization on some cases would not automatically falsify the mathematical method; it may just mean our current implementation strategy is not yet GPU-native enough.

### 5. Monte Carlo is the right baseline, but not a trivial one

The 2017 GPU Monte Carlo paper is useful because it restates the main baseline honestly:

- Monte Carlo error scales as `O(N^{-1/2})`
- it is dimension-robust in a way sparse grids usually are not
- it is relatively easy to parallelize

So Monte Carlo should remain our primary baseline.
But the comparison must be fair:

- same evaluation budget
- same precision
- same device class
- same wall-clock accounting
- repeated seeds with variance estimates

## Comparison Items For Our Next Loops

Based on the literature, the next implementation loops should compare the following quantities.

1. Success frontier:
   For each requested mode and dtype, what is the largest dimension reached at each level before timeout, OOM, compile failure, or numerical instability?

2. Error frontier:
   At fixed dimension and level, how does Smolyak absolute or relative error compare with Monte Carlo under the same number of function evaluations?

3. Time-to-accuracy:
   How much wall-clock time does Monte Carlo need to match the Smolyak error, including compile/init overhead where relevant?

4. Work decomposition:
   For each successful case, how do `num_terms`, `num_evaluation_points`, `storage_bytes`, `vectorized_ndim`, and realized mode evolve with `d` and `level`?

5. GPU efficiency:
   Throughput, average utilization, peak utilization, peak memory, and dominant Pstate must be tracked together. Pstate alone is not enough.

6. Sensitivity to integrand structure:
   Smooth isotropic Gaussian, anisotropic Gaussian, exponential, and deliberately adversarial but analytic families should be separated. The literature strongly suggests that regularity and anisotropy decide whether sparse grids shine.

## Critical Review Relative To Our Current Integrator

The literature suggests three gaps in the present implementation.

1. We are currently optimizing storage and batched execution, not adaptivity.
   That is a real improvement path, but it is narrower than the adaptive sparse-grid literature.

2. We do not yet separate "mathematical limit" from "implementation limit" cleanly enough.
   If a case fails at `d=30`, `level=8`, we need to know whether the failure is due to memory layout, JAX compilation, GPU underutilization, or the inherent growth of sparse-grid work.

3. Our best target should not be phrased only as `50D level 10`.
   A better research claim is:
   "For which integrand classes, modes, and dtypes can the current Smolyak implementation reach high dimension and high level on GPU while remaining more accurate or more time-efficient than Monte Carlo?"

## Working Hypotheses For The Next 20 Loops

1. The first ceiling will likely be implementation-driven:
   compile latency, storage growth, or batched kernel structure before pure arithmetic cost dominates.

2. `indexed` and `batched` modes should be the main path beyond the dense frontier.
   If `points` wins too often at higher `d, level`, the thresholding or storage accounting is probably misaligned.

3. Smooth Gaussian families may let Smolyak beat Monte Carlo by error at the same budget well past the point where Smolyak loses on raw wall-clock.

4. Truly adversarial, non-smooth, or highly localized families will probably expose the need for adaptive refinement rather than more aggressive materialization tricks.

## Immediate Implications For Code And Experiment Design

1. Run all requested modes without feasibility pre-filtering.
2. Keep failure-inclusive JSONL so frontier estimation is empirical.
3. Export CSV tables from JSONL so later loops can re-plot without rerunning.
4. Treat GPU monitoring as a first-class output, not a debugging extra.
5. Add adversarial but analytic integrand families to test where plain Smolyak breaks.
6. In the final report, tie each conclusion to a specific figure or table instead of reporting aggregate impressions only.
