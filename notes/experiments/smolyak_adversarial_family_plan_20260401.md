# Adversarial Integrand Family Plan For Smolyak Loops

Date: 2026-04-01

## Purpose

The Gaussian family is useful but too smooth and too forgiving.
For implementation-focused loops we also need integrand families that are still analytic enough to benchmark against exact values, but are more hostile to one or both of:

- plain Smolyak sparse-grid quadrature
- Monte Carlo estimation

The goal is not to make Smolyak fail arbitrarily.
The goal is to identify which failure mode belongs to:

- cancellation
- anisotropy
- non-smoothness
- localization
- oscillation

## Families Already Added To `run_smolyak_mode_matrix.py`

### 1. `balanced_exponential`

Formula:

`f(x) = exp(sum_i c_i x_i) - prod_i phi(c_i)`

with

`phi(c) = 2 sinh(c / 2) / c`, and `phi(0) = 1`.

Exact integral:

`0`.

Why it is useful:

- true value is exactly zero
- sensitive to cancellation and accumulation order
- exposes the difference between absolute and relative error reporting

Main stress point:

- Smolyak can suffer from cancellation error
- Monte Carlo can have large variance around a zero-mean target

### 2. `shifted_anisotropic_gaussian`

Formula:

`f(x) = exp(-sum_i alpha_i (x_i - mu_i)^2)`.

Exact integral:

Product of 1D factors

`I_i = sqrt(pi) / (2 sqrt(alpha_i)) * [erf(sqrt(alpha_i)(0.5 - mu_i)) + erf(sqrt(alpha_i)(0.5 + mu_i))]`.

Why it is useful:

- preserves smoothness, so sparse-grid theory still has a chance
- introduces anisotropy and off-center peaks
- helps separate "good smooth case" from "good only when centered and isotropic"

Main stress point:

- tests whether current mode switching and batching remain efficient when the mass is localized away from the box center

### 3. `shifted_laplace_product`

Formula:

`f(x) = exp(-sum_i beta_i |x_i - mu_i|)`.

Exact integral:

Product of 1D factors

`I_i = [2 - exp(-beta_i (0.5 + mu_i)) - exp(-beta_i (0.5 - mu_i))] / beta_i`.

Why it is useful:

- has cusps, so it breaks the smoothness assumptions more directly than Gaussian
- still admits a clean analytic integral
- gives a controlled non-smooth benchmark instead of a pathological one-off function

Main stress point:

- tests how fast sparse-grid performance degrades once mixed smoothness is weakened

## Families Planned Next

### 4. `oscillatory_cosine_product`

Formula:

`f(x) = prod_i cos(omega_i (x_i - mu_i))`.

Exact integral:

`I_i = 2 cos(omega_i mu_i) sin(omega_i / 2) / omega_i`, with the usual `omega_i -> 0` limit equal to `1`.

Why it is useful:

- introduces repeated sign changes
- can expose under-resolution and aliasing-like behavior
- may create cases where Monte Carlo remains unbiased while sparse-grid structure becomes fragile

### 5. `compact_tent_product`

Formula:

`f(x) = prod_i max(1 - |x_i - mu_i| / w_i, 0)`.

Exact integral:

If the support stays inside `[-0.5, 0.5]`, the integral is simply `prod_i w_i`.

Why it is useful:

- compact support
- off-grid peak
- non-smooth at the tent boundary

This is an especially good "sparse grid vs Monte Carlo under localized mass" benchmark.

## Suggested Parameter Ranges

### `balanced_exponential`

- `coeff_start` to `coeff_stop`: moderate symmetric ranges such as `[-1.5, 1.5]`
- for larger dimensions, prefer coefficients with mixed signs and modest magnitudes to avoid overflow

### `shifted_anisotropic_gaussian`

- `alpha_start` to `alpha_stop`: roughly `[0.2, 1.4]` for broad sweeps, then larger ranges once stability is confirmed
- `shift_start` to `shift_stop`: around `[-0.25, 0.25]`

### `shifted_laplace_product`

- `beta_start` to `beta_stop`: roughly `[1.0, 6.0]` for initial loops
- `shift_start` to `shift_stop`: around `[-0.25, 0.25]`

## Practical Reading Of Results

1. If Smolyak wins clearly on Gaussian but loses sharply on shifted Laplace, the issue is likely regularity, not implementation.
2. If Smolyak loses on shifted anisotropic Gaussian while still smooth, the issue is more likely mode choice, anisotropy handling, or effective-dimension mismatch.
3. If `balanced_exponential` shows unstable error while Gaussian remains stable, the issue is likely accumulation or numerical cancellation rather than frontier size.
