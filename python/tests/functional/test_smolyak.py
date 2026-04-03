from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.special import erf
import python.jax_util.functional.smolyak as smolyak_module
from python.jax_util.functional import (
    Func,
    SmolyakIntegrator,
    clenshaw_curtis_rule,
    difference_rule,
    initialize_smolyak_integrator,
    integrate,
    multi_indices,
)


def _analytic_box_exponential(coefficients: NDArray[np.float64]) -> float:
    factors = np.where(
        np.abs(coefficients) > 1e-15,
        (2.0 * np.sinh(0.5 * coefficients)) / coefficients,
        1.0,
    )
    return float(np.prod(factors))


def _analytic_box_gaussian(alpha: float, dimension: int) -> float:
    factor = np.sqrt(np.pi / alpha) * erf(0.5 * np.sqrt(alpha))
    return float(factor**dimension)


def _reference_clenshaw_curtis_rule(level: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if level < 1:
        raise ValueError("level must be positive.")
    if level == 1:
        return np.asarray([0.0], dtype=np.float64), np.asarray([1.0], dtype=np.float64)

    num_intervals = 2 ** (level - 1)
    coefficients = np.zeros(num_intervals + 1, dtype=np.float64)
    coefficients[0] = 1.0
    for mode in range(1, (num_intervals // 2) + 1):
        frequency = 2 * mode
        if frequency < num_intervals:
            coefficients[frequency] = -1.0 / (4.0 * mode * mode - 1.0)
    if num_intervals % 2 == 0:
        coefficients[num_intervals] = -1.0 / (num_intervals * num_intervals - 1.0)

    transformed = dct(coefficients, type=1)
    weights = np.empty(num_intervals + 1, dtype=np.float64)
    weights[0] = transformed[0] / num_intervals
    weights[-1] = transformed[-1] / num_intervals
    weights[1:-1] = 2.0 * transformed[1:-1] / num_intervals

    theta = np.pi * np.arange(num_intervals + 1, dtype=np.float64) / num_intervals
    nodes = 0.5 * np.cos(theta[::-1])
    return nodes, 0.5 * weights


def _reference_difference_rule(level: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = _reference_clenshaw_curtis_rule(level)
    node_keys = smolyak_module._clenshaw_curtis_node_keys(level)
    if level == 1:
        return nodes, weights

    previous_keys = smolyak_module._clenshaw_curtis_node_keys(level - 1)
    previous_weights = _reference_clenshaw_curtis_rule(level - 1)[1]
    all_keys = np.concatenate([node_keys, previous_keys], axis=0)
    all_weights = np.concatenate([weights, -previous_weights], axis=0)
    unique_keys, inverse = np.unique(all_keys, axis=0, return_inverse=True)
    unique_weights = np.zeros(unique_keys.shape[0], dtype=np.float64)
    np.add.at(unique_weights, inverse, all_weights)
    mask = np.abs(unique_weights) > 1.0e-15
    filtered_keys = unique_keys[mask]
    filtered_weights = unique_weights[mask]
    filtered_nodes = np.asarray(smolyak_module._clenshaw_curtis_nodes_from_keys(filtered_keys))
    order = np.argsort(filtered_nodes)
    return filtered_nodes[order], filtered_weights[order]


def test_clenshaw_curtis_rule_returns_jax_arrays() -> None:
    nodes, weights = clenshaw_curtis_rule(3)
    assert isinstance(nodes, jnp.ndarray)
    assert isinstance(weights, jnp.ndarray)
    assert nodes.dtype == jnp.float64
    assert weights.dtype == jnp.float64
    assert jnp.allclose(jnp.sum(weights), jnp.asarray(1.0, dtype=jnp.float64))
    for level in range(1, 7):
        ref_nodes, ref_weights = _reference_clenshaw_curtis_rule(level)
        cur_nodes, cur_weights = clenshaw_curtis_rule(level)
        assert np.allclose(np.asarray(cur_nodes), ref_nodes, atol=1.0e-12)
        assert np.allclose(np.asarray(cur_weights), ref_weights, atol=1.0e-12)


def test_difference_rule_returns_jax_arrays() -> None:
    nodes, weights = difference_rule(3)
    assert isinstance(nodes, jnp.ndarray)
    assert isinstance(weights, jnp.ndarray)
    assert nodes.dtype == jnp.float64
    assert weights.dtype == jnp.float64
    for level in range(1, 7):
        ref_nodes, ref_weights = _reference_difference_rule(level)
        cur_nodes, cur_weights = difference_rule(level)
        assert np.allclose(np.asarray(cur_nodes), ref_nodes, atol=1.0e-12)
        assert np.allclose(np.asarray(cur_weights), ref_weights, atol=1.0e-12)


def test_multi_indices_returns_compact_dtype() -> None:
    indices_small = multi_indices(2, 5)
    indices_medium = multi_indices(3, 8)
    assert indices_small.dtype == np.uint8
    assert indices_medium.dtype in (np.uint8, np.uint16)
    with pytest.raises(ValueError):
        multi_indices(0, 5)


def test_smolyak_integrator_initializes() -> None:
    integrator = initialize_smolyak_integrator(
        dimension=2,
        level=2,
        dtype=jnp.float64,
    )
    assert isinstance(integrator, SmolyakIntegrator)
    assert integrator.dimension == 2
    assert integrator.level == 2
    assert integrator.prepared_level == 2
    assert integrator.num_terms > 0
    assert integrator.num_evaluation_points > 0
    assert integrator.storage_bytes > 0
    assert integrator.batched_suffix_ndim == 0
    assert integrator.max_suffix_points == 1


def test_smolyak_level_one_precomputes_only_level_one_rules() -> None:
    integrator = initialize_smolyak_integrator(
        dimension=12,
        level=1,
        dtype=jnp.float64,
    )
    assert integrator.rule_lengths.shape == (1,)
    assert int(integrator.rule_lengths[0]) == 1


def test_smolyak_integrator_preserves_constant_function() -> None:
    constant_func = Func(lambda x: jnp.asarray([1.0], dtype=jnp.float64))
    for dimension in (1, 2, 3):
        integrator = initialize_smolyak_integrator(
            dimension=dimension,
            level=2,
            dtype=jnp.float64,
        )
        result = integrate(constant_func, integrator)
        assert np.allclose(float(result[0]), 1.0, rtol=1e-5)


def test_smolyak_integrator_vector_output() -> None:
    func = Func(lambda x: jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64))
    integrator = initialize_smolyak_integrator(dimension=2, level=2, dtype=jnp.float64)
    result = integrate(func, integrator)
    assert result.shape == (3,)
    assert np.allclose(np.asarray(result), np.asarray([1.0, 2.0, 3.0]), rtol=1e-5)


def test_smolyak_exponential_integral_accuracy_improves_with_level() -> None:
    coefficients = np.asarray([0.5, 1.0], dtype=np.float64)
    func = Func(
        lambda x: jnp.asarray(
            [jnp.exp(jnp.dot(jnp.asarray(coefficients, dtype=jnp.float64), x))],
            dtype=jnp.float64,
        )
    )
    analytic = _analytic_box_exponential(coefficients)
    errors: list[float] = []
    for level in (1, 2, 3):
        integrator = initialize_smolyak_integrator(
            dimension=2,
            level=level,
            dtype=jnp.float64,
        )
        numeric = float(integrate(func, integrator)[0])
        errors.append(abs(numeric - analytic))
    assert errors[0] >= errors[1]
    assert errors[1] >= errors[2]


def test_smolyak_gaussian_integral_matches_analytic_value() -> None:
    alpha = 0.8
    integrator = initialize_smolyak_integrator(dimension=3, level=5, dtype=jnp.float64)
    func = Func(
        lambda x: jnp.asarray([jnp.exp(-alpha * jnp.sum(x**2))], dtype=jnp.float64)
    )
    numeric = float(integrate(func, integrator)[0])
    analytic = _analytic_box_gaussian(alpha, 3)
    assert abs(numeric - analytic) < 1.0e-5


def test_smolyak_integrator_refine() -> None:
    integrator_1 = initialize_smolyak_integrator(dimension=2, level=1, dtype=jnp.float64)
    integrator_2 = integrator_1.refine()
    assert integrator_2.level == 2
    assert integrator_2.dimension == integrator_1.dimension
    assert integrator_2.num_terms > integrator_1.num_terms


def test_smolyak_refine_preserves_prepared_level() -> None:
    integrator_1 = initialize_smolyak_integrator(
        dimension=2,
        level=1,
        prepared_level=3,
        batched_suffix_ndim=1,
        dtype=jnp.float64,
    )
    integrator_2 = integrator_1.refine()
    integrator_3 = integrator_2.refine()
    assert integrator_1.rule_nodes is integrator_2.rule_nodes
    assert integrator_2.rule_nodes is integrator_3.rule_nodes
    assert integrator_1.prepared_level == 3
    assert integrator_2.prepared_level == 3
    assert integrator_3.prepared_level == 3
    assert integrator_1.batched_suffix_ndim == 1
    assert integrator_2.batched_suffix_ndim == 1
    assert integrator_3.batched_suffix_ndim == 1


def test_weighted_smolyak_term_selection_reduces_term_count() -> None:
    isotropic = initialize_smolyak_integrator(
        dimension=6,
        level=5,
        dtype=jnp.float64,
    )
    weighted = initialize_smolyak_integrator(
        dimension=6,
        level=5,
        dimension_weights=(1, 2, 3, 4, 5, 6),
        dtype=jnp.float64,
    )
    constant_func = Func(lambda x: jnp.asarray([1.0], dtype=jnp.float64))

    assert weighted.num_terms < isotropic.num_terms
    assert np.allclose(float(integrate(constant_func, weighted)[0]), 1.0, rtol=1e-5)


def test_weighted_smolyak_refine_preserves_weights() -> None:
    weights = (1, 2, 2, 3)
    integrator_1 = initialize_smolyak_integrator(
        dimension=4,
        level=2,
        prepared_level=4,
        dimension_weights=weights,
        dtype=jnp.float64,
    )
    integrator_2 = integrator_1.refine()

    assert integrator_1.dimension_weights == weights
    assert integrator_2.dimension_weights == weights
    assert integrator_1.rule_nodes is integrator_2.rule_nodes


def test_weighted_all_ones_matches_isotropic_smolyak() -> None:
    alpha = 0.8
    integrator_iso = initialize_smolyak_integrator(
        dimension=6,
        level=5,
        dtype=jnp.float64,
    )
    integrator_all_ones = initialize_smolyak_integrator(
        dimension=6,
        level=5,
        dimension_weights=(1, 1, 1, 1, 1, 1),
        dtype=jnp.float64,
    )
    func = Func(
        lambda x: jnp.asarray([jnp.exp(-alpha * jnp.sum(x**2))], dtype=jnp.float64)
    )

    assert integrator_all_ones.active_axis_count == 6
    assert integrator_iso.num_terms == integrator_all_ones.num_terms
    assert integrator_iso.num_evaluation_points == integrator_all_ones.num_evaluation_points
    assert np.allclose(
        np.asarray(integrate(func, integrator_iso)),
        np.asarray(integrate(func, integrator_all_ones)),
        atol=1.0e-12,
    )


def test_axis_level_ceilings_expose_linear_weight_tail_freezing() -> None:
    axis_level_ceilings = smolyak_module._axis_level_ceilings_numpy(
        100,
        20,
        dimension_weights=tuple(range(1, 101)),
    )
    assert axis_level_ceilings.shape == (100,)
    assert int(np.count_nonzero(axis_level_ceilings > 1)) == 19
    assert np.all(axis_level_ceilings[19:] == 1)
    assert int(axis_level_ceilings[0]) == 20
    assert int(axis_level_ceilings[18]) == 2


def test_smolyak_plan_integrator_is_chunk_size_invariant() -> None:
    coarse_chunks = initialize_smolyak_integrator(
        dimension=2,
        level=4,
        dtype=jnp.float64,
        chunk_size=8,
    )
    fine_chunks = initialize_smolyak_integrator(
        dimension=2,
        level=4,
        dtype=jnp.float64,
        chunk_size=256,
    )
    integrand = Func(
        lambda x: jnp.asarray(
            [
                jnp.exp(0.3 * x[0] - 0.2 * x[1]),
                x[0] ** 2 + x[1] ** 2,
            ],
            dtype=jnp.float64,
        )
    )
    coarse_value = integrate(integrand, coarse_chunks)
    fine_value = integrate(integrand, fine_chunks)
    assert jnp.allclose(coarse_value, fine_value, atol=1.0e-12)


def test_smolyak_internal_plan_matches_integrate() -> None:
    integrator = initialize_smolyak_integrator(
        dimension=4,
        level=4,
        batched_suffix_ndim=2,
        dtype=jnp.float64,
    )

    integrand = Func(
        lambda x: jnp.asarray(
            [
                jnp.exp(0.2 * x[0] - 0.1 * x[1] + 0.3 * x[2] - 0.25 * x[3]),
                jnp.sum(x**2),
            ],
            dtype=jnp.float64,
        )
    )

    integrated_value = integrate(integrand, integrator)
    internal_value = smolyak_module._smolyak_plan_integral(
        integrand,
        integrator.dimension,
        integrator.dtype,
        integrator.rule_nodes,
        integrator.rule_weights,
        integrator.rule_offsets,
        integrator.rule_lengths,
        integrator.generation_weights,
        integrator.term_budget,
        integrator.num_terms,
        chunk_size=integrator.chunk_size,
        batched_suffix_ndim=integrator.batched_suffix_ndim,
        max_suffix_points=integrator.max_suffix_points,
    )
    assert jnp.allclose(integrated_value, internal_value, atol=1.0e-12)


def test_smolyak_reports_positive_term_counts() -> None:
    integrator = initialize_smolyak_integrator(
        dimension=4,
        level=4,
        dtype=jnp.float64,
    )
    assert integrator.num_terms == 35
    assert integrator.num_evaluation_points > integrator.num_terms


def test_smolyak_reports_active_axis_count_for_weighted_case() -> None:
    integrator = initialize_smolyak_integrator(
        dimension=10,
        level=6,
        dimension_weights=tuple(range(1, 11)),
        dtype=jnp.float64,
    )
    assert integrator.active_axis_count == 5
    assert np.all(np.asarray(integrator.axis_level_ceilings)[5:] == 1)


def test_smolyak_batched_suffix_ndim_is_invariant() -> None:
    integrand = Func(
        lambda x: jnp.asarray(
            [
                jnp.exp(0.2 * x[0] - 0.1 * x[1] + 0.3 * x[2] - 0.25 * x[3]),
                jnp.sum(x**2),
            ],
            dtype=jnp.float64,
        )
    )
    baseline = initialize_smolyak_integrator(
        dimension=4,
        level=4,
        chunk_size=128,
        batched_suffix_ndim=0,
        dtype=jnp.float64,
    )
    suffix_one = initialize_smolyak_integrator(
        dimension=4,
        level=4,
        chunk_size=128,
        batched_suffix_ndim=1,
        dtype=jnp.float64,
    )
    suffix_all = initialize_smolyak_integrator(
        dimension=4,
        level=4,
        chunk_size=128,
        batched_suffix_ndim=4,
        dtype=jnp.float64,
    )
    baseline_value = integrate(integrand, baseline)
    suffix_one_value = integrate(integrand, suffix_one)
    suffix_all_value = integrate(integrand, suffix_all)
    assert jnp.allclose(baseline_value, suffix_one_value, atol=1.0e-12)
    assert jnp.allclose(baseline_value, suffix_all_value, atol=1.0e-12)
    assert suffix_one.max_suffix_points > 1
    assert suffix_all.max_suffix_points >= suffix_one.max_suffix_points


def test_smolyak_supports_forced_index_dtype_when_values_fit() -> None:
    integrator = initialize_smolyak_integrator(
        dimension=2,
        level=2,
        index_dtype="uint8",
        dtype=jnp.float64,
    )
    assert integrator.index_dtype_name == "uint8"
    assert integrator.rule_offsets.dtype == jnp.uint8
    assert integrator.rule_lengths.dtype == jnp.uint8


def test_smolyak_forced_index_dtype_raises_on_overflow() -> None:
    with pytest.raises(OverflowError):
        initialize_smolyak_integrator(
            dimension=4,
            level=8,
            index_dtype="int8",
            dtype=jnp.float64,
        )


def test_smolyak_rejects_out_of_range_batched_suffix_ndim() -> None:
    with pytest.raises(ValueError):
        initialize_smolyak_integrator(
            dimension=3,
            level=2,
            batched_suffix_ndim=4,
            dtype=jnp.float64,
        )


class _ExponentialEqxIntegrand(eqx.Module):
    coeffs: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.asarray([jnp.exp(jnp.dot(self.coeffs, x))], dtype=self.coeffs.dtype)


def test_integrate_custom_vmap_matches_lax_map_for_eqx_module_batch() -> None:
    integrator = initialize_smolyak_integrator(
        dimension=3,
        level=3,
        chunk_size=4096,
        dtype=jnp.float64,
    )
    coeff_batch = jnp.linspace(
        0.1,
        1.5,
        num=513 * 3,
        dtype=jnp.float64,
    ).reshape(513, 3)
    batched_integrands = _ExponentialEqxIntegrand(coeff_batch)

    vmapped_values = jax.vmap(lambda current_f: integrate(current_f, integrator))(batched_integrands)
    mapped_values = lax.map(lambda current_f: integrate(current_f, integrator), batched_integrands)

    assert vmapped_values.shape == (513, 1)
    assert jnp.allclose(vmapped_values, mapped_values, atol=1.0e-12)
