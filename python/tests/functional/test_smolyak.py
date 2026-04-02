from __future__ import annotations

import os

from jax_util.xla_env import build_cpu_env, build_gpu_env

for _key, _value in {
    **build_cpu_env(),
    **build_gpu_env(disable_preallocation=True),
}.items():
    os.environ.setdefault(_key, _value)

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import erf

from jax_util.functional import (
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


def test_clenshaw_curtis_rule_returns_jax_arrays() -> None:
    nodes, weights = clenshaw_curtis_rule(3)
    assert isinstance(nodes, jnp.ndarray)
    assert isinstance(weights, jnp.ndarray)
    assert nodes.dtype == jnp.float64
    assert weights.dtype == jnp.float64
    assert jnp.allclose(jnp.sum(weights), jnp.asarray(1.0, dtype=jnp.float64))


def test_difference_rule_returns_jax_arrays() -> None:
    nodes, weights = difference_rule(3)
    assert isinstance(nodes, jnp.ndarray)
    assert isinstance(weights, jnp.ndarray)
    assert nodes.dtype == jnp.float64
    assert weights.dtype == jnp.float64


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
        dtype=jnp.float64,
    )
    integrator_2 = integrator_1.refine()
    integrator_3 = integrator_2.refine()
    assert integrator_1.rule_nodes is integrator_2.rule_nodes
    assert integrator_2.rule_nodes is integrator_3.rule_nodes
    assert integrator_1.prepared_level == 3
    assert integrator_2.prepared_level == 3
    assert integrator_3.prepared_level == 3


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
