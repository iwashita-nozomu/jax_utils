from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jax_util.functional import (
    Func,
    SmolyakIntegrator,
    clenshaw_curtis_rule,
    integrate,
    smolyak_grid,
)


SOURCE_FILE = Path(__file__).name


def _to_host_list(values: jnp.ndarray) -> list[float]:
    host_values = np.asarray(values)
    return [float(component) for component in host_values.reshape(-1)]


# 責務: exp(a^T x) の [-0.5, 0.5]^d 上積分の解析解を返す。
def _analytic_box_exponential_integral(coefficients: NDArray[np.float64]) -> float:
    factors = (2.0 * np.sinh(0.5 * coefficients)) / coefficients
    return float(np.prod(factors))


# 責務: Clenshaw-Curtis 則が nested なノード列と正規化された重みを返すことを確認する。
def test_clenshaw_curtis_rule_is_nested_and_normalized() -> None:
    coarse_nodes, coarse_weights = clenshaw_curtis_rule(3)
    fine_nodes, fine_weights = clenshaw_curtis_rule(4)

    print(json.dumps({
        "case": "functional_clenshaw_curtis_nested",
        "source_file": SOURCE_FILE,
        "test": "test_clenshaw_curtis_rule_is_nested_and_normalized",
        "coarse_nodes": _to_host_list(coarse_nodes),
        "fine_nodes": _to_host_list(fine_nodes),
        "coarse_weight_sum": float(np.asarray(jnp.sum(coarse_weights))),
        "fine_weight_sum": float(np.asarray(jnp.sum(fine_weights))),
    }))
    assert jnp.allclose(jnp.sum(coarse_weights), jnp.asarray(1.0))
    assert jnp.allclose(jnp.sum(fine_weights), jnp.asarray(1.0))
    for node in coarse_nodes:
        assert bool(jnp.any(jnp.isclose(fine_nodes, node)))


# 責務: 構成した疎格子が正規化領域内にあり、重み和が 1 になることを確認する。
def test_smolyak_grid_is_normalized_on_unit_volume_cube() -> None:
    points, weights = smolyak_grid(2, 3)

    print(json.dumps({
        "case": "functional_smolyak_grid_normalization",
        "source_file": SOURCE_FILE,
        "test": "test_smolyak_grid_is_normalized_on_unit_volume_cube",
        "num_points": int(points.shape[1]),
        "weight_sum": float(np.asarray(jnp.sum(weights))),
        "min": float(np.asarray(jnp.min(points))),
        "max": float(np.asarray(jnp.max(points))),
    }))
    assert points.shape[0] == 2
    assert bool(jnp.all(points >= -0.5))
    assert bool(jnp.all(points <= 0.5))
    assert jnp.allclose(jnp.sum(weights), jnp.asarray(1.0))


# 責務: 対称領域上で定数項と奇項の積分が期待値どおりになることを確認する。
def test_smolyak_integrates_symmetric_terms_exactly() -> None:
    integrator = SmolyakIntegrator(dimension=2, level=3)
    expected = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)

    value = integrate(
        Func(
            lambda x: jnp.asarray(
                [
                    1.0,
                    x[0] + x[1] ** 3,
                    x[0] * x[1],
                ],
                dtype=jnp.float64,
            )
        ),
        integrator,
    )

    print(json.dumps({
        "case": "functional_smolyak_symmetric_terms",
        "source_file": SOURCE_FILE,
        "test": "test_smolyak_integrates_symmetric_terms_exactly",
        "expected": _to_host_list(expected),
        "actual": _to_host_list(value),
    }))
    assert jnp.allclose(value, expected, atol=1e-12)


# 責務: 解析解のある指数関数積分を高レベル Smolyak 格子で高精度に近似できることを確認する。
def test_smolyak_resolves_analytic_exponential_integral_with_numeric_accuracy() -> None:
    coefficients = np.asarray([0.8, -0.4], dtype=np.float64)
    integrator = SmolyakIntegrator(dimension=2, level=5)
    expected = jnp.asarray(
        [_analytic_box_exponential_integral(coefficients)],
        dtype=jnp.float64,
    )

    value = integrate(
        Func(
            lambda x: jnp.asarray(
                [jnp.exp(coefficients[0] * x[0] + coefficients[1] * x[1])],
                dtype=jnp.float64,
            )
        ),
        integrator,
    )
    abs_err = jnp.abs(value - expected)

    print(json.dumps({
        "case": "functional_smolyak_analytic_exponential",
        "source_file": SOURCE_FILE,
        "test": "test_smolyak_resolves_analytic_exponential_integral_with_numeric_accuracy",
        "coefficients": [float(value) for value in coefficients],
        "expected": _to_host_list(expected),
        "actual": _to_host_list(value),
        "abs_err": _to_host_list(abs_err),
    }))
    assert float(jnp.max(abs_err)) < 1.0e-10


# 責務: refine がより高いレベルと多い格子点数を持つ積分器を返すことを確認する。
def test_smolyak_integrator_refine_increases_resolution() -> None:
    integrator = SmolyakIntegrator(dimension=3, level=2, dtype=jnp.float32)
    refined = integrator.refine()

    print(json.dumps({
        "case": "functional_smolyak_refine",
        "source_file": SOURCE_FILE,
        "test": "test_smolyak_integrator_refine_increases_resolution",
        "level_before": integrator.level,
        "level_after": refined.level,
        "points_before": int(integrator.points.shape[1]),
        "points_after": int(refined.points.shape[1]),
        "dtype_before": str(integrator.points.dtype),
        "dtype_after": str(refined.points.dtype),
    }))
    assert refined.level == integrator.level + 1
    assert refined.points.shape[1] > integrator.points.shape[1]
    assert integrator.points.dtype == jnp.float32
    assert refined.points.dtype == jnp.float32


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_clenshaw_curtis_rule_is_nested_and_normalized()
    test_smolyak_grid_is_normalized_on_unit_volume_cube()
    test_smolyak_integrates_symmetric_terms_exactly()
    test_smolyak_resolves_analytic_exponential_integral_with_numeric_accuracy()
    test_smolyak_integrator_refine_increases_resolution()


if __name__ == "__main__":
    _run_all_tests()
