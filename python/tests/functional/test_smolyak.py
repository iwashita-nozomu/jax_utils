from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from jax_util.functional import (
    Func,
    SmolyakIntegrator,
    clenshaw_curtis_rule,
    clenshaw_curtis_node_ids,
    integrate,
    initialize_smolyak_integrator,
    multi_indices,
)

SOURCE_FILE = Path(__file__).name


def _to_host_list(values: jnp.ndarray) -> list[float]:
    host_values = np.asarray(values)
    return [float(component) for component in host_values.reshape(-1)]


# 責務: exp(a^T x) の [-0.5, 0.5]^d 上積分の解析解を返す。
def _analytic_box_exponential_integral(coefficients: NDArray[np.float64]) -> float:
    factors = (2.0 * np.sinh(0.5 * coefficients)) / coefficients
    return float(np.prod(factors))


# 責務: exp(-sum_i a_i x_i^2) の [-0.5, 0.5]^d 上積分の解析解を返す。
def _analytic_box_gaussian_integral(coefficients: NDArray[np.float64]) -> float:
    factors = np.sqrt(np.pi / coefficients) * erf(0.5 * np.sqrt(coefficients))
    return float(np.prod(factors))


# 責務: Clenshaw-Curtis 則が nested なノード列と正規化された重みを返すことを確認する。
def test_clenshaw_curtis_rule_is_nested_and_normalized() -> None:
    coarse_nodes, coarse_weights = clenshaw_curtis_rule(3)
    fine_nodes, fine_weights = clenshaw_curtis_rule(4)

    print(
        json.dumps(
            {
                "case": "functional_clenshaw_curtis_nested",
                "source_file": SOURCE_FILE,
                "test": "test_clenshaw_curtis_rule_is_nested_and_normalized",
                "coarse_nodes": _to_host_list(coarse_nodes),
                "fine_nodes": _to_host_list(fine_nodes),
                "coarse_weight_sum": float(np.asarray(jnp.sum(coarse_weights))),
                "fine_weight_sum": float(np.asarray(jnp.sum(fine_weights))),
            }
        )
    )
    assert jnp.allclose(jnp.sum(coarse_weights), jnp.asarray(1.0))
    assert jnp.allclose(jnp.sum(fine_weights), jnp.asarray(1.0))
    for node in coarse_nodes:
        assert bool(jnp.any(jnp.isclose(fine_nodes, node)))


# 責務: plan ベース積分器が定数関数の積分を 1 として保存することを確認する。
def test_smolyak_integrator_preserves_constant_integral_on_unit_volume_cube() -> None:
    integrator = initialize_smolyak_integrator(dimension=2, level=3, dtype=jnp.float64)
    value = integrate(
        Func(lambda x: jnp.asarray([1.0], dtype=jnp.float64)),
        integrator,
    )

    print(
        json.dumps(
            {
                "case": "functional_smolyak_constant_normalization",
                "source_file": SOURCE_FILE,
                "test": "test_smolyak_integrator_preserves_constant_integral_on_unit_volume_cube",
                "num_terms": integrator.num_terms,
                "num_evaluation_points": integrator.num_evaluation_points,
                "value": _to_host_list(value),
            }
        )
    )
    assert float(value[0]) == float(jnp.asarray(1.0, dtype=jnp.float64))


# 責務: canonical ID と multi-index が必要最小限の整数幅へ圧縮されることを確認する。
def test_smolyak_index_helpers_use_compact_integer_dtypes() -> None:
    cc_small = clenshaw_curtis_node_ids(6)
    cc_large = clenshaw_curtis_node_ids(12)
    indices = multi_indices(4, 6)

    print(
        json.dumps(
            {
                "case": "functional_smolyak_compact_integer_dtypes",
                "source_file": SOURCE_FILE,
                "test": "test_smolyak_index_helpers_use_compact_integer_dtypes",
                "cc_small_dtype": str(cc_small.dtype),
                "cc_large_dtype": str(cc_large.dtype),
                "indices_dtype": str(indices.dtype),
            }
        )
    )
    assert cc_small.dtype == np.uint8
    assert cc_large.dtype == np.uint16
    assert indices.dtype == np.uint8


# 責務: 対称領域上で定数項と奇項の積分が期待値どおりになることを確認する。
def test_smolyak_integrates_symmetric_terms_exactly() -> None:
    integrator = initialize_smolyak_integrator(dimension=2, level=3)
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

    print(
        json.dumps(
            {
                "case": "functional_smolyak_symmetric_terms",
                "source_file": SOURCE_FILE,
                "test": "test_smolyak_integrates_symmetric_terms_exactly",
                "expected": _to_host_list(expected),
                "actual": _to_host_list(value),
            }
        )
    )
    assert jnp.allclose(value, expected, atol=1e-12)


# 責務: 解析解のある指数関数積分を高レベル Smolyak 格子で高精度に近似できることを確認する。
def test_smolyak_resolves_analytic_exponential_integral_with_numeric_accuracy() -> None:
    coefficients = np.asarray([0.8, -0.4], dtype=np.float64)
    integrator = initialize_smolyak_integrator(dimension=2, level=5)
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

    print(
        json.dumps(
            {
                "case": "functional_smolyak_analytic_exponential",
                "source_file": SOURCE_FILE,
                "test": "test_smolyak_resolves_analytic_exponential_integral_with_numeric_accuracy",
                "coefficients": [float(value) for value in coefficients],
                "expected": _to_host_list(expected),
                "actual": _to_host_list(value),
                "abs_err": _to_host_list(abs_err),
            }
        )
    )
    assert float(jnp.max(abs_err)) < 1.0e-10


# 責務: 非線形なガウス型 integrand でも解析解に高精度で一致することを確認する。
def test_smolyak_resolves_analytic_gaussian_integral_with_numeric_accuracy() -> None:
    coefficients = np.asarray([0.7, 1.1, 1.6], dtype=np.float64)
    integrator = initialize_smolyak_integrator(dimension=3, level=7, dtype=jnp.float64)
    expected = jnp.asarray(
        [_analytic_box_gaussian_integral(coefficients)],
        dtype=jnp.float64,
    )

    value = integrate(
        Func(
            lambda x: jnp.asarray(
                [jnp.exp(-jnp.sum(jnp.asarray(coefficients, dtype=jnp.float64) * (x**2)))],
                dtype=jnp.float64,
            )
        ),
        integrator,
    )
    abs_err = jnp.abs(value - expected)

    print(
        json.dumps(
            {
                "case": "functional_smolyak_analytic_gaussian",
                "source_file": SOURCE_FILE,
                "test": "test_smolyak_resolves_analytic_gaussian_integral_with_numeric_accuracy",
                "coefficients": [float(value) for value in coefficients],
                "level": integrator.level,
                "expected": _to_host_list(expected),
                "actual": _to_host_list(value),
                "abs_err": _to_host_list(abs_err),
            }
        )
    )
    assert float(jnp.max(abs_err)) < 1.0e-8


# 責務: chunk 分割を変えても plan ベース積分器の値が不変であることを確認する。
def test_smolyak_plan_integrator_is_chunk_size_invariant() -> None:
    coarse_chunks = initialize_smolyak_integrator(
        dimension=2, level=4, dtype=jnp.float64, chunk_size=32
    )
    fine_chunks = initialize_smolyak_integrator(
        dimension=2, level=4, dtype=jnp.float64, chunk_size=2048
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

    print(
        json.dumps(
            {
                "case": "functional_smolyak_chunk_size_invariance",
                "source_file": SOURCE_FILE,
                "test": "test_smolyak_plan_integrator_is_chunk_size_invariant",
                "coarse_chunk_size": coarse_chunks.chunk_size,
                "fine_chunk_size": fine_chunks.chunk_size,
                "coarse_value": _to_host_list(coarse_value),
                "fine_value": _to_host_list(fine_value),
            }
        )
    )
    assert jnp.allclose(coarse_value, fine_value, atol=1.0e-12)


# 責務: 初期化関数が plan ベースの積分器を構築し、メタデータが整合することを確認する。
def test_initialize_smolyak_integrator_builds_plan_metadata() -> None:
    integrator = initialize_smolyak_integrator(dimension=2, level=3, dtype=jnp.float32)

    print(
        json.dumps(
            {
                "case": "functional_smolyak_initialize",
                "source_file": SOURCE_FILE,
                "test": "test_initialize_smolyak_integrator_builds_plan_metadata",
                "num_terms": integrator.num_terms,
                "num_evaluation_points": integrator.num_evaluation_points,
                "rule_nodes_shape": list(integrator.rule_nodes.shape),
                "rule_weights_shape": list(integrator.rule_weights.shape),
                "rule_offsets_shape": list(integrator.rule_offsets.shape),
                "term_levels_shape": list(integrator.term_levels.shape),
                "term_num_points_shape": list(integrator.term_num_points.shape),
                "dtype": str(integrator.rule_nodes.dtype),
            }
        )
    )
    assert isinstance(integrator, SmolyakIntegrator)
    assert integrator.num_terms > 0
    assert integrator.num_evaluation_points > 0
    assert integrator.rule_nodes.dtype == jnp.float32
    assert integrator.rule_weights.dtype == jnp.float32
    assert integrator.rule_nodes.ndim == 1
    assert integrator.rule_weights.ndim == 1
    assert integrator.rule_offsets.ndim == 1
    assert integrator.term_levels.shape[1] == 2


# 責務: refine がより高いレベルと大きい plan を持つ積分器を返すことを確認する。
def test_smolyak_integrator_refine_increases_resolution() -> None:
    integrator = initialize_smolyak_integrator(dimension=3, level=2, dtype=jnp.float32)
    refined = integrator.refine()

    print(
        json.dumps(
            {
                "case": "functional_smolyak_refine",
                "source_file": SOURCE_FILE,
                "test": "test_smolyak_integrator_refine_increases_resolution",
                "level_before": integrator.level,
                "level_after": refined.level,
                "terms_before": integrator.num_terms,
                "terms_after": refined.num_terms,
                "eval_points_before": integrator.num_evaluation_points,
                "eval_points_after": refined.num_evaluation_points,
                "dtype_before": str(integrator.rule_nodes.dtype),
                "dtype_after": str(refined.rule_nodes.dtype),
            }
        )
    )
    assert refined.level == integrator.level + 1
    assert refined.num_terms > integrator.num_terms
    assert refined.num_evaluation_points > integrator.num_evaluation_points
    assert integrator.rule_nodes.dtype == jnp.float32
    assert refined.rule_nodes.dtype == jnp.float32


# 責務: prepared_level を指定すると既存の 1 次元則 storage を再利用しながら refine できることを確認する。
def test_smolyak_integrator_reuses_prepared_rules() -> None:
    integrator = initialize_smolyak_integrator(
        dimension=3,
        level=2,
        prepared_level=4,
        dtype=jnp.float32,
    )
    refined = integrator.refine()

    print(
        json.dumps(
            {
                "case": "functional_smolyak_prepared_level",
                "source_file": SOURCE_FILE,
                "test": "test_smolyak_integrator_reuses_prepared_rules",
                "level_before": integrator.level,
                "level_after": refined.level,
                "prepared_level_before": integrator.prepared_level,
                "prepared_level_after": refined.prepared_level,
                "rule_nodes_shape_before": list(integrator.rule_nodes.shape),
                "rule_nodes_shape_after": list(refined.rule_nodes.shape),
            }
        )
    )
    assert integrator.prepared_level == 4
    assert refined.prepared_level == 4
    assert refined.level == 3
    assert integrator.rule_nodes.shape == refined.rule_nodes.shape
    assert integrator.rule_offsets.shape == refined.rule_offsets.shape


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_clenshaw_curtis_rule_is_nested_and_normalized()
    test_smolyak_integrator_preserves_constant_integral_on_unit_volume_cube()
    test_smolyak_index_helpers_use_compact_integer_dtypes()
    test_smolyak_integrates_symmetric_terms_exactly()
    test_smolyak_resolves_analytic_exponential_integral_with_numeric_accuracy()
    test_smolyak_resolves_analytic_gaussian_integral_with_numeric_accuracy()
    test_smolyak_plan_integrator_is_chunk_size_invariant()
    test_initialize_smolyak_integrator_builds_plan_metadata()
    test_smolyak_integrator_refine_increases_resolution()
    test_smolyak_integrator_reuses_prepared_rules()


if __name__ == "__main__":
    _run_all_tests()
