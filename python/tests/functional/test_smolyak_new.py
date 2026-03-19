"""Smolyak 積分器の新しいテストスイート。

責務:
- 公開 API (integrate, SmolyakIntegrator) の安全性確認
- 基本的な数値検証
- 初期化時間・メモリの基礎的な測定
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
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


SOURCE_FILE = Path(__file__).name


# ==================== 解析解 ====================

def _analytic_box_exponential(coefficients: NDArray[np.float64]) -> float:
    """exp(a^T x) の [-0.5, 0.5]^d 上積分の解析解。
    
    $$\\int_{[-0.5, 0.5]^d} \\exp(a^T x) dx = \\prod_{k=1}^d \\frac{2\\sinh(a_k/2)}{a_k}$$
    
    ただし $a_k = 0$ では極限値 1 を取る。
    """
    factors = np.where(
        np.abs(coefficients) > 1e-15,
        (2.0 * np.sinh(0.5 * coefficients)) / coefficients,
        1.0,
    )
    return float(np.prod(factors))


def _analytic_box_gaussian(alpha: float, dimension: int) -> float:
    """exp(-alpha ||x||^2) の [-0.5, 0.5]^d 上積分の解析解。
    
    $$\\int_{[-0.5, 0.5]^d} \\exp(-\\alpha\\|x\\|_2^2) dx = 
    \\left(\\sqrt{\\pi/\\alpha} \\operatorname{erf}(\\sqrt{\\alpha}/2)\\right)^d$$
    """
    factor = np.sqrt(np.pi / alpha) * erf(0.5 * np.sqrt(alpha))
    return float(factor ** dimension)


# ==================== 基本的な公開 API テスト ====================

def test_clenshaw_curtis_rule_returns_jax_arrays() -> None:
    """Clenshaw-Curtis ルールが JAX 配列を返す。"""
    nodes, weights = clenshaw_curtis_rule(3)
    assert isinstance(nodes, jnp.ndarray)
    assert isinstance(weights, jnp.ndarray)
    # dtype は DEFAULT_DTYPE (float64) に統一される
    assert nodes.dtype == jnp.float64
    assert weights.dtype == jnp.float64


def test_difference_rule_returns_jax_arrays() -> None:
    """差分ルールが JAX 配列を返す。"""
    nodes, weights = difference_rule(3)
    assert isinstance(nodes, jnp.ndarray)
    assert isinstance(weights, jnp.ndarray)
    assert nodes.dtype == jnp.float64
    assert weights.dtype == jnp.float64


def test_multi_indices_returns_compact_dtype() -> None:
    """multi_indices は最小限の unsigned dtype を返す。"""
    # 小さいケース: uint8
    indices_small = multi_indices(2, 5)  # positional-only
    assert indices_small.dtype == np.uint8
    
    # 中程度のケース: uint8 (もう少し大きくても uint8)
    indices_medium = multi_indices(3, 8)
    assert indices_medium.dtype in [np.uint8, np.uint16]  # 環境依存
    
    # 次元ゼロは ValueError
    try:
        multi_indices(0, 5)
        assert False, "should raise ValueError"
    except ValueError:
        pass


# ==================== SmolyakIntegrator の基本テスト ====================

def test_smolyak_integrator_initializes() -> None:
    """SmolyakIntegrator が初期化可能。"""
    integrator = initialize_smolyak_integrator(
        dimension=2,
        level=2,
        dtype=jnp.float64,
    )
    assert integrator.dimension == 2
    assert integrator.level == 2
    assert integrator.prepared_level == 2  # prepared_level 未指定時は level と同じ
    assert integrator.num_terms > 0
    assert integrator.num_evaluation_points > 0


def test_smolyak_integrator_preserves_constant_function() -> None:
    """Smolyak 積分器が定数関数 f(x)=1 の積分を 1.0 に保つ。
    
    責務: 規格化の基本的な正確性を確認。
    """
    constant_func = Func(lambda x: jnp.asarray([1.0], dtype=jnp.float64))
    
    for dimension in [1, 2, 3]:
        integrator = initialize_smolyak_integrator(
            dimension=dimension,
            level=2,
            dtype=jnp.float64,
        )
        result = integrate(constant_func, integrator)
        expected = np.prod([1.0] * dimension)  # Unit volume
        
        assert np.allclose(
            float(result[0]),
            expected,
            rtol=1e-5,
        ), f"Dimension {dimension} failed: got {float(result[0])}, expected {expected}"


def test_smolyak_integrator_vector_output() -> None:
    """Smolyak 積分器が多成分出力関数を処理可能。"""
    def multi_component(x):
        # dtype を DEFAULT_DTYPE (float64) に統一
        return jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    
    func = Func(multi_component)
    integrator = initialize_smolyak_integrator(dimension=2, level=2, dtype=jnp.float64)
    result = integrate(func, integrator)
    
    assert result.shape == (3,)
    assert np.allclose(result[0], 1.0, rtol=1e-5)
    assert np.allclose(result[1], 2.0, rtol=1e-5)
    assert np.allclose(result[2], 3.0, rtol=1e-5)


# ==================== 数値精度テスト ====================

def test_smolyak_exponential_integral_accuracy() -> None:
    """指数関数積分で解析解と数値解を比較。
    
    責務: Smolyak 法の精度が有意に改善するかを確認。
    """
    dimension = 2
    coefficients = np.array([0.5, 1.0], dtype=np.float64)
    
    def exponential_func(x):
        return jnp.asarray(
            [jnp.exp(jnp.dot(coefficients.astype(jnp.float64), x))],
            dtype=jnp.float64,
        )
    
    func = Func(exponential_func)
    analytic = _analytic_box_exponential(coefficients)
    
    # level ごとに誤差を比較
    errors = []
    for level in [1, 2, 3]:
        integrator = initialize_smolyak_integrator(
            dimension=dimension,
            level=level,
            dtype=jnp.float64,
        )
        numeric = float(integrate(func, integrator)[0])
        error = abs(numeric - analytic)
        errors.append(error)
    
    # 低いレベルより高いレベルの方が精度が良い
    assert errors[0] >= errors[1], f"Level 1 error {errors[0]} should be >= Level 2 error {errors[1]}"
    assert errors[1] >= errors[2], f"Level 2 error {errors[1]} should be >= Level 3 error {errors[2]}"
    
    print(json.dumps({
        "case": "smolyak_exponential_accuracy",
        "source_file": SOURCE_FILE,
        "dimension": dimension,
        "analytic": analytic,
        "errors_by_level": errors,
    }))


# ==================== refinement テスト ====================

def test_smolyak_integrator_refine() -> None:
    """SmolyakIntegrator.refine() がより細かいレベルを返す。"""
    integrator_1 = initialize_smolyak_integrator(dimension=2, level=1, dtype=jnp.float64)
    integrator_2 = integrator_1.refine()
    
    assert integrator_2.level == 2
    assert integrator_2.dimension == integrator_1.dimension
    assert integrator_2.num_terms > integrator_1.num_terms


def test_smolyak_refine_preserves_prepared_level() -> None:
    """refine() が prepared_level 内なら既存ルール storage を再利用。"""
    integrator_1 = initialize_smolyak_integrator(
        dimension=2,
        level=1,
        prepared_level=3,
        dtype=jnp.float64,
    )
    integrator_2 = integrator_1.refine()
    integrator_3 = integrator_2.refine()
    
    # 同じ rule storage を参照（メモリ効率）
    assert integrator_1.rule_nodes is integrator_2.rule_nodes
    assert integrator_2.rule_nodes is integrator_3.rule_nodes
    assert integrator_1.prepared_level == integrator_2.prepared_level == integrator_3.prepared_level == 3


# ==================== 初期化・メモリプロファイル ====================

def test_smolyak_initialization_time_and_memory() -> None:
    """SmolyakIntegrator の初期化時間を簡易計測。
    
    責務: 初期化が完了することを確認（詳細プロファイルは別途）。
    """
    results = []
    
    for dimension in [1, 2, 3, 4]:
        # 初期化時間計測（高次元ほど時間がかかる可能性）
        start_time = time.time()
        try:
            integrator = initialize_smolyak_integrator(
                dimension=dimension,
                level=1,  # level を最小に
                dtype=jnp.float64,
            )
            init_time = time.time() - start_time
            success = True
        except Exception as e:
            init_time = time.time() - start_time
            success = False
        
        results.append({
            "dimension": dimension,
            "init_time_sec": init_time,
            "num_terms": integrator.num_terms if success else None,
            "num_evaluation_points": integrator.num_evaluation_points if success else None,
        })
    
    # 初期化は成功すること
    for result in results:
        assert result["num_terms"] is not None, f"Dimension {result['dimension']} initialization failed"
    
    print(json.dumps({
        "case": "smolyak_initialization_profile",
        "source_file": SOURCE_FILE,
        "level_fixed_to": 1,
        "results": results,
    }))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
