from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_util.functional import Func, MonteCarloIntegrator, integrate
from jax_util.functional.monte_carlo import uniform_cube_samples


SOURCE_FILE = Path(__file__).name


def _to_host_list(values: jnp.ndarray) -> list[float]:
    host_values = np.asarray(values)
    return [float(component) for component in host_values.reshape(-1)]


class _MidpointCubeSampler:
    # 責務: 1 次元立方体上の対称な中点列を返し、要求条件を記録する。
    def __init__(self):
        self.requests: list[tuple[int, int]] = []

    def __call__(self, key: jax.Array, dimension: int, count: int, /) -> tuple[jax.Array, jnp.ndarray]:
        self.requests.append((dimension, count))
        if dimension != 1:
            raise ValueError("This deterministic sampler supports only dimension=1.")

        next_key, _ = jax.random.split(key)
        points = (jnp.arange(count, dtype=jnp.float32) + 0.5) / count - 0.5
        return next_key, points[None, :]


# 責務: 積分器が定数関数を厳密に保存し、要求サンプル数も正しく伝えることを確認する。
def test_integrate_preserves_constant_functions() -> None:
    sampler = _MidpointCubeSampler()
    integrator = MonteCarloIntegrator(
        dimension=1,
        num_samples=128,
        key=jax.random.PRNGKey(0),
        sampler=sampler,
    )
    constant = jnp.asarray([3.5], dtype=jnp.float32)

    value = integrate(Func(lambda x: constant), integrator)

    print(json.dumps({
        "case": "functional_integrate_constant",
        "source_file": SOURCE_FILE,
        "test": "test_integrate_preserves_constant_functions",
        "expected": _to_host_list(constant),
        "actual": _to_host_list(value),
        "requests": sampler.requests,
    }))
    assert jnp.allclose(value, constant)
    assert sampler.requests == [(1, 128)]


# 責務: 対称サンプル列に対して奇関数の積分値が 0 になることを確認する。
def test_integrate_cancels_odd_function_on_symmetric_samples() -> None:
    integrator = MonteCarloIntegrator(
        dimension=1,
        num_samples=257,
        key=jax.random.PRNGKey(1),
        sampler=_MidpointCubeSampler(),
    )
    expected = jnp.asarray([0.0], dtype=jnp.float32)

    value = integrate(Func(lambda x: jnp.asarray([x[0] ** 3 - 0.25 * x[0]], dtype=jnp.float32)), integrator)

    print(json.dumps({
        "case": "functional_integrate_odd",
        "source_file": SOURCE_FILE,
        "test": "test_integrate_cancels_odd_function_on_symmetric_samples",
        "expected": _to_host_list(expected),
        "actual": _to_host_list(value),
    }))
    assert jnp.allclose(value, expected, atol=1e-6)


# 責務: 中点列サンプラで 2 次モーメントを高精度に近似できることを確認する。
def test_integrate_resolves_quadratic_moment_with_numeric_accuracy() -> None:
    integrator = MonteCarloIntegrator(
        dimension=1,
        num_samples=4096,
        key=jax.random.PRNGKey(2),
        sampler=_MidpointCubeSampler(),
    )
    expected = jnp.asarray([1.0 / 12.0], dtype=jnp.float32)

    value = integrate(Func(lambda x: jnp.asarray([x[0] ** 2], dtype=jnp.float32)), integrator)
    abs_err = jnp.abs(value - expected)

    print(json.dumps({
        "case": "functional_integrate_quadratic_moment",
        "source_file": SOURCE_FILE,
        "test": "test_integrate_resolves_quadratic_moment_with_numeric_accuracy",
        "expected": _to_host_list(expected),
        "actual": _to_host_list(value),
        "abs_err": _to_host_list(abs_err),
    }))
    assert float(jnp.max(abs_err)) < 5.0e-6


# 責務: ベクトル値の被積分関数に対して成分ごとの積分値を返すことを確認する。
def test_integrate_supports_vector_valued_integrands() -> None:
    integrator = MonteCarloIntegrator(
        dimension=1,
        num_samples=4096,
        key=jax.random.PRNGKey(5),
        sampler=_MidpointCubeSampler(),
    )
    expected = jnp.asarray([1.0, 1.0 / 12.0], dtype=jnp.float32)

    value = integrate(Func(lambda x: jnp.asarray([1.0, x[0] ** 2], dtype=jnp.float32)), integrator)
    abs_err = jnp.max(jnp.abs(value - expected))

    print(json.dumps({
        "case": "functional_integrate_vector_valued",
        "source_file": SOURCE_FILE,
        "test": "test_integrate_supports_vector_valued_integrands",
        "expected": _to_host_list(expected),
        "actual": _to_host_list(value),
        "max_abs_err": float(abs_err),
    }))
    assert jnp.allclose(value, expected, atol=5.0e-6)


# 責務: デフォルト sampler が [-0.5, 0.5]^d 内の点列を返すことを確認する。
def test_uniform_cube_samples_returns_points_inside_normalized_domain() -> None:
    _, samples = uniform_cube_samples(jax.random.PRNGKey(3), 3, 256)

    print(json.dumps({
        "case": "functional_uniform_cube_sampler_bounds",
        "source_file": SOURCE_FILE,
        "test": "test_uniform_cube_samples_returns_points_inside_normalized_domain",
        "shape": list(samples.shape),
        "min": float(np.asarray(jnp.min(samples))),
        "max": float(np.asarray(jnp.max(samples))),
    }))
    assert samples.shape == (3, 256)
    assert bool(jnp.all(samples >= -0.5))
    assert bool(jnp.all(samples <= 0.5))


# 責務: 積分器のサンプル更新が新しい鍵とサンプル列を返すことを確認する。
def test_monte_carlo_integrator_update_samples_returns_new_integrator() -> None:
    integrator = MonteCarloIntegrator(
        dimension=2,
        num_samples=64,
        key=jax.random.PRNGKey(4),
    )
    updated = integrator.update_samples()

    print(json.dumps({
        "case": "functional_monte_carlo_update_samples",
        "source_file": SOURCE_FILE,
        "test": "test_monte_carlo_integrator_update_samples_returns_new_integrator",
        "old_shape": list(integrator.samples.shape),
        "new_shape": list(updated.samples.shape),
        "same_key": bool(np.array_equal(np.asarray(integrator.key), np.asarray(updated.key))),
    }))
    assert updated.samples.shape == integrator.samples.shape
    assert not bool(jnp.array_equal(integrator.key, updated.key))
    assert not bool(jnp.array_equal(integrator.samples, updated.samples))


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_integrate_preserves_constant_functions()
    test_integrate_cancels_odd_function_on_symmetric_samples()
    test_integrate_resolves_quadratic_moment_with_numeric_accuracy()
    test_integrate_supports_vector_valued_integrands()
    test_uniform_cube_samples_returns_points_inside_normalized_domain()
    test_monte_carlo_integrator_update_samples_returns_new_integrator()


if __name__ == "__main__":
    _run_all_tests()
