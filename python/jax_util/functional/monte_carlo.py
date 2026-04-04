from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
from jax import lax
from jax import numpy as jnp

from ..base import Matrix, Vector
from .protocols import Function


def _integrator_key(integrator: "MonteCarloIntegrator") -> jax.Array:
    return integrator.key


# 責務: 単位体積立方体 [-0.5, 0.5]^d から一様サンプルを生成する。
def uniform_cube_samples(
    key: jax.Array,
    dimension: int,
    count: int,
) -> tuple[jax.Array, Matrix]:
    next_key, sample_key = jax.random.split(key)
    samples = jax.random.uniform(
        sample_key,
        shape=(dimension, count),
        minval=-0.5,
        maxval=0.5,
    )
    return next_key, samples


# 責務: サンプル列上の平均を取り、単位体積領域の積分値を推定する。
def monte_carlo_integral(f: Function, samples: Matrix, /) -> Vector:
    values = jax.vmap(f, in_axes=-1, out_axes=-1)(samples)
    return jnp.mean(values, axis=-1)


def monte_carlo_integral_chunked(
    f: Function,
    *,
    key: jax.Array,
    dimension: int,
    num_samples: int,
    chunk_size: int,
    sampler: Callable[[jax.Array, int, int], tuple[jax.Array, Matrix]],
) -> Vector:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    effective_chunk_size = min(chunk_size, num_samples)
    first_key, first_samples = sampler(key, dimension, effective_chunk_size)
    first_values = jax.vmap(f, in_axes=-1, out_axes=-1)(first_samples)
    total_sum = jnp.sum(first_values, axis=-1)

    num_chunks = (num_samples + effective_chunk_size - 1) // effective_chunk_size
    if num_chunks == 1:
        return total_sum / jnp.asarray(num_samples, dtype=total_sum.dtype)

    def body_fun(chunk_index: int, carry: tuple[jax.Array, Vector]) -> tuple[jax.Array, Vector]:
        current_key, current_sum = carry
        next_key, chunk_samples = sampler(current_key, dimension, effective_chunk_size)
        chunk_values = jax.vmap(f, in_axes=-1, out_axes=-1)(chunk_samples)
        chunk_start = chunk_index * effective_chunk_size
        remaining = jnp.maximum(
            jnp.asarray(num_samples - chunk_start, dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int64),
        )
        active_count = jnp.minimum(
            remaining,
            jnp.asarray(effective_chunk_size, dtype=jnp.int64),
        )
        mask = jnp.arange(effective_chunk_size, dtype=jnp.int64) < active_count
        masked_sum = jnp.tensordot(
            chunk_values,
            mask.astype(chunk_values.dtype),
            axes=([-1], [0]),
        )
        return next_key, current_sum + masked_sum

    _, total_sum = lax.fori_loop(
        1,
        num_chunks,
        body_fun,
        (first_key, total_sum),
    )
    return total_sum / jnp.asarray(num_samples, dtype=total_sum.dtype)


class MonteCarloIntegrator(eqx.Module):
    dimension: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    sampler: Callable[[jax.Array, int, int], tuple[jax.Array, Matrix]] = eqx.field(static=True)
    key: jax.Array

    def __init__(
        self,
        dimension: int,
        num_samples: int,
        key: jax.Array,
        chunk_size: int = 16384,
        sampler: Callable[[jax.Array, int, int], tuple[jax.Array, Matrix]] = uniform_cube_samples,
    ):
        self.dimension = dimension
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.sampler = sampler
        self.key = key


    def integrate(self, f: Function, /) -> Vector:
        return monte_carlo_integral_chunked(
            f,
            key=self.key,
            dimension=self.dimension,
            num_samples=self.num_samples,
            chunk_size=self.chunk_size,
            sampler=self.sampler,
        )

    # 責務: 次の乱数状態で新しいサンプル列を持つ積分器を返す。
    def update_samples(self) -> "MonteCarloIntegrator":
        next_key, _ = jax.random.split(self.key)
        return eqx.tree_at(_integrator_key, self, next_key)


__all__ = [
    "MonteCarloIntegrator",
    "uniform_cube_samples",
    "monte_carlo_integral",
    "monte_carlo_integral_chunked",
]
