from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
from jax import numpy as jnp

from ..base import Matrix, Vector
from .protocols import Function

def _integrator_key(integrator: "MonteCarloIntegrator") -> jax.Array:
    return integrator.key


def _integrator_samples(integrator: "MonteCarloIntegrator") -> Matrix:
    return integrator.samples


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


class MonteCarloIntegrator(eqx.Module):
    dimension: int
    num_samples: int
    sampler: Callable[[jax.Array, int, int], tuple[jax.Array, Matrix]] = eqx.field(static=True)
    key: jax.Array
    samples: Matrix

    def __init__(
        self,
        dimension: int,
        num_samples: int,
        key: jax.Array,
        sampler: Callable[[jax.Array, int, int], tuple[jax.Array, Matrix]] = uniform_cube_samples,
    ):
        self.dimension = dimension
        self.num_samples = num_samples
        self.sampler = sampler
        self.key, self.samples = sampler(key, dimension, num_samples)


    def __call__(self, f: Function, /) -> Vector:
        return monte_carlo_integral(f, self.samples)

    # 責務: 次の乱数状態で新しいサンプル列を持つ積分器を返す。
    def update_samples(self) -> "MonteCarloIntegrator":
        next_key, next_samples = self.sampler(self.key, self.dimension, self.num_samples)
        updated = eqx.tree_at(_integrator_key, self, next_key)
        return eqx.tree_at(_integrator_samples, updated, next_samples)


__all__ = [
    "MonteCarloIntegrator",
    "uniform_cube_samples",
    "monte_carlo_integral",
]
