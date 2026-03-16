from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, cast

import equinox as eqx
import jax
from jax import flatten_util
from jax import numpy as jnp
from jax.typing import DTypeLike

from ..base import DEFAULT_DTYPE, Matrix, Scalar, Vector
from .protocols import Params, Static


class StandardCtx(eqx.Module):
    ...


class StandardCarry(eqx.Module):
    z: Matrix


class StandardNNLayer(eqx.Module):
    W: Matrix = eqx.field(static=False)
    b: Vector = eqx.field(static=False)
    activation: Callable[[Matrix], Matrix] = eqx.field(static=True)

    def __call__(self, carry: StandardCarry, ctx: StandardCtx) -> StandardCarry:
        z = self.W @ carry.z + self.b[:, None]
        return StandardCarry(z=self.activation(z))


def standard_nn_layer_factory(
    input_dim: int,
    output_dim: int,
    activation: Callable[[Matrix], Matrix],
    random_key: Scalar,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> tuple[StandardNNLayer, Scalar]:
    k1, k2 = jax.random.split(random_key, 2)
    weight_scale = _he_scale(denominator_dim=input_dim, dtype=dtype)
    W = jax.random.normal(k1, (output_dim, input_dim), dtype=dtype) * weight_scale
    b = jnp.zeros((output_dim,), dtype=dtype)
    return StandardNNLayer(W=W, b=b, activation=activation), k2


class ICNNCtx(eqx.Module):
    x: Matrix


class ICNNCarry(eqx.Module):
    z: Matrix


class ICNNLayer(eqx.Module):
    W: Matrix = eqx.field(static=False)
    W_x: Matrix = eqx.field(static=False)
    b: Vector = eqx.field(static=False)
    activation: Callable[[Matrix], Matrix] = eqx.field(static=True)

    def __call__(self, carry: ICNNCarry, ctx: ICNNCtx) -> ICNNCarry:
        z = self.W @ carry.z + self.W_x @ ctx.x + self.b[:, None]
        return ICNNCarry(z=self.activation(z))


def icnn_layer_factory(
    input_dim: int,
    output_dim: int,
    x_dim: int,
    activation: Callable[[Matrix], Matrix],
    random_key: Scalar,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> tuple[ICNNLayer, Scalar]:
    k1, k2, k3 = jax.random.split(random_key, 3)
    hidden_scale = _he_scale(denominator_dim=output_dim, dtype=dtype)
    input_scale = _he_scale(denominator_dim=x_dim, dtype=dtype)
    W = jnp.abs(jax.random.normal(k1, (output_dim, input_dim), dtype=dtype)) * hidden_scale
    W_x = jax.random.normal(k2, (output_dim, x_dim), dtype=dtype) * input_scale
    b = jnp.zeros((output_dim,), dtype=dtype)
    return ICNNLayer(W=W, W_x=W_x, b=b, activation=activation), k3


M = TypeVar("M", bound=eqx.Module)


@dataclass(frozen=True)
class RebuildState(Generic[M]):
    unravel_fn: Callable[[Vector], Params]
    static: Static


def _he_scale(denominator_dim: int, dtype: DTypeLike) -> Scalar:
    return jnp.sqrt(jnp.asarray(2.0 / denominator_dim, dtype=dtype))


def module_to_vector(module: M) -> tuple[Vector, RebuildState[M]]:
    param, static = eqx.partition(module, eqx.is_inexact_array)
    flat_params, unravel_fn = flatten_util.ravel_pytree(param)
    return flat_params, RebuildState(unravel_fn, static)


def vector_to_module(vector: Vector, st: RebuildState[M]) -> M:
    param = st.unravel_fn(vector)
    return cast(M, eqx.combine(param, st.static))


__all__ = [
    "StandardCtx",
    "StandardCarry",
    "StandardNNLayer",
    "standard_nn_layer_factory",
    "ICNNCtx",
    "ICNNCarry",
    "ICNNLayer",
    "icnn_layer_factory",
    "RebuildState",
    "module_to_vector",
    "vector_to_module",
]
