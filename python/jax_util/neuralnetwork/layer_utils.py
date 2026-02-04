from __future__ import annotations

from typing import Callable, Tuple

import equinox as eqx
import jax

from jax import numpy as jnp

from ..base import Matrix, Vector, Scalar

class StandardCtx(eqx.Module):
    ...
class StandardCarry(eqx.Module):
    z: Matrix

class StandardNNLayer(eqx.Module):
    W: "Matrix" = eqx.field(static=False)
    b: "Vector" = eqx.field(static=False)
    activation: Callable[["Matrix"], "Matrix"] = eqx.field(static=True)
    
    def __call__(self, carry: StandardCarry, ctx: StandardCtx) -> StandardCarry:
        z = self.W @ carry.z + self.b[:, None]
        return StandardCarry(z=self.activation(z))

def standardNN_layer_factory(
    input_dim: int,
    output_dim: int,
    activation: Callable[["Matrix"], "Matrix"],
    random_key: Scalar,
) -> Tuple[StandardNNLayer, Scalar]:
    k1, k2 = jax.random.split(random_key, 2)
    W = jax.random.normal(k1, (output_dim, input_dim)) * jnp.sqrt(2.0 / input_dim)
    b = jnp.zeros((output_dim,))
    return StandardNNLayer(
        W=W,
        b=b,
        activation=activation
    ), k2

class IcnnCtx(eqx.Module):
    x: Matrix

class IcnnCarry(eqx.Module):
    z: Matrix

class IcnnLayer(eqx.Module):
    W: "Matrix" = eqx.field(static=False)  # 重み行列 非負
    W_x: "Matrix" = eqx.field(static=False)  # 入力からの重み行列 非負
    b: "Vector" = eqx.field(static=False)  # バイアス項
    activation: Callable[["Matrix"], "Matrix"] = eqx.field(static=True)

    def __call__(self, carry: IcnnCarry, ctx: IcnnCtx) -> IcnnCarry:
        z = self.W @ carry.z + self.W_x @ ctx.x + self.b[:, None]
        return IcnnCarry(z=self.activation(z))
    
def icnn_layer_factory(
    input_dim: int,
    output_dim: int,
    x_dim: int,
    activation: Callable[["Matrix"], "Matrix"],
    random_key: Scalar,
) -> Tuple[IcnnLayer, Scalar]:
    k1, k2, k3 = jax.random.split(random_key, 3)
    W = jnp.abs(jax.random.normal(k1, (output_dim, input_dim))) * jnp.sqrt(2.0 / output_dim)
    W_x = jax.random.normal(k2, (output_dim, x_dim)) * jnp.sqrt(2.0 / x_dim)
    b = jnp.zeros((output_dim,))
    return IcnnLayer(
        W=W,
        W_x=W_x,
        b=b,
        activation=activation
    ), k3

__all__ = [
    "StandardCtx",
    "StandardCarry",
    "StandardNNLayer",
    "standardNN_layer_factory",
    "IcnnCtx",
    "IcnnCarry",
    "IcnnLayer",
    "icnn_layer_factory",
]