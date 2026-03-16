from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
from jax import numpy as jnp

from ..base import Matrix, Scalar
from .layer_utils import (
    ICNNCarry,
    ICNNCtx,
    StandardCarry,
    StandardCtx,
    icnn_layer_factory,
    standard_nn_layer_factory,
)
from .protocols import Carry, Ctx, NeuralNetworkLayer


class NeuralNetwork(eqx.Module):
    layers: Tuple[NeuralNetworkLayer, ...] = eqx.field(static=False)
    network_type: str = eqx.field(static=True)
    layer_sizes: Tuple[int, ...] = eqx.field(static=True)

    def __call__(self, x: Matrix) -> Matrix:
        ctx, carry = initialize_state(
            network_type=self.network_type,
            x=x,
            layer_sizes=self.layer_sizes,
        )
        for layer in self.layers:
            carry = layer(carry, ctx)
        return carry.z


def initialize_state(
    network_type: str,
    x: Matrix,
    layer_sizes: Tuple[int, ...] | None,
) -> tuple[Ctx, Carry]:
    """ネットワークの初期状態を作成します。"""
    if network_type == "standard":
        return StandardCtx(), StandardCarry(z=x)

    if network_type == "icnn":
        if layer_sizes is None or len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements for icnn.")
        return ICNNCtx(x=x), ICNNCarry(z=x)

    raise ValueError(f"Unsupported network type: {network_type}")


def build_neural_network(
    network_type: str,
    layer_sizes: Tuple[int, ...],
    activation: str,
    random_key: Scalar,
) -> NeuralNetwork:
    """標準的な NN または ICNN を構築するファクトリ関数。"""

    def identity(x: Matrix) -> Matrix:
        return x

    activation_functions = {
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid,
        "softplus": jax.nn.softplus,
        "identity": identity,
    }
    if activation not in activation_functions:
        raise ValueError(f"Unsupported activation: {activation}")

    activation_fn = activation_functions[activation]
    layers = []
    key = random_key

    if network_type == "standard":
        for input_dim, output_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layer, key = standard_nn_layer_factory(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=activation_fn,
                random_key=key,
            )
            layers.append(layer)
    elif network_type == "icnn":
        x_dim = layer_sizes[0]
        for input_dim, output_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layer, key = icnn_layer_factory(
                input_dim=input_dim,
                output_dim=output_dim,
                x_dim=x_dim,
                activation=activation_fn,
                random_key=key,
            )
            layers.append(layer)
    else:
        raise ValueError(f"Unsupported network type: {network_type}")

    return NeuralNetwork(
        layers=tuple(layers),
        network_type=network_type,
        layer_sizes=layer_sizes,
    )


def forward_with_cache(x: Matrix, network: NeuralNetwork) -> tuple[Matrix, tuple[Carry, ...], Ctx]:
    """中間状態をキャッシュしながら順伝播を行います。"""
    ctx, carry = initialize_state(
        network_type=network.network_type,
        x=x,
        layer_sizes=network.layer_sizes,
    )

    carries = []
    for layer in network.layers:
        carry = layer(carry, ctx)
        carries.append(carry)
    return carry.z, tuple(carries), ctx


__all__ = [
    "NeuralNetwork",
    "initialize_state",
    "build_neural_network",
    "forward_with_cache",
]
