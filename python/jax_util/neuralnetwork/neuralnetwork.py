from __future__ import annotations

from typing import Tuple

from ..base import *

from .protocols import *
from .layer_utils import *
import equinox as eqx
import jax
from jax import numpy as jnp



class NeuralNetwork(eqx.Module):
    layers: Tuple[NeuralNetworkLayer,...] = eqx.field(static=False)
    network_type: str = eqx.field(static=True)
    layer_sizes: Tuple[int, ...] = eqx.field(static=True)

    def __call__(self, x: Matrix) -> Matrix:
        ctx, carry = state_initializer(
            network_type=self.network_type,
            x=x,
            layer_sizes=self.layer_sizes,
        )
        for layer in self.layers:
            carry = layer(carry, ctx)
        return carry.z


def state_initializer(
    network_type: str,
    x: Matrix,
    layer_sizes: Tuple[int, ...] | None,
) -> tuple[Ctx, Carry]:
    """ネットワークの初期状態を作成します。"""
    if network_type == "standard":
        ctx = StandardCtx()
        carry = StandardCarry(z=x)
        return ctx, carry

    if network_type == "icnn":
        if layer_sizes is None or len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements for icnn.")
        ctx = IcnnCtx(x=x)
        carry = IcnnCarry(z=x)
        return ctx, carry

    raise ValueError(f"Unsupported network type: {network_type}")


def build_neuralnetwork(
    network_type: str,
    layer_sizes: Tuple[int,...],
    activation: str,

    random_key: Scalar,
)->NeuralNetwork:
    """標準的な NN または ICNN を構築するファクトリ関数。"""
    def identity(x: Matrix) -> Matrix:
        return x

    activations_dict = {
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid,
        "softplus": jax.nn.softplus,
        "identity": identity,
    }
    if activation not in activations_dict:
        raise ValueError(f"Unsupported activation: {activation}")
    activation_fn = activations_dict[activation]

    layers = []
    key = random_key
    if network_type == "standard":
        for i in range(len(layer_sizes) - 1):
            layer, key = standardNN_layer_factory(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                activation=activation_fn,
                random_key=key,
            )
            layers.append(layer)
        return NeuralNetwork(
            layers=tuple(layers),
            network_type=network_type,
            layer_sizes=layer_sizes,
        )

    elif network_type == "icnn":
        x_dim = layer_sizes[0]
        for i in range(1, len(layer_sizes)):
            layer, key = icnn_layer_factory(
                input_dim=layer_sizes[i - 1],
                output_dim=layer_sizes[i],
                x_dim=x_dim,
                activation=activation_fn,
                random_key=key,
            )
            layers.append(layer)
        return NeuralNetwork(
            layers=tuple(layers),
            network_type=network_type,
            layer_sizes=layer_sizes,
        )

    else:
        raise ValueError(f"Unsupported network type: {network_type}")
