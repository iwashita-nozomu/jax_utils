from __future__ import annotations

from typing import Callable,Tuple

from ..base import *

from .protocols import *
from .layer_utils import *
import equinox as eqx
import jax
from jax import numpy as jnp

class NeuralNetwork(eqx.Module):
    Layers: Tuple[NeuralNetworkLayer,...] = eqx.field(static=True)
    network_type: str = eqx.field(static=True)

    def __call__(self,x:Matrix):

        if self.network_type == "standard":
            ctx = StandardCtx()
        elif self.network_type == "icnn":
            ctx = IcnnCtx(x=x)
        else:
            raise ValueError(f"Unsupported network type: {self.network_type}")
        
        for layer in self.Layers:
            x,ctx = layer(x,ctx)
            
        return x
        
    

    


def build_neuralnetwork(
    networktype: str,
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
    if networktype == "standard":
        for i in range(len(layer_sizes) - 1):
            layer, key = standardNN_layer_factory(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                activation=activation_fn,
                random_key=key,
            )
            layers.append(layer)
        
        def nn_forward(x: Matrix) -> Matrix:
            z = x
            for layer in layers:
                z = layer(z)
            return z
        
        return NeuralNetwork(
            Layers=tuple(layers),
            forward=nn_forward,
        )

    elif networktype == "icnn":
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
        
        def icnn_forward(x: Matrix) -> Matrix:
            z = jnp.zeros((layer_sizes[1], x.shape[1]), dtype=DEFAULT_DTYPE)
            for layer in layers:
                z = layer(z, x)
            return z
        
        return NeuralNetwork(
            Layers=tuple(layers),
            forward=icnn_forward,
        )

    else:
        raise ValueError(f"Unsupported network type: {networktype}")
