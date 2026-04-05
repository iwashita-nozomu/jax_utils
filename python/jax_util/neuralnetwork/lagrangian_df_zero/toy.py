"""Define toy losses and toy networks for lagrangian_df_zero."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ...base import Matrix, Scalar
from ..neuralnetwork import NeuralNetwork, build_neuralnetwork


def mean_squared_output_loss(output: Matrix, target: Matrix) -> Scalar:
    """Return the half squared output loss."""
    return 0.5 * jnp.sum(jnp.square(output - target))


def build_standard_identity_network(
    layer_sizes: tuple[int, ...],
    random_key: jax.Array,
) -> NeuralNetwork:
    """Build a standard network with identity activations."""
    return build_neuralnetwork(
        network_type="standard",
        layer_sizes=layer_sizes,
        activation="identity",
        random_key=random_key,
    )


__all__ = [
    "mean_squared_output_loss",
    "build_standard_identity_network",
]
