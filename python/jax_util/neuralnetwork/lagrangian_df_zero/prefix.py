"""Build the frozen prefix state for suffix solves."""

from __future__ import annotations

from ...base import Matrix
from ..layer_utils import StandardCarry, StandardCtx
from ..neuralnetwork import NeuralNetwork
from .types import PrefixTape


def build_prefix_tape(network: NeuralNetwork, x: Matrix, layer_index: int) -> PrefixTape:
    """Build the state right before the suffix starts."""
    if network.network_type != "standard":
        raise NotImplementedError("lagrangian_df_zero v1 supports only standard networks.")
    if layer_index < 0 or layer_index >= len(network.layers):
        raise ValueError("layer_index must point to an existing suffix start layer.")

    carry = StandardCarry(z=x)
    ctx = StandardCtx()
    for layer in network.layers[:layer_index]:
        carry = layer(carry, ctx)
    return PrefixTape(layer_index=layer_index, ctx=ctx, z_prefix=carry.z)


def suffix_input(prefix_tape: PrefixTape) -> StandardCarry:
    """Return the initial carry for suffix rollout."""
    return StandardCarry(z=prefix_tape.z_prefix)


__all__ = [
    "build_prefix_tape",
    "suffix_input",
]
