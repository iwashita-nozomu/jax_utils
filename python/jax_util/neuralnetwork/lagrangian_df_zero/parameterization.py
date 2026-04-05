"""Pack and unpack layer parameters and suffix variables."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from ...base import Matrix, Vector
from ..layer_utils import module_to_vector, vector_to_module
from ..protocols import NeuralNetworkLayer
from .types import BlockSlice, LayerVectorization, SuffixVariables, VariableLayout


def flatten_layer_params(layer: NeuralNetworkLayer) -> tuple[Vector, LayerVectorization]:
    """Flatten one layer and keep the rebuild metadata."""
    flat_params, rebuild_state = module_to_vector(layer)
    vectorization = LayerVectorization(
        rebuild_state=rebuild_state,
        param_size=int(flat_params.shape[0]),
        output_shape=(0,),
    )
    return flat_params, vectorization


def rebuild_layer(vector: Vector, layer_vectorization: LayerVectorization) -> NeuralNetworkLayer:
    """Rebuild one layer from a flat parameter vector."""
    return vector_to_module(vector, layer_vectorization.rebuild_state)


def build_layer_vectorizations(
    layers: Sequence[NeuralNetworkLayer],
    output_shapes: Sequence[tuple[int, ...]],
) -> tuple[LayerVectorization, ...]:
    """Build vectorization metadata for the suffix layers."""
    if len(layers) != len(output_shapes):
        raise ValueError("layers and output_shapes must have the same length.")

    vectorizations: list[LayerVectorization] = []
    for layer, output_shape in zip(layers, output_shapes, strict=True):
        flat_params, vectorization = flatten_layer_params(layer)
        vectorizations.append(
            LayerVectorization(
                rebuild_state=vectorization.rebuild_state,
                param_size=int(flat_params.shape[0]),
                output_shape=output_shape,
            )
        )
    return tuple(vectorizations)


def build_variable_layout(
    param_sizes: Sequence[int],
    output_shapes: Sequence[tuple[int, ...]],
) -> VariableLayout:
    """Build the packed block layout for the suffix variables."""
    if len(param_sizes) != len(output_shapes):
        raise ValueError("param_sizes and output_shapes must have the same length.")

    current = 0
    theta_slices: list[BlockSlice] = []
    z_slices: list[BlockSlice] = []
    p_slices: list[BlockSlice] = []

    for param_size in param_sizes:
        theta_slices.append(BlockSlice(current, current + param_size, (param_size,)))
        current += param_size

    for output_shape in output_shapes:
        block_size = output_shape[0] * output_shape[1]
        z_slices.append(BlockSlice(current, current + block_size, output_shape))
        current += block_size

    for output_shape in output_shapes:
        block_size = output_shape[0] * output_shape[1]
        p_slices.append(BlockSlice(current, current + block_size, output_shape))
        current += block_size

    return VariableLayout(
        theta_slices=tuple(theta_slices),
        z_slices=tuple(z_slices),
        p_slices=tuple(p_slices),
        total_size=current,
    )


def pack_variables(variables: SuffixVariables, layout: VariableLayout) -> Vector:
    """Pack suffix variables into one flat vector."""
    if len(variables.theta_tail) != len(layout.theta_slices):
        raise ValueError("theta_tail length does not match the layout.")
    if len(variables.z_tail) != len(layout.z_slices):
        raise ValueError("z_tail length does not match the layout.")
    if len(variables.p_tail) != len(layout.p_slices):
        raise ValueError("p_tail length does not match the layout.")

    blocks = [
        *(theta.reshape(-1) for theta in variables.theta_tail),
        *(z.reshape(-1) for z in variables.z_tail),
        *(p.reshape(-1) for p in variables.p_tail),
    ]
    if not blocks:
        return jnp.zeros((0,))
    return jnp.concatenate(blocks, axis=0)


def unpack_variables(vector: Vector, layout: VariableLayout) -> SuffixVariables:
    """Unpack one flat vector into suffix variables."""
    if int(vector.shape[0]) != layout.total_size:
        raise ValueError("packed vector size does not match the layout.")

    theta_tail = tuple(
        vector[block.as_slice()].reshape(block.shape) for block in layout.theta_slices
    )
    z_tail = tuple(vector[block.as_slice()].reshape(block.shape) for block in layout.z_slices)
    p_tail = tuple(vector[block.as_slice()].reshape(block.shape) for block in layout.p_slices)
    return SuffixVariables(theta_tail=theta_tail, z_tail=z_tail, p_tail=p_tail)


def residual_block_norms(
    primal_blocks: Sequence[Matrix],
    adjoint_blocks: Sequence[Matrix],
    theta_blocks: Sequence[Vector],
) -> dict[str, float]:
    """Compute blockwise norms for the split residual."""
    def _norm(blocks: Sequence[Matrix | Vector]) -> float:
        if not blocks:
            return 0.0
        flat = jnp.concatenate([block.reshape(-1) for block in blocks], axis=0)
        return float(jnp.linalg.norm(flat))

    return {
        "primal": _norm(primal_blocks),
        "adjoint": _norm(adjoint_blocks),
        "parameter": _norm(theta_blocks),
    }


__all__ = [
    "flatten_layer_params",
    "rebuild_layer",
    "build_layer_vectorizations",
    "build_variable_layout",
    "pack_variables",
    "unpack_variables",
    "residual_block_norms",
]
