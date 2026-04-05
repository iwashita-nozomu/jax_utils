"""Define internal data structures for DF=0 stationary problems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ...base import Matrix, Scalar, Vector
from ..layer_utils import RebuildState, StandardCtx
from ..protocols import NeuralNetworkLayer


@dataclass(frozen=True)
class LayerVectorization:
    """Store vectorization metadata for one layer."""

    rebuild_state: RebuildState[NeuralNetworkLayer]
    param_size: int
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class PrefixTape:
    """Store the frozen prefix state before the suffix starts."""

    layer_index: int
    ctx: StandardCtx
    z_prefix: Matrix


@dataclass(frozen=True)
class BlockSlice:
    """Describe one packed block inside a flat vector."""

    start: int
    stop: int
    shape: tuple[int, ...]

    def as_slice(self) -> slice:
        """Return the block interval as a built-in slice."""
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class VariableLayout:
    """Store the block layout of the packed variables."""

    theta_slices: tuple[BlockSlice, ...]
    z_slices: tuple[BlockSlice, ...]
    p_slices: tuple[BlockSlice, ...]
    total_size: int


@dataclass(frozen=True)
class SuffixVariables:
    """Bundle the variables of the suffix stationary problem."""

    theta_tail: tuple[Vector, ...]
    z_tail: tuple[Matrix, ...]
    p_tail: tuple[Matrix, ...]


@dataclass(frozen=True)
class StationaryProblem:
    """Bundle the fixed data of the suffix stationary problem."""

    prefix_tape: PrefixTape
    layer_vectorizations: tuple[LayerVectorization, ...]
    layout: VariableLayout
    target: Matrix
    loss_from_output: Callable[[Matrix, Matrix], Scalar]
    initial_theta_tail: tuple[Vector, ...]
    initial_z_tail: tuple[Matrix, ...]


@dataclass(frozen=True)
class StationarySolveState:
    """Store one packed iterate during the Newton solve."""

    packed_variables: Vector
    iteration: int
    residual_norm: Scalar


@dataclass(frozen=True)
class StationarySolveInfo:
    """Summarize the dense stationary solve."""

    converged: bool
    iterations: int
    residual_norm: float
    residual_breakdown: dict[str, float]


__all__ = [
    "LayerVectorization",
    "PrefixTape",
    "BlockSlice",
    "VariableLayout",
    "SuffixVariables",
    "StationaryProblem",
    "StationarySolveState",
    "StationarySolveInfo",
]
