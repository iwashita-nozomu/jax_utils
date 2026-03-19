from __future__ import annotations

from ..base import (
    ConstrainedOptimizationProblem,
    ConstrainedOptimizationState,
    Vector,
    OptimizationProblem,
    OptimizationState,
)
from typing import Protocol


class VectorOptimizationProblem(OptimizationProblem[Vector], Protocol):
    variable_dim: int
    ...


class ConstrainedVectorOptimizationProblem(
    ConstrainedOptimizationProblem[Vector, Vector, Vector],
    VectorOptimizationProblem,
    Protocol,
):
    constraint_eq_dim: int
    constraint_ineq_dim: int
    ...


class VectorOptimizationState(OptimizationState[Vector], Protocol): ...


class ConstrainedVectorOptimizationState(
    ConstrainedOptimizationState[Vector, Vector],
    VectorOptimizationState,
    Protocol,
): ...


__all__ = [
    "VectorOptimizationProblem",
    "VectorOptimizationState",
    "ConstrainedVectorOptimizationProblem",
    "ConstrainedVectorOptimizationState",
]
