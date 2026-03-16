from __future__ import annotations

from ..base import (
    Vector,
    OptimizationProblem,
    ConstraintedOptimizationProblem,
    OptimizationState,
    ConstrainedOptimizationState,
)
from typing import Protocol

class VectorOptimizationProblem(OptimizationProblem[Vector], Protocol):
    variable_dim: int
    ... #note:: 目的関数はベクトル空間上の関数であることに注意してください。

class ConstrainedVectorOptimizationProblem(ConstraintedOptimizationProblem[Vector,Vector,Vector],
                                                  VectorOptimizationProblem,
                                                    Protocol):
    # variable_dim: int
    constraint_eq_dim: int
    constraint_ineq_dim: int
    ... #note:: 目的関数,制約関数はベクトル空間上の関数であることに注意してください。


class VectorOptimizationState(OptimizationState[Vector], Protocol):
    ... #note:: xはベクトル空間上の変数であることに注意してください。

class ConstrainedVectorOptimizationState(ConstrainedOptimizationState[Vector, Vector],
                                         VectorOptimizationState,
                                          Protocol):
    ... #note:: xはベクトル空間上の変数であることに注意してください。lam_eq, lam_ineqは双対空間上の変数であることに注意してください。

__all__ = [
    "VectorOptimizationProblem",
    "VectorOptimizationState",
    "ConstrainedVectorOptimizationProblem",
    "ConstrainedVectorOptimizationState",

]
