from __future__ import annotations

from ..base import (
    ScalarFn,
    VectorFn,
    Vector,
)
from typing import Protocol

class OptimizeProblem(Protocol):
    objective:ScalarFn #min f
    variable_dim: int

class OptimizeProblemWithConstraint(OptimizeProblem, Protocol):
    # objective:ScalarFn #min f
    constraint_eq : VectorFn #c_eq==0
    constraint_ineq : VectorFn #c_ineq<=0

    # variable_dim: int
    constraint_eq_dim: int
    constraint_ineq_dim: int


class OptimizeProblemState(Protocol):
    x : Vector

class OptimizeProblemStateWithConstraint(OptimizeProblemState, Protocol):
    # x : Vector

    lam_eq : Vector
    lam_ineq : Vector
    slack : Vector

__all__ = [
    "OptimizeProblem",
    "OptimizeProblemState",
    "OptimizeProblemWithConstraint",
    "OptimizeProblemStateWithConstraint",
    
]