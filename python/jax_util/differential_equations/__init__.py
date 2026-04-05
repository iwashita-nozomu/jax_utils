"""Differential-equation problem catalogs."""

from .problem import (
    ConditionKind,
    DifferentialEquationProblem,
    EquationKind,
)
from .protocols import (
    DifferentialEquationOperator,
    DifferentialEquationTag,
    DifferentialEquationTerm,
    ResidualFunction,
    StateFunction,
)

__all__ = [
    "ConditionKind",
    "DifferentialEquationProblem",
    "DifferentialEquationOperator",
    "DifferentialEquationTag",
    "DifferentialEquationTerm",
    "EquationKind",
    "ResidualFunction",
    "StateFunction",
]
