"""Protocols for autodiff-friendly differential-equation operators.

Concrete problems are expected to build their terms with JAX autodiff
primitives such as ``jax.grad``, ``jax.jvp``, ``jax.jacfwd``, and
``jax.jacrev`` instead of manual finite-difference approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from ..base import Vector


@runtime_checkable
class StateFunction(Protocol):
    """Unknown function in a differential equation."""

    def __call__(self, x: Vector, /) -> Vector:
        """Evaluate the unknown function at one input point."""
        ...


@runtime_checkable
class ResidualFunction(Protocol):
    """Residual function produced by a differential operator."""

    def __call__(self, x: Vector, /) -> Vector:
        """Evaluate the residual at one input point."""
        ...


DifferentialEquationTag = Literal[
    "equation",
    "initial_condition",
    "boundary_condition",
    "terminal_condition",
    "constraint",
    "forcing",
    "observation",
    "other",
]


@runtime_checkable
class DifferentialEquationOperator(Protocol):
    """Map an unknown function to its residual function.

    The returned residual function is expected to remain compatible with
    JAX autodiff and transforms when the input function and implementation
    are written in JAX-friendly form. Concrete operators should be
    assembled from autodiff primitives rather than hand-coded numerical
    differentiation.
    """

    def __call__(self, f: StateFunction, /) -> ResidualFunction:
        """Build the residual function for the given unknown function."""
        ...


@dataclass(frozen=True, slots=True)
class DifferentialEquationTerm:
    """One tagged operator term inside a differential-equation problem."""

    name: str
    operator: DifferentialEquationOperator
    tags: tuple[DifferentialEquationTag, ...] = ("equation",)
    description: str = ""

    def __post_init__(self) -> None:
        """Validate tag metadata for one differential-equation term."""
        if not self.name:
            raise ValueError("Term name must not be empty.")
        if not self.tags:
            raise ValueError("Differential-equation term must have at least one tag.")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("Differential-equation term tags must be unique.")

    @property
    def assumes_zero_rhs(self) -> bool:
        """Whether this term is interpreted as residual == 0."""
        return "equation" in self.tags


__all__ = [
    "StateFunction",
    "ResidualFunction",
    "DifferentialEquationTag",
    "DifferentialEquationOperator",
    "DifferentialEquationTerm",
]
