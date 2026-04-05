"""Core metadata for one differential-equation problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .protocols import DifferentialEquationTerm

EquationKind = Literal["ode", "pde", "dae", "integro_differential", "other"]
ConditionKind = Literal["initial_value", "boundary_value", "terminal_value", "mixed", "other"]


@dataclass(frozen=True, slots=True)
class DifferentialEquationProblem:
    """Metadata for one differential-equation problem."""

    name: str
    equation_kind: EquationKind
    condition_kind: ConditionKind
    state_dimension: int
    spatial_dimension: int = 0
    time_interval: tuple[float, float] | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    terms: tuple[DifferentialEquationTerm, ...] = ()

    def __post_init__(self) -> None:
        """Validate problem metadata and tagged operator terms."""
        if not self.name:
            raise ValueError("Problem name must not be empty.")
        if self.state_dimension < 1:
            raise ValueError("state_dimension must be positive.")
        if self.spatial_dimension < 0:
            raise ValueError("spatial_dimension must be non-negative.")
        if self.time_interval is not None:
            start, end = self.time_interval
            if end <= start:
                raise ValueError("time_interval must satisfy start < end.")
        term_names = [term.name for term in self.terms]
        if len(term_names) != len(set(term_names)):
            raise ValueError("Term names must be unique inside one problem.")

    @property
    def equation_terms(self) -> tuple[DifferentialEquationTerm, ...]:
        """Return terms interpreted as residual equations."""
        return tuple(term for term in self.terms if term.assumes_zero_rhs)

    @property
    def equation_term_names(self) -> tuple[str, ...]:
        """Return the names of equation terms in this problem."""
        return tuple(term.name for term in self.equation_terms)


__all__ = [
    "ConditionKind",
    "DifferentialEquationProblem",
    "EquationKind",
]
