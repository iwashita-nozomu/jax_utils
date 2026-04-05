"""Core types for differential equation problem catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


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

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True)
class ProblemSet:
    """Named collection of differential-equation problems."""

    name: str
    problems: tuple[DifferentialEquationProblem, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Problem set name must not be empty.")
        names = [problem.name for problem in self.problems]
        if len(names) != len(set(names)):
            raise ValueError("Problem names must be unique inside one problem set.")

    @property
    def problem_names(self) -> tuple[str, ...]:
        return tuple(problem.name for problem in self.problems)


def build_problem_set(
    name: str,
    problems: Iterable[DifferentialEquationProblem],
) -> ProblemSet:
    """Build a validated problem set from an iterable."""

    return ProblemSet(name=name, problems=tuple(problems))
