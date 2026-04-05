"""Tests for differential-equation problem metadata."""

from __future__ import annotations

import pytest
from jax_util.differential_equations import DifferentialEquationProblem, DifferentialEquationTerm


class _IdentityResidualOperator:
    """Test helper that forwards the input function as residual."""

    def __call__(self, f, /):
        """Return the input function unchanged."""
        return f


def test_problem_rejects_duplicate_term_names() -> None:
    """Problem metadata should reject duplicate term names."""
    term = DifferentialEquationTerm(
        name="governing_equation",
        operator=_IdentityResidualOperator(),
        tags=("equation",),
    )

    with pytest.raises(ValueError, match="unique"):
        DifferentialEquationProblem(
            name="duplicate_term_problem",
            equation_kind="ode",
            condition_kind="initial_value",
            state_dimension=1,
            time_interval=(0.0, 1.0),
            terms=(term, term),
        )


def test_problem_exposes_equation_term_names() -> None:
    """Problems should expose the ordered names of equation terms."""
    term = DifferentialEquationTerm(
        name="governing_equation",
        operator=_IdentityResidualOperator(),
        tags=("equation",),
    )
    problem = DifferentialEquationProblem(
        name="lotka_volterra",
        equation_kind="ode",
        condition_kind="initial_value",
        state_dimension=2,
        time_interval=(0.0, 10.0),
        terms=(term,),
    )

    assert problem.equation_term_names == ("governing_equation",)
