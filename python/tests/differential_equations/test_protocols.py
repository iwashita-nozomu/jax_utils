"""Tests for differential-equation protocols and tagged terms."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax_util.differential_equations import (
    DifferentialEquationOperator,
    DifferentialEquationTerm,
    StateFunction,
)


class IdentityResidualOperator:
    """Test helper that forwards the input function as residual."""

    def __call__(self, f: StateFunction, /) -> StateFunction:
        """Return the input function unchanged."""
        return f


def test_equation_tag_assumes_zero_rhs() -> None:
    """An equation tag should imply zero right-hand-side semantics."""
    term = DifferentialEquationTerm(
        name="governing_equation",
        operator=IdentityResidualOperator(),
        tags=("equation", "boundary_condition"),
    )

    assert term.assumes_zero_rhs is True


def test_term_requires_at_least_one_tag() -> None:
    """A differential-equation term should require at least one tag."""
    with pytest.raises(ValueError, match="at least one tag"):
        DifferentialEquationTerm(
            name="untagged_term",
            operator=IdentityResidualOperator(),
            tags=(),
        )
def test_operator_protocol_can_return_residual_function() -> None:
    """A structural operator should satisfy the runtime protocol."""
    operator = IdentityResidualOperator()

    assert isinstance(operator, DifferentialEquationOperator)

    residual = operator(lambda x: x)

    assert jnp.allclose(residual(jnp.asarray([1.0, 2.0])), jnp.asarray([1.0, 2.0]))
