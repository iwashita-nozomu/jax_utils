"""Tests for per-problem differential-equation modules."""

from __future__ import annotations

import jax.numpy as jnp
from jax_util.differential_equations.heat_1d_dirichlet import (
    governing_equation_term as heat_governing_equation_term,
)
from jax_util.differential_equations.heat_1d_dirichlet import (
    initial_condition_term as heat_initial_condition_term,
)
from jax_util.differential_equations.heat_1d_dirichlet import (
    left_boundary_condition_term,
    right_boundary_condition_term,
)
from jax_util.differential_equations.heat_1d_dirichlet import (
    problem as heat_problem,
)
from jax_util.differential_equations.lotka_volterra import (
    governing_equation_term as lotka_volterra_governing_equation_term,
)
from jax_util.differential_equations.lotka_volterra import (
    initial_condition_term as lotka_volterra_initial_condition_term,
)
from jax_util.differential_equations.lotka_volterra import (
    problem as lotka_volterra_problem,
)


def test_problem_modules_are_importable_from_direct_paths() -> None:
    """Direct per-problem imports should expose canonical problem objects."""
    assert heat_problem.name == "heat_1d_dirichlet"
    assert lotka_volterra_problem.name == "lotka_volterra"


def test_lotka_volterra_terms_vanish_for_canonical_solution() -> None:
    """The canonical Lotka-Volterra terms should vanish on a simple exact solution."""

    def exact_solution(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([jnp.exp(1.5 * t[0]), 0.0])

    governing_residual = lotka_volterra_governing_equation_term.operator(exact_solution)
    initial_residual = lotka_volterra_initial_condition_term.operator(exact_solution)

    assert jnp.allclose(governing_residual(jnp.asarray([0.25])), jnp.zeros(2))
    assert jnp.allclose(initial_residual(jnp.asarray([0.0])), jnp.zeros(2))


def test_heat_equation_terms_vanish_for_canonical_solution() -> None:
    """Heat-equation terms should vanish on the standard sinusoidal solution."""

    def exact_solution(xt: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [
                jnp.exp(-(jnp.pi**2) * xt[1]) * jnp.sin(jnp.pi * xt[0]),
            ]
        )

    governing_residual = heat_governing_equation_term.operator(exact_solution)
    initial_residual = heat_initial_condition_term.operator(exact_solution)
    left_boundary_residual = left_boundary_condition_term.operator(exact_solution)
    right_boundary_residual = right_boundary_condition_term.operator(exact_solution)

    assert jnp.allclose(governing_residual(jnp.asarray([0.25, 0.1])), jnp.zeros(1))
    assert jnp.allclose(initial_residual(jnp.asarray([0.25, 0.0])), jnp.zeros(1))
    assert jnp.allclose(left_boundary_residual(jnp.asarray([0.0, 0.2])), jnp.zeros(1))
    assert jnp.allclose(right_boundary_residual(jnp.asarray([1.0, 0.2])), jnp.zeros(1))
