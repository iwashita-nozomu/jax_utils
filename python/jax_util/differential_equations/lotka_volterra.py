"""Lotka-Volterra initial-value problem.

This module follows the one-problem-per-file convention for
``jax_util.differential_equations`` and exposes importable symbols such as
``jax_util.differential_equations.lotka_volterra.problem``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ..base import DEFAULT_DTYPE, Vector
from .problem import DifferentialEquationProblem
from .protocols import (
    DifferentialEquationOperator,
    DifferentialEquationTerm,
    ResidualFunction,
    StateFunction,
)


def make_governing_equation_operator(
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    gamma: float = 3.0,
    delta: float = 1.0,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> DifferentialEquationOperator:
    """Build the Lotka-Volterra governing-equation residual operator."""
    alpha_value = jnp.asarray(alpha, dtype=dtype)
    beta_value = jnp.asarray(beta, dtype=dtype)
    gamma_value = jnp.asarray(gamma, dtype=dtype)
    delta_value = jnp.asarray(delta, dtype=dtype)

    def operator(f: StateFunction, /) -> ResidualFunction:
        def residual(t: Vector, /) -> Vector:
            t_value = jnp.asarray(t, dtype=dtype)
            state = jnp.asarray(f(t_value), dtype=dtype)
            state_dt = jnp.asarray(jax.jacfwd(f)(t_value), dtype=dtype)[:, 0]
            prey = state[0]
            predator = state[1]
            rhs = jnp.asarray(
                [
                    alpha_value * prey - beta_value * prey * predator,
                    delta_value * prey * predator - gamma_value * predator,
                ],
                dtype=dtype,
            )
            return state_dt - rhs

        return residual

    return operator


def make_initial_condition_operator(
    *,
    initial_state: tuple[float, float] = (1.0, 0.0),
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> DifferentialEquationOperator:
    """Build the canonical initial-condition residual operator."""
    initial_state_value = jnp.asarray(initial_state, dtype=dtype)

    def operator(f: StateFunction, /) -> ResidualFunction:
        def residual(t: Vector, /) -> Vector:
            t_value = jnp.asarray(t, dtype=dtype)
            return jnp.asarray(f(t_value), dtype=dtype) - initial_state_value

        return residual

    return operator


def make_problem(
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    gamma: float = 3.0,
    delta: float = 1.0,
    initial_state: tuple[float, float] = (1.0, 0.0),
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> DifferentialEquationProblem:
    """Build the canonical Lotka-Volterra problem metadata."""
    governing_equation_term = DifferentialEquationTerm(
        name="governing_equation",
        operator=make_governing_equation_operator(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            dtype=dtype,
        ),
        tags=("equation",),
        description="Lotka-Volterra residual L(f)=0.",
    )
    initial_condition_term = DifferentialEquationTerm(
        name="initial_condition",
        operator=make_initial_condition_operator(
            initial_state=initial_state,
            dtype=dtype,
        ),
        tags=("initial_condition",),
        description="Canonical initial state at t=0.",
    )
    return DifferentialEquationProblem(
        name="lotka_volterra",
        equation_kind="ode",
        condition_kind="initial_value",
        state_dimension=2,
        time_interval=(0.0, 10.0),
        description="Predator-prey benchmark used in classical dynamical-systems literature.",
        tags=("paper_benchmark", "nonlinear", "predator_prey"),
        references=(
            "Lotka (1925), Elements of Physical Biology.",
            "Volterra (1926), Variazioni e fluttuazioni del numero "
            "d'individui in specie animali conviventi.",
        ),
        terms=(governing_equation_term, initial_condition_term),
    )


problem = make_problem()
governing_equation_term = problem.terms[0]
initial_condition_term = problem.terms[1]


__all__ = [
    "make_governing_equation_operator",
    "make_initial_condition_operator",
    "make_problem",
    "governing_equation_term",
    "initial_condition_term",
    "problem",
]
