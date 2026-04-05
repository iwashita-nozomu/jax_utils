"""One-dimensional heat equation with homogeneous Dirichlet boundaries.

This module follows the one-problem-per-file convention for
``jax_util.differential_equations`` and exposes importable symbols such as
``jax_util.differential_equations.heat_1d_dirichlet.problem``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ..base import DEFAULT_DTYPE, Scalar, Vector
from .problem import DifferentialEquationProblem
from .protocols import (
    DifferentialEquationOperator,
    DifferentialEquationTerm,
    ResidualFunction,
    StateFunction,
)


def make_governing_equation_operator(
    *,
    diffusivity: float = 1.0,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> DifferentialEquationOperator:
    """Build the heat-equation residual operator."""
    diffusivity_value = jnp.asarray(diffusivity, dtype=dtype)

    def operator(f: StateFunction, /) -> ResidualFunction:
        def scalar_state(xt: Vector, /) -> Scalar:
            return jnp.asarray(f(xt), dtype=dtype)[0]

        state_grad = jax.grad(scalar_state)

        def state_x(xt: Vector, /) -> Scalar:
            return state_grad(xt)[0]

        def residual(xt: Vector, /) -> Vector:
            xt_value = jnp.asarray(xt, dtype=dtype)
            grad_value = state_grad(xt_value)
            state_t = grad_value[1]
            state_xx = jax.grad(state_x)(xt_value)[0]
            return jnp.asarray([state_t - diffusivity_value * state_xx], dtype=dtype)

        return residual

    return operator


def make_initial_condition_operator(
    *,
    amplitude: float = 1.0,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> DifferentialEquationOperator:
    """Build the canonical sinusoidal initial-condition operator."""
    amplitude_value = jnp.asarray(amplitude, dtype=dtype)

    def operator(f: StateFunction, /) -> ResidualFunction:
        def residual(xt: Vector, /) -> Vector:
            xt_value = jnp.asarray(xt, dtype=dtype)
            target = amplitude_value * jnp.sin(jnp.pi * xt_value[0])
            return jnp.asarray(f(xt_value), dtype=dtype) - jnp.asarray([target], dtype=dtype)

        return residual

    return operator


def make_boundary_condition_operator(
    *,
    boundary_value: float = 0.0,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> DifferentialEquationOperator:
    """Build a homogeneous Dirichlet boundary-condition operator."""
    boundary_value_array = jnp.asarray([boundary_value], dtype=dtype)

    def operator(f: StateFunction, /) -> ResidualFunction:
        def residual(xt: Vector, /) -> Vector:
            xt_value = jnp.asarray(xt, dtype=dtype)
            return jnp.asarray(f(xt_value), dtype=dtype) - boundary_value_array

        return residual

    return operator


def make_problem(
    *,
    diffusivity: float = 1.0,
    amplitude: float = 1.0,
    boundary_value: float = 0.0,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> DifferentialEquationProblem:
    """Build the canonical one-dimensional heat-equation problem metadata."""
    governing_equation_term = DifferentialEquationTerm(
        name="governing_equation",
        operator=make_governing_equation_operator(
            diffusivity=diffusivity,
            dtype=dtype,
        ),
        tags=("equation",),
        description="Heat-equation residual u_t - kappa u_xx.",
    )
    initial_condition_term = DifferentialEquationTerm(
        name="initial_condition",
        operator=make_initial_condition_operator(
            amplitude=amplitude,
            dtype=dtype,
        ),
        tags=("initial_condition",),
        description="Sinusoidal initial profile at t=0.",
    )
    left_boundary_condition_term = DifferentialEquationTerm(
        name="left_boundary_condition",
        operator=make_boundary_condition_operator(
            boundary_value=boundary_value,
            dtype=dtype,
        ),
        tags=("boundary_condition",),
        description="Left Dirichlet boundary condition at x=0.",
    )
    right_boundary_condition_term = DifferentialEquationTerm(
        name="right_boundary_condition",
        operator=make_boundary_condition_operator(
            boundary_value=boundary_value,
            dtype=dtype,
        ),
        tags=("boundary_condition",),
        description="Right Dirichlet boundary condition at x=1.",
    )
    return DifferentialEquationProblem(
        name="heat_1d_dirichlet",
        equation_kind="pde",
        condition_kind="boundary_value",
        state_dimension=1,
        spatial_dimension=1,
        time_interval=(0.0, 1.0),
        description="One-dimensional heat equation with homogeneous Dirichlet boundaries.",
        tags=("paper_benchmark", "parabolic", "dirichlet"),
        references=(
            "Raissi, Perdikaris, and Karniadakis (2019), Physics-informed neural networks.",
        ),
        terms=(
            governing_equation_term,
            initial_condition_term,
            left_boundary_condition_term,
            right_boundary_condition_term,
        ),
    )


problem = make_problem()
governing_equation_term = problem.terms[0]
initial_condition_term = problem.terms[1]
left_boundary_condition_term = problem.terms[2]
right_boundary_condition_term = problem.terms[3]


__all__ = [
    "make_governing_equation_operator",
    "make_initial_condition_operator",
    "make_boundary_condition_operator",
    "make_problem",
    "governing_equation_term",
    "initial_condition_term",
    "left_boundary_condition_term",
    "right_boundary_condition_term",
    "problem",
]
