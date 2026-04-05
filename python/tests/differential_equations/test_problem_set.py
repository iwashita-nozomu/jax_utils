from __future__ import annotations

import pytest

from jax_util.differential_equations import DifferentialEquationProblem, build_problem_set


def test_build_problem_set_preserves_problem_order() -> None:
    problem_a = DifferentialEquationProblem(
        name="heat_1d_dirichlet",
        equation_kind="pde",
        condition_kind="boundary_value",
        state_dimension=1,
        spatial_dimension=1,
        time_interval=(0.0, 1.0),
        tags=("parabolic", "dirichlet"),
    )
    problem_b = DifferentialEquationProblem(
        name="lotka_volterra_ivp",
        equation_kind="ode",
        condition_kind="initial_value",
        state_dimension=2,
        time_interval=(0.0, 10.0),
        tags=("nonlinear", "ivp"),
    )

    problem_set = build_problem_set("starter_set", [problem_a, problem_b])

    assert problem_set.problem_names == ("heat_1d_dirichlet", "lotka_volterra_ivp")


def test_build_problem_set_rejects_duplicate_problem_names() -> None:
    problem = DifferentialEquationProblem(
        name="duplicate_name",
        equation_kind="ode",
        condition_kind="initial_value",
        state_dimension=1,
        time_interval=(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="unique"):
        build_problem_set("duplicate_set", [problem, problem])
