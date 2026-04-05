"""Test the toy stationary solver for lagrangian_df_zero."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax_util.neuralnetwork.lagrangian_df_zero import (
    build_stationary_problem,
    initialize_suffix_variables,
    kkt_residual,
    mean_squared_output_loss,
    solve_stationary_suffix,
)
from jax_util.neuralnetwork.lagrangian_df_zero.toy import build_standard_identity_network


def test_zero_network_with_zero_target_has_zero_stationary_residual() -> None:
    """Produce a zero KKT residual for the trivial zero problem."""
    model = build_standard_identity_network((2, 1), jax.random.PRNGKey(2))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    zero_params = jax.tree_util.tree_map(jnp.zeros_like, params)
    zero_model = eqx.combine(zero_params, static)

    x = jnp.array([[1.0], [-1.0]])
    y = jnp.zeros((1, 1))
    problem = build_stationary_problem(
        network=zero_model,
        x=x,
        y=y,
        layer_index=0,
        loss_from_output=mean_squared_output_loss,
    )
    variables = initialize_suffix_variables(problem)
    residual = kkt_residual(variables, problem)

    assert float(jnp.linalg.norm(residual)) < 1.0e-7


def test_dense_newton_solver_reduces_kkt_residual_norm() -> None:
    """Reduce the KKT residual norm on a one-layer toy problem."""
    model = build_standard_identity_network((2, 1), jax.random.PRNGKey(3))
    x = jnp.array([[1.0], [-1.0]])
    y = jnp.zeros((1, 1))
    problem = build_stationary_problem(
        network=model,
        x=x,
        y=y,
        layer_index=0,
        loss_from_output=mean_squared_output_loss,
    )
    init_variables = initialize_suffix_variables(problem)
    initial_norm = float(jnp.linalg.norm(kkt_residual(init_variables, problem)))

    _, info = solve_stationary_suffix(
        problem=problem,
        init_variables=init_variables,
        tol=jnp.asarray(1.0e-8),
        maxiter=4,
    )

    assert info.residual_norm < initial_norm
