"""Test residual construction for lagrangian_df_zero."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_util.neuralnetwork.lagrangian_df_zero import (
    adjoint_residual,
    build_stationary_problem,
    initialize_suffix_variables,
    mean_squared_output_loss,
    primal_residual,
    theta_residual,
)
from jax_util.neuralnetwork.lagrangian_df_zero.toy import build_standard_identity_network


def test_warm_start_makes_primal_and_adjoint_residuals_zero() -> None:
    """Make the rollout and adjoint residuals vanish at warm start."""
    model = build_standard_identity_network((2, 3, 2, 1), jax.random.PRNGKey(1))
    x = jnp.ones((2, 3))
    y = jnp.zeros((1, 3))

    problem = build_stationary_problem(
        network=model,
        x=x,
        y=y,
        layer_index=1,
        loss_from_output=mean_squared_output_loss,
    )
    variables = initialize_suffix_variables(problem)

    primal = primal_residual(variables, problem.prefix_tape, problem.layer_vectorizations)
    adjoint = adjoint_residual(
        variables=variables,
        prefix_tape=problem.prefix_tape,
        vectorizations=problem.layer_vectorizations,
        loss_from_output=problem.loss_from_output,
        target=problem.target,
    )
    theta = theta_residual(variables, problem.prefix_tape, problem.layer_vectorizations)

    assert problem.prefix_tape.z_prefix.shape == (3, 3)
    assert max(float(jnp.linalg.norm(block)) for block in primal) < 1.0e-6
    assert max(float(jnp.linalg.norm(block)) for block in adjoint) < 1.0e-6
    assert len(theta) == len(variables.theta_tail)
