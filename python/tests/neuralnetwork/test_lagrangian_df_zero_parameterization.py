"""Test packing and unpacking for lagrangian_df_zero."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_util.neuralnetwork.lagrangian_df_zero import (
    build_stationary_problem,
    initialize_suffix_variables,
    mean_squared_output_loss,
    pack_variables,
    unpack_variables,
)
from jax_util.neuralnetwork.lagrangian_df_zero.toy import build_standard_identity_network


def test_pack_unpack_roundtrip_preserves_suffix_variables() -> None:
    """Keep all suffix blocks unchanged across pack and unpack."""
    model = build_standard_identity_network((2, 3, 1), jax.random.PRNGKey(0))
    x = jnp.ones((2, 4))
    y = jnp.zeros((1, 4))

    problem = build_stationary_problem(
        network=model,
        x=x,
        y=y,
        layer_index=0,
        loss_from_output=mean_squared_output_loss,
    )
    variables = initialize_suffix_variables(problem)
    packed = pack_variables(variables, problem.layout)
    rebuilt = unpack_variables(packed, problem.layout)

    assert packed.shape == (problem.layout.total_size,)
    for lhs, rhs in zip(variables.theta_tail, rebuilt.theta_tail, strict=True):
        assert jnp.allclose(lhs, rhs)
    for lhs, rhs in zip(variables.z_tail, rebuilt.z_tail, strict=True):
        assert jnp.allclose(lhs, rhs)
    for lhs, rhs in zip(variables.p_tail, rebuilt.p_tail, strict=True):
        assert jnp.allclose(lhs, rhs)
