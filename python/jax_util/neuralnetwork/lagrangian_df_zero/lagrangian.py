"""Build the local Lagrangian and KKT residuals."""

from __future__ import annotations

from jax import numpy as jnp

from ...base import Matrix, Scalar, Vector
from .constraints import adjoint_residual, primal_residual, theta_residual
from .parameterization import residual_block_norms
from .types import StationaryProblem, SuffixVariables

ResidualTerms = tuple[tuple[Matrix, ...], tuple[Matrix, ...], tuple[Vector, ...]]


def local_lagrangian(variables: SuffixVariables, problem: StationaryProblem) -> Scalar:
    """Evaluate the suffix Lagrangian."""
    primal = primal_residual(
        variables=variables,
        prefix_tape=problem.prefix_tape,
        vectorizations=problem.layer_vectorizations,
    )
    loss_value = problem.loss_from_output(variables.z_tail[-1], problem.target)
    constraint_term = sum(
        jnp.vdot(multiplier, residual)
        for multiplier, residual in zip(variables.p_tail, primal, strict=True)
    )
    return loss_value + constraint_term


def kkt_residual_terms(
    variables: SuffixVariables,
    problem: StationaryProblem,
) -> ResidualTerms:
    """Return the split KKT residual blocks."""
    primal = primal_residual(
        variables=variables,
        prefix_tape=problem.prefix_tape,
        vectorizations=problem.layer_vectorizations,
    )
    adjoint = adjoint_residual(
        variables=variables,
        prefix_tape=problem.prefix_tape,
        vectorizations=problem.layer_vectorizations,
        loss_from_output=problem.loss_from_output,
        target=problem.target,
    )
    theta = theta_residual(
        variables=variables,
        prefix_tape=problem.prefix_tape,
        vectorizations=problem.layer_vectorizations,
    )
    return primal, adjoint, theta


def kkt_residual(variables: SuffixVariables, problem: StationaryProblem) -> Vector:
    """Return the packed KKT residual."""
    primal, adjoint, theta = kkt_residual_terms(variables=variables, problem=problem)
    blocks = [
        *(block.reshape(-1) for block in primal),
        *(block.reshape(-1) for block in adjoint),
        *(block.reshape(-1) for block in theta),
    ]
    if not blocks:
        return jnp.zeros((0,))
    return jnp.concatenate(blocks, axis=0)


def packed_kkt_residual(vector: Vector, problem: StationaryProblem) -> Vector:
    """Return the packed KKT residual for one packed iterate."""
    from .parameterization import unpack_variables

    variables = unpack_variables(vector, problem.layout)
    return kkt_residual(variables=variables, problem=problem)


def residual_breakdown(variables: SuffixVariables, problem: StationaryProblem) -> dict[str, float]:
    """Summarize the norms of the split residual blocks."""
    primal, adjoint, theta = kkt_residual_terms(variables=variables, problem=problem)
    return residual_block_norms(primal_blocks=primal, adjoint_blocks=adjoint, theta_blocks=theta)


__all__ = [
    "local_lagrangian",
    "kkt_residual_terms",
    "kkt_residual",
    "packed_kkt_residual",
    "residual_breakdown",
]
