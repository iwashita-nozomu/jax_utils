"""Build residual blocks for the suffix stationary problem."""

from __future__ import annotations

from collections.abc import Callable

import jax
from jax import numpy as jnp

from ...base import Matrix, Scalar, Vector
from ..layer_utils import StandardCarry
from .parameterization import rebuild_layer
from .types import LayerVectorization, PrefixTape, SuffixVariables


def _apply_layer(
    input_z: Matrix,
    theta: Vector,
    vectorization: LayerVectorization,
    prefix_tape: PrefixTape,
) -> Matrix:
    """Apply one rebuilt layer to the current activation."""
    layer = rebuild_layer(theta, vectorization)
    return layer(StandardCarry(z=input_z), prefix_tape.ctx).z


def rollout_suffix(
    prefix_tape: PrefixTape,
    theta_tail: tuple[Vector, ...],
    vectorizations: tuple[LayerVectorization, ...],
) -> tuple[Matrix, ...]:
    """Roll out the suffix layers and return their outputs."""
    if len(theta_tail) != len(vectorizations):
        raise ValueError("theta_tail and vectorizations must have the same length.")

    outputs: list[Matrix] = []
    current = prefix_tape.z_prefix
    for theta, vectorization in zip(theta_tail, vectorizations, strict=True):
        current = _apply_layer(current, theta, vectorization, prefix_tape)
        outputs.append(current)
    return tuple(outputs)


def primal_residual(
    variables: SuffixVariables,
    prefix_tape: PrefixTape,
    vectorizations: tuple[LayerVectorization, ...],
) -> tuple[Matrix, ...]:
    """Return the primal residual blocks."""
    predicted = rollout_suffix(prefix_tape, variables.theta_tail, vectorizations)
    return tuple(
        current_z - predicted_z
        for current_z, predicted_z in zip(variables.z_tail, predicted, strict=True)
    )


def adjoint_residual(
    variables: SuffixVariables,
    prefix_tape: PrefixTape,
    vectorizations: tuple[LayerVectorization, ...],
    loss_from_output: Callable[[Matrix, Matrix], Scalar],
    target: Matrix,
) -> tuple[Matrix, ...]:
    """Return the adjoint recursion residual blocks."""
    num_layers = len(vectorizations)
    if num_layers == 0:
        return ()

    residuals: list[Matrix] = [jnp.zeros_like(p) for p in variables.p_tail]

    def _terminal_loss(z: Matrix) -> Scalar:
        return loss_from_output(z, target)

    terminal_grad = jax.grad(_terminal_loss)(variables.z_tail[-1])
    residuals[-1] = terminal_grad + variables.p_tail[-1]

    for local_index in range(num_layers - 2, -1, -1):
        z_current = variables.z_tail[local_index]
        p_next = variables.p_tail[local_index + 1]
        theta_next = variables.theta_tail[local_index + 1]
        vectorization_next = vectorizations[local_index + 1]

        def contracted_output(z_prev: Matrix) -> Scalar:
            output = _apply_layer(z_prev, theta_next, vectorization_next, prefix_tape)
            return jnp.vdot(output, p_next)

        pullback = jax.grad(contracted_output)(z_current)
        residuals[local_index] = variables.p_tail[local_index] - pullback

    return tuple(residuals)


def theta_residual(
    variables: SuffixVariables,
    prefix_tape: PrefixTape,
    vectorizations: tuple[LayerVectorization, ...],
) -> tuple[Vector, ...]:
    """Return the parameter stationarity residual blocks."""
    residuals: list[Vector] = []
    for local_index, (theta, p, vectorization) in enumerate(
        zip(variables.theta_tail, variables.p_tail, vectorizations, strict=True)
    ):
        input_z = prefix_tape.z_prefix if local_index == 0 else variables.z_tail[local_index - 1]

        def contracted_output(theta_vec: Vector) -> Scalar:
            output = _apply_layer(input_z, theta_vec, vectorization, prefix_tape)
            return jnp.vdot(output, p)

        residuals.append(jax.grad(contracted_output)(theta))
    return tuple(residuals)


def backward_multipliers(
    theta_tail: tuple[Vector, ...],
    z_tail: tuple[Matrix, ...],
    prefix_tape: PrefixTape,
    vectorizations: tuple[LayerVectorization, ...],
    loss_from_output: Callable[[Matrix, Matrix], Scalar],
    target: Matrix,
) -> tuple[Matrix, ...]:
    """Build warm-start multipliers consistent with the current rollout."""
    num_layers = len(theta_tail)
    if num_layers == 0:
        return ()

    multipliers: list[Matrix] = [jnp.zeros_like(z) for z in z_tail]

    def _terminal_loss(z: Matrix) -> Scalar:
        return loss_from_output(z, target)

    multipliers[-1] = -jax.grad(_terminal_loss)(z_tail[-1])

    for local_index in range(num_layers - 2, -1, -1):
        z_current = z_tail[local_index]
        p_next = multipliers[local_index + 1]
        theta_next = theta_tail[local_index + 1]
        vectorization_next = vectorizations[local_index + 1]

        def contracted_output(z_prev: Matrix) -> Scalar:
            output = _apply_layer(z_prev, theta_next, vectorization_next, prefix_tape)
            return jnp.vdot(output, p_next)

        multipliers[local_index] = jax.grad(contracted_output)(z_current)

    return tuple(multipliers)


__all__ = [
    "rollout_suffix",
    "primal_residual",
    "adjoint_residual",
    "theta_residual",
    "backward_multipliers",
]
